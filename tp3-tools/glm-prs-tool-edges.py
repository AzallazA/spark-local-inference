#!/usr/bin/env python3
"""Live GLM-5.3-Flash tool-argument edge cases for the #59/#179 parser change (glm53-tp3-prs).

Each case sends one tool-requiring prompt REPS times (non-streaming, plus streaming for the cases marked so) and
classifies every reply:
  pass      HTTP 200, finish_reason tool_calls, the expected tool, arguments parse and satisfy the case's check
  mismatch  a tool call whose arguments do not satisfy the check (the value is recorded; off-schema values are
            retained by #179, so for the off_schema_* cases any completed call is a pass)
  no_call   the model answered without calling a tool (model behaviour, not a parser result)
  error     HTTP error, aborted stream or an error event: the parser failed the call (#39 class)
The off_schema_* cases ask the model to pass a value of the wrong JSON type; before #179 the server failed those
calls ("tool parameter value does not match its declared JSON type"), after it they must complete. The
markup_in_value case puts a literal </arg_value> inside a string value (non-streaming and streaming). Before the
glm53-tp3 </arg_value>-in-value fix the parser split at the first closing tag (mismatch or error); with it a closing
tag ends a value only before the next <arg_key> or the end of the call, so that case must pass too. Parser errors in
every case count against PASS; EXEMPT_MARKUP=1 restores the old exemption for binaries without the fix.

usage: glm-prs-tool-edges.py OUT.json [REPS=4]   (FLARE_URL, MODEL, TOOLS_EFFORT from the environment)
"""
import json
import os
import sys
import urllib.error
import urllib.request

URL = os.environ.get("FLARE_URL", "http://127.0.0.1:15015").rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("MODEL", "GLM-5.3-Flash")
EFFORT = os.environ.get("TOOLS_EFFORT", "low")


def tool(name, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": f"{name} tool.", "parameters": {
        "type": "object", "properties": properties, "required": list(required)}}}


SET_FLAG = tool("set_flag", {"name": {"type": "string"}, "enabled": {"type": "boolean"}}, ["name", "enabled"])
TAG_FILE = tool("tag_file", {"path": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
                ["path", "tags"])
CONFIGURE = tool("configure", {"settings": {"type": "object"}}, ["settings"])
SET_LIMIT = tool("set_limit", {"limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}, ["limit"])
WRITE_FILE = tool("write_file", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])
SET_COUNT = tool("set_count", {"count": {"type": "integer"}}, ["count"])

# name, tools, prompt, expected tool, check(arguments) -> bool, also streamed
CASES = [
    ("bool_true", [SET_FLAG], "Enable the feature flag named dark_mode.", "set_flag",
     lambda a: a.get("enabled") is True, True),
    ("array_tags", [TAG_FILE], "Tag the file /srv/a.txt with the tags alpha, beta and gamma.", "tag_file",
     lambda a: isinstance(a.get("tags"), list) and sorted(a["tags"]) == ["alpha", "beta", "gamma"], False),
    ("nested_object", [CONFIGURE], "Configure the server: port 8080, and tls enabled with cert /etc/c.pem and "
     "protocols h2 and http/1.1. Put tls in a nested object.", "configure",
     lambda a: isinstance(a.get("settings"), dict) and any(isinstance(v, dict) for v in a["settings"].values()), False),
    ("anyof_null", [SET_LIMIT], "Remove the limit: call set_limit with limit set to null.", "set_limit",
     lambda a: "limit" in a and a["limit"] is None, False),
    ("empty_string", [WRITE_FILE], "Create the empty file /tmp/empty.txt (its content is the empty string).",
     "write_file", lambda a: a.get("content") == "", False),
    ("markup_in_value", [WRITE_FILE], "Write the file /tmp/t.xml whose content is exactly this text: "
     "<a></arg_value><b/></a>", "write_file", lambda a: "</arg_value>" in a.get("content", ""), True),
    ("off_schema_int", [SET_COUNT], "Call set_count and pass count as the literal word many (the string many, "
     "even though the schema says integer; do not convert it).", "set_count", lambda a: "count" in a, True),
    ("off_schema_bool", [SET_FLAG], "Call set_flag for the flag named beta with enabled set to the literal string "
     "yes (not a boolean; pass the word yes exactly).", "set_flag", lambda a: "enabled" in a, False),
]


def body(tools, prompt, stream):
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "tools": tools,
            "tool_choice": "auto", "max_tokens": 8192, "temperature": 1.0, "top_p": 0.95,
            "reasoning_effort": EFFORT, "stream": stream}


def post(payload):
    request = urllib.request.Request(URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=900)


def call_once(tools, prompt, stream):
    """(finish_reason, [(name, arguments string)], error or None)."""
    try:
        with post(body(tools, prompt, stream)) as response:
            if not stream:
                choice = json.load(response)["choices"][0]
                calls = [(c["function"]["name"], c["function"]["arguments"])
                         for c in choice["message"].get("tool_calls") or []]
                return choice.get("finish_reason"), calls, None
            finish, calls, done = None, {}, False
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    return None, [], f"stream error event: {str(event['error'])[:300]}"
                for choice in event.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    for delta in choice.get("delta", {}).get("tool_calls") or []:
                        slot = calls.setdefault(delta.get("index", 0), ["", ""])
                        function = delta.get("function") or {}
                        slot[0] += function.get("name") or ""
                        slot[1] += function.get("arguments") or ""
            if not done:
                return finish, [], "stream ended without [DONE]"
            return finish, [tuple(calls[i]) for i in sorted(calls)], None
    except urllib.error.HTTPError as error:
        return None, [], f"HTTP {error.code}: {error.read()[:300]!r}"
    except Exception as error:  # noqa: BLE001
        return None, [], f"{type(error).__name__}: {error}"


def classify(expected, check, finish, calls, error):
    if error:
        return "error", error
    if not calls:
        return "no_call", finish
    name, arguments = calls[0]
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return "mismatch", f"arguments are not JSON: {arguments[:200]}"
    if name != expected or not isinstance(parsed, dict):
        return "mismatch", f"{name} {arguments[:200]}"
    return ("pass" if check(parsed) and finish == "tool_calls" else "mismatch"), parsed


def main():
    out, reps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
    results, summary = [], {}
    for name, tools, prompt, expected, check, streamed in CASES:
        for stream in (False, True) if streamed else (False,):
            label = f"{name}{'/stream' if stream else ''}"
            for rep in range(reps):
                verdict, detail = classify(expected, check, *call_once(tools, prompt, stream))
                results.append({"case": label, "rep": rep, "verdict": verdict, "detail": detail})
                summary.setdefault(label, {}).setdefault(verdict, 0)
                summary[label][verdict] += 1
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=1, default=str)
    exempt = os.environ.get("EXEMPT_MARKUP") == "1"
    errors = sum(v.get("error", 0) for k, v in summary.items() if not (exempt and k.startswith("markup_in_value")))
    print(" | ".join(f"{k} " + ",".join(f"{v}{n}" for v, n in sorted(c.items())) for k, c in summary.items()))
    scope = "outside markup_in_value" if exempt else "in all cases"
    print(f"tool edges: {'PASS' if errors == 0 else 'FAIL'} ({errors} parser errors {scope})")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

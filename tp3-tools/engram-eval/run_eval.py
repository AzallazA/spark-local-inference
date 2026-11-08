"""Run one Engram-precision measurement against the vLLM server on :15015.

  run_eval.py logprobs <label>   teacher-forced top-20 prompt logprobs for passages.jsonl -> out/<label>-logprobs.json.gz
  run_eval.py triviaqa <label>   greedy, thinking off -> out/<label>-triviaqa.jsonl
  run_eval.py mmlupro  <label>   greedy, thinking off -> out/<label>-mmlupro.jsonl
"""
import gzip, json, os, re, string, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
URL = "http://127.0.0.1:15015/v1"
MODEL = "DeepSeek-V4.1-Flash"
# vLLM's DeepSeek template takes `thinking`; Flare rejects it and takes `enable_thinking` (and clamps temperature to 1.0).
THINKING_OFF = {"enable_thinking": False} if os.environ.get("ENGINE") == "flare" else {"thinking": False}


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def chat(prompt, max_tokens):
    r = post("/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                   "max_tokens": max_tokens, "temperature": 0, "top_p": 1.0,
                                   "chat_template_kwargs": THINKING_OFF})
    return r["choices"][0]["message"]["content"] or ""


def logprobs(label):
    passages = [json.loads(l) for l in open(HERE / "passages.jsonl")]

    def one(p):
        r = post("/completions", {"model": MODEL, "prompt": p["ids"], "max_tokens": 1, "temperature": 0,
                                  "prompt_logprobs": 20})
        rows = []
        for pos, entry in enumerate(r["choices"][0]["prompt_logprobs"][1:], start=1):
            actual = p["ids"][pos]
            top = {int(t): v["logprob"] for t, v in entry.items()}
            rows.append([actual, top.get(actual), sorted(top.items(), key=lambda kv: -kv[1])[:20]])
        return {"id": p["id"], "kind": p["kind"], "rows": rows}

    # One passage at a time: full-vocabulary prompt logits for several 1K-token prompts at once run out of VRAM.
    res = [one(p) for p in passages]
    with gzip.open(OUT / f"{label}-logprobs.json.gz", "wt") as f:
        json.dump(res, f)
    print(label, "logprobs:", len(res), "passages")


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def triviaqa(label):
    qs = [json.loads(l) for l in open(HERE / "triviaqa.jsonl")]

    def one(q):
        a = chat("Answer the trivia question with only the answer, no explanation.\nQuestion: " + q["question"], 48)
        n = norm(a)
        return {"id": q["id"], "answer": a, "exact": n in q["aliases"],
                "contains": any(f" {al} " in f" {n} " for al in q["aliases"] if al)}

    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(one, qs))
    with open(OUT / f"{label}-triviaqa.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(label, "triviaqa: exact", sum(r["exact"] for r in res), "contains", sum(r["contains"] for r in res), "/", len(res))


def mmlupro(label):
    qs = [json.loads(l) for l in open(HERE / "mmlupro.jsonl")]
    letters = "ABCDEFGHIJ"

    def one(q):
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(q["options"]))
        # ~20% of replies reason step by step despite the instruction: allow room and take the stated final answer.
        a = chat(f"{q['question']}\n\n{opts}\n\nAnswer with only the letter of the correct option.", 2048)
        m = (re.findall(r"answer is:?\s*\(?\*{0,2}([A-J])\b", a, re.I) or re.findall(r"answer:?\s*\*{0,2}\(?([A-J])\b", a, re.I)
             or re.findall(r"\b([A-J])\b", a.strip().splitlines()[-1] if a.strip() else ""))
        pred = m[-1].upper() if m else None
        return {"id": q["id"], "category": q["category"], "answer": a, "pred": pred, "correct": pred == q["answer"]}

    # The first FP8 run used an 8-token cap; fp8b re-runs fp8a too, so both FP8 runs share the 2048-token protocol.
    for lab in [label] + (["fp8a"] if label == "fp8b" else []):
        with ThreadPoolExecutor(8) as pool:
            res = list(pool.map(one, qs))
        with open(OUT / f"{lab}-mmlupro.jsonl", "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        print(lab, "mmlu-pro:", sum(r["correct"] for r in res), "/", len(res), "unparsed", sum(r["pred"] is None for r in res))


if __name__ == "__main__":
    {"logprobs": logprobs, "triviaqa": triviaqa, "mmlupro": mmlupro}[sys.argv[1]](sys.argv[2])

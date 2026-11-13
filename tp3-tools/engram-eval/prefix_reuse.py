"""Host prefix-cache reuse test on the running Flare server (:15015).

Sends N distinct ~120K-token prompts (unique from the first token, so no shared prefix), then re-sends a sample of
them and measures time to first token (TTFT):
  cold        first send of each prompt (full prefill)
  gpu hit     the most recent prompt again (its KV is still in the GPU pool)
  host hit?   prompts evicted from the 2.37M-token GPU pool, re-sent oldest-first
Usage: prefix_reuse.py <label> <N>     -> out/prefix-<label>.json
"""
import json, random, sys, time, urllib.request
from pathlib import Path
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
TOK = Tokenizer.from_file("/path/to/ckpt/ds41-1017-engram-mxfp4/tokenizer.json")
WORDS = [w for w in open("/usr/share/dict/words").read().split() if w.isalpha() and w.islower()] \
    if Path("/usr/share/dict/words").exists() else None
TARGET = 120_000


def prompt(i):
    rng = random.Random(1000 + i)
    vocab = WORDS or [f"w{n}" for n in range(20000)]
    words, toks = [f"Document {i} ({rng.random():.12f})."], 0
    while toks < TARGET:
        chunk = " ".join(rng.choice(vocab) for _ in range(4000))
        words.append(chunk); toks += len(TOK.encode(chunk, add_special_tokens=False).ids)
    return " ".join(words) + "\n\nReply with the word OK."


def ttft(text):
    body = {"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": text}], "max_tokens": 4,
            "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:15015/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if line.startswith(b"data: ") and b'"content"' in line:
                first = time.monotonic() - t
                break
        else:
            first = None
        for _ in r:
            pass
    return first


def metrics():
    m = {}
    for l in urllib.request.urlopen("http://127.0.0.1:15015/metrics", timeout=10).read().decode().splitlines():
        if l.startswith("flare_host_cache_lookup_outcomes_total") or l.startswith("flare_host_cache_records") \
                or l.startswith("flare_host_cache_bytes") or l.startswith("flare_prefix_cache"):
            k, v = l.rsplit(" ", 1); m[k] = float(v)
    return m


label, n = sys.argv[1], int(sys.argv[2])
texts = [prompt(i) for i in range(n)]
res = {"label": label, "n": n, "cold": [], "gpu_hit": None, "reuse": []}
for i, t in enumerate(texts):
    res["cold"].append(ttft(t))
    if i % 10 == 9:
        print(f"{label}: sent {i + 1}/{n}, last cold TTFT {res['cold'][-1]:.2f} s", flush=True)
res["gpu_hit"] = ttft(texts[-1])
for i in [0, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60, n - 10]:
    if i < n - 1:
        res["reuse"].append([i, ttft(texts[i])])
res["metrics"] = metrics()
(HERE / "out").mkdir(exist_ok=True)
json.dump(res, open(HERE / f"out/prefix-{label}.json", "w"), indent=1)
cold = sorted(x for x in res["cold"] if x)
print(f"{label}: cold TTFT median {cold[len(cold) // 2]:.2f} s | gpu hit {res['gpu_hit']:.2f} s | reuse (prompt, TTFT s): "
      + ", ".join(f"{i}:{t:.2f}" for i, t in res["reuse"]))
print("host cache:", {k.split("{")[-1].rstrip("}"): v for k, v in res["metrics"].items() if "lookup_outcomes" in k})

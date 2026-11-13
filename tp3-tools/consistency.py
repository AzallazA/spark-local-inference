#!/usr/bin/env python3
"""Decode-vs-prefill consistency for any model, including meaningless truncated ones.

Each prompt is sent as token IDs; generated tokens are mapped back to IDs with the local tokenizer (a sequence stops at
the first token string that maps to several IDs), so every teacher-forced prefill sees exactly the decoded IDs - no
re-tokenization drift. Prefill always runs Flare's own path, so a decode-only change (BRRR layers) must stay within the
floor measured on a build without it. Reports |delta logprob| of the decode top-1 token and top-1 agreement.
  .venv/bin/python consistency.py [tokens=48] [prompts=4] [tokenizer dir]
"""
import json
import statistics
import sys
import urllib.request

from transformers import PreTrainedTokenizerFast

URL = "http://localhost:15015/v1/completions"
PROMPTS = ["def fibonacci(n):\n    ", "The capital of France is", "for (int i = 0; i < n; ++i) {\n    ",
           "Once upon a time, in a small village,", "import numpy as np\nx = np.", "SELECT name, age FROM users WHERE"]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 48
k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
root = sys.argv[3] if len(sys.argv) > 3 else "/path/to/ckpt/ds41-1017-engram-mxfp4"
tok = PreTrainedTokenizerFast(tokenizer_file=f"{root}/tokenizer.json")
pieces = {}
for i in range(len(tok)):
    pieces.setdefault(tok.decode([i]), []).append(i)


def post(body):
    req = urllib.request.Request(URL, json.dumps({"model": "DeepSeek-V4.1-Flash", **body}).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["logprobs"]


deltas, agree, total, stopped = [], 0, 0, 0
for prompt in PROMPTS[:k]:
    ids = [tok.bos_token_id or 0] + tok.encode(prompt, add_special_tokens=False)
    lp = post({"prompt": ids, "max_tokens": n, "top_p": 1.0, "logprobs": 5})
    generated = []
    for text in lp["tokens"]:
        match = pieces.get(text, [])
        if len(match) != 1:
            stopped += 1
            break
        generated.append(match[0])
    for i in range(len(generated)):
        ref = post({"prompt": ids + generated[:i], "max_tokens": 1, "top_p": 1.0, "logprobs": 5})["top_logprobs"][0]
        dec = lp["top_logprobs"][i]
        best = max(dec, key=dec.get)
        total += 1
        agree += best == max(ref, key=ref.get)
        deltas.append(abs(dec[best] - ref[best]) if best in ref else float("inf"))
finite = sorted(d for d in deltas if d != float("inf"))
q = lambda p: finite[min(len(finite) - 1, int(p * len(finite)))] if finite else float("nan")
print(f"{total} positions ({stopped} sequences cut at ambiguous tokens): top-1 agree {agree / max(total, 1):.3f}, "
      f"decode top-1 missing from prefill top-5 {len(deltas) - len(finite)}, |dlogprob| median "
      f"{statistics.median(finite) if finite else float('nan'):.4f} p90 {q(0.9):.4f} p99 {q(0.99):.4f} "
      f"max {finite[-1] if finite else float('nan'):.4f}")

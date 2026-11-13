"""Build the fixed inputs for the FP8 vs MXFP4 Engram comparison.

passages.jsonl  : 120 WikiText-103 test + validation articles + 80 source files, each cut to 1024 tokens (BOS + 1023), as token IDs
triviaqa.jsonl  : 1000 TriviaQA rc.nocontext validation questions (seed 0) with normalized answer aliases
mmlupro.jsonl   : 1000 MMLU-Pro test questions, stratified over the 14 categories (seed 0)
"""
import json, random, re
from pathlib import Path

from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # the fork checkout: the code corpus is its ignition/ and helios/ trees
import pandas as pd
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
TOK = Tokenizer.from_file("/path/to/ckpt/ds41-1017-engram-mxfp4/tokenizer.json")
BOS = TOK.token_to_id("<｜begin▁of▁sentence｜>")
LEN = 1024
rng = random.Random(0)


def cut(text):
    ids = TOK.encode(text, add_special_tokens=False).ids
    return [BOS] + ids[: LEN - 1] if len(ids) >= LEN - 1 else None


passages = []
wiki = [t for split in ("test", "validation") for t in pd.read_parquet(
    HERE / f"data/wikitext/wikitext-103-raw-v1/{split}-00000-of-00001.parquet")["text"].tolist()]
articles, cur = [], []
for line in wiki:
    if re.match(r"^ = [^=].* = \n$", line) and cur:
        articles.append("".join(cur)); cur = []
    cur.append(line)
articles.append("".join(cur))
for text in articles:
    ids = cut(text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", "."))
    if ids:
        passages.append({"kind": "wiki", "ids": ids})
    if sum(p["kind"] == "wiki" for p in passages) == 120:
        break

code_files = sorted(p for root in (str(REPO / "ignition"), str(REPO / "helios" / "src"))
                    for p in Path(root).rglob("*") if p.suffix in (".py", ".rs") and p.stat().st_size > 8000)
rng.shuffle(code_files)
for path in code_files:
    ids = cut(path.read_text(errors="ignore"))
    if ids:
        passages.append({"kind": "code", "ids": ids, "source": path.name})
    if sum(p["kind"] == "code" for p in passages) == 80:
        break
with open(HERE / "passages.jsonl", "w") as f:
    for i, p in enumerate(passages):
        f.write(json.dumps({"id": i, **p}) + "\n")

tq = pd.read_parquet(HERE / "data/trivia_qa/rc.nocontext/validation-00000-of-00001.parquet")
rows = tq.sample(n=1000, random_state=0)
with open(HERE / "triviaqa.jsonl", "w") as f:
    for _, r in rows.iterrows():
        ans = r["answer"]
        f.write(json.dumps({"id": r["question_id"], "question": r["question"],
                            "aliases": sorted(set(ans["normalized_aliases"].tolist()) | {ans["normalized_value"]})}) + "\n")

mp = pd.read_parquet(HERE / "data/MMLU-Pro/data/test-00000-of-00001.parquet")
cats = sorted(mp["category"].unique())
per = {c: 1000 // len(cats) + (1 if i < 1000 % len(cats) else 0) for i, c in enumerate(cats)}
with open(HERE / "mmlupro.jsonl", "w") as f:
    for c in cats:
        for _, r in mp[mp["category"] == c].sample(n=per[c], random_state=0).iterrows():
            f.write(json.dumps({"id": int(r["question_id"]), "category": c, "question": r["question"],
                                "options": list(r["options"]), "answer": r["answer"]}) + "\n")
print(len(passages), "passages,", len(rows), "trivia,", sum(per.values()), "mmlu-pro over", len(cats), "categories")

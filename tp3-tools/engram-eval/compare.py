"""Compare teacher-forced logprobs and benchmark answers between two runs.

  compare.py <ref label> <test label>
KL is estimated on the reference run's top-20 tokens: p from the reference, q from the test run's top-20, with
tokens missing from the test top-20 given the test run's 20th logprob (an upper bound on q, so KL is slightly
underestimated for those tokens). Both distributions are renormalized over the reference top-20 plus a residual bucket.
"""
import gzip, json, math, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def load(label):
    with gzip.open(OUT / f"{label}-logprobs.json.gz", "rt") as f:
        return {p["id"]: p for p in json.load(f)}


def stats(ref, test):
    agg = {}
    for pid, rp in ref.items():
        tp = test[pid]
        for (act, ra, rtop), (_, ta, ttop) in zip(rp["rows"], tp["rows"]):
            s = agg.setdefault(rp["kind"], {"n": 0, "top1": 0, "kl": 0.0, "nll_r": 0.0, "nll_t": 0.0, "dlp": 0.0, "miss": 0})
            s["n"] += 1
            s["top1"] += rtop[0][0] == ttop[0][0]
            if ra is None or ta is None:
                s["miss"] += 1
            else:
                s["nll_r"] -= ra; s["nll_t"] -= ta; s["dlp"] += abs(ra - ta)
            tmap = dict((int(t), v) for t, v in ttop)
            floor = ttop[-1][1]
            p = [math.exp(v) for _, v in rtop]
            q = [math.exp(tmap.get(int(t), floor)) for t, _ in rtop]
            p.append(max(1e-12, 1 - sum(p))); q.append(max(1e-12, 1 - sum(q)))
            zp, zq = sum(p), sum(q)
            s["kl"] += sum((a / zp) * math.log((a / zp) / (b / zq)) for a, b in zip(p, q) if a > 0)
    return agg


def report(ref_l, test_l):
    agg = stats(load(ref_l), load(test_l))
    print(f"teacher-forced {ref_l} -> {test_l}")
    tot = {k: sum(a[k] for a in agg.values()) for k in ("n", "top1", "kl", "nll_r", "nll_t", "dlp", "miss")}
    for kind, s in [*sorted(agg.items()), ("all", tot)]:
        m = s["n"] - s["miss"]
        print(f"  {kind:5s} tokens {s['n']:7d}  top-1 agree {100 * s['top1'] / s['n']:6.2f}%  mean KL {s['kl'] / s['n']:.5f} nats"
              f"  ppl {math.exp(s['nll_r'] / m):.4f} -> {math.exp(s['nll_t'] / m):.4f}  mean |dlogp| {s['dlp'] / m:.4f}")
    for bench in ("triviaqa", "mmlupro"):
        a, b = OUT / f"{ref_l}-{bench}.jsonl", OUT / f"{test_l}-{bench}.jsonl"
        if not (a.exists() and b.exists()):
            continue
        ra = {r["id"]: r for r in map(json.loads, open(a))}; rb = {r["id"]: r for r in map(json.loads, open(b))}
        key = "exact" if bench == "triviaqa" else "correct"
        same = sum(ra[i]["answer"].strip() == rb[i]["answer"].strip() for i in ra)
        only_a = sum(ra[i][key] and not rb[i][key] for i in ra); only_b = sum(rb[i][key] and not ra[i][key] for i in ra)
        extra = ""
        if bench == "triviaqa":
            extra = f"  contains {sum(r['contains'] for r in ra.values())} vs {sum(r['contains'] for r in rb.values())}"
        print(f"  {bench}: {sum(r[key] for r in ra.values())} vs {sum(r[key] for r in rb.values())} / {len(ra)}{extra}"
              f"  identical answers {same}  right only in {ref_l} {only_a}, only in {test_l} {only_b}")


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2])

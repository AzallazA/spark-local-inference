"""One-line summary of llm_decode_bench JSON results for chain logs: per cell tok/s, steps/s, tokens/step, acceptance.

    python3 tp3-tools/bench_cell.py RESULT.json [RESULT.json ...]
Prints e.g. `c1 305/116.9s/2.61t/0.39a c4 ... c8 ...` (tok/s, steps/s, tok per step, draft acceptance). Cells whose
aggregate is not positive (bench loop artifact) print `c4 ERR`.
"""
import json, sys

parts = []
for path in sys.argv[1:]:
    try:
        results = json.load(open(path))["results"]
    except Exception as exc:  # noqa: BLE001
        parts.append(f"{path.rsplit('/', 1)[-1]}:ERR({exc.__class__.__name__})")
        continue
    for r in results:
        label = f"c{r.get('concurrency', '?')}"
        tps = r.get("aggregate_tps", 0) or 0
        if tps <= 0:
            parts.append(f"{label} ERR")
            continue
        f = r.get("server_flare_metrics") or {}
        if f.get("valid"):
            c = f["counter_delta"]
            steps = f["verification_graphs_per_second"]
            tok_step = c["raw_emitted_tokens"] / max(c["verification_graphs"], 1)
            parts.append(f"{label} {tps:.0f}/{steps:.1f}s/{tok_step:.2f}t/{f['raw_draft_acceptance']:.2f}a")
        else:
            parts.append(f"{label} {tps:.0f}")
print(" ".join(parts))

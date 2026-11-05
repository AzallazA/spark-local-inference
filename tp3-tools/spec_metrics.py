"""Snapshot/diff Helios speculative metrics: per (batch,width) rounds, GPU ms/round, tokens/round, and GPU busy share.
usage: spec_metrics.py snap OUT.json | spec_metrics.py diff A.json B.json"""
import json, re, sys, time, urllib.request
NAMES = ("rounds", "gpu_nanoseconds", "emitted_tokens", "accepted_drafts", "offered_drafts")

def snap():
    text = urllib.request.urlopen("http://127.0.0.1:15015/metrics", timeout=10).read().decode()
    out = {"t": time.time(), "m": {}}
    for line in text.splitlines():
        m = re.match(r'flare_speculative_(\w+)_total\{batch="(\d+)",width="(\d+)"\} (\S+)', line)
        if m and m[1] in NAMES:
            out["m"].setdefault(f"{m[2]},{m[3]}", {})[m[1]] = float(m[4])
    return out

if sys.argv[1] == "snap":
    json.dump(snap(), open(sys.argv[2], "w"))
else:
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    wall = b["t"] - a["t"]; gpu_total = 0
    print(f"wall {wall:.1f} s")
    print(f"{'batch,width':>11} {'rounds':>8} {'GPU ms/round':>12} {'tok/round':>9} {'accept':>6}")
    for k in sorted(b["m"], key=lambda s: tuple(map(int, s.split(",")))):
        d = {n: b["m"][k].get(n, 0) - a["m"].get(k, {}).get(n, 0) for n in NAMES}
        if d["rounds"] <= 0: continue
        gpu_total += d["gpu_nanoseconds"]
        acc = d["accepted_drafts"] / d["offered_drafts"] if d["offered_drafts"] else 0
        print(f"{k:>11} {d['rounds']:8.0f} {d['gpu_nanoseconds']/d['rounds']/1e6:12.3f} {d['emitted_tokens']/d['rounds']:9.2f} {acc:6.2f}")
    print(f"GPU busy (speculative) {gpu_total/1e9:.1f} s of {wall:.1f} s wall = {gpu_total/1e9/wall*100:.0f}%")

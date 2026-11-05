"""Append a Helios speculative-metrics snapshot every 2 s to a JSONL file (for steps/s per bench cell)."""
import json, sys, time, urllib.request, re
out = open(sys.argv[1], "a")
while True:
    try:
        text = urllib.request.urlopen("http://127.0.0.1:15015/metrics", timeout=5).read().decode()
        m = {}
        for line in text.splitlines():
            r = re.match(r'flare_speculative_(rounds|emitted_tokens|gpu_nanoseconds)_total\{batch="(\d+)",width="(\d+)"\} (\S+)', line)
            if r: m.setdefault(r[2], {}).setdefault(r[1], 0.0); m[r[2]][r[1]] += float(r[4])
        out.write(json.dumps({"t": time.time(), "m": m}) + "\n"); out.flush()
    except Exception:
        out.write(json.dumps({"t": time.time(), "m": None}) + "\n"); out.flush()
    time.sleep(2)

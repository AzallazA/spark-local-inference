#!/bin/bash
# Prefix reuse at --prefix-cache-memory 16G (current) and 32G, 80 x 127K-token prompts each, then restore the default.
cd $HOME/engram-eval
pids(){ ps -eo pid,args | awk '$2=="./flare/target/release/flare" && $3=="run"{print $1}'; }
restart(){ for p in $(pids); do kill -INT $p; done; for i in $(seq 1 60); do [ -z "$(pids)" ] && break; sleep 3; done
  for p in $(pids); do kill -KILL $p; done
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 2; done
  (cd $HOME/flare-up && /usr/bin/env PCM=$1 nohup ./serve-best.sh > $HOME/flare-up-v3/serve-prefix.log 2>&1 &)
  for i in $(seq 1 120); do sleep 5; curl -s -m 2 localhost:15015/v1/models | grep -q '"id"' && break; done
  echo "=== $(date +%H:%M) serving PCM=$1, $(grep -o 'host prefix cache.*' $HOME/flare-up-v3/serve-prefix.log | head -1 | cut -c1-90)"; }
restart 16G; .venv/bin/python prefix_reuse.py pcm16 80
restart 32G; .venv/bin/python prefix_reuse.py pcm32 80
restart 16G; echo "=== prefix chain done (default serving)"

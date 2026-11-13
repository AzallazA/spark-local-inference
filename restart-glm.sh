#!/bin/bash
# Restart the GLM-5.3-Flash server: stop any `flare run`, wait until every GPU is below 1000 MiB, start serve-glm.sh
# (env overrides pass through), wait for /v1/models. Log: ~/bench-results/serve-glm-<date>.log (not /tmp: tmpfs).
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-15015}
LOG=${LOG:-$HOME/bench-results/serve-glm-$(date +%m%d-%H%M).log}
mkdir -p "$(dirname "$LOG")"
pids(){ ps -eo pid,args | awk '$2 ~ /(^|\/)flare$/ && $3=="run"{print $1}'; }
for p in $(pids); do kill -INT "$p"; done
for i in $(seq 1 60); do [ -z "$(pids)" ] && break; sleep 3; done
for p in $(pids); do kill -KILL "$p"; done
echo "waiting for GPUs < 1000 MiB"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 2; done
nohup "$D/serve-glm.sh" > "$LOG" 2>&1 &
echo "started serve-glm.sh (pid $!), log $LOG"
t0=$(date +%s)
for i in $(seq 1 360); do
  sleep 5
  if curl -s -m 2 "localhost:$PORT/v1/models" | grep -q '"id"'; then echo "ready after $(( $(date +%s) - t0 )) s"; exit 0; fi
  [ -z "$(pids)" ] && { echo "flare exited; tail of log:"; tail -20 "$LOG"; exit 1; }
done
echo "not ready after 30 min; log $LOG"; exit 1

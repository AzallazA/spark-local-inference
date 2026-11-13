#!/bin/bash
# Mini export: export a truncated DeepSeek-V4.1-Flash checkpoint view (6 or 8 layers; no parking, no Engram) at TP3,
# serve it, run tp3-tools/consistency.py (decode vs prefill), stop the server. About 10 minutes. It catches compile,
# trace and wiring faults in risky kernel or wiring changes; it says nothing about speed (handbook D.2 rule 1).
#   SNAP    truncated checkpoint view directory (required; 8-layer views carry native scales for parking/BRRR tests)
#   LAYERS  BRRR layers, e.g. 0-5 (default: none)
#   LABEL   label for the log directory (default: mini)
#   OUT     log directory (default: ./mini-<label>-<MMDD-HHMM>)
#   GPUS    default 0,1,2; PORT is fixed at 15015 (consistency.py's URL)
#   FLARE_BIN  server binary (default: this checkout's flare-cli/target/release/flare; worktrees without a Rust build
#              point it at the serving checkout's binary)
# Further FLARE_DS41_* settings pass through from the environment (e.g. FLARE_DS41_FUSED_QKV_NORM=1). Run from any
# worktree: the export imports this checkout (PYTHONPATH), and the log records which ignition it imported.
# The GPUs must be idle: stop any server first; this script refuses to start next to one and starts none afterwards.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT"
: "${SNAP:?set SNAP to a truncated checkpoint view}"
LABEL=${LABEL:-mini}
OUT=${OUT:-$PWD/mini-$LABEL-$(date +%m%d-%H%M)}
mkdir -p "$OUT"
pids(){ ps -eo pid,args | awk '$2 ~ /(^|\/)flare$/ && $3=="run"{print $1}'; }
[ -z "$(pids)" ] || { echo "a flare server is running: stop it first (export and serving cannot share the GPUs)" >&2; exit 2; }
export PYTHONPATH="$ROOT" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}" HF_HUB_OFFLINE=1 \
       TORCH_EXTENSIONS_DIR="$ROOT/.brrr-ext" FLARE_DS41_TP3_GRAPH_GIB=1 FLARE_DS41_TP3_VOCAB_PARALLEL=1 \
       FLARE_DS41_MINI=1 FLARE_DS41_BRRR_LAYERS="${LAYERS:-}" FLARE_DS41_BRRR_LOG=1 FLARE_DS41_BRRR_CKPT="$SNAP"
export PATH="$CUDA_HOME/bin:$PATH"
.venv/bin/python -c 'import ignition; print("ignition imported from", ignition.__file__)' | tee "$OUT/export.log"
start=$(date +%s)
rc=0; timeout 3600 .venv/bin/ignition export "$SNAP" --gpus "${GPUS:-0,1,2}" --max-concurrency 8 \
  --prefill-chunk-size 4096 --mtp-draft-tokens 5 --draft-head-quantization nvfp4 --markov-head-quantization fp8 \
  --vocabulary-parallel-sampling --verbose >> "$OUT/export.log" 2>&1 || rc=$?
echo "export exit $rc after $(( ($(date +%s) - start) / 60 )) min" | tee "$OUT/SUMMARY.txt"
if [ "$rc" != 0 ]; then
  tr '\r' '\n' < "$OUT/export.log" | grep -E "Error|Traceback" | grep -v "^b12x" | tail -4 | cut -c1-300 | tee -a "$OUT/SUMMARY.txt"
  exit "$rc"
fi
P=$(grep -oE "[^ ]*/(packages|cores)/models/[^ ]*" "$OUT/export.log" | tail -1)
echo "package $P" | tee -a "$OUT/SUMMARY.txt"
"${FLARE_BIN:-$ROOT/flare-cli/target/release/flare}" run "$P" --gpus "${GPUS:-0,1,2}" --port 15015 --host 127.0.0.1 \
  --served-model-name DeepSeek-V4.1-Flash --mtp-draft-tokens 5 > "$OUT/serve.log" 2>&1 &
ready=0
for _ in $(seq 1 120); do
  sleep 5
  curl -s -m 2 localhost:15015/v1/models | grep -q '"id"' && { ready=1; break; }
  [ -n "$(pids)" ] || break
done
if [ "$ready" = 1 ]; then
  grep -E "OK.*Allocate cache" "$OUT/serve.log" | cut -c1-120 | tee -a "$OUT/SUMMARY.txt" || true
  .venv/bin/python tp3-tools/consistency.py 48 6 "$SNAP" | tee "$OUT/consistency.txt" | tail -3 | tee -a "$OUT/SUMMARY.txt"
else
  echo "server did not come up; see $OUT/serve.log" | tee -a "$OUT/SUMMARY.txt"
fi
for p in $(pids); do kill -INT "$p"; done
for _ in $(seq 1 60); do [ -z "$(pids)" ] && break; sleep 3; done
[ "$ready" = 1 ]

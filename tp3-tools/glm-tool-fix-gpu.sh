#!/bin/bash
# GPU check for the GLM </arg_value>-in-value parser fix (branch glm53-tp3, ~/flare-up-glm). Written 2026-11-09;
# not yet run. Helios parser only: no export, ~40 min.
#
#   tp3-tools/glm-tool-fix-gpu.sh             # REPS=4 edge repetitions by default
#
# Stops any flare server, serves the GLM a16 default package 082232 with this worktree's binary, runs the live
# tool-argument edge cases (tp3-tools/glm-prs-tool-edges.py; markup_in_value now counts, streaming included) and
# tools-60 from the glm-qual suite, counts parser failures in the server log, and restores the DeepSeek default with
# /tmp/restart-default.sh at the end (and on any exit). Log: ~/bench-results/glm-tool-fix-<mmdd-HHMM>/CHAIN.md.
# Never run this while another chain or export is queued on the GPUs (handbook C.4, D.2.9): it waits for running
# exports and chain scripts first.
set -uo pipefail
W=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
QUAL=${QUAL:-$HOME/flare-up-glm-qual/tp3-tools/glm-qual/qual_all.sh}
CKPT=${CKPT:-/path/to/ckpt/glm53-fp4-local}
P=${P:-/path/to/archive-a/flare-packages/models/glm53-fp4-local/gb202-tp3-202611082232}
REPS=${REPS:-4}
GF="--port 15015 --host 127.0.0.1 --served-model-name GLM-5.3-Flash --prefix-cache-memory 32G --mtp-draft-tokens 5"
R=$HOME/bench-results/glm-tool-fix-$(date +%m%d-%H%M); mkdir -p "$R"
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH HF_HUB_OFFLINE=1
K=$HOME/.cache/flare/kernels; export TRITON_CACHE_DIR=$K/triton B12X_COMPILE_CACHE_DIR=$K/b12x TORCHINDUCTOR_CACHE_DIR=$K/inductor

log(){ echo "$(date +%H:%M) $*" | tee -a "$R/CHAIN.md"; }
# stop/ready helpers as in /tmp/glm-704-chain.sh
pids(){ ps -eo pid,args | awk '$2 ~ /(^|\/)flare$/ && $3=="run"{print $1}'; }
stop(){ for p in $(pids); do kill -INT $p; done; for i in $(seq 1 60); do [ -z "$(pids)" ] && break; sleep 3; done
        for p in $(pids); do kill -KILL $p; done
        until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 2; done; }
restore(){ stop; /tmp/restart-default.sh > /dev/null 2>&1; log "default restored; glm-tool-fix chain done"; }
parser_failures(){ grep -cE "requires an argument key|unterminated GLM|failed conversion|does not match its declared|does not match any declared|GLM argument|GLM tool call|incomplete GLM tool call" "$1" 2>/dev/null; }

while ps -eo args | grep -qE '[i]gnition export|[c]hain-?[a-z0-9-]*\.sh'; do
  [ "${waited:-0}" = 0 ] && log "waiting for: $(ps -eo args | grep -E '[i]gnition export|[c]hain-?[a-z0-9-]*\.sh' | cut -c1-120 | tr '\n' ';')"
  waited=1; sleep 120; done
[ -x "$W/flare/target/release/flare" ] || { log "no release binary in $W (cargo build --locked --release --manifest-path flare/Cargo.toml)"; exit 2; }
trap restore EXIT   # set after the wait, so an interrupted wait leaves the default server alone
log "chain start: worktree $W ($(git -C "$W" log -1 --format=%h)), package $(basename "$P"), reps $REPS"

stop; SERVE_LOG=$R/serve.log
(cd "$W" && nohup ./flare/target/release/flare run "$P" --gpus 0,1,2 $GF > "$SERVE_LOG" 2>&1 &)
ok=0; for i in $(seq 1 360); do sleep 5; curl -s -m 2 localhost:15015/v1/models | grep -q '"id"' && { ok=1; break; }; [ -z "$(pids)" ] && break; done
[ $ok = 1 ] || { log "serve FAILED: $(tail -2 "$SERVE_LOG" | tr '\n' ' ' | cut -c1-200)"; exit 1; }
curl -s -m 600 localhost:15015/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 17*23? Reply with just the number."}],"max_tokens":2000,"reasoning_effort":"low"}' > "$R/smoke.json"
log "served; smoke $(python3 -c "import json;print(json.load(open('$R/smoke.json'))['choices'][0]['message']['content'].strip()[:30])" 2>&1 | tail -1)"

(cd "$W" && python3 tp3-tools/glm-prs-tool-edges.py "$R/edges.json" "$REPS" > "$R/edges.log" 2>&1); erc=$?
log "edges exit $erc: $(tail -2 "$R/edges.log" | tr '\n' ' ')"
log "markup_in_value: $(python3 -c "import json;s=json.load(open('$R/edges.json'))['summary'];print({k:v for k,v in s.items() if k.startswith('markup_in_value')})" 2>&1 | tail -1)"
log "server log parser failures after edges: $(parser_failures "$SERVE_LOG")"

QUAL_OUT=$R/qual LABEL=glm-tool-fix FLARE_LOG=$SERVE_LOG CKPT=$CKPT "$QUAL" tools > "$R/qual.log" 2>&1
log "tools-60: $(tail -1 "$R/qual/tools.log" 2>/dev/null)"
log "server log parser failures total: $(parser_failures "$SERVE_LOG")"
grep -E "requires an argument key|unterminated GLM|failed conversion|GLM argument|GLM tool call" "$SERVE_LOG" | head -20 > "$R/parser-failures.txt"

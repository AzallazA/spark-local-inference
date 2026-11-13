#!/bin/bash
# GPU plan for branch glm53-tp3-prs (upstream PRs #59+#179 tool arguments, #172 docs, #217 width-one in place,
# #218 shared draft selection) on GLM-5.3-Flash FP4-Local TP3. Written 2026-11-09; not yet run.
#
#   tp3-tools/glm-prs-gpu.sh              # phase a then phase b
#   PHASES=a tp3-tools/glm-prs-gpu.sh     # tool-call fixes only (no export, ~40 min)
#   PHASES=b tp3-tools/glm-prs-gpu.sh     # GPU tests, re-export, A/B, qualification if faster (~4-5 h)
#
# Phase a (helios parser only, no re-export): serve the a16 default package 082232 with this worktree's binary,
#   smoke, live tool-argument edge cases (tp3-tools/glm-prs-tool-edges.py), tools-60 from the glm-qual suite, and the
#   server log's parser failures. OLD_EDGES=1 (default) repeats the edge cases on the old binary (~/flare-up-v3's,
#   whose Rust equals glm53-tp3) so the off_schema_* cases show #179's before/after.
# Phase b (recipe changes, re-export): GPU tests of #217/#218 at TP3 shapes (gate), a16 export with the 082232
#   flags (glm53-export-tp3.sh defaults: no CP, chunk 4096, 5 drafts, concurrency 8, no overrides), dry runs,
#   3 alternating rounds base/candidate (decode c1/c4/c8 at 32K, 60 s, both on this worktree's binary so only the
#   package differs) with per-width round counts and GPU time from /metrics, then eval/tools/hotel at c8 on the
#   candidate only if it is faster (MIN_GAIN, default +1.0 % mean over c1/c4/c8 and no cell below -1.5 %).
# The DeepSeek default is restored at the end (and on any exit) with /tmp/restart-default.sh.
# Never run this while another chain or export is queued on the GPUs (handbook C.4, D.2.9); it waits for running
# exports and chain-*.sh scripts first, and logs while it waits.
set -uo pipefail
W=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
V3=$HOME/flare-up-v3                      # old binary only (read-only use, as every GLM chain serves)
BENCH=$HOME/llm-inference-bench
QUAL=${QUAL:-$HOME/flare-up-glm-qual/tp3-tools/glm-qual/qual_all.sh}
CKPT=${CKPT:-/path/to/ckpt/glm53-fp4-local}
BASE=${BASE:-/path/to/archive-a/flare-packages/models/glm53-fp4-local/gb202-tp3-202611082232}
M=${M:-$HOME/.cache/flare/packages/models/glm53-fp4-local}   # store the export writes to (CONFIG unset)
PHASES=${PHASES:-a b}; OLD_EDGES=${OLD_EDGES:-1}; REPS=${REPS:-4}; MIN_GAIN=${MIN_GAIN:-1.0}; MAX_LOSS=${MAX_LOSS:--1.5}
GF="--port 15015 --host 127.0.0.1 --served-model-name GLM-5.3-Flash --prefix-cache-memory 32G --mtp-draft-tokens 5"
R=$HOME/bench-results/glm-prs-$(date +%m%d-%H%M); mkdir -p "$R"
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH HF_HUB_OFFLINE=1
K=$HOME/.cache/flare/kernels; export TRITON_CACHE_DIR=$K/triton B12X_COMPILE_CACHE_DIR=$K/b12x TORCHINDUCTOR_CACHE_DIR=$K/inductor

log(){ echo "$(date +%H:%M) $*" | tee -a "$R/CHAIN.md"; }
pids(){ ps -eo pid,args | awk '$2 ~ /(^|\/)flare$/ && $3=="run"{print $1}'; }
stop(){ for p in $(pids); do kill -INT $p; done; for i in $(seq 1 60); do [ -z "$(pids)" ] && break; sleep 3; done
        for p in $(pids); do kill -KILL $p; done
        until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 2; done; }
restore(){ stop; /tmp/restart-default.sh > /dev/null 2>&1; log "default restored; glm-prs chain done"; }
# serve LABEL PACKAGE BINARY_TREE -> 0 when /v1/models answers (archive drive A loads are slower than the boot drive: 30 min)
serve(){ stop; SERVE_LOG=$R/serve-$1.log
  (cd "$3" && nohup ./flare/target/release/flare run "$2" --gpus 0,1,2 $GF > "$SERVE_LOG" 2>&1 &)
  for i in $(seq 1 360); do sleep 5; curl -s -m 2 localhost:15015/v1/models | grep -q '"id"' && return 0; [ -z "$(pids)" ] && break; done
  log "serve $1 FAILED: $(tail -2 "$SERVE_LOG" | tr '\n' ' ' | cut -c1-200)"; return 1; }
smoke(){ curl -s -m 600 localhost:15015/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 17*23? Reply with just the number."}],"max_tokens":2000,"reasoning_effort":"low"}' > "$R/smoke-$1.json"
  python3 -c "import json;print(json.load(open('$R/smoke-$1.json'))['choices'][0]['message']['content'].strip()[:30])" 2>&1 | tail -1; }
parser_failures(){ grep -cE "requires an argument key|failed conversion|does not match its declared|does not match any declared|GLM argument|GLM tool call" "$1" 2>/dev/null; }
metrics(){ curl -s -m 10 localhost:15015/metrics > "$1"; }
# per-width round counts and GPU time between two /metrics scrapes
widths(){ python3 - "$1" "$2" <<'EOF'
import re, sys
def read(path):
    out = {}
    for line in open(path):
        m = re.match(r'flare_speculative_(rounds|gpu_nanoseconds)_total\{batch="(\d+)",width="(\d+)"\} (\S+)', line)
        if m:
            key = int(m[3]); out.setdefault(key, [0, 0])[m[1] == "gpu_nanoseconds"] += float(m[4])
    return out
a, b = read(sys.argv[1]), read(sys.argv[2])
d = {w: [b[w][0] - a.get(w, [0, 0])[0], b[w][1] - a.get(w, [0, 0])[1]] for w in sorted(b)}
total = sum(v[0] for v in d.values()) or 1
print(" ".join(f"w{w}:{v[0] / total:.0%}/{v[1] / max(v[0], 1) / 1e3:.0f}us" for w, v in d.items() if v[0]))
EOF
}

# --- wait for the GPUs' other users (exports and chains), never stop them --------------------------------------
while ps -eo args | grep -qE '[i]gnition export|[c]hain-?[a-z0-9-]*\.sh'; do
  [ "${waited:-0}" = 0 ] && log "waiting for: $(ps -eo args | grep -E '[i]gnition export|[c]hain-?[a-z0-9-]*\.sh' | cut -c1-120 | tr '\n' ';')"
  waited=1; sleep 120; done
trap restore EXIT   # set after the wait, so an interrupted wait leaves the default server alone
log "chain start: worktree $W ($(git -C "$W" log -1 --format=%h)), phases: $PHASES"
[ -x "$W/flare/target/release/flare" ] || { log "no release binary in $W (cargo build --locked --release --manifest-path flare/Cargo.toml)"; exit 2; }

# --- phase a: tool-call parser (#59 + #179), no re-export -----------------------------------------------------
if [[ " $PHASES " == *" a "* ]]; then
  if serve a-new "$BASE" "$W"; then
    log "phase a: 082232 on the new binary; smoke $(smoke a-new)"
    (cd "$W" && python3 tp3-tools/glm-prs-tool-edges.py "$R/edges-new.json" "$REPS" > "$R/edges-new.log" 2>&1)
    log "edges (new binary): $(tail -2 "$R/edges-new.log" | tr '\n' ' ')"
    QUAL_OUT=$R/qual-a LABEL=glm-prs-a FLARE_LOG=$SERVE_LOG CKPT=$CKPT "$QUAL" tools > "$R/qual-a.log" 2>&1
    log "tools-60 (new binary): $(tail -1 "$R/qual-a/tools.log")"
    log "server log parser failures (new binary): $(parser_failures "$SERVE_LOG")"
  fi
  if [ "$OLD_EDGES" = 1 ] && serve a-old "$BASE" "$V3"; then
    (cd "$W" && python3 tp3-tools/glm-prs-tool-edges.py "$R/edges-old.json" "$REPS" > "$R/edges-old.log" 2>&1)
    log "edges (old binary, before #179): $(tail -2 "$R/edges-old.log" | tr '\n' ' ') | server log parser failures $(parser_failures "$SERVE_LOG")"
  fi
fi

# --- phase b: #217 + #218, re-export and A/B -------------------------------------------------------------------
if [[ " $PHASES " == *" b "* ]]; then
  stop
  T=(/usr/bin/env FLARE_RUN_CUDA_TESTS=1 FLARE_GLM53_SNAPSHOT=$CKPT PYTHONPATH=$W)
  PY=$W/.venv/bin/python; [ -e "$W/.venv" ] || ln -s "$V3/.venv" "$W/.venv"   # as ~/flare-up-glm/.venv
  log "ignition imports from $(cd "$W" && "${T[@]}" CUDA_VISIBLE_DEVICES= "$PY" -c 'import ignition;print(ignition.__file__)')"
  # #218: reused selection at every TP3 rank shape (24 heads), TP2 and TP4 (1 GPU). Gate.
  (cd "$W" && timeout 3600 "${T[@]}" CUDA_VISIBLE_DEVICES=0 "$PY" -m pytest -q -p no:cacheprovider \
     tests/recipes/test_glm53_mla_cuda.py -k reused_selection > "$R/gpu-218-mla.log" 2>&1); g1=$?
  # #217: the in-place path is the plain one-column KDA decode plan; at TP3 it runs 24 heads per rank (16 real on
  # rank 2), which the recorded verification path never exercised. world-3 decode case of the KDA TP test. Gate.
  K3=$(cd "$W" && "${T[@]}" CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q --collect-only -p no:cacheprovider \
       tests/recipes/test_glm53_kda_tp_cuda.py 2>/dev/null | grep -E '\[.*decode.*\]' | grep -E '(\[3-|-3\])')
  if [ -n "$K3" ]; then
    (cd "$W" && timeout 3600 "${T[@]}" CUDA_VISIBLE_DEVICES=0,1,2 "$PY" -m pytest -q -p no:cacheprovider $K3 \
       > "$R/gpu-217-kda-tp3.log" 2>&1); g2=$?
  else g2=9; echo "world-3 decode case not collected" > "$R/gpu-217-kda-tp3.log"; fi
  # #217: width one in place vs recorded across width-one / recorded / width-one rounds (TP4 head partition,
  # layer 0). Informational: the PR's own test, not at TP3 shapes.
  (cd "$W" && timeout 3600 "${T[@]}" CUDA_VISIBLE_DEVICES=0 "$PY" -m pytest -q -p no:cacheprovider \
     tests/recipes/test_glm53_mtp_state_cuda.py > "$R/gpu-217-mtp-state.log" 2>&1); g3=$?
  log "GPU tests: 218 mla reused selection exit $g1 ($(tail -1 "$R/gpu-218-mla.log" | cut -c1-80)) | 217 KDA TP3 decode exit $g2 ($(tail -1 "$R/gpu-217-kda-tp3.log" | cut -c1-80)) | 217 mtp state exit $g3 (info: $(tail -1 "$R/gpu-217-mtp-state.log" | cut -c1-80))"
  if [ $g1 != 0 ] || [ $g2 != 0 ]; then
    [ "${SKIP_GATE:-0}" = 1 ] && log "gate failed, SKIP_GATE=1: exporting anyway" || { log "gate failed: no export (SKIP_GATE=1 overrides)"; exit 1; }
  fi
  free=$(df --output=avail -BG "$(dirname "$M")" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 150 ] || { log "only ${free} GB free on the package store: no export (handbook D.2.9)"; exit 1; }
  t0=$(date +%s); (cd "$W" && timeout 28800 ./tp3-tools/glm53-export-tp3.sh > "$R/export.log" 2>&1); erc=$?
  CAND=$(grep -o "$M/gb202-tp3-[0-9]*" "$R/export.log" | tail -1)
  log "export exit $erc after $(( ($(date +%s)-t0)/60 )) min: ${CAND:-none} $( [ $erc = 0 ] || tr '\r' '\n' < "$R/export.log" | grep -E 'Error' | tail -1 | cut -c1-200)"
  [ $erc = 0 ] && [ -n "$CAND" ] || exit 1
  for P in "$BASE" "$CAND"; do
    (cd "$W" && timeout 900 ./flare/target/release/flare run "$P" --gpus 0,1,2 $GF --dry-run > "$R/dryrun-$(basename "$P").log" 2>&1)
    log "dry run $(basename "$P"): $(grep -o '[0-9,]* KV tokens' "$R/dryrun-$(basename "$P").log" | tail -1)"
  done
  for round in 1 2 3; do for pair in "base:$BASE" "cand:$CAND"; do L=${pair%%:*}; P=${pair#*:}
    serve "$L-r$round" "$P" "$W" || continue
    s=$(smoke "$L-r$round"); metrics "$R/metrics-$L-r$round-0.txt"
    (cd "$BENCH" && .venv/bin/python llm_decode_bench.py --port 15015 --model GLM-5.3-Flash --concurrency 1,4,8 \
       --contexts 32k --duration 60 --display-mode plain --output "$R/$L-32k-r$round.json" > "$R/$L-32k-r$round.log" 2>&1)
    metrics "$R/metrics-$L-r$round-1.txt"
    log "bench $L r$round: $(python3 "$V3/tp3-tools/bench_cell.py" "$R/$L-32k-r$round.json") | widths $(widths "$R/metrics-$L-r$round-0.txt" "$R/metrics-$L-r$round-1.txt") | smoke $s"
  done; done
  python3 - "$R" "$MIN_GAIN" "$MAX_LOSS" > "$R/decision.txt" <<'EOF'
import glob, json, sys
R, gain, loss = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
def cells(label):
    out = {}
    for path in sorted(glob.glob(f"{R}/{label}-32k-r*.json")):
        for r in json.load(open(path))["results"]:
            if (r.get("aggregate_tps") or 0) > 0:
                out.setdefault(r["concurrency"], []).append(r["aggregate_tps"])
    return {c: sum(v) / len(v) for c, v in out.items()}
b, c = cells("base"), cells("cand")
deltas = {k: 100 * (c[k] / b[k] - 1) for k in sorted(b) if k in c}
mean = sum(deltas.values()) / max(len(deltas), 1)
faster = len(deltas) == 3 and mean >= gain and min(deltas.values()) >= loss
print(" ".join(f"c{k} {b[k]:.0f}->{c[k]:.0f} ({d:+.1f}%)" for k, d in deltas.items()), f"mean {mean:+.1f}%",
      "FASTER" if faster else "NOT-FASTER")
EOF
  log "A/B (3 rounds, means): $(cat "$R/decision.txt")"
  if grep -q " FASTER" "$R/decision.txt"; then
    if serve qual-cand "$CAND" "$W"; then
      QUAL_OUT=$R/qual-b LABEL=glm-prs-cand QC=8 FLARE_LOG=$SERVE_LOG CKPT=$CKPT "$QUAL" eval tools hotel > "$R/qual-b.log" 2>&1
      log "candidate qualification (eval/tools/hotel c8):"; sed -n '/qualification start/,$p' "$R/qual-b/QUAL.md" | cut -d'—' -f2- | tee -a "$R/CHAIN.md"
      log "server log parser failures (candidate): $(parser_failures "$SERVE_LOG")"
    fi
  else
    log "candidate not faster: no qualification run (handbook D.6 rule 0)"
  fi
fi

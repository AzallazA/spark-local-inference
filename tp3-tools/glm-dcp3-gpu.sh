#!/bin/bash
# GPU stages for GLM-5.3-Flash decode context parallelism over three ranks (DCP3), branch glm53-tp3-dcp3.
# Written 2026-11-09 after the CPU stage (stage 0 of ~/ds41-tp3/analysis/glm-dcp3-design-20261108.md 4.3);
# not yet run. Stops at the first failing stage; the DeepSeek default server is restored at the end (and on any exit).
#
#   tp3-tools/glm-dcp3-gpu.sh                 # stages 1 2 3 4
#   STAGES="1 2" tp3-tools/glm-dcp3-gpu.sh    # kernel and collective tests only (~25 min, no export)
#   STAGES="4" CAND=<package> tp3-tools/glm-dcp3-gpu.sh   # rerun the A/B against an existing DCP3 package
#
# Stage 1, 1 GPU (~15 min): b12x sparse MLA GLM_NEXT at 72 heads with natural LSE (three token shards combine to the
#   whole-context result; empty shards and inactive rows give LSE -inf and zero output; 72 heads agree with 3 x 24;
#   rows 1/6/8/48/64/96 with the planned split counts printed), the existing head-count test (16/24/32 pass, 22
#   strict xfail), and the dcp.py kernels on the GPU at worlds 1-4 (test_glm53_dcp_cuda.py,
#   test_glm53_dcp_records_cuda.py; the interpreter versions passed on CPU).
# Stage 2, 3 GPUs (~10 min): b12x all_gather_heads (24 x 512 per rank -> 72 heads), lse_reduce_scatter (72 heads)
#   at rows 1..48 and the 6144-lane candidate gather at rows 1..512 through PreparedAllGather and
#   PreparedLseReduceScatter, against torch references, with -inf LSE ranks and rows; the >8-row world-3
#   reduce-scatter (vLLM never ran it natively) is the main risk. Then CUDA-graph timing of each exchange. Gate:
#   exposed time per MLA layer (candidates + LSE reduce-scatter + half the query gather, which the query chain
#   overlaps) <= 35 us at 6 rows (c1 verify) and <= 130 us at 48 rows (c8 verify). SKIP_GATE=1 continues anyway.
# Stage 3, export + correctness gate (~1.5-2 h): GLM has no truncated mini view (the recipe checks 45 layers and
#   the MTP layer 45; DS41's FLARE_DS41_MINI has no GLM counterpart), so the wiring check the handbook's mini
#   export would give (D.2.1) runs on the full export: DCP=1 a16 export with the 082232 flags (glm53-export-tp3.sh
#   defaults: chunk 4096, 5 drafts, concurrency 8, static tuning, language model only), plan line and page size
#   checks, then serve it and the base package on this worktree's binary for smoke, consistency (decode vs
#   teacher-forced prefill, 48 x 4) and verify_fix (20); the 43-question eval on DCP3. Gate: smoke 391, 0 TRUE
#   corruptions, consistency top-1 agreement within 0.02 of the base's and no worse p99 than 2x the base's.
# Stage 4, KV and speed (~1.5-2 h): dry runs (expect about 12.3M KV tokens vs 4.21M, 768 tokens per page), three
#   alternating rounds base/DCP3 (decode c1/c4/c8 at 32K, 60 s) on this binary, then long retrieval 128K/900K on
#   DCP3. Design gate (reported, not enforced): c1 >= 94 % and c8 >= 92 % of the base.
#
# Serving uses this worktree's flare binary for both packages: the DCP3 package's 2^20 context is not a whole
# number of 768-token pages, which the old Helios catalog check refused (helios/src/model/catalog.rs). Build it
# first: cargo build --release --manifest-path flare/Cargo.toml (built 2026-11-09 on CPU).
# Never run this while another chain or export is queued on the GPUs (handbook C.4, D.2.9); it waits for running
# exports and chain scripts first.
set -uo pipefail
W=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
V3=$HOME/flare-up-v3                      # bench_cell.py only (read-only)
BENCH=$HOME/llm-inference-bench
QUAL=${QUAL:-$HOME/flare-up-glm-qual/tp3-tools/glm-qual/qual_all.sh}
CKPT=${CKPT:-/path/to/ckpt/glm53-fp4-local}
BASE=${BASE:-/path/to/archive-a/flare-packages/models/glm53-fp4-local/gb202-tp3-202611082232}
M=${M:-$HOME/.cache/flare/packages/models/glm53-fp4-local}   # store the export writes to (CONFIG unset)
STAGES=${STAGES:-1 2 3 4}; CAND=${CAND:-}
GF="--port 15015 --host 127.0.0.1 --served-model-name GLM-5.3-Flash --prefix-cache-memory 32G --mtp-draft-tokens 5"
R=$HOME/bench-results/glm-dcp3-$(date +%m%d-%H%M); mkdir -p "$R"
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH HF_HUB_OFFLINE=1
K=$HOME/.cache/flare/kernels; export TRITON_CACHE_DIR=$K/triton B12X_COMPILE_CACHE_DIR=$K/b12x TORCHINDUCTOR_CACHE_DIR=$K/inductor

log(){ echo "$(date +%H:%M) $*" | tee -a "$R/CHAIN.md"; }
pids(){ ps -eo pid,args | awk '$2 ~ /(^|\/)flare$/ && $3=="run"{print $1}'; }
stop(){ for p in $(pids); do kill -INT $p; done; for i in $(seq 1 60); do [ -z "$(pids)" ] && break; sleep 3; done
        for p in $(pids); do kill -KILL $p; done
        until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 2; done; }
restore(){ stop; /tmp/restart-default.sh > /dev/null 2>&1; log "default restored; glm dcp3 chain done"; }
fail(){ log "STAGE $1 FAILED: $2"; exit 1; }
# serve LABEL PACKAGE -> 0 when /v1/models answers (archive drive A loads are slower than the boot drive: 30 min)
serve(){ stop; SERVE_LOG=$R/serve-$1.log
  (cd "$W" && nohup ./flare/target/release/flare run "$2" --gpus 0,1,2 $GF > "$SERVE_LOG" 2>&1 &)
  for i in $(seq 1 360); do sleep 5; curl -s -m 2 localhost:15015/v1/models | grep -q '"id"' && return 0; [ -z "$(pids)" ] && break; done
  log "serve $1 FAILED: $(tail -2 "$SERVE_LOG" | tr '\n' ' ' | cut -c1-200)"; return 1; }
smoke(){ curl -s -m 600 localhost:15015/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 17*23? Reply with just the number."}],"max_tokens":2000,"reasoning_effort":"low"}' > "$R/smoke-$1.json"
  python3 -c "import json;print(json.load(open('$R/smoke-$1.json'))['choices'][0]['message']['content'].strip()[:30])" 2>&1 | tail -1; }
# tier-1 checks on the running server: consistency (48 tokens x 4 prompts) and verify_fix (20 runs)
tier1(){ QUAL_OUT=$R/qual-$1 LABEL=glm-dcp3-$1 FLARE_LOG=$SERVE_LOG CKPT=$CKPT "$QUAL" consistency verify ${2:-} > "$R/qual-$1.log" 2>&1
  log "tier 1 $1: $(grep -h 'consistency:\|verify_fix:\|eval-43:' "$R/qual-$1/QUAL.md" | cut -d'—' -f2- | tr '\n' '|')"; }

# --- wait for the GPUs' other users (exports and chains), never stop them --------------------------------------
while ps -eo args | grep -qE '[i]gnition export|[c]hain-?[a-z0-9-]*\.sh'; do
  [ "${waited:-0}" = 0 ] && log "waiting for: $(ps -eo args | grep -E '[i]gnition export|[c]hain-?[a-z0-9-]*\.sh' | cut -c1-120 | tr '\n' ';')"
  waited=1; sleep 120; done
# Docker always runs unrelated services here (gitea, monitoring); only refuse if a container holds a GPU.
for c in $(docker ps -q 2>/dev/null); do docker inspect --format '{{.Name}} {{json .HostConfig.DeviceRequests}}' "$c" 2>/dev/null; done | grep -qi gpu \
  && { log "a docker container requests GPUs (vLLM?): stop it first"; exit 2; }
trap restore EXIT   # set after the wait, so an interrupted wait leaves the default server alone
log "chain start: worktree $W ($(git -C "$W" log -1 --format=%h)), stages: $STAGES, base $BASE"
[ -x "$W/flare/target/release/flare" ] || { log "no release binary in $W (cargo build --locked --release --manifest-path flare/Cargo.toml)"; exit 2; }
PY=$W/.venv/bin/python
T=(/usr/bin/env FLARE_RUN_CUDA_TESTS=1 FLARE_GLM53_SNAPSHOT=$CKPT PYTHONPATH=${B12X_OVERLAY:+$B12X_OVERLAY:}$W)
log "ignition imports from $(cd "$W" && "${T[@]}" CUDA_VISIBLE_DEVICES= "$PY" -c 'import ignition;print(ignition.__file__)')"
stop

# --- stage 1: sparse MLA at 72 heads with LSE; dcp.py kernels on the GPU (1 GPU) --------------------------------
if [[ " $STAGES " == *" 1 "* ]]; then
  (cd "$W" && timeout 3600 "${T[@]}" CUDA_VISIBLE_DEVICES=0 "$PY" -m pytest -q -s -p no:cacheprovider \
     tests/recipes/test_glm53_sparse_mla_heads_cuda.py > "$R/s1-sparse-mla.log" 2>&1); a=$?
  (cd "$W" && timeout 3600 "${T[@]}" CUDA_VISIBLE_DEVICES=0 "$PY" -m pytest -q -p no:cacheprovider \
     tests/recipes/test_glm53_dcp_cuda.py tests/recipes/test_glm53_dcp_records_cuda.py > "$R/s1-dcp-kernels.log" 2>&1); b=$?
  log "stage 1: sparse MLA exit $a ($(tail -1 "$R/s1-sparse-mla.log" | cut -c1-80)) | splits $(grep -o '72 heads, [0-9]* rows.*' "$R/s1-sparse-mla.log" | sed 's/72 heads, //;s/(chunks, splits, chunks per split) = //' | tr '\n' ';') | dcp kernels exit $b ($(tail -1 "$R/s1-dcp-kernels.log" | cut -c1-80))"
  [ $a = 0 ] && [ $b = 0 ] || fail 1 "see $R/s1-*.log"
fi

# --- stage 2: world-3 exchanges, correctness then timing (3 GPUs) -----------------------------------------------
if [[ " $STAGES " == *" 2 "* ]]; then
  (cd "$W" && timeout 3600 "${T[@]}" CUDA_VISIBLE_DEVICES=0,1,2 FLARE_DCP3_TIMING=$R/s2-timing.json "$PY" -m pytest \
     -q -s -p no:cacheprovider tests/recipes/test_glm53_dcp3_collectives_cuda.py > "$R/s2-collectives.log" 2>&1); a=$?
  log "stage 2: collectives exit $a ($(tail -1 "$R/s2-collectives.log" | cut -c1-80)) | $(grep -E 'verify \(' "$R/s2-collectives.log" | tr '\n' ';')"
  [ $a = 0 ] && [ -f "$R/s2-timing.json" ] || fail 2 "see $R/s2-collectives.log"
  gate=$(python3 - "$R/s2-timing.json" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))
def exposed(rows, candidate_rows):
    g, l, c = t["head_gather"][str(rows)], t["lse_reduce_scatter"][str(rows)], t["candidate_gather"][str(candidate_rows)]
    return c + l + g / 2, c + l + g
c1, c1_all = exposed(6, 6)
c8, c8_all = exposed(48, 48)
ok = c1 <= 35 and c8 <= 130
print(f"{'PASS' if ok else 'FAIL'} c1 6 rows {c1:.1f} us exposed ({c1_all:.1f} all three) <= 35 | "
      f"c8 48 rows {c8:.1f} us exposed ({c8_all:.1f}) <= 130; per call 1/8/48 rows: gather "
      f"{t['head_gather']['1']:.1f}/{t['head_gather']['8']:.1f}/{t['head_gather']['48']:.1f}, lse "
      f"{t['lse_reduce_scatter']['1']:.1f}/{t['lse_reduce_scatter']['8']:.1f}/{t['lse_reduce_scatter']['48']:.1f}, candidates "
      f"{t['candidate_gather']['1']:.1f}/{t['candidate_gather']['8']:.1f}/{t['candidate_gather']['48']:.1f}/{t['candidate_gather']['512']:.1f}")
EOF
)
  log "stage 2 timing gate: $gate"
  if [[ "$gate" != PASS* ]]; then
    [ "${SKIP_GATE:-0}" = 1 ] && log "timing gate failed, SKIP_GATE=1: continuing" || fail 2 "timing gate (SKIP_GATE=1 overrides)"
  fi
fi

# --- stage 3: DCP3 export, then serve-level correctness before any benchmark -----------------------------------
if [[ " $STAGES " == *" 3 "* ]]; then
  stop
  free=$(df --output=avail -BG "$(dirname "$M")" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 150 ] || fail 3 "only ${free} GB free on the package store (handbook D.2.9)"
  t0=$(date +%s)
  (cd "$W" && DCP=1 FLARE_GLM53_EXPERT_ACTIVATION=a16 timeout 28800 ./tp3-tools/glm53-export-tp3.sh > "$R/export.log" 2>&1); erc=$?
  CAND=$(grep -o "$M/gb202-tp3-[0-9]*" "$R/export.log" | tail -1)
  log "export exit $erc after $(( ($(date +%s)-t0)/60 )) min: ${CAND:-none} $( [ $erc = 0 ] || tr '\r' '\n' < "$R/export.log" | grep -E 'Error' | tail -1 | cut -c1-200)"
  [ $erc = 0 ] && [ -n "$CAND" ] || fail 3 "export"
  log "export plan: $(tr '\r' '\n' < "$R/export.log" | grep -iE 'context.parallel|page.tokens|tokens per page' | head -3 | cut -c1-160 | tr '\n' '|') pages: $(tr '\r' '\n' < "$R/export.log" | grep -oE '[0-9.]+ GiB per GPU · [0-9,]+ tokens' | tail -1)"
  tr '\r' '\n' < "$R/export.log" | grep -qiE 'context.parallel.{0,4}(True|true|on)' || log "warning: no 'context_parallel True' in the export log; check the plan line"
  echo "$CAND" > "$R/candidate.txt"
  serve dcp3-check "$CAND" || fail 3 "serve DCP3"
  s=$(smoke dcp3-check); log "DCP3 smoke: $s | KV $(curl -s localhost:15015/metrics | awk '/^flare_total_kv_tokens/{print $2}')"
  tier1 dcp3 eval
  serve base-check "$BASE" || fail 3 "serve base"
  sb=$(smoke base-check); log "base smoke: $sb"
  tier1 base
  verdict=$(python3 - "$R" "$s" <<'EOF'
import re, sys
R, smoke = sys.argv[1], sys.argv[2]
def read(label):
    text = open(f"{R}/qual-{label}/QUAL.md").read()
    agree = re.search(r"top-1 agree ([0-9.]+)", text); p99 = re.search(r"p99 ([0-9.]+)", text)
    bad = re.search(r"TRUE corruptions (\d+)", text)
    return (float(agree[1]) if agree else -1, float(p99[1]) if p99 else 1e9, int(bad[1]) if bad else -1)
c, b = read("dcp3"), read("base")
ok = "391" in smoke and c[2] == 0 and c[0] >= b[0] - 0.02 and c[1] <= 2 * max(b[1], 0.01)
print(f"{'PASS' if ok else 'FAIL'} smoke {smoke!r}; DCP3 agree {c[0]:.3f} p99 {c[1]:.4f} corruptions {c[2]} | "
      f"base agree {b[0]:.3f} p99 {b[1]:.4f} corruptions {b[2]}")
EOF
)
  log "stage 3 correctness gate: $verdict"
  [[ "$verdict" == PASS* ]] || fail 3 "correctness gate"
fi

# --- stage 4: dry-run KV, alternating A/B, long retrieval ------------------------------------------------------
if [[ " $STAGES " == *" 4 "* ]]; then
  [ -n "$CAND" ] || CAND=$(cat "$R/candidate.txt" 2>/dev/null)
  [ -n "$CAND" ] && [ -d "$CAND" ] || fail 4 "no DCP3 package (set CAND=<package>)"
  stop
  for P in "$BASE" "$CAND"; do
    (cd "$W" && timeout 900 ./flare/target/release/flare run "$P" --gpus 0,1,2 $GF --dry-run > "$R/dryrun-$(basename "$P").log" 2>&1)
    log "dry run $(basename "$P"): $(grep -o '[0-9,]* KV tokens' "$R/dryrun-$(basename "$P").log" | tail -1) | $(grep -oiE '[0-9,]+ tokens per page|page[_ ]tokens[^,]*' "$R/dryrun-$(basename "$P").log" | head -1)"
  done
  for round in 1 2 3; do for pair in "base:$BASE" "dcp3:$CAND"; do L=${pair%%:*}; P=${pair#*:}
    serve "$L-r$round" "$P" || continue
    s=$(smoke "$L-r$round")
    (cd "$BENCH" && .venv/bin/python llm_decode_bench.py --port 15015 --model GLM-5.3-Flash --concurrency 1,4,8 \
       --contexts 32k --duration 60 --display-mode plain --output "$R/$L-32k-r$round.json" > "$R/$L-32k-r$round.log" 2>&1)
    log "bench $L r$round: $(python3 "$V3/tp3-tools/bench_cell.py" "$R/$L-32k-r$round.json") | KV $(curl -s localhost:15015/metrics | awk '/^flare_total_kv_tokens/{print $2}') | smoke $s"
  done; done
  python3 - "$R" > "$R/decision.txt" <<'EOF'
import glob, json, sys
R = sys.argv[1]
def cells(label):
    out = {}
    for path in sorted(glob.glob(f"{R}/{label}-32k-r*.json")):
        for r in json.load(open(path))["results"]:
            if (r.get("aggregate_tps") or 0) > 0:
                out.setdefault(r["concurrency"], []).append(r["aggregate_tps"])
    return {c: sum(v) / len(v) for c, v in out.items()}
b, c = cells("base"), cells("dcp3")
ratio = {k: c[k] / b[k] for k in sorted(b) if k in c}
ok = len(ratio) == 3 and ratio.get(1, 0) >= 0.94 and ratio.get(8, 0) >= 0.92
print(" ".join(f"c{k} {b[k]:.0f}->{c[k]:.0f} ({100 * (r - 1):+.1f}%)" for k, r in ratio.items()),
      "WITHIN-DESIGN-GATE" if ok else "BELOW-DESIGN-GATE (c1 >= 94 %, c8 >= 92 % of base)")
EOF
  log "A/B (3 rounds, means): $(cat "$R/decision.txt")"
  if serve retrieval "$CAND"; then
    QUAL_OUT=$R/qual-retrieval LABEL=glm-dcp3 FLARE_LOG=$SERVE_LOG CKPT=$CKPT "$QUAL" retrieval > "$R/qual-retrieval.log" 2>&1
    log "DCP3 long retrieval: $(grep -h 'retrieval' "$R/qual-retrieval/QUAL.md" | tail -1 | cut -d'—' -f2-)"
  fi
fi

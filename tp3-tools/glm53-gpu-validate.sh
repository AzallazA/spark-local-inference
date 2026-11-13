#!/bin/bash
# GLM 5.3 Flash TP3: the GPU validation of the padded three-rank partition as one scripted run.
#
#   tp3-tools/glm53-gpu-validate.sh LOGDIR
#
# Runs from the worktree root with its .venv. Steps run in order and the script stops at the first
# failure; every step logs to LOGDIR/NN-name.log and prints one summary line. Steps 1-4 need one free
# GPU (CUDA_VISIBLE_DEVICES=0); steps 5-6 need three (0,1,2) and spawn one rank process per GPU.
# The checkpoint is the plain directory FLARE_GLM53_SNAPSHOT (default /path/to/ckpt/glm53-fp4-local,
# revision 76348ddb); the tests skip when it is absent, so a run full of skips means the path is wrong.
# Check the GPUs before using them (TP3-GUIDELINES C.1): the script refuses GPUs with more than
# $BUSY_MIB (default 2000) MiB in use unless FORCE=1.
#
# What each step proves (CPU tests already cover the loader arithmetic; these run the kernels):
#   1 nvidia_weights   NVIDIA checkpoint partitions at TP1/2/3/4: padded block-FP8/MXFP8/BF16 shards equal the
#                      checkpoint rows with zero tails (rank 2 of 3 holds heads 44..65, columns 1536..2303).
#   2 weights          NVFP4-CSF experts: packed w13/w2 columns and the expanded CSF scales; rank 2's padded
#                      rows/columns must decode to zero (CsfSource padded_rows/padded_columns, padding_base 0),
#                      plus the MTP layer's experts and attention; also at 704 routed columns per rank
#                      (FLARE_GLM53_EXPERT_COLUMNS=704: ranks 0/1 exact, rank 2 holds 1408..2047 + 64 zero).
#   3 moe_overlap      routed + shared expert graph at 768 columns on the last rank (b12x MoE kernel acceptance),
#                      and at 704 routed / 768 shared columns ([3-704]).
#   3b sparse_mla_heads multi-row sparse decode at 16/22/24/32 heads: inactive rows zero, rows independent
#                      (the 22-head two-row case is where step 4 first failed; 24 is the 24/24/16 alternative).
#   4 mla              sparse MLA replay with 22 heads on the last rank (absorb/expand BMM and sparse_mla tiling).
#   5 kda_tp           three-rank KDA graphs with the PCIe all-reduce, prefill and decode, 22 heads per rank.
#   6 mlp_tp           three-rank dense (4096) and shared-expert (768) MLP graphs with the PCIe all-reduce.
set -euo pipefail
LOGDIR=${1:?usage: glm53-gpu-validate.sh LOGDIR}
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT"
mkdir -p "$LOGDIR"

export PYTHONPATH="$ROOT" TORCH_EXTENSIONS_DIR="$ROOT/.brrr-ext" CUDA_HOME=/usr/local/cuda-13.1
export PATH="/usr/local/cuda-13.1/bin:$PATH"
# Pin every kernel cache the way `ignition export` does (ignition/cli/environment.py apply()). Tests that
# tune then compile_forward() a planned Triton kernel need TRITON_CACHE_DIR set up front: b12x compiles the
# planned program into Triton's cache during session.tune(), and the first Inductor kernel compiled by
# compile_forward() otherwise points TRITON_CACHE_DIR at /tmp/torchinductor_<user>/triton/<device>
# (torch/_inductor/runtime/triton_heuristics.py, `if os.getenv("TRITON_CACHE_DIR") is None`), so the deferred
# load then fails with "planned Triton program has not been compiled" (glm_pool _fwht_quant_kernel, step 04).
KERNEL_CACHE=${KERNEL_CACHE:-$HOME/.cache/flare/kernels}
export TRITON_CACHE_DIR="$KERNEL_CACHE/triton" B12X_COMPILE_CACHE_DIR="$KERNEL_CACHE/b12x" \
       TORCHINDUCTOR_CACHE_DIR="$KERNEL_CACHE/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$B12X_COMPILE_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export FLARE_GLM53_SNAPSHOT=${FLARE_GLM53_SNAPSHOT:-/path/to/ckpt/glm53-fp4-local}
export FLARE_RUN_CUDA_TESTS=1 FLARE_RUN_TP_CUDA_TESTS=1 HF_HUB_OFFLINE=1
PYTEST="$ROOT/.venv/bin/python -m pytest -q -p no:cacheprovider -x"

[ -d "$FLARE_GLM53_SNAPSHOT" ] || { echo "checkpoint directory $FLARE_GLM53_SNAPSHOT is missing" >&2; exit 2; }
[ -f "$FLARE_GLM53_SNAPSHOT/model.safetensors.index.json" ] || \
  echo "warning: $FLARE_GLM53_SNAPSHOT has no model.safetensors.index.json yet (download incomplete?)" >&2

check_gpus() {
  # $1 = comma-separated GPU indices; refuse GPUs another job holds.
  local busy
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$1" | awk -F', ' -v limit="${BUSY_MIB:-2000}" '$2 > limit {print "GPU " $1 " uses " $2 " MiB"}')
  if [ -n "$busy" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "GPUs are in use (set FORCE=1 to override):" >&2; echo "$busy" >&2; exit 3
  fi
}

step() {
  # step NN name gpus pytest-args...
  local number=$1 name=$2 gpus=$3; shift 3
  local log="$LOGDIR/$number-$name.log" start status summary
  start=$(date +%s)
  if CUDA_VISIBLE_DEVICES="$gpus" $PYTEST "$@" >"$log" 2>&1; then status=PASS; else status=FAIL; fi
  summary=$(grep -E "^[0-9]+ (passed|failed|skipped|error)|passed|failed" "$log" | tail -1)
  printf '%s %-16s gpus=%-5s %4ds  %s  (%s)\n' "$status" "$name" "$gpus" "$(( $(date +%s) - start ))" "${summary:-no pytest summary}" "$log"
  [ "$status" = PASS ] || { echo "stopping at the first failure; see $log" >&2; exit 1; }
}

echo "GLM 5.3 TP3 GPU validation: logs in $LOGDIR, checkpoint $FLARE_GLM53_SNAPSHOT, worktree $ROOT"
check_gpus 0
step 01 nvidia_weights 0 tests/recipes/test_glm53_nvidia_weights.py
step 02 weights        0 tests/recipes/test_glm53_weights.py -k "nvfp4_shards or mtp_experts"
step 03 moe_overlap    0 tests/recipes/test_glm53_moe_overlap.py
# Kernel-only check of multi-row sparse decode at 16/22/24/32 heads (inactive row zero, rows independent):
# isolates the [3-decode-2-*] failure of step 04 from the recipe. Both steps report every case (--maxfail).
step 03b sparse_mla_heads 0 tests/recipes/test_glm53_sparse_mla_heads_cuda.py --maxfail=50
step 04 mla            0 tests/recipes/test_glm53_mla_cuda.py --maxfail=50
check_gpus 0,1,2
step 05 kda_tp         0,1,2 tests/recipes/test_glm53_kda_tp_cuda.py
step 06 mlp_tp         0,1,2 tests/recipes/test_glm53_mlp_tp_cuda.py
echo "all steps passed; next: tp3-tools/glm53-export-tp3.sh"

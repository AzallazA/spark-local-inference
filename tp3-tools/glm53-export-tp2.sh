#!/bin/bash
# GLM-5.3-Flash at TP2 on two RTX PRO 6000: upstream's declared rank count (no TP3 padding involved: 32 heads,
# 1024 expert columns and 77440 vocabulary rows per rank, exact), with Flare's native decode context parallelism
# (DCP2, 512-token pages) left on. GPUS defaults to 0,1.
#
#   tp3-tools/glm53-export-tp2.sh                       # 11-10 defaults: a16, KDA side chain, overlay (push at world 2)
#   B12X_PCIE_DCP_GATHER_PULL=1 tp3-tools/glm53-export-tp2.sh   # world-2 pull gather (overlay, opt-in, unmeasured)
#   DCP=0 / CONC=4 / VISION=1 / CONFIG=<flare config.toml> / B12X_OVERLAY= (empty: no overlay)
#
# TP3 improvements carried over (2026-11-10, branch glm53-tp2-update; defaults overridable from the environment):
#   FLARE_GLM53_EXPERT_ACTIVATION=a16   A4 prefill / A16 decode experts. This is already the recipe default on SM120
#                           (a4 only on SM103), so the 11-08 package 082052 is a16 too; set explicitly to pin it.
#   FLARE_GLM53_KDA_SIDE_CHAIN=1   f_a -> f_b, b and g_a -> g_b beside the KDA q/k/v GEMM, no staging copies (TP3:
#                           +2.2/+1.3/+1.1 % c1/c4/c8, bitwise-identical outputs). At TP2: b_proj N=32 and f_a N=128
#                           GEMVs instead of one N=160 launch (new tuning shapes; the rest of the cache is warm).
#   B12X_OVERLAY            <worktree>/.b12x-overlay when present (tp3-tools/make-b12x-overlay.sh). Its pull gather is
#                           world-3 only under B12X_PCIE_DCP_GATHER_PULL=auto (default), so a TP2 export with the
#                           overlay compiles the same push kernels as without it; =1 selects the world-2 pull plane.
#   FLARE_GLM53_EMBED_HOST=1   (branch glm53-vram, opt-in, unmeasured) the replicated BF16 token embedding (1210 MiB per
#                           rank) lives in device-visible host memory; KV pages take the VRAM (~+0.38M KV tokens at TP2).
#   FLARE_GLM53_VISION_STAGE=1   (with VISION=1; GPU-validated 11-12, release setting with IMAGE_CONCURRENCY=2) the replicated
#                           vision tower (373 MiB per rank) in host memory, staged through 22 MiB of VRAM slots per image.
# Not carried over (rejected at TP3): NVFP4 draft head / MXFP8 eh_proj, a4 decode, LIL checkpoint, FP32 partials,
# 704 expert columns (TP3-only; TP2 splits 2048 exactly). The script refuses the draft-precision and column overrides.
#
# Check in the log: plan line (TP2, context_parallel True), free device bytes after weights (~95 - 83 GiB), KV pages.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT"
CKPT=${CKPT:-/path/to/ckpt/glm53-fp4-local}
[ -d "$CKPT" ] || { echo "checkpoint directory $CKPT is missing" >&2; exit 2; }
for name in DRAFT_HEAD DRAFT_EH_PROJ EXPERT_COLUMNS; do
  var=FLARE_GLM53_$name
  [ -z "${!var:-}" ] || { echo "$var=${!var} is a TP3 option (rejected or not applicable at TP2); unset it" >&2; exit 2; }
done
export FLARE_GLM53_EXPERT_ACTIVATION=${FLARE_GLM53_EXPERT_ACTIVATION:-a16}
export FLARE_GLM53_KDA_SIDE_CHAIN=${FLARE_GLM53_KDA_SIDE_CHAIN:-1}

B12X_OVERLAY=${B12X_OVERLAY-$( [ -d "$ROOT/.b12x-overlay/b12x_tree" ] && echo "$ROOT/.b12x-overlay" )}
export PYTHONPATH="${B12X_OVERLAY:+$B12X_OVERLAY:}$ROOT" TORCH_EXTENSIONS_DIR="$ROOT/.brrr-ext" CUDA_HOME=/usr/local/cuda-13.1
export PATH="/usr/local/cuda-13.1/bin:$PATH" HF_HUB_OFFLINE=1
TEXT_ONLY=(--language-model-only)
[ "${VISION:-0}" = "1" ] && TEXT_ONLY=()
# IMAGE_CONCURRENCY=<n> (with VISION=1): --image-concurrency n feature slots (64 MiB each per rank; default min(4, CONC)).
IMAGE_FLAGS=(); [ -n "${IMAGE_CONCURRENCY:-}" ] && IMAGE_FLAGS=(--image-concurrency "$IMAGE_CONCURRENCY")
CONFIG_FLAGS=(); [ -n "${CONFIG:-}" ] && CONFIG_FLAGS=(--config "$CONFIG")
# MTP_ONLY=1: pass --mtp-only (upstream b547413b): omit the ordinary (no-draft) graph set ("-no-mtp" prefill and
# resume/init/restore/snapshot graphs, plain decode graphs); the package then refuses --mtp-draft-tokens 0.
# Default 0 = unchanged export command.
case "${MTP_ONLY:-0}" in
  0) MTP_ONLY_FLAGS=() ;;
  1) MTP_ONLY_FLAGS=(--mtp-only) ;;
  *) echo "MTP_ONLY must be 0 or 1, not ${MTP_ONLY}" >&2; exit 2 ;;
esac
case "${DCP:-1}" in
  1) CP_FLAG=() ;;
  0) CP_FLAG=(--no-context-parallel) ;;
  *) echo "DCP must be 0 or 1, not ${DCP}" >&2; exit 2 ;;
esac
echo "glm53-export-tp2: GPUS=${GPUS:-0,1} DCP=${DCP:-1} activation=$FLARE_GLM53_EXPERT_ACTIVATION" \
     "kda_side_chain=$FLARE_GLM53_KDA_SIDE_CHAIN embed_host=${FLARE_GLM53_EMBED_HOST:-0} vision=${VISION:-0} vision_stage=${FLARE_GLM53_VISION_STAGE:-0} image_concurrency=${IMAGE_CONCURRENCY:-default} overlay=${B12X_OVERLAY:-none}" \
     "gather_pull=${B12X_PCIE_DCP_GATHER_PULL:-auto} commit=$(git -C "$ROOT" log -1 --format=%h 2>/dev/null)"
exec .venv/bin/ignition export "$CKPT" ${CONFIG_FLAGS[@]+"${CONFIG_FLAGS[@]}"} --gpus "${GPUS:-0,1}" \
  ${CP_FLAG[@]+"${CP_FLAG[@]}"} --prefill-chunk-size 4096 --mtp-draft-tokens 5 \
  --max-concurrency "${CONC:-8}" "${TEXT_ONLY[@]}" ${IMAGE_FLAGS[@]+"${IMAGE_FLAGS[@]}"} ${MTP_ONLY_FLAGS[@]+"${MTP_ONLY_FLAGS[@]}"} --verbose

#!/bin/bash
# First GLM 5.3 Flash TP3 export from the plain checkpoint directory (revision 76348ddb) on three GB202 GPUs.
#
#   tp3-tools/glm53-export-tp3.sh            # after tp3-tools/glm53-gpu-validate.sh passed
#   CONC=32 VISION=1 tp3-tools/glm53-export-tp3.sh
#   DCP=1 tp3-tools/glm53-export-tp3.sh      # decode context parallelism over the three ranks (DCP3, opt-in)
#
# Flags the glm53 recipe needs at world size 3:
#   --no-context-parallel   (default, DCP=0) three ranks shard heads and experts only. The recipe declares no
#                           context-parallel page size for three ranks unless FLARE_GLM53_DCP3=1 (recipe.py
#                           `_costs`), so the planner keeps it off anyway; the flag makes the choice explicit.
#   DCP=1                   exports with FLARE_GLM53_DCP3=1 and without --no-context-parallel: 768-token pages
#                           (256 local records on each rank), page tables rounded up to 1366 pages for 2^20,
#                           72-head query/LSE exchange and 3 x 512 candidate merge (dcp.py). Not qualified yet:
#                           tp3-tools/glm-dcp3-gpu.sh runs the GPU stages. The served context stays 1048576;
#                           serving needs a flare binary with the partial-last-page catalog check (this branch).
#   --prefill-chunk-size 4096   the recipe default; the TP3 prefill all-reduce is one PCIe runtime sized for
#                           this chunk (bootstrap.py, mirrored from deepseek41), 32 MiB per layer reduction.
#   --mtp-draft-tokens 5    the recipe's default drafts (Speculation(draft_tokens=5, token_capacity=128)).
#   --collective-tuning runtime --collective-ladder gb202_pcie_tp3.json
#                           the ladder is accepted at world 3: planning validates world_size (3) and compute
#                           capability (12, 0) only, and the GLM bootstrap passes collective_tuning to every
#                           PCIe all-reduce (joint kernels). The buckets are byte-sized, so DeepSeek's
#                           5120-wide ladder applies to 4096-wide rows.
#   --max-concurrency ${CONC:-8}   first export at 8 (as the DeepSeek TP3 release); the recipe default is 32.
#   --language-model-only   unless VISION=1: the replicated vision tower costs VRAM on every rank and is not
#                           part of the TP3 question.
# No --gpus auto-selection is involved: three GPUs are listed, which is also the only way a four-GPU host
# gets TP3 (recipe exact_rank_counts).
# Fork-only export overrides (recipe.py `_tp3_override`, unset = upstream choice) pass through the environment:
#   FLARE_GLM53_EXPERT_ACTIVATION=a4|a16, FLARE_GLM53_EXPAND_SCALES=1|0,
#   FLARE_GLM53_EXPERT_COLUMNS=768|704   routed experts 704 columns per rank instead of 768 (shared expert
#                           stays 768); new MoE shapes, so the first such export tunes them cold.
#   FLARE_GLM53_KDA_SIDE_CHAIN=1   (branch glm53-tp3-speed-dcp3) run f_a -> f_b, b and g_a -> g_b beside the KDA q/k/v GEMM
#                           and drop the two per-layer staging copies (same weights; ~6-8 us per KDA layer at c1).
#   FLARE_GLM53_DRAFT_HEAD=nvfp4   NVFP4 proposal head built from the MXFP8 head (draft passes only).
#   FLARE_GLM53_DRAFT_EH_PROJ=mxfp8   the draft layer's BF16 eh_proj as MXFP8 (draft passes only).
#   FLARE_GLM53_VISION_STAGE=1   (with VISION=1; GPU-validated 11-12, release setting with IMAGE_CONCURRENCY=4) park the
#                           replicated vision tower (373 MiB per rank) in device-visible host memory and stage it
#                           through 22 MiB of VRAM slots per image (DeepSeek's TP3 vision weight staging).
# MTP_ONLY=1   pass --mtp-only (upstream b547413b): omit the ordinary (no-draft) graph set ("-no-mtp" prefill and
#              resume/init/restore/snapshot graphs, plain decode graphs); vision and width-1 speculative rounds stay.
#              The package then refuses to load with 0 drafts. Default 0 = unchanged export command.
#
# Expect a cold tuning cache: every GEMM/MoE/attention shape of the 22-head, 768-column, 51648-row shards is
# new, so the first export autotunes everything (hours, as the first DeepSeek TP3 export did); the second
# export of the same shapes reuses the shared tuning cache. A cold cache also needs more free VRAM than
# serving; if tuning runs out of memory, retry with CONC=4.
#
# What to check in the export log (--verbose):
#   * the plan line: tensor_parallel 3, context_parallel False, prefill chunk 4096, drafts 5, concurrency.
#   * padded shard shapes on every rank: KDA q/k/v 2816 rows (22 heads), MLA q_b 5632 rows / kv_b 11264 rows,
#     experts intermediate 768 (plan geometry), shared gate/up 1536 rows, lm_head 51648 rows; rank 2's loads
#     report the padded tails (no "exceeds its ... checkpoint extent" error).
#   * collectives: one PCIe all-reduce runtime shared by decode and prefill (no DMA all-reduce line), the
#     vocabulary gather with 154944 columns, no head exchange / candidate gather (no DCP).
#   * "after weights": free device bytes per rank (expect roughly 96 GB - 63 GB weights); "after model
#     state" and the final memory report: KV pages per rank and the free bytes the planner left.
#   * the KV estimate versus the plan: pages * 256 tokens; compare with `flare run <pkg> --dry-run` later.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT"
CKPT=${CKPT:-/path/to/ckpt/glm53-fp4-local}
[ -d "$CKPT" ] || { echo "checkpoint directory $CKPT is missing" >&2; exit 2; }

# B12X_OVERLAY=<dir>: a patched b12x copy ahead of the venv. Default: <worktree>/.b12x-overlay when present (the
# world-3 pull gather, built by tp3-tools/make-b12x-overlay.sh); B12X_OVERLAY= (empty) exports without it.
B12X_OVERLAY=${B12X_OVERLAY-$( [ -d "$ROOT/.b12x-overlay/b12x_tree" ] && echo "$ROOT/.b12x-overlay" )}
export PYTHONPATH="${B12X_OVERLAY:+$B12X_OVERLAY:}$ROOT" TORCH_EXTENSIONS_DIR="$ROOT/.brrr-ext" CUDA_HOME=/usr/local/cuda-13.1
export PATH="/usr/local/cuda-13.1/bin:$PATH" HF_HUB_OFFLINE=1
TEXT_ONLY=(--language-model-only)
[ "${VISION:-0}" = "1" ] && TEXT_ONLY=()
# IMAGE_CONCURRENCY=<n> (with VISION=1): --image-concurrency n feature slots (64 MiB each per rank; default min(4, CONC)).
IMAGE_FLAGS=(); [ -n "${IMAGE_CONCURRENCY:-}" ] && IMAGE_FLAGS=(--image-concurrency "$IMAGE_CONCURRENCY")
# Static collective tuning, as the DS41 TP3 release export: the TP3 runtime ladder stops at 24 MiB and a
# 4096-row GLM prefill chunk reduces 32 MiB per call (export 1108-1557 failed "no bucket for 33554432 bytes").
# CONFIG=<flare config.toml> selects another package store (paths.package_cache), e.g. on a drive with more room.
CONFIG_FLAGS=(); [ -n "${CONFIG:-}" ] && CONFIG_FLAGS=(--config "$CONFIG")
LADDER_FLAGS=(); [ "${RUNTIME_LADDER:-0}" = "1" ] && LADDER_FLAGS=(--collective-tuning runtime --collective-ladder ignition/preparation/ladders/gb202_pcie_tp3.json)
# DCP3 is opt-in: the recipe declares 768-token context-parallel pages at three ranks only under FLARE_GLM53_DCP3=1.
case "${DCP:-0}" in
  0) CP_FLAGS=(--no-context-parallel); export FLARE_GLM53_DCP3=0 ;;
  1) CP_FLAGS=(); export FLARE_GLM53_DCP3=1 ;;
  *) echo "DCP must be 0 or 1, not ${DCP}" >&2; exit 2 ;;
esac
case "${MTP_ONLY:-0}" in
  0) MTP_ONLY_FLAGS=() ;;
  1) MTP_ONLY_FLAGS=(--mtp-only) ;;
  *) echo "MTP_ONLY must be 0 or 1, not ${MTP_ONLY}" >&2; exit 2 ;;
esac

exec .venv/bin/ignition export "$CKPT" ${CONFIG_FLAGS[@]+"${CONFIG_FLAGS[@]}"} --gpus "${GPUS:-0,1,2}" \
  ${CP_FLAGS[@]+"${CP_FLAGS[@]}"} --prefill-chunk-size 4096 --mtp-draft-tokens 5 \
  --max-concurrency "${CONC:-8}" "${TEXT_ONLY[@]}" ${IMAGE_FLAGS[@]+"${IMAGE_FLAGS[@]}"} \
  ${LADDER_FLAGS[@]+"${LADDER_FLAGS[@]}"} ${MTP_ONLY_FLAGS[@]+"${MTP_ONLY_FLAGS[@]}"} \
  --verbose

#!/bin/bash
# Export the package from a checkpoint view (make_checkpoint_view.sh). Run it from the fork checkout with its .venv
# (uv sync, then re-apply tp3-patches/ to the installed FlashInfer; Flare's torch/Triton fixes are applied in memory
# by ignition.runtime_patches, there is no separate ignition.patches step any more).
#   CKPT     checkpoint view directory (required)
#   HOT      hot experts per parked layer (default 342 = 2.37M KV; each expert fewer adds ~92K KV tokens, see EXPORT.md)
#   PREFILL  prefill chunk size (default 4096)
#   DRAFTS   DSpark drafts (default 5); serve with the same DRAFTS (serve.sh)
#   MAXC     maximum concurrent requests the package captures graphs for (default 8; 16 = the c16 package)
#   FLARE_DS41_VISION_STAGE 1 (default) = vision tower parked in host RAM and its weights staged through VRAM slots per
#            block (image 0.11 s instead of 1.70 s, no KV cost); 0 = no staging
#   FLARE_DS41_ENGRAM_CHUNK_ROWS 1024 (default) + FLARE_DS41_SHARE_ENGRAM_RESIDUAL 1 (default) = graph scratch: chunked Engram
#            projection and engram_residual on ffn_hc.residual (KV 2.46M -> 3.01M at HOT 342, decode level); 0/0 = previous layout
#   FLARE_DS41_VISION_HOST  1 (default) = vision tower in host RAM (implied by VISION_STAGE=1); VISION_HOST=0 together
#            with VISION_STAGE=0 keeps the tower in VRAM, which needs a lower HOT (unmeasured, no longer useful)
# A cold tuning cache needs more free VRAM than serving. If tuning runs out of memory, export once with HOT=300,
# then export the target. The second export reuses the shared tuning cache.
set -euo pipefail
: "${CKPT:?set CKPT to the checkpoint view directory}"
HERE=$(dirname "$(readlink -f "$0")")
export FLARE_DS41_TP3_GRAPH_GIB=1 FLARE_DS41_HOST_EXPERT_LAYERS=20-33 FLARE_DS41_HOT_EXPERTS=${HOT:-342} \
       FLARE_DS41_MERGED_BANK=1 FLARE_DS41_TP3_VOCAB_PARALLEL=1 \
       FLARE_DS41_EXPERT_COUNTS_DIR=${COUNTS:-$HERE/expert-order} HF_HUB_OFFLINE=1 \
       FLARE_DS41_BRRR_LAYERS=0-39 FLARE_DS41_BRRR_CKPT="$CKPT" FLARE_DS41_BRRR_PARKED_MAX_ROWS=12 \
       FLARE_DS41_EMBED_HOST=1 FLARE_DS41_VISION_HOST="${FLARE_DS41_VISION_HOST:-1}" FLARE_DS41_VISION_STAGE="${FLARE_DS41_VISION_STAGE:-1}" \
       FLARE_DS41_AR_ROWS=1024 FLARE_DS41_ENGRAM_CHUNK_ROWS="${FLARE_DS41_ENGRAM_CHUNK_ROWS:-1024}" \
       FLARE_DS41_SHARE_ENGRAM_RESIDUAL="${FLARE_DS41_SHARE_ENGRAM_RESIDUAL:-1}"
exec .venv/bin/ignition export "$CKPT" --gpus "${GPUS:-0,1,2}" --max-concurrency "${MAXC:-8}" \
  --prefill-chunk-size "${PREFILL:-4096}" --mtp-draft-tokens "${DRAFTS:-5}" --draft-head-quantization nvfp4 \
  --markov-head-quantization fp8 --vocabulary-parallel-sampling --verbose

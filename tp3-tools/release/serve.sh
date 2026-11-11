#!/bin/bash
# Serve the exported package on 3 GPUs with both Engram tables in host RAM (~97 GiB).
#   FLARE         path to this fork's flare binary   (default: flare/target/release/flare in the fork checkout)
#   PACKAGE       model name (= the checkpoint directory name the package was exported from) or a package path
#                 (default ds41-1017-engram-mxfp4 = the view name in EXPORT.md; Flare picks its default version)
#   ENGRAM_RAM=0  read the Engram tables from disk instead: ~90 GiB less host RAM, decode ~4% slower, prefill unchanged
#   PREFIX_CACHE  host RAM for reusable prompt KV (default 16G; 32G/48G cost only RAM)
#   BUDGET        max verification rows per step (default 24; 128 = no cap, ~1-2% slower at c8)
#   DRAFTS        DSpark drafts (default 5); must equal the DRAFTS the package was exported with
#   ALPHA         adaptive-verification alpha (default 0.8; 0.6 and 1.0 measured within noise)
#   SCHEDULER     scheduler policy (default interactive = balanced + a 50 ms first-token target: a new prompt waits
#                 ~0.37 s instead of 2-4 s behind 3-7 decoding streams, decode speed level; balanced = the previous default)
#   GPUS, PORT, HOST   default 0,1,2 / 15015 / 127.0.0.1
set -euo pipefail
FLARE=${FLARE:-flare/target/release/flare}
PACKAGE=${PACKAGE:-ds41-1017-engram-mxfp4}
ENV=() STORAGE=()
if [ "${ENGRAM_RAM:-1}" = 1 ]; then ENV=(FLARE_ENGRAM_RAM_TABLES=0); else STORAGE=(--embedding-table-storage disk); fi
# /usr/bin/env explicitly: some shells put another `env` (for example uv's) first on PATH.
exec /usr/bin/env "${ENV[@]}" "$FLARE" run "$PACKAGE" --gpus "${GPUS:-0,1,2}" --port "${PORT:-15015}" \
  --host "${HOST:-127.0.0.1}" --served-model-name DeepSeek-V4.1-Flash --prefix-cache-memory "${PREFIX_CACHE:-16G}" \
  --mtp-draft-tokens "${DRAFTS:-5}" --mtp-adaptive-verification --mtp-max-verification-token-budget "${BUDGET:-24}" \
  --mtp-adaptive-verification-alpha "${ALPHA:-0.8}" --scheduler-policy "${SCHEDULER:-interactive}" "${STORAGE[@]}"

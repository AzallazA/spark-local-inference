#!/bin/bash
# GLM-5.3-Flash on Flare TP3 - default serve (modeled on ~/flare-up-v3/serve-best.sh).
# 11-10 18:30: DEFAULT = 704 expert columns + KDA, pkg gb202-tp3-202611102126 (user): vs 101557 3 rounds c1 271 vs 261,
# c4 607 vs 575 (+5.6% every round), c8 833 vs 826, KV 14.70M vs 12.39M (+19%) (bench-results/glm-704-a16-1110-1731).
# Exported with FLARE_GLM53_EXPERT_COLUMNS=704 (no tile patch). Previous: 101557.
# 11-10 19:25: fallbacks 101557 (768+KDA), 092348 (768 DCP3), 092215 (DCP1) and 102210 (704+patch) archived to archive drive B:
#   P=/path/to/archive-b/flare-packages/models/glm53-fp4-local/gb202-tp3-<version> (serve by path, or copy back to the boot store).
# 11-10 13:00: DEFAULT = KDA side chain on DCP3, pkg gb202-tp3-202611101557 (user): vs 092348 3 rounds c1 +2.2% / c4 +1.3% /
# c8 +1.1%, KV 12.39M same, eval 43/43, tools 60/60, verify 0; KDA side chain bitwise-identical in the kernel tests
# (bench-results/glm-speed-dcp1-1110-0815). Previous default: DCP3 092348 (P=.../gb202-tp3-202611092348).
# 11-10 06:10: DCP3 092348 default again: drift explained (bench-results/glm-dcp3-drift-1110-0542). At 1500-token
# context DCP3 and DCP1 agree equally with prefill (top-1 0.927 vs 0.929); the short-context gap is one extra BF16
# rounding of per-rank partial attention, which DCP1 also gets once attention splits (DCP1 unsplit at 1500: 0.946).
# History, 11-09 21:15: default back to DCP1 092215 (user) until the DCP3 decode drift is explained: larger consistency check
# (~1,000 positions, greedy) DCP3 top-1 0.955/0.949 vs DCP1 0.974, p99 |dlogprob| 0.83 vs 0.39, one deterministic
# position where DCP3 decode picks a token outside prefill top-5 (max 3.32). DCP3 returns as default once resolved.
# DCP3 package (P=.../gb202-tp3-202611092348): DCP3 = FP4-Local a16 + world-3 pull gather + decode context parallelism over the three
# ranks, gb202-tp3-202611092348 (768-token pages; needs this worktree's binary: Helios partial-page fix). vs DCP1 092215
# (3 alternating rounds, 450 W, 32K): c1 266->262 / c4 582->562 / c8 829->819 tok/s (rounds/s c1 ~-7 %, c4/c8 ~-2 %),
# KV 12.39M vs 4.21M (2.94x); eval 43/43, verify_fix 0, long retrieval 128K + 900K pass; consistency top-1 0.943 vs
# 0.974 on 192 positions (borderline, larger run owed).
# DCP1 alternative (kept): P=$HOME/.cache/flare/packages/models/glm53-fp4-local/gb202-tp3-202611092215
# (a16 + pull gather, no DCP: c1 249->267 / c4 533->577 / c8 758->821 vs 082232; eval 43/43, tools 60/60, hotel 28/0/2).
# Older: 082232 (archive drive A). a4 (+7% c1) not adopted: prose ppl +1.9%.
# Binary: this worktree (~/flare-up-glm: #59/#179 tool-argument fixes + </arg_value>-in-value fix); falls back to ~/flare-up-v3 if that build is missing.
# Overrides: P (package), K (drafts, default 5), PCM (host prefix cache, default 32G), SCHED (balanced; interactive = TTFT guard, see docs/tp3/GLM53-TP3-PLAN.md),
# PORT (15015), FLARE_BIN, GPUS (0,1,2; a TP2 package needs GPUS=0,1); BUDGET / ALPHA add the verification budget / alpha flags (default: package setting).
P=${P:-$HOME/.cache/flare/packages/models/glm53-fp4-local/gb202-tp3-202611112036}   # 11-11 17:40 default: c32 pkg with PR #253 + --mtp-only (KV 13.42M; needs this worktree's binary, K>0); before: gb202-tp3-202611110835; c8 = gb202-tp3-202611102126
ALPHA=${ALPHA-0.8}   # 11-11 c32 sweep winner; ALPHA= (empty) uses the package setting
if [ -z "$FLARE_BIN" ]; then
  # Since the 11-11 upstream sync the binary is built by flare-cli (flare-cli/target/release/flare); older trees: flare/.
  for FLARE_BIN in $HOME/flare-up-glm/flare-cli/target/release/flare $HOME/flare-up-glm/flare/target/release/flare \
                   $HOME/flare-up-v3/flare/target/release/flare; do [ -x "$FLARE_BIN" ] && break; done
fi
cd "$(dirname "$FLARE_BIN")/../../.." || exit 1
KC=$HOME/.cache/flare/kernels
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$KC/triton} B12X_COMPILE_CACHE_DIR=${B12X_COMPILE_CACHE_DIR:-$KC/b12x} \
  TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$KC/inductor}
EXTRA=()
[ -n "$BUDGET" ] && EXTRA+=(--mtp-max-verification-token-budget "$BUDGET")
[ -n "$ALPHA" ] && EXTRA+=(--mtp-adaptive-verification-alpha "$ALPHA")
exec "$FLARE_BIN" run "$P" --gpus "${GPUS:-0,1,2}" --port "${PORT:-15015}" --host 127.0.0.1 \
  --served-model-name GLM-5.3-Flash --prefix-cache-memory "${PCM:-32G}" --mtp-draft-tokens "${K:-5}" \
  --scheduler-policy "${SCHED:-balanced}" "${EXTRA[@]}"

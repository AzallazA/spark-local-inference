<!-- GitHub Release note for tag hyperflare-tp3-20261111. Title: "Hyperflare TP3 2026-11-11: graph scratch (+22% KV), TTFT guard"
     Filled from the release chain (package 202611111227); 20261111 / 2026-11-11 = the publish day, set before tagging. -->

**Flare = Hyperflare.** Upstream renamed the project to Hyperflare (NVIDIA/hyperflare-inference); the binary, CLI and
cache paths are still `flare`. From this release on the fork's tags are `hyperflare-tp3-<date>`; the earlier
`flare-tp3-*` tags stay as published. Same base as before: upstream `972e28e4` plus this branch.

Changes since `flare-tp3-20261108.1`:

- **Graph scratch (export default): KV 2,467,200 → 3,014,656 tokens** at the same settings (342 hot experts,
  verification row cap 24, 32G prefix cache). The Engram projection, all-reduce and mix run in 1024-row chunks and
  `engram_residual` shares the `ffn_hc.residual` buffer (~465 MiB per GPU of graph scratch freed). Decode level over 4
  alternating rounds (c1/c4/c8 0.99/1.01/1.00×), prefill −1.5% (32K) / −0.6% (128K), long-prompt logprobs equal.
  Switches `FLARE_DS41_ENGRAM_CHUNK_ROWS=1024`, `FLARE_DS41_SHARE_ENGRAM_RESIDUAL=1` (set both to 0 for the old layout).
- **TTFT guard in Helios: `--scheduler-policy interactive`** (balanced + `--scheduler-ttft-target-ms 50`), the new
  `serve.sh` default. A prompt still without output 50 ms after arrival gets the next rounds, its prefill chunks
  interleaved with decode (`--scheduler-ttft-decode-interleave`, default 1; `--scheduler-ttft-max-chunk`). Also new:
  the `--scheduler-*` value/fairness knobs and `--no-mixed-prefill`. With the guard off, plan selection is unchanged
  (all scheduler tests pass; a trace test compares plans with and without inert guard knobs).
  Measured (3 alternating rounds, median): a 1.9K-token prompt arriving while 3 / 7 streams decode gets its first token
  after 0.36 / 0.38 s instead of 2.0 / 4.0 s; a 4-stream agent replay finishes 519 instead of 448 turns in 300 s (p90
  TTFT 0.93 vs 1.97 s); decode c1/c4/c8 level (300/708/965 vs 302/713/942 tok/s).
- `tp3-tools/release/export.sh`: `MAXC` (request slots; 8 is the qualified package) and `PARKED_ROWS` (BRRR row limit
  of the parked layers, default 12; 24 measured below and not adopted).
- BRRR kernel: CSF descriptors kept in registers in `routed()`; extension builds with `-fno-gnu-unique` (several kernel
  variants in one process); opt-in profile pass and phase marks for the layer test.
- Handbook (`docs/TP3-GUIDELINES.md`): 11-09/11-10 lessons (shell traps, GPU queue, pristine upstream reproductions,
  signing keys of forked configs, dry-run accuracy), and the fork README's Hyperflare note.

**Full c8 qualification of package `gb202-tp3-202611111227`, exported from this tag's code** (`export.sh` defaults:
HOT 342, MAXC 8, PARKED_ROWS 12; served at 450 W, k5, alpha 0.8, interactive, cap 24, 32G prefix cache; KV 3,014,656):
corruption check 0 true corruptions (10 runs), eval 42/43, tool calls 60/60, lavd 30/30, hotel-lights 30/30, long
retrieval 128K/900K pass/pass, long generation 4/4 (worst 8-gram repeat 0.0013, natural stop), natural-EOS decode 32K
c1/c4/c8 300/713/950 tok/s with 0 loops (20261108.1: 290/717/963), LiveBench coding 83.5 (20261108.1: 81.6), c8 KV
stress PASS (16/16, pinger 68/68, 9.5M prompt tokens). Decode 32K c1/c4/c8 301/734/988 tok/s (ignore-EOS, mean of the
completed cells of 5 alternating rounds; 20261108.1: 305/734/986), 128K/512K c1 289/271 (1 run).

**Not adopted:** BRRR in parked layers up to 24 rows at c8 (`PARKED_ROWS=24`, release A/B against 12 on two exports
from this tag, 5 alternating rounds): c8 +2.7 / +1.6 / −0.8 / +3.1% in the four rounds where both arms completed the c8 cell
(gate: ≥ +1% in every round), c4 0.98× in round 1, 128K / 512K c1 0.98× / 0.97× (on the 16-slot package, not part of
this release, 24 measured +3.4% at c16).
Default-off switches (verdicts in the code): BRRR item 6 (split tail and counter parity slower in the
layer A/B; fast route −0.5 to −2 µs per layer, level end to end); BRRR stage 2 wide rows (c4 −5%).

Release folder: `DeepSeek-V4.1-Flash Hyperflare-tp3-20261111` in https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes
(`source.patch` regenerated from this tag, 164 files, 110 new).

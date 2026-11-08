# 01. MXFP4 Engram table encoding as an explicit field

## Problem

DeepSeek V4.1 Flash ships its two Engram n-gram tables as FP8 E4M3 rows (256
bytes) with E8M0 block scales (8 bytes per row). NVIDIA's
`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` checkpoint, and the vLLM change it
targets, store the same tables as packed MXFP4 E2M1 nibbles (128 bytes per row,
same scales). Upstream Flare cannot load those tables. The packed form halves
host RAM per table (31.5 GiB to 16.2 GiB per rank per layer measured in the
fork at TP3) and lets both tables stay RAM-resident on 192 to 256 GB hosts.
Fork measurement (TP3, GB202): decode 79.5 to 84.4 steps/s at 32k context,
concurrency 1, with both tables in RAM instead of one on disk; WikiText
perplexity +2.6 %, code perplexity unchanged, the fork's 43-item answer check
43/43 (details in the next section). The tables are lossy (about 12 % relative RMS against FP8).

## Quality evidence (fork measurements, 2026-11-05 to 11-08)

Full write-up with method and raw numbers: `docs/tp3/ENGRAM-PRECISION.md`; tooling in `tp3-tools/engram-eval/`.
The comparison baseline is the engine's own run-to-run noise: vLLM on this stack is not bit-reproducible (fused
kernels reduce with float atomics), so FP8 was run twice and MXFP4 is judged against that floor.

**Teacher-forced next-token distributions**, same engine and day, only the tables changed (197 fixed 1,024-token
passages, 201,531 scored positions, top-20 logprobs at every position, KL over the reference top-20 plus a residual
bucket):

| | FP8 vs FP8 (noise floor) | FP8 vs MXFP4 |
| --- | --- | --- |
| Encyclopedia text (117 WikiText-103 articles): top-1 agreement | 95.5 % | 93.8 % |
| Encyclopedia text: mean KL (nats/token) | 0.024 | 0.055 |
| Encyclopedia text: perplexity | 2.735 to 2.737 (+0.07 %) | 2.735 to 2.805 (+2.6 %) |
| Code (80 source files): top-1 agreement | 96.7 % | 96.2 % |
| Code: mean KL | 0.010 | 0.013 |
| Code: perplexity | +0.05 % | -0.2 % |

The loss concentrates in encyclopedic prose, where the n-gram tables carry the most signal; code is at the noise
floor.

**Benchmarks**, greedy, thinking off (vLLM): TriviaQA 1,000 questions exact/contains FP8 836/881 and 844/886 (two
runs) vs MXFP4 838/882; MMLU-Pro 1,000 questions FP8 774 and 774 vs MXFP4 768. Question-level churn between the two
FP8 runs (26 flips each way on MMLU-Pro) is as large as the FP8-vs-MXFP4 churn (33 vs 27), so both gaps sit inside
the engine's own noise.

**Table-level checks**: value error of NVIDIA's MXFP4 tables against the FP8 tables, relative RMS 0.120 (first 262K
rows of layer 1); our own round-to-nearest conversion of the FP8 tables reproduces NVIDIA's block scales on 100 % of
blocks and 89 % of packed bytes (the rest are rounding ties); the packed-FP4 lookup kernel is bit-exact against a
reference decode on real rows.

**Whole-model qualification on the MXFP4 tables** (Flare TP3 releases 20261105 and 20261107, sampling at
temperature 1): 43-question eval 43/43 (FP8 on vLLM: 41-42/43); LiveBench coding 82.9 and 82.2 against 81.3 for the
single FP8 run and 81.5-84.2 across seven MXFP4 vLLM runs, i.e. no difference beyond run-to-run spread; hotel-lights
and lavd 30/30 at c8; long retrieval at 128K and 900K pass; long generation after a 581K-token conversation 4/4 with
worst 8-gram repeat 0.0011 (FP8: 0.0005); tool calls 60/60; vision 60/60 and 59/60 on the two image sets. Greedy
open-ended generations diverge from FP8 after the first token whose probability shifts, the expected fingerprint of
a lossy table; in every check we read, the content stayed equivalent.

**What this buys**: both tables (97 GiB) RAM-resident on a 256 GB host instead of 189 GiB for FP8. On vLLM at TP3 that
was +6 % decode (one table no longer read from disk); on Flare it is the difference between both tables in RAM with
~100 GiB free and either ~14 GiB free or one table on disk (9 % slower decode, the other table loses the page cache).
It also answers the disk-mode first-touch prefill complaint in upstream issue 78.

**Not yet measured** (candidates for the PR's evidence section if maintainers ask): a greedy FP8-vs-MXFP4 A/B on
Flare itself (Flare samples at temperature 1), a larger prose perplexity set, and the per-layer contribution (layer 1
vs layer 14 tables) to the +2.6 %.

## What the fork does today (`ds41-tp3-v3`, 2026-11-08; line numbers as of that tip)

- Protobuf: `EngramTable.Encoding.MXFP4_E2M1_E8M0 = 3` in
  `helios/proto/package.proto:253` and `helios/proto/runtime_package.proto:533`.
- Helios plane geometry `(256, 32, 128, 8)` for that value in
  `helios/src/runtime_package/engram.rs:117-119`.
- Ignition component: `EngramInputs.allocate(..., packed_fp4=False)` picks a
  128- or 256-byte staging row (`ignition/components/engram.py:47-53`); the
  lookup op and plan invocation decide the encoding from
  `weight.shape[-1] == 128` (`engram.py:27`, `:73-75`).
- Recipe: `Tables.prepare` detects `dtype == "U8" and shape == (rows, 128)` in
  the safetensors manifest (`ignition/recipes/deepseek41/engram.py:54-75`),
  carries `packed_fp4` (`:35`, `:82`), and emits row bytes and the encoding
  name (`:98`, `:110`).
- FlashInfer: `tp3-patches/flashinfer-engram-mxfp4.patch` adds a `PACKED_FP4`
  constexpr to the Triton `_lookup` kernel (nibble decode: sign bit 3,
  exponent bits 2:1, mantissa bit 0; even column in the low nibble) and
  `packed_fp4` plumbing through `_tuning.py`, `_preparation.py`, `_impl.py`.
- Converter (fork, vLLM side):
  `DeepSeek-V4.1-Flash Kraken-tp3-20261019/patch-nvfp4/make_engram_mxfp4.py` (this repo)
  rewrites only the two Engram shards of the stock checkpoint, recomputing
  UE8M0 block scales so `|x| / 2^(e-127) <= 6`, round-to-nearest to E2M1, and
  writes `quantization_config.engram_embed_quant = "mxfp4"`,
  `engram_embed_block_size = 32`, `engram_embed_scale_dtype = "ue8m0"`,
  `engram_embed_layers`. `test_packed_lookup.py` in the same directory checks
  the Triton decode bit-exactly against a LUT reference.

The shape-based detection is the review finding: a 128-wide row is also what a
future 128-column FP8 table would look like, and the encoding is decided in
three places.

## Proposed upstream design

1. `ignition/components/engram.py`: an `EngramEncoding` enum (`FP8_E4M3_E8M0`,
   `MXFP4_E2M1_E8M0`) with `row_bytes`, `scale_bytes`, `torch_view` properties.
   `EngramInputs` gains an `encoding: EngramEncoding` field;
   `allocate(..., encoding=EngramEncoding.FP8_E4M3_E8M0)` sizes staging from it.
   `PreparedEngramLookup` reads `inputs.encoding`, passes
   `invocation={"operation": "lookup", "compact_rows": True, "encoding": encoding.b12x_name}`
   and the matching weight view; the custom op `_lookup` receives the encoding
   as an int argument instead of inspecting shapes.
2. `ignition/recipes/deepseek41/engram.py`: read the encoding from checkpoint
   metadata (`quantization_config.engram_embed_quant`, default FP8 when absent),
   then validate manifest dtype and shape against it. Detection becomes
   validation, satisfying design-review principle 9 ("Read quantization from
   checkpoint metadata").
3. Helios: keep the fork's enum value and geometry; the loader already
   validates width, block, and row bytes per encoding.
4. Converter: a documented, explicit, lossy tool. Either an `ignition convert`
   form (the CLI already has `ignition convert ... --form packed`) or a script
   under `ignition/tools/`. It must write the metadata keys above so the recipe
   never infers encoding from shapes.
5. FlashInfer (separate PR to local-inference-lab/flashinfer): the patch as is,
   with `packed_fp4` renamed to an `encoding` string if maintainers prefer one
   vocabulary across both repositories. Main (6b867fdb, 2026-11-08) has no
   `PACKED_FP4` in `sequence/engram/_kernels.py`.

## API and schema changes

- Protobuf: enum value 3 in both files (additive; absent or 0 remains invalid
  as today, 1 remains FP8).
- Python: `EngramInputs.encoding`, `EngramInputs.allocate(encoding=...)`,
  `Tables.encoding` replacing `packed_fp4`.
- b12x: `invocation["encoding"]` or `invocation["packed_fp4"]`; `Plan.weight_shape`
  depends on it.

## Migration and compatibility

Packages exported before this change carry encoding 1 and load unchanged.
Packages with encoding 3 require an engine that knows the value; older engines
reject them with "unknown n-gram encoding", which is the correct failure.
Checkpoints without `engram_embed_quant` are FP8.

## Tests

- `tests/components/test_engram_mxfp4_cuda.py`: build random packed rows and
  scales, run the lookup op with `MXFP4_E2M1_E8M0`, compare bit-exactly in
  BF16 with a LUT-times-`2^(e-127)` reference; include the 16-code table at
  scale `2^0` (port of `test_packed_lookup.py`).
- Converter round trip: quantize synthetic FP8 rows with the converter
  function, decode through the lookup op, assert error within one E2M1 step
  times the block scale, and assert the recomputed scale bound `|x|/scale <= 6`.
- Helios unit test in `helios/src/runtime_package/engram.rs`: encoding 3 with
  128-byte weight planes decodes; 256-byte planes with encoding 3 are rejected.
- Mixed encodings across layers rejected (fork already raises at `engram.py:65-66`).

## Open questions

- Is a changed weight-tensor encoding a "shape variant" of the lookup
  operation or a new operation contract? AGENTS.md requires prior approval from
  the project owner for a new captured-graph operation family; the lookup
  family and its outputs are unchanged, but the input encoding is not. Ask
  before submitting.
- Do maintainers accept a lossy table encoding at all, given principle 9
  ("Never silently requantize")? The converter is explicit and user-invoked;
  the recipe never requantizes.
- Authorship of the Triton decode must be stated (see README).

## PR plan

FlashInfer: one commit, "Decode packed MXFP4 Engram rows in the lookup
kernel", with the kernel test.

Flare, three commits:
1. Protobuf enum value and Helios geometry with unit test.
2. `EngramEncoding` in the component; recipe reads metadata and validates
   shapes; lookup test.
3. Converter and documentation in `docs/model-preparation.md` (DeepSeek V4.1
   section) stating the loss figures and conditions above.

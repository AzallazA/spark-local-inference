# GLM-5.3-Flash on Flare at TP3: plan and status (2026-11-08)

Goal: run upstream Flare's `glm53` recipe on three RTX PRO 6000 Blackwell (GB202, 96 GB each) at tensor-parallel 3,
with the full 1M-token context, speculation and images, then qualify and release it the same way as the DeepSeek
V4.1 Flash TP3 build. Worktree `~/flare-up-glm`, branch `glm53-tp3` (on `ds41-tp3-v3` at 6c3aef27). Nothing committed
yet. Checkpoints on disk, both SHA-256 verified against the Hub:

| Checkpoint | Path | Role |
| --- | --- | --- |
| `nvidia/GLM-5.3-Flash-FP4-Local` rev 76348ddb (165 GiB) | `/path/to/ckpt/glm53-fp4-local` | first target: the checkpoint upstream's recipe declares (PTQ NVFP4 experts + CSF scales, MXFP8 dense/head, block-FP8 MLA/shared) |
| `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` branch `qad2-kda-w8a16-csf` head 96f27f8c (165 GiB) | `/path/to/ckpt/glm53-qad2-kda-w8a16-csf` | second target: QAD experts, KDA W8A16 (its card: 516K lookup 79.6 % vs 76.5 % W8A8 vs 76.2 % qad2); needs a W8A16/BF16 KDA projection path in the recipe |

## Why TP3 and the shape of the port

Upstream supports TP1, TP2 and TP4. On 3 x 96 GB, TP2 leaves ~5 GB per card for KV; TP3 leaves ~25 GB. TP3 is a
sharding problem, not a kernel problem: no expert parking, no Engram, no BRRR. The arithmetic that does not divide by 3
is padded with zero weights (checkpoint bytes untouched), as the DeepSeek attention groups were:

| Dimension | Value | TP3 |
| --- | --- | --- |
| KDA and MLA heads | 64 | padded to 66, 22 per rank (q/k/v/o columns, A_log, dt_bias, conv1d, kv_b rows zero; inert by the kernels' own math: L2 norm with eps, sigmoid decay, zero k/v keep the state zero, gated RMSNorm with eps) |
| Indexer heads | 32 | replicated (kernel capability forbids 33) |
| Routed and shared expert width | 2048 | padded to 2304, 768 per rank (the shard the b12x kernels are tuned for at DeepSeek TP3); ~6 GiB per rank of padding; rank 2 stores ~6 GB of zero w2 pad columns because the package format has a `WeightPadding` field but no producer or consumer for it |
| Dense MLP | 12288 | 4096 per rank, divides |
| Vocabulary | 154880 | 3 x 51648 (64-aligned, what the MXFP8 head loader needs); the 64 extra logits are sliced by the existing padded vocabulary projection |
| Decode context parallelism (DCP) | ranks 1/2/4 | DCP3 opt-in (branch `glm53-tp3-dcp3`, item 9): `FLARE_GLM53_DCP3=1` declares 768-token pages at 3 (`DCP=1` in the export script); unset keeps `page_tokens=None` at 3 so the planner leaves it off |
| Prefill all-reduce | DMA path needs rows % (world x 128) | one shared PCIe runtime for decode and prefill (DeepSeek's workaround), `ar_rows` chunking not ported yet |

Reused from the DeepSeek port: the world-3 collective ladder `ignition/preparation/ladders/gb202_pcie_tp3.json`, the
FlashInfer tuning and DCP all-to-all patches for three ranks (in the venv), vocabulary-parallel sampling at three ranks,
export/serve/qualification scripts in `tp3-tools/release/`, the handbook rules.

## Done (CPU, 230 tests green, the two failing planning tests pre-exist upstream)

1. `ignition/recipes/glm53/shards.py`: `Shards.of(world)` with padded heads/experts/dense/vocabulary; the ~150
   hardcoded divisions in recipe, state, pool, decoder, stack, mtp, resume, moe, kda, mla, bootstrap, weights replaced;
   TP1/2/4 byte-identical (tests assert the exact widths, partitions, scratch and state shapes, and `_costs`).
2. Rank count 3 declared with derived state-row costs (the formula reproduces upstream's 1/2/4 entries exactly);
   `Recipe.exact_rank_counts={3}` so a 4-GPU host still auto-selects TP4; `--context-parallel` refused at 3.
3. Padded loaders in `ignition/components/weights.py`: `pad=` on linear-group loads for MXFP8 (zero values, scale 127),
   block-FP8 (zero values, zero scales), BF16/FP32 vectors; `load_nvfp4_experts(pad=)` copies the covered expert
   columns into zero banks and clamps the CSF scale ranges (`padded_rows`/`padded_columns`, zero base); provenance
   recorded for the covered part; clear errors for uncovered ranges and non-paddable formats.
4. Tests: `test_glm53_shards.py`, `test_glm53_padded_weights.py` (synthetic checkpoint with the real tensor names and
   formats, byte-level TP1/2/4 equivalence, TP3 zero regions, negative control), `test_glm53_csf_padding.py` (CPU
   reference of both CSF pad semantics: zero on the Triton A16 path, row base on the compact A4 payload path, and
   the b12x in-place W4A16 repack round trip), world-3 parametrizations of the six GPU test modules, planning test.
5. Scripts: `tp3-tools/glm53-gpu-validate.sh LOGDIR` (six GPU steps, stops at the first failure) and
   `tp3-tools/glm53-export-tp3.sh` (the first export command with the TP3 flags and what to check in its log).
6. Two fresh code reviews (no wrong-bytes defects; findings fixed: planner auto-selection, provenance, comment,
   padded-range start check, column guard, naming).

## GPU validation so far

| Run | Result |
| --- | --- |
| 1 (11:03) | 01 nvidia_weights PASS (real-checkpoint partitions incl. TP3 ranks 0 and 2); 02 weights FAIL on `[3-2]`: padded down-projection CSF scales on the compact A4 path decode to the row base, not zero (inert: the packed values there are zero) |
| 2 (11:36) | `[3-2-a4]` PASS with the corrected expectation; `[3-2-a16]` FAIL at the packed-weight compare: the A16 path repacks the banks in place into W4A16 MMA tiles (int32 `[E, K/16, (N/64)*128]`), the test compared raw bytes. Fixed on CPU with b12x's own inverse decode and a round-trip test |
| 3 (queued) | reruns all six steps; pending steps: moe_overlap, mla (1 GPU), kda_tp, mlp_tp (3 GPUs) |

## Remaining steps

| # | Step | Needs | Estimate | Exit criterion |
| --- | --- | --- | --- | --- |
| 1 | GPU validation run 3 | 3 GPUs, 15 min | queued | all six steps PASS; sparse-MLA head tiling at 22 and MoE at 768 accepted |
| 2 | Mini-model path: build a 6- to 8-layer truncated GLM view (as `/path/to/ckpt/ds41-mini8` for DeepSeek), export at TP3, serve, `tp3-tools/consistency.py` decode-vs-prefill | 3 GPUs, ~1 h | first captured graphs at TP3; catches wiring and collective faults before the long export |
| 3 | First full export from `tp3-tools/glm53-export-tp3.sh` (cold tuning cache: every 22-head/768-column/51648-row shape is new, hours; retry with `CONC=4` if tuning runs out of VRAM) | 3 GPUs, 3-5 h | package exported; check the plan line, padded shapes, the single PCIe all-reduce, free device bytes (~96 - 63 GB), `flare run --dry-run` KV |
| 4 | Serve and smoke: `verify_fix`-style corruption check adapted to GLM, 43-question eval, tool calls, one image | 3 GPUs, 1 h | 0 true corruptions; answers sane |
| 5 | Speed: alternating A/B is not possible against another Flare GLM build, so measure c1/c4/c8 at 32K with steps/s and tokens/step, and compare with the vLLM Kraken TP3 release (288/282/266 tok/s at c1 for 0/32K/128K on its default) and upstream's TP4 numbers | 3 GPUs, 1 h | numbers recorded with power, cap, prefix cache |
| 6 | Qualification suite (adapt `qual_all.sh`: GLM model name, top-p 0.95 default, its thinking switch) | 3 GPUs, 1.5 h | all stages pass |
| 7 | Release: dated recipes folder `GLM-5.3-Flash Flare-tp3-YYYYMMDD`, fork tag, GitHub release, source patch verified | CPU | per handbook B.5 |
| 8 | QAD checkpoint as a second declaration: `w8a16` KDA projections (or BF16) need a projection path the recipe lacks; declare the checkpoint, add the path, export, run the suite plus a long-context recall test like LIL's 516K lookup, compare with FP4-Local | 3 GPUs, 2-4 days | numbers for both checkpoints |
| 9 | DCP at TP3 (optional): 3-way candidate merge kernel (the Triton merge assumes a power-of-two candidate count) and the 72-head exchange (24 per rank incl. rank 2's 8 zero heads; b12x needs equal shards). CPU stage done 2026-11-09 on `glm53-tp3-dcp3`: masked 2048-key merge, 768-token pages with ceil page counts (Python and Helios), opt-in `FLARE_GLM53_DCP3`, interpreter tests at W=2/3/4, Helios 768-page tests; GPU stages in `tp3-tools/glm-dcp3-gpu.sh` (not run) | 3 GPUs, 2 days | KV capacity x3 (estimate 12.3M vs 4.21M) |
| 10 | Follow-ups: port `ar_rows` chunking to the GLM prefill all-reduce (32 MiB per layer call at a 4096 chunk today); `expand_expert_scales` on at TP3 if the ~6.5 GB per rank is affordable after KV sizing; `collective_ladder.py` word-exchange validation at TP3 |
| 11 | Draft-layer `kv_b_proj` back on MXFP8 at TP3 (commit b135bb16 decodes the layer-45 MXFP8 weights exactly to BF16 because b12x qualifies its BF16 x MXFP8 BMM for 8/16/32 heads only and TP3 holds 24; cost ~+6 MiB/rank and ~4 us per draft pass, ~0.2 % of a c1 round). Options: (a) three 8-head calls per BMM, qualified shapes, two extra launches per pass; (b) one 32-head call with 8 zero heads, a third wasted work; (c) qualify batch 24 in b12x `_QUALIFIED_BATCHES` with its numerical tests (cleanest, a b12x change). Also extend the TP3 runtime collective ladder past 24 MiB (export uses static tuning since 442a7f86) | 3 GPUs, profile first | only if a c1 trace shows the draft absorb/expand BMMs taking real time; otherwise close |

## Risks

- Sparse-MLA kernel head tiling at a non-power-of-two head count (22): only the GPU test shows it.
- W4A16 NVFP4 kernel at intermediate 768 with padded columns: equivalence proven for scales; values need run 3.
- Cold tuning at new shapes may exceed VRAM at concurrency 8 (DeepSeek needed the HOT=300 warm-up trick; here the
  equivalent is `CONC=4` first).
- The shared PCIe prefill all-reduce without `ar_rows` chunking may cost prefill throughput.
- The vLLM Kraken TP3 release serves the QAD checkpoint; the first Flare release will serve FP4-Local, so the two are
  not directly comparable on quality until step 8.

## Complete porting checklist (every point, in dependency order)

1. **Rank-count declaration**: `ranks[3]`/`plain_ranks[3]` with derived state-row costs; `exact_rank_counts={3}`;
   `page_tokens=None` at 3 (no DCP). Done.
2. **Shard geometry** (`Shards`): heads 66/22, indexer 32 replicated, expert width 2304/768, dense 4096, vocabulary
   3 x 51648, KDA channels 2816 per rank; every former `// world_size` goes through it. Done; TP1/2/4 byte-identical.
3. **Scratch and state**: KDA state rows, MLA scratch, MLP scratch, MTP scratch, resume buffers sized from `Shards`. Done.
4. **Padded weight loads**: MXFP8 (zero values, UE8M0 127), block-FP8 (zero values, zero FP32 scales), BF16/FP32 KDA
   vectors (`A_log`, `dt_bias`, conv1d), MLA `q_b`/`o_proj`/`kv_b`, routed experts (w13 rows, w2 columns, CSF scale
   ranges with zero base, both decode semantics), shared expert, head rows. Done on CPU; expert padding proven on GPU
   for the compact path; A16 path rerun pending.
5. **Provenance**: covered pieces recorded for padded owners; ranks 0 and 1 fully derivable; rank 2 stores ~6 GB of
   interleaved zero w2 columns (format gap: `WeightPadding` exists in the proto with no producer or consumer; an
   upstream proposal if the size matters). Done.
6. **Collectives**: one shared PCIe all-reduce runtime for decode and prefill when 4096 % world != 0; padded
   vocabulary gather 154944 columns; world-3 ladder; tuning patch. Done in code, untested on GPUs. To do: port
   DeepSeek's `ar_rows` chunking (32 MiB per layer call at a 4096-token chunk today), and check
   `collective_ladder.py:52` (word-exchange rows validated for TP2/TP4 only) before `--collective-tuning runtime`.
7. **Kernels at the new shapes**: KDA decode/prefill, short conv, gated norm, sparse MLA (22 heads: unverified tiling),
   absorbed k/v batched GEMMs, MoE at 768 columns, W4A16 in-place repack. GPU steps 03-06.
8. **Planner**: 3 selectable only on 3-GPU hosts or explicit `--gpus`; shortfall suggestions; `--dry-run` KV for the
   GLM package (check the plan reproduces the live "Allocate cache" line as it does for DeepSeek). Partly done.
9. **Export**: `tp3-tools/glm53-export-tp3.sh` (no context parallel, chunk 4096, 5 drafts, concurrency 8, ladder);
   cold tuning cache; `CONC=4` fallback. Mini-model view first (build `glm53-mini8` like `ds41-mini8`).
10. **Serving**: GLM model name `GLM-5.3-Flash`; sampling defaults temperature 1.0, top-p 0.95 (ranges 0.95-1.0);
    the recipe's thinking switch; images (vision tower replicated; `--language-model-only` for text-only exports);
    serve script variant in `tp3-tools/release/`.
11. **Correctness**: a GLM `verify_fix` (decode vs teacher-forced prefill on fixed prompts), 43-question eval, tool
    calls, hotel-lights/lavd/retrieval/long-generation/stress via `qual_all.sh` adapted (model name, top-p, thinking
    kwarg), image set.
12. **Speed**: c1/c4/c8 at 32K with steps/s and tokens/step; compare with the vLLM Kraken TP3 release (default:
    287.6/282.1/266.3 tok/s at c1 for 0/32K/128K, 32 slots; DCP3 KV 10.5M) and upstream TP4.
13. **Release**: dated folder, fork tag, GitHub release, source patch verified, hygiene grep (B.5).
14. **Second checkpoint**: `qad2-kda-w8a16-csf` declaration; W8A16 or BF16 KDA projection path; suite plus a 516K
    lookup test; compare with FP4-Local.
15. **DCP at TP3** (optional): 3-way candidate merge (power-of-two assumption in the Triton merge), 72-head exchange,
    `page_tokens` 768 at 3 ranks, costs for the DCP state layout. Design study
    `~/ds41-tp3/analysis/glm-dcp3-design-20261108.md`; CPU part on branch `glm53-tp3-dcp3` (item 9).

## Concepts from our vLLM GLM builds worth porting (Kraken TP3 releases 20260927, 20261008, CSF 20261015/16)

| vLLM concept (measured effect there) | What it did | Flare status / how it maps |
| --- | --- | --- |
| **TP3 padding set** (heads, vocabulary, sinks) | zero-padded shards so TP3 works at all | the core of this port (items 2-4); same approach, now with upstream's loader |
| **Small-row MXFP8 GEMV** for decode steps of <= 8 rows (+5.5 % c1 steps/s; c8 neutral) | reads packed MXFP8 weights in place with an exact E4M3->BF16 path for the skinny decode GEMMs (KDA q/k/v/o, dense, head) | Flare already races kernel candidates per shape at export (tuning) and upstream has `perf/glm53-bf16-head-projections` (Triton tile vs cuBLAS for the BF16 head) and `perf/glm53-width-one-in-place`; first measure the c1 step map on Flare (trace) and only port a GEMV if the MXFP8 projections show up as a cuBLAS floor |
| **Finer CUDA-graph ladder above 32 rows** (+10-11 % at c20/c28) and MTP graph max following the slot count | no padding to the next captured width | Flare captures a speculation width ladder from the recipe (`Speculation.widths`, `verification_rows`); check the captured widths at concurrency 8 and whether the 8-slot default even needs it; a wider ladder costs VRAM |
| **One-shot all-reduce cutoff 256 KB -> 1 MB** (~+1 %, c16-c32 latency halved) | keeps decode all-reduces on the one-shot path up to 32 slots | Flare's PCIe runtime is sized by the ladder JSON buckets (`gb202_pcie_tp3.json`) and `max_tokens`; re-tune the bucket boundaries for GLM's 4096-wide rows (DeepSeek's were tuned for 5120) |
| **DCP3 sparse-MLA split fix** (+7.5 % steps/s at c32, DCP3 only) | one split instead of 33 when a split exceeds 3 waves | only matters with DCP at TP3 (item 15); check whether upstream's sparse-MLA wrapper has the same `_wave_balanced_num_splits` behaviour |
| **MTP3 / `DRAFT_TOKENS`** (3 drafts default; 2 drafts +3 % at c8/c32 but slower at c1) | speculative decoding with the checkpoint's MTP layer | upstream recipe defaults to 5 drafts with token capacity 128; sweep drafts 3/4/5 at c1 and c8 like the DeepSeek k5/k7 sweep (expect the same trade) |
| **DFlash2 draft model** (MXFP8 draft; slower than MTP3 at c1: 206 vs 288 tok/s; `incoai`/`local-inference-lab` DFlash2 on disk) | external 5-layer draft model instead of the MTP layer | not in upstream Flare; lower priority since MTP won on vLLM; revisit only if MTP acceptance is poor at c8 |
| **DCP1/DCP3** (KV 4.15M vs 10.5M at 32 slots) | KV sharded across ranks, attention merged | upstream Flare has native DCP for 2/4 ranks; TP3 needs the 3-way merge (item 15); the single biggest KV lever |
| **KV reclaim** (explicit KV budget, BF16 LM head released after the draft builds its head, shared TP/EP communicator: 2.95M -> 3.32M KV) | memory bookkeeping at startup | Flare sizes KV from the plan deterministically (`--dry-run` shows it); the equivalents are padding cost (6 GiB/rank), `expand_expert_scales` off, and `--language-model-only`; no runtime reclaim needed |
| **EP3 (expert parallel) vs MOE_TP=1** | expert placement per rank instead of TP shards | not applicable: Flare's recipe shards experts by width; EP would be a new design |
| **Expert order / promotion, VMM cold experts** (DeepSeek-era) | hot experts in VRAM, cold in host RAM | not needed for GLM: the whole model fits at TP3 |
| **MAX_PARALLEL_PREFILLS=auto** (TTFT of a short request during a 420K prefill 42 s -> seconds) | prefills share steps round-robin | Flare schedules prefill in 4096-token chunks per step with `--scheduler-policy balanced`; measure the same scenario (short request during a long prefill) on Flare before assuming it is solved |
| **LMCache L1** (host-RAM prefix tier, 96 GB) | reload of long prefixes from host RAM | Flare has `--prefix-cache-memory` (host RAM, 32G on DeepSeek, ~78 warm 127K prompts); same feature, already native |
| **fastokens Rust tokenizer** (cached-prefix TTFT 0.58 -> 0.21 s at 396K) | tokenizer speed | Flare tokenizes in Rust (Helios) already; measure cached-prefix TTFT at 400K and 900K to confirm |
| **Uncensored checkpoint, MOE_TP, Marlin TP experts** | alternative weights | checkpoint declaration only, if ever wanted; out of scope |
| **`lil-bench`, qualification protocols** (lavd, hotel-lights, retrieval 128K/900K, long generation 581K, c8 stress, DCP3 launch checks) | test suite | reuse as is: `qual_all.sh` already runs them against an OpenAI endpoint; add the GLM thinking switch and top-p |
| **Half-L2 / `VLLM_B12X_*` cache knobs** | b12x L2 persistence settings | Flare's b12x is the same library at a different pin; the tuning step chooses; nothing to port unless the trace shows L2 thrash |

Order of value for the Flare GLM build once TP3 works: DCP3 (KV x2.5), draft-count sweep, ladder re-tuning for
4096-wide rows, then the GEMV question if the trace shows a cuBLAS floor. Everything else is already native or not
applicable.

## Decision 2026-11-08 (after GPU run 6 and the head-count kernel test): heads 24/24/16, no head padding

`tests/recipes/test_glm53_sparse_mla_heads_cuda.py` (kernel-only sparse-MLA decode, last row inactive): 16, 24 and 32
heads per rank pass at 2 and 3 rows; **22 heads fails** at 2 and 3 rows (the inactive row gets up to 2e-2 of output,
1024 nonzero values), i.e. the b12x sparse-MLA GLM_NEXT kernel mis-tiles head counts that do not divide 64 when a tile
spans two rows. This is why `test_glm_mla_checkpoint_native_replay[3-decode-2-True]` failed in run 6. Conclusion: do not
pad heads to 66. Use an exact, uneven head split per rank: **24 / 24 / 16** (KDA and MLA; the KDA kernels take any head
count, the sparse-MLA kernel is proven at 24 and 16). Consequences:

- `Shards` gains per-rank head counts (`heads_of(rank)`), KDA channels 3072/3072/2048, MLA scratch and state rows per
  rank, `_costs` for the largest rank (24 heads) since the planner sizes by the limiting rank.
- Head padding in the loaders becomes unused for GLM (keep the `pad=` path for experts and the vocabulary only); the
  o_proj column partition and q_b/kv_b row partitions become exact per-rank ranges (64 heads x 24/24/16).
- The collectives see uneven per-rank attention widths: check the all-reduce (sums, width-agnostic) and any
  all-gather of head-major tensors (the DCP path is refused anyway) for equal-size assumptions.
- Tests: world-3 cases must run for every rank (0, 1, 2) since ranks now differ in shape; the `[3-decode-2]` MLA case
  becomes the 16-head rank.
- Expert and vocabulary padding stay as validated (runs 4-6).

Validation status at the pause: run 6 (qualification Heliopause build, pinned kernel caches): 01 partitions PASS,
02 padded weights PASS (13, incl. the MTP layer), 03 MoE overlap PASS (36, incl. the padded 768 shard), 04 MLA 13/14
(the 22-head case), 05/06 (KDA/MLP three-rank) not yet reached in that configuration. Next GPU step after the head
split: `tp3-tools/glm53-gpu-validate.sh` from step 03b.


## Measured 2026-11-08: TP2 vs TP3

FP4-Local, language model only, 450 W, `--prefix-cache-memory 32G`, adaptive verification on, 32K context, 3 rounds
per variant (`~/bench-results/glm-compare-k-1108-1735`; TP2 baseline repeated in `glm-compare-1108-1610`). Full write-up:
`$HOME/ds41-tp3/analysis/GLM53-TP2-vs-TP3-20261108.md`.

| Package (5 drafts) | KV tokens | c1 tok/s (steps/s) | c4 tok/s | c8 tok/s |
| --- | --- | --- | --- | --- |
| TP2 `gb202-tp2-202611082052` (DCP2, 32 heads/rank) | 1,914,368 | 224.1 (105.2) | 489.4 | 680.4 |
| TP3 `gb202-tp3-202611082232` (24/24/16 heads, 768 expert columns, no DCP) | 4,229,888 | 246.9 (114.3) | 542.1 | 748.3 |
| TP3 vs TP2 | 2.21x | +10.1% (+8.6%) | +10.8% | +10.0% |

- TP3 is about 10% faster at c1, c4 and c8, all from steps/s (tokens/step and acceptance unchanged), and holds 2.2x the
  KV (27 GiB cache budget per GPU vs 6.7 GiB).
- 3 vs 5 drafts: no measurable difference (within +-2% at every cell, both rank counts), because adaptive verification
  offers about 1.8 drafts per round. Keep 5 until the fixed-width sweep reports.
- Smoke ("17 x 23") returns "391" on all 12 server starts; no bench loop detections. The quality suite is still to run.
- TP3 export 39 min (TP2 33 min) after three failed attempts: runtime collective ladder stops at 24 MiB (static tuning,
  442a7f86), MXFP8 BMM qualified for 8/16/32 heads only (draft kv_b_proj decoded to BF16, b135bb16), and a
  variable-shadowing bug in the first version of that fix.
- vLLM Kraken TP3 (dec48abd, MTP3, 350 W) is 13-21% faster at c1 and 7-10% at c8 at equal c1 steps/s; the gap is
  tokens/step (2.16 vs 2.48), confounded by checkpoint, power and draft width. Next: draft sweep (running), a4 +
  resident scales export, 704 expert columns per rank, dec48abd export, DCP3.

## Measured 2026-11-09: default decision, a4 vs a16, LIL vs FP4-Local

Decision: GLM default = **FP4-Local a16** (package gb202-tp3-202611082232). Full tables in
`~/ds41-tp3/analysis/glm-lil-vs-fp4-20261109.md` (private).

- **a4 vs a16** (teacher-forced decode-path distribution, 302K positions, noise floor from a repeated a16 run): a4 costs
  prose perplexity +1.9 %, code +0.2-0.34 %, KL 4.8-6.4x the floor on both checkpoints; task tests equal
  (eval 42-43/43, tools 60/60, hotel-lights 26/30 both); a4 is +7 % c1, +3-5 % c4/c8. Not adopted (measurable loss).
- **LIL CSF-QAD (dec48abd) vs FP4-Local**: LIL is **faster** (a4: c1 +1-2 %, c4 +3-6 %, c8 +0.5-1 %) but **less
  accurate** (perplexity +0.36 % overall, +0.32 % prose, +0.40 % code, a16 vs a16) and has **~1M less KV**
  (3.22M vs 4.25M at a4). Not adopted. LIL packages live in the m2-2 store, signed with its own development key.
- **704 expert columns** (a4): +18 % KV, -3..-4.5 % speed. Not adopted.
- **Resident expanded scales** (a16): +5.3 % c1, +2 % c4/c8, -0.48M KV. Open.
- Next: tool-call PR fixes (#59 diagnostics + #179, branch glm53-tp3-prs), a16 c1 profile, DCP3 (item 9).

## Decision 2026-11-11: c32 default = PR #253 + `--mtp-only` package; TTFT guard shipped, off by default

**Package.** `gb202-tp3-202611112036` (`glm53-pr253` d3c27126: upstream PR #253 dead-prefill-storage reuse +
port of upstream `--mtp-only`, same switches as 202611110835: DCP=1 a16 KDA side chain, 704 expert columns, CONC=32).
68 CUDA graphs instead of 108 (no no-draft graphs; the package refuses `--mtp-draft-tokens 0`), KV 13,416,960 served
(+2.0% over 13,154,304). Decode vs 110835 (3 rounds vs the c32 sweep rounds at alpha 0.8): steps/s 1.025 / 1.013 /
1.022 / 1.006 / 1.005x at c1 / c4 / c8 / c16 / c32; tok/s 0.970-1.008x (draft-acceptance swing; gate since 11-11 =
steps/s >= 0.98x AND tok/s >= 0.96x). Quick checks: eval-43 43/43, tools-60 60/60 at c32, verify_fix 0 TRUE
corruptions, 0 server errors. Needs a binary with the `mtp_only` field (this branch since 7e4a1cd0).

**TTFT guard** (`--scheduler-policy interactive` = balanced + `--scheduler-ttft-target-ms 50`; Helios f1135624, same
code as DS41). Merged here (fast-forward to 7e4a1cd0) but `serve-glm.sh` keeps `SCHED=balanced`: with the target at 0
the guard path is inert (trace test: identical plan sequence), so no quality run is needed for the default.
`SCHED=interactive ./serve-glm.sh` turns it on. Measured on 202611112036 (bench-results/ttft-ab-glm-1111-1615, decode
"off" = glm-tp3-newbench-1111-1540, both 3 rounds, 32K 60 s):

| | guard off (`balanced`) | guard on (`interactive`) |
|---|---|---|
| Decode tok/s c1 / c4 / c8 / c16 / c32 | 264 / 612 / 774 / 1183 / 1652 | 264 / 604 / 786 / 1176 / 1635 (0.99-1.02x) |
| Decode steps/s c1 / c4 / c8 / c16 / c32 | 112.1 / 67.5 / 43.5 / 31.4 / 22.3 | 112.5 / 68.4 / 41.9 / 30.5 / 21.7 (c16/c32 -3%, c8 baseline swings 41.6-44.8) |
| First token p90, 1.9K prompt while 3 / 7 / 15 / 31 streams decode | 2.90 / 6.18 / 11.5 / 19.6 s | **0.47 / 0.50 / 0.52 / 0.54 s** |
| Running streams' tok/s while that prompt prefills (c4 / c8) | ~520 / ~740 free, 431 / 699 during | 486 / 755 free, 113 / 203 during |
| Agent replay 8 streams, 300 s: turns / output tok/s / TTFT p50 | 308 / 266 / 4.23 s | **360** / 304 / 0.62 s |
| Agent replay 16 streams, 300 s: turns / output tok/s / TTFT p50 | 246 / 329 / 8.46 s | **418** / 310 / 0.92 s |

Why off by default on GLM (DS41 has it on): the c32 package serves more batch-style traffic, where the c16/c32
steps/s dip (~3%, not yet confirmed by a same-session alternating A/B) and the 6% lower output at 16 agent streams
matter more than first-token latency. For chat or agents, `interactive` finishes 17-70% more turns. Before making it
the default: same-session alternating decode A/B (3x3) + the guard quality run (`ttft-quality-glm.sh`).

# DeepSeek V4.1 Flash on Flare TP3: future plan (speed first, then VRAM, then fixes)

State on 2026-11-08 (default package 081024, fork tag `flare-tp3-20261108.1`): 32K decode c1/c4/c8 ≈ 303/726/985 tok/s
at 450 W, prefill ≈ 14.5K tok/s, 2,461,184 KV tokens, cold image 0.11 s, full c8 qualification PASS. Everything below
is unmeasured until stated otherwise; the rules in `docs/TP3-GUIDELINES.md` D.1 apply (alternating A/B, 3+ rounds,
read steps/s and tokens/step next to tok/s, trace estimates are upper bounds).

## Where the time and the memory go (measured)

**c1 decode round, 9.88 ms, 1060 kernels (node trace, 2x build):** BRRR fused layer tail 6.30 ms (60.7 %,
157.5 µs/layer against a ~147 µs bandwidth floor: both all-reduces, mHC, router, experts are inside it); attention-side
dense GEMMs 1.31 ms (qkv_a 43 x 10.3 µs, q_b 40 x 7.0 µs + split-K reduce 43 x 5.9 µs, LM head 228 µs); MLA decode
plus merge plus KV snapshot ≈ 0.8 ms; mHC 0.36; RoPE 0.24; elementwise 0.21; indexer 0.12; norm/quant 0.09. GPU busy
100 % of the step. Speculation: 2.6 tokens per step at c1 (draft acceptance 0.38 per draft), 8.9 at c4 (0.61), 16.6 at
c8 (0.75).

**Per rank VRAM (dry run, 95.06 GiB total):** fixed allocations 91.65 GiB (weights: 26 resident expert layers, 342 hot
experts in 14 parked layers, dense/attention/heads; execution state; graph scratch pools 1,816 MiB of which five
160-200 MiB residual/Engram buffers; operator scratch 296 MiB; vision staging slots 190 MiB), kernels and graphs
553 MiB (129 graphs, 7,643 kernels), headroom 64 MiB, prefix-cache device share 35 MiB, KV cache 2.19 GiB. KV costs
890 bytes per token per rank (128-token pages of 113,920 B: compressed latent 288 B + index 68 B per record), so
**1 GiB of VRAM freed = ~1.2M KV tokens, and one hot expert (~82 MB) = ~92K tokens.**

## Speed ideas, ranked by expected value

| # | Idea | Why it could pay | Expected | Effort | How to measure |
| --- | --- | --- | --- | --- | --- |
| 1 | **Speculation acceptance at c1.** 2.6 tokens/step is the whole c1 story: +0.5 tokens/step ≈ +19 % tok/s, more than every kernel idea combined. Options: per-position acceptance study (where do drafts die?), draft sampling (temperature/typical) vs the target's temperature 1.0, upstream's DSpark proposal work (`astra/dspark-proposal-pipeline-pinned`, `dspark-block-broadcast`, `dspark-markov-fusion`: canonical proposals, Markov fusion), draft head precision (nvfp4 today; fp8 draft head = larger but more accurate drafts), more drafts only if acceptance per draft rises (k7 lost 3-6 % at c4/c8) | the lever with the most room | +10-25 % c1 if acceptance moves 0.38 → 0.45-0.5 | medium-large | `spec_metrics.py` per-position acceptance from the serve log; A/B per change |
| 2 | **Expert activation precision (a16 vs a8 vs a4).** The DeepSeek recipe runs the b12x MoE with `activation_mode="auto"`; GLM on SM103 uses a4 and on vLLM the a4 experts were ~10 % faster decode than a16. Find what auto resolves to on GB202 for MXFP4-CSF experts and whether a8/a4 is available; it affects every MoE call outside BRRR (prefill, wide verify rows at c8, parked layers above 12 rows) | prefill and c8 both run many MoE calls through FlashInfer | prefill +5-15 %, c8 +2-5 % if a8/a4 applies; quality must be re-checked (teacher-forced KL, eval) | small to try, medium to qualify | export with the mode forced, prefill bench + c8 A/B, `verify_fix`, eval-43, LiveBench |
| 3 | **Verification row cap by concurrency.** Cap 24 won +1-2 %; on vLLM a scaled cap (higher cap only above N requests) won more. Sweep cap 16/20/24/32 at c8 and c1 separately; if the optimum differs, propose a per-concurrency cap (upstream has adaptive alpha, not a scaled cap) | cheap config sweep | +1-3 % c8 | small | serve-flag A/B, 3 rounds |
| 4 | **Prefill all-reduce chunking (`FLARE_DS41_AR_ROWS` 1024) and prefill chunk.** 1024 was set once; sweep 512/1024/2048 for prefill tok/s; 8192-token chunks were rejected for VRAM, not speed | prefill is 14.5K tok/s with ~0.5 ms of all-reduce per layer call | prefill +2-5 % | small | prefill bench at 32K/128K |
| 5 | **Padded attention group on rank 2.** Attention groups are padded 8 → 9, so rank 2 computes a zero group: MLA decode runs 3 groups where 2.67 would do, and every rank waits for the slowest. Skip the padded group's attention on rank 2 (zero-weight group → zero output) | ~11 % of the 0.8 ms MLA side | ≤1 % c1 | small-medium (per-rank head count in the MLA launch) | trace rank 2 vs rank 0 MLA kernel time; A/B |
| 6 | **BRRR tail headroom (157 vs 147 µs/layer, ≈4 % of the round).** Remaining ideas inside the kernel: dense weight restreaming is already one pass at m ≤ 6 (OTOK 16); the fp8 row reads of resident experts could use 16-byte vector loads where they do not; the two in-kernel all-reduces' spin-wait could overlap the shared-expert GEMV. Each is a 1-2 % item and needs the layer test plus the FP64 bounds | known floor | 1-3 % c1 | medium each | `test_deepseek41_brrr_tail_cuda.py`, mini export, A/B |
| 7 | **MTP draft layer through BRRR.** The draft (MTP) layer has its own experts and runs every step; it goes through the stock path. Fusing its tail like the 40 target layers would cut the draft's per-step cost | draft cost per step is part of the ~36 % attention-side time | 1-3 % c1 | medium (weights plumbing for the draft layer, hot_rows not needed) | trace the draft step; layer test on the draft layer |
| 8 | **Kernel-count reduction on the attention side (1060 kernels per round).** Launch granularity is 5-10 µs per kernel; CUDA graphs and PDL already hide most of it. Fusions tried (prologue, qkv_a norm, sparse decode) were level, so only a fusion that removes ≥ 5 kernels per layer is worth trying (RoPE + KV snapshot + merge in one, 3 x 43 launches) | 0.24 + 0.16 ms of small kernels | ≤2 % c1 | medium | trace first: per-stream busy intervals, not kernel sums |

Closed, do not reopen without a new angle: wide-row BRRR restreaming (stage 2, -5 % c4), extra PDL trigger (-2 % c8),
fused prologue, fused qkv_a norm, sparse decode attention, FP8 LM head for speed, L2 prefetch, EP all-reduce over
b12x, CPU-side experts, DCP3, dynamic expert cache (0.40 % cold routes), route reuse, k7 drafts, 8192 prefill chunk,
600 W, +100/+150 core offsets, ungated parked layers (c8 +1 %, c1 level).

## VRAM ideas (each GiB = ~1.2M KV tokens, or ~12 hot experts)

| # | Idea | Saves per rank | Cost / risk | Effort |
| --- | --- | --- | --- | --- |
| 1 | **Graph scratch pools (1,816 MiB).** Five 160-200 MiB residual/Engram buffers are sized for the widest graph; check which graphs need them at full width and whether the pool's first-fit leaves fragmentation (the vision staging found 90 MiB of free chunk to alias) | 300-600 MiB | none if sized by analysis; needs the capture to agree | medium |
| 2 | **FP8 LM head (BF16 today, 129,280 x 5,120 per rank-share).** Measured no speed gain, but it is ~0.4 GiB per rank of BF16 the FP32-logit path could read as FP8 | ~0.4 GiB (~500K KV) | logit precision; re-run eval/LiveBench | small (export flag exists upstream) |
| 3 | **Fewer captured graphs.** 129 graphs, 553 MiB of kernels/graphs estimate; the cap 24 already dropped 13 speculative graphs (+87K KV). Drop unused capacities (image sizes never used, verify widths above the cap) | 50-150 MiB | fewer ready shapes; only drop what the serve flags make unreachable | small |
| 4 | **Operator scratch (296 MiB) and the vision arena (160 MiB).** The vision arena stays allocated with no image in flight; staging slots (190 MiB) alias scratch already | 100-300 MiB | vision latency if the arena is lazily mapped | medium |
| 5 | **Hot experts vs KV.** The current 342 is the sweep's speed optimum; every expert fewer is +92K KV at a measured ~0.5 % c1 per 20 experts (H300 is 4 % slower, 2.99M KV). Offer two presets rather than optimising one | user choice | speed | none (export knob) |
| 6 | **Engram scratch on device (200 MiB projected + 12 MiB weights x2).** Check whether the projected buffer is sized for prefill width and could be split per width | ~100 MiB | capture complexity | medium |

Not worth it: expert padding (none at TP3), KV precision (already a compressed 288-byte latent; FP8 KV would not apply),
embedding/vision in host (done).

## General fixes and hygiene

1. **Upstream sync** (B.6): main is 4 commits ahead with a FlashInfer pin bump; required before the first upstream PR.
   Rebase, re-apply the six `tp3-patches/` to the venv, rebuild, layer test, export, verify.
2. **FlashInfer `tiny_decode` SwiGLU clamp**: still unfixed upstream (head 07c41f32); a PR with the 45-line patch and
   the 10 → 0 corruption A/B is self-contained and benefits every clamped-SwiGLU model at TP1-3.
3. **Proposals 01-06** (Gitea-only branch `upstream-proposals`) (MXFP4 Engram encoding first, with the quality evidence now
   attached), each as a small PR after the sync; proposal 03 can drop the now-redundant top-p range.
4. **Cold tuning cache at HOT=342** needs the HOT=300 warm-up; upstream's 11-07 "bound host memory during TP kernel
   preparation" may remove it; re-test after the sync.
5. **Prefill `ar_rows` chunking** is DeepSeek-only; make it a recipe-level option so GLM can use it (B.6 of the GLM plan).
6. **Package format `WeightPadding`** exists in the proto with no producer or consumer; a derivable zero-fill would let
   padded owners reference the checkpoint (relevant for GLM rank 2's 6 GB today; DeepSeek's padded attention group
   is small).
7. **Release automation**: a script for handbook B.5 (tag, regenerate and verify the patch, dated folder, hygiene
   grep, both pushes) so a release is one command; the three slips of 11-08 (results with usernames, `.brrr-ext`
   committed, pre-cleanup copies of files) were all manual steps.
8. **Qualification runner**: the "grid" items miss because the model answers verbosely ("Therefore, the answer is
   **green**"); the grader should accept a bolded final answer; 49/60 would read 60/60 on both the staged and
   unstaged vision runs.
9. **`verify_fix` and `consistency.py`** as a GLM-capable pair (model name and tokenizer parametrised) before the GLM
   export so the first package has the same checks as DeepSeek.
10. **Expert-order refresh**: the blended order is from 11-05 traffic; re-count routes from a week of real traffic
    when convenient (the dynamic-cache study showed 0.40 % cold routes, so the gain is small but the data is cheap).

# 05. Small generic fixes to send separately

Each item is one short PR. FlashInfer checks below were made read-only against
local-inference-lab/flashinfer main at 6b867fdb (2026-11-08) with
`gh api repos/local-inference-lab/flashinfer/commits?path=...` and raw file
reads.

## A. FlashInfer: tiny-decode MoE kernel ignores `swiglu_limit`

Problem. `_w4a8_mx_micro_supported` in
`flashinfer/experimental/b12x/moe/fused_moe/_impl.py` (main lines 2606-2648)
selects the tiny-decode kernel from token count, shapes, activation name, and
determinism only. `_tiny_decode_supports` (main line 13032) checks
`activation == "silu"`. The kernel implements plain SiLU-GLU; DeepSeek V4.1
sets `swiglu_limit = 10`, so a decode step routed to it runs unclamped.

Effect seen in the fork (TP3, DeepSeek V4.1 Flash): experts whose activations
exceed the clamp (for example layer 39 expert 313 on rank 0) blow up the last
layer's MoE output 16-600x and produce one junk decode token, after which the
reply recovers. Same package A/B with the guard: 10 to 0 true corruptions in 20
`verify_fix` runs. The tiny kernel is chosen when the per-rank expert width is a
multiple of 128 (TP1/2/3: 2304/1152/768); TP4 (576) takes the compact micro
kernel, which clamps.

This is not the cause of flare#193. That issue (TP4) was traced upstream to the
checkpoint itself: DeepSeek's own API and the #118 CPU reference give the same
flat next-token distribution at the same positions (thread, 2026-11-06/07).
Fork comments that cite flare#193 date from before that conclusion; they mean
the same symptom (one junk token mid-reply), not the same cause.

Fork patch: `tp3-patches/flashinfer-tiny-decode-swiglu.patch` adds
`swiglu_limit`, `swiglu_alpha`, `swiglu_beta` parameters and requires all three
to be `None` before preferring the tiny kernel, passing them from
`MoeDecodeQuery` at the two call sites.

Status on main: not present. Main also added a compact micro path for
`source_format == "fp4_e8m0_k32" and n % 128 == 64` in the same function;
check whether it honors the clamp before submitting, and cover it if not.

Test: a decode query with `swiglu_limit=10` must not select the tiny kernel;
a numeric test comparing tiny-kernel-eligible shapes with and without a limit
against the reference MoE path.

## B. FlashInfer: world size 3 for `DcpAllToAll`

Fork patches `flashinfer-dcp-a2a-world3.patch` (`comm/pcie/pcie_dcp_a2a.py`,
`SUPPORTED_WORLD_SIZES` gains 3) and `flashinfer-tuning-world3.patch`
(`comm/pcie/_tuning.py`, `DcpAllToAll*.lse_reduce_scatter` and
`all_gather_heads` tables gain 3). Status on main: `SUPPORTED_WORLD_SIZES =
(2, 4, 8, 9, 10, 12, 16)` at line 42 and the tables at lines 27-28 and 32-33
lack 3. `PCIeTwoShotBF16.all_gather` already lists 3 (line 24).

Needed before submission: a b12x test exercising the all-to-all at world 3
(the fork validated only through Flare's vocabulary tests, proposal 03), and a
statement of which code path serves a non-power-of-two world.

## C. FlashInfer: packed MXFP4 Engram lookup

Covered by proposal 01. Status on main: `sequence/engram/_kernels.py` has no
`PACKED_FP4`; last changes to that file are from 2026-10-06.

## D. `--history-drafts` with DSpark: copied-history inputs

Helios accepts `--history-drafts` only when every speculative graph with token
capacity above one declares the copy inputs
(`helios/src/model/catalog.rs:93-118`, error "copied proposals require prepared
token, generation, cursor, and mask inputs"). Qwen3.8 declares them
(`ignition/recipes/qwen38/speculative_model.py:77-79`); DeepSeek's DSpark did
not, so `--history-drafts` was rejected for DeepSeek packages.

Fork: `SpeculativePreparation.COPY_INPUTS` and `copy_buffers()`
(`ignition/components/dspark.py:32-36`, buffers at `:79-82`),
`PreparedParallelDraftRound._apply_copied_proposals`
(`ignition/components/parallel_draft.py:368-387`, called at `:431`) replaces
learned drafts with the host-supplied continuation as point-mass proposals
over the matched prefix, and `bootstrap.py:269-273` advertises the inputs only
when vocabulary-parallel sampling is prepared, because only that branch
consumes them.

Rule to state in the PR: a graph advertises copy inputs only when its draft
path consumes them; advertising unconsumed inputs would make Helios accept
`--history-drafts` and silently do nothing. The DSpark path without
vocabulary-parallel sampling therefore keeps rejecting the flag until it
consumes the inputs too.

Evidence needed: acceptance length and tokens/s with and without
`--history-drafts` on a repetitive workload, at TP3 (fork) and TP4 (not
available to the fork). The fork has no recorded measurement of this feature.
Depends on proposal 03.

## E. Embedding-table host accounting and row-cache split

Fork changes in `helios/src/runtime_package/model.rs:651-690` and `:755-784`
and `helios/src/engram.rs:362-405` count RAM-resident tables toward the host
memory check, split the row-cache budget only among disk-backed tables, and
pass `resident_scales` only to disk readers. They exist to support per-table
storage selection through `FLARE_ENGRAM_RAM_TABLES` and
`FLARE_ENGRAM_RAM_LAYERS`, which are environment switches.

Upstream today has one storage mode for all tables (`--embedding-table-storage
disk|ram`), `embedding_cache_bytes` defaults to zero in RAM mode
(`helios/src/runtime_package/selection.rs:57-67`), and the reader ignores
`resident_scales` when resident (`helios/src/engram/reader.rs:788-812`). With
one mode there is no accounting defect to fix; the fork's accounting is only
meaningful together with a per-table setting. If maintainers want that
setting, the honest proposal is a CLI flag such as
`--embedding-table-ram-layers LIST`, carried in `LoadOptions`, plus this
accounting. Without that request, do not submit.

Separately, `reader.rs:724-741` splits gathers of 8192 or more rows from
RAM-resident tables across up to 16 threads. This is a generic prefill-side
change; it needs a measurement (prefill tokens/s with RAM tables, before and
after) before it qualifies as "a measured performance ... problem".

## F. `Recipe.host_offload_bytes`

`ignition/recipes/declaration.py:248`, `:303-305`. Only useful with proposal
02; ship there as plan-carried data, not a callable.

## Not proposed from this group

- `helios/src/execution.rs:43-48` `FLARE_SINGLE_SUBMISSION`: an environment
  debug switch.
- `bootstrap.py:657` tokenizer `sha256` in assets: Helios does not read it
  (no consumer found in `helios/src`); drop unless a consumer is added.

# Upstream contribution proposals

Proposals for carrying parts of the `ds41-tp3-v3` fork (HEAD 6f9fcba, based on
upstream 972e28e4 of https://github.com/NVIDIA/flare-local-inference) into
upstream Flare and the b12x FlashInfer fork
(https://github.com/local-inference-lab/flashinfer). Each file states the
problem, what the fork does today with file and line references at 6f9fcba,
a generic upstream design, compatibility, tests, open questions, and a commit
plan. Nothing here has been submitted; the fork's line numbers are the only
code evidence.

## Rules that shape every proposal

AGENTS.md, "Contribution scope and rejection criteria": "Every contribution
must address an explicit maintainer request, a demonstrated defect, a measured
performance or memory problem, or inaccurate documentation." and "Reject
unsupported abstractions and churn. Do not add frameworks, extension points,
speculative compatibility, fallback layers, or configuration switches without a
concrete requirement."

AGENTS.md, "Configuration": "Flare, Helios, recipes, and components never read
an environment variable to choose behavior. Feature toggles, experiment
switches, tuning overrides, debug modes, and alternate code paths selected
through the environment are prohibited, including temporary ones." Every
`FLARE_DS41_*`, `FLARE_ENGRAM_*`, and `FLARE_SINGLE_SUBMISSION` read in the
fork must become a recipe declaration, configuration value, or CLI flag, or
stay out.

AGENTS.md, "Architecture and ownership": "Admission, page reservation and
growth, eviction, preemption, readmission, and offload are shared runtime
policy and mechanisms. Do not select them by model or recipe identity". The
fork's expert placement (which experts live in host memory) is model-specific
offload policy and stays fork-only; only the storage mechanism is proposed.

AGENTS.md, "CUDA kernel and model optimization authorship": "Claude Opus 5.5
has repeatedly produced defective low-level kernel changes in this project. Do
not use it to design or implement low-level CUDA kernels or changes to existing
model fusion and GPU execution optimizations. This includes CUDA C++, PTX, CuTe
DSL, and Triton kernels ... Do not incorporate its generated changes to these
mechanisms through another contributor or agent." The rule names one model, not
AI authorship in general. The Triton `PACKED_FP4` decode (proposal 1) and the
BRRR and fused-QKV kernels are kernel changes; the submitter must state who or
what produced each and withhold anything produced by that model.

CONTRIBUTING.md requires `git commit -s` (DCO sign-off) on every commit.

## Recommended submission order

| Order | Proposal | Repository | Size | Blocking need |
|---|---|---|---|---|
| 1 | [05 small fixes](05-small-generic-fixes.md), tiny-decode `swiglu_limit` guard | flashinfer | 1 file | Standalone reproducer (TP1-3 shapes; independent of flare#193, which was checkpoint behaviour) |
| 2 | 05, world-size-3 DCP all-to-all tables | flashinfer | 2 files | A b12x test at world 3 |
| 3 | [03 vocabulary-parallel world sizes](03-vocabulary-parallel-world-sizes.md) | flare | small | Order 2 merged; TP4 regression run we cannot do |
| 4 | [04 shared all-reduce sizing](04-chunked-all-reduce-sizing.md) | flare | small | Confirm the DMA all-reduce divisibility constraint in b12x; prefill measurements |
| 5 | [01 MXFP4 Engram encoding](01-engram-mxfp4-encoding.md) | flashinfer, then flare | medium | Kernel authorship statement; decode round-trip test; maintainer view on lossy tables |
| 6 | [02 mixed device/host allocations](02-mixed-device-host-allocations.md) | flare | large | Explicit maintainer decision: upstream has no consumer without a placement policy |
| 7 | [06 memory dry run](06-kv-memory-dry-run.md) (`flare run --dry-run`) | flare | medium, Rust only | Calibration of the idle reserve and kernel/graph constants on SM103/SM121; maintainer view on storing the constants in the package |

05 also covers the `--history-drafts` input rule (ships with 03) and the
embedding-table accounting (only meaningful with a per-table storage setting
that does not exist upstream).

## What we can and cannot validate

Hardware available to the fork: three GB202 (SM120, 96 GB) GPUs on PCIe, one
host, Linux. We can run TP1 through TP3, `FLARE_RUN_TP_CUDA_TESTS=1` suites at
world 2 and 3, and export plus `flare run` qualification at TP3.

We cannot run: TP4 or larger, SM103, SM121 integrated GPUs (DGX Spark, GB10),
multi-host RoCE, Windows, or loading packages exported by upstream releases
other than 972e28e4. Every proposal lists these as unrun checks; upstream
reviewers or CI must cover them.

## Fork-only, not proposed

- Route-count expert placement: `hot_expert_ids`, `global_cold_plan`,
  `host_expert_layers`, `merged_bank` in `ignition/recipes/deepseek41/weights.py:129-195`
  and `_parked_bytes` in `ignition/recipes/deepseek41/recipe.py:52-67`. Model-specific
  offload policy driven by environment variables and external route-count files.
- Probes: `ignition/recipes/deepseek41/probe.py` and the `_probe.*` calls in
  `decoder.py`. Development instrumentation; AGENTS.md: "Do not create development
  audit reports, receipts, provenance bundles, or hash inventories."
- BRRR kernels and wiring (`brrr.py`, `attention.py` `brrr_sparse`, `decoder.py`
  tail, `fused_qkv_norm.py`): fused GPU execution changes selected through
  `TP3Config`; the prologue (no measured gain, commit 953859e) was removed
  2026-11-08; authorship rule.
- Debug toggles: `FLARE_SINGLE_SUBMISSION` (`helios/src/execution.rs:43-48`),
  `FLARE_DS41_NO_ENGRAM`, `FLARE_DS41_EAGER_PROFILE`, `FLARE_DS41_MINI`,
  `FLARE_DS41_JOINT_HEADROOM_MIB`, `FLARE_DS41_TP3_GRAPH_GIB`,
  `FLARE_DS41_EMBED_HOST`, `FLARE_DS41_VISION_HOST`, `FLARE_ENGRAM_RAM_TABLES`,
  `FLARE_ENGRAM_RAM_LAYERS`. Each is an environment switch (the `FLARE_DS41_*`
  ones are read only by `TP3Config.from_environment()` in the fork;
  `FLARE_DS41_QB_ATOMIC` was removed 2026-11-08).
- FlashInfer `flashinfer-dense-splitk-deferral.patch` and
  `flashinfer-tuning-route-reuse.patch`: experiments; route reuse measured no gain
  at 450 W.
- `ignition/preparation/ladders/gb202_pcie_tp3.json`: a measured offline ladder
  for three GB202 GPUs. Usable upstream only with `--collective-tuning runtime`;
  offered as an optional commit in proposal 04.

## Effort

Orders 1 to 4 are each under a day of engineering plus the GPU runs listed.
Proposal 01 is two to three days across two repositories. Proposal 02 is a week
or more: schema in two Protobuf files, Rust mapping and six accounting sites,
Python storage and export, planner hook, and tests on both sides, and it cannot
land without a maintainer decision on the consumer.

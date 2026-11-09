# 06. `flare run --dry-run`: memory plan and KV capacity without loading

## Problem

The only way to learn the KV capacity of a package under a set of serving flags
(`--prefix-cache-memory`, the verification budget, `--kv-cache-limit`,
`--memory-limit`, the GPUs chosen) is a full `flare run`. A load takes minutes,
needs the GPUs idle, and so means stopping the server that is using them. While
tuning the TP3 fork we repeatedly blamed a build for a KV change that came from
a serving flag (a 32 GiB host prefix cache costs device memory for its intake;
verification budget 24 instead of 128 drops 13 graphs and adds about 87K KV
tokens). A planner that reproduces the loader's arithmetic answers the same
question in seconds while the server keeps running.

## What the fork does

`flare run PACKAGE [serving flags] --dry-run` verifies the package and resolves
the configuration file, model defaults and `--gpus` exactly as a load does,
then prints the per-rank memory plan and the announced cache line (bytes, KV
tokens, shared snapshot slots, what limited the pool) plus the host RAM the
load commits. It creates no CUDA context, allocates no GPU memory and reads no
weights; the driver is initialized only to read each GPU's total
(`cuDeviceTotalMem`). Design text: `docs/cli-design.md`, section "Memory dry run".

- `flare/src/dry_run.rs` (new, ~280 lines): report and the CLI entry point.
- `helios/src/runtime_package/plan.rs` (new, ~900 lines): `plan_memory`,
  `device_total_memory`, `IDLE_DEVICE_RESERVE_BYTES` (606 MiB), unit tests.
- `helios/src/runtime_package/memory.rs`, `model.rs`: cache sizing factored so
  the loader and the planner share one implementation of the page, plane and
  row arithmetic (the planner cannot drift from `Allocate cache`).
- `flare/src/cli.rs`: `--dry-run` (conflicts with `--hosts`, `--rank`,
  `--rendezvous`, `--stage-only`) and `--dry-run-free-memory SIZE[,SIZE...]`.
- Fork commits: db9dea41 (feature), 34716793 (calibration against the driver total).

Plan per rank: (1) assumed free memory = driver total minus 606 MiB, or
`--dry-run-free-memory`; (2) fixed device allocations exactly (weights,
execution state, scratch, device parts of mixed banks); (3) an estimate of
driver memory for the selected kernels and graphs (135.875 MiB base, module
images plus 11,776 B per module, module globals, 3,584 B per graph node, after
the serving flags select graphs and prune kernels); (4) runtime headroom
(package or `--memory-limit` reserve plus 64 MiB); (5) the host prefix cache's
device share; (6) rows and KV pages as `Allocate cache` sizes them, the smallest
rank governing.

## Accuracy (fork measurements, 3x RTX PRO 6000 Blackwell 96 GB)

| Package | Flags | Dry run | Served | Error |
| --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash TP3 (calibration) | 32G prefix cache, budget 24 | 2,461,184 | 2,461,184 | exact |
| DeepSeek-V4.1-Flash TP3 (calibration) | 16G, budget 128 | 2,373,760 | 2,373,760 | exact |
| DeepSeek-V4.1-Flash TP3, 3 other recorded loads | various | | | exact |
| GLM-5.3-Flash TP3 (not in calibration) | 32G, 5 drafts | 4,214,016 | 4,229,888 | -0.4 % |
| GLM-5.3-Flash TP2, DCP2 (not in calibration) | 32G, 5 drafts | 1,890,304 | 1,914,368 | -1.3 % |

Items 2, 4, 5 and 6 are exact by construction. The residual is item 1 (576 to
606 MiB unavailable before loading across ranks; the larger value is assumed so
the plan is conservative) and the kernel/graph estimate of item 3, calibrated
on one model family. The two GLM packages, a different architecture with
different kernels, under-predict by 0.4 % and 1.3 %. One MiB of error on the
limiting rank moves DeepSeek TP3's capacity by about 1,200 tokens.

## Upstream design

As in the fork, no change needed for upstream rules: no environment variables,
flags only, no model or recipe identity in the planner. Open points for
maintainers:

- The 606 MiB idle reserve and the kernel/graph constants were measured on SM120
  (GB202). SM103 and SM121 (unified memory on GB10) need their own measurement;
  GB10 may need a different notion of "free device memory" entirely.
- Should the constants live in the package (measured at export, as graph memory
  already is) instead of in Helios? That would remove the calibration residual.
- Multi-host and `--prefix-cache-pool` are refused today; their capacity depends
  on rank agreement and pool registration.
- Output format: human report only today; a `--json` form would let tooling use it.

## Tests

Unit tests in `plan.rs` (four) and `dry_run.rs` (report shape, explicit free
memory). Validation we ran: the five DeepSeek loads above plus two GLM packages.
Unrun: TP4+, SM103, SM121, multi-host, packages from upstream releases other
than 972e28e4.

## Commit plan

1. Factor cache sizing out of `memory.rs`/`model.rs` into a shared function (no
   behaviour change; existing load tests).
2. `plan.rs` with unit tests.
3. `flare run --dry-run` and `--dry-run-free-memory` plus `docs/cli-design.md`.

Size: about 1.4K added lines in Rust plus docs. It does not depend on any other
proposal and touches no kernels, so the authorship rule does not apply.

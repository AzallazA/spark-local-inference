# 02. Allocations with a device prefix and a host-memory tail

## Problem

A tensor-parallel rank can hold more weights than its VRAM if part of a weight
bank lives in pinned host memory that the GPU reads over PCIe. For a routed
expert bank the useful split is by rows (experts): the frequently routed rows
on the device, the rest in host memory, in one contiguous virtual range so a
single MoE kernel call reads either. Upstream has two memory kinds, `DEVICE`
and `MAPPED_HOST`, and no way to express one allocation spanning both. The
fork's published TP3 build uses this to fit DeepSeek V4.1 Flash on three 96 GB
GPUs with 2.2 M KV tokens; without it the model does not fit at TP3 with
usable KV capacity.

## What the fork does today (6f9fcba)

- `ignition/preparation/package_storage.py:260-278` `mixed_allocate(name, shape, dtype, *, device_rows)`
  and `MixedRegion` (`:495-543`): two `cuMemCreate` calls (location type 1
  device, location type 3 host NUMA from `cuDeviceGetAttribute(134)`), two
  `cuMemMap` calls into one range at the device allocation cursor, access for
  the owning device, zero fill. The boundary page stays on the device.
- `ignition/preparation/runtime_package.py:206-210` exports each region as a
  `DEVICE` allocation named `mixed-<index>-<device bytes>`.
- Helios parses that name: `helios/src/runtime_package/mod.rs:1220-1231`
  (`mixed_device_bytes`, `mixed_device_bytes_named`), maps prefix and tail with
  two create/map pairs (`mod.rs:498-531`), records two mappings for unmapping
  (`:587-593`), and uses the device prefix at six accounting sites:
  `mod.rs:462`, `memory.rs:230`, `model.rs:160` and `:704-708`,
  `protection.rs:2640`, `describe.rs:328`.
- Planner: `Recipe.host_offload_bytes: Callable[[int], int]`
  (`ignition/recipes/declaration.py:248`, used at `:303-305`) subtracts parked
  bytes from the per-rank resident estimate.
- Loader: `_park_on_host` (`ignition/components/weights.py:715-719`) and the
  `host_allocator` argument of `load_fp4_experts`/`load_mxfp4_experts`
  (`:1453-1455`), plus a guard that b12x did not re-materialize parked tensors
  on the device (`:1489-1497`).
- Policy (fork-only): which layers and experts are parked comes from
  environment variables and route-count files
  (`ignition/recipes/deepseek41/weights.py:129-195`, `recipe.py:52-67`).

Encoding the split in an allocation name is the defect to fix before any of
this can go upstream.

## Proposed upstream design

Schema. Add `optional uint64 device_bytes` to `Allocation` in
`helios/proto/package.proto` (field 8) and `helios/proto/runtime_package.proto`
(field 10). Meaning: for `memory == DEVICE`, bytes `[0, device_bytes)` are
device memory and `[device_bytes, nbytes)` are pinned host memory on the
device's NUMA node, mapped contiguously at `address`. Absent, or equal to
`nbytes`, means a plain device allocation. The alternative, a `MIXED` memory
kind with `device_prefix_bytes`, forces every `match` on `Memory` to grow a
third arm; a field on `DEVICE` keeps every existing consumer correct by default
and is the recommended form.

Validation (`helios/src/runtime_package/format.rs` `validate`, around
`:299-330`): `device_bytes` only with `DEVICE`; `0 < device_bytes <= nbytes`;
`device_bytes % alignment == 0` (alignment is the VMM granularity); not
combined with `sizing`; the whole range inside a reservation as today.

Helios mapping (`mod.rs:498-531`): the fork's two-segment code, keyed on
`allocation.device_bytes` instead of the name. Peer access: the owning rank's
`cuMemSetAccess` (`mod.rs:610`) covers the whole range so peer ranks that
reference it see both segments.

Accounting: a `format::Allocation::device_resident_bytes()` helper replaces the
six name-parsing call sites; host tail bytes count as host memory in
`model.rs:704-708`, `describe.rs` (`mapped_host_bytes` plus a new
`device_host_tail_bytes`), and `memory.rs` `fixed_bytes`.

Ignition: `PackageStorage.mixed_allocate` stays; export sets
`Allocation.device_bytes = region.device_bytes` and a plain name.

Planner: `Recipe.host_offload_bytes` becomes data the plan carries, not a
callable reading the environment. The amount is a preparation setting
(AGENTS.md: "A setting that only experiments need is an advanced flag carried
through the plan into the recipe's `PreparationConfig`"), so the planner reads
`PreparationConfig.host_parked_bytes` and the report prints it. Which rows are
parked remains the recipe's or fork's business.

## Migration and compatibility

Packages without `device_bytes` load exactly as today. The fork's name-encoded
packages were never produced by upstream, so upstream should not learn the name
form; AGENTS.md rejects "speculative compatibility". The fork keeps its name
parser for one release and re-exports.

## Tests

Rust:
- `format.rs`: accept a valid split; reject `device_bytes > nbytes`, misaligned
  prefix, `MAPPED_HOST` with `device_bytes`, `sizing` with `device_bytes`.
- `describe.rs` (extend `rank_bytes_split_by_memory_kind_and_graphs_report_packed_tokens`,
  `:587`): prefix counted as device, tail as host.
- `memory.rs`: `fixed_bytes` uses the prefix.

Python:
- `tests/preparation/test_package_storage_cuda.py`: `mixed_allocate`, write
  through `device_view`, read back both segments, close.
- `tests/preparation/test_runtime_package.py`: exported `Allocation` carries
  `device_bytes`; a package with a host tail loads in Helios on one GPU and a
  graph reading across the boundary reproduces reference values.

Not runnable by the fork: TP4, SM103, integrated GPUs (where "host NUMA"
memory is the same DRAM; `cuMemCreate` with location type 3 may behave
differently and needs a decision, probably rejection on integrated devices).

## Open questions

- Consumer: upstream has no recipe that parks weights. AGENTS.md: "Hypothetical
  future consumers ... do not establish a need for a change." This needs an
  explicit maintainer request, for example for a DeepSeek V4.1 TP3
  configuration on 96 GB GPUs, before the mechanism is accepted.
- Does `cuMemSetAccess` from a peer device work on host-NUMA physical memory?
  The fork grants access only to the owner.
- Boundary rounding: the fork rounds the device prefix up to granularity; the
  schema must store the rounded value so Helios maps exactly what Ignition did.
- Signing: `protection.rs` hashes allocation metadata; confirm the field joins
  the signed set.

## PR plan

1. Protobuf field, `format.rs` validation, unit tests.
2. Helios two-segment mapping and the accounting helper at the six sites, with
   `describe.rs`/`memory.rs` tests.
3. Ignition `MixedRegion` export with `device_bytes`, storage and export
   tests.
4. Planner `host_parked_bytes` through `PreparationConfig` and the report.
5. Loader `host_allocator` support in `load_fp4_experts` with the
   re-materialization guard, only once a recipe uses it.

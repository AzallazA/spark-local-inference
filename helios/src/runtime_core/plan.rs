/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Size a core's caches for selected GPUs without initializing CUDA.
//!
//! The loader sizes KV pages from what is free on each GPU after weights, kernels, and graphs
//! are resident. This module reproduces that arithmetic from the core metadata, the serving
//! options, and an assumed free memory per device, so a serving configuration can be judged
//! without a load. Fixed allocations, cache geometry, headroom, and the prefix cache's device
//! share are exact; the memory the CUDA driver takes for loaded modules and instantiated graphs
//! is estimated from the module and graph counts the core declares.

use super::{
    AuthorizedCore, KvCacheLimit, LoadOptions, MemoryLimit,
    format::{Memory, Core, invalid},
    memory::{self, CacheSizing, MemorySnapshot},
    model::{Staged, engram_resident_bytes},
};
use crate::{Result, startup::StartupProgress};

/// Device memory assumed unavailable to Flare on an otherwise idle GPU before loading: the CUDA
/// context, driver bookkeeping, and the display, measured against the device total the CUDA
/// driver reports (`cuDeviceTotalMem`). That total already excludes the memory `nvidia-smi` lists
/// as reserved (547 MiB on a 97,887 MiB RTX PRO 6000 Blackwell, whose driver total is 97,340.875
/// MiB). The dry run subtracts this reserve from the driver total unless the caller supplies the
/// free memory. Calibrated with the constants below.
pub const IDLE_DEVICE_RESERVE_BYTES: u64 = 606 << 20;

/// Device memory one rank's load takes beyond its fixed allocations and the per-module and
/// per-node terms: streams, events, graph executables, and page tables of the reserved virtual
/// ranges. 135.875 MiB.
const RUNTIME_BASE_BYTES: u64 = 139_136 << 10;

/// Device memory the driver takes per loaded module beyond its stored image: function
/// descriptors, constant banks, and allocation granularity.
const MODULE_OVERHEAD_BYTES: u64 = 11776;

/// Device memory per instantiated graph node: kernel parameter blocks and launch descriptors
/// uploaded with the graph executable.
const GRAPH_NODE_BYTES: u64 = 3584;

// Calibration of the three constants above and of IDLE_DEVICE_RESERVE_BYTES: five recorded loads
// of DeepSeek-V4.1-Flash tensor-parallel-3 cores on three RTX PRO 6000 Blackwell GPUs (driver
// total 102,069,305,344 bytes each), two complete cores with 6,138 or 7,026 selected modules
// and 91,448 or 109,408 graph nodes and two truncated test cores with 5,663 or 6,911 modules
// and 38,587 or 45,928 nodes. Each load's "Allocate cache" line fixes the limiting rank's cache
// budget to within one page step (8 MiB for the complete cores, 2 MiB for the truncated
// ones); the constants place every planned budget inside its recorded step. The split between
// the idle reserve and the per-rank base is not observable separately; only their sum,
// 741.875 MiB below the driver total, is measured.

/// Assumed state of one selected GPU before loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedDevice {
    /// The user-visible GPU index.
    pub label: i32,
    pub total_bytes: u64,
    /// Memory assumed free before Flare allocates anything.
    pub free_bytes: u64,
}

/// Driver-managed device memory one rank's kernels and graphs are estimated to take.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeEstimate {
    pub modules: usize,
    /// Stored (compressed) bytes of the module images.
    pub module_bytes: u64,
    /// Bytes of module global variables, which the loader uploads to device memory.
    pub global_bytes: u64,
    pub graphs: usize,
    pub graph_nodes: u64,
    /// The estimate: the per-rank base, globals, stored module images with per-module overhead,
    /// and per-node graph memory.
    pub bytes: u64,
}

/// Estimate the device memory the selected kernels and graphs of `core` take once loaded.
pub(super) fn runtime_estimate(core: &Core) -> RuntimeEstimate {
    let module_bytes = core
        .modules
        .iter()
        .filter_map(|module| core.resources.get(&module.file))
        .map(|span| span.nbytes)
        .sum::<u64>();
    let global_bytes = core
        .modules
        .iter()
        .flat_map(|module| &module.globals)
        .map(|global| global.nbytes as u64)
        .sum::<u64>();
    let graph_nodes = core
        .graphs
        .iter()
        .map(|graph| graph.node_count as u64)
        .sum::<u64>();
    let bytes = RUNTIME_BASE_BYTES
        .saturating_add(global_bytes)
        .saturating_add(module_bytes)
        .saturating_add((core.modules.len() as u64).saturating_mul(MODULE_OVERHEAD_BYTES))
        .saturating_add(graph_nodes.saturating_mul(GRAPH_NODE_BYTES));
    RuntimeEstimate {
        modules: core.modules.len(),
        module_bytes,
        global_bytes,
        graphs: core.graphs.len(),
        graph_nodes,
        bytes,
    }
}

/// One rank's planned device memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankPlan {
    pub device: PlannedDevice,
    /// Device bytes of fixed allocations: weights, execution state, scratch, and the device
    /// prefix of mixed device/host expert banks.
    pub fixed_device_bytes: u64,
    /// Host bytes of mixed device/host allocations.
    pub mixed_host_bytes: u64,
    /// Bytes of mapped host allocations.
    pub mapped_host_bytes: u64,
    /// Payload bytes the loader copies into the fixed allocations.
    pub stored_bytes: u64,
    pub runtime: RuntimeEstimate,
    /// Memory left free below the budget: the core reserve or the memory limit's reserve, plus
    /// engine headroom.
    pub headroom_bytes: u64,
    /// Device bytes set aside for host prefix cache transfer streams and intake.
    pub prefix_cache_device_bytes: u64,
    /// Memory available for execution rows and KV pages.
    pub cache_budget: u64,
    /// Bytes the planned execution rows and KV pages occupy on this rank.
    pub cache_bytes: u64,
}

/// Host memory the load commits beyond the process itself.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostPlan {
    /// Mixed device/host expert banks across ranks.
    pub mixed_host_bytes: u64,
    /// Mapped host allocations across ranks.
    pub mapped_host_bytes: u64,
    /// Embedding table planes kept in RAM: every plane of a RAM table, the scale plane of a disk
    /// table with resident scales.
    pub engram_resident_bytes: u64,
    /// Pinned staging frames for embedding gathers.
    pub engram_staging_bytes: u64,
    /// Row cache for disk-backed embedding tables.
    pub embedding_cache_bytes: u64,
    /// Configured warm retention memory for prefix checkpoints and prepared inputs.
    pub prefix_cache_memory_bytes: u64,
    /// Host memory the prefix cache occupies in total: retention memory, intake slots, input
    /// staging, and its index.
    pub prefix_cache_host_bytes: u64,
    /// Ordinary memory for serialized prepared image inputs.
    pub input_window_bytes: u64,
}

/// The memory plan of one serving configuration: what the loader would allocate and the KV
/// capacity it would announce, under the assumed free memory of each device.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryPlan {
    pub ranks: Vec<RankPlan>,
    pub host: HostPlan,
    pub active: usize,
    /// Continuation rows one active request needs.
    pub width: u64,
    /// Rows backed for execution: one null row plus `active * width`.
    pub execution_rows: u64,
    /// Total continuation rows, including snapshot slots aliased into KV backing.
    pub rows: u64,
    pub snapshots: u64,
    pub pages: u64,
    /// Tokens per logical page, as the scheduler counts them.
    pub page_tokens: u64,
    pub kv_tokens: u64,
    pub limiting: String,
    /// Cache bytes across ranks, as the "Allocate cache" stage reports.
    pub cache_bytes: u64,
    pub kv_cache_limit: KvCacheLimit,
    pub memory_limit: Option<MemoryLimit>,
}

/// Host-side accounting the plan reports and, on integrated GPUs, the loader would budget.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct EmbeddingAccounting {
    pub resident_bytes: u64,
    pub staging_bytes: u64,
    pub cache_bytes: u64,
}

/// Plan `authorized` on `devices` under `options` without initializing CUDA.
///
/// Every device is treated as discrete: an integrated GPU's shared-memory budget is not modeled.
/// A multi-host core and a prefix cache pool are refused, because their capacities depend on
/// rank agreement and pool registration at startup.
pub fn plan_memory(
    authorized: AuthorizedCore,
    devices: &[PlannedDevice],
    options: &LoadOptions,
) -> Result<MemoryPlan> {
    if devices.is_empty() {
        return Err(invalid("the memory dry run requires at least one device"));
    }
    if options.prefix_cache.pool.is_some() {
        return Err(invalid(
            "the memory dry run does not model a prefix cache pool; its device charges are known only at registration",
        ));
    }
    let options = LoadOptions {
        device_labels: Some(devices.iter().map(|device| device.label).collect()),
        ..options.clone()
    };
    let ordinals: Vec<i32> = (0..devices.len() as i32).collect();
    let staged = Staged::prepare(authorized, &ordinals, &options, &StartupProgress::silent())?;
    if staged.cluster.is_some() {
        return Err(invalid(
            "the memory dry run plans a single-host core; multi-host ranks agree on capacity at rendezvous",
        ));
    }
    let resident_scales = staged
        .options
        .embedding_resident_scales(staged.has_disk_tables, false);
    let embedding = EmbeddingAccounting {
        resident_bytes: engram_resident_bytes(
            &staged.engram,
            &staged.table_storage,
            resident_scales,
        )?,
        staging_bytes: staged.staging_bytes,
        cache_bytes: staged
            .options
            .embedding_cache_bytes(staged.has_disk_tables, false),
    };
    plan_ranks(
        &staged.cores,
        &staged.server_config,
        &staged.options,
        staged.active,
        devices,
        embedding,
    )
}

/// Size caches for `cores` as the loader would, from assumed device memory instead of
/// observed memory.
pub(super) fn plan_ranks(
    cores: &[Core],
    server_config: &serde_json::Value,
    options: &LoadOptions,
    active: usize,
    devices: &[PlannedDevice],
    embedding: EmbeddingAccounting,
) -> Result<MemoryPlan> {
    if cores.len() != devices.len() {
        return Err(invalid(
            "planned device count must match the core's ranks",
        ));
    }
    let CacheSizing {
        reserve,
        page_tokens,
        contract,
        width,
        input_window,
        offload_bytes,
        offload_host,
        ..
    } = memory::cache_sizing(cores, server_config, options, active)?;
    let headroom = match options.memory_limit {
        None => reserve,
        Some(MemoryLimit::Reserve(user)) => user
            .checked_add(memory::HEADROOM_BYTES)
            .ok_or_else(|| invalid("memory reserve overflows the device budget"))?,
        Some(_) => memory::HEADROOM_BYTES,
    };
    let mut ranks = Vec::with_capacity(cores.len());
    let mut available = Vec::with_capacity(cores.len());
    let mut host = HostPlan {
        engram_resident_bytes: embedding.resident_bytes,
        engram_staging_bytes: embedding.staging_bytes,
        embedding_cache_bytes: embedding.cache_bytes,
        prefix_cache_memory_bytes: options.prefix_cache.memory_bytes.unwrap_or(0),
        prefix_cache_host_bytes: offload_host,
        input_window_bytes: input_window,
        ..Default::default()
    };
    for (core, &device) in cores.iter().zip(devices) {
        let fixed_device_bytes = memory::fixed_bytes(core, false)?;
        let (mut mixed_host_bytes, mut mapped_host_bytes, mut stored_bytes) = (0u64, 0u64, 0u64);
        for allocation in &core.allocations {
            match allocation.memory {
                Memory::Device => {
                    let device_bytes =
                        super::mixed_device_bytes(allocation).unwrap_or(allocation.nbytes);
                    mixed_host_bytes += (allocation.nbytes - device_bytes) as u64;
                }
                Memory::MappedHost => mapped_host_bytes += allocation.nbytes as u64,
                Memory::Peer => {}
            }
            stored_bytes += allocation.stored_bytes() as u64;
        }
        host.mixed_host_bytes += mixed_host_bytes;
        host.mapped_host_bytes += mapped_host_bytes;
        let runtime = runtime_estimate(core);
        let loaded = fixed_device_bytes
            .checked_add(runtime.bytes)
            .ok_or_else(|| invalid("loaded memory footprint overflows the device budget"))?;
        let snapshot = MemorySnapshot {
            free: device.free_bytes.saturating_sub(loaded),
            total: device.total_bytes,
        };
        let cache_budget = memory::limited_budget(
            snapshot,
            loaded,
            options.memory_limit,
            reserve,
            false,
            offload_bytes,
        )?;
        available.push(cache_budget);
        ranks.push(RankPlan {
            device,
            fixed_device_bytes,
            mixed_host_bytes,
            mapped_host_bytes,
            stored_bytes,
            runtime,
            headroom_bytes: headroom,
            prefix_cache_device_bytes: offload_bytes,
            cache_budget,
            cache_bytes: 0,
        });
    }
    let context = server_config["max_sequence_tokens"]
        .as_u64()
        .ok_or_else(|| invalid("model requires a prepared context ceiling"))?;
    let ceiling = memory::page_ceiling(options.kv_cache_limit, context, active as u64, &contract)?;
    let (rows, pages, snapshots) = memory::capacities(
        cores,
        &available,
        active as u64,
        width,
        page_tokens,
        ceiling,
    )?;
    let execution_rows = rows - snapshots;
    let mut cache_bytes = 0u64;
    for (rank, core) in ranks.iter_mut().zip(cores) {
        rank.cache_bytes = memory::rank_cache_bytes(core, execution_rows, pages)?;
        cache_bytes += rank.cache_bytes;
    }
    let logical_page_tokens = contract.cache.page_tokens as u64;
    Ok(MemoryPlan {
        ranks,
        host,
        active,
        width,
        execution_rows,
        rows,
        snapshots,
        pages,
        page_tokens: logical_page_tokens,
        kv_tokens: pages * logical_page_tokens,
        limiting: memory::limiting_factor(cores, ceiling, pages, options.kv_cache_limit),
        cache_bytes,
        kv_cache_limit: options.kv_cache_limit,
        memory_limit: options.memory_limit,
    })
}

/// Total memory of each selected CUDA device, by ordinal under the process's device visibility,
/// read through the driver without creating a context or allocating device memory.
pub fn device_total_memory(ordinals: &[i32]) -> Result<Vec<u64>> {
    let driver = super::driver::Driver::load()?;
    // SAFETY: Driver signatures are checked against cuda.h; the output storage is valid, and
    // cuInit precedes every device query.
    unsafe {
        driver.check("cuInit", (driver.init)(0))?;
        ordinals
            .iter()
            .map(|&ordinal| {
                let (mut device, mut total) = (0, 0usize);
                driver.check("cuDeviceGet", (driver.device)(&mut device, ordinal))?;
                driver.check(
                    "cuDeviceTotalMem_v2",
                    (driver.device_total_memory)(&mut total, device),
                )?;
                Ok(total as u64)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        format::{Allocation, Context, Global, Graph, Module},
        protobuf::wire,
    };
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    fn allocation(
        name: &str,
        nbytes: usize,
        memory: Memory,
        sizing: Option<wire::Sizing>,
    ) -> Allocation {
        Allocation {
            name: name.into(),
            nbytes,
            alignment: 65536,
            file: None,
            data_spans: if sizing.is_none() && nbytes > 0 {
                vec![[0, nbytes]]
            } else {
                vec![]
            },
            sources: vec![],
            relocations: vec![],
            address: Some(0x200000000000),
            memory,
            sizing,
        }
    }

    /// One rank: `weights` fixed device bytes, a mixed expert bank with a 1 KiB device prefix and a
    /// 3 KiB host tail, a 2 MiB mapped host allocation, 4 KiB continuation rows, and 1 MiB KV pages.
    fn core(weights: usize, row_bytes: u64, page_bytes: u64) -> Core {
        let mut resources = std::collections::HashMap::new();
        for (file, nbytes) in [("a.cubin", 1000u64), ("b.cubin", 2000)] {
            resources.insert(
                file.to_owned(),
                wire::Span {
                    nbytes,
                    ..Default::default()
                },
            );
        }
        Core {
            resources,
            events: vec![],
            architecture: [12, 0],
            modules: [("a", "a.cubin", 100usize), ("b", "b.cubin", 0)]
                .into_iter()
                .map(|(name, file, global)| Module {
                    name: name.into(),
                    file: file.into(),
                    globals: (global > 0)
                        .then(|| Global {
                            symbol: "table".into(),
                            file: "table.bin".into(),
                            nbytes: global,
                            relocations: vec![],
                        })
                        .into_iter()
                        .collect(),
                })
                .collect(),
            functions: vec![],
            graphs: [10usize, 20, 30]
                .into_iter()
                .enumerate()
                .map(|(index, node_count)| Graph {
                    name: format!("graph-{index}"),
                    flags: 0,
                    resource: format!("graph-{index}"),
                    node_count,
                    edge_count: node_count.saturating_sub(1),
                    functions: vec![],
                })
                .collect(),
            reservations: vec![],
            execution: Some(wire::Execution {
                abi_version: crate::execution::ABI_VERSION,
                max_active: 4,
                graphs: vec![wire::ExecutionGraph {
                    name: "Decode-1-1".into(),
                    kind: "decode".into(),
                    sampling_mode: "general".into(),
                    batch_capacity: 1,
                    token_capacity: 1,
                    ..Default::default()
                }],
                ..Default::default()
            }),
            context: Context::default(),
            allocations: vec![
                allocation("weights", weights, Memory::Device, None),
                allocation("mixed-0-1024", 4096, Memory::Device, None),
                allocation("host-table", 2 << 20, Memory::MappedHost, None),
                allocation(
                    "state",
                    0,
                    Memory::Device,
                    Some(wire::Sizing {
                        capacity: 0,
                        unit_bytes: row_bytes,
                        maximum: 4096,
                    }),
                ),
                allocation(
                    "kv",
                    0,
                    Memory::Device,
                    Some(wire::Sizing {
                        capacity: 1,
                        unit_bytes: page_bytes,
                        maximum: 100_000,
                    }),
                ),
            ],
        }
    }

    fn server_config() -> serde_json::Value {
        serde_json::json!({
            "model_contract": serde_json::to_value(crate::model::testing::metadata()).unwrap(),
            "page_tokens": 64,
            "max_sequence_tokens": 64 * 1000,
            "reserve_bytes": 0,
        })
    }

    fn device(label: i32, free: u64) -> PlannedDevice {
        PlannedDevice {
            label,
            total_bytes: 32 * GIB,
            free_bytes: free,
        }
    }

    #[test]
    fn runtime_estimate_counts_selected_modules_globals_and_graph_nodes() {
        let estimate = runtime_estimate(&core(1 << 30, 4096, 1 << 20));
        assert_eq!(
            estimate,
            RuntimeEstimate {
                modules: 2,
                module_bytes: 3000,
                global_bytes: 100,
                graphs: 3,
                graph_nodes: 60,
                bytes: RUNTIME_BASE_BYTES
                    + 100
                    + 3000
                    + 2 * MODULE_OVERHEAD_BYTES
                    + 60 * GRAPH_NODE_BYTES,
            }
        );
        let empty = runtime_estimate(&Core {
            modules: vec![],
            graphs: vec![],
            ..core(0, 4096, 1 << 20)
        });
        assert_eq!(empty.bytes, RUNTIME_BASE_BYTES);
    }

    #[test]
    fn plan_budgets_each_rank_from_assumed_free_memory_and_sizes_pages_for_the_smallest() {
        let cores = [
            core(10 * GIB as usize, 4096, 1 << 20),
            core(11 * GIB as usize, 4096, 1 << 20),
        ];
        let devices = [device(0, 20 * GIB), device(1, 20 * GIB)];
        let options = LoadOptions::default();
        let embedding = EmbeddingAccounting {
            resident_bytes: 7,
            staging_bytes: 8,
            cache_bytes: 9,
        };
        let plan = plan_ranks(
            &cores,
            &server_config(),
            &options,
            4,
            &devices,
            embedding,
        )
        .unwrap();
        assert_eq!(plan.ranks.len(), 2);
        for (rank, core) in plan.ranks.iter().zip(&cores) {
            // The mixed bank contributes its 1 KiB device prefix to fixed bytes and 3 KiB to the host.
            let fixed = core.allocations[0].nbytes as u64 + 1024;
            assert_eq!(rank.fixed_device_bytes, fixed);
            assert_eq!(rank.mixed_host_bytes, 3072);
            assert_eq!(rank.mapped_host_bytes, 2 << 20);
            assert_eq!(rank.stored_bytes, fixed + 3072 + (2 << 20));
            assert_eq!(rank.runtime, runtime_estimate(core));
            assert_eq!(rank.headroom_bytes, memory::HEADROOM_BYTES);
            assert_eq!(rank.prefix_cache_device_bytes, 0);
            assert_eq!(
                rank.cache_budget,
                20 * GIB - fixed - rank.runtime.bytes - memory::HEADROOM_BYTES
            );
            assert_eq!(
                rank.cache_bytes,
                memory::rank_cache_bytes(core, plan.execution_rows, plan.pages).unwrap()
            );
            assert!(rank.cache_bytes <= rank.cache_budget);
        }
        // Rank 1 holds 1 GiB more weights, so its budget bounds the shared page count.
        let tight = &plan.ranks[1];
        assert_eq!(plan.execution_rows, 1 + 4);
        assert!(tight.cache_budget - tight.cache_bytes < MIB + 65536);
        assert_eq!(plan.rows, plan.execution_rows + plan.snapshots);
        assert_eq!(plan.kv_tokens, plan.pages * 64);
        assert_eq!(plan.page_tokens, 64);
        assert_eq!(plan.limiting, "memory");
        assert_eq!(
            plan.cache_bytes,
            plan.ranks.iter().map(|r| r.cache_bytes).sum::<u64>()
        );
        assert_eq!(
            plan.host,
            HostPlan {
                mixed_host_bytes: 2 * 3072,
                mapped_host_bytes: 2 * (2 << 20),
                engram_resident_bytes: 7,
                engram_staging_bytes: 8,
                embedding_cache_bytes: 9,
                ..Default::default()
            }
        );
        // Granting the tight rank the same free memory as its peer raises the plan by its deficit.
        let even = plan_ranks(
            &cores,
            &server_config(),
            &options,
            4,
            &[device(0, 20 * GIB), device(1, 21 * GIB)],
            embedding,
        )
        .unwrap();
        assert_eq!(even.pages, plan.pages + 1024);
    }

    #[test]
    fn plan_applies_memory_limits_kv_ceilings_and_prefix_cache_headroom() {
        let cores = [core(10 * GIB as usize, 4096, 1 << 20)];
        let devices = [device(0, 20 * GIB)];
        let config = server_config();
        let base = plan_ranks(
            &cores,
            &config,
            &LoadOptions::default(),
            4,
            &devices,
            Default::default(),
        )
        .unwrap();
        let reserve = plan_ranks(
            &cores,
            &config,
            &LoadOptions {
                memory_limit: Some(MemoryLimit::Reserve(GIB)),
                ..Default::default()
            },
            4,
            &devices,
            Default::default(),
        )
        .unwrap();
        assert_eq!(
            reserve.ranks[0].headroom_bytes,
            GIB + memory::HEADROOM_BYTES
        );
        assert_eq!(
            reserve.ranks[0].cache_budget,
            base.ranks[0].cache_budget - GIB
        );
        assert_eq!(reserve.pages, base.pages - 1024);
        let occupied = plan_ranks(
            &cores,
            &config,
            &LoadOptions {
                memory_limit: Some(MemoryLimit::Bytes(12 * GIB)),
                ..Default::default()
            },
            4,
            &devices,
            Default::default(),
        )
        .unwrap();
        let loaded = base.ranks[0].fixed_device_bytes + base.ranks[0].runtime.bytes;
        assert_eq!(
            occupied.ranks[0].cache_budget,
            12 * GIB - loaded - memory::HEADROOM_BYTES
        );
        assert!(
            plan_ranks(
                &cores,
                &config,
                &LoadOptions {
                    memory_limit: Some(MemoryLimit::Bytes(10 * GIB)),
                    ..Default::default()
                },
                4,
                &devices,
                Default::default(),
            )
            .is_err()
        );
        let capped = plan_ranks(
            &cores,
            &config,
            &LoadOptions {
                kv_cache_limit: KvCacheLimit::Context,
                ..Default::default()
            },
            4,
            &devices,
            Default::default(),
        )
        .unwrap();
        // One prepared context of 64,000 tokens is 1,000 pages of 64 tokens.
        assert_eq!((capped.pages, capped.limiting.as_str()), (1000, "context"));
        let warm = plan_ranks(
            &cores,
            &config,
            &LoadOptions {
                prefix_cache: crate::offload::PrefixCacheOffload {
                    memory_bytes: Some(4 * GIB),
                    ..Default::default()
                },
                ..Default::default()
            },
            4,
            &devices,
            Default::default(),
        )
        .unwrap();
        assert!(warm.ranks[0].prefix_cache_device_bytes > 0);
        assert_eq!(
            warm.ranks[0].cache_budget,
            base.ranks[0].cache_budget - warm.ranks[0].prefix_cache_device_bytes
        );
        assert!(warm.pages < base.pages);
        assert_eq!(warm.host.prefix_cache_memory_bytes, 4 * GIB);
        assert!(warm.host.prefix_cache_host_bytes > 4 * GIB);
        assert!(
            plan_ranks(
                &cores,
                &config,
                &LoadOptions::default(),
                4,
                &[],
                Default::default()
            )
            .is_err()
        );
    }

    /// The default DeepSeek-V4.1-Flash tensor-parallel core of this host, when present, loaded
    /// with the serving flags of `serve-best.sh`; the asserted capacities are the "Allocate cache"
    /// lines of its recorded loads.
    /// Recorded loads on this host, each identified by its core directory under the shared
    /// model cache, its `--prefix-cache-memory` and `--mtp-max-verification-token-budget`, and
    /// the "Allocate cache" line of its serve log: the default DeepSeek-V4.1-Flash
    /// tensor-parallel-3 core (`serve-best.log`, `serve-v3-2x.log`, `serve-prefix.log` in the
    /// repository root) and two truncated test cores (`serve-mini-v3m8.log`,
    /// `serve-mini-v3m8fused.log`). The three GPUs are RTX PRO 6000 Blackwell with a driver total of
    /// 102,069,305,344 bytes each.
    /// Absent cores or an absent development publisher profile skip the case or the test.
    #[test]
    fn actual_dry_run_reproduces_recorded_kv_capacities() {
        let models = std::path::Path::new("~/.cache/flare/cores/models");
        let profile = std::env::var_os("HOME").map(|home| {
            std::path::PathBuf::from(home).join(".config/flare/development-publisher.json")
        });
        let Some(profile) = profile.filter(|p| p.is_file()) else {
            eprintln!("skipping: development publisher profile is absent");
            return;
        };
        let publishers: Vec<super::super::Publisher> =
            serde_json::from_slice(&std::fs::read(profile).unwrap()).unwrap();
        // The CLI reads each device's total through the CUDA driver; without a usable driver the
        // test assumes the calibration hosts' driver total, which nvidia-smi shows as 97,887 MiB.
        let totals = device_total_memory(&[0, 1, 2]).unwrap_or_else(|error| {
            eprintln!("assuming the calibration hosts' device total: {error}");
            vec![102_069_305_344; 3]
        });
        let devices: Vec<_> = totals
            .iter()
            .enumerate()
            .map(|(label, &total)| PlannedDevice {
                label: label as i32,
                total_bytes: total,
                free_bytes: total - IDLE_DEVICE_RESERVE_BYTES,
            })
            .collect();
        let mut failures = Vec::new();
        for (core, prefix_cache_memory, budget, expected_tokens, expected_snapshots) in [
            (
                "ds41-1017-engram-mxfp4/gb202-tp3-202611070243",
                32u64 << 30,
                24usize,
                2_461_184u64,
                233u64,
            ),
            (
                "ds41-1017-engram-mxfp4/gb202-tp3-202611070243",
                16 << 30,
                128,
                2_373_760,
                225,
            ),
            (
                "ds41-1017-engram-mxfp4/gb202-tp3-202611070243",
                16 << 30,
                24,
                2_467_200,
                234,
            ),
            (
                "ds41-mini8/gb202-tp3-202611070111",
                16 << 30,
                128,
                414_056_448,
                76_314,
            ),
            (
                "ds41-mini8/gb202-tp3-202611072222",
                32 << 30,
                24,
                414_260_224,
                76_352,
            ),
        ] {
            let core = models.join(core);
            if !core.is_dir() {
                eprintln!("skipping absent core {}", core.display());
                continue;
            }
            let options = LoadOptions {
                mtp_draft_tokens: Some(5),
                adaptive_verification: Some(true),
                mtp_max_verification_token_budget: Some(budget),
                embedding_table_storage_per_table: Some(
                    super::super::EmbeddingTableStorageSpec::parse("0").unwrap(),
                ),
                prefix_cache: crate::offload::PrefixCacheOffload {
                    memory_bytes: Some(prefix_cache_memory),
                    ..Default::default()
                },
                ..Default::default()
            };
            let authorized = match super::super::ModelCore::open(
                &core,
                &publishers,
                crate::core::protection::Selection::Deployment {
                    deployment: 0,
                    rank: None,
                },
                "local-development",
                false,
            )
            .unwrap()
            {
                super::super::ModelCore::Legacy(authorized) => authorized,
                super::super::ModelCore::Payload(payload) => AuthorizedCore {
                    kind: super::super::protection::AuthorizedKind::Payload(payload),
                },
            };
            let plan = plan_memory(authorized, &devices, &options).unwrap();
            for rank in &plan.ranks {
                eprintln!(
                    "DATA rank {} fixed {} modules {} module_bytes {} globals {} nodes {} estimate {} budget {} cache {} offload {}",
                    rank.device.label,
                    rank.fixed_device_bytes,
                    rank.runtime.modules,
                    rank.runtime.module_bytes,
                    rank.runtime.global_bytes,
                    rank.runtime.graph_nodes,
                    rank.runtime.bytes,
                    rank.cache_budget,
                    rank.cache_bytes,
                    rank.prefix_cache_device_bytes,
                );
            }
            eprintln!(
                "DATA plan {} prefix {prefix_cache_memory} budget {budget}: tokens {} snapshots {} pages {} rows {} execution {} cache {} expected {expected_tokens} {expected_snapshots}",
                core.display(),
                plan.kv_tokens,
                plan.snapshots,
                plan.pages,
                plan.rows,
                plan.execution_rows,
                plan.cache_bytes
            );
            eprintln!("DATA host {:?}", plan.host);
            if (plan.kv_tokens, plan.snapshots) != (expected_tokens, expected_snapshots) {
                failures.push(format!(
                    "{} at {prefix_cache_memory} bytes prefix cache, verification budget {budget}: planned {} tokens and {} snapshots, recorded {expected_tokens} and {expected_snapshots}",
                    core.display(), plan.kv_tokens, plan.snapshots
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! `flare run --dry-run`: the memory plan and KV capacity of a serving configuration, computed
//! from package metadata and assumed device memory without a CUDA context.

use std::fmt::Write as _;

use flare_helios::runtime_package::{
    AuthorizedPackage,
    plan::{
        IDLE_DEVICE_RESERVE_BYTES, MemoryPlan, PlannedDevice, device_total_memory, plan_memory,
    },
};

use crate::inspect::{bytes, thousands};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Plan `authorized` for the GPUs `args` selected and print the report.
pub fn run(args: &crate::cli::Run, authorized: AuthorizedPackage) -> Result<()> {
    let labels = &args.gpus;
    let ordinals: Vec<i32> = (0..labels.len() as i32).collect();
    let totals = device_total_memory(&ordinals)?;
    let assumed = match args.dry_run_free_memory.as_slice() {
        [] => None,
        [one] => Some(vec![*one; labels.len()]),
        many if many.len() == labels.len() => Some(many.to_vec()),
        many => {
            return Err(format!(
                "--dry-run-free-memory names {} sizes for {} selected GPUs; give one size or one per GPU",
                many.len(),
                labels.len()
            )
            .into());
        }
    };
    let devices: Vec<PlannedDevice> = labels
        .iter()
        .zip(&totals)
        .enumerate()
        .map(|(index, (&label, &total_bytes))| PlannedDevice {
            label,
            total_bytes,
            free_bytes: match &assumed {
                Some(free) => free[index],
                None => total_bytes.saturating_sub(IDLE_DEVICE_RESERVE_BYTES),
            },
        })
        .collect();
    let plan = plan_memory(authorized, &devices, &args.load_options())?;
    print!("{}", render(&plan, assumed.is_none()));
    Ok(())
}

/// The plan as aligned text: one block per rank, the cache the loader would announce, and the
/// host memory the load commits.
pub fn render(plan: &MemoryPlan, default_free: bool) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Memory dry run: no CUDA context was created, no GPU memory was allocated, and no weights were read."
    );
    if default_free {
        let _ = writeln!(
            out,
            "Assumed free memory per GPU: the CUDA driver's device total minus {} for the CUDA \
             context, driver, and display; --dry-run-free-memory overrides it.",
            bytes(IDLE_DEVICE_RESERVE_BYTES)
        );
    } else {
        let _ = writeln!(out, "Assumed free memory per GPU: --dry-run-free-memory.");
    }
    let _ = writeln!(
        out,
        "Kernel and graph memory is an estimate from the selected module and graph node counts; \
         fixed allocations, cache geometry, headroom, and the prefix cache's device share are exact."
    );
    let width = 70;
    let row = |out: &mut String, label: &str, value: String| {
        let _ = writeln!(out, "  {label:<width$}{value:>14}");
    };
    for rank in &plan.ranks {
        let _ = writeln!(
            out,
            "\nGPU {} · {} total · {} assumed free",
            rank.device.label,
            bytes(rank.device.total_bytes),
            bytes(rank.device.free_bytes)
        );
        row(
            &mut out,
            "weights, execution state, and scratch (fixed device allocations)",
            bytes(rank.fixed_device_bytes),
        );
        row(
            &mut out,
            &format!(
                "kernels and graphs (estimate: {} modules, {} graph nodes)",
                thousands(rank.runtime.modules as u64),
                thousands(rank.runtime.graph_nodes)
            ),
            bytes(rank.runtime.bytes),
        );
        row(
            &mut out,
            "headroom left free (reserve and engine)",
            bytes(rank.headroom_bytes),
        );
        if rank.prefix_cache_device_bytes != 0 {
            row(
                &mut out,
                "host prefix cache transfer streams and intake",
                bytes(rank.prefix_cache_device_bytes),
            );
        }
        row(&mut out, "cache budget", bytes(rank.cache_budget));
        row(
            &mut out,
            &format!(
                "cache planned ({} execution rows + {} KV pages)",
                thousands(plan.execution_rows),
                thousands(plan.pages)
            ),
            bytes(rank.cache_bytes),
        );
        if rank.mixed_host_bytes != 0 {
            row(
                &mut out,
                "mixed expert banks, host part",
                bytes(rank.mixed_host_bytes),
            );
        }
        if rank.mapped_host_bytes != 0 {
            row(
                &mut out,
                "mapped host allocations",
                bytes(rank.mapped_host_bytes),
            );
        }
    }
    let _ = writeln!(
        out,
        "\nAllocate cache: {:.1} GiB · {} KV tokens · {} shared snapshot slots · limited by {}",
        plan.cache_bytes as f64 / (1u64 << 30) as f64,
        thousands(plan.kv_tokens),
        thousands(plan.snapshots),
        plan.limiting
    );
    let mut settings = format!(
        "  concurrency {} · {} continuation rows per request · {} tokens per page · KV cache limit {}",
        plan.active, plan.width, plan.page_tokens, plan.kv_cache_limit
    );
    if let Some(limit) = plan.memory_limit {
        let _ = write!(settings, " · memory limit {limit}");
    }
    let _ = writeln!(out, "{settings}");
    let host = &plan.host;
    let _ = writeln!(out, "\nHost RAM committed by the load");
    row(
        &mut out,
        "mixed expert banks, host part, all ranks",
        bytes(host.mixed_host_bytes),
    );
    row(
        &mut out,
        "mapped host allocations, all ranks",
        bytes(host.mapped_host_bytes),
    );
    row(
        &mut out,
        "embedding table planes resident in RAM",
        bytes(host.engram_resident_bytes),
    );
    row(
        &mut out,
        "embedding gather staging",
        bytes(host.engram_staging_bytes),
    );
    row(
        &mut out,
        "embedding row cache",
        bytes(host.embedding_cache_bytes),
    );
    row(
        &mut out,
        &format!(
            "host prefix cache ({} configured plus intake, input staging, and index)",
            bytes(host.prefix_cache_memory_bytes)
        ),
        bytes(host.prefix_cache_host_bytes),
    );
    if host.input_window_bytes != 0 {
        row(
            &mut out,
            "prepared image inputs",
            bytes(host.input_window_bytes),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use flare_helios::runtime_package::{
        KvCacheLimit, MemoryLimit,
        plan::{HostPlan, RankPlan, RuntimeEstimate},
    };

    #[test]
    fn report_states_assumptions_and_the_cache_line_the_loader_would_print() {
        let plan = MemoryPlan {
            ranks: vec![RankPlan {
                device: PlannedDevice {
                    label: 2,
                    total_bytes: 96 << 30,
                    free_bytes: (96 << 30) - IDLE_DEVICE_RESERVE_BYTES,
                },
                fixed_device_bytes: 91 << 30,
                mixed_host_bytes: 3 << 30,
                mapped_host_bytes: 2 << 30,
                stored_bytes: 94 << 30,
                runtime: RuntimeEstimate {
                    modules: 6138,
                    module_bytes: 31 << 20,
                    global_bytes: 700 << 10,
                    graphs: 86,
                    graph_nodes: 91448,
                    bytes: 531 << 20,
                },
                headroom_bytes: 64 << 20,
                prefix_cache_device_bytes: 34 << 20,
                cache_budget: 2 << 30,
                cache_bytes: (2 << 30) - (1 << 20),
            }],
            host: HostPlan {
                mixed_host_bytes: 3 << 30,
                mapped_host_bytes: 2 << 30,
                engram_resident_bytes: 97 << 30,
                engram_staging_bytes: 51 << 20,
                embedding_cache_bytes: 0,
                prefix_cache_memory_bytes: 32 << 30,
                prefix_cache_host_bytes: (32 << 30) + (200 << 20),
                input_window_bytes: 20 << 20,
            },
            active: 8,
            width: 6,
            execution_rows: 49,
            rows: 282,
            snapshots: 233,
            pages: 19228,
            page_tokens: 128,
            kv_tokens: 2_461_184,
            limiting: "memory".into(),
            cache_bytes: 7_027_556_352,
            kv_cache_limit: KvCacheLimit::Memory,
            memory_limit: Some(MemoryLimit::Reserve(1 << 30)),
        };
        let report = render(&plan, true);
        assert!(report.contains(
            "Assumed free memory per GPU: the CUDA driver's device total minus 606.0 MiB"
        ));
        assert!(report.contains("GPU 2 · 96.00 GiB total · 95.41 GiB assumed free"));
        assert!(report.contains("estimate: 6,138 modules, 91,448 graph nodes"));
        assert!(report.contains("cache planned (49 execution rows + 19,228 KV pages)"));
        assert!(report.contains(
            "Allocate cache: 6.5 GiB · 2,461,184 KV tokens · 233 shared snapshot slots · limited by memory"
        ));
        assert!(report.contains("KV cache limit memory · memory limit -1GiB"));
        assert!(report.contains(
            "host prefix cache (32.00 GiB configured plus intake, input staging, and index)"
        ));
        assert!(report.contains("prepared image inputs"));
        let explicit = render(&plan, false);
        assert!(explicit.contains("Assumed free memory per GPU: --dry-run-free-memory."));
        assert!(!explicit.contains("device total minus"));
    }
}

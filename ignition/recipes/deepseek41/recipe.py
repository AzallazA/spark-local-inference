# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4.1 Flash checkpoint and tensor-parallel memory geometry."""

from types import MappingProxyType

from ignition.planning import HEADROOM_BYTES
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange, Speculation
from .tp3_config import TP3Config


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    # TP3 fork: the FLARE_DS41_* environment is read once here and travels with the preparation config.
    tp3 = TP3Config.from_environment()
    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_token_capacity": plan.speculative_token_capacity,
        "speculative_graph_budget": speculation.graph_budget,
        "speculative_fixed_width": speculation.fixed_width, "packed_verification": speculation.packed_verification,
        "draft_prefix_selection": speculation.draft_prefix_selection,
        "draft_head_quantization": speculation.draft_head_quantization,
        "markov_head_quantization": speculation.markov_head_quantization,
        "vocabulary_parallel_sampling": speculation.vocabulary_parallel_sampling}
    config = PreparationConfig(mtp_only=plan.speculation is not None and plan.speculation.mtp_only,
        collective_tuning=plan.collective_tuning,
        collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        joint_tuning=plan.joint_tuning,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        image_concurrency=plan.image_concurrency,
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts, roce_ring=plan.roce_ring, roce_chain=plan.roce_chain, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, max_active=plan.concurrency, context=plan.context,
        prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache,
        dspark=speculation is not None,
        stream_overlap=plan.stream_overlap, expert_scale_compression=plan.expert_scale_compression,
        mla_compute_mode=plan.mla_compute_mode, tp3=tp3, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


def _parked_bytes(world: int, tp3: TP3Config | None = None) -> int:
    """Routed-expert bytes per rank parked in host memory (TP3Config host_expert_layers / hot_experts / global_cold).

    The planner calls this (``host_offload_bytes``) only once this recipe is the chosen one, right before
    ``prepare`` would build the same config, so the full validated construction is appropriate here."""
    from .weights import global_cold_plan, host_expert_layers
    tp3 = TP3Config.from_environment() if tp3 is None else tp3
    local = -(-2304 // world)  # routed-expert intermediate width per rank
    per_expert = 2 * local * 5120 // 2 + 5120 * local // 2           # MXFP4 w13 + w2 bytes (scales stay on GPU)
    plan = global_cold_plan(tp3)
    if plan is not None:
        return sum(len(cold) for cold in plan.values()) * per_expert
    layers = host_expert_layers(tp3)
    if not layers:
        return 0
    cold = 384 - min(tp3.hot_experts, 384)
    return len(layers) * cold * per_expert


def _graph_bytes(world: int, tp3: TP3Config | None = None) -> int:
    """Per-rank captured-graph estimate; TP3Config.tp3_graph_gib overrides it for world size 3 (96 GB GPUs).

    Evaluated at import time for the recipe declaration (every ignition command imports every recipe), so without a
    config it peeks at the one variable it needs and never validates the rest of the environment."""
    gib = TP3Config.graph_gib_from_environment() if tp3 is None else tp3.tp3_graph_gib
    if world == 3 and gib is not None:
        return int(gib * (1 << 30))
    return 8 << 30


RECIPE = Recipe(
    name="deepseek41", served_model_name="DeepSeek-V4.1-Flash", prepare=prepare, usage=usage,
    mla_compute_modes=("auto", "bf16"), mla_compute_mode="bf16", expert_scale_compression="csf",
    draft_head_quantizations=("nvfp4", "mxfp8"), markov_head_quantizations=("nvfp4", "fp8"),
    # Recommended sampling, including top-p 1.0: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference.
    # Repetition-penalty bounds enable measured use over prompt + output, matching vLLM/HF semantics.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1,
        ranges={"repetition_penalty": SamplingRange(1., 1.1)}),
    checkpoints=(Checkpoint(alias="deepseek-v4.1-flash", repository="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        revision="1bae0e1088993b9b82de08f59de5be5a34e42170", architecture="DeepseekV41ForCausalLM",
        stored_bytes=510_286_023_000, device_bytes=306_557_453_640, draft_bytes=7_932_874_632,
        # Estimate half of backbone scale bytes saved; retain the full draft budget.
        scale_compressed_device_bytes={"csf": 298_063_988_040}),),
    context=1048576, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True,
                            draft_prefix_selection="gpu", graph_budget=64),
    max_draft_tokens=7, max_images=4, image_bytes=32 << 20,
    # Single-row decode runs the shared expert on an auxiliary stream.
    stream_overlap=True,
    host_offload_bytes=_parked_bytes,
    # TP3 on 96 GB SM120: the shared 8 GiB graph estimate leaves no room for 92.5 GiB of TP3 weights;
    # TP3Config.tp3_graph_gib (FLARE_DS41_TP3_GRAPH_GIB) overrides the TP3 graph estimate (measured value goes here once known).
    ranks=MappingProxyType({world: RankCosts(3_043_840, 113_920, _graph_bytes(world)) for world in (3, 4)}),
    plain_ranks=MappingProxyType({world: RankCosts(2_841_088, 113_920, _graph_bytes(world)) for world in (3, 4)}),
)

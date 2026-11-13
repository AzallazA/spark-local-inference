# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualified checkpoint, serving range, and memory costs of the GLM 5.3 recipe."""

import os
from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange, Speculation

# Each sparse-attention layer stores four latent blocks and one index block per
# 256-token page on every rank. The target has eleven such layers; learned drafts
# add the draft layer's.
_PAGE_LAYER_BYTES = 4 * 64 * 528 + 64 * 132
_TARGET_PAGE_LAYERS = 11
# Rough estimate of what a rank holds beyond stored weights, state rows, and
# pages: operator scratch, graphs, collectives, and the CUDA context. It serves
# every rank count and speculation setting.
_GRAPH_BYTES = 6656 << 20
# State rows hold one rank's KDA recurrent and convolution state for its heads plus the replicated
# index tails, partial pages and hidden output: `continuation_layout(..., world).row_bytes` in pool.py,
# which tests/recipes/test_glm53_shards.py checks against every declared rank count.


def _tp3_override(name, choices, default):
    """Fork-only export override FLARE_GLM53_<name> (TP3 experiments), read in the recipe shim like DeepSeek's TP3Config.

    Empty or unset keeps upstream's choice. a4 experts on SM120 and resident expanded scales at three ranks are
    measured against the defaults before either becomes one (plan follow-ups; vLLM Kraken used both).
    EXPERT_COLUMNS=704 narrows the routed experts' three-rank shard from 768 to 704 columns (shards.py; the
    shared expert stays at 768 for its 128 x 128 block-FP8 scales); it applies at three ranks only.
    """
    value = os.environ.get(f"FLARE_GLM53_{name}", "").strip().lower()
    if not value:
        return default
    if value not in choices:
        raise ValueError(f"FLARE_GLM53_{name}={value!r}: expected one of {', '.join(choices)}")
    return value


def _expert_columns(world_size):
    """FLARE_GLM53_EXPERT_COLUMNS=768|704 at three ranks; unset keeps the upstream 128-aligned shard (None)."""
    value = _tp3_override("EXPERT_COLUMNS", ("768", "704"), "")
    if not value:
        return None
    if world_size != 3:
        raise ValueError(f"FLARE_GLM53_EXPERT_COLUMNS={value} applies to three ranks only, not {world_size}")
    return int(value)


def _context_parallel_three_ranks():
    """FLARE_GLM53_DCP3=1|0 (fork-only, default 0): declare a context-parallel page size at three ranks.

    Decode context parallelism over three ranks (DCP3) puts 256 local records of every page on each rank, so a
    page covers 768 positions and KV tokens triple at constant page bytes. The planner turns context
    parallelism on whenever the rank count declares a page size (planning.py), so the declaration is opt-in
    until DCP3 is qualified: unset keeps three ranks at 256-token pages with heads and experts sharded only.
    Read at import, like the other export overrides; export with `DCP=1 tp3-tools/glm53-export-tp3.sh`.
    """
    return _tp3_override("DCP3", ("1", "0"), "0") == "1"


def _costs(state_row_bytes, graph_bytes, *, ranks, speculating, context_parallel_three=None):
    layers = _TARGET_PAGE_LAYERS + int(speculating)
    # Decode context parallelism shards each page's latent records over the ranks: 256 * ranks positions per
    # page. Three ranks declare it only with FLARE_GLM53_DCP3=1 (768-token pages; 2^20 is not a multiple of
    # 768, so the bootstrap and planner round page counts up); otherwise the planner keeps every page on every rank.
    if context_parallel_three is None:
        context_parallel_three = _context_parallel_three_ranks()
    return RankCosts(state_row_bytes, layers * _PAGE_LAYER_BYTES, graph_bytes,
                     page_tokens=None if ranks == 3 and not context_parallel_three else 256 * ranks)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    """Prepare the plan's catalog and speculation settings on process-local GPUs 0 through N-1.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    KV pages fill the free memory measured after graph capture, not the plan's estimate.
    """
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_token_capacity": plan.speculative_token_capacity, "speculative_fixed_width": speculation.fixed_width}
    config = PreparationConfig(mtp_only=plan.speculation is not None and plan.speculation.mtp_only,
        collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts, compile_decoder=compile_forwards, retain_graphs=retain_graphs,
        image_concurrency=plan.image_concurrency, language_model_only=plan.language_model_only,
        max_active=plan.concurrency, roce_ring=plan.roce_ring, roce_chain=plan.roce_chain, context_parallel=plan.context_parallel, stream_overlap=plan.stream_overlap,
        context=plan.recipe.context, prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, kernel_cache=kernel_cache,
        # SM103 runs routed experts through trtllm-gen with NVFP4 activations at every size.
        expert_activation=_tp3_override("EXPERT_ACTIVATION", ("a4", "a16"),
            "a4" if any(gpu.capability == (10, 3) for gpu in plan.gpus) else "a16"),
        # Expanded scales take 2 x 288 x 4096 x 2048 / 16 bytes per routed layer over the ranks: about
        # 4.9 GB per rank at TP4, which GB10 holds beside its weights, but 9.7 GB at TP2, which it does not.
        expand_expert_scales=_tp3_override("EXPAND_SCALES", ("1", "0"), "1" if plan.world_size >= 4 else "0") == "1",
        expert_columns=_expert_columns(plan.world_size),
        # Fork-only decode-speed experiments (GLM speed step, 2026-11-09), unset = upstream behaviour:
        # KDA_SIDE_CHAIN=1 runs every small KDA projection (f_a -> f_b, b, g_a -> g_b) beside the q/k/v GEMM
        # and drops the two per-layer staging copies; DRAFT_HEAD=nvfp4 proposes with an NVFP4 copy of the
        # MXFP8 head; DRAFT_EH_PROJ=mxfp8 runs the draft layer's BF16 eh_proj as MXFP8.
        kda_side_chain=_tp3_override("KDA_SIDE_CHAIN", ("1", "0"), "0") == "1",
        draft_head=_tp3_override("DRAFT_HEAD", ("target", "nvfp4"), "target"),
        draft_eh_proj=_tp3_override("DRAFT_EH_PROJ", ("bf16", "mxfp8"), "bf16"),
        # Fork-only VRAM switch (2026-11-10): EMBED_HOST=1 parks the replicated BF16 token embedding
        # (1210 MiB per rank at every rank count) in device-visible host memory; KV pages take the VRAM.
        embed_host=_tp3_override("EMBED_HOST", ("1", "0"), "0") == "1",
        # Fork-only VRAM switch (2026-11-12, DeepSeek's TP3 vision weight staging): VISION_STAGE=1 parks the
        # replicated vision tower in device-visible host memory and stages it through VRAM slots per image.
        # Only meaningful with images (export without --language-model-only).
        vision_stage=_tp3_override("VISION_STAGE", ("1", "0"), "0") == "1",
        mtp=speculation is not None, **drafting)
    if plan.world_size == 1:
        return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, core_storage=core_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="glm53", served_model_name="GLM-5.3-Flash", prepare=prepare, usage=usage,
    # Bounds span HLE, BabyVision, and coding presets: https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1,
        ranges={"temperature": SamplingRange(.95, 1.), "top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(
        alias="glm-5.3-flash", repository="nvidia/GLM-5.3-Flash-FP4-Local",
        revision="76348ddb2be2e35e39ecfc00e75dc07703f4f3d5", architecture="Glm5NextForConditionalGeneration",
        stored_bytes=177_421_592_008,
        # Language model, draft layer, output head, and replicated vision tower.
        device_bytes=177_421_592_008,
        # Decoder layer 45, the MTP module.
        draft_bytes=4_301_658_112),),
    context=1048576, page_tokens=256, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5, token_capacity=128), max_draft_tokens=7, verification_rows=0,
    # Shared experts, KDA gate projections, and decode query chains run on an auxiliary stream.
    stream_overlap=True, max_images=4, image_bytes=32 << 20, language_model_only=True,
    # Three ranks pad heads, expert columns and the vocabulary (shards.py); a four-GPU host keeps the exact TP4.
    exact_rank_counts=frozenset({3}),
    ranks=MappingProxyType({
        # DGX Station can hold GLM on one GPU; TP1 must remain a supported rank count.
        # One rank holds every head: twice TP2's recurrent rows less the replicated part.
        1: _costs(161_267_200, _GRAPH_BYTES, ranks=1, speculating=True),
        2: _costs(81_608_192, _GRAPH_BYTES, ranks=2, speculating=True),
        # Three ranks hold 24 heads each (the last 16 real + 8 zero; shards.py), so every rank's rows cover 24 heads.
        3: _costs(61_693_440, _GRAPH_BYTES, ranks=3, speculating=True),
        4: _costs(41_778_688, _GRAPH_BYTES, ranks=4, speculating=True),
    }),
    plain_ranks=MappingProxyType({
        1: _costs(149_230_592, _GRAPH_BYTES, ranks=1, speculating=False),
        2: _costs(75_420_672, _GRAPH_BYTES, ranks=2, speculating=False),
        3: _costs(56_968_192, _GRAPH_BYTES, ranks=3, speculating=False),
        4: _costs(38_515_712, _GRAPH_BYTES, ranks=4, speculating=False),
    }),
)

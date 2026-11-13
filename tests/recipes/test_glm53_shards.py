# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Rank-count-derived GLM shard widths: exact for one, two and four ranks, padded for three."""

import os

import pytest

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.glm53.decoder import LayerScratch
from ignition.recipes.glm53.kda import KDAScratch
from ignition.recipes.glm53.mla import MLAScratch
from ignition.recipes.glm53.moe import MoeScratch
from ignition.recipes.glm53.mtp import MTPScratch
from ignition.recipes.glm53.pool import continuation_layout
from ignition.recipes.glm53.recipe import RECIPE
from ignition.recipes.glm53.shards import (EXPERT_WIDTHS, HEAD_COUNTS, ROUTED_WIDTHS, WORLDS, Shards, padded_width,
                                           routed_expert_columns)
from ignition.recipes.glm53.stack import DecoderScratch
from ignition.recipes.glm53.state import DecoderState

LAYER_TYPES = tuple("indexed_attention" if index % 4 == 3 else "linear_attention" for index in range(45))


@pytest.mark.parametrize("world", [1, 2, 4])
def test_dividing_rank_counts_keep_the_exact_partitions(world):
    shards = Shards.of(world)
    assert shards == Shards(world, 64 // world, 2048 // world, 12288 // world, 154880 // world)
    assert shards.padded_heads == 64 and shards.padded_vocabulary == 154880
    assert shards.kda_channels == 8192 // world
    shard = TensorParallelShard(world - 1, world)
    for rank in range(world):
        exact = TensorParallelShard(rank, world)
        assert shards.head_range(rank) == exact.partition(64)
        for width in (128, 256, 512):
            assert shards.head_channels(rank, width) == exact.partition(64 * width)
    for extent, alignment in ((64 * 256, 256), (64 * 512, 512), (8192, 128), (64, 1), (12288, 128), (2048, 128),
                              (154880, 64)):
        assert shard.padded_partition(extent, alignment=alignment) == shard.partition(extent, alignment=alignment)


def test_three_ranks_pad_heads_experts_and_vocabulary():
    shards = Shards.of(3)
    assert shards == Shards(3, 24, 768, 4096, 51648)
    assert shards.padded_heads == 72 and shards.kda_channels == 3072
    # Exact checkpoint heads 24/24/16; the last rank's 8 extra heads are zero padding.
    assert [shards.head_range(rank) for rank in range(3)] == [(0, 24), (24, 48), (48, 72)]
    assert [shards.checkpoint_heads(rank) for rank in range(3)] == [24, 24, 16]
    assert [shards.head_channels(rank, 128) for rank in range(3)] == [(0, 3072), (3072, 6144), (6144, 9216)]
    assert shards.padded_vocabulary == 154944 and shards.padded_vocabulary % (3 * 64) == 0
    assert shards.expert_width % 128 == 0 and shards.vocabulary % 64 == 0
    assert TensorParallelShard(2, 3).padded_partition(2048, alignment=128) == (1536, 2304)
    with pytest.raises(ValueError):
        shards.head_range(3)


def test_three_ranks_can_narrow_the_routed_experts_to_704_columns():
    """FLARE_GLM53_EXPERT_COLUMNS=704: routed 704/704/640+64 zero, shared expert unchanged at 768 (block-FP8 128)."""
    assert routed_expert_columns(3) == {768, 704}
    assert all(routed_expert_columns(world) == {2048 // world} for world in (1, 2, 4))
    default, narrow = Shards.of(3), Shards.of(3, expert_columns=704)
    assert default.routed_width == default.expert_width == 768 and Shards.of(3, expert_columns=768) == default
    assert narrow == Shards(3, 24, 768, 4096, 51648, 704)
    assert [narrow.routed_columns(rank) for rank in range(3)] == [(0, 704), (704, 1408), (1408, 2112)]
    assert [default.routed_columns(rank) for rank in range(3)] == [
        TensorParallelShard(rank, 3).padded_partition(2048, alignment=128) for rank in range(3)]
    # 704 is the smallest 64-aligned (and 32-aligned) width whose three shards cover 2048.
    assert padded_width(2048, 3, 64) == padded_width(2048, 3, 32) == 704 and 3 * 640 < 2048
    assert all(width % 64 == 0 and (2 * width) % 128 == 0 for width in (704, 768))
    for world, columns in ((3, 640), (3, 736), (3, 2048), (2, 704), (4, 768), (1, 1024)):
        with pytest.raises(ValueError):
            Shards.of(world, expert_columns=columns)
    with pytest.raises(ValueError):
        narrow.routed_columns(3)
    # Shared-expert scratch keeps the 128-aligned width whatever the routed width.
    assert MoeScratch.allocate(5, 3, "meta").shared.intermediate.shape == (5, 768)


def test_supported_widths_and_invalid_rank_counts():
    assert WORLDS == (1, 2, 3, 4)
    assert HEAD_COUNTS == {64, 32, 24, 16}  # every count the sparse-MLA kernel test passes
    assert EXPERT_WIDTHS == {2048, 1024, 768, 512}
    assert ROUTED_WIDTHS == {2048, 1024, 768, 704, 512}
    assert padded_width(64, 3, 8) == 24 and padded_width(2048, 3, 128) == 768
    for world in (0, 5, 8, -1, 2.0):
        with pytest.raises(ValueError):
            Shards.of(world)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_scratch_and_state_shapes_are_unchanged_for_dividing_rank_counts(world):
    heads, local, dense, vocabulary = 64 // world, 2048 // world, 12288 // world, 154880 // world
    kda = KDAScratch.allocate(5, Shards.of(world).heads, "meta")
    assert kda.projected_input.shape == (5, 3 * heads * 128 + heads + 128)
    mla = MLAScratch.allocate(5, Shards.of(world).heads, 2, "decode", "meta")
    assert mla.query.shape == (5, heads, 256) and mla.absorbed_query.shape == (5, heads, 512)
    moe = MoeScratch.allocate(5, world, "meta")
    assert moe.shared.intermediate.shape == (5, local) and moe.shared.gate_up.shape == (5, 2 * local)
    layer = LayerScratch.allocate(5, world, "meta", sparse=False, block_type="linear_attention", mode="decode")
    assert layer.mlp.intermediate.shape == (5, dense) and layer.attention.recurrence.shape == (5, heads * 128)
    stack = DecoderScratch.allocate(5, 2, "decode", "meta", shard=TensorParallelShard(0, world), sample_rows=3)
    assert stack.local_logits.shape == (3, vocabulary) and stack.logits.shape == (3, 154880)
    assert stack.dense.intermediate.shape == (5, dense) and stack.kda.z.shape == (5, heads * 128)
    draft = MTPScratch.allocate(5, world, 2, "decode", "meta")
    assert draft.attention.query.shape == (5, heads, 256) and draft.moe.shared.intermediate.shape == (5, local)
    state = DecoderState.allocate(LAYER_TYPES[:4], world, 3, 1, "meta")
    assert state.layers[0].recurrent.shape == (3, heads, 128, 128)
    assert state.layers[0].convolution.shape == (3, 3 * heads * 128, 3)


def test_three_rank_scratch_and_state_use_padded_heads():
    stack = DecoderScratch.allocate(5, 2, "decode", "meta", shard=TensorParallelShard(1, 3), sample_rows=3)
    assert stack.kda.projected_input.shape == (5, 3 * 24 * 128 + 24 + 128)
    assert stack.mla.query.shape == (5, 24, 256) and stack.moe.shared.intermediate.shape == (5, 768)
    assert stack.dense.intermediate.shape == (5, 4096) and stack.local_logits.shape == (3, 51648)
    state = DecoderState.allocate(LAYER_TYPES[:4], 3, 3, 1, "meta")
    assert state.layers[0].recurrent.shape == (3, 24, 128, 128)
    with pytest.raises(ValueError):
        KDAScratch.allocate(5, 22, "meta")
    with pytest.raises(ValueError):
        MLAScratch.allocate(5, 22, 2, "decode", "meta")  # 22 is the failing kernel head count


@pytest.mark.parametrize("world", WORLDS)
def test_declared_state_row_bytes_match_the_continuation_layout(world):
    assert RECIPE.ranks[world].state_row_bytes == continuation_layout(LAYER_TYPES, world, mtp=True).row_bytes
    assert RECIPE.plain_ranks[world].state_row_bytes == continuation_layout(LAYER_TYPES, world, mtp=False).row_bytes
    # Three ranks declare no context-parallel page size unless FLARE_GLM53_DCP3=1 (opt-in DCP3, 768-token
    # pages), so by default the planner keeps it off there.
    dcp3 = os.environ.get("FLARE_GLM53_DCP3", "").strip() == "1"
    assert RECIPE.ranks[world].page_tokens == (None if world == 3 and not dcp3 else 256 * world)
    assert RECIPE.ranks[world].page_bytes == 12 * (4 * 64 * 528 + 64 * 132)
    assert RECIPE.plain_ranks[world].page_bytes == 11 * (4 * 64 * 528 + 64 * 132)


def test_three_rank_state_rows_hold_twenty_four_heads():
    per_head = 34 * (128 * 128 * 4 + 3 * 128 * 2 * 10)
    replicated = RECIPE.ranks[1].state_row_bytes - 64 * per_head
    assert RECIPE.ranks[2].state_row_bytes == 32 * per_head + replicated
    assert RECIPE.ranks[4].state_row_bytes == 16 * per_head + replicated
    assert RECIPE.ranks[3].state_row_bytes == 24 * per_head + replicated == 61_693_440
    assert RECIPE.plain_ranks[3].state_row_bytes == 56_968_192
    assert RECIPE.tensor_parallel == (1, 2, 3, 4)
    # A four-GPU host keeps the exact TP4 partition; three ranks are chosen only with exactly three GPUs.
    assert RECIPE.exact_rank_counts == frozenset({3})


def test_expert_columns_export_override(monkeypatch):
    """FLARE_GLM53_EXPERT_COLUMNS (fork-only, recipe shim): unset keeps the upstream shard; 704 at three ranks only."""
    from ignition.recipes.glm53.recipe import _expert_columns

    monkeypatch.delenv("FLARE_GLM53_EXPERT_COLUMNS", raising=False)
    assert [_expert_columns(world) for world in WORLDS] == [None] * 4
    for value, expected in (("704", 704), (" 768 ", 768), ("", None)):
        monkeypatch.setenv("FLARE_GLM53_EXPERT_COLUMNS", value)
        assert _expert_columns(3) == expected
    monkeypatch.setenv("FLARE_GLM53_EXPERT_COLUMNS", "640")
    with pytest.raises(ValueError, match="expected one of 768, 704"):
        _expert_columns(3)
    monkeypatch.setenv("FLARE_GLM53_EXPERT_COLUMNS", "704")
    for world in (1, 2, 4):
        with pytest.raises(ValueError, match="three ranks only"):
            _expert_columns(world)


def test_three_rank_context_parallel_export_override(monkeypatch):
    """FLARE_GLM53_DCP3 (fork-only, recipe shim): 1 declares 768-token context-parallel pages at three ranks."""
    from ignition.recipes.glm53.recipe import _context_parallel_three_ranks, _costs

    monkeypatch.delenv("FLARE_GLM53_DCP3", raising=False)
    assert _context_parallel_three_ranks() is False
    assert _costs(1, 2, ranks=3, speculating=True).page_tokens is None
    for world in (1, 2, 4):
        assert _costs(1, 2, ranks=world, speculating=True).page_tokens == 256 * world
    for value, expected in (("1", True), (" 1 ", True), ("0", False), ("", False)):
        monkeypatch.setenv("FLARE_GLM53_DCP3", value)
        assert _context_parallel_three_ranks() is expected
        assert _costs(1, 2, ranks=3, speculating=False).page_tokens == (768 if expected else None)
    monkeypatch.setenv("FLARE_GLM53_DCP3", "yes")
    with pytest.raises(ValueError, match="expected one of 1, 0"):
        _context_parallel_three_ranks()
    # Page bytes per rank do not change: each rank stores 256 records of every page whatever the rank count.
    sharded = _costs(1, 2, ranks=3, speculating=True, context_parallel_three=True)
    whole = _costs(1, 2, ranks=3, speculating=True, context_parallel_three=False)
    assert sharded.page_bytes == whole.page_bytes == 12 * (4 * 64 * 528 + 64 * 132)
    assert (sharded.page_tokens, whole.page_tokens) == (768, None)

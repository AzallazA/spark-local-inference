# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact checkpoint partitioning for GLM tensor-parallel weights."""

import os
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.glm53.shards import Shards
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader, _expert_scales
from ignition.reference.qwen38_quant import decode_csf_scales, decode_csf_slab_scales, expand_csf_scales_padded
from tests.recipes.glm53_a16_layout import a16_logical_weights, a16_packed_scales


@pytest.fixture(scope="module")
def manifest():
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    return inspect_indexed_checkpoint(snapshot)


def read(manifest, name, extent=None):
    with safe_open(str(manifest.snapshot / manifest.tensors[name].shard), framework="pt") as source:
        if extent is not None:
            return source.get_slice(name)[extent].clone()
        return source.get_tensor(name).clone()


def read_padded(manifest, name, rows=None, columns=None, fill=0):
    """Checkpoint rows/columns with the range past the tensor's extent filled (three-rank padding)."""
    shape = manifest.tensors[name].shape
    r0, r1 = (0, shape[0]) if rows is None else rows
    nr = min(r1, shape[0]) - r0
    if columns is not None:  # 1-D tensors (A_log, dt_bias) are read by rows only
        c0, c1 = columns
        nc = min(c1, shape[1]) - c0
    covered = read(manifest, name, slice(r0, r0 + nr) if columns is None else (slice(r0, r0 + nr), slice(c0, c0 + nc)))
    target_shape = list(covered.shape)
    target_shape[0] = r1 - r0
    if columns is not None:
        target_shape[1] = c1 - c0
    if target_shape == list(covered.shape):
        return covered
    target = torch.zeros(target_shape, dtype=covered.dtype)
    if fill:
        target.view(torch.uint8).fill_(fill)
    if columns is None:
        target[:nr] = covered
    else:
        target[:nr, :nc] = covered
    return target


def pad_rows(value, rows):
    """Zero rows appended past a covered expert column range."""
    return torch.cat((value, torch.zeros((rows - value.shape[0], *value.shape[1:]), dtype=value.dtype)))


def pad_columns(value, columns):
    return torch.cat((value, torch.zeros((value.shape[0], columns - value.shape[1]), dtype=value.dtype)), dim=1)


@pytest.mark.parametrize("world", [2, 4])
def test_tp_partitions_cover_checkpoint_exactly(world):
    for extent, alignment in ((64, 1), (8192, 128), (16384, 256), (32768, 512), (2048, 128)):
        partitions = [TensorParallelShard(rank, world).partition(extent, alignment=alignment)
                      for rank in range(world)]
        assert partitions[0][0] == 0 and partitions[-1][1] == extent
        assert all(a[1] == b[0] for a, b in pairwise(partitions))
        assert all(a % alignment == b % alignment == 0 for a, b in partitions)


@pytest.mark.parametrize("rank,world", [(1, 1), (0, 0), (-1, 2), (2, 2), (True, 2)])
def test_invalid_tp(rank, world):
    with pytest.raises(ValueError, match="positive world size and a rank within it"):
        TensorParallelShard(rank, world)


def test_loader_construction_leaves_cuda_uninitialized(manifest):
    initialized = torch.cuda.is_initialized()
    loader = GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 2))
    with pytest.raises(RuntimeError, match="open GlmWeightLoader"):
        loader.load_tensor(f"{TEXT_PREFIX}.embed_tokens.weight")
    assert torch.cuda.is_initialized() == initialized


def test_gate_up_activation_scale_uses_larger_calibration():
    values = {f"expert.{name}.{suffix}": torch.tensor(value) for name in ("up_proj", "gate_proj", "down_proj")
              for suffix, value in (("input_scale", .25), ("weight_scale_2", .01))}
    values["expert.gate_proj.input_scale"] = torch.tensor(.5)
    scales = _expert_scales(values, ("expert",))
    assert scales[2].item() == 2
    assert scales[3].item() == 4
    values["expert.gate_proj.weight_scale_2"] *= 2
    with pytest.raises(ValueError, match="global scales differ"):
        _expert_scales(values, ("expert",))


cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")


def assert_mxfp8(manifest, packed, name, *, rows=None, columns=None, output_rows=None):
    expected = read_padded(manifest, name + ".weight", rows, columns).view(torch.uint8)
    scale_columns = None if columns is None else (columns[0] // 32, columns[1] // 32)
    # Padded rows and columns hold zero values under the neutral UE8M0 scale 127.
    expected_scales = read_padded(manifest, name + ".weight_scale", rows, scale_columns, fill=127)
    actual = packed.weight.values.view(torch.uint8).reshape(packed.out_features, -1).cpu()
    scales = packed.weight.scale_rows.view(torch.uint8).reshape(packed.out_features, -1).cpu()
    if output_rows is not None:
        actual, scales = actual[slice(*output_rows)], scales[slice(*output_rows)]
    assert torch.equal(actual[:, :expected.shape[1]], expected)
    assert torch.equal(scales[:, :expected_scales.shape[1]], expected_scales)


@cuda
@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_actual_dense_and_attention_shards(manifest, world, rank):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    shard = TensorParallelShard(rank, world)
    shards = Shards.of(world)
    prefixes = tuple(f"{TEXT_PREFIX}.layers.{layer}." for layer in (0, 3))
    with GlmWeightLoader(manifest, "cuda:0", shard=shard, prefixes=prefixes) as loader:
        kda = loader.load_attention(0)
        mla = loader.load_attention(3)
        dense = loader.load_mlp(0)
        residual = loader.load_residual(3)
        assert loader._session.torch_copy_bytes == 0
    channels = shards.head_channels(rank, 128)
    prefix = f"{TEXT_PREFIX}.layers.0.self_attn"
    width = channels[1] - channels[0]
    for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
        assert_mxfp8(manifest, kda["input_proj"], f"{prefix}.{name}", rows=channels,
                     output_rows=(index * width, (index + 1) * width))
    for name in ("f_b_proj", "g_b_proj"):
        assert_mxfp8(manifest, kda[name], f"{prefix}.{name}", rows=channels)
    heads = shards.heads
    assert_mxfp8(manifest, kda["input_proj"], f"{prefix}.b_proj", rows=shards.head_range(rank),
                 output_rows=(3 * width, 3 * width + heads))
    assert_mxfp8(manifest, kda["input_proj"], f"{prefix}.f_a_proj",
                 output_rows=(3 * width + heads, 3 * width + heads + 128))
    assert_mxfp8(manifest, kda["o_proj"], f"{prefix}.o_proj", columns=channels)
    assert_mxfp8(manifest, kda["g_a_proj"], f"{prefix}.g_a_proj")
    expected_conv = torch.cat([read_padded(manifest, f"{prefix}.{name}_conv1d.weight", channels)
                               for name in ("q", "k", "v")]).squeeze(1)
    assert torch.equal(kda["conv1d.weight"].cpu(), expected_conv)
    assert torch.equal(kda["A_log"].cpu(), read_padded(manifest, prefix + ".A_log", shards.head_range(rank)))
    assert torch.equal(kda["dt_bias"].cpu(), read_padded(manifest, prefix + ".dt_bias", channels))
    prefix = f"{TEXT_PREFIX}.layers.3.self_attn"
    for name, width, alignment in (("q_b_proj", 16384, 256), ("kv_b_proj", 32768, 512)):
        assert_mxfp8(manifest, mla[name], f"{prefix}.{name}", rows=shards.head_channels(rank, width // 64))
    assert_mxfp8(manifest, mla["o_proj"], f"{prefix}.o_proj", columns=shards.head_channels(rank, 256))
    assert_mxfp8(manifest, mla["qkv_a_proj"], f"{prefix}.q_a_proj", output_rows=(0, 1536))
    assert_mxfp8(manifest, mla["qkv_a_proj"], f"{prefix}.kv_a_proj_with_mqa", output_rows=(1536, 2048))
    assert_mxfp8(manifest, mla["qkv_a_proj"], f"{prefix}.indexer.wk", output_rows=(2048, 2176))
    assert_mxfp8(manifest, mla["qkv_a_proj"], f"{prefix}.indexer.weights_proj", output_rows=(2176, 2208))
    assert_mxfp8(manifest, mla["indexer.wq_b"], f"{prefix}.indexer.wq_b")
    prefix = f"{TEXT_PREFIX}.layers.0.mlp"
    extent = slice(*shard.padded_partition(12288, alignment=128))
    expected = torch.cat([read(manifest, f"{prefix}.{name}.weight", extent) for name in ("gate_proj", "up_proj")])
    assert torch.equal(dense.gate_up.cpu(), expected)
    assert torch.equal(dense.down.cpu(), read(manifest, prefix + ".down_proj.weight", (slice(None), extent)))
    for name, value in residual.items():
        assert torch.equal(value.cpu(), read(manifest, f"{TEXT_PREFIX}.layers.3.{name}"))


@cuda
@pytest.mark.parametrize("world,rank,activation,expert_columns", [(2, 0, "a4", None), (2, 1, "a4", None), (3, 0, "a4", None), (3, 1, "a4", None),
                                                   (3, 2, "a4", None), (3, 2, "a16", None), (4, 0, "a4", None), (4, 1, "a4", None),
                                                   (4, 2, "a4", None), (4, 3, "a4", None),
                                                   # FLARE_GLM53_EXPERT_COLUMNS=704: ranks 0/1 exact, rank 2 padded by 64.
                                                   (3, 0, "a16", 704), (3, 1, "a16", 704), (3, 1, "a4", 704),
                                                   (3, 2, "a16", 704), (3, 2, "a4", 704)])
def test_actual_nvfp4_shards_and_shared_scale_storage(manifest, world, rank, activation, expert_columns):
    """Three ranks pad 2048 expert columns to 2304: rank 2 holds columns 1536..2047 and 256 zero columns.
    Padded up/gate rows decode to zero scales on both b12x paths; padded down columns decode to zero on the
    A16 Triton expansion (what SM120 serves) and to each row's base on the compact A4 payload
    (expand_csf_scales_padded documents both); the packed values there are zero either way."""
    from b12x.moe import fused_moe as moe

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    shard = TensorParallelShard(rank, world)
    ids = (0, 287)
    prefix = f"{TEXT_PREFIX}.layers.3.mlp.experts"
    with GlmWeightLoader(manifest, "cuda:0", shard=shard, expert_activation=activation, expert_columns=expert_columns,
                         prefixes=tuple(f"{prefix}.{i}." for i in ids)) as loader:
        scratch = loader.allocate_expert_scale_scratch(len(ids))
        experts = loader.load_experts(3, expert_ids=ids, scale_scratch=scratch)
        assert loader._session.torch_copy_bytes == 0
    assert experts.plan.activation.swiglu_limit == 10
    assert experts._impl.w1_blockscale.data_ptr() == scratch[0].data_ptr()
    assert experts._impl.w2_blockscale.data_ptr() == scratch[1].data_ptr()
    moe.expand_scales(experts)
    shards = Shards.of(world, expert_columns=expert_columns)
    local = shards.routed_width
    from b12x.moe._shared.execution import WeightPreparationTransform

    # A16 on SM120 repacks the banks in place into W4A16 MMA tiles and permutes the expanded scales
    # (tests/recipes/glm53_a16_layout.py); A4 keeps the packed bytes and the swizzled E4M3 scale grid.
    a16_packed = WeightPreparationTransform.W4A16_PACKED in experts._impl.plan.transforms
    assert a16_packed == (activation == "a16")
    rotation = local if experts.plan.source.w13_layout.value == "w13" else 0
    decoded = []
    if not a16_packed:
        for storage, rows, columns in ((scratch[0], 2 * local, 256), (scratch[1], 4096, local // 16)):
            logical = storage.view(torch.uint8).reshape(len(ids), rows // 128, columns // 4, 32, 4, 4)
            decoded.append(logical.permute(0, 1, 4, 3, 2, 5).reshape(len(ids), rows, columns).cpu())
    first, last = shards.routed_columns(rank)
    end = min(last, 2048)
    expected_scales = {"w13": [], "w2": []}
    for index, expert in enumerate(ids):
        scales, weights = {}, {}
        for name, rows, columns in (("up_proj", 2048, 256), ("gate_proj", 2048, 256), ("down_proj", 4096, 128)):
            name_prefix = f"{prefix}.{expert}.{name}"
            if f"{name_prefix}.weight_scale.csf32_bases" in manifest.tensors:
                planes = [read(manifest, f"{name_prefix}.weight_scale.csf32_{suffix}")
                          for suffix in ("bases", "codes", "page_bases", "offsets", "exceptions")]
                full = decode_csf_scales(*planes, rows=rows, columns=columns)
                bases = planes[0]
            else:
                fixed = read(manifest, f"{name_prefix}.weight_scale.nvfp4_csf_fixed")
                full = decode_csf_slab_scales(
                    fixed, read(manifest, f"{name_prefix}.weight_scale.nvfp4_csf_exceptions"),
                    rows=rows, columns=columns)
                bases = fixed[:, :16].reshape(rows)
            # `expand_scales` decodes the compact payload into the scratch in both activation modes, so padded
            # down-projection columns carry each row's base (lifted and clamped at A16); padded rows are zero.
            compact = True
            if name == "down_proj":
                scales[name] = expand_csf_scales_padded(full, bases, row_range=(0, rows),
                    column_range=(first // 16, end // 16), padded_columns=local // 16, compact=compact)
                weights[name] = pad_columns(read(manifest, f"{name_prefix}.weight", (slice(None), slice(first // 2, end // 2))),
                                            local // 2)
            else:
                scales[name] = expand_csf_scales_padded(full, bases, row_range=(first, end),
                    column_range=(0, columns), padded_rows=local, compact=compact)
                weights[name] = pad_rows(read(manifest, f"{name_prefix}.weight", slice(first, end)), local)
        if a16_packed:
            actual_w13 = a16_logical_weights(experts._impl.w1_fp4[index], size_k=4096, size_n=2 * local,
                                             row_rotation=rotation)
            actual_w2 = a16_logical_weights(experts._impl.w2_fp4[index], size_k=local, size_n=4096)
        else:
            actual_w13, actual_w2 = experts._impl.w1_fp4[index].cpu(), experts._impl.w2_fp4[index].cpu()
        assert torch.equal(actual_w13, torch.cat((weights["up_proj"], weights["gate_proj"])))
        assert torch.equal(actual_w2, weights["down_proj"])
        expected_scales["w13"].append(torch.cat((scales["up_proj"], scales["gate_proj"])))
        expected_scales["w2"].append(scales["down_proj"])
        if a16_packed:
            # The scales are checked for the whole batch below, in the permuted W4A16 grid.
            continue
        assert torch.equal(decoded[0][index], torch.cat((scales["up_proj"], scales["gate_proj"])))
        actual_down, expected_down = decoded[1][index], scales["down_proj"]
        if not torch.equal(actual_down, expected_down):
            differing = (actual_down != expected_down).nonzero()
            columns_hit = sorted(set(differing[:, 1].tolist()))
            rows_hit = differing[:, 0]
            raise AssertionError(
                f"down_proj scales differ at {differing.shape[0]} elements; columns {columns_hit[:8]}..{columns_hit[-8:]}"
                f" (covered 0..{(end - first) // 16 - 1}, padded {(end - first) // 16}..{local // 16 - 1}), rows "
                f"{int(rows_hit.min())}..{int(rows_hit.max())}; first differences (row, col, actual, expected, base): "
                + ", ".join(f"({int(r)}, {int(c)}, {int(actual_down[r, c])}, {int(expected_down[r, c])}, {int(bases[r])})"
                            for r, c in differing[:6].tolist()))
        if end < last:
            # Padded up/gate rows are zero on both paths; padded down columns are zero (A16) or the row base (A4).
            assert not decoded[0][index][end - first:local].count_nonzero()
            assert not decoded[0][index][local + end - first:].count_nonzero()
            padded_columns = decoded[1][index][:, (end - first) // 16:]
            if compact:
                assert torch.equal(padded_columns, bases[:, None].expand_as(padded_columns))
            else:
                assert not padded_columns.count_nonzero()
    if a16_packed:
        # After expand_scales the scratch holds `native.w13_scale`/`w2_scale`: the Triton-expanded padded
        # scales (zero pad) permuted by _permute_nvfp4_scales for the whole two-expert batch.
        globals_ = {name: torch.stack([read(manifest, f"{prefix}.{expert}.{projection}.weight_scale_2").reshape(())
                                       for expert in ids])
                    for name, projection in (("w13", "up_proj"), ("w2", "down_proj"))}
        io_dtype = experts.plan.activation.io_dtype
        pad_groups = None if end == last else ((end - first) // 16, local // 16)
        for name, size_k, size_n, rot, pads in (("w13", 4096, 2 * local, rotation, None),
                                                ("w2", local, 4096, None, pad_groups)):
            expected = a16_packed_scales(torch.stack(expected_scales[name]), globals_[name], size_k=size_k,
                                         size_n=size_n, io_dtype=io_dtype, row_rotation=rot, pad_groups=pads)
            storage = scratch[0] if name == "w13" else scratch[1]
            actual = storage.view(torch.uint8).reshape(-1)[:expected.numel()].view(torch.float8_e4m3fn).view_as(expected)
            actual, expected = actual.cpu().view(torch.uint8), expected.view(torch.uint8)
            if not torch.equal(actual, expected):
                differing = (actual != expected).nonzero()
                raise AssertionError(f"{name}: {differing.shape[0]} grid bytes differ; grid rows (K groups) "
                                     f"{sorted(set(differing[:, 1].tolist()))[:12]}; first: "
                                     + ", ".join(f"{tuple(d.tolist())} actual {int(actual[tuple(d)])} expected "
                                                 f"{int(expected[tuple(d)])}" for d in differing[:6]))


@cuda
@pytest.mark.parametrize("world,rank,expert_columns", [(2, 1, None), (3, 2, None), (4, 3, None), (3, 1, 704), (3, 2, 704)])
def test_mtp_experts_preserve_packed_weights_and_e4m3_scales(manifest, world, rank, expert_columns, monkeypatch):
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.moe import fused_moe

    packed = {}
    prepare = fused_moe.prepare_weights

    def capture(*, plan, weights, **kwargs):
        # load_fp4_experts forwards scale_scratch (None here) since the CSF-scale preparation change.
        for name in ("w13", "w2", "w13_block_scales", "w2_block_scales"):
            packed[name] = getattr(weights, name).clone()
        return prepare(plan=plan, weights=weights, **kwargs)

    monkeypatch.setattr(fused_moe, "prepare_weights", capture)

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    shard = TensorParallelShard(rank, world)
    ids = (0, 287)
    prefix = f"{TEXT_PREFIX}.layers.45.mlp.experts"
    with GlmWeightLoader(manifest, "cuda:0", shard=shard, expert_columns=expert_columns,
                         prefixes=tuple(f"{prefix}.{i}." for i in ids)) as loader:
        experts = loader.load_experts(45, expert_ids=ids)
        assert loader._session.torch_copy_bytes == 0
    assert experts.plan.activation.mode == "a16"
    first, last = Shards.of(world, expert_columns=expert_columns).routed_columns(rank)
    local, end = last - first, min(last, 2048)
    weights, scales = {}, {}
    for projection in ("up_proj", "gate_proj", "down_proj"):
        extent = (slice(None), slice(first // 2, end // 2)) if projection == "down_proj" else slice(first, end)
        scale_extent = (slice(None), slice(first // 16, end // 16)) if projection == "down_proj" else slice(first, end)
        pad = (lambda value, width: pad_columns(value, width)) if projection == "down_proj" else pad_rows
        weights[projection] = torch.stack([pad(read(manifest, f"{prefix}.{expert}.{projection}.weight", extent),
                                               local // 2 if projection == "down_proj" else local) for expert in ids])
        scales[projection] = torch.stack([pad(read(manifest, f"{prefix}.{expert}.{projection}.weight_scale", scale_extent),
                                              local // 16 if projection == "down_proj" else local) for expert in ids])
    assert torch.equal(packed["w13"].cpu(), torch.cat((weights["up_proj"], weights["gate_proj"]), dim=1))
    assert torch.equal(packed["w2"].cpu(), weights["down_proj"])
    for actual, expected in ((packed["w13_block_scales"], torch.cat((scales["up_proj"], scales["gate_proj"]), dim=1)),
                             (packed["w2_block_scales"], scales["down_proj"])):
        expected = swizzle_block_scale(expected.cuda())
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

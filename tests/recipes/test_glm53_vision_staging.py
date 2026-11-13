# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fork-only GLM vision weight staging (FLARE_GLM53_VISION_STAGE=1), DeepSeek's TP3 `vision_stage` for GLM.

CPU checks cover the packed byte layout (bytes, geometry and aliasing survive packing and binding), the split of
the tower into parked blocks, parked tail and resident weights at the checkpoint's real geometry, slot binding
(block ``i`` reads slot ``i % 2``; the tail aliases the gate scratch), the staging schedule, and the export
switch's plumbing. They run without a GPU:

    CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q tests/recipes/test_glm53_vision_staging.py

The CUDA check (FLARE_RUN_CUDA_TESTS=1, one GPU, the GLM checkpoint) parks the tower in pinned host memory and
compares staged replay with the resident tower bit for bit under every forward compiler:

    FLARE_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 FLARE_GLM53_SNAPSHOT=/path/to/ckpt/glm53-fp4-local \\
        .venv/bin/python -m pytest -q tests/recipes/test_glm53_vision_staging.py
"""

from __future__ import annotations

import dataclasses
import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.vision_staging import PackedLayout, map_tensors, pack_tree, tensor_leaves
from ignition.recipes.glm53 import recipe as glm_recipe
from ignition.recipes.glm53.bootstrap import PreparationConfig
from ignition.recipes.glm53.vision import (
    VISION_BLOCKS, VisionScratch, VisionStaging, VisionWeights, staged_blocks,
)

_CUDA = os.environ.get("FLARE_RUN_CUDA_TESTS") == "1"

if _CUDA:
    from tests.recipes.test_glm53_vision import manifest  # noqa: F401  (checkpoint fixture for the CUDA check)

# The checkpoint's vision_config (glm53-fp4-local config.json).
VISION_CONFIG = dict(attention_bias=True, depth=24, hidden_act="silu", hidden_size=1024, image_size=448,
                     in_channels=3, intermediate_size=4096, num_heads=16, out_hidden_size=4096, patch_size=14,
                     projection_intermediate_size=10240, rms_norm_eps=1e-5, spatial_merge_size=2,
                     swiglu_limit=10.0, temporal_patch_size=2)


def _bytes(tensor):
    """Raw bytes of a tensor (NaN-safe comparison of uninitialized storage)."""
    return tensor.contiguous().view(-1).view(torch.uint8)


def _same_bytes(a, b):
    return all(torch.equal(_bytes(x), _bytes(y)) and x.shape == y.shape and x.stride() == y.stride()
               and x.dtype == y.dtype for x, y in zip(tensor_leaves(a), tensor_leaves(b), strict=True))


# Packed layout ------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Rows:
    values: torch.Tensor
    scale_rows: torch.Tensor | None
    scale_mma: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _Packed:
    weight: _Rows
    in_features: int
    out_features: int


def test_packed_layout_keeps_bytes_geometry_aliasing_and_fields():
    shared = torch.arange(4096, dtype=torch.int32).view(torch.uint8)
    tree = {"projection": _Packed(_Rows(torch.randn(48, 64).to(torch.float8_e4m3fn),
                                        shared[1000:1384].view(48, 8), shared[2048:2432].view(4, 96).t()), 64, 48),
            "bias": torch.randn(48).bfloat16(), "norm": torch.randn(7, dtype=torch.float32)[1:]}
    layout, flat, bound = pack_tree(tree, lambda n: torch.zeros(n, dtype=torch.uint8))
    assert flat.numel() == layout.nbytes and layout.nbytes % 256 == 0
    assert _same_bytes(tree, bound)
    # Non-tensor fields survive; views of one storage keep sharing it at the same relative offset.
    assert (bound["projection"].in_features, bound["projection"].out_features) == (64, 48)
    rows, mma = bound["projection"].weight.scale_rows, bound["projection"].weight.scale_mma
    assert rows.untyped_storage().data_ptr() == mma.untyped_storage().data_ptr() == flat.data_ptr()
    assert (mma.data_ptr() - rows.data_ptr()) == 2048 - 1000
    # Only the covered span of a shared storage is packed, from a 256-byte-aligned start (alignment kept).
    assert layout.spans[0] == (0, 64 * 48)
    assert layout.spans[1] == (768, 2048 + 384)
    assert (rows.data_ptr() - flat.data_ptr()) % 256 == 1000 % 256
    # Rebinding to another buffer (a VRAM slot) gives the same geometry; filling it reproduces the tree.
    slot = torch.zeros(layout.nbytes + 512, dtype=torch.uint8)[256:]
    again = layout.bind(bound, slot)
    slot[:layout.nbytes].copy_(flat)
    assert _same_bytes(tree, again)
    # The layout keeps no reference to the source storage.
    assert not any(isinstance(value, torch.UntypedStorage) for value in vars(layout).values())


def test_packed_layout_rejects_mismatched_trees_and_buffers():
    tree = {"a": torch.zeros(16, dtype=torch.bfloat16), "b": torch.zeros(3, dtype=torch.float32)}
    layout = PackedLayout(tree)
    with pytest.raises(ValueError, match="layout"):
        layout.bind({"a": torch.zeros(17, dtype=torch.bfloat16), "b": torch.zeros(3)}, torch.zeros(layout.nbytes, dtype=torch.uint8))
    with pytest.raises(ValueError, match="flat uint8"):
        layout.bind(tree, torch.zeros(layout.nbytes - 1, dtype=torch.uint8))
    with pytest.raises(ValueError, match="256-byte"):
        layout.bind(tree, torch.zeros(layout.nbytes + 8, dtype=torch.uint8)[8:])
    with pytest.raises(ValueError, match="layout"):
        layout.pack({"a": torch.zeros(16, dtype=torch.bfloat16)}, torch.zeros(layout.nbytes, dtype=torch.uint8))
    assert PackedLayout(map_tensors(tree, torch.clone)).signature == layout.signature


# Checkpoint geometry ------------------------------------------------------------------------------------------

def _mxfp8(n, k):
    from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    scales = torch.empty((n, k // 32), dtype=torch.uint8)
    return MXFP8LinearWeight(MXFP8Rows(torch.empty((n, k), dtype=torch.float8_e4m3fn), scales.view(1, n, k // 32),
                                       torch.empty(n * k // 32, dtype=torch.uint8)), k, k, n)


def _nvfp4(n, k):
    from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight

    return NVFP4LinearWeight(torch.empty((n, k // 2), dtype=torch.uint8), torch.empty(n * k // 16, dtype=torch.uint8),
                             torch.ones(1, dtype=torch.float32), "multiplier", k, n)


def _projection(path, shape):
    if path.startswith("blocks.") and ".attn." in path:
        return _mxfp8(*shape)
    return _nvfp4(*shape)


def _checkpoint_weights():
    """VisionWeights as `VisionWeights.load` builds them, at the checkpoint's shapes (uninitialized bytes)."""
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    with torch.device("meta"):
        module = Glm5NextVisionModel(Glm5NextVisionConfig(**VISION_CONFIG))
    quantized = {f"blocks.{i}.{p}" for i in range(24) for p in
                 ("attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    quantized |= {f"merger.{p}" for p in ("proj", "gate_proj", "up_proj", "down_proj")}
    parameters, projections = {}, {}
    for path, parameter in module.named_parameters():
        prefix = path.removesuffix(".weight")
        if path.endswith(".weight") and prefix in quantized:
            projections[prefix] = _projection(prefix, tuple(parameter.shape))
        else:
            parameters[path] = torch.randn(parameter.shape).to(torch.bfloat16)
    downsample = parameters["downsample.weight"].permute(0, 2, 3, 1).reshape(4096, 4096).contiguous()
    return VisionWeights(parameters, projections, downsample)


class _Host:
    """Stand-in for core storage host regions: flat uint8 buffers named like the export's regions."""

    def __init__(self):
        self.regions = {}

    def __call__(self, name, nbytes):
        flat = torch.zeros(nbytes, dtype=torch.uint8)
        self.regions[name] = flat
        return flat


@pytest.fixture(scope="module")
def checkpoint():
    weights = _checkpoint_weights()
    host = _Host()
    return weights, weights.park(host), host


def test_parking_splits_the_tower_into_blocks_tail_and_resident_set(checkpoint):
    weights, parked, host = checkpoint
    parking = parked.parking
    assert list(host.regions) == [*(f"model.visual.blocks.{i}" for i in range(VISION_BLOCKS)), "model.visual.tail"]
    # Every block shares one layout of ~11 MiB; the tail (downsample + merger) is ~109 MiB.
    assert len({flat.nbytes for flat in parking.block_sources}) == 1
    block, tail = parking.block_sources[0].nbytes, parking.tail_source.nbytes
    assert 10 << 20 < block < 12 << 20 and 100 << 20 < tail < 115 << 20
    # The convolution-layout downsample weight is dropped; everything else is parked or resident.
    assert set(parked.parameters) == set(weights.parameters) - {"downsample.weight"}
    assert set(parked.projections) == set(weights.projections)
    resident = {k for k, v in parked.parameters.items()
                if not any(v.data_ptr() >= f.data_ptr() and v.data_ptr() < f.data_ptr() + f.nbytes
                           for f in host.regions.values())}
    assert resident == {"patch_embed.proj.weight", "patch_embed.proj.bias", "post_layernorm.weight"}
    # Parked bytes equal the loaded ones.
    assert _same_bytes(weights.projections, parked.projections)
    assert _same_bytes({k: weights.parameters[k] for k in parked.parameters}, parked.parameters)
    assert torch.equal(_bytes(weights.downsample), _bytes(parked.downsample))
    # The parked geometry trees hold no loaded tensors.
    loaded = {t.untyped_storage().data_ptr() for t in tensor_leaves((weights.parameters, weights.projections))}
    assert not loaded & {t.untyped_storage().data_ptr()
                         for t in tensor_leaves((parking.block_tree, parking.tail_tree))}


def test_staging_binds_blocks_to_alternating_slots_and_the_tail_to_gate_scratch(checkpoint):
    weights, parked, _ = checkpoint
    scratch = VisionScratch.allocate(1024, "cpu")
    with pytest.raises(ValueError, match="parked"):
        VisionStaging(weights, scratch)
    # 1024 patches leave 8 MiB of gate scratch: too small for the tail, which then gets its own slot.
    small = VisionStaging(parked, scratch)
    assert not small.tail_aliased and small.allocated_bytes == small.blocks.nbytes + small.tail.nbytes
    scratch = VisionScratch.allocate(16384, "cpu")
    staging = VisionStaging(parked, scratch)
    assert staging.tail_aliased
    assert staging.allocated_bytes == 2 * parked.parking.block_sources[0].nbytes
    gate = scratch.gate.untyped_storage().data_ptr()
    assert staging.tail.slot(0)["tail"].untyped_storage().data_ptr() == gate
    parameters, projections, downsample = staging.bound(parked)
    assert set(parameters) == set(parked.parameters) and set(projections) == set(parked.projections)

    def within(tensor, flat):
        return flat.data_ptr() <= tensor.data_ptr() < flat.data_ptr() + flat.nbytes

    for index in range(VISION_BLOCKS):
        slot = staging.blocks.slot(index)["block"]
        assert all(within(t, slot) for k, t in {**parameters, **projections}.items()
                   for t in tensor_leaves(t) if k.startswith(f"blocks.{index}."))
    assert within(downsample, staging.tail.slot(0)["tail"])
    # Emulate the copies: after staging block i (and the tail) the bound weights equal the loaded ones.
    for index in (0, 1, 7, 23):
        staging.blocks.slot(index)["block"].copy_(staging.block_sources[index]["block"])
        prefix = f"blocks.{index}."
        assert _same_bytes({k: v for k, v in projections.items() if k.startswith(prefix)},
                           {k: v for k, v in weights.projections.items() if k.startswith(prefix)})
        assert _same_bytes({k: v for k, v in parameters.items() if k.startswith(prefix)},
                           {k: v for k, v in weights.parameters.items() if k.startswith(prefix)})
    staging.tail.slot(0)["tail"].copy_(staging.tail_sources["tail"])
    assert torch.equal(_bytes(downsample), _bytes(weights.downsample))
    assert _same_bytes({k: v for k, v in projections.items() if k.startswith("merger.")},
                       {k: v for k, v in weights.projections.items() if k.startswith("merger.")})


class _Recorder:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def stage(self, index, sources):
        self.log.append((self.name, "stage", index))

    def join(self, index, dependency):
        self.log.append((self.name, "join", index, dependency))


def test_block_schedule_overlaps_the_next_copy_with_the_current_block():
    log = []
    staging = SimpleNamespace(blocks=_Recorder(log, "blocks"), block_sources=tuple(range(VISION_BLOCKS)))

    def run(block, hidden):
        log.append(("run", block))
        return hidden + 1

    assert staged_blocks(list(range(VISION_BLOCKS)), 0, run, staging) == VISION_BLOCKS
    assert log[0] == ("blocks", "stage", 0)
    for index in range(VISION_BLOCKS):
        position = log.index(("run", index))
        # Block i waits for its own copy, which followed block i - 1's output ...
        assert log.index(("blocks", "join", index, index)) < position
        # ... and block i + 1's copy (into the slot block i - 1 read) is issued before block i runs.
        if index + 1 < VISION_BLOCKS:
            stage = log.index(("blocks", "stage", index + 1))
            assert log.index(("blocks", "join", index, index)) < stage < position
            if index:
                assert log.index(("run", index - 1)) < stage
    assert ("blocks", "stage", VISION_BLOCKS) not in log
    # Without staging the tower runs unchanged.
    assert staged_blocks([1, 2, 3], 0, lambda block, hidden: hidden + block, None) == 6


# Export switch ------------------------------------------------------------------------------------------------

def test_vision_stage_defaults_off_and_is_validated(monkeypatch):
    assert PreparationConfig(devices=(0, 1)).vision_stage is False
    assert PreparationConfig(devices=(0, 1), vision_stage=True).vision_stage is True
    with pytest.raises(TypeError, match="vision_stage"):
        PreparationConfig(devices=(0, 1), vision_stage=1)
    monkeypatch.setenv("FLARE_GLM53_VISION_STAGE", "yes")
    with pytest.raises(ValueError, match="VISION_STAGE"):
        glm_recipe._tp3_override("VISION_STAGE", ("1", "0"), "0")


@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_export_environment_reaches_the_preparation_config(monkeypatch, world, value, expected):
    import ignition.preparation.rank_group as rank_group
    from tests.recipes.test_glm53_vram_options import _plan

    for name in ("EXPERT_ACTIVATION", "EXPAND_SCALES", "EXPERT_COLUMNS", "KDA_SIDE_CHAIN", "DRAFT_HEAD",
                 "DRAFT_EH_PROJ", "EMBED_HOST", "VISION_STAGE"):
        monkeypatch.delenv(f"FLARE_GLM53_{name}", raising=False)
    if value is not None:
        monkeypatch.setenv("FLARE_GLM53_VISION_STAGE", value)
    seen = {}
    monkeypatch.setattr(rank_group, "prepare_group",
                        lambda snapshot, config, function, **kwargs: seen.setdefault("config", config))
    plan = _plan(world)
    plan.language_model_only = False
    glm_recipe.prepare("/nonexistent", plan, compile_forwards=False, kernel_cache=None, progress=None)
    config = seen["config"]
    assert config.vision_stage is expected
    assert (config.embed_host, config.kda_side_chain, config.language_model_only) == (False, False, False)


# GPU: staged replay equals the resident tower ------------------------------------------------------------------

@pytest.mark.skipif(not _CUDA, reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")
@torch.no_grad()
@pytest.mark.parametrize("compiler", ["none", "worker"])
@pytest.mark.parametrize("overlap", [True, False])
def test_staged_replay_matches_the_resident_tower(manifest, compiler, overlap, tmp_path):
    from functools import partial

    from b12x.preparation import PreparationSession
    from transformers.vision_utils import get_vision_position_ids

    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.glm53.vision import PreparedVision, VisionMetadata
    from tests.recipes.test_glm53_vision import _input, _module
    from tests.recipes.test_glm53_vram_options import _PinnedHost

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    with DirectWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(_module(manifest), loader)
    host = _PinnedHost(device)
    parked = weights.park(lambda name, nbytes: host(name, (nbytes,), torch.uint8).device_view)
    capacity = 4096
    inputs = _input([(14 * 56, 14 * 56)], 0)
    count = inputs.pixel_values.shape[0]
    outputs = {}
    for variant, source in (("resident", weights), ("staged", parked)):
        module = _module(manifest)
        scratch = VisionScratch.allocate(16384, device)
        metadata = VisionMetadata.allocate(capacity, 1, device)
        pixels = torch.zeros((capacity, 1176), dtype=torch.float32, device=device)
        arena = ScratchArena(device, overlap=overlap)
        with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
            staging = (VisionStaging(parked, scratch, stream=arena.auxiliary_stream if overlap else None)
                       if variant == "staged" else None)
            # Workspaces as in the native replay test (none); the arena only supplies the copy stream.
            prepared = PreparedVision(module, scratch.rows(capacity), metadata, source, session=session,
                                      staging=staging)
            session.freeze()
            forward = lambda: prepared(pixels)  # noqa: E731
            if compiler == "worker":
                from ignition.preparation.aot import ForwardCache
                from ignition.preparation.execution import GraphSignature

                cache = ForwardCache(namespace={"vision": "glm53-staged"}, device=device, root=tmp_path / variant)
                forward = cache.prepare_many({"vision": partial(prepared, pixels)},
                                             {"vision": GraphSignature("vision", 1, capacity)})["vision"]
            metadata.position_ids[:count].copy_(get_vision_position_ids(inputs.image_grid_thw, 2))
            metadata.cu_seqlens.copy_(torch.tensor([0, count], dtype=torch.int32))
            metadata.num_patches.fill_(count)
            pixels[:count].copy_(inputs.pixel_values)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                forward()
            results = []
            for _ in range(3):
                scratch.output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                results.append(scratch.output[:count // 4].clone())
            outputs[variant] = results
    for result in (*outputs["resident"], *outputs["staged"]):
        torch.testing.assert_close(result, outputs["resident"][0], rtol=0, atol=0)

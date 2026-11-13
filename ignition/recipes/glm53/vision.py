# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured GLM vision tower with canonical blocks and prepared projections."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)

from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.components.vision_staging import PackedLayout, WeightStaging, pack_tree
from ignition.preparation.tensor_metadata import record_numeric_tensor, unrecorded_numeric_tensors

VISION_BLOCKS = 24
# Weight staging (fork-only, FLARE_GLM53_VISION_STAGE=1): the 24 blocks share one geometry and are staged
# through two VRAM slots; the tail (downsample and merger) is staged after the last block into the dead
# feed-forward gate scratch; the patch embedding and final norm stay resident.
RESIDENT_PREFIXES = ("patch_embed.", "post_layernorm.")
# VisionScratch elements per patch row (see VisionScratch.allocate).
SCRATCH_WIDTH = 2048 + 11264
# Row bands for the FP32 RMSNorm / rotary / LayerNorm temporaries: the per-row math is unchanged, but eager
# capture (graph memory measurement) keeps 1/ROW_BANDS of the rows' FP32 temporaries alive instead of all of
# them, so the vision graphs need no more shared graph pool than the prefill graphs.
ROW_BANDS = 8


@dataclass(frozen=True)
class VisionMetadata:
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    num_patches: torch.Tensor

    @classmethod
    def allocate(cls, patches, images, device):
        if patches <= 0 or patches % 4 or images <= 0:
            raise ValueError("GLM vision requires positive image capacity and patch capacity divisible by four")
        return cls(torch.zeros((patches, 2), dtype=torch.int64, device=device),
                   torch.zeros(images + 1, dtype=torch.int32, device=device),
                   torch.zeros(1, dtype=torch.int32, device=device))


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    hidden: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention: torch.Tensor
    projected: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    merged: torch.Tensor
    merger_gate: torch.Tensor
    merger_up: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, device):
        """One BF16 buffer laid out by lifetime (fork): ``hidden`` (the residual stream, updated in place) and
        ``projected`` (RMSNorm outputs and projection results) persist through the tower; every other field
        lives in one phase region whose fields are live in disjoint phases. Patches are read only by the patch
        embedding; the attention fields (qkv, query/key/value, attention) and the feed-forward fields (gate,
        up) alternate inside each block; the merger fields are live only after the last block and sit past
        ``gate``, which the staged tail weights alias. 13312 elements per patch row (832 MiB at 32768
        patches) instead of 25752 (1609.5 MiB) with one storage per field."""
        if patches <= 0 or patches % 4:
            raise ValueError("GLM vision patch capacity must be positive and divisible by four")
        flat = torch.zeros(patches * SCRATCH_WIDTH, dtype=torch.bfloat16, device=device)

        def field(start, *shape):
            count = 1
            for size in shape:
                count *= size
            return flat[patches * start:patches * start + count].view(shape)

        quarter, phase = patches // 4, 2048
        return cls(field(phase, patches, 1176), field(0, patches, 1024), field(phase, patches, 3072),
                   field(phase + 3072, patches, 16, 64), field(phase + 4096, patches, 16, 64),
                   field(phase + 5120, patches, 16, 64), field(phase + 6144, patches, 1024),
                   field(1024, patches, 1024),
                   field(phase, patches, 4096), field(phase + 4096, patches, 4096),
                   field(phase + 4096, quarter, 4096), field(phase + 5120, quarter, 10240),
                   field(phase + 7680, quarter, 10240), field(phase + 10240, quarter, 4096))

    def rows(self, patches):
        if patches <= 0 or patches % 4 or patches > self.patches.shape[0]:
            raise ValueError("GLM vision view exceeds its patch capacity")
        merged = {"merged", "merger_gate", "merger_up", "output"}
        return type(self)(*(getattr(self, name)[:patches // 4 if name in merged else patches]
                            for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class VisionParking:
    """Host-parked block and tail weights: one flat region per block (sharing `block_layout`) and one
    for the tail. `block_tree`/`tail_tree` give the geometry the layouts bind (keys relative to a block)."""
    block_layout: PackedLayout
    block_sources: tuple
    block_tree: dict
    tail_layout: PackedLayout
    tail_source: torch.Tensor
    tail_tree: dict

    @property
    def parked_bytes(self):
        return sum(flat.nbytes for flat in self.block_sources) + self.tail_source.nbytes


def _block_tree(parameters, projections, index):
    prefix = f"blocks.{index}."
    return {"parameters": {k.removeprefix(prefix): v for k, v in parameters.items() if k.startswith(prefix)},
            "projections": {k.removeprefix(prefix): v for k, v in projections.items() if k.startswith(prefix)}}


def _is_tail(path):
    return not path.startswith(("blocks.", *RESIDENT_PREFIXES))


@dataclass(frozen=True)
class VisionWeights:
    parameters: dict[str, torch.Tensor]
    projections: dict[str, object]
    downsample: torch.Tensor
    parking: VisionParking | None = None

    def park(self, allocate):
        """Move the block and tail weights into ``allocate(name, nbytes)`` flat uint8 regions (device-visible
        host memory); the returned weights view the parked bytes and keep the resident set in place.

        The checkpoint's convolution-layout ``downsample.weight`` is dropped: only its transposed copy
        (`downsample`) is read."""
        parameters = {k: v for k, v in self.parameters.items() if k != "downsample.weight"}
        trees = tuple(_block_tree(parameters, self.projections, i) for i in range(VISION_BLOCKS))
        layout = PackedLayout(trees[0])
        if any(PackedLayout(tree).signature != layout.signature for tree in trees):
            raise ValueError("GLM vision staging requires every block to share one weight layout")
        if any(k.startswith("blocks.") and not k.startswith(tuple(f"blocks.{i}." for i in range(VISION_BLOCKS)))
               for k in (*parameters, *self.projections)):
            raise ValueError("GLM vision weights name a block outside the 24-block tower")
        sources, parked_blocks, parked_parameters, parked_projections = [], [], {}, {}
        for index, tree in enumerate(trees):
            _, flat, parked = pack_tree(tree, lambda n, i=index: allocate(f"model.visual.blocks.{i}", n))
            sources.append(flat)
            parked_blocks.append(parked)
            parked_parameters.update({f"blocks.{index}.{k}": v for k, v in parked["parameters"].items()})
            parked_projections.update({f"blocks.{index}.{k}": v for k, v in parked["projections"].items()})
        tail = {"parameters": {k: v for k, v in parameters.items() if _is_tail(k)},
                "projections": {k: v for k, v in self.projections.items() if _is_tail(k)},
                "downsample": self.downsample}
        tail_layout, tail_source, tail_parked = pack_tree(tail, lambda n: allocate("model.visual.tail", n))
        parked_parameters.update(tail_parked["parameters"])
        parked_projections.update(tail_parked["projections"])
        parked_parameters.update({k: v for k, v in parameters.items() if k.startswith(RESIDENT_PREFIXES)})
        if set(parked_parameters) != set(parameters) or set(parked_projections) != set(self.projections):
            raise ValueError("GLM vision weights must split into blocks, tail and resident sets")
        # Geometry trees view the parked bytes, so no reference to the loaded device copies survives.
        parking = VisionParking(layout, tuple(sources), parked_blocks[0], tail_layout, tail_source, tail_parked)
        return replace(self, parameters=parked_parameters, projections=parked_projections,
                       downsample=tail_parked["downsample"], parking=parking)

    @classmethod
    def load(cls, module, loader):
        parameters, projections = {}, {}
        declarations = loader.quantization
        for path, _ in module.named_parameters():
            prefix = "model.visual." + path.removesuffix(".weight")
            if path.endswith(".weight") and prefix in declarations:
                projections[path.removesuffix(".weight")] = loader.load_linear(prefix)
            else:
                parameters[path] = loader.load_tensor("model.visual." + path)
        # Patch rows are ordered [merge_h, merge_w, channel]. Conv2d stores
        # weights as [out, channel, merge_h, merge_w]; transpose once at load.
        downsample = parameters["downsample.weight"].permute(0, 2, 3, 1).reshape(4096, 4096).contiguous()
        return cls(parameters, projections, downsample)


def _byte_view(tensor, nbytes):
    """Flat uint8 view of the first ``nbytes`` of a contiguous tensor's bytes."""
    if not tensor.is_contiguous() or tensor.nbytes < nbytes:
        raise ValueError("aliased staging storage must be contiguous and large enough")
    start = tensor.storage_offset() * tensor.element_size()
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage(), start, (nbytes,))


class VisionStaging:
    """VRAM slots that every GLM vision graph of one rank reads its staged weights from.

    Two block slots double-buffer the tower; the tail slot aliases ``scratch.gate`` at full patch
    capacity, which no stage reads after the last block (the downsample and merger use their own
    scratch). Graphs replay serially, so all patch capacities share these slots.
    """

    def __init__(self, weights, scratch, *, stream=None, kernel_copy=False):
        parking = weights.parking
        if parking is None:
            raise ValueError("GLM vision staging requires host-parked weights (VisionWeights.park)")
        options = dict(stream=stream, kernel_copy=kernel_copy)
        self.block_sources = tuple({"block": flat} for flat in parking.block_sources)
        self.tail_sources = {"tail": parking.tail_source}
        self.blocks = WeightStaging(self.block_sources[0], slots=2, **options)
        if not all(self.blocks.matches(sources) for sources in self.block_sources):
            raise ValueError("GLM vision staging requires every block to share one packed size")
        self.tail_aliased = scratch.gate.nbytes >= parking.tail_source.nbytes
        tail = ({"slots": 1} if not self.tail_aliased else
                {"storage": ({"tail": _byte_view(scratch.gate, parking.tail_source.nbytes)},)})
        self.tail = WeightStaging(self.tail_sources, token=self.blocks.token, **tail, **options)
        # The parked bytes are the checkpoint storage the package stores and the slots are filled from.
        for flat in (*parking.block_sources, parking.tail_source):
            record_numeric_tensor(flat)
        self.block_trees = tuple(parking.block_layout.bind(parking.block_tree, self.blocks.slot(i)["block"])
                                 for i in range(2))
        self.tail_tree = parking.tail_layout.bind(parking.tail_tree, self.tail.slot(0)["tail"])

    @property
    def allocated_bytes(self):
        """VRAM the slots add: the two block slots, and the tail slot unless it aliases scratch."""
        return self.blocks.nbytes + (0 if self.tail_aliased else self.tail.nbytes)

    def bound(self, weights):
        """Parameters, projections and downsample weight with block and tail weights read from the slots."""
        parameters = {k: v for k, v in weights.parameters.items() if k.startswith(RESIDENT_PREFIXES)}
        projections = {}
        for index in range(VISION_BLOCKS):
            tree = self.block_trees[index % 2]
            parameters.update({f"blocks.{index}.{k}": v for k, v in tree["parameters"].items()})
            projections.update({f"blocks.{index}.{k}": v for k, v in tree["projections"].items()})
        parameters.update(self.tail_tree["parameters"])
        projections.update(self.tail_tree["projections"])
        if set(parameters) != set(weights.parameters) or set(projections) != set(weights.projections):
            raise ValueError("staged GLM vision weights do not cover the parked weights")
        return parameters, projections, self.tail_tree["downsample"]


def staged_blocks(blocks, hidden, run, staging):
    """Run ``hidden = run(block, hidden)`` over the tower, staging block ``i + 1`` while block ``i`` runs.

    Block ``i`` reads slot ``i % 2``. Its join follows the previous block's output (``hidden``); the
    next block's copies fork only after that join, when the slot's previous reader (block ``i - 1``)
    has been enqueued."""
    if staging is not None:
        staging.blocks.stage(0, staging.block_sources[0])
    for index, block in enumerate(blocks):
        if staging is not None:
            staging.blocks.join(index, hidden)
            if index + 1 < len(blocks):
                staging.blocks.stage(index + 1, staging.block_sources[index + 1])
        hidden = run(block, hidden)
    return hidden


def _rowwise(function, source, out, bands=ROW_BANDS):
    """``out[rows] = function(source[rows])`` one row band at a time: the same per-row math (and the same
    BF16 results) as one call, with the function's temporaries sized for one band."""
    rows = source.shape[0]
    step = max(1, -(-rows // bands))
    for start in range(0, rows, step):
        out[start:start + step].copy_(function(source[start:start + step]))
    return out


def _swiglu_(gate, up, limit):
    """``silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)`` in place in ``gate`` (HF's clamped SwiGLU,
    op for op, without its three gate-sized temporaries)."""
    gate.clamp_(max=limit)
    up.clamp_(min=-limit, max=limit)
    return torch.nn.functional.silu(gate, inplace=True).mul_(up)


class PreparedVision:
    """Encode packed patches with a fixed capacity and device-resident lengths.

    The HF module supplies the canonical RMS normalization, Q/K normalization, rotary positions and
    LayerNorm/GELU; this class runs the block and merger dataflow itself so that every intermediate lands
    in VisionScratch: residual adds and the clamped SwiGLU run in place, and the FP32 normalization and
    rotary temporaries are bounded to one row band. Results equal the HF forward bit for bit in eager mode.
    Prepared projections retain checkpoint precision. Each capacity owns a canonical module tree;
    weights and sequential scratch may be shared across capacities.
    """

    def __init__(self, module, scratch, metadata, weights, *, session, name="vision",
                 projection_workspace=None, attention_workspace=None, staging=None):
        config = module.config
        expected = dict(hidden_size=1024, num_heads=16, intermediate_size=4096, depth=24,
                        out_hidden_size=4096, projection_intermediate_size=10240,
                        patch_size=14, temporal_patch_size=2, spatial_merge_size=2, in_channels=3,
                        hidden_act="silu")
        if any(getattr(config, key) != value for key, value in expected.items()):
            raise ValueError("GLM vision geometry differs from the checkpoint contract")
        capacity, device = scratch.hidden.shape[0], scratch.hidden.device
        if metadata.position_ids.shape != (capacity, 2):
            raise ValueError("GLM vision metadata and scratch capacities differ")
        if metadata.num_patches.item() or bool(metadata.cu_seqlens.any().item()):
            raise ValueError("GLM vision preparation requires inactive metadata")
        self.module, self.scratch, self.metadata, self.staging = module, scratch, metadata, staging
        self.limit = config.swiglu_limit
        parameters, projections, downsample = (weights.parameters, weights.projections, weights.downsample) \
            if staging is None else staging.bound(weights)
        for path, tensor in parameters.items():
            parent, _, attribute = path.rpartition(".")
            setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(tensor, requires_grad=False))
        self.patch_embed = UnquantizedProjection(scratch.patches, module.patch_embed.proj.weight.flatten(1),
            scratch.hidden, session=session, name=f"{name}.patch_embed")
        self.patch_bias = module.patch_embed.proj.bias
        module.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(config).to(device)
        # Staged kernels read slot tensors: scratch rewritten on every replay, not checkpoint storage, so
        # they are not recorded as numeric storage (VisionStaging recorded the parked sources).
        with unrecorded_numeric_tensors() if staging is not None else nullcontext():
            self._prepare_stages(module, scratch, metadata, projections, downsample, session=session, name=name,
                                 projection_workspace=projection_workspace, attention_workspace=attention_workspace)
        self.output_rows = torch.arange(capacity // 4, device=device)

    def _prepare_stages(self, module, scratch, metadata, projections, downsample, *, session, name,
                        projection_workspace, attention_workspace):
        s = scratch

        def project(key, source, output, **options):
            original = module.get_submodule(key)
            return QuantizedProjection(source, projections[key], output, session=session, name=f"{name}.{key}",
                                       bias=original.bias, workspace=projection_workspace, **options)

        self.blocks = []
        for index in range(len(module.blocks)):
            prefix = f"blocks.{index}"
            self.blocks.append(SimpleNamespace(
                qkv=project(f"{prefix}.attn.qkv", s.projected, s.qkv),
                proj=project(f"{prefix}.attn.proj", s.attention, s.projected),
                gate=project(f"{prefix}.mlp.gate_proj", s.projected, s.gate, activation_mode="auto"),
                up=project(f"{prefix}.mlp.up_proj", s.projected, s.up, activation_mode="auto"),
                down=project(f"{prefix}.mlp.down_proj", s.gate, s.projected, activation_mode="auto")))
        self.attention = PreparedVarlenAttention(s.query, s.key, s.value, metadata.cu_seqlens,
            s.attention.view_as(s.query), session=session, workspace=attention_workspace, name=name)
        self.downsample = UnquantizedProjection(s.projected.view(-1, 4096), downsample, s.merged,
                                                session=session, name=f"{name}.downsample")
        self.downsample_bias = module.downsample.bias
        self.merger = SimpleNamespace(
            proj=project("merger.proj", s.merged, s.output, activation_mode="auto"),
            gate=project("merger.gate_proj", s.merged, s.merger_gate, activation_mode="auto"),
            up=project("merger.up_proj", s.merged, s.merger_up, activation_mode="auto"),
            down=project("merger.down_proj", s.merger_gate, s.output, activation_mode="auto"))

    def _block(self, index, hidden, rope):
        s, block, run = self.scratch, self.module.blocks[index], self.blocks[index]
        cos, sin = rope
        _rowwise(block.norm1, hidden, s.projected)
        qkv = run.qkv(s.projected).view(-1, 3, 16, 64)
        rows = qkv.shape[0]
        step = max(1, -(-rows // ROW_BANDS))
        for start in range(0, rows, step):
            band = slice(start, start + step)
            query, key = apply_rotary_pos_emb_vision(block.attn.q_norm(qkv[band, 0]),
                                                     block.attn.k_norm(qkv[band, 1]), cos[band], sin[band])
            s.query[band].copy_(query)
            s.key[band].copy_(key)
        s.value.copy_(qkv[:, 2])
        self.attention()
        hidden.add_(run.proj(s.attention))
        _rowwise(block.norm2, hidden, s.projected)
        gate = _swiglu_(run.gate(s.projected), run.up(s.projected), self.limit)
        return hidden.add_(run.down(gate))

    def __call__(self, pixels):
        module, scratch, metadata = self.module, self.scratch, self.metadata
        scratch.patches.copy_(pixels)
        hidden = self.patch_embed(scratch.patches)
        if self.patch_bias is not None:
            hidden.add_(self.patch_bias)
        rope = module.rotary_pos_emb(hidden, metadata.position_ids)
        staging = self.staging
        hidden = staged_blocks(range(len(self.blocks)), hidden,
                               lambda index, value: self._block(index, value, rope), staging)
        if staging is not None:
            # The tail slot aliases the gate scratch: fork only after the last block has been enqueued.
            staging.tail.stage(0, staging.tail_sources)
        normed = _rowwise(module.post_layernorm, hidden, scratch.projected)
        if staging is not None:
            staging.tail.join(0, normed)
        merged = self.downsample(normed.view(-1, 4096))
        if self.downsample_bias is not None:
            merged.add_(self.downsample_bias)
        merger, m = module.merger, self.merger
        output = m.proj(merged)
        _rowwise(lambda rows: merger.act1(merger.post_projection_norm(rows)), output, scratch.merged)
        gate = _swiglu_(m.gate(scratch.merged), m.up(scratch.merged), merger.swiglu_limit)
        output = m.down(gate)
        output.masked_fill_((self.output_rows >= metadata.num_patches // 4)[:, None], 0)
        return output

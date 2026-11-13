# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fork-only GLM vision VRAM layout: lifetime-phased VisionScratch and the in-place tower dataflow.

The vision package measured +632 MiB of shared CUDA graph pool and +327 MiB of scratch beyond the prefill
overlay per rank (analysis/glm-vision-vram-20261112.md). These CPU checks pin the fixes:

- VisionScratch fields that are live together never overlap, the merger fields stay clear of the gate
  scratch the staged tail weights alias, and the whole scratch is 13312 BF16 elements per patch row;
- PreparedVision (in-place residuals and SwiGLU, row-banded FP32 norm/rotary temporaries, every
  intermediate in VisionScratch) equals the HF block/merger dataflow it replaced bit for bit (CPU eager,
  dense stand-ins for the b12x projections and varlen attention);
- its eager transient peak is a small fraction of the HF dataflow's.

    CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q tests/recipes/test_glm53_vision_lean.py
"""

from __future__ import annotations

import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ignition.recipes.glm53 import vision as glm_vision
from ignition.recipes.glm53.vision import SCRATCH_WIDTH, PreparedVision, VisionMetadata, VisionScratch, VisionWeights

from tests.recipes.test_glm53_vision_staging import VISION_CONFIG


PHASES = {
    "patch": ("patches",),
    "attention": ("qkv", "query", "key", "value", "attention"),
    "feed-forward": ("gate", "up"),
    "merger": ("merged", "merger_gate", "merger_up", "output"),
}
PERSISTENT = ("hidden", "projected")


def _span(tensor):
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def _overlap(a, b):
    (a0, a1), (b0, b1) = _span(a), _span(b)
    return a0 < b1 and b0 < a1


@pytest.mark.parametrize("patches", [64, 1024, 32768])
def test_scratch_fields_live_together_never_overlap(patches):
    scratch = VisionScratch.allocate(patches, "meta" if patches > 1024 else "cpu")
    if scratch.hidden.device.type == "meta":
        # Meta tensors have no addresses: check the element offsets instead.
        scratch = VisionScratch.allocate(1024, "cpu")
    fields = {name: getattr(scratch, name) for name in scratch.__dataclass_fields__}
    flat = fields["hidden"].untyped_storage()
    assert all(t.untyped_storage().data_ptr() == flat.data_ptr() for t in fields.values())
    rows = fields["hidden"].shape[0]
    assert flat.nbytes() == rows * SCRATCH_WIDTH * 2
    for names in (*PHASES.values(), PERSISTENT):
        group = [*names, *PERSISTENT]
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a != b:
                    assert not _overlap(fields[a], fields[b]), (a, b)
    # The staged tail weights alias `gate` after the last block: the merger fields must not touch it.
    for name in PHASES["merger"]:
        assert not _overlap(fields[name], fields["gate"]), name
    # Smaller capacities view row prefixes of every field, inside the same regions.
    small = scratch.rows(64)
    for name in fields:
        assert getattr(small, name).data_ptr() == fields[name].data_ptr()


def test_scratch_is_smaller_than_the_prefill_overlay():
    # 32768 patches: 832 MiB (was 1609.5 MiB); the TP3/DCP3 prefill-4096 overlay is 1282.6 MiB, TP2/DCP2 1330.4.
    assert 32768 * SCRATCH_WIDTH * 2 == 832 << 20
    assert 32768 * SCRATCH_WIDTH * 2 < 1282 << 20


# Dense stand-ins for the b12x kernels (same output-buffer semantics) ----------------------------------------

class _Quantized:
    def __init__(self, source, weight, output, *, session=None, name="", bias=None, workspace=None, **options):
        self.weight, self.output, self.bias = weight, output, bias

    def __call__(self, source):
        torch.matmul(source, self.weight.t(), out=self.output)
        if self.bias is not None:
            self.output.add_(self.bias)
        return self.output


class _Unquantized:
    def __init__(self, source, weight, output, *, session=None, name="", bias=None):
        self.weight, self.output = weight, output

    def __call__(self, source):
        return torch.matmul(source, self.weight.t(), out=self.output)


def _varlen(q, k, v, cu_seqlens):
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds, bounds[1:]):
        if stop > start:
            out[start:stop] = torch.nn.functional.scaled_dot_product_attention(
                q[start:stop].transpose(0, 1), k[start:stop].transpose(0, 1), v[start:stop].transpose(0, 1),
                scale=64 ** -0.5).transpose(0, 1)
    return out


class _Attention:
    def __init__(self, q, k, v, cu_seqlens, out, *, session=None, workspace=None, name=""):
        self.q, self.k, self.v, self.cu, self.out = q, k, v, cu_seqlens, out

    def __call__(self):
        self.out.copy_(_varlen(self.q, self.k, self.v, self.cu))
        return self.out


def _weights():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    generator = torch.Generator().manual_seed(0)
    with torch.device("meta"):
        module = Glm5NextVisionModel(Glm5NextVisionConfig(**VISION_CONFIG))
    quantized = {f"blocks.{i}.{p}" for i in range(24) for p in
                 ("attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    quantized |= {f"merger.{p}" for p in ("proj", "gate_proj", "up_proj", "down_proj")}
    parameters, projections = {}, {}
    for path, parameter in module.named_parameters():
        prefix = path.removesuffix(".weight")
        values = torch.randn(parameter.shape, generator=generator)
        if path.endswith(".weight") and prefix in quantized:
            projections[prefix] = (values * parameter.shape[1] ** -0.5).to(torch.bfloat16)
        elif "norm" in path and path.endswith(".weight"):
            parameters[path] = (1 + 0.1 * values).to(torch.bfloat16)
        else:
            parameters[path] = (values * 0.05).to(torch.bfloat16)
    parameters["patch_embed.proj.weight"] = (parameters["patch_embed.proj.weight"] * 0.5)
    downsample = parameters["downsample.weight"].permute(0, 2, 3, 1).reshape(4096, 4096).contiguous() * 0.1
    return VisionWeights(parameters, projections, downsample)


def _encoder(monkeypatch, weights, patches):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    monkeypatch.setattr(glm_vision, "QuantizedProjection", _Quantized)
    monkeypatch.setattr(glm_vision, "UnquantizedProjection", _Unquantized)
    monkeypatch.setattr(glm_vision, "PreparedVarlenAttention", _Attention)
    with torch.device("meta"):
        module = Glm5NextVisionModel(Glm5NextVisionConfig(**VISION_CONFIG))
    module.eval().requires_grad_(False)
    scratch = VisionScratch.allocate(patches, "cpu")
    metadata = VisionMetadata.allocate(patches, 1, "cpu")
    return PreparedVision(module, scratch, metadata, weights, session=None)


def _reference(encoder, weights, pixels, positions, count):
    """The replaced dataflow: HF block, MLP and merger forwards, op for op, out of place."""
    from transformers.models.glm5_next.modeling_glm5_next import apply_rotary_pos_emb_vision

    module, p, w = encoder.module, weights.parameters, weights.projections
    act = torch.nn.functional.silu

    def linear(key, x):
        out = torch.matmul(x, w[key].t())
        bias = p.get(f"{key}.bias")
        return out.add_(bias) if bias is not None else out

    def swiglu(prefix, x, limit):
        gate = linear(f"{prefix}.gate_proj", x)
        up = linear(f"{prefix}.up_proj", x)
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        return linear(f"{prefix}.down_proj", act(gate) * up)

    cu = torch.tensor([0, count], dtype=torch.int32)
    hidden = torch.matmul(pixels.to(torch.bfloat16), p["patch_embed.proj.weight"].flatten(1).t())
    hidden.add_(p["patch_embed.proj.bias"])
    cos, sin = module.rotary_pos_emb(hidden, positions)
    for index, block in enumerate(module.blocks):
        prefix = f"blocks.{index}"
        projected = linear(f"{prefix}.attn.qkv", block.norm1(hidden)).view(-1, 3, 16, 64)
        query, key = apply_rotary_pos_emb_vision(block.attn.q_norm(projected[:, 0]),
                                                 block.attn.k_norm(projected[:, 1]), cos, sin)
        attention = _varlen(query, key, projected[:, 2].contiguous(), cu).reshape(-1, 1024)
        hidden = hidden + linear(f"{prefix}.attn.proj", attention)
        hidden = hidden + swiglu(f"{prefix}.mlp", block.norm2(hidden), block.mlp.swiglu_limit)
    hidden = module.post_layernorm(hidden)
    merged = torch.matmul(hidden.view(-1, 4096), weights.downsample.t()).add_(p["downsample.bias"])
    merger = module.merger
    output = merger.act1(merger.post_projection_norm(linear("merger.proj", merged)))
    output = swiglu("merger", output, merger.swiglu_limit)
    return output.masked_fill((torch.arange(output.shape[0]) >= count // 4)[:, None], 0)


def _inputs(patches, count, seed=1):
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randn((patches, 1176), generator=generator)
    side = 8
    rows = torch.arange(patches) // side
    columns = torch.arange(patches) % side
    positions = torch.stack((rows, columns), dim=1).to(torch.int64)
    live = torch.arange(patches) < count
    return pixels * live[:, None], positions * live[:, None]


@pytest.fixture(scope="module")
def weights():
    return _weights()


@pytest.mark.parametrize("count", [256, 96])
def test_in_place_tower_matches_the_hf_dataflow_bit_for_bit(monkeypatch, weights, count):
    patches = 256
    encoder = _encoder(monkeypatch, weights, patches)
    pixels, positions = _inputs(patches, count)
    encoder.metadata.position_ids.copy_(positions)
    encoder.metadata.cu_seqlens.copy_(torch.tensor([0, count], dtype=torch.int32))
    encoder.metadata.num_patches.fill_(count)
    with torch.no_grad():
        expected = _reference(encoder, weights, pixels, positions, count)
        encoder.scratch.patches.copy_(pixels)
        actual = encoder(encoder.scratch.patches)
    assert actual.data_ptr() == encoder.scratch.output.data_ptr()
    assert torch.isfinite(expected).all() and expected[: count // 4].abs().max() > 0
    assert torch.equal(actual, expected)


class _LiveBytes(TorchDispatchMode):
    """Peak bytes of tensors created inside the mode and still referenced (CPU eager stand-in for the graph
    pool's live set). Views of a counted storage are not counted again."""

    def __init__(self, exclude):
        super().__init__()
        self.exclude, self.counted, self.live, self.peak = set(exclude), [], 0, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        known = self.exclude | {key for _, _, key in self.counted}
        for tensor in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(tensor, torch.Tensor):
                key = tensor.untyped_storage().data_ptr()
                if key not in known and tensor.untyped_storage().nbytes():
                    known.add(key)
                    self.counted.append((weakref.ref(tensor), tensor.untyped_storage().nbytes(), key))
                    self.live += self.counted[-1][1]
        alive = []
        for ref, nbytes, key in self.counted:
            if ref() is None:
                self.live -= nbytes
            else:
                alive.append((ref, nbytes, key))
        self.counted = alive
        self.peak = max(self.peak, self.live)
        return out


def test_in_place_tower_keeps_eager_transients_small(monkeypatch, weights):
    patches = 1024
    encoder = _encoder(monkeypatch, weights, patches)
    pixels, positions = _inputs(patches, patches)
    encoder.metadata.position_ids.copy_(positions)
    encoder.metadata.cu_seqlens.copy_(torch.tensor([0, patches], dtype=torch.int32))
    encoder.metadata.num_patches.fill_(patches)
    encoder.scratch.patches.copy_(pixels)
    owned = [*weights.parameters.values(), *weights.projections.values(), weights.downsample,
             encoder.scratch.hidden, pixels, positions,
             *(t for t in vars(encoder.metadata).values() if isinstance(t, torch.Tensor)),
             encoder.output_rows, *(b for b in encoder.module.buffers())]
    exclude = {t.untyped_storage().data_ptr() for t in owned}
    with torch.no_grad():
        lean = _LiveBytes(exclude)
        with lean:
            encoder(encoder.scratch.patches)
        hf = _LiveBytes(exclude)
        with hf:
            _reference(encoder, weights, pixels, positions, patches)
    # Scaled to 32768 patches: HF dataflow ~1.2 GiB (four live 256 MiB SwiGLU tensors), lean ~0.16 GiB.
    scale = 32768 / patches
    assert hf.peak * scale > 1100 << 20
    print(f"eager transient peak at 32768 patches: HF {hf.peak * scale / 2**20:.0f} MiB, lean {lean.peak * scale / 2**20:.0f} MiB")
    assert lean.peak * scale < 192 << 20

# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Zero-padded GLM weight partitions for three ranks; exact partitions for one, two and four.

A synthetic checkpoint carries the real tensor names, formats and partitioned extents (8192 KDA
channels, 64 heads, 2048 expert columns, 154880 vocabulary rows) with small unpartitioned widths,
once with csf_32x2_v1 routed-expert scales (FP4-Local) and once with the legacy `csf` slab encoding
(LIL's CSF-QAD revisions such as dec48abd: `weight_scale.nvfp4_csf_fixed` + `nvfp4_csf_exceptions`).
The loader runs on the CPU through a session double, so b12x's CUDA-only packers are stubbed to
return the bytes they were handed.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from ignition.components.block_fp8 import BlockFP8Weight
from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import GroupedLinearWeights
from ignition.preparation import provenance
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.glm53.bootstrap import PreparationConfig
from ignition.recipes.glm53.shards import Shards
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

HIDDEN = 256  # stands in for the 4096-wide unpartitioned axes of dense projections
EXPERT_HIDDEN = 4096  # the recipe fixes the routed experts' hidden width
KDA = f"{TEXT_PREFIX}.layers.0.self_attn"
DENSE = f"{TEXT_PREFIX}.layers.0.mlp"
MLA = f"{TEXT_PREFIX}.layers.3.self_attn"
MOE = f"{TEXT_PREFIX}.layers.3.mlp"
DRAFT_MOE = f"{TEXT_PREFIX}.layers.45.mlp"  # plain W4A16 NVFP4 experts (no CSF), as the checkpoint declares
EXPERTS = (0, 1)


def _bytes(*shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8)


def _e4m3(*shape):
    return _bytes(*shape).view(torch.float8_e4m3fn)


def _bf16(*shape):
    return torch.randn(shape, dtype=torch.bfloat16)


def _csf(rows, columns, encoding="csf_32x2_v1"):
    if encoding == "csf":
        # Slab encoding: [rows / 16, 16 bases + 8 * columns code bytes] and u32 exception words.
        return {"nvfp4_csf_fixed": _bytes(rows // 16, 16 * (1 + columns // 2)),
                "nvfp4_csf_exceptions": torch.randint(0, 1 << 24, (5,), dtype=torch.int64).to(torch.uint32)}
    groups = (rows + 127) // 128 * ((columns + 3) // 4)
    pages = (groups + 31) // 32
    return {"csf32_bases": _bytes(rows), "csf32_codes": _bytes(rows * columns // 2),
            "csf32_page_bases": torch.randint(0, 1 << 16, (pages,), dtype=torch.int32).to(torch.uint32),
            "csf32_offsets": torch.randint(0, 1 << 15, (rows // 32 * ((columns + 1) // 2) + pages,),
                                           dtype=torch.int32).to(torch.uint16),
            "csf32_exceptions": torch.randint(0, 1 << 15, (8,), dtype=torch.int32).to(torch.uint16)}


def _tensors(encoding="csf_32x2_v1"):
    t = {}
    for name in ("q_proj", "k_proj", "v_proj"):
        t[f"{KDA}.{name}.weight"], t[f"{KDA}.{name}.weight_scale"] = _e4m3(8192, HIDDEN), _bytes(8192, HIDDEN // 32)
        t[f"{KDA}.{name[0]}_conv1d.weight"] = _bf16(8192, 1, 4)
    t[f"{KDA}.b_proj.weight"], t[f"{KDA}.f_a_proj.weight"], t[f"{KDA}.g_a_proj.weight"] = (
        _bf16(64, HIDDEN), _bf16(128, HIDDEN), _bf16(128, HIDDEN))
    t[f"{KDA}.f_b_proj.weight"], t[f"{KDA}.g_b_proj.weight"] = _bf16(8192, 128), _bf16(8192, 128)
    t[f"{KDA}.o_proj.weight"], t[f"{KDA}.o_proj.weight_scale"] = _e4m3(HIDDEN, 8192), _bytes(HIDDEN, 8192 // 32)
    t[f"{KDA}.A_log"], t[f"{KDA}.dt_bias"] = torch.randn(64), torch.randn(8192)
    t[f"{KDA}.o_norm.weight"] = _bf16(128)
    for name in ("gate_proj", "up_proj"):
        t[f"{DENSE}.{name}.weight"], t[f"{DENSE}.{name}.weight_scale"] = _e4m3(12288, HIDDEN), _bytes(12288, HIDDEN // 32)
    t[f"{DENSE}.down_proj.weight"], t[f"{DENSE}.down_proj.weight_scale"] = _e4m3(HIDDEN, 12288), _bytes(HIDDEN, 12288 // 32)
    t[f"{MLA}.q_a_proj.weight"], t[f"{MLA}.q_a_proj.weight_scale"] = _e4m3(1536, HIDDEN), torch.rand(12, 1, HIDDEN // 128, 1)
    t[f"{MLA}.kv_a_proj_with_mqa.weight"] = _e4m3(512, HIDDEN)
    t[f"{MLA}.kv_a_proj_with_mqa.weight_scale"] = torch.rand(4, 1, HIDDEN // 128, 1)
    t[f"{MLA}.indexer.wk.weight"], t[f"{MLA}.indexer.weights_proj.weight"] = _bf16(128, HIDDEN), _bf16(32, HIDDEN)
    t[f"{MLA}.indexer.wq_b.weight"] = _bf16(64, 48)
    t[f"{MLA}.q_b_proj.weight"], t[f"{MLA}.q_b_proj.weight_scale"] = _e4m3(16384, 128), torch.rand(128, 1, 1, 1)
    t[f"{MLA}.kv_b_proj.weight"] = _bf16(32768, 64)
    t[f"{MLA}.o_proj.weight"], t[f"{MLA}.o_proj.weight_scale"] = _e4m3(HIDDEN, 16384), torch.rand(HIDDEN // 128, 1, 128, 1)
    for name, shape in (("q_a_layernorm.weight", (1536,)), ("kv_a_layernorm.weight", (512,)),
                        ("indexer.k_norm.weight", (128,)), ("indexer.k_norm.bias", (128,)),
                        ("indexer.index_kpool_compress_ape", (4, 128)), ("indexer.index_kpool_compress_gate", (128, HIDDEN))):
        t[f"{MLA}.{name}"] = _bf16(*shape)
    t[f"{MOE}.gate.weight"], t[f"{MOE}.gate.e_score_correction_bias"] = _bf16(288, HIDDEN), torch.randn(288)
    for name in ("gate_proj", "up_proj"):
        t[f"{MOE}.shared_experts.{name}.weight"] = _e4m3(2048, HIDDEN)
        t[f"{MOE}.shared_experts.{name}.weight_scale"] = torch.rand(16, 1, HIDDEN // 128, 1)
    t[f"{MOE}.shared_experts.down_proj.weight"] = _e4m3(HIDDEN, 2048)
    t[f"{MOE}.shared_experts.down_proj.weight_scale"] = torch.rand(HIDDEN // 128, 1, 16, 1)
    for expert in EXPERTS:
        prefix = f"{MOE}.experts.{expert}"
        for name in ("gate_proj", "up_proj", "down_proj"):
            t[f"{prefix}.{name}.weight"] = (_bytes(2048, EXPERT_HIDDEN // 2) if name != "down_proj"
                                            else _bytes(EXPERT_HIDDEN, 1024))
            t[f"{prefix}.{name}.weight_scale_2"] = torch.tensor([2.0 if name != "down_proj" else 3.0])
            t[f"{prefix}.{name}.input_scale"] = torch.tensor([0.5 + expert])
            for suffix, value in _csf(2048 if name != "down_proj" else EXPERT_HIDDEN,
                                      EXPERT_HIDDEN // 16 if name != "down_proj" else 128, encoding).items():
                t[f"{prefix}.{name}.weight_scale.{suffix}"] = value
    for expert in EXPERTS:
        prefix = f"{DRAFT_MOE}.experts.{expert}"
        for name in ("gate_proj", "up_proj", "down_proj"):
            down = name == "down_proj"
            t[f"{prefix}.{name}.weight"] = _bytes(EXPERT_HIDDEN, 1024) if down else _bytes(2048, EXPERT_HIDDEN // 2)
            t[f"{prefix}.{name}.weight_scale"] = _e4m3(EXPERT_HIDDEN, 128) if down else _e4m3(2048, EXPERT_HIDDEN // 16)
            t[f"{prefix}.{name}.weight_scale_2"] = torch.tensor([1.5 if down else 2.5])
    t["lm_head.weight"], t["lm_head.weight_scale"] = _e4m3(154880, 32), _bytes(154880, 1)
    return {name: value.contiguous() for name, value in t.items()}


def _config(encoding="csf_32x2_v1"):
    text = {"hidden_size": 4096, "num_hidden_layers": 45, "intermediate_size": 12288, "moe_intermediate_size": 2048,
            "n_routed_experts": 288, "n_shared_experts": 1, "first_k_dense_replace": 3, "num_attention_heads": 64,
            "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 256, "qk_rope_head_dim": 0,
            "v_head_dim": 256, "linear_num_heads": 64, "linear_head_dim": 128, "linear_conv_kernel_dim": 4,
            "index_n_heads": 32, "index_head_dim": 128, "swiglu_limit": 10.0, "hc_mult": 4}
    mxfp8 = {"quant_algo": "MXFP8", "weight_block_size": [1, 32]}
    block = {"quant_algo": "FP8_PB_WO", "weight_block_size": [128, 128]}
    layers = {"lm_head": mxfp8, f"{MOE}.experts": {"group_size": 16, "quant_algo": "NVFP4",
                                                    "weight_scale_encoding": encoding},
              f"{DRAFT_MOE}.experts": {"group_size": 16, "quant_algo": "W4A16_NVFP4"}}
    layers.update({f"{KDA}.{name}": mxfp8 for name in ("q_proj", "k_proj", "v_proj", "o_proj")})
    layers.update({f"{DENSE}.{name}": mxfp8 for name in ("gate_proj", "up_proj", "down_proj")})
    layers.update({f"{MLA}.{name}": block for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")})
    layers.update({f"{MOE}.shared_experts.{name}": block for name in ("gate_proj", "up_proj", "down_proj")})
    return {"model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"], "text_config": text,
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                    "quantized_layers": layers}}


ENCODINGS = ("csf_32x2_v1", "csf")


@pytest.fixture(scope="module", params=ENCODINGS)
def checkpoint(tmp_path_factory, request):
    torch.manual_seed(0)
    snapshot = tmp_path_factory.mktemp(f"glm53-fake-{request.param}")
    tensors = _tensors(request.param)
    save_file(tensors, str(snapshot / "model.safetensors"))
    (snapshot / "config.json").write_text(json.dumps(_config(request.param)))
    return inspect_indexed_checkpoint(snapshot), tensors


class _Session:
    payload_bytes = 0

    def flush(self):
        pass

    def concatenate(self, destination, sources):
        destination.view(-1).copy_(torch.cat([source.reshape(-1) for source in sources]))


class CpuLoader(GlmWeightLoader):
    """The GLM loader over host tensors: the direct-transfer session copies through torch."""

    def __enter__(self):
        self._sources = load_file(str(self.manifest.snapshot / "model.safetensors"))
        self._session = _Session()
        self.device = torch.device(self.device)
        return self

    def __exit__(self, *exc):
        self._session, self._sources = None, {}

    def _copy(self, destination, source, *, numeric=True):
        destination.copy_(source)


@pytest.fixture
def loader_factory(checkpoint, monkeypatch):
    import b12x.gemm.blockscaled
    import b12x.moe.fused_moe as moe

    def pack_weight(values, scales=None, *, recipe):
        return SimpleNamespace(values=values, scale_rows=scales, recipe=recipe,
                               out_features=values.shape[0], in_features=values.shape[1])

    captured = []
    monkeypatch.setattr(b12x.gemm.blockscaled, "pack_weight", pack_weight)
    monkeypatch.setattr(moe, "plan_weights", lambda **kwargs: SimpleNamespace(**kwargs))

    def prepare_weights(*, plan, weights):
        captured.append((plan, weights))
        packed = weights.packed if hasattr(weights, "packed") else weights
        # The owner shape the provenance recorder inspects: native w13/w2 banks without scale owners.
        impl = SimpleNamespace(w1_fp4=packed.w13, w2_fp4=packed.w2, w1_blockscale=None, w2_blockscale=None,
                               plan=SimpleNamespace(transforms=()))
        return SimpleNamespace(plan=plan, weights=weights, _impl=impl)
    monkeypatch.setattr(moe, "prepare_weights", lambda *, plan, weights, **kwargs: prepare_weights(plan=plan, weights=weights))
    manifest, _ = checkpoint

    def make(rank, world, expert_activation="a16", expert_columns=None):
        return CpuLoader(manifest, "cpu", shard=TensorParallelShard(rank, world),
                         expert_activation=expert_activation, expert_columns=expert_columns), captured
    return make


def _padded(source, rows=None, columns=None):
    """Checkpoint slice with zero rows/columns past the source; FP8 values as uint8 views."""
    r0, r1 = (0, source.shape[0]) if rows is None else rows
    c0, c1 = (0, 0) if columns is None else columns
    shape = [r1 - r0, *source.shape[1:]]
    if columns is not None:
        shape[1] = c1 - c0
    target = torch.zeros(shape, dtype=source.dtype)
    nr = min(r1, source.shape[0]) - r0
    nc = min(c1, source.shape[1]) - c0 if columns is not None else None
    if nr > 0 and (nc is None or nc > 0):
        if nc is None:
            target[:nr] = source[r0:r0 + nr]
        else:
            target[:nr, :nc] = source[r0:r0 + nr, c0:c0 + nc]
    return target.view(torch.uint8) if target.dtype == torch.float8_e4m3fn else target


def _neutral(scale, rows):
    """MXFP8 scale rows with the neutral UE8M0 127 (1.0) past the source."""
    target = torch.full((rows[1] - rows[0], scale.shape[1]), 127, dtype=torch.uint8)
    covered = min(rows[1], scale.shape[0]) - rows[0]
    target[:covered] = scale[rows[0]:rows[0] + covered]
    return target


def _same(actual, expected):
    actual = actual.view(torch.uint8) if actual.dtype == torch.float8_e4m3fn else actual
    expected = expected.view(torch.uint8) if expected.dtype == torch.float8_e4m3fn else expected
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert torch.equal(actual, expected)


def _encoding(tensors):
    return "csf_32x2_v1" if f"{MOE}.experts.0.up_proj.weight_scale.csf32_bases" in tensors else "csf"


def _check_csf_sources(scales13, scales2, tensors, first, covered, local):
    """Both CSF sources select the covered rectangle, pad to the rank's width, and carry the checkpoint planes."""
    encoding = _encoding(tensors)
    fields = (("csf32_bases", "csf32_codes", "csf32_page_bases", "csf32_offsets", "csf32_exceptions")
              if encoding == "csf_32x2_v1" else ("nvfp4_csf_fixed", "nvfp4_csf_exceptions"))
    assert scales13.encoding == scales2.encoding == encoding
    assert len(scales13.components) == len(scales2.components) == len(fields)
    assert scales13.row_range == (first, first + covered) and scales13.rows == 2 * local and scales13.join == 2
    assert scales13.source_rows == 2048 and scales13.source_columns == EXPERT_HIDDEN // 16
    assert scales2.column_range == (first // 16, (first + covered) // 16) and scales2.columns == local // 16
    assert scales2.source_rows == EXPERT_HIDDEN and scales2.source_columns == 128
    assert scales13.padding_base == scales2.padding_base == 0
    if covered == local:
        assert scales13.padded_rows is None and scales2.padded_columns is None
    else:
        assert scales13.padded_rows == local and scales2.padded_columns == local // 16
    for source in (scales13, scales2):
        projections = ([(e, p) for e in EXPERTS for p in ("up_proj", "gate_proj")] if source is scales13
                       else [(e, "down_proj") for e in EXPERTS])
        assert source.segments == len(projections)
        source.validate()
        records = []
        for plane, (expert, projection) in enumerate(projections):
            prefix = f"{MOE}.experts.{expert}.{projection}.weight_scale"
            for field, component in zip(fields[:-1], source.components[:-1], strict=True):
                _same(component[plane], tensors[f"{prefix}.{field}"])
            records.append(tensors[f"{prefix}.{fields[-1]}"])
        _same(source.components[-1], torch.cat(records))
        assert source.starts.tolist() == [0, *torch.tensor([r.numel() for r in records]).cumsum(0).tolist()]


def _check_rank(loader, captured, tensors, rank, world, expert_columns=None):
    shards = Shards.of(world, expert_columns=expert_columns)
    shard = TensorParallelShard(rank, world)
    channels = shards.head_channels(rank, 128)
    heads = shards.head_range(rank)
    with loader:
        kda = loader.load_attention(0)
        dense = loader.load_mlp(0)
        mla = loader.load_attention(3)
        moe = loader.load_mlp(3, scale_scratch=loader.allocate_expert_scale_scratch(len(EXPERTS)), expert_ids=EXPERTS)
        draft = loader.load_experts(45, expert_ids=EXPERTS)
        head = loader.load_linear("lm_head", rows=shard.padded_partition(154880, alignment=64), pad=True)
    # KDA: MXFP8 q/k/v rows and BF16 beta rows pad; f_a stays whole; scales pad with the neutral 127.
    assert isinstance(kda["input_proj"], GroupedLinearWeights)
    qkv, rest = kda["input_proj"].parts
    assert kda["input_proj"].widths == (3 * shards.kda_channels, shards.heads + 128)
    _same(qkv.values, torch.cat([_padded(tensors[f"{KDA}.{n}.weight"], channels) for n in ("q_proj", "k_proj", "v_proj")]))
    _same(qkv.scale_rows, torch.cat([_neutral(tensors[f"{KDA}.{n}.weight_scale"], channels)
                                     for n in ("q_proj", "k_proj", "v_proj")]))
    _same(rest, torch.cat([_padded(tensors[f"{KDA}.b_proj.weight"], heads), tensors[f"{KDA}.f_a_proj.weight"]]))
    for name in ("f_b_proj", "g_b_proj"):
        _same(kda[name], _padded(tensors[f"{KDA}.{name}.weight"], channels))
    _same(kda["o_proj"].values, _padded(tensors[f"{KDA}.o_proj.weight"], None, channels))
    assert kda["o_proj"].in_features == shards.kda_channels
    _same(kda["A_log"], _padded(tensors[f"{KDA}.A_log"], heads))
    _same(kda["dt_bias"], _padded(tensors[f"{KDA}.dt_bias"], channels))
    _same(kda["conv1d.weight"], torch.cat([_padded(tensors[f"{KDA}.{n}_conv1d.weight"], channels) for n in "qkv"]).squeeze(1))
    # Dense MLP divides for every supported rank count.
    extent = shard.padded_partition(12288, alignment=128)
    _same(dense.gate_up.values, torch.cat([_padded(tensors[f"{DENSE}.{n}.weight"], extent) for n in ("gate_proj", "up_proj")]))
    _same(dense.down.values, _padded(tensors[f"{DENSE}.down_proj.weight"], None, extent))
    # MLA: block-FP8 query rows and output columns, BF16 absorbed key/value rows.
    head_channels = shards.head_channels(rank, 256)
    assert isinstance(mla["q_b_proj"], BlockFP8Weight) and mla["q_b_proj"].out_features == shards.heads * 256
    _same(mla["q_b_proj"].values, _padded(tensors[f"{MLA}.q_b_proj.weight"], head_channels))
    _same(mla["q_b_proj"].scales, _padded(tensors[f"{MLA}.q_b_proj.weight_scale"].reshape(128, 1),
                                          (head_channels[0] // 128, head_channels[1] // 128)))
    kv_rows = shards.head_channels(rank, 512)
    _same(mla["kv_b_proj"], _padded(tensors[f"{MLA}.kv_b_proj.weight"], kv_rows))
    recorded = provenance.lookup(mla["kv_b_proj"])
    assert recorded is not None and recorded["sources"][0]["tensor"] == f"{MLA}.kv_b_proj.weight"
    assert recorded["sources"][0]["ranges"] == ((kv_rows[0], min(kv_rows[1], 32768)), (0, 64))
    if kv_rows[1] > 32768:
        piece = recorded["sources"][0]["pieces"][0]
        assert piece == dict(source_offset=kv_rows[0] * 64 * 2, destination_offset=0, contiguous_bytes=128,
                             extents=(32768 - kv_rows[0],), source_strides=(128,), destination_strides=(128,))
    _same(mla["o_proj"].values, _padded(tensors[f"{MLA}.o_proj.weight"], None, head_channels))
    _same(mla["o_proj"].scales, _padded(tensors[f"{MLA}.o_proj.weight_scale"].reshape(HIDDEN // 128, 128), None,
                                        (head_channels[0] // 128, head_channels[1] // 128)))
    # Shared expert: block-FP8 gate/up rows and down columns padded to the 128-aligned expert shard.
    columns = shard.padded_partition(2048, alignment=128)
    _same(moe.shared.gate_up.values, torch.cat([_padded(tensors[f"{MOE}.shared_experts.{n}.weight"], columns)
                                                for n in ("gate_proj", "up_proj")]))
    _same(moe.shared.down.values, _padded(tensors[f"{MOE}.shared_experts.down_proj.weight"], None, columns))
    _same(moe.shared.down.scales, _padded(tensors[f"{MOE}.shared_experts.down_proj.weight_scale"].reshape(HIDDEN // 128, 16),
                                          None, (columns[0] // 128, columns[1] // 128)))
    # Routed experts: packed NVFP4 columns and CSF scale selections (the draft experts were captured last).
    # Their shard may be narrower than the shared expert's (704 vs 768 at three ranks, shards.py).
    plan, weights = captured[-2]
    first, last = shards.routed_columns(rank)
    covered = min(last, 2048) - first
    local = last - first
    assert plan.geometry.intermediate_size == local == shards.routed_width
    assert (first, last) == (rank * local, (rank + 1) * local) and covered % 32 == 0
    packed = weights.packed if hasattr(weights, "packed") else weights
    assert packed.w13.shape == (len(EXPERTS), 2 * local, EXPERT_HIDDEN // 2)
    assert packed.w2.shape == (len(EXPERTS), EXPERT_HIDDEN, local // 2)
    for index, expert in enumerate(EXPERTS):
        prefix = f"{MOE}.experts.{expert}"
        _same(packed.w13[index, :covered], tensors[f"{prefix}.up_proj.weight"][first:first + covered])
        _same(packed.w13[index, local:local + covered], tensors[f"{prefix}.gate_proj.weight"][first:first + covered])
        assert not packed.w13[index, covered:local].count_nonzero() and not packed.w13[index, local + covered:].count_nonzero()
        _same(packed.w2[index, :, :covered // 2], tensors[f"{prefix}.down_proj.weight"][:, first // 2:(first + covered) // 2])
        assert not packed.w2[index, :, covered // 2:].count_nonzero()
    # Provenance maps the covered columns as affine pieces; a padded owner maps only those.
    w13_sources, w2_sources = provenance.lookup(packed.w13)["sources"], provenance.lookup(packed.w2)["sources"]
    assert [s["tensor"] for s in w13_sources] == [f"{MOE}.experts.{e}.{p}.weight" for e in EXPERTS for p in ("up_proj", "gate_proj")]
    for source in w13_sources:
        assert source["ranges"] == ((first, first + covered), (0, EXPERT_HIDDEN // 2))
        assert source["pieces"][0]["extents"] == (covered,) and source["pieces"][0]["contiguous_bytes"] == EXPERT_HIDDEN // 2
    for source in w2_sources:
        assert source["ranges"] == ((0, EXPERT_HIDDEN), (first // 2, (first + covered) // 2))
        assert source["pieces"][0]["contiguous_bytes"] == covered // 2 and source["pieces"][0]["extents"] == (EXPERT_HIDDEN,)
    _check_csf_sources(weights.w13_scales, weights.w2_scales, tensors, first, covered, local)
    # Draft-layer experts: plain NVFP4 values and E4M3 block scales, both zero in the padded columns.
    from b12x._lib.intrinsics.reference import swizzle_block_scale

    draft_plan, draft_weights = captured[-1]
    assert draft_plan.geometry.intermediate_size == local and draft_plan.activation.mode == "a16"
    assert draft_weights.w13.shape == (len(EXPERTS), 2 * local, EXPERT_HIDDEN // 2)
    assert draft_weights.w2.shape == (len(EXPERTS), EXPERT_HIDDEN, local // 2)
    expected_s13, expected_s2 = [], []
    for index, expert in enumerate(EXPERTS):
        prefix = f"{DRAFT_MOE}.experts.{expert}"
        rows = lambda name: torch.cat((tensors[f"{prefix}.{name}"][first:first + covered],
                                       torch.zeros((local - covered, tensors[f"{prefix}.{name}"].shape[1]),
                                                   dtype=tensors[f"{prefix}.{name}"].dtype)))
        _same(draft_weights.w13[index], torch.cat((rows("up_proj.weight"), rows("gate_proj.weight"))))
        down = tensors[f"{prefix}.down_proj.weight"][:, first // 2:(first + covered) // 2]
        _same(draft_weights.w2[index, :, :covered // 2], down)
        assert not draft_weights.w2[index, :, covered // 2:].count_nonzero()
        expected_s13.append(torch.cat((rows("up_proj.weight_scale"), rows("gate_proj.weight_scale"))))
        down_scales = torch.zeros((EXPERT_HIDDEN, local // 16), dtype=torch.float8_e4m3fn)
        down_scales[:, :covered // 16] = tensors[f"{prefix}.down_proj.weight_scale"][:, first // 16:(first + covered) // 16]
        expected_s2.append(down_scales)
    _same(draft_weights.w13_block_scales, swizzle_block_scale(torch.stack(expected_s13)))
    _same(draft_weights.w2_block_scales, swizzle_block_scale(torch.stack(expected_s2)))
    assert torch.equal(draft_weights.w13_global_scales, torch.tensor([2.5, 2.5]))
    assert torch.equal(draft_weights.w2_global_scales, torch.tensor([1.5, 1.5]))
    # Vocabulary head: MXFP8 rows padded to the 64-aligned shard; the gather excludes the extra logits.
    rows = shard.padded_partition(154880, alignment=64)
    assert head.out_features == shards.vocabulary
    _same(head.values, _padded(tensors["lm_head.weight"], rows))
    _same(head.scale_rows, _neutral(tensors["lm_head.weight_scale"], rows))
    return kda, mla, moe, head


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 1), (4, 0), (4, 3)])
def test_dividing_rank_counts_load_exact_checkpoint_slices(loader_factory, checkpoint, world, rank):
    loader, captured = loader_factory(rank, world)
    _check_rank(loader, captured, checkpoint[1], rank, world)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_three_ranks_zero_fill_the_padded_tails(loader_factory, checkpoint, rank):
    loader, captured = loader_factory(rank, 3)
    kda, mla, moe, head = _check_rank(loader, captured, checkpoint[1], rank, 3)
    assert kda["A_log"].shape == (24,) and kda["dt_bias"].shape == (3072,) and kda["conv1d.weight"].shape == (3 * 3072, 4)
    assert mla["q_b_proj"].values.shape == (6144, 128) and mla["kv_b_proj"].shape == (12288, 64)
    assert moe.shared.gate_up.values.shape == (1536, HIDDEN) and moe.shared.down.values.shape == (HIDDEN, 768)
    assert head.values.shape == (51648, 32)
    if rank == 2:
        # The last rank holds heads 48..63 and 8 zero heads, expert columns 1536..2303, vocabulary rows past 154879.
        assert not kda["A_log"][16:].count_nonzero() and not kda["dt_bias"][2048:].count_nonzero()
        assert not kda["conv1d.weight"].view(3, 3072, 4)[:, 2048:].count_nonzero()
        assert not mla["q_b_proj"].values[4096:].view(torch.uint8).count_nonzero()
        assert not mla["kv_b_proj"][8192:].count_nonzero()
        assert not mla["o_proj"].values[:, 4096:].view(torch.uint8).count_nonzero()
        assert not moe.shared.down.values[:, 512:].view(torch.uint8).count_nonzero()
        assert not moe.shared.down.scales[:, 4:].count_nonzero()
        assert not head.values[51584:].view(torch.uint8).count_nonzero()
    else:
        # Ranks 0 and 1 hold 24 checkpoint heads each, exactly.
        assert kda["A_log"].count_nonzero() == 24 and head.values[-1].view(torch.uint8).count_nonzero()
        assert mla["q_b_proj"].values[-1].view(torch.uint8).count_nonzero()


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_three_ranks_narrow_routed_experts_to_704_columns(loader_factory, checkpoint, rank):
    """FLARE_GLM53_EXPERT_COLUMNS=704: routed columns 0..703, 704..1407 exact, 1408..2047 + 64 zero; the shared
    expert keeps its 128-aligned 768-column shard (block-FP8 128 x 128 scales)."""
    loader, captured = loader_factory(rank, 3, expert_columns=704)
    _, _, moe, _ = _check_rank(loader, captured, checkpoint[1], rank, 3, expert_columns=704)
    assert moe.shared.gate_up.values.shape == (1536, HIDDEN) and moe.shared.down.values.shape == (HIDDEN, 768)
    plan, weights = captured[-2]
    assert plan.geometry.intermediate_size == 704
    assert weights.packed.w13.shape == (len(EXPERTS), 1408, EXPERT_HIDDEN // 2)
    assert weights.packed.w2.shape == (len(EXPERTS), EXPERT_HIDDEN, 352)
    draft_plan, draft_weights = captured[-1]
    assert draft_plan.geometry.intermediate_size == 704 and draft_weights.w2.shape == (len(EXPERTS), EXPERT_HIDDEN, 352)
    padded = rank == 2
    assert (weights.w13_scales.padded_rows == 704) is padded and (weights.w2_scales.padded_columns == 44) is padded
    expected13 = ((1408, 2048) if padded else (704 * rank, 704 * (rank + 1)))
    assert weights.w13_scales.row_range == expected13
    assert weights.w2_scales.column_range == (expected13[0] // 16, expected13[1] // 16)


def test_expert_column_override_is_validated(loader_factory):
    for world, columns in ((3, 640), (3, 736), (2, 704), (4, 704), (1, 768)):
        with pytest.raises(ValueError, match="routed experts"):
            loader_factory(0, world, expert_columns=columns)
    loader, _ = loader_factory(0, 3, expert_columns=768)
    assert loader.shards == Shards.of(3)
    with pytest.raises(ValueError, match="routed experts"):
        PreparationConfig(devices=(0, 1, 2), expert_columns=640)
    assert PreparationConfig(devices=(0, 1, 2), expert_columns=704).expert_columns == 704
    with pytest.raises(TypeError):
        PreparationConfig(devices=(0, 1, 2), expert_columns="704")


class OffByOneLoader(CpuLoader):
    """Negative control: every multi-row copy lands one row late, so padded tails hide a shifted row."""

    def _copy(self, destination, source, *, numeric=True):
        if destination.ndim >= 2 and destination.shape[0] > 1:
            destination[1:].copy_(source[:-1])
            destination[0].zero_()
        else:
            destination.copy_(source)


@pytest.mark.parametrize("expert_columns", [None, 704])
def test_negative_control_detects_a_one_row_misplacement(loader_factory, checkpoint, expert_columns):
    loader, captured = loader_factory(2, 3, expert_columns=expert_columns)
    broken = OffByOneLoader(loader.manifest, "cpu", shard=TensorParallelShard(2, 3), expert_activation="a16",
                            expert_columns=expert_columns)
    with pytest.raises(AssertionError):
        _check_rank(broken, captured, checkpoint[1], 2, 3, expert_columns)
    _check_rank(loader, captured, checkpoint[1], 2, 3, expert_columns)


def test_uncovered_partitions_fail_with_the_tensor_name(loader_factory):
    loader, _ = loader_factory(2, 3)
    with loader:
        with pytest.raises(ValueError, match=r"q_proj\.weight: padded partition \(8192, 11008\) starts past its 8192"):
            loader.load_linear(f"{KDA}.q_proj", rows=(8192, 11008), pad=True)
        with pytest.raises(ValueError, match=r"q_proj\.weight: partition \(6144, 9216\) exceeds its 8192"):
            loader.load_linear(f"{KDA}.q_proj", rows=(6144, 9216))
        with pytest.raises(ValueError, match=r"q_b_proj\.weight: partition \(12288, 18432\) exceeds its 16384"):
            loader.load_linear(f"{MLA}.q_b_proj", rows=(12288, 18432))
        with pytest.raises(ValueError, match=r"kv_b_proj\.weight: partition"):
            loader.load_linear_group((f"{MLA}.kv_b_proj",), row_ranges=((24576, 36864),))
        with pytest.raises(ValueError, match=r"up_proj\.weight: partition \(1536, 2304\) exceeds its 2048"):
            loader.load_nvfp4_experts((f"{MOE}.experts.0",), hidden_size=EXPERT_HIDDEN, intermediate_size=2048,
                                      intermediate_range=(1536, 2304), scale_vectors=(torch.ones(1),) * 4,
                                      scale_scratch=loader.allocate_expert_scale_scratch(1), activation_mode="a16")
        with pytest.raises(ValueError, match=r"up_proj\.weight: partition \(1536, 2304\) exceeds its 2048"):
            loader.load_fp4_experts((f"{DRAFT_MOE}.experts.0",), hidden_size=EXPERT_HIDDEN, intermediate_size=2048,
                                    format="nvfp4", activation_mode="a16", intermediate_range=(1536, 2304))


@pytest.mark.parametrize("expand", [False, True], ids=["scratch", "expanded"])
@pytest.mark.parametrize("activation", ["a4", "a16"])
@pytest.mark.parametrize("world,rank,expert_columns", [(3, 2, None), (3, 0, None), (2, 1, None),
                                                       (3, 0, 704), (3, 1, 704), (3, 2, 704)])
def test_routed_csf_experts_for_each_activation_and_scale_residency(
        loader_factory, checkpoint, monkeypatch, world, rank, activation, expand, expert_columns):
    """a4 and a16, shared-scratch and resident expanded scales: the same padded CSF sources, either encoding.

    The expanded path (`expand_expert_scales`, no scratch) hands each source to b12x's load-time Triton
    expansion with the swizzled native grid as output; on the CPU that call is recorded instead
    (`test_glm53_csf_padding.py` runs the kernel itself, interpreted, on both encodings).
    """
    import b12x._lib.quant._csf_prepare as prepare

    expansions = []

    def expand_source_scales(source, output, *, swizzled):
        expansions.append((source, output, swizzled))
        output.view(torch.uint8).fill_(len(expansions))
    monkeypatch.setattr(prepare, "expand_source_scales", expand_source_scales)
    monkeypatch.setattr(prepare, "validate_source", lambda source: source.validate())
    tensors = checkpoint[1]
    loader, captured = loader_factory(rank, world, expert_activation=activation, expert_columns=expert_columns)
    first, last = Shards.of(world, expert_columns=expert_columns).routed_columns(rank)
    if expert_columns is None:
        assert (first, last) == TensorParallelShard(rank, world).padded_partition(2048, alignment=128)
    local, covered = last - first, min(last, 2048) - first
    with loader:
        scratch = None if expand else loader.allocate_expert_scale_scratch(len(EXPERTS))
        moe = loader.load_mlp(3, scale_scratch=scratch, expert_ids=EXPERTS)
    assert moe.experts_a4_prefill == (activation == "a16")
    plan, weights = captured[-1]
    assert getattr(plan.activation.mode, "value", plan.activation.mode) == activation
    assert plan.geometry.intermediate_size == local
    shapes = ((len(EXPERTS), (2 * local + 127) // 128 * 128, EXPERT_HIDDEN // 16),
              (len(EXPERTS), EXPERT_HIDDEN, (local // 16 + 3) // 4 * 4))
    if expand:
        assert not hasattr(weights, "packed") and len(expansions) == 2
        _check_csf_sources(expansions[0][0], expansions[1][0], tensors, first, covered, local)
        for index, (_, output, swizzled) in enumerate(expansions):
            assert swizzled and output.dtype == torch.float8_e4m3fn and tuple(output.shape) == shapes[index]
        assert weights.w13_block_scales is expansions[0][1] and weights.w2_block_scales is expansions[1][1]
    else:
        assert not expansions
        _check_csf_sources(weights.w13_scales, weights.w2_scales, tensors, first, covered, local)
        assert weights.packed.w13_block_scales is scratch[0] and weights.packed.w2_block_scales is scratch[1]
        assert tuple(scratch[0].shape) == shapes[0] and tuple(scratch[1].shape) == shapes[1]
    packed = weights.packed if hasattr(weights, "packed") else weights
    for index, expert in enumerate(EXPERTS):
        prefix = f"{MOE}.experts.{expert}"
        _same(packed.w13[index, :covered], tensors[f"{prefix}.up_proj.weight"][first:first + covered])
        _same(packed.w13[index, local:local + covered], tensors[f"{prefix}.gate_proj.weight"][first:first + covered])
        _same(packed.w2[index, :, :covered // 2], tensors[f"{prefix}.down_proj.weight"][:, first // 2:(first + covered) // 2])
    assert not packed.w13[:, covered:local].count_nonzero() and not packed.w13[:, local + covered:].count_nonzero()
    assert not packed.w2[:, :, covered // 2:].count_nonzero()


def test_three_ranks_accept_decode_context_parallelism_with_partial_last_pages(monkeypatch):
    """DCP3: 768-token pages, page tables rounded up so a 2^20 context fits (1366 pages, the last partial)."""
    plain = PreparationConfig(devices=(0, 1, 2), context=1048576, max_active=2)
    assert (plain.page_tokens, plain.blocks, plain.page_capacity) == (256, 4096, 2 * 4096)
    config = PreparationConfig(devices=(0, 1, 2), context_parallel=True, context=1048576, max_active=2, mtp=True)
    assert config.page_tokens == 768 and config.blocks == 1366 and config.blocks * 768 >= 1048576
    assert config.page_capacity == 2 * (1366 + 1)
    assert PreparationConfig(devices=(0, 1, 2), context_parallel=True, context=262144).blocks == 342
    # Two and four ranks keep their exact page counts.
    assert PreparationConfig(devices=(0, 1), context_parallel=True, context=1048576).blocks == 2048
    assert PreparationConfig(devices=(0, 1, 2, 3), context_parallel=True, context=1048576).blocks == 1024
    # Contexts stay 256-aligned (whole local records) and at least one page.
    with pytest.raises(ValueError, match="256-aligned"):
        PreparationConfig(devices=(0, 1, 2), context_parallel=True, context=1048576 - 64)
    with pytest.raises(ValueError, match="256-aligned"):
        PreparationConfig(devices=(0, 1, 2), context_parallel=True, context=512)
    # The exchange needs 24 query heads per rank (72 in total, equal b12x shards).
    from ignition.recipes.glm53 import bootstrap

    class Narrow:
        heads, padded_heads = 22, 66

        @staticmethod
        def of(world, *, expert_columns=None):
            return Narrow

    monkeypatch.setattr(bootstrap, "Shards", Narrow)
    with pytest.raises(ValueError, match="requires 24 heads per rank"):
        PreparationConfig(devices=(0, 1, 2), context_parallel=True)
    PreparationConfig(devices=(0, 1, 2))

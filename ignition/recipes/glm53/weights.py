# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM checkpoint weights preserving per-projection precision across TP partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ignition.components.parallel import TensorParallelShard
from ignition.planning import SUPPORTED_ARCHITECTURES, SUPPORTED_CAPABILITIES
from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import IndexedCheckpoint

from .shards import Shards

if TYPE_CHECKING:
    import torch
    from b12x.moe.fused_moe import PreparedExperts

TEXT_PREFIX = "model.language_model"


@dataclass(frozen=True)
class MoEWeights:
    router: torch.Tensor
    correction: torch.Tensor
    experts: PreparedExperts
    shared: MLPWeights
    # A16 experts prepared with A4 capability: prefill may bind quantized activations.
    experts_a4_prefill: bool = False


@dataclass(frozen=True)
class LayerWeights:
    attention: dict[str, object]
    mlp: MLPWeights | MoEWeights
    residual: dict[str, torch.Tensor]


@dataclass(frozen=True)
class MTPWeights:
    attention: dict[str, object]
    mlp: MoEWeights
    tensors: dict[str, torch.Tensor]
    # Proposal head; None uses the target vocabulary projection at checkpoint precision.
    draft_head: object | None = None


@dataclass(frozen=True)
class ModelWeights:
    """One text rank's canonical modules and all packed decoder weights.

    Embeddings and normalization are replicated. Vocabulary weights partition
    their output rows. Routed layers either share one scale-expansion workspace
    and execute serially, or hold their CSF block scales expanded once at load.
    Vision weights are replicated when image graphs are prepared.
    """

    tree: torch.nn.Module
    shard: TensorParallelShard
    layers: tuple[LayerWeights, ...]
    expert_scale_scratch: tuple[torch.Tensor, torch.Tensor] | None
    allocated_bytes: int
    head: object
    mtp: MTPWeights | None = None
    vision: object | None = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, mtp=False, expert_activation="a4",
             expand_expert_scales=False, language_model_only=False, expert_columns=None,
             draft_head="target", draft_eh_proj="bf16", host_allocator=None, vision_host_allocator=None):
        """`host_allocator` (fork-only FLARE_GLM53_EMBED_HOST, package exports) moves the replicated token
        embedding into device-visible host memory after loading (`park_embedding`). `vision_host_allocator`
        (fork-only FLARE_GLM53_VISION_STAGE) parks the vision tower's block and tail weights the same way
        (`VisionWeights.park`); bootstrap then stages them through VRAM slots per image."""
        import torch

        from .model import create_model_tree

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GLM model weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
            raise ValueError(f"GLM prepared execution requires {SUPPORTED_ARCHITECTURES}")
        tree = create_model_tree(manifest.snapshot)
        before = torch.cuda.memory_allocated(device)
        layers = []
        with GlmWeightLoader(manifest, device, shard=shard,
                             progress=progress if shard.rank == 0 else None,
                             expert_activation=expert_activation, expert_columns=expert_columns) as loader:
            names = [f"{TEXT_PREFIX}.embed_tokens.weight", f"{TEXT_PREFIX}.norm.weight"]
            for name in names:
                path, attribute = name.rsplit(".", 1)
                setattr(tree.get_submodule(path), attribute,
                        torch.nn.Parameter(loader.load_tensor(name), requires_grad=False))
            # Three ranks pad the vocabulary to 3 x 51648 rows; the last rank's 64 extra rows are zero.
            head = loader.load_linear("lm_head", rows=shard.padded_partition(tree.lm_head.out_features, alignment=64),
                                      pad=True)
            if isinstance(head, torch.Tensor):
                tree.lm_head.weight = torch.nn.Parameter(head, requires_grad=False)
            tree.lm_head.out_features = head.shape[0] if isinstance(head, torch.Tensor) else head.out_features
            scales = None if expand_expert_scales else loader.allocate_expert_scale_scratch()
            for index in range(len(tree.model.language_model.layers)):
                layers.append(LayerWeights(loader.load_attention(index), loader.load_mlp(index, scale_scratch=scales),
                                           loader.load_residual(index)))
            draft = loader.load_mtp(scales) if mtp else None
            if draft is not None and isinstance(head, torch.Tensor):
                from ignition.components.vocabulary import nvfp4_draft_head

                draft = replace(draft, draft_head=nvfp4_draft_head(head))
            elif draft is not None and draft_head == "nvfp4":
                # Fork-only TP3 option: the MXFP8 head decodes exactly to BF16 (E4M3 x 2^k), chunk by chunk,
                # and requantizes to NVFP4 for proposals; verification keeps the MXFP8 head.
                from ignition.components.vocabulary import nvfp4_draft_head_from_mxfp8

                draft = replace(draft, draft_head=nvfp4_draft_head_from_mxfp8(head))
            if draft is not None and draft_eh_proj == "mxfp8":
                from ignition.components.vocabulary import mxfp8_draft_head

                # Proposal-only precision: eh_proj feeds the draft layer alone.
                tensors = dict(draft.tensors)
                tensors["eh_proj"] = mxfp8_draft_head(tensors["eh_proj"].contiguous())
                draft = replace(draft, tensors=tensors)
        if host_allocator is not None:
            park_embedding(tree, host_allocator)
        vision = None
        if not language_model_only:
            from .vision import VisionWeights

            with DirectWeightLoader(manifest, device, prefixes=("model.visual.",),
                                    progress=progress if shard.rank == 0 else None,
                                    progress_label="Load vision weights") as loader:
                vision = VisionWeights.load(tree.model.visual, loader)
            if vision_host_allocator is not None:
                # The loader's direct transfers have finished once it closes; drop the device copies after.
                torch.cuda.synchronize(device)
                vision = vision.park(lambda name, nbytes: vision_host_allocator(name, (nbytes,), torch.uint8).device_view)
                torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        return cls(tree, shard, tuple(layers), scales, torch.cuda.memory_allocated(device) - before, head, draft, vision)


def park_embedding(tree, host_allocator):
    """Move the token embedding into device-visible host memory; return the parked tensor.

    The BF16 table (154880 x 4096, 1210 MiB) is replicated on every rank, but each step gathers only its
    tokens' rows: at most the speculative rows (48) in decode and one prefill chunk (4096 rows, 32 MiB) in
    prefill, read zero-copy over PCIe by the same b12x embedding kernel (DeepSeek's TP3 `embed_host`, which
    measured c1 +0.0 %). The loader's direct transfers have finished when it closes, so the copy runs after
    the loader context; the device table is released once the module holds the parked view.
    """
    import torch

    from ignition.components.weights import _park_on_host

    module = tree.model.language_model.embed_tokens
    weight = module.weight
    if weight.device.type != "cuda":
        raise ValueError("GLM embedding parking requires the loaded CUDA table")
    torch.cuda.synchronize(weight.device)
    parked = _park_on_host(host_allocator, f"{TEXT_PREFIX}.embed_tokens.weight", weight.detach())
    torch.cuda.synchronize(weight.device)
    module.weight = torch.nn.Parameter(parked, requires_grad=False)
    del weight
    return parked


class GlmWeightLoader(DirectWeightLoader):
    """Head and intermediate sharding with replicated latent attention and mHC.

    KDA and MLA output projections produce partial hidden vectors that require
    an all-reduce. Indexer projections, low-rank inputs, normalization and mHC
    remain replicated. Routed experts partition their intermediate dimension,
    preserving all 288 experts on every rank.
    """

    def __init__(self, manifest: IndexedCheckpoint, device, *, shard: TensorParallelShard,
                 prefixes=None, progress=None, expert_activation="a4", expert_columns=None):
        super().__init__(manifest, device, prefixes=prefixes,
                         progress=progress)
        self.shard = shard
        # Routed expert columns per rank (shards.py): None keeps the 128-aligned shard of the shared expert.
        self.shards = Shards.of(shard.world_size, expert_columns=expert_columns)
        if expert_activation not in ("a4", "a16"):
            raise ValueError("GLM NVFP4-CSF experts execute with A4 or A16 activations")
        # A16 keeps calibrated activation scales so prefill can still bind A4 launches.
        self.expert_activation = expert_activation
        config = json.loads((manifest.snapshot / "config.json").read_text())
        if (config.get("model_type") != "glm5_next"
                or config.get("architectures") != ["Glm5NextForConditionalGeneration"]):
            raise ValueError("GLM weights require Glm5NextForConditionalGeneration")
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

        text = Glm5NextConfig.from_dict(config).text_config.to_dict()
        for key, expected in {
            "hidden_size": 4096, "num_hidden_layers": 45, "intermediate_size": 12288,
            "moe_intermediate_size": 2048, "n_routed_experts": 288, "n_shared_experts": 1,
            "first_k_dense_replace": 3, "num_attention_heads": 64, "q_lora_rank": 1536,
            "kv_lora_rank": 512, "qk_nope_head_dim": 256, "qk_rope_head_dim": 0,
            "v_head_dim": 256, "linear_num_heads": 64, "linear_head_dim": 128,
            "linear_conv_kernel_dim": 4, "index_n_heads": 32, "index_head_dim": 128,
            "swiglu_limit": 10.0, "hc_mult": 4,
        }.items():
            if text.get(key) != expected:
                raise ValueError(f"GLM {key} requires {expected}, got {text.get(key)}")

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if name.startswith((TEXT_PREFIX + ".", "lm_head."))}

    def _layer(self, layer):
        if type(layer) is not int or not 0 <= layer <= 45:
            raise ValueError("GLM decoder layer must be in [0, 45]")
        return f"{TEXT_PREFIX}.layers.{layer}"

    def _projection(self, prefix, *, rows=None, columns=None, pad=False):
        return self.load_linear(prefix, rows=rows, columns=columns, pad=pad)

    def _load_rows(self, name, rows):
        """Rows of a vector or [rows, ...] tensor; rows past the checkpoint extent are zero."""
        if rows[1] <= self._sources[name].shape[0]:
            return self.load_tensor(name, extent=slice(*rows))
        return self.load_padded_tensor(name, rows=rows)

    def load_attention(self, layer):
        prefix = self._layer(layer) + ".self_attn"
        if layer < 45 and layer % 4 != 3:
            return self._load_kda(prefix)
        # Padded partitions equal the exact ones for one, two and four ranks; three ranks hold 24 heads each (the last 16 real + 8 zero, shards.py).
        # A padded head has zero query rows, zero absorbed/expanded key-value rows and zero output columns,
        # so it attends uniformly to zero values and contributes nothing to the reduced output.
        head_channels = Shards.of(self.shard.world_size).head_channels(self.shard.rank, 256)
        result = {"indexer.wq_b": self._projection(f"{prefix}.indexer.wq_b")}
        # Adjacent projections share a GEMM only when their stored precision is compatible.
        result["qkv_a_proj"] = self.load_mixed_linear_group(
            tuple(f"{prefix}.{name}" for name in ("q_a_proj", "kv_a_proj_with_mqa", "indexer.wk", "indexer.weights_proj")))
        result["q_b_proj"] = self._projection(f"{prefix}.q_b_proj", rows=head_channels, pad=True)
        if self.linear_format(f"{prefix}.kv_b_proj") not in ("mxfp8", "bfloat16"):
            raise ValueError(f"{prefix}.kv_b_proj: MLA absorption requires BF16 or MXFP8 weights")
        result["kv_b_proj"] = self.load_linear_group(
            (f"{prefix}.kv_b_proj",), row_ranges=(Shards.of(self.shard.world_size).head_channels(self.shard.rank, 512),),
            pad=True)
        result["o_proj"] = self._projection(f"{prefix}.o_proj", columns=head_channels, pad=True)
        for name in ("q_a_layernorm.weight", "kv_a_layernorm.weight", "indexer.k_norm.weight",
                     "indexer.k_norm.bias", "indexer.index_kpool_compress_ape", "indexer.index_kpool_compress_gate"):
            result[name] = self.load_tensor(f"{prefix}.{name}")
        return result

    def _load_kda(self, prefix):
        import torch

        # Three ranks hold heads 0-23, 24-47 and 48-63 + 8 zero heads (shards.py). A padded head is inert with all-zero weights: zero q, k
        # and v projections and convolution taps keep its recurrent state at zero (the delta update is k-scaled;
        # q/k L2 normalization is eps-safe: x * rsqrt(sum(x^2) + 1e-6)), zero A_log/dt_bias/forget rows give the
        # finite decay lower_bound * sigmoid(0), the gated RMSNorm of a zero vector is zero (eps-safe), and the
        # output projection's columns for the head are zero. No masking is needed.
        shards = Shards.of(self.shard.world_size)
        channels = shards.head_channels(self.shard.rank, 128)
        heads = shards.head_range(self.shard.rank)
        result = {name: self._projection(f"{prefix}.{name}", rows=channels, pad=True)
                  for name in ("f_b_proj", "g_b_proj")}
        result["input_proj"] = self.load_mixed_linear_group(
            tuple(f"{prefix}.{name}" for name in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj")),
            row_ranges=(channels,) * 3 + (heads, None), pad=True)
        result["g_a_proj"] = self._projection(f"{prefix}.g_a_proj")
        result["o_proj"] = self._projection(f"{prefix}.o_proj", columns=channels, pad=True)
        result["A_log"] = self._load_rows(f"{prefix}.A_log", heads)
        result["dt_bias"] = self._load_rows(f"{prefix}.dt_bias", channels)
        result["o_norm.weight"] = self.load_tensor(f"{prefix}.o_norm.weight")
        result["conv1d.weight"] = torch.cat([
            self._load_rows(f"{prefix}.{name}_conv1d.weight", channels) for name in ("q", "k", "v")
        ]).squeeze(1)
        return result

    def load_mlp(self, layer, *, scale_scratch=None, expert_ids=None):
        """Dense or routed MLP weights; `expert_ids` restricts the routed experts (tests and probes)."""
        prefix = self._layer(layer) + ".mlp"
        if layer < 3:
            extent = self.shard.padded_partition(12288, alignment=128)
            gate_up = self.load_linear_group(tuple(f"{prefix}.{name}" for name in ("gate_proj", "up_proj")),
                                            row_ranges=(extent, extent), pad=True)
            down = self._projection(f"{prefix}.down_proj", columns=extent, pad=True)
            return MLPWeights(gate_up, down)
        # Three ranks pad the shared expert's 2048 columns to 2304 (768 per rank, 128-aligned for its block-FP8
        # scales): zero gate/up rows and down columns. Routed experts use `self.shards.routed_columns` (768 or 704).
        extent = self.shard.padded_partition(2048, alignment=128)
        shared_prefix = prefix + ".shared_experts"
        shared = MLPWeights(
            self.load_linear_group(tuple(f"{shared_prefix}.{name}" for name in ("gate_proj", "up_proj")),
                                 row_ranges=(extent, extent), pad=True),
            self._projection(f"{shared_prefix}.down_proj", columns=extent, pad=True),
        )
        from ignition.preparation.quantization import expert_format

        csf = expert_format(self.quantization, prefix + ".experts")[0] == "nvfp4_csf"
        return MoEWeights(
            self.load_tensor(f"{prefix}.gate.weight"),
            self.load_tensor(f"{prefix}.gate.e_score_correction_bias"),
            self.load_experts(layer, expert_ids=expert_ids, scale_scratch=scale_scratch), shared,
            experts_a4_prefill=csf and self.expert_activation == "a16",
        )

    def load_experts(self, layer, *, expert_ids=None, scale_scratch=None):
        if layer < 3:
            raise ValueError("GLM routed experts begin at layer 3")
        prefix = self._layer(layer) + ".mlp.experts"
        ids = tuple(range(288)) if expert_ids is None else tuple(expert_ids)
        if (not ids or len(set(ids)) != len(ids)
                or any(type(i) is not int or not 0 <= i < 288 for i in ids)):
            raise ValueError("GLM expert IDs must be distinct integers in [0, 288)")
        prefixes = tuple(f"{prefix}.{i}" for i in ids)
        from ignition.preparation.quantization import expert_format

        fmt, mode = expert_format(self.quantization, prefix)
        if fmt != "nvfp4_csf":
            # The draft layer's experts ship plain W4A16 NVFP4 scales; three ranks pad their columns too.
            return self.load_fp4_experts(prefixes, hidden_size=4096, intermediate_size=2048,
                format=fmt, activation_mode=mode,
                intermediate_range=self.shards.routed_columns(self.shard.rank), swiglu_limit=10.0,
                scale_vectors=_expert_scales(self._sources, prefixes) if mode == "a4" else None, pad=True)
        return self.load_nvfp4_experts(
            prefixes, hidden_size=4096, intermediate_size=2048,
            intermediate_range=self.shards.routed_columns(self.shard.rank),
            scale_vectors=_expert_scales(self._sources, prefixes), scale_scratch=scale_scratch,
            swiglu_limit=10.0, activation_mode=self.expert_activation, pad=True,
        )

    def allocate_expert_scale_scratch(self, expert_count=288):
        import torch

        if type(expert_count) is not int or not 1 <= expert_count <= 288:
            raise ValueError("GLM expert count must be in [1, 288]")
        local = self.shards.routed_width
        return tuple(torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device) for shape in (
            (expert_count, 2 * local, 256), (expert_count, 4096, local // 16)))

    def load_mtp(self, scale_scratch):
        prefix = self._layer(45)
        return MTPWeights(self.load_attention(45), self.load_mlp(45, scale_scratch=scale_scratch),
            {name: self.load_tensor(f"{prefix}.{name}.weight") for name in (
                "enorm", "hnorm", "eh_proj", "input_layernorm", "post_attention_layernorm", "shared_head.norm")})

    def load_residual(self, layer):
        prefix = self._layer(layer)
        # mHC parameters execute in FP32; converting once lets every prepared graph variant share them.
        return {name: self.load_tensor(f"{prefix}.{name}").float() if name.startswith("hc_")
                else self.load_tensor(f"{prefix}.{name}") for name in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale")}


def _expert_scales(sources, prefixes):
    import torch

    w13, w2, a1, a2 = [], [], [], []
    for prefix in prefixes:
        values = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_scale_2", "input_scale"):
                name = f"{prefix}.{projection}.{suffix}"
                value = sources[name]
                if (value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() != 1
                        or not bool((torch.isfinite(value) & (value > 0)).all())):
                    raise ValueError(f"{name}: expected a finite positive FP32 scalar")
                values[projection, suffix] = value.item()
        if values["gate_proj", "weight_scale_2"] != values["up_proj", "weight_scale_2"]:
            raise ValueError(f"{prefix}: gate/up weight global scales differ")
        w13.append(values["up_proj", "weight_scale_2"])
        w2.append(values["down_proj", "weight_scale_2"])
        a1.append(max(values["gate_proj", "input_scale"], values["up_proj", "input_scale"]))
        a2.append(values["down_proj", "input_scale"])
    return (torch.tensor(w13, dtype=torch.float32), torch.tensor(w2, dtype=torch.float32),
            torch.tensor(a1, dtype=torch.float32).reciprocal(), torch.tensor(a2, dtype=torch.float32).reciprocal())

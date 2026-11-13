# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native FP8/MXFP4 checkpoint transfers and tensor-parallel partitions."""

from dataclasses import dataclass, replace

import torch

from ignition.components.weights import MLPWeights
from ignition.components.block_fp8_weights import BlockFP8WeightLoader
from ignition.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
from ignition.components.markov_sampling import fp8_markov_head

from .config import ModelConfig
from .tp3_config import TP3Config


@dataclass(frozen=True)
class ExpertWeights:
    router: torch.Tensor
    correction: torch.Tensor
    experts: object
    shared: MLPWeights
    vision_correction: torch.Tensor | None = None
    # TP3 expert promotion: a parked layer keeps its hottest experts on the GPU (`experts`) and the rest in
    # device-visible host memory (`cold`). The maps translate global router IDs to each bank's local slot
    # (-1 = not in this bank, which the b12x W4A8 kernels treat as an inactive route).
    cold: object = None
    hot_map: torch.Tensor | None = None
    cold_map: torch.Tensor | None = None
    # Merged bank only: bank slots < hot_rows are VRAM rows, the rest pinned host memory (None = no host tail).
    hot_rows: int | None = None


@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    experts: ExpertWeights
    residual: dict
    engram: dict | None


@dataclass(frozen=True)
class DSparkWeights:
    layers: tuple[LayerWeights, ...]
    main_projection: object
    context_kv: object
    main_norm: torch.Tensor
    norm: torch.Tensor
    markov_embedding: torch.Tensor
    markov_head: object
    confidence_head: torch.Tensor | None = None
    draft_head: object | None = None

    @classmethod
    def load(cls, loader, *, confidence=False):
        config = loader.config.draft
        last = f"mtp.{config.layers - 1}"
        return cls(tuple(loader.load_layer(i, draft=True) for i in range(config.layers)),
            loader.load_linear("mtp.0.main_proj"),
            loader.load_linear_group(tuple(f"mtp.{i}.attn.wkv" for i in range(config.layers))),
            loader.load_tensor("mtp.0.main_norm.weight"),
            loader.load_tensor(last + ".norm.weight"), loader.load_tensor(last + ".markov_head.embed.weight"),
            loader.load_tensor(last + ".markov_head.head.weight"),
            loader.load_tensor(last + ".confidence_head.proj.weight") if confidence else None)


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[LayerWeights, ...]
    draft: DSparkWeights | None
    allocated_bytes: int
    vision: dict | None = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, dspark=False, confidence=False,
             expert_scale_compression=None, draft_head_quantization=None, markov_head_quantization=None,
             vocabulary_parallel_sampling=False, host_allocator=None, tp3=None):
        tp3 = TP3Config() if tp3 is None else tp3
        config = ModelConfig.load(manifest.metadata_path, mini=tp3.mini)
        before = torch.cuda.memory_allocated(device)
        with DeepSeekWeightLoader(manifest, device, shard=shard, progress=progress, dspark=dspark,
                                 expert_scale_compression=expert_scale_compression,
                                 host_allocator=host_allocator, tp3=tp3) as loader:
            embedding, norm = loader.load_tensors(("embed.weight", "norm.weight"))
            if tp3.embed_host and host_allocator is not None:
                # TP3: the 1.23 GiB token embedding is replicated on every rank but each step gathers only its tokens'
                # rows (<= 48 in decode, one prefill chunk otherwise); read it from device-visible host memory and give
                # the VRAM to hot experts / KV.
                from ignition.components.weights import _park_on_host
                embedding = _park_on_host(host_allocator, "embed.weight", embedding)
            head = loader.load_padded_tensor("head.weight", rows=shard.padded_partition(129280, alignment=64))
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
            draft = DSparkWeights.load(loader, confidence=confidence) if dspark else None
            if draft is not None:
                if vocabulary_parallel_sampling:
                    first, last = shard.padded_partition(129280, alignment=64)
                    local = torch.zeros((last - first, draft.markov_head.shape[1]),
                                        device=device, dtype=draft.markov_head.dtype)
                    covered = min(last, 129280) - first
                    local[:covered].copy_(draft.markov_head[first:first + covered])
                    draft = replace(draft, markov_head=local)
                if draft_head_quantization == "nvfp4":
                    draft = replace(draft, draft_head=nvfp4_draft_head(head))
                elif draft_head_quantization == "mxfp8":
                    draft = replace(draft, draft_head=mxfp8_draft_head(head))
                if markov_head_quantization == "nvfp4":
                    draft = replace(draft, markov_head=nvfp4_draft_head(draft.markov_head))
                elif markov_head_quantization == "fp8":
                    draft = replace(draft, markov_head=fp8_markov_head(draft.markov_head))
            vision = ({name: loader.load_tensor(name) for name in loader._sources
                       if name.startswith(("vision.", "aligner.", "image_"))}
                      if config.vision is not None else None)
            if vision and host_allocator is not None and tp3.vision_host:
                # TP3: the vision tower (~0.8 GiB, replicated) runs only for image prompts; its projections consume these
                # tensors as-is, so read them from device-visible host memory and give the VRAM to KV.
                # tp3.vision_stage keeps the tower parked and additionally stages each block's weights through
                # double-buffered VRAM slots inside the vision graphs (ignition/components/vision_staging.py). Without a
                # host allocator nothing is parked, and bootstrap (`vision_host_parked`) then skips staging as well.
                from ignition.components.weights import _park_on_host
                vision = {name: _park_on_host(host_allocator, name, tensor) for name, tensor in vision.items()}
        return cls(config, shard, embedding, norm, head, layers, draft,
                   torch.cuda.memory_allocated(device) - before, vision)


def vision_host_parked(vision: dict, host_regions) -> bool:
    """Whether every vision/aligner tensor lives in one of the package's mapped host regions (`_park_on_host`)."""
    spans = [(region.address, region.address + region.extent) for region in host_regions if not region.closed]
    tensors = [tensor for name, tensor in vision.items() if name.startswith(("vision.", "aligner."))]
    return bool(tensors) and all(any(start <= tensor.data_ptr() < end for start, end in spans) for tensor in tensors)


def hot_expert_ids(layer: int, experts: int, tp3: TP3Config) -> list | None:
    """TP3 expert promotion: the ``tp3.hot_experts`` hottest experts of a parked layer, by the route counts in
    ``tp3.expert_counts_dir``/expert-routes-rank0-layer{L}.json (vLLM route-monitor format). None = park all."""
    plan = global_cold_plan(tp3)
    if plan is not None:
        cold = plan.get(layer, frozenset())
        return [e for e in range(experts) if e not in cold] if cold else None
    hot = tp3.hot_experts
    if hot <= 0:
        return None
    if hot >= experts:
        return list(range(experts))
    import json
    if tp3.expert_counts_dir is None:
        raise ValueError(f"TP3 config: hot_experts={hot} for layer {layer} needs expert_counts_dir")
    path = tp3.expert_counts_dir / f"expert-routes-rank0-layer{layer}.json"
    counts = json.loads(path.read_text())["counts"]
    if len(counts) != experts:
        raise ValueError(f"{path}: expected {experts} expert counts, got {len(counts)}")
    order = sorted(range(experts), key=lambda e: (-counts[e], e))
    return sorted(order[:hot])


_GLOBAL_PLAN = {}


def global_cold_plan(tp3: TP3Config) -> dict | None:
    """``tp3.global_cold`` = N: the N least-routed (layer, expert) pairs across all MoE layers live in host
    memory (merged banks); every other expert stays in VRAM. Route counts as for ``hot_experts``.
    Returns {layer: frozenset(cold experts)}. DROPPED 2026-11 (N1 global placement: c8 -2..-7%); kept opt-in."""
    budget = tp3.global_cold
    if budget <= 0:
        return None
    key = (budget, tp3.expert_counts_dir)
    if key not in _GLOBAL_PLAN:
        import json
        root = tp3.expert_counts_dir
        pairs = []
        for path in sorted(root.glob("expert-routes-rank0-layer*.json")):
            layer = int(path.stem.rsplit("layer", 1)[1])
            counts = json.loads(path.read_text())["counts"]
            total = sum(counts) or 1
            pairs.extend((count / total, layer, expert) for expert, count in enumerate(counts))
        pairs.sort()
        plan = {}
        for _, layer, expert in pairs[:budget]:
            plan.setdefault(layer, set()).add(expert)
        _GLOBAL_PLAN[key] = {layer: frozenset(cold) for layer, cold in plan.items()}
    return _GLOBAL_PLAN[key]


def host_expert_layers(tp3: TP3Config) -> frozenset:
    """TP3 expert parking: the MoE layers whose routed-expert banks live in device-visible host memory (read over
    PCIe when routed): ``tp3.host_expert_layers``, or the layers of the global-cold plan when that is active."""
    plan = global_cold_plan(tp3)
    if plan is not None:
        return frozenset(plan)
    return tp3.host_expert_layers


class DeepSeekWeightLoader(BlockFP8WeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None, dspark=False,
                 expert_scale_compression=None, host_allocator=None, tp3=None):
        if expert_scale_compression not in (None, "csf"):
            raise ValueError("DeepSeek V4.1 expert scale compression must be None or 'csf'")
        self.dspark = dspark
        self.expert_scale_compression = expert_scale_compression
        self.tp3 = TP3Config() if tp3 is None else tp3
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config = ModelConfig.load(manifest.metadata_path, mini=self.tp3.mini)
        self.shard = shard
        self._expert_scale_scratch = {}
        self._host_allocator = host_allocator
        self._host_layers = host_expert_layers(self.tp3) if host_allocator is not None else frozenset()

    def load_experts(self, prefixes, interval, *, host=False):
        text = self.config.text
        # TP3 BRRR experiment: the fused layer kernel reads native E8M0 expert scales, so its layers skip CSF.
        native = (self.tp3.brrr_native_scales and prefixes and prefixes[0].startswith("layers.")
                  and int(prefixes[0].split(".")[1]) in self.tp3.brrr_layers)
        return self.load_mxfp4_experts(prefixes, hidden_size=text["hidden_size"],
            intermediate_size=text["moe_intermediate_size"], intermediate_range=interval,
            swiglu_limit=text["swiglu_limit"], scale_name="scale", projection_names=("w3", "w1", "w2"),
            scale_compression=None if native else self.expert_scale_compression,
            scale_scratch=None if native else self._expert_scale_scratch,
            host_allocator=self._host_allocator if host else None)

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if (name.startswith("layers.") or (self.dspark and name.startswith("mtp."))
                    or name.startswith(("vision.", "aligner.", "image_"))
                    or name in ("embed.weight", "norm.weight", "head.weight"))
                and ".engram.embed." not in name}

    def load_layer(self, layer, *, draft=False):
        prefix = f"mtp.{layer}" if draft else f"layers.{layer}"
        text, shard = self.config.text, self.shard
        attn = prefix + ".attn"
        first, last = shard.padded_partition(8)
        heads = first * 8 * 512, last * 8 * 512
        groups = first * 1024, last * 1024
        result = {
            "qkv_a": self.load_linear_group((attn + ".wq_a", attn + ".wkv")),
            "q_b": self.load_linear_group((attn + ".wq_b",), row_ranges=(heads,), pad=True),
            "output": self.load_attention_output(attn, groups, pad=True),
            "q_norm": self.load_tensor(attn + ".q_norm.weight"),
            "kv_norm": self.load_tensor(attn + ".kv_norm.weight"),
            "sink": self.load_padded_tensor(attn + ".attn_sink", rows=(first * 8, last * 8)),
        }
        if not draft and layer in text["kv_source_layer_ids"]:
            ratio = text["compress_ratios"][layer]
            result["compressor"] = {
                "values": self.load_tensor(attn + ".compressor.wkv.weight"),
                "norm": self.load_tensor(attn + ".compressor.norm.weight", dtype=torch.float32),
            }
            if ratio == 2:
                result["compressor"]["gates"] = self.load_tensor(attn + ".compressor.wgate.weight")
            result["index_k"] = self.load_tensor(attn + ".indexer.wk.weight")
            result["index_norm"] = self.load_tensor(attn + ".indexer.k_norm.weight")
        if not draft and layer in text["index_source_layer_ids"]:
            # Replicated index heads avoid a context-sized TP score reduction.
            result["index_q"] = self.load_linear(attn + ".indexer.wq_b")
            result["index_weights"] = self.load_tensor(attn + ".indexer.weights_proj.weight")
        ffn = prefix + ".ffn"
        interval = shard.partition(text["moe_intermediate_size"], alignment=32)
        count = self.config.draft.experts if draft else text["n_routed_experts"]
        parked = not draft and layer in self._host_layers
        hot = hot_expert_ids(layer, count, self.tp3) if parked else None
        cold = hot_map = cold_map = hot_rows = None
        if hot is None or len(hot) == count:
            experts = self.load_experts([f"{ffn}.experts.{i}" for i in range(count)], interval,
                                        host=parked and hot is None)
        elif self.tp3.merged_bank and getattr(self._host_allocator, "__self__", None) is not None:
            # One bank, hot experts first: VRAM rows for hot, host rows for cold (one MoE call per layer).
            storage = self._host_allocator.__self__
            hot_set = set(hot)
            order = list(hot) + [e for e in range(count) if e not in hot_set]
            host_allocator = self._host_allocator
            self._host_allocator = lambda name, shape, dtype: storage.mixed_allocate(
                name, shape, dtype, device_rows=len(hot))
            try:
                experts = self.load_experts([f"{ffn}.experts.{i}" for i in order], interval, host=True)
            finally:
                self._host_allocator = host_allocator
            hot_map = torch.empty((count,), dtype=torch.int32)
            hot_map[order] = torch.arange(count, dtype=torch.int32)
            hot_map = hot_map.to(self.device)
            hot_rows = len(hot)
        else:
            hot_set = set(hot)
            cold_ids = [e for e in range(count) if e not in hot_set]
            experts = self.load_experts([f"{ffn}.experts.{i}" for i in hot], interval, host=False)
            cold = self.load_experts([f"{ffn}.experts.{i}" for i in cold_ids], interval, host=True)
            hot_map = torch.full((count,), -1, dtype=torch.int32)
            hot_map[hot] = torch.arange(len(hot), dtype=torch.int32)
            cold_map = torch.full((count,), -1, dtype=torch.int32)
            cold_map[cold_ids] = torch.arange(len(cold_ids), dtype=torch.int32)
            hot_map, cold_map = hot_map.to(self.device), cold_map.to(self.device)
        shared = MLPWeights(self.load_linear_group((ffn + ".shared_experts.w1", ffn + ".shared_experts.w3"),
            row_ranges=(interval, interval)), self.load_linear(ffn + ".shared_experts.w2", columns=interval))
        routed = ExpertWeights(self.load_tensor(ffn + ".gate.weight"),
            self.load_tensor(ffn + ".gate.bias"), experts, shared,
            self.load_tensor(ffn + ".gate.bias_vl") if self.config.vision is not None else None,
            cold=cold, hot_map=hot_map, cold_map=cold_map, hot_rows=hot_rows)
        residual = {name: self.load_tensor(prefix + "." + name) for name in (
            "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
            "attn_norm.weight", "ffn_norm.weight")}
        engram = None
        if not draft and layer in text["engram_layer_ids"]:
            engram = {"projection": self.load_linear_columns(prefix + ".engram.wkv",
                ((head * 256, (head + 1) * 256) for head in range(shard.rank, 24, shard.world_size))),
                "weight": self.load_tensor(prefix + ".engram.q_weight").float()
                          * self.load_tensor(prefix + ".engram.k_weight").float()}
        return LayerWeights(result, routed, residual, engram)

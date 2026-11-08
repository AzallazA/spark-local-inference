# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel DeepSeek text stack with lagged hyperconnections."""

from dataclasses import dataclass, fields, is_dataclass, replace

import os

import torch

from ignition.components.collectives import PreparedAllReduce, PreparedChunkedAllReduce
from . import probe as _probe
from ignition.components.embedding import PreparedEmbedding
from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.ring_commit import PreparedRingCommit
from ignition.components.linear import UnquantizedProjection
from ignition.components.mhc import MHCWeights, MHCScratch, PreparedMHC, PreparedMHCPost
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import RouterScratch, PreparedRouter, PreparedRoutedExperts
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.scratch import GraphBufferPool, GraphBuffers

from .attention import AttentionIndices, AttentionScratch, PreparedAttention, PreparedGlobalKV, Rotary, normalize
from .decoder_prefill import PreparedDecoderSuffix
from .engram import EngramScratch, PreparedEngram, PreparedTableRequests


# Debug A/B (flare#193): FLARE_DS41_NO_ENGRAM=1 skips the Engram contribution in every captured forward.
_NO_ENGRAM = os.environ.get("FLARE_DS41_NO_ENGRAM", "") not in ("", "0")

# TP3 BRRR: each rank decides per layer and graph whether the fused tail (and prologue) runs, from its own environment
# (FLARE_DS41_BRRR_LAYERS, FLARE_DS41_BRRR_PARKED_MAX_ROWS, FLARE_DS41_HOST_EXPERT_LAYERS, FLARE_DS41_HOT_EXPERTS, ...)
# and the graph's row count. The kernel's in-kernel all-reduce needs every rank to make the same decision: a rank that
# runs Flare's path while a peer waits in the kernel deadlocks. PreparedLayer records its decisions here and the bootstrap
# compares them across ranks (check_brrr_decisions) in plain construction code, before any capture.
_BRRR_DECISIONS = []
_BRRR_CHECKED = 0
_BRRR_DECISION_FIELDS = ("graph", "layer", "mode", "rows", "brrr", "prologue", "gated")


def record_brrr_decision(name, index, mode, rows, uses_brrr, uses_prologue=False, gated=False):
    _BRRR_DECISIONS.append((str(name), int(index), str(mode), int(rows), bool(uses_brrr), bool(uses_prologue), bool(gated)))


def check_brrr_decisions(group):
    """All-gather the BRRR decisions recorded since the previous check and raise when ranks disagree.

    Collective over ``group`` (the Gloo group create_engine used): every rank must call it at the same point, from plain
    construction code, never inside a traced or captured forward. No-op without a BRRR engine (no engine, no kernel to
    deadlock) or without a group. Returns the number of decisions compared.
    """
    global _BRRR_CHECKED
    from ignition.recipes.deepseek41.brrr import engine

    if group is None or engine() is None:
        return 0
    import torch.distributed as dist

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("check_brrr_decisions must run outside CUDA graph capture")
    pending, _BRRR_CHECKED = list(_BRRR_DECISIONS[_BRRR_CHECKED:]), len(_BRRR_DECISIONS)
    gathered = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, pending, group=group)
    root = gathered[0]
    for rank, other in enumerate(gathered[1:], 1):
        if other == root:
            continue
        if len(other) != len(root):
            detail = f"rank 0 recorded {len(root)} layer decisions, rank {rank} recorded {len(other)}"
        else:
            at = next(i for i, (a, b) in enumerate(zip(root, other)) if a != b)
            detail = (f"rank 0 decided {dict(zip(_BRRR_DECISION_FIELDS, root[at]))}, "
                      f"rank {rank} decided {dict(zip(_BRRR_DECISION_FIELDS, other[at]))}")
        raise RuntimeError("BRRR layer decisions differ between ranks (the fused kernel's all-reduce would deadlock): "
                           f"{detail}. Every rank must see the same FLARE_DS41_BRRR_* and FLARE_DS41_HOST_EXPERT_LAYERS/"
                           "FLARE_DS41_HOT_EXPERTS/FLARE_DS41_GLOBAL_COLD environment and graph catalog.")
    return len(pending)

@dataclass(frozen=True)
class Scratch:
    """BF16 decoder activations and FP32 target vocabulary logits."""

    logits_dtype = torch.float32

    embedding: torch.Tensor
    residual: torch.Tensor
    engram_residual: torch.Tensor
    engram: EngramScratch
    pre_mix: torch.Tensor
    attn_mix: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    attention: AttentionScratch
    attention_output: torch.Tensor
    shared: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    routed: torch.Tensor
    mlp_partial: torch.Tensor
    mlp_output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor
    buffers: GraphBuffers

    def rows(self, tokens, samples):
        def select(value, count):
            if isinstance(value, torch.Tensor):
                return value[:count]
            if isinstance(value, GraphBuffers):
                return value.fork()
            if is_dataclass(value):
                return type(value)(*(select(getattr(value, field.name), count) for field in fields(value)))
            raise TypeError("decoder scratch must contain tensors and tensor dataclasses")
        return type(self)(*(select(getattr(self, field.name), samples if field.name in
            ("hidden", "local_logits", "logits") else tokens) for field in fields(self)))

    @classmethod
    def allocate(cls, tokens, samples, world, device, *, experts=384, top_k=6, buffers=None,
                 candidate_tokens=None):
        if buffers is None:
            buffers = GraphBufferPool(device, chunk_bytes=2 << 20).cursor()
        def zeros(*shape, dtype=torch.bfloat16, cursor=buffers):
            return cursor.zeros(shape, dtype)
        residual, engram_residual = zeros(tokens, 4, 5120), zeros(tokens, 4, 5120)
        pre_mix, attn_mix = zeros(tokens, 4, dtype=torch.float32), zeros(tokens, 4, dtype=torch.float32)
        attention_hc = MHCScratch.allocate(tokens, 5120, device, buffers=buffers)
        ffn_hc = MHCScratch.allocate(tokens, 5120, device, buffers=buffers)
        attention = AttentionScratch.allocate(tokens, world, device, buffers=buffers, candidate_tokens=candidate_tokens)
        attention_output, mlp_output = zeros(tokens, 5120), zeros(tokens, 5120)
        _, vocabulary = TensorParallelShard(0, world).padded_partition(129280, alignment=64)
        hidden = zeros(samples, 5120)
        local_logits = zeros(samples, vocabulary, dtype=cls.logits_dtype)
        logits = zeros(samples, 129280, dtype=cls.logits_dtype)
        # Embedding, Engram projection, and feed-forward intermediates execute serially.
        embed, lookup, ffn = buffers.fork(), buffers.fork(), buffers.fork()
        embedding = zeros(tokens, 5120, cursor=embed)
        engram = EngramScratch.allocate(tokens, device, buffers=lookup, world=world)
        shared = MLPScratch.allocate(tokens, 5120, 2304 // world, device, buffers=ffn)
        router_logits = zeros(tokens, experts, dtype=torch.float32, cursor=ffn)
        router = RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k, buffers=ffn)
        routed, partial = zeros(tokens, 5120, cursor=ffn), zeros(tokens, 5120, cursor=ffn)
        buffers.join(embed, lookup, ffn)
        return cls(embedding, residual, engram_residual, engram, pre_mix, attn_mix, attention_hc, ffn_hc,
            attention, attention_output, shared, router_logits, router, routed, partial, mlp_output,
            hidden, local_logits, logits, buffers)


class DraftScratch(Scratch):
    """Retain BF16 proposal logits independently of FP32 target logits."""

    logits_dtype = torch.bfloat16


def _mhc_weights(weights, kind, norm):
    residual = weights.residual
    return MHCWeights(residual[f"hc_{kind}_fn"], residual[f"hc_{kind}_scale"],
                      residual[f"hc_{kind}_base"], residual[norm + ".weight"])


class PreparedDecoderBoundary:
    """Produce complete global KV, then gather the decoder's bounded replay rows."""

    def __init__(self, index, weights, source, target, packing, pool, rotary, *, carry,
                 session, name, workspace, window_metadata, source_indices, buffers=None):
        self.packing = packing
        self.window_metadata = window_metadata
        self.incoming = ((source.mlp_output, source.ffn_hc.residual, source.ffn_hc.post, source.ffn_hc.comb)
                         if carry else (source.residual,))
        text = pool.config.text
        self.mhc = PreparedMHC(self.incoming, _mhc_weights(weights, "attn", "attn_norm"), source.attention_hc,
            session=session, name=name + ".attn_hc", workspace=workspace,
            pre_mix=source.pre_mix, pre_out=source.attn_mix, rms_eps=text["rms_norm_eps"],
            norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"], sinkhorn_iters=text["hc_sinkhorn_iters"])
        self.normalized = source.attention_hc.normalized
        self.global_kv = PreparedGlobalKV(index, weights.attention, self.normalized, source.attention,
            packing.source, pool, rotary, session=session, name=name + ".global_kv",
            indices=source_indices, buffers=buffers)
        self.rows = tuple((getattr(source.attention_hc, field), getattr(target.attention_hc, field))
                          for field in ("normalized", "residual", "post", "comb"))
        self.rows += ((source.attn_mix, target.attn_mix),)

    def __call__(self, indexer):
        self.packing.begin()
        # Packed positions and replay bounds must precede their shared addresses.
        self.window_metadata()
        self.mhc(*self.incoming)
        self.global_kv(self.normalized, indexer)
        for source, output in self.rows:
            self.packing.gather(source, output)


class PreparedLayer:
    def __init__(self, index, weights, tables, scratch, metadata, pool, rotary, *, shard, mode,
                 runtime, coordinator, session, name, workspace, carry, defer_post,
                 boundary=None, replay_start=None, window_staging=None, mla_compute_mode="auto", buffers=None,
                 window_metadata=None):
        s, text = scratch, pool.config.text
        self.scratch, self.carry = s, carry
        self.probe_index, self.probe_mode = index, mode
        if _probe.ENABLED:
            _probe._state(scratch.residual.device)  # allocate eagerly, outside traced forwards
        self.metadata = metadata
        buffers = scratch.buffers if buffers is None else buffers
        self.image_mask = (buffers.zeros(metadata.token_ids.shape, torch.bool)
                           if mode == "prefill" and weights.experts.vision_correction is not None else None)
        self.boundary = boundary
        self.engram = None
        if weights.engram is not None:
            self.engram = PreparedEngram(index, weights.engram, tables, s.residual, s.engram_residual,
                metadata, pool,
                scratch=s.engram, session=session, name=name + ".engram", workspace=workspace,
                runtime=runtime, coordinator=coordinator)
        source = s.engram_residual if self.engram is not None else s.residual
        options = dict(rms_eps=text["rms_norm_eps"], norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"],
                       sinkhorn_iters=text["hc_sinkhorn_iters"])
        incoming = ((s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb) if carry else (source,))
        self.attention_hc = None if boundary is not None else PreparedMHC(
            incoming, _mhc_weights(weights, "attn", "attn_norm"), s.attention_hc,
            session=session, name=name + ".attn_hc", workspace=workspace, pre_mix=s.pre_mix, pre_out=s.attn_mix, **options)
        self.attention = PreparedAttention(index, weights.attention, s.attention_hc.normalized, s.attention,
            metadata, pool, rotary, mode=mode, world=shard.world_size,
            session=session, name=name + ".attention", workspace=workspace, replay_start=replay_start,
            window_staging=window_staging, mla_compute_mode=mla_compute_mode, buffers=buffers,
            window_metadata=window_metadata)
        # Chunked: a decode-sized runtime (FLARE_DS41_AR_ROWS) still reduces prefill rows; one part in decode.
        self.attention_reduce = PreparedChunkedAllReduce(s.attention.output, s.attention_output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".attention_reduce")
        self.ffn_hc = PreparedMHC((s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
            _mhc_weights(weights, "ffn", "ffn_norm"), s.ffn_hc, session=session, name=name + ".ffn_hc", workspace=workspace,
            pre_mix=s.attn_mix, pre_out=s.pre_mix, **options)
        self.router_projection = UnquantizedProjection(s.ffn_hc.normalized, weights.experts.router, s.router_logits,
            session=session, name=name + ".router")
        self.router = PreparedRouter(s.router_logits, s.router, session=session, name=name + ".topk",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction,
            image_correction_bias=weights.experts.vision_correction if self.image_mask is not None else None,
            image_mask=self.image_mask)
        self.cold_experts = self.hot_map = None
        if weights.experts.cold is not None:
            # TP3 expert promotion: hot experts on the GPU, cold experts in host memory. Each bank sees the
            # router's IDs remapped to its local slots; IDs outside a bank are inactive routes there.
            self.hot_map, self.cold_map = weights.experts.hot_map, weights.experts.cold_map
            self.hot_ids = buffers.zeros(tuple(s.router.ids.shape), s.router.ids.dtype)
            self.cold_ids = buffers.zeros(tuple(s.router.ids.shape), s.router.ids.dtype)
            self.cold_routed = buffers.zeros(tuple(s.routed.shape), s.routed.dtype)
            route_ids = self.hot_ids
        elif weights.experts.hot_map is not None:
            # Merged bank: one remap from global expert IDs to the bank's hot-first slots.
            self.hot_map, self.cold_map = weights.experts.hot_map, None
            self.hot_ids = buffers.zeros(tuple(s.router.ids.shape), s.router.ids.dtype)
            route_ids = self.hot_ids
        else:
            route_ids = s.router.ids
        self.experts = PreparedRoutedExperts(s.ffn_hc.normalized, weights.experts.experts, route_ids,
            s.router.weights, s.routed, session=session, name=name + ".experts", workspace=workspace,
            router_plan=self.router.plan)
        if weights.experts.cold is not None:
            self.cold_experts = PreparedRoutedExperts(s.ffn_hc.normalized, weights.experts.cold, self.cold_ids,
                s.router.weights, self.cold_routed, session=session, name=name + ".cold_experts",
                workspace=workspace, router_plan=self.router.plan)
        self.shared = PreparedMLP(s.ffn_hc.normalized, weights.experts.shared, s.shared.gate_up,
            s.shared.intermediate, s.shared.output, session=session, name=name + ".shared", workspace=workspace,
            swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto")
        self.concurrent_shared = None
        if mode in ("decode", "verify", "draft") and s.ffn_hc.normalized.shape[0] <= 128 and workspace.overlap:
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.router_projection.plan, self.router.plan, self.experts.plan,
                                *(() if self.cold_experts is None else (self.cold_experts.plan,))),
                session=session, workspace=workspace,
                defer_launch=s.ffn_hc.normalized.shape[0] > 4)
        self.mlp_reduce = PreparedChunkedAllReduce(s.mlp_partial, s.mlp_output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".mlp_reduce")
        self.post = None if defer_post else PreparedMHCPost(s.mlp_output, s.ffn_hc.residual,
            s.ffn_hc.post, s.ffn_hc.comb, s.residual, session=session, name=name + ".post")
        self.brrr = None
        self.brrr_prologue = None
        from ignition.recipes.deepseek41.brrr import brrr_layers
        # FLARE_DS41_BRRR_PARKED_MAX_ROWS=N: parked (host-expert) layers use BRRR only in graphs of at most N rows; wide
        # verification of parked layers stays on Flare's MoE path (BRRR-40 measured c8 -2.7% vs H300, BRRR-26 level).
        from ignition.recipes.deepseek41.weights import host_expert_layers
        parked_rows = int(os.environ.get("FLARE_DS41_BRRR_PARKED_MAX_ROWS", "0") or 0)
        rows = metadata.positions.numel()
        # A layer counts as parked only if it actually has a host tail (HOT >= 384 loads it fully resident, no hot_map).
        parked = index in host_expert_layers() and self.hot_map is not None
        gated = bool(parked_rows) and parked and rows > parked_rows
        if index in brrr_layers() and mode in ("decode", "verify") and not gated:
            # TP3 BRRR experiment: one fused kernel from the attention output to the MLP all-reduce.
            if self.cold_experts is not None or self.image_mask is not None:
                raise ValueError(f"BRRR layer {index} needs one expert bank (resident or merged hot/cold)")
            from ignition.recipes.deepseek41.brrr import PreparedBRRRTail
            self.attention.external_tail = True
            self.brrr = PreparedBRRRTail(index, weights, s, self.attention.window_metadata.position,
                                         rotary.frequencies[bool(self.attention.ratio)], config=pool.config)
            # FLARE_DS41_BRRR_PROLOGUE=1: fused attention prologue too, on plain layers of graphs with <= 8 rows (BRRR's
            # PROLOGUE_MAXM). Excluded: no carried state (post materialized), Engram, indexer/KV sources (they reuse
            # Flare-format quantized q-low / normalized input), the layer that owns window metadata.
            att = self.attention
            if (os.environ.get("FLARE_DS41_BRRR_PROLOGUE") == "1" and carry and boundary is None and self.engram is None
                    and s.residual.shape[0] <= 8 and not att.owns_window_metadata and att.global_kv is None and not att.selectors
                    and att.concurrent_query is None):
                from ignition.recipes.deepseek41.brrr import PreparedBRRRPrologue
                self.brrr_prologue = PreparedBRRRPrologue(index, weights, s, att.window_metadata.position,
                                                          rotary.frequencies[bool(att.ratio)], text["rms_norm_eps"], buffers)
                att.external_prologue = True
        if brrr_layers():
            record_brrr_decision(name, index, mode, rows, self.brrr is not None, self.brrr_prologue is not None, gated)

    def __call__(self):
        s = self.scratch
        if self.engram is not None and _NO_ENGRAM:
            # Debug A/B (flare#193): keep the Engram handshake (readiness wait, lookup, timestamps) but
            # discard its contribution, so only Engram's effect on the residual is removed.
            source = self.engram(s.residual)
            source.copy_(s.residual)
        elif self.engram is not None:
            source = self.engram(s.residual)
        else:
            source = s.residual
        incoming = ((s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb)
                    if self.carry else (source,))
        _probe.record(source, self.probe_index, 0, self.probe_mode)
        if self.brrr_prologue is not None:
            # The fused prologue replaces the attention mHC (post_pre + attn_norm) and the q/kv projections.
            self.brrr_prologue()
            self.brrr(self.attention(s.attention_hc.normalized), mix=self.brrr_prologue.mix)
            if self.post is not None:
                self.post(s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb)
            return
        if self.boundary is None:
            self.attention_hc(*incoming)
        else:
            self.boundary(self.attention.selectors[0])
        if self.brrr is not None:
            self.brrr(self.attention(s.attention_hc.normalized))
            if self.post is not None:
                self.post(s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb)
            return
        self.attention_reduce(self.attention(s.attention_hc.normalized))
        _probe.record(s.attention_output, self.probe_index, 1, self.probe_mode)
        self.ffn_hc(s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb)
        if self.concurrent_shared is not None:
            self.concurrent_shared.start(s.ffn_hc.normalized)
        if self.image_mask is not None:
            self.image_mask.copy_(self.metadata.token_ids >= 129280)
        self.router(self.router_projection(s.ffn_hc.normalized))
        _probe.router(s.router.ids, s.router.weights, self.probe_index, self.probe_mode)
        _probe.full(s.ffn_hc.normalized, self.probe_index, "in", self.probe_mode)
        if self.cold_experts is not None:
            flat = s.router.ids.reshape(-1)
            torch.index_select(self.hot_map, 0, flat, out=self.hot_ids.view(-1))
            torch.index_select(self.cold_map, 0, flat, out=self.cold_ids.view(-1))
        elif self.hot_map is not None:
            torch.index_select(self.hot_map, 0, s.router.ids.reshape(-1), out=self.hot_ids.view(-1))
        self.experts()
        if self.cold_experts is not None:
            self.cold_experts()
            s.routed.add_(self.cold_routed)
        _probe.full(s.routed, self.probe_index, "out", self.probe_mode)
        if self.concurrent_shared is None:
            self.shared(s.ffn_hc.normalized)
        else:
            self.concurrent_shared.finish(s.ffn_hc.normalized, s.routed)
        _probe.record(s.routed, self.probe_index, 3, self.probe_mode)
        _probe.record(s.shared.output, self.probe_index, 4, self.probe_mode)
        torch.add(s.routed, s.shared.output, out=s.mlp_partial)
        _probe.record(s.mlp_partial, self.probe_index, 5, self.probe_mode)
        self.mlp_reduce(s.mlp_partial)
        _probe.record(s.mlp_output, self.probe_index, 2, self.probe_mode)
        if self.post is not None:
            self.post(s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb)


class PreparedDecoder:
    def __init__(self, weights, tables, scratch, metadata, pool, *, mode, runtime,
                 vocabulary_runtime, coordinator, session, name, workspace,
                 window_staging=None, suffix_scratch=None, vision=None, mla_compute_mode="auto", scoring=None, vocabulary_parallel=False):
        if scratch.local_logits.dtype != torch.float32 or scratch.logits.dtype != torch.float32:
            raise ValueError("DeepSeek target vocabulary logits require FP32 storage")
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.mode = mode
        self.vision = vision if mode == "prefill" else None
        self.table_requests = PreparedTableRequests(tables, metadata, pool, session=session, name=name + ".table_requests")
        decoder_start = weights.config.decoder_start
        self.packing = None
        if mode == "prefill" and metadata.positions.numel() > 128 and decoder_start is not None and scoring is None:
            if (any(layer.engram is not None for layer in weights.layers[decoder_start:])
                    or any(source.layer > decoder_start for source in weights.config.sources)):
                raise ValueError("decoder suffix replay requires Engram and compressed KV sources before its boundary")
            self.packing = PreparedDecoderSuffix(metadata)
        self.output_metadata = metadata if self.packing is None else self.packing.metadata
        self.output_scratch = scratch if self.packing is None else (
            suffix_scratch if suffix_scratch is not None else Scratch.allocate(
                128, scratch.hidden.shape[0], weights.shard.world_size, scratch.embedding.device))
        self.aux_layers = () if weights.draft is None else weights.config.draft.target_layers
        if self.packing is not None and any(layer <= decoder_start for layer in self.aux_layers):
            raise ValueError("bounded decoder feedback requires draft target layers after its global-KV boundary")
        buffers = scratch.buffers
        self.auxiliary = buffers.zeros((self.output_metadata.token_ids.numel(), len(self.aux_layers) * 5120), torch.bfloat16)
        self.safe_tokens = buffers.zeros(metadata.token_ids.shape, metadata.token_ids.dtype) if self.vision is not None else metadata.token_ids
        self.embedding = PreparedEmbedding(weights.embedding, self.safe_tokens, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.rotary = Rotary(weights.config, scratch.embedding.device)
        self.window_commit = None
        if mode == "verify":
            self.window_commit = PreparedRingCommit(pool.reserved_backing,
                [pool.layout.region(f"layer{index}.window").offset for index in range(len(weights.layers))],
                metadata.state_rows, metadata.starts,
                token_capacity=metadata.max_query_tokens,
                window_tokens=128, record_bytes=528, staging=window_staging)
        # These addresses stay live through every layer and precede reusable layer buffers.
        ratios = weights.config.text["compress_ratios"][:len(weights.layers)]
        prefix_ratios = ratios if self.packing is None else ratios[:decoder_start + 1]
        self.window_metadata = AttentionIndices(metadata, mode=mode,
            tokens=metadata.positions.numel(),
            page_size=128 if mode == "decode" else 128 + metadata.max_query_tokens, width=128,
            ratios=set(prefix_ratios) - {0}, buffers=buffers)
        self.suffix_window_metadata = None
        if self.packing is not None:
            self.suffix_window_metadata = AttentionIndices(self.output_metadata, mode=mode,
                tokens=self.output_metadata.positions.numel(),
                page_size=128 + self.output_metadata.max_query_tokens, width=128,
                replay_start=self.packing.replay_start, ratios=set(ratios[decoder_start:]) - {0}, buffers=buffers)
        self.layers, carry = [], False
        layer_cursors = []
        for index, layer in enumerate(weights.layers):
            # Prefill commits each layer before the following layer reuses its private buffers.
            layer_buffers = buffers.fork() if mode == "prefill" else buffers
            layer_cursors.append(layer_buffers)
            boundary = replay_start = None
            layer_scratch, layer_metadata = scratch, metadata
            if self.packing is not None and index >= decoder_start:
                layer_scratch, layer_metadata = self.output_scratch, self.output_metadata
                replay_start = self.packing.replay_start
                if index == decoder_start:
                    boundary = PreparedDecoderBoundary(index, layer, scratch, layer_scratch, self.packing,
                        pool, self.rotary, carry=carry, session=session, name=f"{name}.layer{index}", workspace=workspace,
                        buffers=layer_buffers, window_metadata=self.suffix_window_metadata,
                        source_indices=self.window_metadata)
                    layer = replace(layer, attention={key: value for key, value in layer.attention.items()
                                                      if key not in ("compressor", "index_k", "index_norm")})
            defer = (index + 1 < len(weights.layers) and weights.layers[index + 1].engram is None
                     and index + 1 not in self.aux_layers)
            self.layers.append(PreparedLayer(index, layer, tables, layer_scratch, layer_metadata, pool, self.rotary,
                shard=weights.shard, mode=mode, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}", workspace=workspace,
                carry=carry, defer_post=defer, boundary=boundary, replay_start=replay_start,
                window_staging=None if self.window_commit is None else self.window_commit.staging[index],
                mla_compute_mode=mla_compute_mode, buffers=layer_buffers,
                window_metadata=(self.window_metadata if layer_metadata is metadata else self.suffix_window_metadata)))
            carry = defer
        buffers.join(*layer_cursors)
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=129280, session=session, coordinator=coordinator, name=name + ".prompt", copy_source=True,
            logits_dtype=torch.float32)
        self._prepare_head(vocabulary_parallel=vocabulary_parallel, vocabulary_runtime=vocabulary_runtime,
            coordinator=coordinator, session=session, name=name)

    def _prepare_head(self, *, vocabulary_parallel, vocabulary_runtime, coordinator, session, name):
        scratch, weights = self.scratch, self.weights
        self.compact_head = scratch.hidden.shape[0] > self.output_metadata.token_ids.numel()
        head_rows = self.output_metadata.token_ids.numel() if self.compact_head else scratch.hidden.shape[0]
        self.head_hidden = scratch.hidden[:head_rows]
        head_output = scratch.logits
        if vocabulary_parallel and self.compact_head:
            # Physical projection rows must remain intact while their logical
            # sample rows are expanded. Local sampling does not use gathered
            # logits; the captured round consumes this arena prefix through
            # verification and logprobs before another decoder reuses it.
            head_output = scratch.logits.view(-1)[:scratch.local_logits.numel()].view_as(scratch.local_logits)
        self.head = PreparedVocabularyProjection(self.head_hidden, weights.head, scratch.local_logits[:head_rows],
            head_output, runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            row_indices=self.output_metadata.sample_indices if self.compact_head else None, gather=not vocabulary_parallel)

        self.local_logits = self.head.output if vocabulary_parallel else scratch.local_logits

    def __call__(self):
        s, m = self.scratch, self.metadata
        _probe.step(s.residual.device, self.mode)
        _probe.positions(m.positions, self.mode)
        self.table_requests()
        if self.vision is not None:
            self.safe_tokens.copy_(m.token_ids.clamp(0, self.weights.embedding.shape[0] - 1))
        self.embedding()
        if self.vision is not None:
            self.vision.inject(m, s.embedding)
        s.residual.copy_(s.embedding[:, None, :])
        s.pre_mix.zero_()
        s.pre_mix[:, 0].fill_(1)
        self.window_metadata()
        for index, layer in enumerate(self.layers):
            if index in self.aux_layers:
                column = self.aux_layers.index(index)
                feedback = layer.scratch.residual.float().mean(1)
                if self.packing is not None:
                    feedback = torch.where(self.packing.live[:, None], feedback, 0)
                self.auxiliary[:, column * 5120:(column + 1) * 5120].copy_(feedback)
            layer()
        samples = self.output_metadata.sample_indices
        residual = (self.output_scratch.residual if self.compact_head
                    else self.output_scratch.residual[samples]).float()
        premix = (self.output_scratch.pre_mix if self.compact_head
                  else self.output_scratch.pre_mix[samples])[:, :, None]
        collapsed = (residual * premix).sum(1).bfloat16()
        if self.scoring is not None:
            collapsed = (s.residual.float() * s.pre_mix[:, :, None]).sum(1).bfloat16()
            normalize(collapsed, self.weights.norm, self.scoring.source, self.weights.config.text["rms_norm_eps"])
            self.scoring()
            self.head_hidden.copy_(self.scoring.source[samples])
        else:
            normalize(collapsed, self.weights.norm, self.head_hidden, self.weights.config.text["rms_norm_eps"])
        self.head(self.head_hidden)
        if self.compact_head:
            s.hidden.copy_(self.head_hidden[samples])
        if self.mode != "verify":
            self.table_requests.commit(m.lengths)
        return s.hidden

    def commit(self, counts):
        if self.window_commit is None:
            raise ValueError("accepted decoder commits require verification staging")
        self.window_commit(counts)
        for layer in self.layers:
            layer.attention.commit_pending(counts)
        self.table_requests.commit(counts)

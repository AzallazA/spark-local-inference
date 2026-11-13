# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared GLM KDA projections, convolution, recurrence and gated output norm."""

from dataclasses import dataclass, fields

import torch
from b12x.sequence import gdn_decode, kda_prefill, short_conv

from ignition.components import sm103
from ignition.components.kda_speculation import KDARecords, _record, register_recording
from ignition.components.gated_norm import PreparedGatedRMSNorm
from ignition.components.linear import prepare_projection
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.recurrent import (
    DeltaMetadata,
    DeltaState,
    prepare_sequence_binding,
)

from .shards import HEAD_COUNTS

# Fork-only TP3 decode-speed switch (PreparationConfig.kda_side_chain, set once per rank by bootstrap.prepare_rank
# before any scratch is allocated). Off keeps the upstream layout and schedule.
_SIDE_CHAIN = False


def configure(*, side_chain):
    """Select the KDA projection schedule for scratch allocated and layers prepared afterwards."""
    global _SIDE_CHAIN
    if type(side_chain) is not bool:
        raise TypeError("KDA side chain must be a boolean")
    _SIDE_CHAIN = side_chain


def side_chain_enabled():
    return _SIDE_CHAIN


@dataclass(frozen=True)
class KDAScratch:
    projected_input: torch.Tensor
    projected_qkv: torch.Tensor
    convolved_qkv: torch.Tensor
    forget_input: torch.Tensor
    gate_input: torch.Tensor
    raw_g: torch.Tensor
    raw_beta: torch.Tensor
    z: torch.Tensor
    recurrence: torch.Tensor
    gated: torch.Tensor
    output: torch.Tensor

    @property
    def split(self):
        """Side-chain layout: q/k/v, beta and the forget input each own contiguous rows, so the q/k/v GEMM
        writes its output in place (no staging copy) and f_b reads f_a's rows directly (no copy)."""
        return self.projected_input.shape[-1] == self.projected_qkv.shape[-1]

    @classmethod
    def allocate(cls, tokens, heads, device, *, split=None):
        if type(tokens) is not int or tokens <= 0 or heads not in HEAD_COUNTS:
            raise ValueError("GLM KDA scratch requires positive capacity and a supported rank-count head partition")
        split = _SIDE_CHAIN if split is None else split
        width = heads * 128
        def buffer(channels):
            return torch.zeros((tokens, channels), dtype=torch.bfloat16, device=device)
        if split:
            # Same bytes as the packed layout: [3w] + [heads] + [128] per row.
            qkv = buffer(3 * width)
            return cls(qkv, qkv, buffer(3 * width), buffer(128), buffer(128), buffer(width), buffer(heads),
                       buffer(width), buffer(width), buffer(width), buffer(4096))
        projected = buffer(3 * width + heads + 128)
        qkv, beta, forget = projected.split((3 * width, heads, 128), dim=-1)
        if tokens > 1:
            forget = buffer(128)
        return cls(projected, qkv, buffer(3 * width), forget, buffer(128), buffer(width), beta,
                   buffer(width), buffer(width), buffer(width), buffer(4096))

    def rows(self, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.output.shape[0]:
            raise ValueError("KDA rows exceed scratch capacity")
        views = {field.name: getattr(self, field.name)[:tokens] for field in fields(self)}
        if tokens == 1 and not self.split:
            views["forget_input"] = views["projected_input"][:, -128:]
        return KDAScratch(**views)


class PreparedGlmKDA:
    """One rank's KDA branch before the row-parallel output reduction.

    Decode and prefill use b12x recurrence kernels. Prefill output normalization
    fuses the canonical sigmoid-gated FP32 RMSNorm and zeroes inactive rows. Device
    metadata selects live rows and persistent state slots; zero live counts are
    required while preparing and tuning kernels.
    """

    def __init__(self, module, weights, source, scratch: KDAScratch, state: DeltaState,
                 metadata: DeltaMetadata, *, mode, session, name, workspace, checkpoint_export=False,
                 records=None):
        if (source.ndim != 2 or source.shape[1] != 4096 or source.dtype != torch.bfloat16
                or source.device.type != "cuda"):
            raise ValueError("GLM KDA requires BF16 [token_capacity, 4096] CUDA input")
        if mode not in ("decode", "prefill"):
            raise ValueError("KDA mode must be decode or prefill")
        if (module.num_heads != 64 or module.head_dim != 128 or module.conv_kernel_size != 4
                or module.o_norm.activation != "sigmoid" or module.forget_gate.safe_gate_lower_bound != -5.0):
            raise ValueError("canonical KDA geometry differs from GLM 5.3 Flash")
        self.heads = state.recurrent.shape[1]
        self.tokens, self.device = source.shape[0], source.device
        self.width = self.heads * 128
        self.sequences = metadata.query_start_loc.numel() - 1
        self.columns = metadata.state_indices.shape[1]
        self.drafts = state.convolution.shape[-1] - 3
        if (self.heads not in HEAD_COUNTS or state.recurrent.shape[2:] != (128, 128)
                or state.convolution.shape[:2] != (state.recurrent.shape[0], 3 * self.width)):
            raise ValueError("GLM KDA state must match one supported rank-count head partition")
        if metadata.num_tokens.item() or metadata.num_seqs.item():
            raise ValueError("KDA preparation requires zero live counts")
        if mode == "decode" and (self.tokens > self.sequences * self.columns or self.columns > self.drafts + 1):
            raise ValueError("KDA decode capacity exceeds metadata or convolution history")
        if records is not None and mode != "decode":
            raise ValueError("KDA verification records belong to decode rounds")
        self.module, self.weights, self.scratch = module, weights, scratch
        self.records = self.verification = None
        self.state, self.metadata, self.mode = state, metadata, mode
        module.o_norm.weight = torch.nn.Parameter(weights["o_norm.weight"], requires_grad=False)
        self.projections = {}
        self.side_chain = scratch.split
        self.side_chains = ()
        if self.side_chain:
            self._prepare_side_chain(weights, source, scratch, session=session, name=name, workspace=workspace)
        else:
            self._prepare_projections(weights, source, scratch, session=session, name=name, workspace=workspace)
        self._prepare_recurrence(module, weights, scratch, state, metadata, records,
                                 mode=mode, session=session, name=name, workspace=workspace,
                                 checkpoint_export=checkpoint_export)

    def _prepare_side_chain(self, weights, source, scratch, *, session, name, workspace):
        """q/k/v alone on the main stream; f_a -> f_b, b and g_a -> g_b on the auxiliary stream beside it.

        The checkpoint group keeps its stored parts (MXFP8 q/k/v, BF16 b + f_a); the BF16 part is split by rows
        (contiguous views of the same storage), so the weights and every dot product are unchanged.
        """
        from ignition.components.weights import GroupedLinearWeights

        group = weights["input_proj"]
        width = 3 * self.width
        if (not isinstance(group, GroupedLinearWeights) or len(group.parts) != 2
                or tuple(group.widths) != (width, self.heads + 128)
                or not isinstance(group.parts[1], torch.Tensor)):
            raise ValueError("KDA side chain requires the q/k/v group followed by one BF16 b + f_a part")
        qkv_weight, gates = group.parts
        def project(key, weight, input, output):
            self.projections[key] = prepare_projection(input, weight, output, session=session,
                name=f"{name}.{key}", workspace=workspace, activation_mode="auto")
        project("input_proj", qkv_weight, source, scratch.projected_qkv)
        project("b_proj", gates[:self.heads], source, scratch.raw_beta)
        project("f_a_proj", gates[self.heads:], source, scratch.forget_input)
        project("f_b_proj", weights["f_b_proj"], scratch.forget_input, scratch.raw_g)
        project("g_a_proj", weights["g_a_proj"], source, scratch.gate_input)
        project("g_b_proj", weights["g_b_proj"], scratch.gate_input, scratch.z)
        project("o_proj", weights["o_proj"], scratch.gated, scratch.output)
        self.gate_overlap = None
        if workspace.overlap:
            main = self.projections["input_proj"]
            parallel = tuple(main.plans) if hasattr(main, "plans") else (main.plan,)
            self.side_chains = tuple(ConcurrentProjectionChain(tuple(self.projections[key] for key in keys),
                    parallel_plans=parallel, session=session, workspace=workspace)
                for keys in (("g_a_proj", "g_b_proj"), ("f_a_proj", "f_b_proj"), ("b_proj",)))
            # Tests and callers read gate_overlap as "the output-gate chain overlaps the input projection".
            self.gate_overlap = self.side_chains[0]

    def _prepare_projections(self, weights, source, scratch, *, session, name, workspace):
        for key, input, output in (
            ("input_proj", source, scratch.projected_input),
            ("f_b_proj", scratch.forget_input, scratch.raw_g),
            ("g_a_proj", source, scratch.gate_input),
            ("g_b_proj", scratch.gate_input, scratch.z),
            ("o_proj", scratch.gated, scratch.output),
        ):
            self.projections[key] = prepare_projection(input, weights[key], output,
                session=session, name=f"{name}.{key}", workspace=workspace, activation_mode="auto",
                # Convolution consumes the projected rows after this staging has been copied out.
                staging=scratch.convolved_qkv if key == "input_proj" else None)
        # The output-gate chain runs beside the input projections, or after them on the main stream.
        self.gate_overlap = None
        if workspace.overlap:
            self.gate_overlap = ConcurrentProjectionChain(
                (self.projections["g_a_proj"], self.projections["g_b_proj"]),
                parallel_plans=(*self.projections["input_proj"].plans, self.projections["f_b_proj"].plan)
                    if hasattr(self.projections["input_proj"], "plans") else
                    (self.projections["input_proj"].plan, self.projections["f_b_proj"].plan),
                session=session, workspace=workspace)

    def _prepare_recurrence(self, module, weights, scratch, state, metadata, records, *,
                            mode, session, name, workspace, checkpoint_export):
        conv_args = {
            "x": scratch.projected_qkv, "conv_weight": weights["conv1d.weight"],
            "query_start_loc": metadata.query_start_loc, "state_slot_ids": metadata.convolution_slots,
            "state_is_fresh": metadata.fresh, "num_accepted_tokens": metadata.accepted_tokens,
            "num_seqs": metadata.num_seqs, "num_tokens": metadata.num_tokens,
            "conv_state": state.convolution, "out": scratch.convolved_qkv,
        }
        conv_caps = short_conv.Caps(device=self.device, mode=mode, channels=3 * self.width,
            kernel_size=4, max_tokens=self.tokens, max_seqs=self.sequences,
            max_state_slots=state.convolution.shape[0], max_speculative_tokens=self.drafts)
        self.conv_plan = short_conv.plan(conv_caps, invocation=short_conv.invocation_from_tensors(**conv_args))
        self.conv_binding = prepare_sequence_binding(self.conv_plan, conv_args,
            session=session, name=f"{name}.short_conv", public_bind=short_conv.bind,
            mode=mode, bind_plan=True, workspace=workspace,
            set_binding=lambda binding: setattr(self, "conv_binding", binding))
        common = {
            "raw_g": scratch.raw_g.view(self.tokens, self.heads, 128), "raw_beta": scratch.raw_beta,
            "A_log": weights["A_log"], "dt_bias": weights["dt_bias"].view(self.heads, 128),
            "recurrent_state": state.recurrent, "num_seqs": metadata.num_seqs, "num_tokens": metadata.num_tokens,
            # Decode writes normalized output straight into the output projection's input.
            "output": (scratch.gated if mode == "decode" else scratch.recurrence).view(self.tokens, self.heads, 128),
        }
        self.options = {"lower_bound": -5.0}
        if mode == "decode" and records is not None and sm103.is_sm103(self.device):
            # On SM103 Triton records verification (b12x's CuTe recording runs 1.6x slower there); the gated
            # output norm follows as a separate launch.
            from .kda_recovery import PreparedKDAVerification

            self.verification = PreparedKDAVerification(mixed_qkv=scratch.convolved_qkv, raw_g=common["raw_g"],
                raw_beta=scratch.raw_beta, A_log=weights["A_log"], dt_bias=common["dt_bias"],
                state=state.recurrent, metadata=metadata, output=scratch.recurrence, records=records,
                heads=self.heads)
            self.output_norm = PreparedGatedRMSNorm(scratch.recurrence.view(self.tokens, self.heads, 128),
                scratch.z.view(self.tokens, self.heads, 128), module.o_norm.weight,
                scratch.gated.view(self.tokens, self.heads, 128), metadata.num_tokens,
                eps=module.layer_norm_epsilon)
            self.recurrent_plan = None
            return
        if mode == "decode":
            self.options["eps"] = module.layer_norm_epsilon
            # Speculative rounds record each verified token instead of persisting its FP32 state.
            caps = gdn_decode.Caps(device=self.device, max_tokens=self.tokens, max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0], key_heads=self.heads, value_heads=self.heads,
                state_index_columns=self.columns, gate_activation="sigmoid", null_state_index=0,
                recover_speculative_state=records is not None)
            args = dict(common, mixed_qkv=scratch.convolved_qkv, z=scratch.z.view(self.tokens, self.heads, 128),
                norm_weight=module.o_norm.weight, query_start_loc=metadata.query_start_loc,
                num_accepted_tokens=metadata.accepted_tokens, state_indices=metadata.state_indices)
            if records is not None:
                args.update(correction_cache=records[0], kg_cache=records[1])
                self.records = records
                self.handle = register_recording(self)
            self.recurrent_plan = gdn_decode.plan(caps, invocation=gdn_decode.invocation_from_tensors(caps, **args))
            public_bind, bind_method = gdn_decode.bind_kda, "bind_kda"
        else:
            caps = kda_prefill.Caps(device=self.device, max_tokens=self.tokens, max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0], heads=self.heads, null_state_index=0,
                checkpoint_export=checkpoint_export)
            q, k, v = (t.view(self.tokens, self.heads, 128) for t in scratch.convolved_qkv.split(self.width, dim=-1))
            args = dict(common, q=q, k=k, v=v, cu_seqlens=metadata.query_start_loc,
                initial_state_indices=metadata.initial_state_indices, final_state_indices=metadata.final_state_indices,
                checkpoint_state_indices=metadata.checkpoint_state_indices, checkpoint_offsets=metadata.checkpoint_offsets)
            self.recurrent_plan = kda_prefill.plan(caps, invocation=kda_prefill.invocation_from_tensors(**args))
            public_bind, bind_method = kda_prefill.bind, "bind"
        self.recurrence = prepare_sequence_binding(self.recurrent_plan, args,
            session=session, name=f"{name}.{mode}", public_bind=public_bind, bind_method=bind_method,
            mode=mode, workspace=workspace, run_options=self.options,
            set_binding=lambda binding: setattr(self, "recurrence", binding))
        if mode == "prefill":
            # The canonical sigmoid-gated FP32 RMSNorm, fused, writing the output projection's input.
            self.output_norm = PreparedGatedRMSNorm(scratch.recurrence.view(self.tokens, self.heads, 128),
                scratch.z.view(self.tokens, self.heads, 128), module.o_norm.weight,
                scratch.gated.view(self.tokens, self.heads, 128), metadata.num_tokens,
                eps=module.o_norm.variance_epsilon)

    def _side_chain_projections(self, source):
        s, p = self.scratch, self.projections
        if self.side_chains:
            for chain in self.side_chains:
                chain.start(source)
            p["input_proj"](source)
            # Joins follow the q/k/v launch on the main stream; each orders its chain's outputs for the readers.
            for chain in self.side_chains:
                chain.finish(source, s.projected_qkv)
            return
        p["input_proj"](source)
        p["b_proj"](source)
        p["f_b_proj"](p["f_a_proj"](source))
        p["g_b_proj"](p["g_a_proj"](source))

    def __call__(self, source):
        s, p = self.scratch, self.projections
        if self.side_chain:
            self._side_chain_projections(source)
            return self._recurrence_and_output()
        if self.gate_overlap is not None:
            self.gate_overlap.start(source)
        p["input_proj"](source)
        if self.tokens > 1:
            s.forget_input.copy_(s.projected_input[:, -128:])
        p["f_b_proj"](s.forget_input)
        if self.gate_overlap is None:
            p["g_b_proj"](p["g_a_proj"](source))
        else:
            self.gate_overlap.finish(source, s.raw_g)
        return self._recurrence_and_output()

    def _recurrence_and_output(self):
        s, p = self.scratch, self.projections
        short_conv.run(self.conv_binding)
        if self.verification is not None:
            self.verification()
            self.output_norm()
        elif self.mode == "decode" and self.records is not None:
            m = self.metadata
            _record([s.convolved_qkv, s.raw_g, s.raw_beta, s.z, self.state.recurrent, m.query_start_loc,
                     m.state_indices, m.num_seqs, m.num_tokens], s.gated, *self.records, self.handle)
        elif self.mode == "decode":
            gdn_decode.run_kda(self.recurrence, **self.options)
        else:
            kda_prefill.run(self.recurrence, **self.options)
            self.output_norm()
        return p["o_proj"](s.gated)

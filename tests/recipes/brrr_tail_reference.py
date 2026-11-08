# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""FP64 stage references and per-operator error bounds for the TP3 BRRR layer tail (test support only).

Adapted from the persistent-decoder qualification in NVIDIA's proof of concept (persistent_validation.py): every
fused stage is recomputed in FP64 from the stage's own inputs and the kernel's output is accepted only within an
arithmetic bound built from

* gamma(n) = n*u / (1 - n*u), u = 2^-24, times the magnitude sum |a| @ |w| of each FP32 reduction of length n
  (n = K plus the 32-wide MMA blocks and k-part partial sums added on top of it),
* half an ulp of the declared output format at every quantization seam (BF16 outputs: half the local BF16 spacing
  of the kernel's actual value; E4M3 block-quantized activations: half the E4M3 spacing at the block scale),
* and the kernel's activation quantizers emulated in FP64, because its quantized intermediates are not observable
  (one launch; only the stage outputs are written back). An element whose FP64 value lies within the stage's
  uncertainty of a rounding boundary ("fragile") may round differently in the kernel, so it receives one grid step
  of slack that is propagated through the next GEMM as slack @ |w|; a 32-group whose amax lies within the
  uncertainty of a scale threshold may be quantized at the neighbouring power-of-two scale and receives half a step
  at both scales. Non-fragile elements receive no slack: the emulated quantized value is then the kernel's.

Two uncertainty tracks (class Err): `bound` is the worst-case gamma(n) term and is what the kernel's output is
accepted against; `frag` is a statistical FP32 error model, FRAG_SIGMA * u * |a| @ |w| (32 sigma of the rounding
random walk, whose typical size is ~0.2 u |a| @ |w|), and only decides which roundings may flip. The worst case
would mark more than half of the E4M3 inputs fragile and the propagated slack would exceed the gamma term 200x,
making the mlp_output bound vacuous (a 100% error would pass); with the statistical choice the bounds come out at a
few percent of the stage output (measured on an FP32 emulation of the kernel: about 5% of |y| on the attention
output, under 1% on an expert's down projection, 6% of |mlp_output|; the emulation fills about half of them, the
rope negative control exceeds them about 70x) while every deterministic term (BF16/E4M3 half-ulps, propagated slack,
reduction order) remains rigorous. Inputs that are BF16-exact in both the kernel and the emulation (z before WO-B,
x8, the shared activation) have deterministic ties: one in sixteen BF16 values sits exactly on an E4M3 midpoint
and both sides round it to even, so those are not fragile (see _uncertainty).

The reference follows the kernel's declared arithmetic (brrr/kernels/decoder.cu), not Flare's stock path:

  attention  o (BF16) -> inverse RoPE in FP32 (angle = fp32(pos) * fp32(freq), sincosf) -> BF16 -> F16 -> WO-A
             (f16 MMA, FP32 accumulate, E8M0 32x32 weight blocks) -> BF16 -> E4M3 per 32 cols (amax floor 1e-4)
             -> WO-B (block-scaled MMA) -> BF16 rank partial -> FP32 sum of the three ranks -> BF16 y
             -> resid1[j] = bf16(post[j] * y + sum_k comb[k, j] * resid0[k])            [observable: ffn_hc.residual]
  FFN mHC    pre/post/comb from resid1 (FP32, 20 Sinkhorn iterations)                   [observable: pre_mix/post/comb]
  FFN input  xc = bf16(sum_j attn_mix[j] * resid1[j]); x8 = bf16(norm * xc * rsqrt(mean xc^2 + eps))
  router     logits = x8 @ gate^T (bf16 MMA, 40 k-parts added in 2^-32 fixed point), score = sqrt(softplus),
             key = score + bias, top-6                                                   [observable: route_counts probe]
  routed     xq = E4M3(x8); h = xq @ w13_e^T (MXFP4 experts); a = swiglu(h) in FP32 (no BF16 seam);
             hq = E4M3(a); y_e = hq @ w2_e^T; routed = sum_i w_i * y_e_i (FP32 atomics)
  shared     s = xq @ w13^T; a = bf16(swiglu(bf16(g), bf16(u))); hs = E4M3(a); shared = hs @ w2^T
  reduce     partial = bf16(bf16(routed) + bf16(shared)); mlp = bf16(sum of the three ranks)   [observable: mlp_output]

Stage inputs: the attention stage starts from the shared test inputs; the FFN stages start from the kernel's own
resid1 (an exact BF16 kernel output), so each observable output is checked against the FP64 image of its own
input: a wrong attention branch shows on resid1, a wrong FFN on the mHC scalars, the routing or mlp_output.
Weights are decoded in FP64 from the checkpoint bytes the kernel's planes were packed from (E4M3 values with E8M0
32x32 block scales for the dense projections, packed E2M1 nibbles with E8M0 per-32 scales for the routed experts;
the low nibble is the even column as in b12x's reference), which also checks b12x's repack, the CSF scale expansion
and the merged-bank slot addressing. The FP64 top-k must equal the kernel's routing except for pairs whose FP64 key
margin is within the propagated key error bound; the mHC scalars use rtol 2e-5 / atol 4e-5 as in the proof of concept.

Deliberate approximations the FP64 reference documents (and the bound allows) rather than fails:
* RoPE angles are formed in FP32 (both the kernel and Flare's Triton rotary do fp32(pos) * fp32(freq)); at
  position 4096 that is ~1e-4 rad off the exact angle, far beyond any gamma term, so the reference uses the same
  FP32 angle and only cos/sin are exact.
* Activations are E4M3 block-quantized (32 columns, power-of-two scale, amax floor 1e-4) before WO-B, the gate/up
  projections and the down projections; the routed SwiGLU is quantized straight from FP32 while the shared one
  rounds gate, up and the activation to BF16 first.
* WO-A rounds its output to BF16 before quantizing to E4M3 (two seams), and the attention/MLP reductions round the
  rank partials to BF16 before the FP32 sum (three half-ulps plus gamma(2)).
* The gate logits are exact fixed-point sums of 40 FP32 k-part partials (2^-33 per part), not one FP32 chain.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

U = 2.0**-24  # FP32 unit roundoff
FRAG_SIGMA = 32.0  # statistical FP32 error model for the fragility decision: FRAG_SIGMA * u * |a| @ |w|
AMAX_FLOOR = 1e-4  # brrr/kernels/common.cuh AMAX_FLOOR (activation quantizer)
E4M3_MAX = 448.0
E2M1_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
SILU_LIPSCHITZ = 1.1  # max |d/dg (g * sigmoid(g))| = 1.0998
MIX_RTOL, MIX_ATOL = 2e-5, 4e-5  # mHC scalars (persistent_validation._mix)
HID, HC, HDIM, ROPE, GROUP_WIDTH = 5120, 4, 512, 64, 1024


def gamma(n):
    value = float(n) * U
    return value / (1.0 - value)


class Err:
    """Elementwise uncertainty of a kernel FP32 value against its FP64 reference (two tracks, see module doc)."""

    __slots__ = ("bound", "frag")

    def __init__(self, bound, frag=None):
        self.bound = bound
        self.frag = bound if frag is None else frag

    def __add__(self, other):
        if isinstance(other, Err):
            return Err(self.bound + other.bound, self.frag + other.frag)
        return Err(self.bound + other, self.frag + other)

    __radd__ = __add__

    def __mul__(self, factor):
        return Err(self.bound * factor, self.frag * factor)

    __rmul__ = __mul__

    def map(self, fn):
        return Err(fn(self.bound), fn(self.frag))


def _frag_of(err):
    return err.frag if isinstance(err, Err) else err


def _uncertainty(x, err):
    """Fragility radius: the statistical uncertainty plus, for values that are not FP32-representable, one FP32
    rounding of the emulation's own cast. A BF16-exact input with zero uncertainty is held exactly by the kernel
    too, so its ties are resolved identically (round to nearest even) and are not fragile."""
    exact = x.float().double() == x
    return _frag_of(err).double().abs() + torch.where(exact, torch.zeros_like(x), U * x.abs())


def _exp_floor(x):
    """floor(log2(|x|)) as an exact integer tensor (frexp based; |x| = 0 maps to a tiny exponent)."""
    _, exponent = torch.frexp(x.abs().clamp_min(2.0**-200))
    return exponent - 1


def bf16_ulp(magnitude):
    """BF16 spacing just above |x| (max of the two neighbouring spacings), FP64 in, FP64 out."""
    exponent = _exp_floor(magnitude).clamp_min(-126)
    return torch.ldexp(torch.ones_like(magnitude), exponent - 7)


def half_ulp(actual):
    """Half the local spacing of a BF16 or FP32 tensor at its actual values (persistent_validation._rounding)."""
    magnitude = actual.detach().double().abs()
    if actual.dtype == torch.bfloat16:
        return 0.5 * bf16_ulp(magnitude)
    if actual.dtype == torch.float32:
        exponent = _exp_floor(magnitude).clamp_min(-126)
        return 0.5 * torch.ldexp(torch.ones_like(magnitude), exponent - 23)
    raise TypeError(f"half_ulp expects BF16 or FP32 outputs, got {actual.dtype}")


def _is_pow2(x):
    mantissa, _ = torch.frexp(x.abs())
    return (x != 0) & (mantissa == 0.5)


def round_bf16(x, err):
    """Emulate the kernel's FP32 -> BF16 rounding of a value known to FP64 within the uncertainty `err`.

    Returns (q, slack): q is bf16(x) in FP64; |kernel's bf16 - q| <= slack elementwise. slack is zero where the
    whole uncertainty interval rounds to q, else the uncertainty plus one BF16 step.
    """
    x = x.double()
    delta = _uncertainty(x, err)
    q = x.float().to(torch.bfloat16).double()
    above = bf16_ulp(q.abs())
    below = torch.where(_is_pow2(q), 0.5 * above, above)
    distance = torch.minimum((x.abs() - (q.abs() + 0.5 * above)).abs(), (x.abs() - (q.abs() - 0.5 * below)).abs())
    fragile = (distance <= delta) & (delta > 0)
    return q, torch.where(fragile, delta + bf16_ulp(x.abs() + delta), torch.zeros_like(x))


def _e4m3_spacing(u):
    """E4M3 grid spacing at |u| (u already divided by the block scale); subnormal spacing 2^-9 below 2^-6."""
    u = u.abs().clamp_max(E4M3_MAX)
    exponent = _exp_floor(u)
    return torch.where(u >= 2.0**-6, torch.ldexp(torch.ones_like(u), exponent - 3), torch.full_like(u, 2.0**-9))


def _scale_exponent(amax, floor):
    """quant_exp(): ceil(log2(max(amax, floor) / 448)) on FP32 bits, exactly as the kernel computes it."""
    t = torch.maximum(amax.float(), torch.tensor(floor, dtype=torch.float32, device=amax.device))
    t = t * torch.tensor(1.0 / E4M3_MAX, dtype=torch.float32, device=amax.device)
    bits = t.contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def quantize_e4m3(x, err, *, floor=AMAX_FLOOR):
    """Emulate the kernel's per-32-column E4M3 quantizer (quant_exp/quant8/quant_byte in common.cuh).

    x [..., N] (N % 32 == 0) is the FP64 stage value, err its uncertainty against the kernel's FP32 value. Values
    are scaled by the group's power-of-two scale in FP32, rounded to nearest even and saturated at +-448. Returns
    (q, slack) with q the dequantized emulation and |kernel's dequantized value - q| <= slack: zero for elements
    the uncertainty cannot move across an E4M3 rounding boundary, one grid step for fragile elements, half a step
    at this and at the kernel's possible other scale for every element of a group whose amax uncertainty changes
    the scale exponent.
    """
    x = x.double()
    delta = _uncertainty(x, err)
    shape = x.shape
    if shape[-1] % 32:
        raise ValueError("E4M3 quantization groups are 32 columns")
    g, dg = x.reshape(-1, 32), delta.reshape(-1, 32)
    amax = g.abs().amax(-1)
    exponent = _scale_exponent(amax, floor)
    scale = torch.ldexp(torch.ones_like(amax), exponent)[:, None]
    u = g / scale
    q_u = u.float().clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float().double()
    above = _e4m3_spacing(q_u)
    below = torch.where(_is_pow2(q_u) & (q_u.abs() > 2.0**-6), 0.5 * above, above)
    distance = torch.minimum((u.abs() - (q_u.abs() + 0.5 * above)).abs(), (u.abs() - (q_u.abs() - 0.5 * below)).abs()) * scale
    element_fragile = (distance <= dg) & (dg > 0)
    reach = dg.amax(-1) + torch.where(amax.float().double() == amax, torch.zeros_like(amax), U * amax)
    low, high = _scale_exponent((amax - reach).clamp_min(0.0), floor), _scale_exponent(amax + reach, floor)
    group_fragile = ((low != exponent) | (high != exponent))[:, None]
    other_scale = torch.ldexp(torch.ones_like(amax), torch.maximum(high, exponent))[:, None]
    step_here = _e4m3_spacing((g.abs() + dg) / scale) * scale
    half_other = 0.5 * _e4m3_spacing((g.abs() + dg) / other_scale) * other_scale
    slack = torch.where(group_fragile, dg + 0.5 * step_here + half_other,
                        torch.where(element_fragile, dg + step_here, torch.zeros_like(g)))
    return (q_u * scale).reshape(shape), slack.reshape(shape)


def dot(a, slack_a, w, *, n):
    """FP64 a @ w^T; Err(bound = gamma(n) |a| @ |w|^T + slack_a @ |w|^T, frag = FRAG_SIGMA u |a| @ |w|^T + slack)."""
    w_abs = w.abs()
    magnitude, propagated = a.abs() @ w_abs.T, slack_a @ w_abs.T
    return a @ w.T, Err(gamma(n) * magnitude + propagated, FRAG_SIGMA * U * magnitude + propagated)


def swiglu_bound(gate, up, err_gate, err_up, limit):
    """FP64 swiglu(gate, up) with the kernel's clamps, and the propagated uncertainty of its FP32 evaluation."""
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    silu = gate * torch.sigmoid(gate)
    value = silu * up
    own = value.abs()
    return value, err_gate * (SILU_LIPSCHITZ * up.abs()) + err_up * silu.abs() + Err(gamma(8) * own, 8 * U * own)


def check(actual, expected, bound, *, rows=None):
    """Elementwise |actual - expected| <= bound over the selected rows; returns a report dict (fill = dev / bound)."""
    deviation = (actual.detach().double() - expected).abs()
    finite = bool(torch.isfinite(actual).all())
    if rows is not None:
        deviation, bound = deviation[rows], bound[rows]
    if deviation.numel() == 0:
        return dict(ok=finite, checked=0, fill=0.0, max_dev=0.0, over=0)
    fill = deviation / bound.clamp_min(1e-300)
    worst = int(fill.argmax())
    return dict(ok=finite and bool((deviation <= bound).all()), checked=int(deviation.numel()), fill=float(fill.max()),
                max_dev=float(deviation.max()), worst_dev=float(deviation.flatten()[worst]),
                worst_bound=float(bound.flatten()[worst]), over=int((deviation > bound).sum()))


def check_close(actual, expected, *, rtol=MIX_RTOL, atol=MIX_ATOL):
    deviation = (actual.detach().double() - expected).abs()
    allowed = atol + rtol * expected.abs()
    return dict(ok=bool(torch.isfinite(actual).all()) and bool((deviation <= allowed).all()), checked=int(deviation.numel()),
                fill=float((deviation / allowed).max()), max_dev=float(deviation.max()), over=int((deviation > allowed).sum()))


def decode_e4m3_blocks(values_u8, scales_u8):
    """FP64 weight from E4M3 bytes [R, C] and E8M0 bytes [R/32, C/32] (32x32 blocks; byte 0 = 0 like e8m0())."""
    rows, cols = values_u8.shape
    values = values_u8.contiguous().view(torch.float8_e4m3fn).float().double()
    scale = torch.ldexp(torch.ones(scales_u8.shape, dtype=torch.float64, device=values.device), scales_u8.to(torch.int32) - 127)
    scale = torch.where(scales_u8 == 0, torch.zeros_like(scale), scale)
    return values * scale.repeat_interleave(32, 0)[:rows].repeat_interleave(32, 1)[:, :cols]


def decode_mxfp4(packed_u8, scales_u8):
    """FP64 weight from packed E2M1 nibbles [R, C/2] (low nibble = even column) and E8M0 bytes [R, C/32]."""
    rows = packed_u8.shape[0]
    table = torch.tensor(E2M1_TABLE, dtype=torch.float64, device=packed_u8.device)
    codes = torch.stack((packed_u8 & 0xF, packed_u8 >> 4), -1).reshape(rows, -1, 32).long()
    scale = torch.ldexp(torch.ones(scales_u8.shape, dtype=torch.float64, device=packed_u8.device), scales_u8.to(torch.int32) - 127)
    return (table[codes] * scale[..., None]).reshape(rows, -1)


class TailReference:
    """FP64 references and bounds for one rank's share of one BRRR layer.

    checkpoint: ignition.recipes.deepseek41.brrr.Checkpoint (safetensors reader); the dense weights are read once
    and kept in FP64 on `device`; routed experts are read per expert (bytes cached on the host) and decoded on use.
    `last` keeps the attention-stage intermediates of the latest evaluation for diagnostics.
    """

    def __init__(self, *, layer, rank, world, group, device, checkpoint, text, frequencies, oa, inter,
                 gate, gate_bias, fn, hc_scale, hc_base, norm):
        self.layer, self.rank, self.world, self.group, self.device = layer, rank, world, group, device
        self.checkpoint, self.prefix = checkpoint, f"layers.{layer}"
        self.oa, self.inter, self.groups = int(oa), int(inter), int(oa) // GROUP_WIDTH
        self.experts, self.top_k = int(text["n_routed_experts"]), int(text["num_experts_per_tok"])
        self.eps, self.hc_eps, self.sinkhorn = float(text["rms_norm_eps"]), float(text["hc_eps"]), int(text["hc_sinkhorn_iters"])
        self.limit, self.route_scale = float(text["swiglu_limit"]), float(text["routed_scaling_factor"])
        self.frequencies = frequencies.detach().float().to(device)
        self.gate, self.gate_bias = gate.detach().double().to(device), gate_bias.detach().double().to(device)
        self.fn, self.hc_scale, self.hc_base = (t.detach().double().to(device) for t in (fn, hc_scale, hc_base))
        self.norm = norm.detach().double().to(device)
        self._dense = None
        self._expert_bytes = {}
        self.last = {}

    # -- weights -------------------------------------------------------------------------------------------------

    def _get(self, name, rows=slice(None), cols=slice(None)):
        return self.checkpoint.get(name, rows, cols)

    def _padded(self, name, rows=slice(None), cols=slice(None)):
        """Zero-filled rank shard (TP3 pads the 8 o-groups to 9; brrr.Engine.raw_padded does the same)."""
        shape = self.checkpoint.shape(name)
        r0, r1 = rows.start or 0, rows.stop if rows.stop is not None else shape[0]
        c0, c1 = cols.start or 0, cols.stop if cols.stop is not None else shape[1]
        got = self._get(name, slice(r0, max(min(r1, shape[0]), r0)), slice(c0, max(min(c1, shape[1]), c0))).view(torch.uint8)
        out = torch.zeros((r1 - r0, c1 - c0), dtype=torch.uint8)
        out[:got.shape[0], :got.shape[1]] = got
        return out.to(self.device)

    def _shard(self, size):
        return slice(self.rank * size, (self.rank + 1) * size)

    @property
    def dense(self):
        if self._dense is None:
            p, oa, inter = self.prefix, self.oa, self.inter
            u8 = lambda name, rows=slice(None), cols=slice(None): self._get(name, rows, cols).view(torch.uint8).to(self.device)
            self._dense = dict(
                wo_a=decode_e4m3_blocks(self._padded(f"{p}.attn.wo_a.weight", self._shard(oa)),
                                        self._padded(f"{p}.attn.wo_a.scale", self._shard(oa // 32))),
                wo_b=decode_e4m3_blocks(self._padded(f"{p}.attn.wo_b.weight", cols=self._shard(oa)),
                                        self._padded(f"{p}.attn.wo_b.scale", cols=self._shard(oa // 32))),
                w1=decode_e4m3_blocks(u8(f"{p}.ffn.shared_experts.w1.weight", self._shard(inter)),
                                      u8(f"{p}.ffn.shared_experts.w1.scale", self._shard(inter // 32))),
                w3=decode_e4m3_blocks(u8(f"{p}.ffn.shared_experts.w3.weight", self._shard(inter)),
                                      u8(f"{p}.ffn.shared_experts.w3.scale", self._shard(inter // 32))),
                w2=decode_e4m3_blocks(u8(f"{p}.ffn.shared_experts.w2.weight", cols=self._shard(inter)),
                                      u8(f"{p}.ffn.shared_experts.w2.scale", cols=self._shard(inter // 32))))
        return self._dense

    def expert(self, index):
        """FP64 (w1 gate [inter, HID], w3 up [inter, HID], w2 down [HID, inter]) of routed expert `index`, rank shard."""
        if index not in self._expert_bytes:
            p, inter = f"{self.prefix}.ffn.experts.{index}", self.inter
            rows, cols = self._shard(inter), self._shard(inter // 2)
            self._expert_bytes[index] = tuple(self._get(name, r, c).view(torch.uint8) for name, r, c in (
                (f"{p}.w1.weight", rows, slice(None)), (f"{p}.w1.scale", rows, slice(None)),
                (f"{p}.w3.weight", rows, slice(None)), (f"{p}.w3.scale", rows, slice(None)),
                (f"{p}.w2.weight", slice(None), cols), (f"{p}.w2.scale", slice(None), self._shard(inter // 32))))
        w1, s1, w3, s3, w2, s2 = (t.to(self.device) for t in self._expert_bytes[index])
        return decode_mxfp4(w1, s1), decode_mxfp4(w3, s3), decode_mxfp4(w2, s2)

    # -- stages --------------------------------------------------------------------------------------------------

    def _all_reduce(self, *tensors):
        stacked = torch.stack([t.double().cpu() for t in tensors])
        dist.all_reduce(stacked, group=self.group)
        return tuple(t.to(self.device) for t in stacked)

    def attention(self, attended, positions, resid0, post0, comb0, resid1_actual):
        """resid1 reference and bound from the shared inputs (inverse RoPE, WO-A, WO-B, all-reduce, residual mix)."""
        m = attended.shape[0]
        a = attended.detach().double().reshape(m, -1, HDIM).clone()
        slack = torch.zeros_like(a)
        pairs = torch.arange(ROPE // 2, device=self.device)
        angle = (positions.to(torch.float32)[:, None] * self.frequencies[pairs][None, :]).double()  # FP32 angle: declared
        cos, sin = torch.cos(angle)[:, None, :], torch.sin(angle)[:, None, :]
        even, odd = a[..., HDIM - ROPE::2].clone(), a[..., HDIM - ROPE + 1::2].clone()
        # FP32 products and add (gamma(3)), sincosf within 2 ulp (2^-22 absolute on cos/sin).
        delta = gamma(3) * (even.abs() * cos.abs() + odd.abs() * sin.abs()) + 2.0**-22 * (even.abs() + odd.abs())
        for columns, value in ((slice(HDIM - ROPE, None, 2), even * cos + odd * sin),
                               (slice(HDIM - ROPE + 1, None, 2), odd * cos - even * sin)):
            a[..., columns], slack[..., columns] = round_bf16(value, delta)
        slack = slack + torch.where(a.abs() < 2.0**-14, torch.full_like(a, 2.0**-25), torch.zeros_like(a))  # BF16 -> F16
        a, slack = a.reshape(m, self.groups, -1), slack.reshape(m, self.groups, -1)
        wo_a = self.dense["wo_a"].reshape(self.groups, GROUP_WIDTH, -1)
        k = a.shape[-1]
        z = torch.einsum("mgk,gnk->mgn", a, wo_a).reshape(m, self.oa)
        magnitude = torch.einsum("mgk,gnk->mgn", a.abs(), wo_a.abs()).reshape(m, self.oa)
        propagated = torch.einsum("mgk,gnk->mgn", slack, wo_a.abs()).reshape(m, self.oa)
        err_z = Err(gamma(k + k // 32 + 32) * magnitude + propagated, FRAG_SIGMA * U * magnitude + propagated)
        z, slack_z = round_bf16(z, err_z)  # finish_woa: bf() before the E4M3 quantizer
        z, slack_z = quantize_e4m3(z, slack_z)
        y, err_y = dot(z, slack_z, self.dense["wo_b"], n=self.oa + self.oa // 32 + 32)
        partial_bound = err_y.bound + 0.5 * bf16_ulp(y.abs() + err_y.bound)  # rank partial posted as BF16
        total, bound, magnitude = self._all_reduce(y, partial_bound, y.abs())
        delta_total = bound + gamma(2) * magnitude
        delta_total = delta_total + 0.5 * bf16_ulp(total.abs() + delta_total)  # reduced y stored as BF16
        post, comb, r0 = post0.detach().double(), comb0.detach().double(), resid0.detach().double()
        resid1 = post[..., None] * total[:, None, :] + torch.einsum("bkj,bkh->bjh", comb, r0)
        magnitude = post.abs()[..., None] * total.abs()[:, None, :] + torch.einsum("bkj,bkh->bjh", comb.abs(), r0.abs())
        bound = post.abs()[..., None] * delta_total[:, None, :] + gamma(5) * magnitude + half_ulp(resid1_actual)
        self.last = dict(z=z, slack_z=slack_z, y=y, err_y=err_y, total=total, delta_total=delta_total)
        return resid1, bound

    def mix(self, resid1):
        """FFN mHC scalars (pre, post, comb) in FP64 from the kernel's resid1."""
        flat = resid1.detach().double().flatten(1)
        projected = (flat @ self.fn.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        pre = torch.sigmoid(projected[:, :HC] * self.hc_scale[0] + self.hc_base[:HC]) + self.hc_eps
        post = 2 * torch.sigmoid(projected[:, HC:2 * HC] * self.hc_scale[1] + self.hc_base[HC:2 * HC])
        comb = torch.softmax(projected[:, 2 * HC:].reshape(-1, HC, HC) * self.hc_scale[2] + self.hc_base[2 * HC:].reshape(HC, HC), -1)
        comb = comb + self.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    def router(self, resid1, attn_mix):
        """FFN input (BF16 x8 with slack), router keys with their uncertainty, FP64 top-k."""
        r1, pre = resid1.detach().double(), attn_mix.detach().double()
        terms = pre[..., None] * r1
        xc, slack_xc = round_bf16(terms.sum(1), gamma(HC) * terms.abs().sum(1))
        xsq = xc.square().sum(-1, keepdim=True)
        cross = (2 * xc.abs() * slack_xc + slack_xc.square()).sum(-1, keepdim=True)
        err_xsq = Err(gamma(HID + HID // 32 + 1) * xsq + cross, FRAG_SIGMA * U * xsq + cross)
        v = xsq / HID + self.eps
        err_v = err_xsq * (1.0 / HID) + 2 * U * v
        rs = torch.rsqrt(v)
        err_rs = err_v * (0.5 * rs / v) + 2.0**-22 * rs
        x8 = self.norm * xc * rs
        err_x8 = err_rs * (self.norm.abs() * xc.abs()) + self.norm.abs() * rs * slack_xc + gamma(2) * x8.abs()
        x8, slack_x8 = round_bf16(x8, err_x8)
        logits, err_logits = dot(x8, slack_x8, self.gate, n=128 + 8)  # FP32 only within one 128-k part
        err_logits = err_logits + (40 * 2.0**-33 + U * logits.abs())  # fixed-point k-part sums, final FP32 rounding
        softplus = torch.nn.functional.softplus(logits)
        err_sp = err_logits * torch.sigmoid(logits) + (U + 2.0**-21 * softplus)  # 1 + expf rounding, expf/logf ulps
        score = torch.sqrt(softplus)

        def through_sqrt(delta):
            safe = softplus > 4 * delta
            return torch.where(safe, delta / (2 * torch.sqrt((softplus - delta).clamp_min(1e-300))), torch.sqrt(delta))

        err_score = err_sp.map(through_sqrt) + 2.0**-23 * score
        key = score + self.gate_bias
        err_key = err_score + U * key.abs()
        return dict(x8=x8, slack_x8=slack_x8, logits=logits, score=score, err_score=err_score, key=key,
                    err_key=err_key, top=key.topk(self.top_k, dim=-1).indices)

    def check_routing(self, kernel_rows, keys):
        """Kernel's per-row expert sets vs the FP64 top-k: every swap must lie within the key error bound.

        Returns (ids [m, top_k] of the kernel's choice or -1 rows, flipped-row mask, swap count, errors).
        """
        m = kernel_rows.shape[0]
        key, delta, top = keys["key"].cpu(), keys["err_key"].bound.cpu(), keys["top"].cpu()
        counts = kernel_rows.cpu().long()
        ids = torch.full((m, self.top_k), -1, dtype=torch.long)
        flipped = torch.zeros((m,), dtype=torch.bool)
        swaps, errors = 0, []
        for row in range(m):
            chosen = counts[row].nonzero().flatten()
            if len(chosen) != self.top_k or int(counts[row].sum()) != self.top_k:
                flipped[row] = True  # reported by the stock reconciliation as a routing error
                continue
            ids[row] = chosen
            kernel_set, reference_set = set(chosen.tolist()), set(top[row].tolist())
            dropped = sorted(reference_set - kernel_set, key=lambda e: -float(key[row, e]))
            added = sorted(kernel_set - reference_set, key=lambda e: -float(key[row, e]))
            for old, new in zip(dropped, added):
                margin = float(key[row, old] - key[row, new])
                allowed = float(delta[row, old] + delta[row, new])
                swaps += 1
                flipped[row] = True
                if margin > allowed:
                    errors.append(f"row {row}: kernel routed expert {new} instead of {old} at FP64 key margin {margin:.3e} > bound {allowed:.3e}")
        return ids, flipped, swaps, errors

    def ffn(self, keys, ids, mlp_actual):
        """mlp_output reference and bound from the FFN input: routed experts with the kernel's routing, shared
        expert, BF16 rank partial, all-reduce. Rows with ids == -1 get a zero reference (callers exclude them)."""
        x8, slack_x8, score, err_score = keys["x8"], keys["slack_x8"], keys["score"], keys["err_score"]
        m, device = x8.shape[0], x8.device
        xq, slack_xq = quantize_e4m3(x8, slack_x8)
        ids = ids.to(device)
        valid = (ids >= 0).all(1)
        safe_ids = ids.clamp_min(0)
        s, ds = score.gather(1, safe_ids), err_score.bound.gather(1, safe_ids)
        total = s.sum(1, keepdim=True)
        w = self.route_scale * s / total
        delta_w = self.route_scale * (ds / total + s * ds.sum(1, keepdim=True) / total.square()) + gamma(self.top_k + 2) * w.abs()
        routed = torch.zeros((m, HID), dtype=torch.float64, device=device)
        bound_routed, magnitude = torch.zeros_like(routed), torch.zeros_like(routed)
        n_up, n_down = HID + HID // 32 + 16, self.inter + self.inter // 32 + 8
        for expert in sorted(set(ids[valid].flatten().tolist())):
            tokens, slots = ((ids == expert) & valid[:, None]).nonzero(as_tuple=True)
            w1, w3, w2 = self.expert(expert)
            gate, err_gate = dot(xq[tokens], slack_xq[tokens], w1, n=n_up)
            up, err_up = dot(xq[tokens], slack_xq[tokens], w3, n=n_up)
            act, err_act = swiglu_bound(gate, up, err_gate, err_up, self.limit)  # finish_r1: FP32, no BF16 seam
            hq, slack_hq = quantize_e4m3(act, err_act)
            y, err_y = dot(hq, slack_hq, w2, n=n_down)
            weight, delta_weight = w[tokens, slots, None], delta_w[tokens, slots, None]
            routed.index_add_(0, tokens, weight * y)
            bound_routed.index_add_(0, tokens, weight.abs() * err_y.bound + delta_weight * y.abs())
            magnitude.index_add_(0, tokens, (weight * y).abs())
            del w1, w3, w2
        bound_routed = bound_routed + gamma(self.top_k + 1) * magnitude  # w * v products and 6 FP32 route adds
        dense = self.dense
        gate, err_gate = dot(xq, slack_xq, dense["w1"], n=n_up)
        up, err_up = dot(xq, slack_xq, dense["w3"], n=n_up)
        gate, slack_gate = round_bf16(gate, err_gate)  # finish_sh1: bf(swiglu(bf(g), bf(u)))
        up, slack_up = round_bf16(up, err_up)
        act, err_act = swiglu_bound(gate, up, Err(slack_gate), Err(slack_up), self.limit)
        act, slack_act = round_bf16(act, err_act)
        hs, slack_hs = quantize_e4m3(act, slack_act)
        shared, err_shared = dot(hs, slack_hs, dense["w2"], n=n_down)
        partial = routed + shared
        bound_partial = (bound_routed + 0.5 * bf16_ulp(routed.abs() + bound_routed)
                         + err_shared.bound + 0.5 * bf16_ulp(shared.abs() + err_shared.bound) + U * partial.abs())
        bound_partial = bound_partial + 0.5 * bf16_ulp(partial.abs() + bound_partial)
        total, bound, magnitude = self._all_reduce(partial, bound_partial, partial.abs())
        bound = bound + gamma(2) * magnitude + half_ulp(mlp_actual)
        return total, bound

    def evaluate(self, *, inputs, outputs, kernel_rows=None):
        """All stage checks for one launch. inputs: attended, residual, post, comb, attn_mix, positions (the
        kernel's inputs); outputs: the kernel's mlp_output, ffn_residual, ffn_post, ffn_comb, pre_mix; kernel_rows:
        per-row route counts from the probe (None: routing and mlp_output are not bound-checked).
        Runs collectives: call on every rank in the same order. Returns {stage: report}; reports carry "ok"."""
        reports = {}
        resid1, bound = self.attention(inputs["attended"], inputs["positions"], inputs["residual"], inputs["post"],
                                       inputs["comb"], outputs["ffn_residual"])
        reports["ffn_residual"] = check(outputs["ffn_residual"], resid1, bound)
        pre, post, comb = self.mix(outputs["ffn_residual"])
        for name, actual, expected in (("pre_mix", outputs["pre_mix"], pre), ("ffn_post", outputs["ffn_post"], post),
                                       ("ffn_comb", outputs["ffn_comb"], comb)):
            reports[name] = check_close(actual, expected)
        keys = self.router(outputs["ffn_residual"], inputs["attn_mix"])
        if kernel_rows is None:
            reports["routing"] = dict(ok=True, checked=0, swaps=0, note="route_counts unavailable: routing and mlp_output not bound-checked")
            return reports
        ids, flipped, swaps, errors = self.check_routing(kernel_rows, keys)
        reports["routing"] = dict(ok=not errors, checked=int(kernel_rows.shape[0]), swaps=swaps,
                                  flipped_rows=int(flipped.sum()), **({"error": "; ".join(errors)} if errors else {}))
        mlp, bound = self.ffn(keys, ids, outputs["mlp_output"])
        reports["mlp_output"] = check(outputs["mlp_output"], mlp, bound, rows=(~flipped).to(mlp.device))
        reports["mlp_output"]["excluded_rows"] = int(flipped.sum())
        return reports


def summarize(reports):
    """One line per launch: fill = max |deviation| / bound (must be <= 1)."""
    parts = []
    for name, report in reports.items():
        if "fill" in report:
            parts.append(f"{name} fill={report['fill']:.3f}" + (f" over={report['over']}" if report.get("over") else "")
                         + (f" excl={report['excluded_rows']}" if report.get("excluded_rows") else ""))
        else:
            parts.append(f"{name} swaps={report.get('swaps', 0)}" + ("" if report["ok"] else " ERROR"))
    return " ".join(parts)

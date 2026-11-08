# SPDX-License-Identifier: Apache-2.0
"""TP3: qkv_a's 4-way split-K reduction fused into the MLA norm + MXFP8 encode (FLARE_DS41_FUSED_QKV_NORM=1).

MEASURED 2026-11-07: bit-identical output, no speed gain (alternating A/B level with the stock kernels) -> NOT adopted;
kept opt-in for reference. Not part of any published build.

b12x reduces qkv_a's four FP32 K partials in its own kernel, writes the BF16 sum, and mla_norm_mxfp8 then reads it
back. Here the dense GEMM leaves the partials in its split-K workspace (b12x DEFERRED_SPLIT_K) and one kernel sums
them, rounds the sum to BF16 exactly as the reduce kernel stores it, and runs mla_norm_mxfp8's arithmetic on it:
bit-identical outputs, one launch and one round trip fewer per layer. Graph widths whose GEMM did not choose 4-way
split-K keep the stock norm kernel.
"""

import os

import torch
import triton
import triton.language as tl
from b12x.gemm.blockscaled._quantize import _scale_offset


def fused_qkv_norm_enabled() -> bool:
    return os.environ.get("FLARE_DS41_FUSED_QKV_NORM", "0") == "1"


@triton.jit
def _split4_mla_norm_mxfp8(P, TOTAL, QW, KW, QBF, KV, Q8, ROWS, S,
                           Q_FEATURES: tl.constexpr, KV_FEATURES: tl.constexpr, EPSILON: tl.constexpr,
                           MIN_AMAX: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)
    total = TOTAL.to(tl.int64)
    width: tl.constexpr = Q_FEATURES + KV_FEATURES
    if task == 0:
        c = tl.arange(0, 2048)
        m = c < Q_FEATURES
        w = tl.load(QW + c, m, 0).to(tl.float32)
        qp = P + row * width + c
        s = tl.load(qp, m, 0.0) + tl.load(qp + total, m, 0.0)
        s += tl.load(qp + 2 * total, m, 0.0)
        s += tl.load(qp + 3 * total, m, 0.0)
        x = s.to(tl.bfloat16).to(tl.float32)
        variance = tl.sum(x * x, 0) / Q_FEATURES
        y = (x * tl.rsqrt(variance + EPSILON) * w).to(QBF.dtype.element_ty)
        tl.store(QBF + row * Q_FEATURES + c, y, m)
        grouped = tl.reshape(y.to(tl.float32), (64, 32))
        amax = tl.maximum(tl.max(tl.abs(grouped), 1), MIN_AMAX)
        bits = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
        exponent = ((bits >> 23) & 255).to(tl.int32) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = ((exponent + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        quant = tl.minimum(tl.maximum(tl.div_rn(grouped, scale[:, None]), -448.0), 448.0).to(tl.float8e4nv)
        tl.store(Q8 + row * Q_FEATURES + c, tl.reshape(quant, (2048,)), m)
        group = tl.arange(0, 64)
        byte = (exponent + 127).to(tl.uint8)
        groups: tl.constexpr = Q_FEATURES // 32
        tl.store(ROWS + row * groups + group, byte, group < groups)
        tl.store(S + _scale_offset(row, group, groups), byte, group < groups)
    else:
        kc = tl.arange(0, 512)
        kw = tl.load(KW + kc).to(tl.float32)
        kp = P + row * width + Q_FEATURES + kc
        ks = tl.load(kp) + tl.load(kp + total)
        ks += tl.load(kp + 2 * total)
        ks += tl.load(kp + 3 * total)
        kx = ks.to(tl.bfloat16).to(tl.float32)
        ky = kx * tl.rsqrt(tl.sum(kx * kx, 0) / KV_FEATURES + EPSILON) * kw
        tl.store(KV + row * KV_FEATURES + kc, ky)


def launch_split4(partials, qw, kw, query, kv, values, rows_base, mma_base, *, eps, min_amax=1e-4):
    _, m, n = partials.shape
    _split4_mla_norm_mxfp8[(m, 2, 1)](partials, m * n, qw, kw, query, kv, values, rows_base, mma_base,
                                       Q_FEATURES=1280, KV_FEATURES=512, EPSILON=eps, MIN_AMAX=min_amax)


_SEEN = set()


def _log(kind, rows):
    if os.environ.get("FLARE_DS41_BRRR_LOG") == "1" and (kind, rows) not in _SEEN:
        _SEEN.add((kind, rows))
        print(f"[fused_qkv_norm] {kind} norm at {rows} rows", flush=True)


@torch.library.custom_op("flare::fused_qkv_norm_arm", mutates_args=("projected",))
def fused_qkv_norm_arm(projected: torch.Tensor) -> None:
    """Arms the deferral for the qkv_a GEMM that follows in this graph: runs at capture, right before the GEMM, so
    the key is live only between this op and the consumer below (never while another graph executes)."""
    from b12x._lib.dense_gemm.preparation import DEFERRED_SPLIT_K

    DEFERRED_SPLIT_K[projected.data_ptr()] = None


@fused_qkv_norm_arm.register_fake
def _fake_fused_qkv_norm_arm(projected):
    pass


@torch.library.custom_op("flare::fused_qkv_norm",
                         mutates_args=("query", "kv", "values_base", "scale_rows_base", "scale_mma_base"))
def fused_qkv_norm(projected: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor, query: torch.Tensor, kv: torch.Tensor,
                   values_base: torch.Tensor, scale_rows_base: torch.Tensor, scale_mma_base: torch.Tensor,
                   norm_handle: int, eps: float) -> None:
    from b12x._lib.dense_gemm.preparation import DEFERRED_SPLIT_K

    deferred = DEFERRED_SPLIT_K.pop(projected.data_ptr(), None)  # disarm: the key dies with this consumer
    if deferred is not None:
        partials, m, n = deferred
        if m != projected.shape[0] or n != projected.shape[1]:
            raise ValueError("deferred qkv_a partials do not match the projected rows")
        launch_split4(partials, qw, kw, query, kv, values_base, scale_rows_base, scale_mma_base, eps=eps)
        _log("fused", m)
    else:
        _log("stock", projected.shape[0])
        torch.ops.b12x.mla_norm_mxfp8_prepared(projected, qw, kw, query, kv, values_base, scale_rows_base,
                                               scale_mma_base, norm_handle)


@fused_qkv_norm.register_fake
def _fake_fused_qkv_norm(projected, qw, kw, query, kv, values_base, scale_rows_base, scale_mma_base, norm_handle, eps):
    pass


class PreparedFusedQKVNorm:
    """Replaces norm_quant(s.projected) after qkv_a: `arm()` goes right before the qkv_a projection, `__call__` right
    where the stock norm ran. Both are custom ops, so the arm/consume window is baked into each graph."""

    def __init__(self, norm_quant, eps):
        self.norm_quant, self.eps = norm_quant, eps

    def arm(self, projected):
        fused_qkv_norm_arm(projected)

    def __call__(self, projected):
        nq = self.norm_quant
        fused_qkv_norm(projected, *nq.weights, nq.query, nq.kv, *nq.encoded_bases, nq.plan.handle, self.eps)
        return nq.query, nq.kv

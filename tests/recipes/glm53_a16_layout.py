# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Logical views of b12x's W4A16 (MMA-packed) expert storage, for checkpoint comparisons.

With A16 activations on SM120 `fused_moe.prepare_weights` repacks the NVFP4 expert banks in place
(`fused_moe/_impl.py`, `WeightPreparationTransform.W4A16_PACKED`, `reuse_input_storage=True`):
`_impl.w1_fp4`/`w2_fp4` become int32 `[E, K // 16, (N // 64) * 128]` tiles produced by
`b12x/moe/_shared/kernels/w4a16/prepare.py::_repack_weight` (per expert: the `[N, K // 2]` packed
bytes viewed as int32, transposed to `[K // 8, N]`, rows rotated by `intermediate_size` for the
`w13` layout so gate rows come first, then `_repack_4bit_no_perm` into 16 x 64 MMA tiles). The
scales become the permuted FP16-lifted `e4m3_k16` grid of `_permute_nvfp4_scales`. b12x ships the
exact inverse of the weight repack (`_decode_w4a16_packed_expert_to_qweight`), which runs on CPU
tensors; both helpers here call b12x's own functions so the expectation cannot drift from the kernel.
"""

import torch
from b12x.moe._shared.kernels.w4a16.prepare import _permute_nvfp4_scales
from b12x.moe.fused_moe._impl import _decode_w4a16_packed_expert_to_qweight


def a16_logical_weights(packed, *, size_k, size_n, row_rotation=0):
    """Decode one expert's MMA-packed int32 tiles back to its `[size_n, size_k // 2]` packed bytes.

    `row_rotation` is the rotation `_repack_weight` applied (the intermediate size for the `w13`
    layout, which puts the gate rows first); the result is in checkpoint row order again.
    """
    packed = packed.cpu().view(torch.int32)
    qweight = torch.empty((size_n, size_k // 8), dtype=torch.int32)
    _decode_w4a16_packed_expert_to_qweight(packed, qweight, size_k=size_k, size_n=size_n)
    logical = qweight.view(torch.uint8).view(size_n, size_k // 2)
    if row_rotation:
        logical = torch.cat((logical[size_n - row_rotation:], logical[:size_n - row_rotation]))
    return logical


def a16_packed_scales(logical, global_scales, *, size_k, size_n, io_dtype, row_rotation=None, pad_groups=None):
    """Permute `[E, size_n, size_k // 16]` E4M3 scales into the W4A16 `[E, size_k // 16, size_n]` grid.

    This is `_prepare_w4a16_packed_weights`' scale path for `modelopt_nvfp4` sources, whose global
    scale passes through unchanged; the lifting factor depends on the largest nonzero scale of the
    batch, so pass every expert the kernel prepared together. `pad_groups=(first, last)` names the
    K groups (grid rows) that a padded partition left outside the checkpoint: the compact CSF
    decoder fills them with each row's base lifted through the W4A16 table and clamped to 240
    (`_csf_prepare.py::_pack_codes`, `tl.minimum(Table[base], 256 - 16)`), so when `logical` carries
    the compact row-base padding (`expand_csf_scales_padded(compact=True)`) those rows are clamped here.
    """
    logical = logical.cpu().view(torch.float8_e4m3fn)
    packed, _ = _permute_nvfp4_scales(logical, global_scales.cpu().float(), size_k=size_k, size_n=size_n,
                                      a_dtype=io_dtype, row_rotation=row_rotation)
    if pad_groups is not None:
        first, last = pad_groups
        bytes_ = packed.view(torch.uint8)
        bytes_[:, first:last] = bytes_[:, first:last].clamp_max(240)
    return packed

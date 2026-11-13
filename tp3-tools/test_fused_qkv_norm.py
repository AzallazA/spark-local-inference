"""Bit-exactness check: fused split-4 reduce + MLA norm/MXFP8 vs b12x's reduce kernel followed by mla_norm_mxfp8."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b12x._lib.dense_gemm.reductions import _reduce_split_k4_bf16  # noqa: E402
from b12x.quantization import mxfp8  # noqa: E402
from b12x.quantization.mla_norm_mxfp8._kernels import mla_norm_mxfp8_kernel  # noqa: E402
from ignition.recipes.deepseek41.fused_qkv_norm import launch_split4  # noqa: E402

dev = torch.device("cuda", 0)
torch.manual_seed(0)
ok = True
for rows in (1, 6, 8, 24, 48, 64):
    partials = torch.randn(4, rows, 1792, device=dev) * torch.rand(1, rows, 1, device=dev) * 3
    qw = torch.rand(1280, device=dev, dtype=torch.bfloat16) + 0.5
    kw = torch.rand(512, device=dev, dtype=torch.bfloat16) + 0.5
    outs = []
    for fused in (False, True):
        q = torch.zeros(rows, 1280, device=dev, dtype=torch.bfloat16)
        kv = torch.zeros(rows, 512, device=dev, dtype=torch.bfloat16)
        vs, rs, ms = mxfp8.dense_rows_storage_shapes(rows, 1280)
        bases = (torch.zeros(vs, device=dev, dtype=torch.float8_e4m3fn), torch.zeros(rs, device=dev, dtype=torch.uint8),
                 torch.zeros(ms, device=dev, dtype=torch.uint8))
        if fused:
            launch_split4(partials, qw, kw, q, kv, *bases, eps=1e-20)
        else:
            projected = torch.empty(rows, 1792, 1, device=dev, dtype=torch.bfloat16)
            _reduce_split_k4_bf16(partials.permute(1, 2, 0), projected, m=rows, n=1792)
            mla_norm_mxfp8_kernel[(rows, 2, 1)](projected.view(rows, 1792), qw, kw, q, kv, *bases, 1280, 512, 1e-20,
                                                1e-4, False, False)
        torch.cuda.synchronize(dev)
        outs.append((q, kv, *bases))
    same = [torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(*outs)]
    ok &= all(same)
    print(f"rows {rows:2d}: query/kv/q8/scale_rows/scale_mma bit-identical {same}")
print("PASS" if ok else "FAIL")

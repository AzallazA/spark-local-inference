"""Standalone check: BRRR flare_sparse_decode vs b12x's pure-torch compressed sparse MLA reference.

Random DeepSeek-V4.1 window (528 B) and indexed (288 B) caches packed with b12x's reference encoder, random per-row
window slots and logical indexed positions mapped through a random page table, partial lengths and -1 holes.
Prints max/mean relative error per case; bf16 rounding of P and the output bounds it around 1e-2.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brrr.cuext import extension  # noqa: E402
from b12x.attention._shared.mla.compressed_reference import (  # noqa: E402
    compressed_sparse_mla_reference, pack_deepseek_v41_cache_reference)

ext = extension("flare_sparse_decode")
dev = torch.device("cuda", int(sys.argv[1]) if len(sys.argv) > 1 else 0)
torch.manual_seed(0)
H, D = ext.H, ext.D


def case(rows, swa_page, idx_page, idx_width, swa_tokens=2048, idx_tokens=4096, table_width=64, holes=False):
    swa_kv = torch.randn(swa_tokens, D, device=dev) * 0.5
    idx_kv = torch.randn(idx_tokens, D, device=dev) * 0.5
    swa = pack_deepseek_v41_cache_reference(swa_kv, page_size=swa_page, cache_kind="swa")
    idx = pack_deepseek_v41_cache_reference(idx_kv, page_size=idx_page, cache_kind="indexed") if idx_width else None
    q = (torch.randn(rows, H, D, device=dev) * 0.3).bfloat16()
    sink = torch.randn(H, device=dev)
    swa_idx = torch.randint(0, swa.shape[0] * swa_page, (rows, 128), device=dev, dtype=torch.int32)
    swa_len = torch.randint(0, 129, (rows,), device=dev, dtype=torch.int32)
    if rows > 2:
        swa_len[1] = 0  # window-less row
    if holes:
        swa_idx[torch.rand(rows, 128, device=dev) < 0.05] = -1
    args = dict(swa_idx=swa_idx, swa_len=swa_len)
    if idx_width:
        pages = idx.shape[0]
        table = torch.randint(0, pages, (rows, table_width), device=dev, dtype=torch.int32)
        logical = torch.randint(0, table_width * idx_page, (rows, idx_width), device=dev, dtype=torch.int32)
        idx_len = torch.randint(0, idx_width + 1, (rows,), device=dev, dtype=torch.int32)
        if rows > 2:
            idx_len[1] = 0  # row 1 attends to nothing but the sink
        if holes:
            logical[torch.rand(rows, idx_width, device=dev) < 0.05] = -1
        mapped = torch.where(logical >= 0, table.gather(1, (logical.clamp_min(0) // idx_page).long()) * idx_page
                             + logical.clamp_min(0) % idx_page, -1).int()
        mapped[torch.arange(idx_width, device=dev)[None, :] >= idx_len[:, None]] = -1
    scale = D ** -0.5
    out = torch.empty_like(q)
    partial = torch.empty(rows * ext.CL * H * D, device=dev)
    stats = torch.empty(rows * ext.CL * H * 2, device=dev)
    counters = torch.zeros(rows, device=dev, dtype=torch.int32)
    for _ in range(2):  # second launch checks the self-reset counters
        ext.decode(q, out, swa, swa_page, swa_idx, swa_len, idx, idx_page,
                   logical if idx_width else None, idx_len if idx_width else None, table if idx_width else None,
                   sink, scale, partial, stats, counters)
    torch.cuda.synchronize(dev)
    # The reference treats the length as a valid PREFIX and stops at the first -1; the kernel (like b12x's production
    # page-mapping kernel) skips invalid entries wherever they are. Compact each row's valid keys to the front so both
    # see the same key set.
    def compact(indices, lengths):
        out = torch.full_like(indices, -1)
        for r in range(indices.shape[0]):
            valid = indices[r, :int(lengths[r])]
            valid = valid[valid >= 0]
            out[r, :valid.numel()] = valid
        return out
    ref_swa = compact(swa_idx, swa_len)
    kw = {}
    if idx_width:
        kw = dict(extra_k_cache=idx, extra_indices=compact(mapped, torch.full((rows,), idx_width, device=dev)),
                  extra_topk_lengths=torch.full((rows,), idx_width, device=dev, dtype=torch.int32), extra_page_size=idx_page)
    ref = compressed_sparse_mla_reference(q, swa, ref_swa, torch.full((rows,), 128, device=dev, dtype=torch.int32),
        sm_scale=scale, attn_sink=sink, swa_page_size=swa_page, cache_format="deepseek_v41", **kw)
    ref = ref[0] if isinstance(ref, tuple) else ref
    err = (out.float() - ref.float()).abs()
    rel = err.max().item() / ref.float().abs().max().item()
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    print(f"rows {rows:2d} swa_page {swa_page} idx_page {idx_page} idx_width {idx_width}: max rel err {rel:.2e}, cosine {cos:.6f}")
    return cos


cases = [(1, 128, 32, 512, False), (6, 128, 32, 512, False), (24, 128, 32, 512, False), (48, 134, 32, 512, False),
         (6, 128, 1, 512, False), (6, 128, 32, 0, False), (8, 128, 64, 512, False),
         (6, 128, 32, 512, True), (24, 128, 32, 512, True), (64, 134, 32, 512, True)]  # -1 holes mid-row; 64 = capacity
worst = min(case(r, sp, ip, w, holes=h) for r, sp, ip, w, h in cases)
print("PASS" if worst > 0.9999 else "FAIL", f"(worst cosine {worst:.6f})")

# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM decode context parallelism over two, three and four ranks: the dcp.py kernels on Triton's CPU interpreter.

The CUDA tests (`test_glm53_dcp_cuda.py`, `test_glm53_dcp_records_cuda.py`) check the same kernels on a GPU at
two and four ranks. Three ranks (DCP3) gather 3 x 512 candidates, which Triton cannot sort directly (arange and
sort lengths are powers of two), so the merge sorts 2048 keys with the last 512 masked as missing. These cases
run the actual kernels under `TRITON_INTERPRET=1` (chosen when the kernels are decorated, hence one subprocess
per case) against brute-force references: candidate pack and merge (full, ragged, tied, empty shards and empty
rows), merge plus local expansion (positions below 4 * W leave some ranks empty), prefill expansion to global
positions, the CUDA IPC context pull and the windowed pack/scatter path at 768-token pages, and the
log-sum-exp combine of partial attention with ranks whose LSE is -inf (the RoCE head exchange's Triton merge;
the PCIe path's b12x reduce-scatter is GPU-only, tp3-tools/glm-dcp3-gpu.sh stage 2).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

CASES = ("placement", "merge", "merge_expand_local", "expand_tokens", "records", "lse_combine")
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("case", CASES)
def test_dcp_kernels_with_cpu_interpreter(case, world):
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))))
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), case, str(world)],
                            env=env, text=True, capture_output=True, timeout=900)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def test_three_rank_shards_and_pages():
    from ignition.recipes.glm53.dcp import ContextShards

    for world in (1, 2, 3, 4):
        assert ContextShards.on(world - 1, world, "cpu").page_tokens == 256 * world
    with pytest.raises(ValueError, match="1, 2, 3 or 4 ranks"):
        ContextShards.on(0, 5, "cpu")
    with pytest.raises(ValueError):
        ContextShards(3, 3)
    # 2^20 is not a whole number of 768-token pages: 1366 pages, the last holding 256 positions.
    assert -(-1048576 // 768) == 1366 and 1366 * 768 - 1048576 == 512


def owner_local(t, world):
    return (t // 4) % world, (t // (4 * world)) * 4 + t % 4


def _placement(world, device):
    from ignition.recipes.glm53.dcp import ContextShards

    for rank in range(world):
        shards = ContextShards.on(rank, world, device)
        blocks = 24
        table = torch.randperm(4096)[:blocks].int().to(device)
        positions = torch.randint(-1, blocks * 256 * world, (3000,), device=device)
        positions[:16] = torch.arange(-1, 15)
        slots = shards.token_slots(positions, table.expand(positions.numel(), -1))
        for p, s in zip(positions.tolist(), slots.tolist()):
            o, l = owner_local(p, world)
            want = -1 if p < 0 or o != rank else int(table[l // 256]) * 256 + l % 256
            assert s == want, (world, rank, p, s, want)
        lengths = torch.arange(0, 3000, device=device)
        brute = torch.tensor([sum(1 for t in range(n) if owner_local(t, world)[0] == rank) for n in range(3000)])
        assert torch.equal(shards.local_tokens(lengths).cpu(), brute), "local_tokens"
        pools = torch.arange(0, 700, device=device)
        brute = torch.tensor([sum(1 for q in range(n) if q % world == rank) for n in range(700)])
        assert torch.equal(shards.local_pools(pools).cpu(), brute), "local_pools"
        # Every record a page addresses is one rank's: page p of a request covers positions [p * 256W, (p+1) * 256W).
        page = 256 * world
        for t in (0, page - 1, page, 3 * page + 5):
            assert owner_local(t, world)[1] // 256 == t // page


def _candidates(world, rows, device, generator):
    """Each rank's local top pools and scores, with full, ragged, tied, short, empty-shard and empty rows."""
    local_ids = [torch.full((rows, 512), -1, dtype=torch.int32) for _ in range(world)]
    scores = [torch.full((rows, 512), -float("inf")) for _ in range(world)]
    for rank in range(world):
        count = torch.randint(0, 513, (rows,), generator=generator).tolist()
        count[0] = 3 + rank                      # short context: fewer than 512 between the ranks
        count[1] = 512                           # every rank full
        count[2] = 0                             # an empty row
        count[3] = 0 if rank else 100            # empty shards on every rank but the first
        count[4] = 37 if rank == world - 1 else 0  # only the last rank has candidates
        for r, c in enumerate(count):
            local_ids[rank][r, :c] = torch.randperm(5000, generator=generator)[:c].int()
            # Coarse scores tie often; ties prefer the lower global pool id.
            scores[rank][r, :c] = torch.randint(0, 50, (c,), generator=generator).float() / 7
    return [t.to(device) for t in local_ids], [t.to(device) for t in scores]


def _brute_top(local_ids, scores, world, r):
    candidates = []
    for rank in range(world):
        for l, s in zip(local_ids[rank][r].tolist(), scores[rank][r].tolist()):
            if l >= 0:
                candidates.append((-s, l * world + rank))
    candidates.sort()
    return [g for _, g in candidates[:512]]


def _merge(world, device):
    from ignition.recipes.glm53.dcp import ContextShards, merge_candidates, pack_candidates

    generator = torch.Generator().manual_seed(world)
    rows = 9
    local_ids, scores = _candidates(world, rows, device, generator)
    packed = []
    for rank in range(world):
        out = torch.full((rows, 512, 2), 7.0, device=device)
        pack_candidates(local_ids[rank], scores[rank], out, ContextShards.on(rank, world, device))
        # Each rank's list is in descending key order, missing entries (-inf, -1) at the back.
        for r in range(rows):
            present = int((local_ids[rank][r] >= 0).sum())
            ids = out[r, :, 1].tolist()
            assert all(i >= 0 for i in ids[:present]) and all(i == -1 for i in ids[present:]), ("pack", rank, r)
            assert out[r, present:, 0].eq(-float("inf")).all()
            keys = [(-s, i) for s, i in zip(out[r, :present, 0].tolist(), ids[:present])]
            assert keys == sorted(keys), ("pack order", rank, r)
        packed.append(out)
    gathered = torch.cat(packed, dim=1).contiguous()
    merged = torch.full((rows, 512), 99, dtype=torch.int32, device=device)
    merge_candidates(gathered, merged)
    for r in range(rows):
        want = _brute_top(local_ids, scores, world, r)
        got = merged[r].tolist()
        # Expansion reads the first min(pools, 512) entries, so the selected pools lead in any order.
        assert sorted(got[:len(want)]) == sorted(want) and all(v == -1 for v in got[len(want):]), ("merge", world, r)


def _merge_expand_local(world, device):
    from ignition.recipes.glm53.dcp import ContextShards, merge_expand_local

    generator = torch.Generator().manual_seed(10 + world)
    # The interpreter sorts 2048 keys slowly: few rows, and four ranks (also covered on the GPU) one placement.
    seqs, blocks, rows = 3, 12, 8
    tables = torch.stack([torch.randperm(1000, generator=generator)[:blocks] for _ in range(seqs)]).int().to(device)
    requests = torch.randint(0, seqs, (rows,), dtype=torch.int32, generator=generator).to(device)
    positions = torch.randint(-1, blocks * 256 * world - 1, (rows,), generator=generator)
    # Inactive, below one group per rank (some ranks own nothing), short and long contexts.
    positions[:7] = torch.tensor([-1, 0, 3, 4 * world - 2, 4 * world - 1, 32, 1001])
    positions = positions.to(device)
    pools = torch.full((rows, 512), -1, dtype=torch.int32)
    for r in range(rows):
        complete = (int(positions[r]) + 1) // 4
        take = min(complete, 512)
        if take:
            pools[r, :take] = torch.randperm(complete, generator=generator)[:take].int()
    for split in (True,) if world == 4 else (False, True):
        candidates = torch.zeros((rows, world * 512, 2), dtype=torch.float32)
        candidates[:, :, 0] = -float("inf")
        candidates[:, :, 1] = -1
        for r in range(rows):
            listed = [0] * world
            for i, pool in enumerate(pools[r].tolist()):
                if pool >= 0:
                    owner = pool % world if split else 0
                    candidates[r, owner * 512 + listed[owner]] = torch.tensor([1000.0 - i, pool])
                    listed[owner] += 1
        candidates = candidates.to(device)
        for rank in range(world):
            shards = ContextShards.on(rank, world, device)
            output = torch.empty((rows, 2051), dtype=torch.int32, device=device)
            counts = torch.empty(rows, dtype=torch.int32, device=device)
            merged = torch.empty((rows, 512), dtype=torch.int32, device=device)
            merge_expand_local(candidates, merged, positions, requests, tables, output, counts, shards)
            for r in range(rows):
                take = int((pools[r] >= 0).sum())
                selected = merged[r].tolist()
                assert sorted(selected[:take]) == sorted(pools[r, :take].tolist()), ("merge", world, rank, split, r)
                assert all(v == -1 for v in selected[take:]), ("merge tail", world, rank, split, r)
                length = int(positions[r]) + 1
                complete = length // 4
                tokens = [pool * 4 + k for pool in selected[:take] for k in range(4)]
                tokens += list(range(complete * 4, length))
                want = []
                for t in tokens:
                    o, l = owner_local(t, world)
                    if o == rank:
                        want.append(int(tables[int(requests[r]), l // 256]) * 256 + l % 256)
                assert counts[r].item() == len(want), ("count", world, rank, split, r, counts[r].item(), len(want))
                got = output[r].tolist()
                assert got[:len(want)] == want and all(v == -1 for v in got[len(want):]), \
                    ("expand", world, rank, split, r)
                if length <= 4 * rank:
                    assert len(want) == 0, ("empty shard", world, rank, r)


def _expand_tokens(world, device):
    from ignition.recipes.glm53.dcp import ContextShards, expand_tokens, merge_candidates, pack_candidates

    # Prefill: every rank merges the same global pools and expands them to global positions.
    generator = torch.Generator().manual_seed(20 + world)
    rows = 9
    local_ids, scores = _candidates(world, rows, device, generator)
    positions = torch.randint(5000 * 4 * world, 6000 * 4 * world, (rows,), generator=generator)
    positions[0] = 4 * world + 2
    positions = positions.to(device)
    # Pool ids must lie below each row's complete pools.
    for rank in range(world):
        complete = (positions.cpu() + 1) // 4
        for r in range(rows):
            ids = local_ids[rank][r]
            live = ids >= 0
            ids[live & (ids * world + rank >= complete[r])] = -1
    packed = []
    for rank in range(world):
        out = torch.empty((rows, 512, 2), device=device)
        pack_candidates(local_ids[rank], scores[rank], out, ContextShards.on(rank, world, device))
        packed.append(out)
    gathered = torch.cat(packed, dim=1).contiguous()
    merged = torch.empty((rows, 512), dtype=torch.int32, device=device)
    merge_candidates(gathered, merged)
    output = torch.empty((rows, 2051), dtype=torch.int32, device=device)
    counts = torch.empty(rows, dtype=torch.int32, device=device)
    expand_tokens(merged, positions, output, counts)
    for r in range(rows):
        want_pools = _brute_top(local_ids, scores, world, r)
        length = int(positions[r]) + 1
        complete = length // 4
        selected = min(complete, 512)
        got = output[r].tolist()
        count = int(counts[r])
        history = sorted(t for t in got[:count] if t < complete * 4)
        expected = sorted(pool * 4 + k for pool in want_pools for k in range(4))
        # Expansion reads min(complete, 512) entries and skips missing ones, so it covers the present pools.
        assert selected >= len(want_pools) and history == expected, ("history", world, r)
        assert sorted(t for t in got[:count] if t >= complete * 4) == list(range(complete * 4, length))
        assert all(v == -1 for v in got[count:])


def _records(world, device):
    from ignition.recipes.glm53.dcp import (ContextShards, pack_records, pull_context, scatter_records,
                                            window_counts)

    pages, columns, window = 6, 4, 192
    positions = columns * 256 * world
    table = torch.tensor([3, 0, 5, 1], dtype=torch.int32, device=device)
    generator = torch.Generator().manual_seed(30 + world)
    caches = [torch.randint(-2**31, 2**31 - 1, (pages * 256, 132), dtype=torch.int32, generator=generator).to(device)
              for _ in range(world)]
    page = 256 * world
    for end in sorted({1, 3, 4, 5, 4 * world - 1, 4 * world, 4 * world + 1, 37, 1000, page - 1, page, page + 1,
                       2 * page + 255, positions - 1, positions}):
        final = torch.tensor([end], dtype=torch.int64, device=device)
        # Brute force: every position below `end` read from its owner's cache through the page table.
        brute = torch.zeros(positions, 132, dtype=torch.int32)
        for t in range(end):
            o, l = owner_local(t, world)
            brute[t] = caches[o][int(table[l // 256]) * 256 + l % 256].cpu()
        expected = torch.zeros(positions, 132, dtype=torch.int32, device=device)
        addresses = torch.tensor([cache.data_ptr() for cache in caches], dtype=torch.int64, device=device)
        pull_context(addresses, table, final, expected, ContextShards.on(0, world, device))
        assert torch.equal(expected.cpu(), brute), ("pull", world, end)
        slab = torch.zeros_like(expected)
        total = 0
        for first in range(0, columns * 256, window):
            counts = window_counts(final, world, first, window)
            total += int(counts.sum())
            gathered = torch.full((world, window * 528), 255, dtype=torch.uint8, device=device)
            for rank in range(world):
                records = torch.full((world, window * 528), 255, dtype=torch.uint8, device=device)
                pack_records(caches[rank], table, final, records, ContextShards.on(rank, world, device), first)
                gathered[rank, :int(counts[rank])] = records[rank, :int(counts[rank])]
            scatter_records(gathered.view(torch.bfloat16), final, slab, ContextShards.on(world - 1, world, device),
                            first)
        assert total == end * 528, ("window bytes", world, end)
        assert torch.equal(slab.cpu(), brute), ("scatter", world, end)


def lse_reference(partials, lses):
    """Combine per-rank partial attention [W][rows, H, D] with natural-log LSEs [W][rows, H] in FP32.

    Ranks whose LSE is -inf contribute nothing; a head no rank attended combines to zero.
    """
    lse = torch.stack([l.float() for l in lses])
    largest = lse.max(0).values
    base = torch.where(torch.isfinite(largest), largest, torch.zeros_like(largest))
    weights = torch.exp(lse - base)
    total = weights.sum(0)
    combined = sum(w[..., None] * p.float() for w, p in zip(weights, partials))
    return torch.where(total[..., None] > 0, combined / total.clamp_min(1e-30)[..., None],
                       torch.zeros_like(combined))


def _lse_combine(world, device):
    from ignition.components.head_exchange import merge, pack, pack_width

    # Three ranks exchange 72 heads (24 per rank, rank 2's last 8 zero heads); two and four ranks 64.
    heads = 72 if world == 3 else 64
    local, dim, rows = heads // world, 512, 6
    generator = torch.Generator().manual_seed(40 + world)
    partials, lses = [], []
    for rank in range(world):
        partials.append((torch.randn((rows, heads, dim), generator=generator) * 2).to(torch.bfloat16))
        lse = torch.randn((rows, heads), generator=generator) * 3
        # Empty shards: row 1 has no records on ranks above 0, row 2 none on rank 0, row 3 none anywhere.
        if rank > 0:
            lse[1] = -float("inf")
        if rank == 0:
            lse[2] = -float("inf")
        lse[3] = -float("inf")
        # One head where only the last rank attended anything.
        if rank < world - 1:
            lse[4, 5] = -float("inf")
        # Zero heads (q = 0) give a uniform softmax and a finite LSE.
        if world == 3:
            partials[rank][:, 64:] = 0
        lses.append(lse)
    # Inactive ranks leave zero output where their LSE is -inf, as the attention kernel does.
    for rank in range(world):
        partials[rank][~torch.isfinite(lses[rank])] = 0
    reference = lse_reference(partials, lses).to(torch.bfloat16)
    width = pack_width(world, heads, dim)
    sends = []
    for rank in range(world):
        send = torch.zeros((rows, width), dtype=torch.bfloat16, device=device)
        pack(partials[rank].to(device), lses[rank].to(device), send, torch.tensor([rank], device=device), world)
        sends.append(send)
    received = torch.cat(sends, dim=1).contiguous()
    for rank in range(world):
        output = torch.full((rows, local, dim), 7, dtype=torch.bfloat16, device=device)
        merge(received, partials[rank].to(device), lses[rank].to(device), output,
              torch.tensor([rank], device=device), world)
        want = reference[:, rank * local:(rank + 1) * local]
        torch.testing.assert_close(output.cpu().float(), want.float(), rtol=1e-2, atol=2e-2)
        assert not output[3].any(), ("no rank attended", world, rank)
        assert torch.isfinite(output.float()).all()
        if world == 3 and rank == 2:
            assert not output[:, 16:].any(), "zero heads stay zero"


def main(case, world):
    assert os.environ.get("TRITON_INTERPRET") == "1" and not os.environ.get("CUDA_VISIBLE_DEVICES")
    torch.manual_seed(world)
    {"placement": _placement, "merge": _merge, "merge_expand_local": _merge_expand_local,
     "expand_tokens": _expand_tokens, "records": _records, "lse_combine": _lse_combine}[case](world, "cpu")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))

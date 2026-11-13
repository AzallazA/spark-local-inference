# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM sparse latent decode with several rows at head counts that do not tile 64 (TP3's 22, a 24/24/16 split).

`test_glm53_mla_cuda.py::test_glm_mla_checkpoint_native_replay[3-decode-2-True]` found the inactive
second row of a two-row decode (position -1, zero selected length) carrying nonzero attention output on
the 22-head rank, while one-row 22-head decode and two-row 32/64-head decode were clean. This isolates the
kernel: rows must be independent of each other and an inactive row must be zero for every head count.
The 16- and 32-head cases are the TP4/TP2 controls; 24 is the alternative three-rank shard (24/24/16).
"""

import os

import pytest
import torch
from b12x.attention import sparse_mla

from ignition.components.sparse_mla import PreparedLatentCacheWriter, PreparedSparseMLA
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")

TOKENS = 192  # cached latent records: three 64-record pages


def _caps(device, heads, rows):
    return sparse_mla.Caps(device=device, num_q_heads=heads, max_q_rows=rows, max_width=2051,
        softmax_scale=256**-.5, max_batch=rows, max_kv_rows=TOKENS, kv_dtype=torch.uint8,
        head_dim=512, v_head_dim=512, page_size=64, model_type=sparse_mla.ModelType.GLM_NEXT,
        cache_record_bytes=528, mode="decode")


# 22 heads per rank (64 padded to 66 at TP3) is a known b12x sparse-MLA GLM_NEXT limitation: a 64-row M tile that spans
# two requests leaks into an inactive row. The TP3 recipe uses 24 heads per rank instead; strict xfail flags a kernel fix.
@pytest.mark.parametrize("heads", [16, pytest.param(22, marks=pytest.mark.xfail(strict=True,
    reason="b12x sparse-MLA mis-tiles head counts that do not divide 64 across rows")), 24, 32])
@pytest.mark.parametrize("rows", [2, 3])
@torch.no_grad()
def test_decode_rows_are_independent_and_inactive_rows_zero(heads, rows):
    device = torch.device("cuda:0")
    torch.manual_seed(heads * 10 + rows)
    source = torch.randn((TOKENS, 512), dtype=torch.bfloat16, device=device) * .1
    cache = torch.zeros((TOKENS // 64, 64, 528), dtype=torch.uint8, device=device)
    slots = torch.arange(TOKENS, dtype=torch.int64, device=device)
    query = torch.randn((rows, heads, 512), dtype=torch.bfloat16, device=device) * .1
    selected = torch.full((rows, 2051), -1, dtype=torch.int32, device=device)
    lengths = torch.zeros(rows, dtype=torch.int32, device=device)
    output = torch.empty_like(query)
    # The single-row reference shares the cache; it sees one row at a time.
    query1 = torch.empty((1, heads, 512), dtype=torch.bfloat16, device=device)
    selected1 = torch.full((1, 2051), -1, dtype=torch.int32, device=device)
    lengths1 = torch.zeros(1, dtype=torch.int32, device=device)
    output1 = torch.empty_like(query1)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="glm-sparse-mla-heads",
            workload_schema="glm-latent-v1", numerical_policy="default") as session:
        session.register_scratch(arena)
        writer = PreparedLatentCacheWriter(source, cache, slots, session=session, name="cache-write")
        attention = PreparedSparseMLA(query, cache, selected, lengths, lengths, output,
            caps=_caps(device, heads, rows), session=session, name="attention", workspace=arena)
        single = PreparedSparseMLA(query1, cache, selected1, lengths1, lengths1, output1,
            caps=_caps(device, heads, 1), session=session, name="single", workspace=arena)
        session.tune()

        def forward():
            writer()
            attention()
            single()

        compiled = compile_forward(forward)
        compiled()
        torch.cuda.synchronize(device)
        session.freeze()
        # Rows select different prefixes of the cached tokens; the last row is inactive (position -1).
        active = [TOKENS - 1 - 40 * index for index in range(rows - 1)]
        for row, length in enumerate(active):
            selected[row, :length] = torch.arange(length, dtype=torch.int32, device=device)
            lengths[row] = length
        writer()
        torch.cuda.synchronize(device)
        attention()
        torch.cuda.synchronize(device)
        assert torch.isfinite(output).all()
        inactive = output[-1]
        assert not inactive.count_nonzero(), (
            f"{heads} heads, {rows} rows: the inactive row has {int(inactive.count_nonzero())} nonzero outputs, "
            f"max |x| = {inactive.float().abs().max().item():.3e}")
        for row, length in enumerate(active):
            query1.copy_(query[row:row + 1])
            selected1.copy_(selected[row:row + 1])
            lengths1[0] = length
            single()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(output[row], output1[0], rtol=2e-2, atol=2e-3,
                                       msg=lambda m: f"{heads} heads, row {row} of {rows} differs from the one-row kernel: {m}")
        # Activating the last row must not change the other rows.
        before = output[:-1].clone()
        selected[-1, :TOKENS] = torch.arange(TOKENS, dtype=torch.int32, device=device)
        lengths[-1] = TOKENS
        attention()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(output[:-1], before, rtol=2e-2, atol=2e-3)
        query1.copy_(query[-1:])
        selected1.copy_(selected[-1:])
        lengths1[0] = TOKENS
        single()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(output[-1], output1[0], rtol=2e-2, atol=2e-3)


def _lse_caps(device, heads, rows, tokens):
    return sparse_mla.Caps(device=device, num_q_heads=heads, max_q_rows=rows, max_width=2051,
        softmax_scale=256**-.5, max_batch=rows, max_kv_rows=tokens, kv_dtype=torch.uint8,
        head_dim=512, v_head_dim=512, page_size=64, model_type=sparse_mla.ModelType.GLM_NEXT,
        cache_record_bytes=528, mode="decode", return_lse=True, lse_scale="natural")


def _combine(partials, lses):
    """Natural-log LSE combine of [W][rows, H, D] partials; -inf ranks contribute nothing, no rank gives zero."""
    lse = torch.stack([value.float() for value in lses])
    largest = lse.max(0).values
    base = torch.where(torch.isfinite(largest), largest, torch.zeros_like(largest))
    weights = torch.exp(lse - base)
    total = weights.sum(0)
    combined = sum(w[..., None] * p.float() for w, p in zip(weights, partials))
    output = torch.where(total[..., None] > 0, combined / total.clamp_min(1e-30)[..., None], torch.zeros_like(combined))
    return output, torch.where(total > 0, base + torch.log(total.clamp_min(1e-30)), torch.full_like(total, -float("inf")))


# DCP3 attends all 72 exchanged heads (24 per rank, rank 2's last 8 zero) over each rank's records with natural
# LSE (glm53/mla.py). 48 rows is c8 verification at five drafts; 64 and 96 cover more slots. The split planner
# falls back to 33 splits once no split count fits three waves (rows x 5 head blocks > 3 x SMs: >= 113 rows on 188
# SMs); the test prints the planned splits next to each case.
@pytest.mark.parametrize("rows", [1, 6, 8, 48, 64, 96])
@torch.no_grad()
def test_72_head_lse_partials_over_three_shards_combine_to_whole_context_attention(rows):
    from flashinfer.experimental.b12x.attention._shared.mla.kernel import plan_unified_decode_splits

    device = torch.device("cuda:0")
    heads, tokens, world = 72, 768, 3
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    splits = plan_unified_decode_splits(topk=2051, max_chunks=33, num_tokens=rows, h_blocks=5, sm_count=sms)
    print(f"\n72 heads, {rows} rows, {sms} SMs: (chunks, splits, chunks per split) = {splits}")
    torch.manual_seed(rows)
    source = torch.randn((tokens, 512), dtype=torch.bfloat16, device=device) * .1
    cache = torch.zeros((tokens // 64, 64, 528), dtype=torch.uint8, device=device)
    slots = torch.arange(tokens, dtype=torch.int64, device=device)
    query = torch.randn((rows, heads, 512), dtype=torch.bfloat16, device=device) * .3
    query[:, 64:] = 0  # rank 2's padded heads
    buffers = {}

    def attention_buffers(name, count):
        buffers[name] = (torch.empty((rows, count, 512), dtype=torch.bfloat16, device=device),
                         torch.full((rows, 2051), -1, dtype=torch.int32, device=device),
                         torch.zeros(rows, dtype=torch.int32, device=device),
                         torch.empty((rows, count, 512), dtype=torch.bfloat16, device=device),
                         torch.empty((rows, count), dtype=torch.float32, device=device))
        return buffers[name]

    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="glm-sparse-mla-dcp3",
            workload_schema="glm-latent-v1", numerical_policy="default") as session:
        session.register_scratch(arena)
        writer = PreparedLatentCacheWriter(source, cache, slots, session=session, name="cache-write")
        ops = {}
        for name, count in (("whole", heads), ("shard", heads), ("narrow", 24)):
            q, selected, lengths, output, lse = attention_buffers(name, count)
            ops[name] = PreparedSparseMLA(q, cache, selected, lengths, lengths, output,
                caps=_lse_caps(device, count, rows, tokens), session=session, name=name, workspace=arena, lse=lse)
        session.tune()

        def forward():
            writer()
            for op in ops.values():
                op()

        compiled = compile_forward(forward)
        compiled()
        torch.cuda.synchronize(device)
        session.freeze()
        writer()
        # Row lengths: full, ragged, below one group per rank (ranks 1 and 2 own nothing), inactive last row.
        # Every fifth row holds only positions 0..3, one group, so ranks 1 and 2 attend nothing for it.
        picks = [torch.arange(3 + row % 2, device=device).int() if row % 5 == 0 else
                 torch.randperm(tokens, device=device)[:1 + (700 - 37 * row) % 700].sort().values.int()
                 for row in range(rows)]
        if rows > 1:
            picks[-1] = picks[-1][:0]
        q, selected, length, output, lse = buffers["whole"]
        q.copy_(query)
        selected.fill_(-1)
        for row, pick in enumerate(picks):
            selected[row, :len(pick)] = pick
            length[row] = len(pick)
        ops["whole"]()
        torch.cuda.synchronize(device)
        whole, whole_lse = output.clone(), lse.clone()
        assert torch.isfinite(whole).all()
        if rows > 1:
            assert not whole[-1].count_nonzero() and torch.isneginf(whole_lse[-1]).all(), "inactive row"
        # Each "rank" attends the records it owns: positions in groups of four, group g on rank g % 3.
        partials, lses = [], []
        q, selected, length, output, lse = buffers["shard"]
        q.copy_(query)
        for rank in range(world):
            selected.fill_(-1)
            for row, pick in enumerate(picks):
                own = pick[(pick // 4) % world == rank]
                selected[row, :len(own)] = own
                length[row] = len(own)
            ops["shard"]()
            torch.cuda.synchronize(device)
            assert torch.isfinite(output).all()
            empty = length == 0
            assert not output[empty].count_nonzero() and torch.isneginf(lse[empty]).all(), ("empty shard", rank)
            partials.append(output.clone())
            lses.append(lse.clone())
        combined, combined_lse = _combine(partials, lses)
        torch.testing.assert_close(combined, whole.float(), rtol=2e-2, atol=4e-3)
        live = torch.isfinite(whole_lse)
        assert torch.equal(live, torch.isfinite(combined_lse))
        torch.testing.assert_close(combined_lse[live], whole_lse[live], rtol=1e-3, atol=1e-3)
        # Zero query heads attend uniformly: finite LSE, and their output is later multiplied by zero weights.
        assert torch.isfinite(whole_lse[:-1 if rows > 1 else None, 64:]).all()
        # The 72-head launch agrees with three 24-head launches (24 heads per rank is the qualified TP3 shape).
        q24, selected24, length24, output24, lse24 = buffers["narrow"]
        selected24.copy_(buffers["whole"][1])
        length24.copy_(buffers["whole"][2])
        for block in range(world):
            q24.copy_(query[:, 24 * block:24 * (block + 1)])
            ops["narrow"]()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(output24, whole[:, 24 * block:24 * (block + 1)], rtol=2e-2, atol=2e-3)
            torch.testing.assert_close(lse24, whole_lse[:, 24 * block:24 * (block + 1)], rtol=1e-3, atol=1e-3)

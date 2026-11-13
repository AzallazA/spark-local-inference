# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM DCP3 decode exchanges on three GPUs through the production wrappers, against torch references, with timing.

Per MLA layer and decode pass, DCP3 runs three b12x PCIe collectives (glm53/mla.py, indexer.py): the query-head
all-gather (24 x 512 BF16 per rank -> 72 heads), the LSE reduce-scatter of 72-head partial attention back to each
rank's 24 heads, and the candidate all-gather (2048 BF16 lanes of 512 FP32 (score, pool) pairs per rank -> 6144).
b12x serves world 3 through its generic modulo path (tp3-patches/flashinfer-dcp-a2a-world3.patch); vLLM's DCP3 used
it at up to 8 rows only, so rows 1..48 (c8 verification at five drafts) are checked here, including rows and heads
whose LSE is -inf on some or every rank. Gathers must be bit-exact; the combine is checked against b12x's own
reference and an independent FP32 one. Each exchange is then captured in a CUDA graph of 11 calls (GLM's MLA
layers) and timed; FLARE_DCP3_TIMING=<path> writes the per-call microseconds as JSON (rank 0).

    CUDA_VISIBLE_DEVICES=0,1,2 FLARE_RUN_CUDA_TESTS=1 pytest -s tests/recipes/test_glm53_dcp3_collectives_cuda.py
"""

import json
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires WORLD assigned GPUs and FLARE_RUN_CUDA_TESTS=1")

# FLARE_DCP_TEST_WORLD=2 runs the same exchanges at DCP2 (TP2: 64 heads, 32 per rank, no padded heads) to time the
# world-2 gather (upstream TP2 push, or the overlay's pull with B12X_PCIE_DCP_GATHER_PULL=1). Default 3 (DCP3).
WORLD = int(os.environ.get("FLARE_DCP_TEST_WORLD", "3"))
if WORLD not in (2, 3):
    raise ValueError("FLARE_DCP_TEST_WORLD must be 2 or 3")
HEADS, DIM, LAYERS = (72 if WORLD == 3 else 64), 512, 11
HEAD_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 32, 47, 48)
CANDIDATE_ROWS = (1, 6, 8, 9, 48, 100, 256, 511, 512)


def _inputs(rows, source, seed):
    """Rank `source`'s partial attention and LSE, reproducible on every rank."""
    generator = torch.Generator().manual_seed(seed * 1000 + rows * 10 + source)
    partial = (torch.randn((rows, HEADS, DIM), generator=generator) * 2).to(torch.bfloat16)
    lse = torch.randn((rows, HEADS), generator=generator) * 4
    # -inf patterns: row 0 attends only on rank 0 (short context, empty shards), row 1 on no rank (inactive),
    # row 2 on every rank but 0, one head per row only on the last rank, and the padded heads 64..71.
    if source > 0:
        lse[0] = -float("inf")
    if rows > 1:
        lse[1] = -float("inf")
    if rows > 2 and source == 0:
        lse[2] = -float("inf")
    if source < WORLD - 1:
        lse[:, 7] = -float("inf")
    # Ranks with nothing to attend write zero output; large values elsewhere check that -inf weights stay zero.
    partial[torch.isinf(lse)] = 0
    partial[:, 64:] = 0
    return partial, lse


def _reference(partials, lses, rank):
    """Independent FP32 combine of this rank's head shard."""
    local = HEADS // WORLD
    shard = slice(rank * local, (rank + 1) * local)
    lse = torch.stack([value[:, shard] for value in lses])
    largest = lse.max(0).values
    base = torch.where(torch.isfinite(largest), largest, torch.zeros_like(largest))
    weights = torch.exp(lse - base)
    total = weights.sum(0)
    combined = sum(w[..., None] * p[:, shard].float() for w, p in zip(weights, partials))
    return torch.where(total[..., None] > 0, combined / total.clamp_min(1e-30)[..., None], torch.zeros_like(combined))


def _time(call, device, iterations=30):
    """Microseconds per call of `call` captured LAYERS times in one CUDA graph, replayed in lockstep."""
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(LAYERS):
            call()
    torch.cuda.synchronize(device)
    for _ in range(3):
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize(device)
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    torch.cuda.synchronize(device)
    graph.reset()
    return start.elapsed_time(end) * 1e3 / (iterations * LAYERS)


def _rank(rank, rendezvous, timing_path):
    from b12x.comm.pcie.pcie_dcp_a2a import lse_reduce_scatter_reference

    from ignition.components.collectives import PreparedAllGather, PreparedLseReduceScatter
    from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
                                                  create_head_exchange)
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    local = HEADS // WORLD
    rows_max = max(HEAD_ROWS)
    # As bootstrap.py sizes them for eight slots at five drafts (48 rows) and 512-row candidate slices.
    heads = create_head_exchange(group, device, max_rows=rows_max, heads=HEADS, head_dim=DIM)
    candidates = create_all_gather_runtime(group, device, max_rows=max(CANDIDATE_ROWS), columns=2048 * WORLD)
    query = torch.zeros((rows_max, local * DIM), dtype=torch.bfloat16, device=device)
    gathered = torch.zeros((rows_max, HEADS * DIM), dtype=torch.bfloat16, device=device)
    partial = torch.zeros((rows_max, HEADS, DIM), dtype=torch.bfloat16, device=device)
    lse = torch.zeros((rows_max, HEADS), dtype=torch.float32, device=device)
    attended = torch.zeros((rows_max, local, DIM), dtype=torch.bfloat16, device=device)
    packed = torch.zeros((max(CANDIDATE_ROWS), 512, 2), dtype=torch.float32, device=device)
    merged = torch.zeros((max(CANDIDATE_ROWS), WORLD * 512, 2), dtype=torch.float32, device=device)
    timings = {"head_gather": {}, "lse_reduce_scatter": {}, "candidate_gather": {}}
    try:
        with RecordedPreparationSession(device=device, checkpoint="glm-dcp3-collectives",
                workload_schema="glm-dcp3-collectives-v1", numerical_policy="default",
                coordinator=coordinator) as session:
            gathers = {rows: PreparedAllGather(query[:rows], gathered[:rows], runtime=heads, session=session,
                                               coordinator=coordinator, name=f"heads{rows}") for rows in HEAD_ROWS}
            combines = {rows: PreparedLseReduceScatter(partial[:rows], lse[:rows], attended[:rows], runtime=heads,
                                                       session=session, coordinator=coordinator,
                                                       name=f"combine{rows}") for rows in HEAD_ROWS}
            # The indexer gathers FP32 pairs as BF16 lanes (indexer.py).
            selections = {rows: PreparedAllGather(packed[:rows].view(torch.bfloat16).view(rows, 2048),
                                                  merged[:rows].view(torch.bfloat16).view(rows, 2048 * WORLD),
                                                  runtime=candidates, session=session, coordinator=coordinator,
                                                  name=f"candidates{rows}") for rows in CANDIDATE_ROWS}
            session.tune()
            failures = []
            for rows in HEAD_ROWS:
                # Query heads: rank r's 24 heads land at columns [r * 24 * 512, (r + 1) * 24 * 512) on every rank.
                sources = [torch.randint(-2**15, 2**15 - 1, (rows, local * DIM), dtype=torch.int16,
                                         generator=torch.Generator().manual_seed(rows * 10 + r)) for r in range(WORLD)]
                query[:rows].view(torch.int16).copy_(sources[rank])
                gathered.zero_()
                dist.barrier()
                gathers[rows](query[:rows])
                torch.cuda.synchronize(device)
                if not torch.equal(gathered[:rows].view(torch.int16).cpu(), torch.cat(sources, dim=1)):
                    failures.append(f"head gather rows {rows}")
                # LSE reduce-scatter.
                inputs = [_inputs(rows, source, 1) for source in range(WORLD)]
                partial[:rows].copy_(inputs[rank][0])
                lse[:rows].copy_(inputs[rank][1])
                attended.fill_(7)
                dist.barrier()
                combines[rows](partial[:rows], lse[:rows])
                torch.cuda.synchronize(device)
                got = attended[:rows].float().cpu()
                b12x = lse_reduce_scatter_reference(torch.stack([p for p, _ in inputs]),
                                                    torch.stack([l for _, l in inputs]), rank).float()
                mine = _reference([p for p, _ in inputs], [l for _, l in inputs], rank)
                if not torch.isfinite(got).all():
                    failures.append(f"lse rows {rows}: non-finite output")
                for name, want in (("b12x reference", b12x), ("fp32 reference", mine)):
                    try:
                        torch.testing.assert_close(got, want, rtol=1e-2, atol=2e-2)
                    except AssertionError as error:
                        failures.append(f"lse rows {rows} vs {name}: {str(error).splitlines()[0:4]}")
                if rows > 1 and got[1].count_nonzero():
                    failures.append(f"lse rows {rows}: a row no rank attended is not zero")
                if HEADS > 64 and rank == WORLD - 1 and got[:, 64 - (WORLD - 1) * local:].count_nonzero():
                    failures.append(f"lse rows {rows}: padded heads are not zero")
            for rows in CANDIDATE_ROWS:
                sources = [torch.randint(-2**15, 2**15 - 1, (rows, 2048), dtype=torch.int16,
                                         generator=torch.Generator().manual_seed(7000 + rows * 10 + r))
                           for r in range(WORLD)]
                packed[:rows].view(torch.int16).view(rows, 2048).copy_(sources[rank])
                merged.zero_()
                dist.barrier()
                selections[rows](packed[:rows].view(torch.bfloat16).view(rows, 2048))
                torch.cuda.synchronize(device)
                if not torch.equal(merged[:rows].view(torch.int16).view(rows, -1).cpu(), torch.cat(sources, dim=1)):
                    failures.append(f"candidate gather rows {rows}")
            outcome = [None] * WORLD
            dist.all_gather_object(outcome, failures, group=group)
            assert not any(outcome), outcome
            # Timing, after correctness: the same prepared operations inside CUDA graphs.
            with session.capture():
                for rows in HEAD_ROWS:
                    timings["head_gather"][rows] = _time(lambda: gathers[rows](query[:rows]), device)
                    timings["lse_reduce_scatter"][rows] = _time(lambda: combines[rows](partial[:rows], lse[:rows]),
                                                                device)
                for rows in CANDIDATE_ROWS:
                    view = packed[:rows].view(torch.bfloat16).view(rows, 2048)
                    timings["candidate_gather"][rows] = _time(lambda: selections[rows](view), device)
            session.freeze()
        if rank == 0:
            print(f"\nDCP{WORLD} exchanges, us per call (CUDA graph of {LAYERS} calls):")
            for name, table in timings.items():
                print(f"  {name:20s} " + "  ".join(f"{rows}:{value:.1f}" for rows, value in table.items()))
            for label, rows in (("c1 verify", 6), ("c8 verify", 48)):
                total = sum(timings[name][rows] for name in ("head_gather", "lse_reduce_scatter")) + \
                    timings["candidate_gather"][rows if rows in CANDIDATE_ROWS else 8]
                print(f"  {label} ({rows} rows): {total:.1f} us per MLA layer (all three exchanges, none hidden)")
            if timing_path:
                with open(timing_path, "w") as handle:
                    json.dump(timings, handle, indent=1)
    finally:
        heads.close()
        candidates.close()
        dist.destroy_process_group()


def test_dcp3_head_gather_lse_reduce_scatter_and_candidate_gather(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() >= WORLD
    mp.spawn(_rank, args=((tmp_path / "rendezvous").as_uri(), os.environ.get("FLARE_DCP3_TIMING", "")),
             nprocs=WORLD, join=True)

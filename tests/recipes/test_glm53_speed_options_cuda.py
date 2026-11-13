# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GPU checks and microbenchmarks for the fork-only GLM TP3 decode-speed options (one GPU, real FP4-Local weights).

    FLARE_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 FLARE_GLM53_SNAPSHOT=/path/to/ckpt/glm53-fp4-local \\
      FLARE_SPEED_TIMING=out.json pytest -q -s tests/recipes/test_glm53_speed_options_cuda.py

- KDA side chain: the side-chain layer matches the packed layer (output, convolution and recurrent state; bitwise
  unless a GEMV tiling changes the reduction order, recorded) at decode rows 1/4/5/8 and prefill 65, and its captured graph is timed against the packed one (rank-0 TP3
  shard, 24 heads). The gain to expect per layer is ~6-8 us at rows 4-5 (profile: 8.2-8.8 us exposed after q/k/v).
- Draft head: NVFP4-from-MXFP8 head vs the MXFP8 head (rank-0 51,648-row shard) and MXFP8 vs BF16 eh_proj, timed
  at rows 1/4/8 as captured graphs, with a logit-agreement sanity check.
"""

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")

SNAPSHOT = Path(os.environ.get("FLARE_GLM53_SNAPSHOT", "/path/to/ckpt/glm53-fp4-local"))
# FLARE_GLM53_TEST_WORLD=2 checks the KDA side chain on the rank-0 TP2 shard (32 heads; b_proj N=32 beside f_a N=128,
# packed N=160) instead of TP3 (24 heads). Default 3.
WORLD = int(os.environ.get("FLARE_GLM53_TEST_WORLD", "3"))
TIMING = {}


def _record(*keys, value):
    node = TIMING
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    path = os.environ.get("FLARE_SPEED_TIMING")
    if path:
        Path(path).write_text(json.dumps(TIMING, indent=1, sort_keys=True))


def _time_graph(graph, iterations=400):
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return sorted(samples)[len(samples) // 2]


def _snapshot():
    if not SNAPSHOT.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.glm53.model import create_model_tree
    return inspect_indexed_checkpoint(SNAPSHOT), create_model_tree(SNAPSHOT)


def _kda_run(module, weights, split, mode, rows, inputs, device):
    """Prepare one KDA layer variant, run it eagerly and as a captured graph; return results and graph time."""
    from ignition.components.recurrent import DeltaMetadata, DeltaState
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.kda import KDAScratch, PreparedGlmKDA

    from ignition.recipes.glm53.shards import Shards

    heads = Shards.of(WORLD).heads
    decode = mode == "decode"
    state = DeltaState.allocate(rows + 2 if decode else 4, device, heads=heads, channels=3 * heads * 128,
                                speculative_tokens=rows - 1 if decode else 0)
    metadata = DeltaMetadata.allocate(1, device, verification_columns=rows if decode else 1)
    scratch = KDAScratch.allocate(rows, heads, device, split=split)
    source = torch.zeros((rows, 4096), dtype=torch.bfloat16, device=device)
    arena = ScratchArena(device, overlap=True)
    with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-FP4-Local",
                                    workload_schema="glm-kda-speed-v1", numerical_policy="default") as session:
        session.register_scratch(arena)
        operator = PreparedGlmKDA(module, weights, source, scratch, state, metadata, mode=mode, session=session,
                                  name=f"kda.{mode}.{rows}.{int(split)}", workspace=arena,
                                  checkpoint_export=not decode)
        assert operator.side_chain is split and (len(operator.side_chains) == 3) is split
        session.tune()
        metadata.query_start_loc.copy_(torch.tensor([0, rows], dtype=torch.int32))
        metadata.convolution_slots.copy_(torch.tensor([1], dtype=torch.int64))
        metadata.fresh.fill_(not decode)
        if decode:
            metadata.state_indices[0, :rows] = torch.arange(1, rows + 1)
            metadata.accepted_tokens.fill_(1)
        else:
            metadata.initial_state_indices[0] = 0
            metadata.final_state_indices[0] = 1
            metadata.checkpoint_state_indices[0] = 2 if rows >= 16 else 0
            metadata.checkpoint_offsets[0] = 16 if rows >= 16 else 0
        metadata.num_seqs.fill_(1)
        metadata.num_tokens.fill_(rows)
        source.copy_(inputs)
        initial = (state.convolution.clone(), state.recurrent.clone())
        operator(source)
        torch.cuda.synchronize()
        result = (scratch.output.clone(), state.convolution.clone(), state.recurrent.clone())

        def forward():
            state.convolution.copy_(initial[0])
            state.recurrent.copy_(initial[1])
            operator(source)

        compiled = compile_forward(forward)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        graph.instantiate()
        graph.replay()
        torch.cuda.synchronize()
        replayed = (scratch.output.clone(), state.convolution.clone(), state.recurrent.clone())
        for actual, expected in zip(replayed, result, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The layer graph re-copies two small state tensors per replay in both variants; time it.
        microseconds = _time_graph(graph)
        graph.reset()
    return result, microseconds


@pytest.mark.parametrize("mode,rows", [("decode", 1), ("decode", 4), ("decode", 5), ("decode", 8), ("prefill", 65)])
@torch.no_grad()
def test_kda_side_chain_matches_and_is_timed(mode, rows):
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    manifest, model = _snapshot()
    module = model.model.language_model.layers[4].self_attn
    device = torch.device("cuda", 0)
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(0, WORLD),
                         prefixes=(f"{TEXT_PREFIX}.layers.4.self_attn.",)) as loader:
        weights = loader.load_attention(4)
    generator = torch.Generator().manual_seed(4053)
    inputs = (torch.randn((rows, 4096), generator=generator) * .5).to(torch.bfloat16).to(device)
    packed, packed_us = _kda_run(module, weights, False, mode, rows, inputs, device)
    split, split_us = _kda_run(module, weights, True, mode, rows, inputs, device)
    # Same weights and dot products; only the GEMV tiling of b / f_a (one N=152 launch before, N=24 and N=128
    # now) and the autotuned q/k/v candidate can change the reduction order, so expect identity or BF16 rounding.
    differences = []
    for actual, expected in zip(split, packed, strict=True):
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
        differences.append((actual.float() - expected.float()).abs().max().item())
    assert packed[0].count_nonzero()
    _record("kda" if WORLD == 3 else f"kda-tp{WORLD}", f"{mode}-{rows}", value={"packed_us": round(packed_us, 2), "side_chain_us": round(split_us, 2),
                                             "saved_us": round(packed_us - split_us, 2),
                                             "bitwise": not any(differences), "max_abs_diff": differences})
    print(f"KDA {mode} rows {rows}: packed {packed_us:.2f} us, side chain {split_us:.2f} us")


def _projection_graph(source, weight, output, name):
    from ignition.components.linear import QuantizedProjection, UnquantizedProjection
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = source.device
    arena = ScratchArena(device, overlap=False)
    session = RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-FP4-Local",
                                         workload_schema="glm-draft-speed-v1", numerical_policy="default")
    with session:
        session.register_scratch(arena)
        operation = (UnquantizedProjection(source, weight, output, session=session, name=name)
                     if isinstance(weight, torch.Tensor) else
                     QuantizedProjection(source, weight, output, session=session, name=name,
                                         activation_mode="auto", workspace=arena))
        session.tune(show_progress=False)
        compiled = compile_forward(lambda: operation(source))
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        graph.instantiate()
        graph.replay()
        torch.cuda.synchronize()
        result = output.clone()
        microseconds = _time_graph(graph, iterations=100)
        graph.reset()
    return result, microseconds


@pytest.mark.parametrize("rows", [1, 4, 8])
@torch.no_grad()
def test_draft_head_and_eh_proj_precision_options_are_timed(rows):
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.vocabulary import decode_mxfp8_rows, mxfp8_draft_head, nvfp4_draft_head_from_mxfp8
    from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

    manifest, _ = _snapshot()
    device = torch.device("cuda", 0)
    shard = TensorParallelShard(0, 3)
    with GlmWeightLoader(manifest, device, shard=shard,
                         prefixes=("lm_head.", f"{TEXT_PREFIX}.layers.45.eh_proj.")) as loader:
        head = loader.load_linear("lm_head", rows=shard.padded_partition(154880, alignment=64), pad=True)
        eh_proj = loader.load_tensor(f"{TEXT_PREFIX}.layers.45.eh_proj.weight")
    assert not isinstance(head, torch.Tensor) and head.out_features == 51648
    draft = nvfp4_draft_head_from_mxfp8(head)
    generator = torch.Generator().manual_seed(45)
    hidden = (torch.randn((rows, 4096), generator=generator)).to(torch.bfloat16).to(device)
    logits = torch.empty((rows, 51648), dtype=torch.bfloat16, device=device)
    target, target_us = _projection_graph(hidden, head, logits, "head.mxfp8")
    proposal, proposal_us = _projection_graph(hidden, draft, torch.empty_like(logits), "head.nvfp4")
    reference = hidden.float() @ decode_mxfp8_rows(head, 0, 51648).float().T
    error = ((proposal.float() - reference).norm() / reference.norm()).item()
    assert ((target.float() - reference).norm() / reference.norm()).item() < .05
    assert error < .15
    agreement = (proposal.float().argmax(-1) == target.float().argmax(-1)).float().mean().item()
    joined = (torch.randn((rows, 8192), generator=generator) * .5).to(torch.bfloat16).to(device)
    residual = torch.empty((rows, 4096), dtype=torch.bfloat16, device=device)
    bf16, bf16_us = _projection_graph(joined, eh_proj, residual, "eh_proj.bf16")
    mx, mx_us = _projection_graph(joined, mxfp8_draft_head(eh_proj.contiguous()), torch.empty_like(residual),
                                  "eh_proj.mxfp8")
    eh_error = ((mx.float() - bf16.float()).norm() / bf16.float().norm()).item()
    assert eh_error < .08
    _record("draft", f"rows-{rows}", value={
        "head_mxfp8_us": round(target_us, 2), "head_nvfp4_us": round(proposal_us, 2),
        "head_nvfp4_rel_error": round(error, 4), "head_top1_agreement_random_rows": round(agreement, 3),
        "eh_proj_bf16_us": round(bf16_us, 2), "eh_proj_mxfp8_us": round(mx_us, 2),
        "eh_proj_mxfp8_rel_error": round(eh_error, 4)})
    print(f"draft rows {rows}: head {target_us:.1f} -> {proposal_us:.1f} us (rel err {error:.3f}), "
          f"eh_proj {bf16_us:.1f} -> {mx_us:.1f} us (rel err {eh_error:.3f})")

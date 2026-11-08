# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 expert parking: resident, hot + cold banks and the merged bank compute the same routed-expert output.

Run (one free GPU, truncated real-weight checkpoint):

    cd ~/flare-up-v3 && CUDA_VISIBLE_DEVICES=0 FLARE_RUN_CUDA_TESTS=1 \\
        FLARE_DS41_SNAPSHOT=/m2-2/flare-ckpt/ds41-mini8 .venv/bin/pytest -x -s \\
        tests/recipes/test_deepseek41_expert_parking_cuda.py

Optional: FLARE_DS41_PARKING_TEST_LAYER (default 3), FLARE_DS41_PARKING_TEST_HOT (default 64 hot experts of 384),
FLARE_DS41_PARKING_TEST_ROWS (default "1,24,48").

One MoE layer's 384 routed experts are loaded three times through weights.py `load_layer`, in one PackageStorage,
each with its own weight loader and environment (parsed into a TP3Config per configuration):
  resident  no FLARE_DS41_HOST_EXPERT_LAYERS: one VRAM bank, router ids used as-is;
  split     FLARE_DS41_HOST_EXPERT_LAYERS + FLARE_DS41_HOT_EXPERTS: hot bank (VRAM) + cold bank (host_allocate regions),
            hot_map / cold_map with -1 for ids outside a bank (b12x skips negative routes);
  merged    + FLARE_DS41_MERGED_BANK=1: one bank in hot-first order, VRAM prefix + host tail (storage.mixed_allocate),
            hot_map[order] = arange.
The synthetic route counts make experts 1, 7, 13, ... (e % 6 == 1) the hot set, so hot ids are not a prefix of
0..383 and the merged bank's hot-first order is a real permutation. All three paths then see identical normalized
inputs and identical router ids / weights (the router is bypassed: ids are written straight into the router scratch
the way PreparedRouter would) and run exactly decoder.py PreparedLayer.__call__'s expert section: hot_map/cold_map
index_select remaps, experts(), and for the split layout cold_experts() + routed.add_(cold_routed). Every row routes
to one hot and one cold "heavy" expert (> 16 routes each once rows > 16), one more random hot and one more random cold
expert, plus two random experts. Outputs must agree within BF16 rounding (cosine > 0.9999; max_abs within two BF16
ulps of the output's peak magnitude or 4x the resident path's own run-to-run atomic-accumulation noise, whichever is
larger; stats printed), and recipe.py `_parked_bytes(3)` must equal the bytes PackageStorage actually placed in host
memory for that configuration (HostRegion extents + MixedRegion host tails; exact for the TP3 geometry, allowance of
one page / one granule per region otherwise). Geometry is the TP3 rank-0 shard (768 of 2304 intermediate columns):
the routed-expert path has no collectives, so a single process stands in for rank 0 and `_parked_bytes(3)` is the
production number.
"""

import json
import mmap
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")

WORLD = 3  # TP3 rank-0 shard geometry; the routed-expert path needs no collectives
EXPERTS, TOP_K, HIDDEN = 384, 6, 5120
COSINE = 0.9999
BF16_ULP = 2.0 ** -7  # relative spacing of BF16 values (7 explicit mantissa bits)
DEFAULT_ROWS = (1, 24, 48)
SEEDS = (11, 12)


def _rows():
    spec = os.environ.get("FLARE_DS41_PARKING_TEST_ROWS", "")
    return tuple(int(p) for p in spec.split(",") if p.strip()) or DEFAULT_ROWS


def _hot_ids(hot):
    """`hot` expert ids spread over 0..383 (not a prefix), so hot-first bank order is a real permutation."""
    step = max(1, EXPERTS // hot)
    ids = list(range(1, EXPERTS, step))[:hot]
    chosen = set(ids)
    ids += [e for e in range(EXPERTS) if e not in chosen][:hot - len(ids)]
    return sorted(ids)


def _routes(m, hot_ids, cold_ids, heavy, *, seed):
    """Router ids/weights every path sees: both heavy experts in every row, one more hot, one more cold, two random."""
    g = torch.Generator().manual_seed(seed)
    hot_pool = [e for e in hot_ids if e not in heavy]
    cold_pool = [e for e in cold_ids if e not in heavy]
    ids = torch.empty((m, TOP_K), dtype=torch.int32)
    for row in range(m):
        picks = list(heavy)
        picks.append(hot_pool[int(torch.randint(len(hot_pool), (1,), generator=g))])
        picks.append(cold_pool[int(torch.randint(len(cold_pool), (1,), generator=g))])
        for e in torch.randperm(EXPERTS, generator=g).tolist():
            if len(picks) == TOP_K:
                break
            if e not in picks:
                picks.append(e)
        ids[row] = torch.tensor(picks, dtype=torch.int32)
    # Renormalized top-k weights scaled by routed_scaling_factor, as PreparedRouter produces them (FP32).
    weights = torch.softmax(torch.randn((m, TOP_K), generator=g), dim=1) * 1.5
    return ids, weights


def _stats(got, want):
    a, b = got.detach().double().reshape(-1), want.detach().double().reshape(-1)
    diff = a - b
    return dict(cosine=float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30)),
                max_abs=float(diff.abs().max()), rel_norm=float(diff.norm() / b.norm().clamp_min(1e-30)),
                max_rel=float(diff.abs().max() / b.abs().max().clamp_min(1e-30)),
                finite=bool(torch.isfinite(a).all()))


def _base_env(snapshot):
    text = json.loads((Path(snapshot) / "config.json").read_text())["text_config"]
    return {"FLARE_DS41_MINI": "1" if text["num_hidden_layers"] != 40 else None,
            "FLARE_DS41_HOST_EXPERT_LAYERS": None, "FLARE_DS41_HOT_EXPERTS": None, "FLARE_DS41_MERGED_BANK": None,
            "FLARE_DS41_GLOBAL_COLD": None, "FLARE_DS41_EXPERT_COUNTS_DIR": None,
            "FLARE_DS41_BRRR_LAYERS": None, "FLARE_DS41_BRRR_NATIVE_SCALES": None}


def _apply(monkeypatch, env):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def _configs(layer, hot, counts_dir):
    parked = {"FLARE_DS41_HOST_EXPERT_LAYERS": str(layer), "FLARE_DS41_HOT_EXPERTS": str(hot),
              "FLARE_DS41_EXPERT_COUNTS_DIR": str(counts_dir)}
    return {"resident": {}, "split": parked, "merged": {**parked, "FLARE_DS41_MERGED_BANK": "1"}}


def _bank_tensors(experts):
    impl = getattr(experts, "_impl", experts)
    return tuple(getattr(impl, name) for name in ("w1_fp4", "w2_fp4"))


def _within(tensor, regions):
    pointer = tensor.untyped_storage().data_ptr()
    return any(region.address <= pointer < region.address + region.extent for region in regions)


class _Buffers:
    """Router scratch, FFN input and routed output shared by every path (identical inputs by construction)."""

    def __init__(self, capacity, device):
        from ignition.components.moe import RouterScratch

        self.source = torch.zeros((capacity, HIDDEN), dtype=torch.bfloat16, device=device)
        self.logits = torch.zeros((capacity, EXPERTS), dtype=torch.float32, device=device)
        self.router = RouterScratch.allocate(capacity, device, experts=EXPERTS, top_k=TOP_K)
        self.routed = torch.zeros((capacity, HIDDEN), dtype=torch.bfloat16, device=device)


class _Path:
    """decoder.py PreparedLayer's expert section (remaps, experts(), cold_experts() + add) for one expert layout."""

    def __init__(self, name, weights, config, m, buffers, *, session, workspace):
        from ignition.components.moe import PreparedRouter, PreparedRoutedExperts

        text, e = config.text, weights.experts
        source, logits = buffers.source[:m], buffers.logits[:m]
        router, routed = buffers.router.rows(m), buffers.routed[:m]
        self.m, self.ids, self.weights, self.routed = m, router.ids, router.weights, routed
        self.hot_map, self.cold_map, self.hot_rows = e.hot_map, e.cold_map, e.hot_rows
        # Declared (never called): the routed experts tune against the router plan as in decoder.py.
        topk = PreparedRouter(logits, router, session=session, name=name + ".topk",
                              score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
                              routed_scaling_factor=text["routed_scaling_factor"], correction_bias=e.correction)
        self.cold_experts = self.hot_ids = self.cold_ids = self.cold_routed = None
        if e.cold is not None:
            self.hot_ids, self.cold_ids = torch.zeros_like(router.ids), torch.zeros_like(router.ids)
            self.cold_routed = torch.zeros_like(routed)
            route_ids = self.hot_ids
        elif e.hot_map is not None:
            self.hot_ids = torch.zeros_like(router.ids)
            route_ids = self.hot_ids
        else:
            route_ids = router.ids
        self.experts = PreparedRoutedExperts(source, e.experts, route_ids, router.weights, routed, session=session,
                                             name=name + ".experts", workspace=workspace, router_plan=topk.plan)
        if e.cold is not None:
            self.cold_experts = PreparedRoutedExperts(source, e.cold, self.cold_ids, router.weights, self.cold_routed,
                                                      session=session, name=name + ".cold_experts",
                                                      workspace=workspace, router_plan=topk.plan)

    def __call__(self):
        flat = self.ids.reshape(-1)
        if self.cold_experts is not None:
            torch.index_select(self.hot_map, 0, flat, out=self.hot_ids.view(-1))
            torch.index_select(self.cold_map, 0, flat, out=self.cold_ids.view(-1))
        elif self.hot_map is not None:
            torch.index_select(self.hot_map, 0, flat, out=self.hot_ids.view(-1))
        self.experts()
        if self.cold_experts is not None:
            self.cold_experts()
            self.routed.add_(self.cold_routed)
        return self.routed

    def cold_routes(self):
        if self.cold_ids is not None:
            return int((self.cold_ids >= 0).sum())
        if self.hot_rows is not None:
            return int((self.hot_ids >= self.hot_rows).sum())
        return 0


def _check_maps(name, weights, hot_ids, cold_ids):
    e = weights.experts
    hot_map = None if e.hot_map is None else e.hot_map.cpu().long()
    if name == "resident":
        assert e.cold is None and e.hot_map is None and e.cold_map is None and e.hot_rows is None, "resident layer remapped"
        return
    hot_index, cold_index = torch.tensor(hot_ids), torch.tensor(cold_ids)
    assert hot_map is not None
    assert torch.equal(hot_map[hot_index], torch.arange(len(hot_ids))), "hot ids must map to hot slots in id order"
    if name == "split":
        assert e.cold is not None and e.cold_map is not None and e.hot_rows is None
        cold_map = e.cold_map.cpu().long()
        assert bool((hot_map[cold_index] == -1).all()), "cold ids must be inactive in the hot bank"
        assert bool((cold_map[hot_index] == -1).all()), "hot ids must be inactive in the cold bank"
        assert torch.equal(cold_map[cold_index], torch.arange(len(cold_ids)))
    else:
        assert e.cold is None and e.cold_map is None and e.hot_rows == len(hot_ids)
        # order = hot ids then the remaining ids ascending: cold ids take slots hot_rows.. in id order.
        assert torch.equal(hot_map[cold_index], torch.arange(len(hot_ids), EXPERTS)), "merged bank is not hot-first"


@torch.no_grad()
def test_parked_expert_layouts_match_resident_and_parked_bytes(tmp_path, monkeypatch):
    snapshot = os.environ.get("FLARE_DS41_SNAPSHOT")
    if snapshot is None or not (Path(snapshot) / "model.safetensors.index.json").is_file():
        pytest.skip("FLARE_DS41_SNAPSHOT must identify a DeepSeek V4.1 Flash checkpoint (ds41-mini8 is fast)")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    snapshot = str(Path(snapshot).resolve())
    layer = int(os.environ.get("FLARE_DS41_PARKING_TEST_LAYER", "3"))
    hot = int(os.environ.get("FLARE_DS41_PARKING_TEST_HOT", "64"))
    assert 0 < hot < EXPERTS, "the test needs both hot and cold experts"
    rows_sweep = _rows()
    capacity = max(rows_sweep)
    assert capacity > 16, "rows must include a count > 16 so one expert takes more than one 16-route chunk"

    hot_ids = _hot_ids(hot)
    hot_set = set(hot_ids)
    cold_ids = [e for e in range(EXPERTS) if e not in hot_set]
    heavy = (hot_ids[min(1, len(hot_ids) - 1)], cold_ids[len(cold_ids) // 2])
    counts_dir = tmp_path / "counts"
    counts_dir.mkdir()
    (counts_dir / f"expert-routes-rank0-layer{layer}.json").write_text(
        json.dumps({"counts": [2 if e in hot_set else 1 for e in range(EXPERTS)]}))

    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.deepseek41.recipe import _parked_bytes
    from ignition.recipes.deepseek41.tp3_config import TP3Config
    from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader, hot_expert_ids

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    base = _base_env(snapshot)
    per_expert_shard = -(-2304 // WORLD)
    with PackageStorage(rank=0, world=1, device=0) as storage:
        manifest = inspect_indexed_checkpoint(Path(snapshot))
        loaded, config = {}, None
        for name, env in _configs(layer, hot, counts_dir).items():
            _apply(monkeypatch, {**base, **env})
            tp3 = TP3Config.from_environment()
            if env:
                assert hot_expert_ids(layer, EXPERTS, tp3) == hot_ids, "synthetic route counts must select the intended hot set"
            host_before, mixed_before = len(storage.host_regions), len(storage.mixed_regions)
            # Production passes the storage's host allocator to every layer; the TP3Config decides parking.
            with DeepSeekWeightLoader(manifest, device, shard=TensorParallelShard(0, WORLD),
                                      prefixes=(f"layers.{layer}.",), expert_scale_compression="csf",
                                      host_allocator=storage.host_allocate, tp3=tp3) as loader:
                weights, config = loader.load_layer(layer), loader.config
            torch.cuda.synchronize(device)
            assert config.text["moe_intermediate_size"] == 2304 and config.text["n_routed_experts"] == EXPERTS
            host = storage.host_regions[host_before:]
            mixed = storage.mixed_regions[mixed_before:]
            _check_maps(name, weights, hot_ids, cold_ids)

            # Planner bookkeeping vs. what PackageStorage actually mapped in host memory for this configuration.
            expected = _parked_bytes(WORLD, tp3)
            actual = sum(region.extent for region in host) + sum(region.host_bytes for region in mixed)
            slack = len(host) * mmap.PAGESIZE + sum(region.granularity for region in mixed)
            print(f"[parking] layer {layer} {name}: _parked_bytes({WORLD})={expected} host_bytes={actual} "
                  f"regions=host:{len(host)} mixed:{len(mixed)} exact={expected == actual}", flush=True)
            if name == "resident":
                assert expected == 0 and actual == 0 and not host and not mixed, (expected, actual, len(host), len(mixed))
                assert not any(_within(t, storage.host_regions) or _within(t, storage.mixed_regions)
                               for t in _bank_tensors(weights.experts.experts)), "resident bank left VRAM"
            else:
                cold_values = (EXPERTS - hot) * (2 * per_expert_shard * HIDDEN // 2 + HIDDEN * per_expert_shard // 2)
                assert expected == cold_values, "recipe._parked_bytes geometry drifted from the MXFP4 w13 + w2 layout"
                assert actual > 0 and abs(actual - expected) <= slack, (name, expected, actual, slack)
                if name == "split":
                    assert len(host) == 2 and not mixed, (len(host), len(mixed))
                    assert all(_within(t, host) for t in _bank_tensors(weights.experts.cold)), "cold bank is not in host regions"
                    assert not any(_within(t, host) for t in _bank_tensors(weights.experts.experts)), "hot bank parked"
                else:
                    assert len(mixed) == 2 and not host, (len(host), len(mixed))
                    for tensor, region in zip(_bank_tensors(weights.experts.experts), mixed, strict=True):
                        assert tensor.untyped_storage().data_ptr() == region.address, "merged bank is not the mixed region"
                        row = tensor[0].numel() * tensor.element_size()
                        up = lambda n: -(-n // region.granularity) * region.granularity
                        assert region.device_bytes == min(up(hot * row), up(EXPERTS * row)), \
                            "mixed region VRAM prefix does not cover exactly the hot rows"
            loaded[name] = weights
        _apply(monkeypatch, base)

        buffers = _Buffers(capacity, device)
        arena = ScratchArena(device, overlap=False)
        failures = []
        with RecordedPreparationSession(device=device, checkpoint="ds41-expert-parking-test",
                                        workload_schema="tp3-expert-parking", numerical_policy="checkpoint",
                                        compile_workers=8) as session:
            session.register_scratch(arena)
            paths = {m: {name: _Path(f"layer{layer}.{name}.m{m}", weights, config, m, buffers,
                                     session=session, workspace=arena)
                         for name, weights in loaded.items()} for m in rows_sweep}
            session.tune(show_progress=False)
            for m in rows_sweep:
                for seed in SEEDS:
                    ids, route_weights = _routes(m, hot_ids, cold_ids, heavy, seed=seed * 1000 + m)
                    counts = torch.bincount(ids.reshape(-1).long(), minlength=EXPERTS)
                    if m > 16:
                        assert int(counts[heavy[0]]) == m > 16 and int(counts[heavy[1]]) == m > 16, counts[list(heavy)]
                    source = torch.randn((m, HIDDEN), generator=torch.Generator().manual_seed(seed * 7919 + m))
                    outputs = {}
                    for name, path in paths[m].items():
                        runs = 2 if name == "resident" else 1
                        for repeat in range(runs):
                            buffers.source[:m].copy_(source.to(torch.bfloat16))
                            buffers.router.ids[:m].copy_(ids)
                            buffers.router.weights[:m].copy_(route_weights)
                            buffers.routed[:m].fill_(float("nan"))  # an unwritten output cannot pass by luck
                            if path.cold_routed is not None:
                                path.cold_routed.fill_(float("nan"))
                            torch.cuda.synchronize(device)
                            result = path().clone()
                            torch.cuda.synchronize(device)
                            outputs[(name, repeat)] = result
                        if name != "resident":
                            assert path.cold_routes() > 0, f"{name}: no route reached a cold expert"
                            if path.cold_routed is not None:
                                assert bool(torch.isfinite(path.cold_routed).all()) and float(path.cold_routed.abs().max()) > 0
                    reference = outputs[("resident", 0)]
                    assert bool(torch.isfinite(reference).all()), "resident output is not finite"
                    floor = _stats(outputs[("resident", 1)], reference)
                    peak = float(reference.abs().max())
                    bound = max(2 * BF16_ULP * peak, 4 * floor["max_abs"])
                    line = [f"resident-rerun: maxabs={floor['max_abs']:.3e} cos={floor['cosine']:.7f}"]
                    for name in ("split", "merged"):
                        st = _stats(outputs[(name, 0)], reference)
                        line.append(f"{name}: cos={st['cosine']:.7f} maxabs={st['max_abs']:.3e} "
                                    f"maxrel={st['max_rel']:.3e} relnorm={st['rel_norm']:.3e} "
                                    f"cold_routes={paths[m][name].cold_routes()}")
                        if not st["finite"] or st["cosine"] < COSINE or st["max_abs"] > bound:
                            failures.append((name, m, seed, st, bound))
                    print(f"[parking] layer {layer} m={m} seed={seed} peak={peak:.3e} bound={bound:.3e} "
                          f"max_routes/expert={int(counts.max())} | " + " | ".join(line), flush=True)
            del paths
        assert not failures, f"parked expert layouts differ from the resident bank: {failures}"
    torch.cuda.synchronize(device)

# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 expert banks on CPU: resident, split hot + cold, and merged VRAM/host banks.

The CUDA test (test_deepseek41_expert_parking_cuda.py) checks numerics on a real checkpoint; this one checks the
wiring with stubs, without a GPU:

* weights.py ``DeepSeekWeightLoader.load_layer`` (its real ``__init__`` and ``load_experts``; the checkpoint reads and
  ``load_mxfp4_experts`` are stubbed) builds the bank layout TP3Config asks for: the hot set comes from the route counts
  in expert-id space; ``hot_map`` / ``cold_map`` partition the 384 ids and agree with each bank's slot order
  (``bank[map[e]] == e``, i.e. maps are indexed by expert id and hold slots, not the reverse);
* the bytes the loader hands the host allocator (``host_allocate`` regions plus ``mixed_allocate`` host tails) equal
  recipe.py ``_parked_bytes``, the number the planner subtracts from the VRAM budget;
* decoder.py ``PreparedLayer`` builds a second ``PreparedRoutedExperts`` for a cold bank and adds its plan to the
  shared-expert overlap's ``parallel_plans`` exactly when a cold bank exists; a merged bank has one bank and one remap;
* ``PreparedLayer.__call__``'s expert section (index_select remaps, experts(), cold_experts(), routed.add_) delivers
  every router id to exactly one bank row holding that expert, checked with a one-hot stub MoE;
* BRRR refuses a split bank and accepts a merged one.

Each check has a negative control that must fail: swapped hot/cold maps, a merged map stored slot -> id instead of
id -> slot, a hot set read in slot space, a dropped cold plan, and a placement the byte count does not cover.
"""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.deepseek41 import decoder, recipe, weights
from ignition.recipes.deepseek41.tp3_config import TP3Config

WORLD, EXPERTS, TOP_K, HIDDEN, INTERMEDIATE = 3, 384, 6, 5120, 2304
HOT = 64
PARKED = (20, 21)   # parked layers loaded in the loader tests
RESIDENT = 5        # a layer outside host_expert_layers


def _hot_ids():
    """Experts 1, 7, 13, ... (e % 6 == 1): 64 hot ids that are not a prefix of 0..383."""
    return [e for e in range(EXPERTS) if e % 6 == 1]


def _counts(layer):
    """Route counts in expert-id space: hot ids get large, distinct counts; the rest small ones (layer-dependent)."""
    hot = set(_hot_ids())
    return [10_000 + 3 * e + layer if e in hot else (e * 7 + layer) % 97 for e in range(EXPERTS)]


@pytest.fixture
def counts_dir(tmp_path):
    for layer in (*PARKED, RESIDENT):
        (tmp_path / f"expert-routes-rank0-layer{layer}.json").write_text(json.dumps({"counts": _counts(layer)}))
    return tmp_path


# --------------------------------------------------------------------------------------------------------------------
# Loader stubs


class FakeBank:
    """What the stubbed ``load_mxfp4_experts`` returns: the expert ids in slot order and where they were placed."""

    def __init__(self, ids, local, host):
        self.ids, self.local, self.host = tuple(ids), local, host

    def __len__(self):
        return len(self.ids)


class FakeStorage:
    """PackageStorage stand-in: counts the host bytes of host regions and of mixed regions' host tails."""

    def __init__(self):
        self.host_bytes = 0
        self.regions = []

    def host_allocate(self, name, shape, dtype):
        self.host_bytes += math.prod(shape) * dtype.itemsize
        self.regions.append(("host", name, tuple(shape), None))
        return SimpleNamespace(device_view=None)

    def mixed_allocate(self, name, shape, dtype, *, device_rows):
        assert 0 <= device_rows <= shape[0]
        self.host_bytes += (shape[0] - device_rows) * math.prod(shape[1:]) * dtype.itemsize
        self.regions.append(("mixed", name, tuple(shape), device_rows))
        return SimpleNamespace(device_view=None)


def _loader(monkeypatch, tp3, storage):
    """A real DeepSeekWeightLoader (its __init__ and load_experts run) over stubbed checkpoint reads."""
    text = dict(hidden_size=HIDDEN, moe_intermediate_size=INTERMEDIATE, n_routed_experts=EXPERTS, swiglu_limit=10.,
                kv_source_layer_ids=(), index_source_layer_ids=(), engram_layer_ids=(), compress_ratios={})
    config = SimpleNamespace(text=text, vision=None, draft=SimpleNamespace(experts=EXPERTS))
    monkeypatch.setattr(weights.BlockFP8WeightLoader, "__init__",
                        lambda self, manifest, device, **kwargs: setattr(self, "device", torch.device(device)))
    monkeypatch.setattr(weights, "ModelConfig", SimpleNamespace(load=lambda path, mini: config))
    loader = weights.DeepSeekWeightLoader(SimpleNamespace(metadata_path=None), "cpu",
        shard=TensorParallelShard(0, WORLD), expert_scale_compression="csf",
        host_allocator=storage.host_allocate, tp3=tp3)

    def load_mxfp4_experts(prefixes, *, hidden_size, intermediate_range, host_allocator, **kwargs):
        local = intermediate_range[1] - intermediate_range[0]
        ids = [int(prefix.rsplit(".", 1)[1]) for prefix in prefixes]
        if host_allocator is not None:  # the packed w13 / w2 values move to host memory (scales stay on the GPU)
            host_allocator(f"{prefixes[0]}.w13", (len(ids), 2 * local, hidden_size // 2), torch.uint8)
            host_allocator(f"{prefixes[0]}.w2", (len(ids), hidden_size, local // 2), torch.uint8)
        return FakeBank(ids, local, host_allocator is not None)

    loader.load_mxfp4_experts = load_mxfp4_experts
    for name in ("load_tensor", "load_padded_tensor", "load_linear", "load_linear_group", "load_linear_columns",
                 "load_attention_output"):
        setattr(loader, name, lambda *args, **kwargs: object())
    return loader


# --------------------------------------------------------------------------------------------------------------------
# Bank checks (shared by the loader tests, the layer tests and the negative controls)


def _map(tensor):
    assert tensor is not None and tensor.dtype == torch.int32 and tuple(tensor.shape) == (EXPERTS,)
    return tensor.tolist()


def _assert_bank(experts):
    """Check one ExpertWeights layout; return the expert ids held in VRAM slots."""
    if experts.cold is None and experts.hot_map is None:  # one bank, router ids used as-is
        assert experts.experts.ids == tuple(range(EXPERTS)) and experts.cold_map is None
        return set() if experts.experts.host else set(range(EXPERTS))
    hot_map = _map(experts.hot_map)
    if experts.cold is None:  # merged: one hot-first bank, VRAM prefix + host tail
        assert experts.cold_map is None and experts.experts.host
        assert sorted(hot_map) == list(range(EXPERTS)), "merged hot_map must be a permutation of the slots"
        assert all(experts.experts.ids[hot_map[e]] == e for e in range(EXPERTS)), "hot_map disagrees with slot order"
        rows = experts.hot_rows
        assert type(rows) is int and 0 < rows < EXPERTS
        return set(experts.experts.ids[:rows])
    # split: hot bank in VRAM, cold bank in host memory, -1 = not in this bank
    cold_map = _map(experts.cold_map)
    assert not experts.experts.host and experts.cold.host and experts.hot_rows is None
    assert all((h >= 0) != (c >= 0) for h, c in zip(hot_map, cold_map)), "maps must partition the expert ids"
    for bank, mapping in ((experts.experts, hot_map), (experts.cold, cold_map)):
        slots = sorted(slot for slot in mapping if slot >= 0)
        assert slots == list(range(len(bank))), "a map must cover its bank's slots exactly once"
        assert all(bank.ids[slot] == e for e, slot in enumerate(mapping) if slot >= 0), "map disagrees with slot order"
    return set(experts.experts.ids)


def _assert_hot_set(vram_ids, layer):
    """The VRAM experts are the HOT most-routed ids, counts read in expert-id space."""
    counts = _counts(layer)
    expected = set(sorted(range(EXPERTS), key=lambda e: (-counts[e], e))[:HOT])
    assert vram_ids == expected
    assert sum(counts[e] for e in vram_ids) == sum(sorted(counts, reverse=True)[:HOT])


def _per_expert_bytes(local):
    return 2 * local * HIDDEN // 2 + HIDDEN * local // 2  # MXFP4 w13 + w2 values


SPLIT = dict(host_expert_layers=frozenset(PARKED), hot_experts=HOT)
CONFIGS = {
    "resident": dict(),
    "park_all": dict(host_expert_layers=frozenset(PARKED)),
    "split": SPLIT,
    "merged": dict(SPLIT, merged_bank=True),
    "all_hot": dict(host_expert_layers=frozenset(PARKED), hot_experts=EXPERTS),
}


def _load(monkeypatch, counts_dir, name):
    tp3 = TP3Config(**CONFIGS[name], expert_counts_dir=counts_dir)
    storage = FakeStorage()
    loader = _loader(monkeypatch, tp3, storage)
    layers = {layer: loader.load_layer(layer).experts for layer in (*PARKED, RESIDENT)}
    return tp3, storage, layers


# --------------------------------------------------------------------------------------------------------------------
# Loader: layout, maps, hot set, parked bytes


@pytest.mark.parametrize("name", CONFIGS)
def test_loader_builds_the_configured_bank_and_parked_bytes_match(monkeypatch, counts_dir, name):
    tp3, storage, layers = _load(monkeypatch, counts_dir, name)
    local = INTERMEDIATE // WORLD
    host_experts = 0
    for layer, experts in layers.items():
        vram = _assert_bank(experts)
        parked = layer in PARKED and name != "resident"
        if not parked or name == "all_hot":
            assert experts.cold is None and experts.hot_map is None and vram == set(range(EXPERTS))
            continue
        if name == "park_all":
            assert vram == set() and experts.hot_map is None
            host_experts += EXPERTS
            continue
        assert (experts.cold is not None) == (name == "split")
        assert (experts.cold_map is not None) == (name == "split")
        _assert_hot_set(vram, layer)
        host_experts += EXPERTS - HOT
        if name == "merged":
            first = f"layers.{layer}.ffn.experts.{experts.experts.ids[0]}.w13"
            region = next(r for r in storage.regions if r[1] == first)
            assert region[0] == "mixed" and region[3] == experts.hot_rows == HOT
    banks = [bank for experts in layers.values() for bank in (experts.experts, experts.cold) if bank is not None]
    assert all(bank.local == local for bank in banks)
    assert storage.host_bytes == host_experts * _per_expert_bytes(local)
    assert storage.host_bytes == recipe._parked_bytes(WORLD, tp3)
    if name in ("resident", "all_hot"):
        assert recipe._parked_bytes(WORLD, tp3) == 0 and not storage.regions


def test_negative_controls_for_the_bank_checks(monkeypatch, counts_dir):
    tp3, storage, layers = _load(monkeypatch, counts_dir, "split")
    split = layers[PARKED[0]]
    with pytest.raises(AssertionError):  # hot and cold maps swapped
        _assert_bank(replace(split, hot_map=split.cold_map, cold_map=split.hot_map))
    shifted = split.cold_map.clone()
    shifted[shifted >= 0] = (shifted[shifted >= 0] + 1) % len(split.cold)
    with pytest.raises(AssertionError):  # a cold map that partitions the ids but points at the wrong slots
        _assert_bank(replace(split, cold_map=shifted))
    with pytest.raises(AssertionError):  # hot set read in slot space: slots 0..63 taken for ids 0..63
        _assert_hot_set(set(range(HOT)), PARKED[0])
    with pytest.raises(AssertionError):  # bytes of the hot experts too would be parked
        assert storage.host_bytes + HOT * _per_expert_bytes(INTERMEDIATE // WORLD) == \
            recipe._parked_bytes(WORLD, tp3)
    _, _, layers = _load(monkeypatch, counts_dir, "merged")
    merged = layers[PARKED[0]]
    with pytest.raises(AssertionError):  # merged map stored slot -> id (the bank order) instead of id -> slot
        _assert_bank(replace(merged, hot_map=torch.tensor(merged.experts.ids, dtype=torch.int32)))


# --------------------------------------------------------------------------------------------------------------------
# PreparedLayer: plans, remaps and the expert section of __call__


class Component:
    """Callable stub with a unique plan; returns its first argument."""

    def __init__(self, *args, **kwargs):
        self.plan = object()
        self.args, self.kwargs = args, kwargs

    def __call__(self, *args):
        return args[0] if args else None


class OneHotExperts(Component):
    """Stub MoE: for each route with slot >= 0, add its weight to column ``bank.ids[slot]`` (the global expert id)."""

    def __init__(self, inputs, bank, ids, route_weights, output, **kwargs):
        super().__init__(inputs, bank, ids, route_weights, output, **kwargs)
        self.bank, self.ids, self.route_weights, self.output = bank, ids, route_weights, output

    def __call__(self):
        self.output.zero_()
        for row, (slots, values) in enumerate(zip(self.ids.tolist(), self.route_weights.tolist())):
            for slot, value in zip(slots, values):
                if slot >= 0:
                    assert slot < len(self.bank), "route outside the bank"
                    self.output[row, self.bank.ids[slot]] += value


def _layer(monkeypatch, experts, *, rows, mode="decode", overlap=True, tp3=None, attention=None):
    calls, made = [], []

    def concurrent(shared, **kwargs):
        calls.append((shared, kwargs))
        return SimpleNamespace(start=lambda inputs: None, finish=lambda inputs, routed: None)

    def routed(*args, **kwargs):
        made.append(OneHotExperts(*args, **kwargs))
        return made[-1]

    for name in ("PreparedMHC", "PreparedAttention", "PreparedAllReduce", "PreparedChunkedAllReduce",
                 "UnquantizedProjection", "PreparedRouter", "PreparedMLP", "PreparedMHCPost"):
        monkeypatch.setattr(decoder, name, Component)
    if attention is not None:
        monkeypatch.setattr(decoder, "PreparedAttention", lambda *args, **kwargs: attention)
    monkeypatch.setattr(decoder, "PreparedRoutedExperts", routed)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    monkeypatch.setattr(decoder, "ConcurrentSharedExpert", concurrent)
    monkeypatch.setattr(decoder, "_BRRR_DECISIONS", [])
    text = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20,
                scoring_func="sigmoid", norm_topk_prob=True, routed_scaling_factor=1., swiglu_limit=10.)
    scratch = decoder.Scratch.allocate(rows, 1, WORLD, "cpu")
    layer_weights = SimpleNamespace(engram=None, attention=None, experts=experts)
    layer = decoder.PreparedLayer(PARKED[0], layer_weights, None, scratch,
        SimpleNamespace(token_ids=torch.zeros(rows, dtype=torch.int64), positions=torch.zeros(rows, dtype=torch.int64)),
        SimpleNamespace(config=SimpleNamespace(text=text)), SimpleNamespace(frequencies=(object(), object())),
        shard=TensorParallelShard(0, WORLD), mode=mode, runtime=None, coordinator=None, session=object(),
        name="layer", workspace=SimpleNamespace(overlap=overlap), carry=False, defer_post=True, tp3=tp3)
    return layer, made, calls


def _expected_plans(experts, layer):
    """The overlap's parallel plans as derived from the weights: the cold plan iff the weights have a cold bank."""
    plans = (layer.router_projection.plan, layer.router.plan, layer.experts.plan)
    return plans + ((layer.cold_experts.plan,) if experts.cold is not None else ())


def _assert_layer(experts, layer, made, calls, rows):
    s = layer.scratch
    assert (layer.cold_experts is not None) == (experts.cold is not None)
    assert len(made) == (2 if experts.cold is not None else 1)
    hot = made[0]
    assert hot is layer.experts and hot.bank is experts.experts and hot.output is s.routed
    if experts.cold is not None:
        cold = made[1]
        assert cold is layer.cold_experts and cold.bank is experts.cold
        assert cold.ids is layer.cold_ids and cold.output is layer.cold_routed
        assert cold.route_weights is s.router.weights
        assert hot.ids is layer.hot_ids and layer.hot_map is experts.hot_map and layer.cold_map is experts.cold_map
        assert layer.cold_ids.data_ptr() != layer.hot_ids.data_ptr() != s.router.ids.data_ptr()
    elif experts.hot_map is not None:
        assert hot.ids is layer.hot_ids and layer.hot_map is experts.hot_map and layer.cold_map is None
    else:
        assert hot.ids is s.router.ids and layer.hot_map is None
    if rows <= 128:
        shared, options = calls.pop()
        assert shared is layer.shared
        assert options["parallel_plans"] == _expected_plans(experts, layer)
    assert not calls


def _routes(rows, vram, *, seed):
    """Router ids mixing VRAM and host experts in every row; weights are multiples of 1/8 (exact in BF16)."""
    g = torch.Generator().manual_seed(seed)
    vram, host = sorted(vram), sorted(set(range(EXPERTS)) - set(vram))
    ids = torch.empty((rows, TOP_K), dtype=torch.int32)
    for row in range(rows):
        picks = [vram[int(torch.randint(len(vram), (1,), generator=g))],
                 host[int(torch.randint(len(host), (1,), generator=g))]]
        for e in torch.randperm(EXPERTS, generator=g).tolist():
            if len(picks) == TOP_K:
                break
            if e not in picks:
                picks.append(e)
        ids[row] = torch.tensor(picks, dtype=torch.int32)
    route_weights = torch.randint(1, 9, (rows, TOP_K), generator=g).float() / 8
    return ids, route_weights


def _run_experts(layer, ids, route_weights):
    s = layer.scratch
    s.router.ids.copy_(ids)
    s.router.weights.copy_(route_weights)
    layer()
    reference = torch.zeros(ids.shape[0], HIDDEN)
    reference.scatter_add_(1, ids.long(), route_weights)
    return s.routed.float(), reference


@pytest.mark.parametrize("name", ("resident", "split", "merged"))
@pytest.mark.parametrize("rows", (1, 5, 48, 129))
def test_layer_wires_the_bank_and_delivers_every_route_once(monkeypatch, counts_dir, name, rows):
    _, _, layers = _load(monkeypatch, counts_dir, name)
    experts = layers[PARKED[0]]
    vram = _assert_bank(experts)
    layer, made, calls = _layer(monkeypatch, experts, rows=rows)
    _assert_layer(experts, layer, made, calls, rows)
    ids, route_weights = _routes(rows, vram if 0 < len(vram) < EXPERTS else set(range(HOT)), seed=rows)
    routed, reference = _run_experts(layer, ids, route_weights)
    assert torch.equal(routed, reference)
    assert torch.equal(layer.scratch.mlp_partial.float(), reference)  # shared output is zero in the stub


@pytest.mark.parametrize("mode", ("decode", "verify", "draft", "prefill"))
def test_cold_plan_joins_the_overlap_exactly_with_a_cold_bank(monkeypatch, counts_dir, mode):
    expected = mode != "prefill"
    for name in ("resident", "split", "merged"):
        _, _, layers = _load(monkeypatch, counts_dir, name)
        experts = layers[PARKED[0]]
        layer, made, calls = _layer(monkeypatch, experts, rows=8, mode=mode)
        assert (layer.concurrent_shared is not None) == expected
        if not expected:
            assert not calls
            continue
        plans = calls[-1][1]["parallel_plans"]
        assert (layer.cold_experts is not None and layer.cold_experts.plan in plans) == (name == "split")
        assert len(plans) == (4 if name == "split" else 3)
        _assert_layer(experts, layer, made, calls, 8)


def test_negative_controls_for_the_layer_checks(monkeypatch, counts_dir):
    _, _, layers = _load(monkeypatch, counts_dir, "split")
    split = layers[PARKED[0]]
    layer, made, calls = _layer(monkeypatch, split, rows=8)
    options = calls[-1][1]
    with pytest.raises(AssertionError):  # an overlap that drops the cold plan
        assert options["parallel_plans"][:3] == _expected_plans(split, layer)
    # Swapped maps: hot ids land in the cold bank and vice versa (slots out of range or the wrong experts).
    swapped = replace(split, hot_map=split.cold_map, cold_map=split.hot_map)
    layer, _, _ = _layer(monkeypatch, swapped, rows=8)
    ids, route_weights = _routes(8, _assert_bank(split), seed=3)
    with pytest.raises(AssertionError):
        routed, reference = _run_experts(layer, ids, route_weights)
        assert torch.equal(routed, reference)
    # Merged map stored slot -> id: routes reach the wrong experts.
    _, _, layers = _load(monkeypatch, counts_dir, "merged")
    merged = layers[PARKED[0]]
    wrong = replace(merged, hot_map=torch.tensor(merged.experts.ids, dtype=torch.int32))
    layer, _, _ = _layer(monkeypatch, wrong, rows=8)
    routed, reference = _run_experts(layer, *_routes(8, _assert_bank(merged), seed=4))
    assert not torch.equal(routed, reference)


def test_brrr_takes_a_merged_bank_and_refuses_a_split_bank(monkeypatch, counts_dir, tmp_path):
    from ignition.recipes.deepseek41 import brrr
    built = []
    monkeypatch.setattr(brrr, "PreparedBRRRTail", lambda *args, **kwargs: built.append(args) or object())
    # What PreparedBRRRTail's construction reads from the attention.
    attention = SimpleNamespace(window_metadata=SimpleNamespace(position=None), ratio=0, plan=object())
    for name in ("split", "merged"):
        tp3, _, layers = _load(monkeypatch, counts_dir, name)
        tp3 = replace(tp3, brrr_layers=frozenset({PARKED[0]}), brrr_ckpt=tmp_path)
        experts = layers[PARKED[0]]
        if name == "split":
            with pytest.raises(ValueError, match="one expert bank"):
                _layer(monkeypatch, experts, rows=4, tp3=tp3, attention=attention)
            assert not built
        else:
            layer, _, _ = _layer(monkeypatch, experts, rows=4, tp3=tp3, attention=attention)
            assert layer.brrr is not None and len(built) == 1 and attention.external_tail
            assert decoder._BRRR_DECISIONS == [("layer", PARKED[0], "decode", 4, True, False)]

# SPDX-License-Identifier: Apache-2.0
"""TP3 graph-scratch savings (CPU only): chunked Engram projection rows and engram_residual on ffn_hc.residual.

Both are default-off TP3Config switches. The tests pin the default layout, the shapes and aliases of the opt-in
layouts and their pool savings, and check the lifetimes by running the real ``PreparedLayer.__call__`` and
``PreparedEngram.__call__`` (the traced forward code) over CPU stand-ins for the kernels: a shared-storage layout must
give bit-identical results to the separate one, and a negative control (an Engram layer receiving a deferred post)
must diverge.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.scratch import GraphBufferPool
from ignition.recipes.deepseek41 import engram as engram_module
from ignition.recipes.deepseek41.decoder import (PreparedLayer, Scratch, aliases, check_engram_residual,
                                                 defers_post)
from ignition.recipes.deepseek41.probe import Probe
from ignition.recipes.deepseek41.tp3_config import TP3Config

MiB = 1 << 20


# --- configuration ---

def test_switches_default_off_and_parse_from_the_environment():
    assert TP3Config().engram_chunk_rows == 0 and TP3Config().share_engram_residual is False
    assert TP3Config.from_environment({}) == TP3Config()
    config = TP3Config.from_environment({"FLARE_DS41_ENGRAM_CHUNK_ROWS": "1024",
                                         "FLARE_DS41_SHARE_ENGRAM_RESIDUAL": "1"})
    assert config.engram_chunk_rows == 1024 and config.share_engram_residual is True
    assert TP3Config.from_environment({"FLARE_DS41_ENGRAM_CHUNK_ROWS": ""}).engram_chunk_rows == 0
    assert not TP3Config.from_environment({"FLARE_DS41_SHARE_ENGRAM_RESIDUAL": "yes"}).share_engram_residual
    record = config.to_metadata()
    assert record["engram_chunk_rows"] == 1024 and record["share_engram_residual"] is True
    assert TP3Config().to_metadata()["engram_chunk_rows"] == 0
    assert "engram_chunk_rows=1024" in config.describe() and "share_engram_residual=True" in config.describe()
    assert TP3Config().describe() == "TP3 config: defaults"


@pytest.mark.parametrize("env,error", [
    ({"FLARE_DS41_ENGRAM_CHUNK_ROWS": "1000"}, "multiple of 128"),
    ({"FLARE_DS41_ENGRAM_CHUNK_ROWS": "-128"}, "non-negative integer"),
])
def test_invalid_chunk_rows_raise(env, error):
    with pytest.raises(ValueError, match=error):
        TP3Config.from_environment(env)
    with pytest.raises(TypeError):
        TP3Config(share_engram_residual=1)


# --- layouts ---

def _pool(chunk_rows=0, share=False, *, world=3, prefill=4096, decode=192, device="meta"):
    pool = GraphBufferPool(device, chunk_bytes=2 << 20)
    options = dict(engram_chunk_rows=chunk_rows, share_engram_residual=share)
    prefill_scratch = Scratch.allocate(prefill, 1, world, device, buffers=pool.cursor(), candidate_tokens=prefill,
                                       **options)
    Scratch.allocate(128, 1, world, device, buffers=prefill_scratch.buffers, **options)
    decode_scratch = Scratch.allocate(decode, 64, world, device, buffers=pool.cursor(), **options)
    return pool, prefill_scratch, decode_scratch


def _layout(scratch):
    def walk(value, prefix):
        if isinstance(value, torch.Tensor):
            yield prefix, tuple(value.shape), value.dtype, value.storage_offset()
        elif hasattr(value, "__dataclass_fields__"):
            for name in value.__dataclass_fields__:
                yield from walk(getattr(value, name), f"{prefix}.{name}")
    return list(walk(scratch, "scratch"))


def test_default_layout_is_the_base_layout():
    pool, prefill, _ = _pool()
    explicit, _, _ = _pool(0, False)
    plain = GraphBufferPool("meta", chunk_bytes=2 << 20)
    base = Scratch.allocate(4096, 1, 3, "meta", buffers=plain.cursor(), candidate_tokens=4096)
    assert _layout(base) == _layout(prefill)
    assert pool.nbytes == explicit.nbytes
    assert not aliases(prefill.engram_residual, prefill.ffn_hc.residual)
    assert prefill.engram.partial.shape == (4096, 25600) and prefill.engram.local.shape == (4096, 2048)
    # Pin of the modelled base pool (prefill 4096 + suffix 128 + a 192-row decode cursor, TP3): a change here means
    # the default package's activation layout moved.
    assert pool.nbytes == 2060025856  # same value from the base branch (ds41-tp3-v3)


def test_chunked_engram_scratch_holds_one_chunk_and_saves_300_mib():
    base, _, _ = _pool()
    pool, prefill, decode = _pool(1024)
    engram = prefill.engram
    assert engram.mask.shape == (4096,) and engram.gathered.shape == (4096, 6144)
    assert engram.local.shape == (1024, 2048)
    assert engram.partial.shape == engram.projected.shape == (1024, 25600)
    assert decode.engram.partial.shape == (192, 25600)  # narrower graphs keep one pass
    # Graph views: rows(capacity) keeps one chunk in wide prefill graphs and every row in narrow ones.
    assert prefill.rows(2048, 1).engram.partial.shape == (1024, 25600)
    assert prefill.rows(512, 1).engram.partial.shape == (512, 25600)
    assert prefill.rows(512, 1).engram.local.shape == (512, 2048)
    assert base.nbytes - pool.nbytes == 300 * MiB
    # Unsharded (world 1): local becomes its own chunk buffer and partial is the projected output.
    single = Scratch.allocate(4096, 1, 1, "meta", candidate_tokens=4096, engram_chunk_rows=1024).engram
    assert single.local.shape == (1024, 6144) and single.partial is single.projected
    assert Scratch.allocate(512, 1, 1, "meta", engram_chunk_rows=1024).engram.local.shape == (512, 6144)


def test_shared_engram_residual_aliases_ffn_residual_in_every_view_and_saves_its_storage():
    base, _, _ = _pool()
    pool, prefill, decode = _pool(share=True)
    for scratch in (prefill, decode, prefill.rows(512, 1), decode.rows(8, 8)):
        assert scratch.engram_residual.shape == scratch.ffn_hc.residual.shape
        assert aliases(scratch.engram_residual, scratch.ffn_hc.residual)
        assert scratch.engram_residual.storage_offset() == scratch.ffn_hc.residual.storage_offset()
        for other in (scratch.residual, scratch.attention_hc.residual, scratch.mlp_output, scratch.attention_output):
            assert not aliases(scratch.engram_residual, other)
    # 160 MiB for the 4096-row prefill buffer, 5 MiB for the 128-row suffix buffer behind it.
    assert base.nbytes - pool.nbytes == 165 * MiB
    both, _, _ = _pool(1024, True)
    assert base.nbytes - both.nbytes == 465 * MiB


def test_both_switches_keep_cross_rank_shapes_identical():
    """Each rank allocates from its own TP3Config; one config gives every rank the same shapes and offsets."""
    config = TP3Config.from_environment({"FLARE_DS41_ENGRAM_CHUNK_ROWS": "1024",
                                         "FLARE_DS41_SHARE_ENGRAM_RESIDUAL": "1"})
    layouts = [_layout(_pool(config.engram_chunk_rows, config.share_engram_residual)[1]) for _ in range(3)]
    assert layouts[0] == layouts[1] == layouts[2]


# --- lifetimes through the real PreparedLayer.__call__ ---

class _Stand:
    """CPU stand-ins with the real components' operand wiring (outputs fixed at construction, as prepared)."""

    def __init__(self, s, index):
        self.s, self.k = s, float(index + 1)

    def engram(self, source):  # PreparedEngram: reads residual, writes engram_residual (its output)
        self.s.engram_residual.copy_(source * 0.75 + 0.125 * self.k)
        return self.s.engram_residual

    def attention_hc(self, *incoming):  # PreparedMHC pre (one input) or lagged post+pre (four inputs)
        s = self.s
        if len(incoming) == 1:
            x = incoming[0].float()
        else:
            branch, residual, post, comb = (value.float() for value in incoming)
            x = torch.einsum("tij,tjh->tih", comb, residual) + post[..., None] * branch[:, None, :]
        s.attention_hc.residual.copy_(x * 0.5 + 0.25)
        s.attention_hc.post.copy_(x.mean(-1) * 0.01 + 1)
        s.attention_hc.comb.copy_(torch.eye(4) * 0.9 + 0.025)
        s.attention_hc.normalized.copy_(x.sum(1) * 0.25)

    def attention(self, normalized):
        self.s.attention.output.copy_(normalized * 0.5 - 0.0625 * self.k)
        return self.s.attention.output

    def attention_reduce(self, source):
        self.s.attention_output.copy_(source * 3)

    def ffn_hc(self, branch, residual, post, comb):
        s = self.s
        x = torch.einsum("tij,tjh->tih", comb.float(), residual.float()) + post[..., None] * branch.float()[:, None]
        s.ffn_hc.residual.copy_(x)
        s.ffn_hc.post.copy_(x.mean(-1) * 0.02 + 0.5)
        s.ffn_hc.comb.copy_(torch.eye(4) * 0.8 + 0.05)
        s.ffn_hc.normalized.copy_(x.sum(1) * 0.25)

    def router_projection(self, normalized):
        self.s.router_logits.copy_(normalized[:, :self.s.router_logits.shape[1]].float())
        return self.s.router_logits

    def router(self, logits):
        pass

    def experts(self):
        self.s.routed.copy_(self.s.ffn_hc.normalized * 0.3)

    def shared(self, normalized):
        self.s.shared.output.copy_(normalized * 0.2 + 0.03125)

    def mlp_reduce(self, partial):
        self.s.mlp_output.copy_(partial * 3)

    def post(self, branch, residual, post, comb):
        x = torch.einsum("tij,tjh->tih", comb.float(), residual.float()) + post[..., None] * branch.float()[:, None]
        self.s.residual.copy_(x)

    def brrr(self, attended):  # PreparedBRRRTail: attention output .. MLP all-reduce, in one kernel
        s = self.s
        x = s.attention_hc.residual.float() + attended.float()[:, None] * s.attention_hc.post[..., None]
        s.ffn_hc.residual.copy_(x)
        s.ffn_hc.post.copy_(x.mean(-1) * 0.02 + 0.5)
        s.ffn_hc.comb.copy_(torch.eye(4) * 0.8 + 0.05)
        s.mlp_output.copy_(x.sum(1) * 0.1)


def _layer(s, index, *, engram, carry, defer, brrr=False):
    stand = _Stand(s, index)
    return SimpleNamespace(scratch=s, probe=Probe(False), probe_index=index, probe_mode="prefill", no_engram=False,
        engram=stand.engram if engram else None, carry=carry, boundary=None, attention_hc=stand.attention_hc,
        attention=stand.attention, attention_reduce=stand.attention_reduce, ffn_hc=stand.ffn_hc,
        brrr=stand.brrr if brrr else None, concurrent_shared=None, image_mask=None, metadata=None,
        router=stand.router, router_projection=stand.router_projection, cold_experts=None, hot_map=None,
        experts=stand.experts, shared=stand.shared, mlp_reduce=stand.mlp_reduce,
        post=None if defer else stand.post)


def _run(share, engram_layers, *, force_carry_into=None, brrr=False, tokens=8):
    s = Scratch.allocate(tokens, 1, 3, "cpu", candidate_tokens=tokens, share_engram_residual=share)
    generator = torch.Generator().manual_seed(7)
    s.residual.copy_(torch.randn(s.residual.shape, generator=generator))
    for value in (s.ffn_hc.residual, s.engram_residual, s.mlp_output):
        value.copy_(torch.randn(value.shape, generator=generator))  # stale state from an earlier graph
    carry = False
    for index in range(len(engram_layers)):
        defer = defers_post(index, engram_layers)
        if force_carry_into is not None and index + 1 == force_carry_into:
            defer = True  # negative control: break the "no deferred post into an Engram layer" rule
        PreparedLayer.__call__(_layer(s, index, engram=engram_layers[index], carry=carry, defer=defer, brrr=brrr))
        carry = defer
    return s.residual.clone(), s


@pytest.mark.parametrize("brrr", [False, True])
@pytest.mark.parametrize("engram_layers", [
    (False, True, False, False),         # Engram in layer 1 (layers 1 and 14 in the 40-layer model)
    (True, False, False, True, False),   # Engram first, and again after a deferred chain
    (False, True, True, False),          # back-to-back Engram layers
])
def test_shared_residual_matches_separate_storage_through_the_layer_forward(engram_layers, brrr):
    separate, _ = _run(False, engram_layers, brrr=brrr)
    shared, scratch = _run(True, engram_layers, brrr=brrr)
    assert aliases(scratch.engram_residual, scratch.ffn_hc.residual)
    assert torch.equal(separate, shared)


def test_negative_control_a_deferred_post_into_an_engram_layer_diverges_and_is_rejected():
    layers = (False, True, False)
    separate, _ = _run(False, layers, force_carry_into=1)
    shared, scratch = _run(True, layers, force_carry_into=1)
    assert not torch.equal(separate, shared)  # the Engram mix overwrote the carried ffn_hc.residual
    with pytest.raises(ValueError, match="must not receive a deferred mHC post"):
        check_engram_residual(1, scratch, carry=True)
    check_engram_residual(1, scratch, carry=False)
    check_engram_residual(1, Scratch.allocate(8, 1, 3, "cpu", candidate_tokens=8), carry=True)  # separate storage


def test_the_model_never_defers_a_post_into_an_engram_layer():
    for count, engram_ids, aux in ((40, (1, 14), (6, 20, 33)), (8, (1,), ()), (6, (1,), ())):
        flags = [index in engram_ids for index in range(count)]
        for index in range(count):
            if index + 1 < count and flags[index + 1]:
                assert not defers_post(index, flags, aux)
        assert not defers_post(count - 1, flags, aux)
        assert all(not defers_post(layer - 1, flags, aux) for layer in aux)


# --- chunked Engram through the real PreparedEngram ---

class _Projection:
    created = []

    def __init__(self, source, weight, output, *, session, name, workspace, activation_mode):
        assert output.shape[0] == source.shape[0] and output.shape[1] == 25600
        self.source, self.output, self.name = source, output, name
        _Projection.created.append(self)

    def __call__(self, source):
        assert source.shape == self.source.shape and source.data_ptr() == self.source.data_ptr()
        repeats = -(-25600 // source.shape[1])
        self.output.copy_((source.float().repeat(1, repeats)[:, :25600] * 0.25 + source.float()[:, :1]))
        return self.output


class _Reduce:
    created = []

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        self.source, self.output, self.name = source, output, name
        _Reduce.created.append(self)

    def __call__(self, source):
        assert source.shape == self.source.shape and source.data_ptr() == self.source.data_ptr()
        self.output.copy_(source.float() * 3)  # three ranks holding the same partial
        return self.output


class _Lookup:
    def __init__(self, inputs, gathered, live, **_):
        self.gathered = gathered

    def __call__(self, source):
        rows = source.shape[0]
        self.gathered.copy_(source.view(rows, -1)[:, :6144].float() * 0.5 + torch.arange(rows)[:, None] * 0.01)


class _Plan:
    def __init__(self, caps, invocation):
        self.caps = caps

    def request(self, name, prepare_call):
        return name


def _mix(state, projected, norm_weight, *, eps, plan, out, token_mask):
    assert state.shape[0] <= plan.caps.max_tokens and projected.shape[0] == state.shape[0] == out.shape[0]
    out.copy_(state.float() + projected[:, :state.shape[1]].float() * token_mask[:, None] * norm_weight[None])


def _engram(monkeypatch, tokens, chunk_rows, world):
    declared = []
    _Projection.created, _Reduce.created = [], []
    monkeypatch.setattr(engram_module, "QuantizedProjection", _Projection)
    monkeypatch.setattr(engram_module, "PreparedChunkedAllReduce", _Reduce)
    monkeypatch.setattr(engram_module, "PreparedEngramLookup", _Lookup)
    monkeypatch.setattr(engram_module, "declare", lambda session, requests: declared.extend(requests))
    monkeypatch.setattr(engram_module, "hyperconnection", SimpleNamespace(
        Caps=lambda **caps: SimpleNamespace(**caps), plan=_Plan, run_engram_mix=_mix))
    s = Scratch.allocate(tokens, 1, world, "cpu", candidate_tokens=tokens, engram_chunk_rows=chunk_rows)
    shard = TensorParallelShard(0, world)
    columns = torch.tensor([head * 256 + column for head in range(shard.rank, 24, world) for column in range(256)])
    tables = SimpleNamespace(inputs={1: None}, token_map=(), geometry=None, readiness_runtime=None,
                             columns=columns, shard=shard)
    metadata = SimpleNamespace(token_ids=torch.arange(tokens) * 7 % 129300, positions=torch.arange(tokens) - 2,
                               live_counts=torch.tensor([tokens]))
    pool = SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6}))
    weights = {"projection": None, "weight": torch.linspace(0.5, 1.5, 20480)}
    prepared = engram_module.PreparedEngram(1, weights, tables, s.residual, s.engram_residual, metadata, pool,
        scratch=s.engram, session=None, name="prefill.layer1.engram", workspace=None,
        runtime=object() if world > 1 else None)
    s.residual.copy_(torch.randn(s.residual.shape, generator=torch.Generator().manual_seed(11)))
    return prepared, s, declared


@pytest.mark.parametrize("world", [3, 1])
@pytest.mark.parametrize("tokens,chunk_rows,parts", [(12, 4, 3), (10, 4, 3), (8, 8, 1), (8, 16, 1)])
def test_chunked_engram_matches_one_pass(monkeypatch, world, tokens, chunk_rows, parts):
    whole, s_whole, declared_whole = _engram(monkeypatch, tokens, 0, world)
    reference = whole(s_whole.residual).clone()
    assert not whole.chunked and [p.name for p in _Projection.created] == ["prefill.layer1.engram.projection"]
    chunked, s_chunked, declared = _engram(monkeypatch, tokens, chunk_rows, world)
    assert torch.equal(s_whole.residual, s_chunked.residual)
    result = chunked(s_chunked.residual)
    assert result is s_chunked.engram_residual
    assert torch.equal(result, reference)
    assert len(chunked.bounds) == parts and chunked.chunked == (parts > 1)
    names = [p.name for p in _Projection.created]
    if tokens % chunk_rows and parts > 1:
        assert names == ["prefill.layer1.engram.projection", "prefill.layer1.engram.projection.tail"]
    else:
        assert names == ["prefill.layer1.engram.projection"]
    assert len(_Reduce.created) == (len(names) if world > 1 else 0)
    assert declared == declared_whole == ["prefill.layer1.engram.mix"]
    assert chunked.mix_plan.caps.max_tokens == min(tokens, chunk_rows)


def test_engram_rejects_inconsistent_chunk_scratch(monkeypatch):
    prepared, s, _ = _engram(monkeypatch, 8, 4, 3)
    bad = engram_module.EngramScratch(s.engram.mask, s.engram.gathered, s.engram.local[:2], s.engram.partial,
                                      s.engram.projected)
    tables = SimpleNamespace(inputs={1: None}, token_map=(), geometry=None, readiness_runtime=None,
                             columns=prepared.columns, shard=TensorParallelShard(0, 3))
    pool = SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6}))
    with pytest.raises(ValueError, match="contiguous lookup and projection geometry"):
        engram_module.PreparedEngram(1, {"projection": None, "weight": torch.ones(20480)}, tables, s.residual,
            s.engram_residual, prepared.metadata, pool, scratch=bad, session=None, name="x", workspace=None,
            runtime=object())

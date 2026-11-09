# SPDX-License-Identifier: Apache-2.0
"""BRRR item-6 switches (split tail, fast route, counter parity): config parsing and sidecar record, kernel build
selection, engine dispatch and export zero ranges, and a CPU mirror of the kernel's split-tail job layout (CPU only;
the kernel itself is checked by test_deepseek41_brrr_tail_cuda.py on three GPUs)."""

import math
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.deepseek41 import brrr
from ignition.recipes.deepseek41.tp3_config import TP3Config

BRRR_ENV = {"FLARE_DS41_BRRR_LAYERS": "0-39", "FLARE_DS41_BRRR_CKPT": "/c"}


def test_switches_default_off_and_parse_from_the_environment():
    plain = TP3Config.from_environment(BRRR_ENV)
    assert plain.brrr_split_tail == 0 and not plain.brrr_fast_route and not plain.brrr_counter_parity
    assert brrr.kernel_defines(plain) == ()
    config = TP3Config.from_environment({**BRRR_ENV, "FLARE_DS41_BRRR_SPLIT_TAIL": "96", "FLARE_DS41_BRRR_FAST_ROUTE": "1",
                                         "FLARE_DS41_BRRR_COUNTER_PARITY": "1"})
    assert config.brrr_split_tail == 96 and config.brrr_fast_route and config.brrr_counter_parity
    # The split count is an engine setting: one BRRR_SPLIT_TAIL=1 build serves every N.
    assert brrr.kernel_defines(config) == ("BRRR_SPLIT_TAIL=1", "BRRR_FAST_ROUTE=1", "BRRR_COUNTER_PARITY=1")
    other = TP3Config.from_environment({**BRRR_ENV, "FLARE_DS41_BRRR_SPLIT_TAIL": "2", "FLARE_DS41_BRRR_FAST_ROUTE": "1",
                                        "FLARE_DS41_BRRR_COUNTER_PARITY": "1"})
    assert brrr.kernel_defines(other) == brrr.kernel_defines(config)
    assert brrr.engine_key(other) != brrr.engine_key(config) and brrr.engine_key(plain) == ((), 0)
    trap = TP3Config.from_environment({**BRRR_ENV, "FLARE_DS41_BRRR_SPIN_TRAP": "1"})
    assert trap.brrr_spin_trap and brrr.kernel_defines(trap) == ("BRRR_SPIN_TRAP=1",)
    # "1" flags like the other kernel switches; empty = unset; split tail is an "or 0" integer.
    assert not TP3Config.from_environment({**BRRR_ENV, "FLARE_DS41_BRRR_FAST_ROUTE": "yes"}).brrr_fast_route
    assert TP3Config.from_environment({**BRRR_ENV, "FLARE_DS41_BRRR_SPLIT_TAIL": ""}).brrr_split_tail == 0
    assert brrr.kernel_defines(TP3Config.from_environment({**BRRR_ENV, "FLARE_DS41_BRRR_COUNTER_PARITY": "1"})) == (
        "BRRR_COUNTER_PARITY=1",)


@pytest.mark.parametrize("env,error,message", [
    ({"FLARE_DS41_BRRR_SPLIT_TAIL": "2"}, ValueError, "brrr_split_tail needs brrr_layers"),
    ({"FLARE_DS41_BRRR_FAST_ROUTE": "1"}, ValueError, "brrr_fast_route needs brrr_layers"),
    ({"FLARE_DS41_BRRR_COUNTER_PARITY": "1"}, ValueError, "brrr_counter_parity needs brrr_layers"),
    ({"FLARE_DS41_BRRR_SPIN_TRAP": "1"}, ValueError, "brrr_spin_trap needs brrr_layers"),
    ({**BRRR_ENV, "FLARE_DS41_BRRR_SPLIT_TAIL": "-1"}, ValueError, "brrr_split_tail must be a non-negative integer"),
    ({**BRRR_ENV, "FLARE_DS41_BRRR_SPLIT_TAIL": "two"}, ValueError, "invalid literal"),
])
def test_switch_validation(env, error, message):
    with pytest.raises(error, match=message):
        TP3Config.from_environment(env)
    with pytest.raises(TypeError):
        TP3Config(brrr_layers=frozenset({0}), brrr_ckpt="/c", brrr_fast_route=1)


def test_sidecar_records_every_switch():
    import json

    off = TP3Config.from_environment(BRRR_ENV).to_metadata()
    assert (off["brrr_split_tail"], off["brrr_fast_route"], off["brrr_counter_parity"]) == (0, False, False)
    on = TP3Config.from_environment({**BRRR_ENV, "FLARE_DS41_BRRR_SPLIT_TAIL": "24", "FLARE_DS41_BRRR_FAST_ROUTE": "1",
                                     "FLARE_DS41_BRRR_COUNTER_PARITY": "1"})
    record = json.loads(json.dumps(on.to_metadata()))
    assert (record["brrr_split_tail"], record["brrr_fast_route"], record["brrr_counter_parity"]) == (24, True, True)
    described = on.describe()
    assert "brrr_split_tail=24" in described and "brrr_fast_route=True" in described and "brrr_counter_parity=True" in described
    assert "brrr_split_tail" not in TP3Config.from_environment(BRRR_ENV).describe()


def _fake_ext(**switches):
    values = dict(SPLIT_TAIL=0, FAST_ROUTE=0, COUNTER_PARITY=0, SPIN_TRAP=0)
    values.update(switches)
    return SimpleNamespace(**values)


def test_build_switch_contract_check():
    config = TP3Config(brrr_layers=frozenset({0}), brrr_ckpt="/c", brrr_split_tail=8, brrr_counter_parity=True)
    brrr.check_switches(_fake_ext(SPLIT_TAIL=1, COUNTER_PARITY=1), config)
    with pytest.raises(RuntimeError, match="do not match"):
        brrr.check_switches(_fake_ext(SPLIT_TAIL=1), config)  # a build without parity for a parity config
    with pytest.raises(RuntimeError, match="do not match"):
        brrr.check_switches(SimpleNamespace(), config)  # a build without the attributes (older source)
    with pytest.raises(RuntimeError, match="do not match"):
        brrr.check_switches(_fake_ext(SPLIT_TAIL=1, COUNTER_PARITY=1, SPIN_TRAP=1), config)  # a test build


def test_extension_variants_get_their_own_module_name_and_defines(monkeypatch):
    from brrr import cuext

    calls = []
    monkeypatch.setattr(cuext, "load", lambda **kwargs: calls.append(kwargs) or kwargs["name"])
    cuext.extension.cache_clear()
    try:
        plain = cuext.extension("decoder", ())
        variant = cuext.extension("decoder", ("BRRR_SPLIT_TAIL=2", "BRRR_FAST_ROUTE=1"))
        other = cuext.extension("decoder", ("BRRR_SPLIT_TAIL=4",))  # (cuext takes any define value)
        assert plain.startswith("decoder_") and len(plain) == len("decoder_") + 10  # the plain build's name scheme
        assert variant.startswith(plain + "_") and other.startswith(plain + "_") and variant != other
        assert not any(flag.startswith("-D") for flag in calls[0]["extra_cuda_cflags"])
        assert calls[1]["extra_cuda_cflags"][-2:] == ["-DBRRR_SPLIT_TAIL=2", "-DBRRR_FAST_ROUTE=1"]
        cpp = next((name for name, path in cuext.SOURCES.items() if path.suffix == ".cpp"), None)
        if cpp is not None:
            with pytest.raises(ValueError, match="only supported for CUDA"):
                cuext.extension(cpp, ("X=1",))
    finally:
        cuext.extension.cache_clear()


def test_custom_op_dispatches_to_the_engine_of_its_handle(monkeypatch):
    seen = []
    runner = SimpleNamespace(run=lambda *args: seen.append(args))
    monkeypatch.setattr(brrr, "_RUNNERS", [(SimpleNamespace(run=None), 0), (runner, 5)])
    t = torch.zeros(1)
    brrr.brrr_layer_tail(1, 3, t, t.long(), t, t, t.clone(), t.clone(), t.clone(), t.clone(), t.clone(), t.clone(),
                         t.clone(), t.clone())
    assert len(seen) == 1 and seen[0][0] == 5 and seen[0][1] == 3 and seen[0][4] is None  # local handle, slot, no ids


def test_export_zero_ranges_only_with_counter_parity(monkeypatch):
    from ignition.recipes.deepseek41 import bootstrap

    engine = object.__new__(brrr.Engine)
    engine.ext = SimpleNamespace(counters_range=lambda handle: [0x7000_0000 + handle, 2 * 7662 * 4 + 8])
    engine.local_handle = 2
    engine.tp3 = TP3Config(brrr_layers=frozenset({0}), brrr_ckpt="/c")
    assert engine.zero_ranges() == ()
    engine.tp3 = TP3Config(brrr_layers=frozenset({0}), brrr_ckpt="/c", brrr_counter_parity=True)
    assert engine.zero_ranges() == ((0x7000_0002, 2 * 7662 * 4 + 8),)
    monkeypatch.setattr(brrr, "_ENGINE", [engine])
    assert bootstrap._brrr_zero_ranges(engine.tp3) == ((0x7000_0002, 2 * 7662 * 4 + 8),)
    assert bootstrap._brrr_zero_ranges(TP3Config()) == ()  # no BRRR layers: nothing asked of the engine
    monkeypatch.setattr(brrr, "_ENGINE", [])
    with pytest.raises(RuntimeError, match="no BRRR engine"):  # a parity config must not silently skip the zeroing
        bootstrap._brrr_zero_ranges(engine.tp3)
    assert bootstrap._brrr_zero_ranges(TP3Config(brrr_layers=frozenset({0}), brrr_ckpt="/c")) == ()


# --- CPU mirror of decoder.cu's chunk-section layout (lagged, decode, tail_of, tail_slot, split_items) ---
CHUNKS, OTOK, GRID = 40, 16, 376


def _lagged(r, per, grid):
    lag = min(CHUNKS, max(1, -(-grid // per)))
    if r < lag * per:
        return r // per, r % per
    r -= lag * per
    if r < (CHUNKS - lag) * (per + 1):
        k, i = lag + r // (per + 1), r % (per + 1)
        return (k, i) if i < per else (k - lag, per)
    return CHUNKS - lag + r - (CHUNKS - lag) * (per + 1), per


def _tail_of(tg, n, split, grid):
    per = 2 * tg + n
    lag = min(CHUNKS, max(1, -(-grid // per)))
    split = min(split, n * CHUNKS)
    if n <= 0 or split <= 0:
        return dict(per=per, lag=lag, kc=CHUNKS, i0=0, w0=CHUNKS * (per + 1))
    kc = CHUNKS - 1 - (split - 1) // n
    i0 = per - 1 - (split - 1) % n
    w0 = kc * per + i0 if kc < lag else lag * per + (kc - lag) * (per + 1) + i0
    return dict(per=per, lag=lag, kc=kc, i0=i0, w0=w0)


def _split_items(t, k):
    return t["per"] if k > t["kc"] else t["per"] - t["i0"] if k == t["kc"] else 0


def _split_jobs(t):
    return 0 if t["kc"] >= CHUNKS else (t["per"] - t["i0"]) + (CHUNKS - 1 - t["kc"]) * t["per"]


def _tail_slot(t, r):
    per, lag, kc, i0 = t["per"], t["lag"], t["kc"], t["i0"]
    r -= t["w0"]
    first = 2 * (per - i0)
    if r < first:
        return kc, i0 + r // 2, 1 + r % 2
    r -= first
    if kc >= lag:
        if r == 0:
            return kc - lag, per, 0
        r -= 1
    k0 = kc + 1
    plain = max(0, min(lag, CHUNKS) - k0)
    if r < plain * 2 * per:
        return k0 + r // (2 * per), r % (2 * per) // 2, 1 + r % 2
    r -= plain * 2 * per
    kb = max(k0, lag)
    blocks = CHUNKS - kb
    if r < blocks * (2 * per + 1):
        k, off = kb + r // (2 * per + 1), r % (2 * per + 1)
        return (k - lag, per, 0) if off == 2 * per else (k, off // 2, 1 + off % 2)
    r -= blocks * (2 * per + 1)
    return (CHUNKS - lag + r, per, 0) if r < lag else (CHUNKS, 0, 0)


def _chunk_jobs(tg, n, split, grid):
    """(chunk, item, half) per chunk-section job in claim order; item == per is XF(chunk)."""
    t = _tail_of(tg, n, split, grid)
    total = CHUNKS * (t["per"] + 1) + (_split_jobs(t) if split else 0)
    out = []
    for r in range(total):
        if split and r >= t["w0"]:
            out.append(_tail_slot(t, r))
        else:
            k, i = _lagged(r, t["per"], grid)
            out.append((k, i, 0))
    return t, out


CASES = [(tg, n, grid) for tg in (1, 2, 3, 8) for n in (1, 2, 6, 9, 13, 21, 33, 64, 200) for grid in (GRID, 40, 1000)]


def check_split_layout(tg, n, split, grid, jobs, items):
    """Properties of a split-tail chunk-section layout (rows of (chunk, item, half), item == per is XF) against the
    unsplit layout; shared with the GPU test, which feeds it the kernel's own split_tail_layout()."""
    per = 2 * tg + n
    t = _tail_of(tg, n, split, grid)
    _, base = _chunk_jobs(tg, n, 0, grid)
    base_r2 = [(k, i) for k, i, _ in base if 2 * tg <= i < per]
    assert (CHUNKS, 0, 0) not in jobs, "a job index inside the total decoded past the end"
    assert jobs[:t["w0"]] == base[:t["w0"]], "the layout before the window must not move"
    want_r2 = set(base_r2[len(base_r2) - min(split, n * CHUNKS):])
    first = base.index((t["kc"], t["i0"], 0))
    want = {(k, i) for k, i, _ in base[first:] if i < per}
    assert {(k, i) for k, i in want if i >= 2 * tg} == want_r2  # the SH2 items in `want` are the ones behind kc/i0
    seen, arrivals, done = {}, [0] * CHUNKS, [False] * CHUNKS
    for pos, (k, i, h) in enumerate(jobs):
        if i == per:  # XF(k): after every item (both halves) of chunk k, exactly once
            assert not done[k]
            assert arrivals[k] == n + 2 * tg + items[k], (split, k, arrivals[k])
            done[k] = True
            continue
        assert not done[k], "chunk item after its XF"
        seen.setdefault((k, i), []).append((h, pos))
        arrivals[k] += 1
    assert all(done)
    for item, parts in seen.items():
        if item in want:
            assert [h for h, _ in parts] == [1, 2] and parts[1][1] == parts[0][1] + 1, (item, parts)
        else:
            assert [h for h, _ in parts] == [0], (item, parts)
    assert len(seen) == CHUNKS * per
    assert len(jobs) == CHUNKS * (per + 1) + len(want) == CHUNKS * (per + 1) + _split_jobs(t)
    assert sum(items) == _split_jobs(t) == len(want)


def split_values(n):
    return sorted({1, 2, 3, n - 1, n, n + 1, 2 * n + 1, 7 * n, 17 * n + 3, 39 * n, 40 * n, 10 ** 6} - {0})


@pytest.mark.parametrize("tg,n,grid", CASES)
def test_split_tail_layout_mirror(tg, n, grid):
    for split in split_values(n):
        t, jobs = _chunk_jobs(tg, n, split, grid)
        check_split_layout(tg, n, split, grid, jobs, [_split_items(t, k) for k in range(CHUNKS)])


def test_split_tail_is_off_without_entries():
    t = _tail_of(1, 0, 5, GRID)
    assert t["w0"] == CHUNKS * (t["per"] + 1) and _split_jobs(t) == 0
    assert math.isclose(sum(_split_items(t, k) for k in range(CHUNKS)), 0)

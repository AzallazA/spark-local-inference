# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 BRRR fused layer tail vs. Flare's stock layer path, layer-level, on three GPUs.

Run (three free GPUs, truncated real-weight checkpoint):

    cd ~/flare-up-v3 && CUDA_VISIBLE_DEVICES=0,1,2 FLARE_RUN_CUDA_TESTS=1 \\
        FLARE_DS41_SNAPSHOT=/m2-2/flare-ckpt/ds41-mini8 .venv/bin/pytest -x -s \\
        tests/recipes/test_deepseek41_brrr_tail_cuda.py

Optional: FLARE_DS41_BRRR_TEST_LAYER (default 2, resident), FLARE_DS41_BRRR_TEST_PARKED_LAYER (default 3, merged
hot/cold bank), FLARE_DS41_BRRR_TEST_ROWS="1,6,8,12,16,17,24,48", FLARE_DS41_BRRR_TEST_HOT (default 64 hot experts
of 384 in the parked case). FLARE_DS41_BRRR_* variables already in the environment (CSF_CHECK, LOG, ...) pass through.

For one layer each rank builds, on identical realistic inputs, (a) Flare's stock path from the attention output to
the MLP all-reduce (inverse RoPE, grouped o-projection, all-reduce, FFN mHC post/pre, router + hot_map remap,
routed experts, shared expert, all-reduce) and (b) the BRRR tail (one flare::brrr_layer_tail launch), then compares
s.mlp_output, s.ffn_hc.residual/post/comb and s.pre_mix (cosine > 0.999, stats printed). Rows sweep 1..48 across the
kernel's WOA_SMALL=16 / XNORM_MIN=16 / TWO_SHOT_MIN=17 regimes; the "uniform" profile routes every row to the same
six experts so one expert takes > 16 routes (two or three 16-route chunks). Each configuration runs in its own
3-process world because FLARE_DS41_FUSED_ATTN_PER_CHUNK is an engine-creation setting of the kernel. Each rank
builds one TP3Config from its environment (TP3Config.from_environment) and passes it to the loader, the engine and
the tails. The rope negative control (FLARE_DS41_BRRR_BREAK=rope) must fail the comparison.

The stock path's two all-reduces are replaced by an exact FP32 Gloo all-reduce (rounded to BF16) so the reference is
independent of b12x's PCIe runtime.

Routing: both paths compute the router from their own bf16 FFN input (two mHC/RMSnorm implementations), so the fp32 gate
logits differ by ~1e-2 and the top-6 selection flips whenever the 6th/7th key margin is of that order (frequent on random
inputs: ~1 row in 4). A flipped row is a different function (20-35% off), so the kernel's routing is observed per row
(each row is re-launched alone through a 1-row probe tail with add_layer(route_counts=...) counters): every dropped /
added expert pair must be a near-tie (key margin < TIE_MARGIN), then the reference is re-routed with the kernel's
choice and its experts / MLP all-reduce rerun.
mlp_output then differs only by the FP8 (e4m3, 32-group) activation quantization each implementation applies to the
expert inputs and intermediates: two independent quantizers give ~5% relative norm (cosine ~0.9985), so mlp_output uses
MLP_TOLERANCE while the mHC outputs keep TOLERANCE. A dropped route, missing rank partial or wrong expert fails both.

Primary check (brrr_tail_reference.py, after NVIDIA's persistent-decoder qualification): every observable kernel output
is compared elementwise with an FP64 recomputation of its stage from that stage's own inputs, within a per-operator
bound of gamma(n) FP32 accumulation error (n = reduction length) plus half an ulp of each declared BF16/E4M3 seam
(the kernel's activation quantizers are emulated in FP64; elements a 32-sigma FP32 error model could move across a
rounding boundary get one grid step of slack, see brrr_tail_reference.py). ffn_hc.residual is checked from the shared
inputs (inverse RoPE, WO-A, WO-B, attention all-reduce, residual mix); pre_mix/post/comb from the kernel's residual
(rtol 2e-5, atol 4e-5); the kernel's routing must equal the FP64 top-6 of its own FFN input except for swaps inside
the propagated key error bound (such rows are counted and excluded from the mlp_output bound); mlp_output from the
FP64 routed/shared experts (checkpoint MXFP4 and MXFP8 bytes decoded in FP64) with the kernel's routing and the MLP
all-reduce. The cosine statistics above remain as the secondary report. After the width sweep one held-out random
input runs through a freshly constructed PreparedBRRRTail (new scratch, positions and object) under the same bounds.
The rope negative control fails on the ffn_residual bound.

Item-6 kernel switches (TP3Config brrr_split_tail / brrr_fast_route / brrr_counter_parity, compile-time variants of the
kernel, each its own extension build; the split count N is an engine setting of the one split build): the *-split2 /
-splitall / -fastroute / -parity cases run each switch alone through the same sweep, bounds and held-out input,
resident-item6 / parked-item6 all three together, and resident-item6-break-rope is the rope negative control on the
all-switches build. These cases use spin-trap builds (TP3Config.brrr_spin_trap: a wait that never ends traps instead
of hanging), and every case is bounded by FLARE_DS41_BRRR_TEST_TIMEOUT (default 3600 s; the ranks are killed). With
the split tail the FP64 bound counts the extra half-sum roundings (TailReference down_halves), and rank 0 checks the
kernel's own split layout (split_tail_layout) against the CPU mirror in test_deepseek41_brrr_item6.py.
resident-/parked-item6-equivalence compare fast_route and counter_parity with the plain build on the same inputs:
every route's (bank slot, FP32 weight bits) exactly (route_state), against the plain build's launch-to-launch noise,
plus the outputs bitwise; split_tail's output difference is reported only (_equivalence). The full file is 14 cases and 5
extension builds, about 60-70 minutes.
Opt-in A/B timing: FLARE_DS41_BRRR_TEST_AB="split_tail:96,fast_route,counter_parity" (variants joined by "+" combine)
times each variant against the build with all three off, alternating, at FLARE_DS41_BRRR_TEST_AB_ROWS (default 1,6,12;
must be in FLARE_DS41_BRRR_TEST_ROWS) x FLARE_DS41_BRRR_TEST_PROFILE_MIX (default 0.6,0.7), see _ab_profile; JSON
lines go to FLARE_DS41_BRRR_TEST_AB_OUT. Typically with -k resident-perchunk1.
"""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import brrr_tail_reference as reference64  # sibling module; pytest puts tests/recipes on sys.path
except ImportError:  # pragma: no cover - imported from another working directory
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import brrr_tail_reference as reference64

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires three explicitly assigned CUDA devices")

WORLD = 3
TOLERANCE = 0.999
MLP_TOLERANCE = 0.995  # mlp_output: two independent e4m3 activation quantizations (~5% relative norm, cosine ~0.9985)
MLP_ROW_TOLERANCE = 0.99
TIE_MARGIN = 0.05  # kernel/Flare top-k disagreements must be near-ties in key = sqrt(softplus(logit)) + bias
DEFAULT_ROWS = (1, 6, 8, 12, 16, 17, 24, 48)
# (profile, residual scale): rows independent / every row identical (one expert gets m routes) / small residual so
# the attention branch dominates the FFN input (sensitive to o-projection and RoPE errors).
PROFILES = (("random", 1.0), ("uniform", 1.0), ("branch", 0.05))
COMPARED = ("mlp_output", "ffn_residual", "ffn_post", "ffn_comb", "pre_mix")
# Opt-in timing pass (not a correctness check): FLARE_DS41_BRRR_TEST_PROFILE=<iterations> times every width after the
# sweep with L2 flushed before each launch; with DS41_FUSED_PROF=1 it also prints the kernel's per-job accumulators and
# phase marks. FLARE_DS41_BRRR_TEST_PROFILE_MIX lists row correlations (0 = independent random rows, ~36 distinct
# experts at m=6; higher values pull every row toward row 0 so fewer distinct experts are routed, like real decode).
JOB_NAMES = ("WOA", "AMIX", "WOB", "XA", "OWNA", "XNORM", "GATE", "ROUTE", "SH1", "SH2", "MIX", "R1", "R2", "XF")


def _profile_widths(*, rank, group, device, ext, layer, rows_sweep, paths, positions, route_counts, iters):
    """Median per-launch time (min over ranks: the last rank to arrive does not wait on the others) and, with
    DS41_FUSED_PROF=1, the job-phase timeline of that launch relative to the first CTA entry (globaltimer, ~1 us
    resolution on some drivers)."""
    import statistics

    mixes = tuple(float(x) for x in os.environ.get("FLARE_DS41_BRRR_TEST_PROFILE_MIX", "0,0.7").split(",") if x.strip())
    prof = os.environ.get("DS41_FUSED_PROF", "0") not in ("", "0")
    # Zeroed once and flushed by reading: a per-launch zero_() left L2 full of dirty lines whose write-backs inflated
    # the kernel's first reads (WOA 18.4 vs 7.0 us per job unflushed).
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=device)  # > 128 MiB L2
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    seed = 50_000 + 1000 * layer
    for m in rows_sweep:
        _, brrr_rows, _, tail = paths[m]
        for mix in mixes:
            seed += 1
            p0 = _fill_inputs(brrr_rows, m, seed=seed, rank=rank, profile="random", resid_scale=1.0, device=device)
            for t in (brrr_rows.attention.attended, brrr_rows.attention_hc.residual):
                t[:m].copy_(((1 - mix) * t[:m].float() + mix * t[:1].float()).to(t.dtype))
            positions[:m].copy_(p0)
            times, phases, accs, distinct = [], [], [], []
            for _ in range(iters):
                if route_counts is not None:
                    route_counts.zero_()
                flush.sum(dtype=torch.int64)
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
                # Keep the GPU busy while the host enqueues the eager custom op (~30 us of argument checks and launch),
                # so the events time the kernel, as a CUDA graph replay does in production.
                torch.cuda._sleep(200_000)
                start.record()
                tail(brrr_rows.attention.attended)
                stop.record()
                torch.cuda.synchronize(device)
                times.append(start.elapsed_time(stop) * 1e3)
                if route_counts is not None:
                    distinct.append(int((route_counts > 0).sum()))
                if prof:
                    pr = ext.prof_read()
                    accs.append(pr[:16].double())
                    phases.append(pr[16:].double())
            mine = statistics.median(times)
            gathered = [None] * WORLD
            dist.all_gather_object(gathered, mine, group=group)
            best = min(range(WORLD), key=lambda r: gathered[r])
            if rank == best:
                line = (f"[brrr-profile] layer {layer} m={m} mix={mix}: {mine:.1f} us/launch (median of {iters}, L2 flushed;"
                        f" ranks {[round(x, 1) for x in gathered]}), distinct experts "
                        f"{statistics.median(distinct) if distinct else 'n/a'}")
                if prof:
                    ph = torch.stack(phases).median(0).values  # per-launch absolute marks, median over launches
                    acc = torch.stack(accs).median(0).values
                    rel = torch.stack([p - p[15, 0] for p in phases]).median(0).values / 1e3  # us since first CTA entry
                    line += "\n  job    jobs  1st-acq  1st-run  last-run  last-end | sum stage-wait  dep-wait     run (us over CTAs)"
                    for t, name in enumerate(JOB_NAMES):
                        if acc[t, 3] == 0:
                            continue
                        line += (f"\n  {name:6s} {int(acc[t, 3]):5d} {rel[t, 0]:8.1f} {rel[t, 1]:8.1f} {rel[t, 3]:9.1f} {rel[t, 2]:9.1f} |"
                                 f" {acc[t, 0] / 1e3:14.1f} {acc[t, 1] / 1e3:9.1f} {acc[t, 2] / 1e3:7.1f}")
                    line += (f"\n  kernel: last CTA entry {rel[15, 1]:.1f}, first/last consumer exit {rel[15, 2]:.1f}/{rel[15, 3]:.1f},"
                             f" C_FIN first/last {rel[14, 0]:.1f}/{rel[14, 1]:.1f}, counter reset done {rel[14, 2]:.1f} us"
                             f" (marks are absolute; {int(ph[15, 0] > 0)} = recorded)")
                print(line, flush=True)
            dist.barrier(group=group)


# --- item-6 kernel variants (TP3Config brrr_split_tail / brrr_fast_route / brrr_counter_parity) ---
ITEM6_OFF = dict(brrr_split_tail=0, brrr_fast_route=False, brrr_counter_parity=False)


def _observe_routes(ext, route_counts):
    """Make every add_layer of this extension build pass route_counts (idempotent per module). False = unsupported."""
    if getattr(ext, "_route_counts_patched", False):
        return True
    add_layer = ext.add_layer
    if "route_counts" not in (add_layer.__doc__ or ""):
        return False
    ext.add_layer = lambda handle, **kwargs: add_layer(handle, route_counts=route_counts, **kwargs)
    ext._route_counts_patched = True
    return True


def _parse_switches(spec):
    """"split_tail:96+fast_route" -> TP3Config field overrides (one A/B variant)."""
    switches = {}
    for part in filter(None, (p.strip() for p in spec.split("+"))):
        name, _, value = part.partition(":")
        if name == "split_tail":
            switches["brrr_split_tail"] = int(value or 0)
        elif name in ("fast_route", "counter_parity") and not value:
            switches["brrr_" + name] = True
        else:
            raise ValueError(f"unknown BRRR A/B switch {part!r} (split_tail:N, fast_route, counter_parity; join with +)")
    if not any(switches.values()):
        raise ValueError(f"A/B variant {spec!r} switches nothing on")
    return switches


def _variant_tails(tp3, *, group, device, storage, layer, weights, config, frequencies, paths, positions, route_counts,
                   base_tp3, base_tails):
    """(engine, PreparedBRRRTails per width on the sweep's scratch rows) for a kernel variant; the base engine and its
    tails when the variant is what the base config already runs (same build and split count). Collective (engine
    creation): call in the same order on every rank."""
    from ignition.recipes.deepseek41.brrr import Engine, PreparedBRRRTail, engine, engine_key

    if engine_key(tp3) == engine_key(base_tp3):
        return engine(), base_tails
    variant = Engine(group, device, storage.peer_storage, tp3)
    if route_counts is not None:
        _observe_routes(variant.ext, route_counts)
    return variant, {m: PreparedBRRRTail(layer, weights, paths[m][1], positions[:m], frequencies, config=config, tp3=tp3,
                                         engine_=variant) for m in paths}


def _timed(tail, rows, *, flush, device, group, start, stop, route_counts):
    if route_counts is not None:
        route_counts.zero_()
    flush.sum(dtype=torch.int64)
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    torch.cuda._sleep(200_000)  # hide the eager op's host launch time (see _profile_widths)
    start.record()
    tail(rows.attention.attended)
    stop.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(stop) * 1e3


def _bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _route_state(engine, m):
    """(bank slot, weight bits) of the last launch's routes 0 .. m * TOPK - 1 (routes cleared before the launch)."""
    weights, routes = engine.ext.route_state(engine.local_handle)
    slots = torch.full((weights.numel(),), -1, dtype=torch.int32)
    valid = routes >= 0
    slots[routes[valid].long()] = valid.nonzero()[:, 0].to(torch.int32)
    n = m * engine.ext.TOPK
    return torch.stack((slots[:n], weights[:n].view(torch.int32)))


def _equivalence(*, rank, rows_sweep, group, device, storage, layer, weights, config, frequencies, paths, positions,
                 route_counts, tp3):
    """Bit-exact check (FLARE_DS41_BRRR_TEST_EQUIV=1): fast_route and counter_parity must route exactly like the plain build.

    Same inputs through: plain three times (the noise floor: the FP32 red.add order of WOA/WOB and the expert sums is
    timing-dependent, so two plain launches can differ in a few BF16 elements and, through the gate input, in the route
    weights), fast_route, counter_parity, and split_tail over every down job (reported only: it changes the summation
    of whole down projections, but measured 2026-11-09 its 4-22K differing output elements sit inside the plain build's
    own run-to-run floor of 10-24K, so it cannot serve as a negative control; fast_route and counter_parity passed within
    the floor). Primary check, per launch: every route's (bank slot, FP32
    weight bits) read back from the kernel (route_state) must equal the plain launch's; launches where the plain repeats
    already disagree count as noise, and a variant may not mismatch more often than 2 x the noise. Secondary: the five
    outputs bitwise. counter_parity keeps the job layout, so its output difference must stay within 2 x floor + 8 per
    launch; fast_route reserves one ROUTE job's entries in lane order (a different FP32 red order downstream), so its
    output difference is reported only. Returns failures (every rank decides on the same gathered data)."""
    from dataclasses import replace

    base = replace(tp3, **ITEM6_OFF)
    shared = dict(group=group, device=device, storage=storage, layer=layer, weights=weights, config=config,
                  frequencies=frequencies, paths=paths, positions=positions, route_counts=route_counts, base_tp3=tp3,
                  base_tails={m: paths[m][3] for m in paths})
    names = ("plain", "plain2", "plain3", "fast_route", "counter_parity", "split_tail")
    configs = dict(plain=base, plain2=base, plain3=base, fast_route=replace(base, brrr_fast_route=True),
                   counter_parity=replace(base, brrr_counter_parity=True), split_tail=replace(base, brrr_split_tail=1 << 20))
    from ignition.recipes.deepseek41.brrr import engine_key

    built, runs = {}, {}
    for name in names:  # one engine per distinct (build, split count), created in the same order on every rank
        key = engine_key(configs[name])
        if key not in built:
            built[key] = _variant_tails(configs[name], **shared)
        runs[name] = built[key]
    diff = {name: 0 for name in names}
    state_bad = {name: 0 for name in names}
    launches = 0
    seed = 77_000 + 1000 * layer
    for m in rows_sweep:
        brrr_rows = paths[m][1]
        for profile in ("random", "uniform"):
            seed += 1
            p0 = _fill_inputs(brrr_rows, m, seed=seed, rank=rank, profile=profile, resid_scale=1.0, device=device)
            positions[:m].copy_(p0)
            got = {}
            for name in names:
                engine, tails = runs[name]
                engine.ext.clear_routes(engine.local_handle)
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
                tails[m](brrr_rows.attention.attended)
                torch.cuda.synchronize(device)
                got[name] = ({k: _bits(v).clone() for k, v in _outputs(brrr_rows, m).items()}, _route_state(engine, m))
            launches += 1
            ref, ref_state = got["plain"]
            for name in names[1:]:
                out, state = got[name]
                diff[name] += sum(int((out[k] != ref[k]).sum()) for k in COMPARED)
                state_bad[name] += int(not torch.equal(state, ref_state))
    floor = max(diff["plain2"], diff["plain3"])
    state_floor = max(state_bad["plain2"], state_bad["plain3"])
    report = dict(launches=launches, output_floor=floor, route_state_floor=state_floor,
                  **{f"output_diff_{k}": v for k, v in diff.items() if k != "plain"},
                  **{f"route_state_mismatch_{k}": v for k, v in state_bad.items() if k not in ("plain", "split_tail")})
    failures = []
    for name in ("fast_route", "counter_parity"):
        if state_bad[name] > 2 * state_floor:
            failures.append((rank, "equivalence", name, f"route (slot, weight) state differs from the plain build in "
                             f"{state_bad[name]} of {launches} launches (plain repeats: {state_floor})"))
    allowed = 2 * floor + 8 * launches
    if diff["counter_parity"] > allowed:
        failures.append((rank, "equivalence", "counter_parity", f"{diff['counter_parity']} output elements differ (allowed {allowed})"))
    gathered = [None] * WORLD
    dist.all_gather_object(gathered, (report, failures), group=group)
    if rank == 0:
        for r, (rep_, fail) in enumerate(gathered):
            print(f"[brrr-equiv] layer {layer} rank {r}: {rep_}" + (f" | FAIL {fail}" if fail else " | ok"), flush=True)
    return failures


def _split_layout_check(ext, ctas):
    """The split build's own layout (split_tail_layout, the kernel's tail_of / tail_slot / split_items compiled for the
    host) must satisfy the CPU mirror's properties and equal the mirror row for row. Returns failures."""
    try:
        import test_deepseek41_brrr_item6 as mirror
    except ImportError:  # pragma: no cover
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_deepseek41_brrr_item6 as mirror
    failures, checked = [], 0
    for grid in (ctas, 40, 1000):
        for tg in (1, 2, 3, 8):
            for n in (1, 6, 13, 21, 33, 200):
                for split in mirror.split_values(n):
                    rows, items = ext.split_tail_layout(tg, n, split, grid)
                    jobs = [tuple(r) for r in rows.tolist()]
                    t, want = mirror._chunk_jobs(tg, n, split, grid)
                    try:
                        assert jobs == want, "kernel layout differs from the CPU mirror"
                        assert items.tolist() == [mirror._split_items(t, k) for k in range(40)], "split_items differ"
                        mirror.check_split_layout(tg, n, split, grid, jobs, items.tolist())
                    except AssertionError as exc:
                        failures.append(("split-layout", grid, tg, n, split, str(exc)[:200]))
                    checked += 1
    print(f"[brrr-split-layout] {checked} (grid, tg, n, N) layouts checked, {len(failures)} failures", flush=True)
    return failures


def _ab_profile(*, rank, spec, rows_sweep, group, device, storage, layer, weights, config, frequencies, paths, positions,
                route_counts, tp3):
    """Alternating off/on timing of item-6 variants (FLARE_DS41_BRRR_TEST_AB, e.g. "split_tail:96,fast_route,counter_parity";
    join switches with + for a combined variant). "off" is this config with the three switches cleared. Per variant,
    width (FLARE_DS41_BRRR_TEST_AB_ROWS, default 1,6,12; must be in the sweep) and mix (FLARE_DS41_BRRR_TEST_PROFILE_MIX,
    default 0.6,0.7): FLARE_DS41_BRRR_TEST_AB_REPEATS (default 6, >= 5) repeats in ABBA order, each the median of
    FLARE_DS41_BRRR_TEST_AB_ITERS (default 20) launches with L2 flushed (read) and the host launch hidden; the reported
    time per variant is the median over repeats on the rank with the lowest median (min over ranks, as _profile_widths).
    Rank 0 prints one [brrr-ab] line per cell and appends JSON lines to FLARE_DS41_BRRR_TEST_AB_OUT if set."""
    import json
    import statistics
    from dataclasses import replace

    variants = [(item, _parse_switches(item)) for item in filter(None, (p.strip() for p in spec.split(",")))]
    rows = tuple(int(x) for x in os.environ.get("FLARE_DS41_BRRR_TEST_AB_ROWS", "1,6,12").split(",") if x.strip())
    missing = [m for m in rows if m not in paths]
    if missing:
        raise ValueError(f"A/B widths {missing} are not in FLARE_DS41_BRRR_TEST_ROWS {rows_sweep}")
    mixes = tuple(float(x) for x in os.environ.get("FLARE_DS41_BRRR_TEST_PROFILE_MIX", "0.6,0.7").split(",") if x.strip())
    repeats = max(5, int(os.environ.get("FLARE_DS41_BRRR_TEST_AB_REPEATS", "6")))
    iters = int(os.environ.get("FLARE_DS41_BRRR_TEST_AB_ITERS", "20"))
    out_path = os.environ.get("FLARE_DS41_BRRR_TEST_AB_OUT")
    base_tails = {m: paths[m][3] for m in paths}
    shared = dict(group=group, device=device, storage=storage, layer=layer, weights=weights, config=config,
                  frequencies=frequencies, paths=paths, positions=positions, route_counts=route_counts, base_tp3=tp3,
                  base_tails=base_tails)
    off_tp3 = replace(tp3, **ITEM6_OFF)
    _, off_tails = _variant_tails(off_tp3, **shared)
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=device)
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timing = dict(flush=flush, device=device, group=group, start=start, stop=stop, route_counts=route_counts)
    for item, switches in variants:
        _, on_tails = _variant_tails(replace(off_tp3, **switches), **shared)
        seed = 90_000 + 1000 * layer
        for m in rows:
            brrr_rows = paths[m][1]
            for mix in mixes:
                seed += 1
                p0 = _fill_inputs(brrr_rows, m, seed=seed, rank=rank, profile="random", resid_scale=1.0, device=device)
                for t in (brrr_rows.attention.attended, brrr_rows.attention_hc.residual):
                    t[:m].copy_(((1 - mix) * t[:m].float() + mix * t[:1].float()).to(t.dtype))
                positions[:m].copy_(p0)
                reps = {"off": [], "on": []}
                distinct = []
                for rep in range(repeats):
                    for name in (("off", "on") if rep % 2 == 0 else ("on", "off")):
                        tail = (off_tails if name == "off" else on_tails)[m]
                        times = [_timed(tail, brrr_rows, **timing) for _ in range(iters)]
                        reps[name].append(statistics.median(times))
                        if route_counts is not None and name == "off":
                            distinct.append(int((route_counts > 0).sum()))
                mine = {name: statistics.median(v) for name, v in reps.items()}
                wins = sum(on < off for on, off in zip(reps["on"], reps["off"]))
                gathered = [None] * WORLD
                dist.all_gather_object(gathered, (mine, wins, reps), group=group)
                if rank == 0:
                    off = min(g[0]["off"] for g in gathered)
                    on = min(g[0]["on"] for g in gathered)
                    record = dict(variant=item, layer=layer, m=m, mix=mix, off_us=round(off, 2), on_us=round(on, 2),
                                  delta_us=round(on - off, 2), repeats=repeats, iters=iters,
                                  on_faster_repeats=[g[1] for g in gathered],
                                  rank_delta_us=[round(g[0]["on"] - g[0]["off"], 2) for g in gathered],
                                  distinct_experts=statistics.median(distinct) if distinct else None)
                    print(f"[brrr-ab] {item} layer {layer} m={m} mix={mix}: off {off:.1f} us, on {on:.1f} us, "
                          f"delta {on - off:+.2f} us/layer (median of {repeats} ABBA repeats x {iters} launches, min over "
                          f"ranks; per-rank deltas {record['rank_delta_us']}, on faster in {record['on_faster_repeats']} "
                          f"repeats), distinct experts {record['distinct_experts']}", flush=True)
                    if out_path:
                        with open(out_path, "a") as f:
                            f.write(json.dumps(record) + "\n")
                dist.barrier(group=group)


def _rows():
    spec = os.environ.get("FLARE_DS41_BRRR_TEST_ROWS", "")
    return tuple(int(p) for p in spec.split(",") if p.strip()) or DEFAULT_ROWS


def _stats(got, want):
    a, b = got.detach().double().reshape(got.shape[0], -1), want.detach().double().reshape(want.shape[0], -1)
    diff = a - b
    cosine = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
    row_cos = ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-30))
    return dict(cosine=cosine, row_min_cosine=float(row_cos.min()), max_abs=float(diff.abs().max()),
                max_rel=float(diff.abs().max() / b.abs().max().clamp_min(1e-30)),
                rel_norm=float(diff.norm() / b.norm().clamp_min(1e-30)), finite=bool(torch.isfinite(a).all()))


def _fill_inputs(scratch, m, *, seed, rank, profile, resid_scale, device):
    """Replicated hyperconnection state (same on every rank) and this rank's attention-head shard."""
    shared = torch.Generator(device="cpu").manual_seed(seed)
    local = torch.Generator(device="cpu").manual_seed(seed * 7919 + rank + 1)
    rows = 1 if profile == "uniform" else m

    def expand(t):
        return t.expand(m, *t.shape[1:]).contiguous() if rows == 1 else t

    residual = expand(torch.randn((rows, 4, 5120), generator=shared) * resid_scale)
    post = expand(1 + 0.1 * torch.randn((rows, 4), generator=shared))
    comb = expand(torch.eye(4).expand(rows, 4, 4) + 0.05 * torch.randn((rows, 4, 4), generator=shared))
    attn_mix = expand(0.25 + 0.05 * torch.randn((rows, 4), generator=shared))
    positions = expand(torch.randint(1, 4096, (rows,), generator=shared, dtype=torch.int64))
    attended = expand(torch.randn((rows, scratch.attention.attended.shape[1], 512), generator=local) * 0.3)
    scratch.attention_hc.residual[:m].copy_(residual.to(torch.bfloat16))
    scratch.attention_hc.post[:m].copy_(post)
    scratch.attention_hc.comb[:m].copy_(comb)
    scratch.attn_mix[:m].copy_(attn_mix)
    scratch.attention.attended[:m].copy_(attended.to(torch.bfloat16))
    # Outputs poisoned so an unwritten tensor cannot pass by luck.
    for t in (scratch.mlp_output, scratch.ffn_hc.residual, scratch.ffn_hc.post, scratch.ffn_hc.comb, scratch.pre_mix):
        t[:m].fill_(float("nan"))
    return positions.to(device)


class _Reference:
    """Flare's stock components after attention (decoder.py PreparedLayer.__call__ from attention_reduce on)."""

    def __init__(self, m, weights, config, scratch, rotary, compressed, *, session, workspace, name):
        from ignition.components.attention_output import PreparedGroupedOutputProjection
        from ignition.components.linear import UnquantizedProjection
        from ignition.components.mhc import PreparedMHC
        from ignition.components.mlp import PreparedMLP
        from ignition.components.moe import PreparedRouter, PreparedRoutedExperts
        from ignition.recipes.deepseek41.decoder import _mhc_weights

        s, text = scratch, config.text
        self.m, self.s, self.rotary, self.compressed = m, s, rotary, compressed
        options = dict(rms_eps=text["rms_norm_eps"], norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"],
                       sinkhorn_iters=text["hc_sinkhorn_iters"])
        self.output_projection = PreparedGroupedOutputProjection(s.attention.attended.view(m, -1, 4096), weights.attention["output"],
            s.attention.output, session=session, name=name + ".output", workspace=workspace)
        self.ffn_hc = PreparedMHC((s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
            _mhc_weights(weights, "ffn", "ffn_norm"), s.ffn_hc, session=session, name=name + ".ffn_hc", workspace=workspace,
            pre_mix=s.attn_mix, pre_out=s.pre_mix, **options)
        self.router_projection = UnquantizedProjection(s.ffn_hc.normalized, weights.experts.router, s.router_logits,
            session=session, name=name + ".router")
        self.router = PreparedRouter(s.router_logits, s.router, session=session, name=name + ".topk",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction)
        if weights.experts.cold is not None:
            raise ValueError("BRRR layers need one expert bank (resident or merged hot/cold)")
        self.hot_map = weights.experts.hot_map
        self.hot_rows = getattr(weights.experts, "hot_rows", None)
        self.correction = weights.experts.correction
        self.routed_scaling_factor = float(text["routed_scaling_factor"])
        self.renormalize = bool(text["norm_topk_prob"])
        if text["scoring_func"] != "sqrtsoftplus" or not self.renormalize:
            raise ValueError("re-routing assumes sqrtsoftplus scores with renormalized top-k weights")
        self.hot_ids = torch.zeros(tuple(s.router.ids.shape), dtype=s.router.ids.dtype, device=s.router.ids.device)
        route_ids = s.router.ids if self.hot_map is None else self.hot_ids
        self.experts = PreparedRoutedExperts(s.ffn_hc.normalized, weights.experts.experts, route_ids, s.router.weights,
            s.routed, session=session, name=name + ".experts", workspace=workspace, router_plan=self.router.plan)
        self.shared = PreparedMLP(s.ffn_hc.normalized, weights.experts.shared, s.shared.gate_up, s.shared.intermediate,
            s.shared.output, session=session, name=name + ".shared", workspace=workspace,
            swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto")

    @staticmethod
    def _all_reduce(source, output):
        # Exact FP32 sum over Gloo, rounded once to BF16 (stands in for PreparedChunkedAllReduce).
        value = source.float().cpu()
        dist.all_reduce(value)
        output.copy_(value.to(output.dtype))

    def __call__(self, positions):
        s, m = self.s, self.m
        self.rotary.apply(s.attention.attended, positions, compressed=self.compressed, inverse=True)
        self.output_projection(s.attention.attended.view(m, -1, 4096))
        self._all_reduce(s.attention.output, s.attention_output)
        self.ffn_hc(s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb)
        self.router(self.router_projection(s.ffn_hc.normalized))
        self._remap()
        self.experts()
        self.shared(s.ffn_hc.normalized)
        torch.add(s.routed, s.shared.output, out=s.mlp_partial)
        self._all_reduce(s.mlp_partial, s.mlp_output)

    def _remap(self):
        if self.hot_map is not None:
            torch.index_select(self.hot_map, 0, self.s.router.ids.reshape(-1), out=self.hot_ids.view(-1))

    def reconcile(self, kernel_rows, launch_counts):
        """Adopt the kernel's top-k where it differs from Flare's by a near-tie, then rerun experts + MLP all-reduce.

        kernel_rows[row, expert]: the kernel's route counts for each row launched alone (probe); launch_counts: the
        counts of the batched launch (must equal the probe sum). Per row, every expert the kernel chose instead of
        Flare's must be a near-tie (key margin < TIE_MARGIN, strongest dropped paired with strongest added), so a
        real routing bug (wrong expert at a clear margin, wrong route count) is reported as "routing_error". Runs
        collectives: call on all ranks.
        """
        s, m = self.s, self.m
        ids = s.router.ids[:m].cpu().long()
        logits = s.router_logits[:m].float().cpu()
        score = torch.sqrt(torch.nn.functional.softplus(logits))
        key = score + self.correction.float().cpu()
        weights = s.router.weights[:m].float().cpu()
        want = self.routed_scaling_factor * score.gather(1, ids) / score.gather(1, ids).sum(1, keepdim=True)
        formula_err = float((want - weights).abs().max())
        experts = key.shape[1]
        kernel_rows, launch_counts = kernel_rows.cpu().long(), launch_counts.cpu().long()
        swaps, margins, errors = 0, [], []
        if formula_err > 5e-3:
            errors.append(f"Flare's router weights do not follow scale * score / sum(score): max err {formula_err}")
        if not torch.equal(kernel_rows.sum(0), launch_counts):
            errors.append("per-row probe launches route differently from the batched launch: experts "
                          f"{(kernel_rows.sum(0) != launch_counts).nonzero().flatten().tolist()}")
        for row in range(m):
            kernel_set = kernel_rows[row].nonzero().flatten().tolist()
            if int(kernel_rows[row].sum()) != ids.shape[1] or len(kernel_set) != ids.shape[1]:
                errors.append(f"row {row}: kernel routed {kernel_rows[row].sum()} routes to {len(kernel_set)} experts")
                continue
            flare_set = ids[row].tolist()
            dropped = sorted((e for e in flare_set if e not in kernel_set), key=lambda e: -float(key[row, e]))
            added = sorted((e for e in kernel_set if e not in flare_set), key=lambda e: -float(key[row, e]))
            for old, new in zip(dropped, added):
                margin = float(key[row, old] - key[row, new])
                margins.append(margin)
                if margin >= TIE_MARGIN:
                    errors.append(f"row {row}: kernel routed expert {new} instead of {old} at key margin {margin:.4f}")
                ids[row, flare_set.index(old)] = new
                swaps += 1
        report = dict(swapped_routes=swaps, max_swap_margin=max(margins, default=0.0), weight_formula_err=formula_err)
        if errors:
            report["routing_error"] = "; ".join(errors)
            swaps = 0  # keep Flare's routing; the comparison then reports the disagreement as a failure
        if swaps:
            new_weights = self.routed_scaling_factor * score.gather(1, ids) / score.gather(1, ids).sum(1, keepdim=True)
            s.router.ids[:m].copy_(ids.to(s.router.ids.dtype))
            s.router.weights[:m].copy_(new_weights.to(s.router.weights.dtype))
            self._remap()
        # Always rerun (collective) so every rank takes the same path whatever its own swap count.
        self.experts()
        torch.add(s.routed, s.shared.output, out=s.mlp_partial)
        self._all_reduce(s.mlp_partial, s.mlp_output)
        return report

    def route_report(self):
        ids = self.s.router.ids.cpu()
        counts = torch.bincount(ids.reshape(-1).long(), minlength=384)
        report = dict(max_routes_per_expert=int(counts.max()), distinct_experts=int((counts > 0).sum()),
                      weight_sum_min=float(self.s.router.weights.sum(1).min()),
                      weight_sum_max=float(self.s.router.weights.sum(1).max()))
        if self.hot_map is not None and self.hot_rows is not None:
            report["cold_routes"] = int((self.hot_ids.cpu() >= self.hot_rows).sum())
        return report


def _outputs(scratch, m):
    return dict(mlp_output=scratch.mlp_output[:m], ffn_residual=scratch.ffn_hc.residual[:m], ffn_post=scratch.ffn_hc.post[:m],
                ffn_comb=scratch.ffn_hc.comb[:m], pre_mix=scratch.pre_mix[:m])


def _rank(rank, rendezvous, snapshot, env, layer, expect_failure):
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD, timeout=timedelta(minutes=30))
    group = dist.group.WORLD

    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.deepseek41.attention import Rotary
    from ignition.recipes.deepseek41.brrr import PreparedBRRRTail, create_engine, extension, kernel_defines
    from ignition.recipes.deepseek41.decoder import Scratch
    from ignition.recipes.deepseek41.tp3_config import TP3Config
    from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader, host_expert_layers

    tp3 = TP3Config.from_environment()
    assert layer in tp3.brrr_layers, (layer, os.environ.get("FLARE_DS41_BRRR_LAYERS"))
    parked = layer in host_expert_layers(tp3)
    rows_sweep = _rows()
    capacity = max(rows_sweep)
    failures, reports = [], []
    try:
        with PackageStorage(rank=rank, world=WORLD, device=rank) as storage:
            manifest = inspect_indexed_checkpoint(Path(snapshot))
            with DeepSeekWeightLoader(manifest, device, shard=TensorParallelShard(rank, WORLD), prefixes=(f"layers.{layer}.",),
                                      expert_scale_compression="csf",
                                      host_allocator=storage.host_allocate if parked else None, tp3=tp3) as loader:
                weights, config = loader.load_layer(layer), loader.config
            torch.cuda.synchronize(device)
            if parked:
                assert weights.experts.hot_map is not None and weights.experts.hot_rows is not None, "parked layer did not load a merged bank"
                assert weights.experts.cold is None, "BRRR needs the merged bank (FLARE_DS41_MERGED_BANK=1), not hot+cold banks"
            else:
                assert weights.experts.hot_map is None, "resident layer unexpectedly remapped"
            engine = create_engine(group, device, storage.peer_storage, tp3)
            assert engine is not None
            ext = engine.ext  # the build this config selects (item-6 switches are compile-time variants)
            assert ext is extension(kernel_defines(tp3))
            assert ext.RANKS == WORLD and capacity <= ext.MAXM
            # Observe the kernel's routing: per-global-expert route counters (optional add_layer argument).
            route_counts = torch.zeros((ext.EXPERTS,), dtype=torch.int32, device=device)
            if not _observe_routes(ext, route_counts):
                route_counts = None
            tail_kwargs = {"config": config, "tp3": tp3}
            compressed = bool(config.text["compress_ratios"][layer])
            rotary = Rotary(config, device)
            frequencies = rotary.frequencies[compressed]
            reference_scratch = Scratch.allocate(capacity, capacity, WORLD, device)
            brrr_scratch = Scratch.allocate(capacity, capacity, WORLD, device)
            positions = torch.zeros((capacity,), dtype=torch.int64, device=device)
            arena = ScratchArena(device, overlap=False)
            with RecordedPreparationSession(device=device, checkpoint="ds41-brrr-tail-test", workload_schema="tp3-brrr-tail",
                                            numerical_policy="checkpoint", compile_workers=8) as session:
                session.register_scratch(arena)
                paths = {}
                for m in rows_sweep:
                    ref_rows, brrr_rows = reference_scratch.rows(m, m), brrr_scratch.rows(m, m)
                    reference = _Reference(m, weights, config, ref_rows, rotary, compressed, session=session,
                                           workspace=arena, name=f"layer{layer}.m{m}")
                    tail = PreparedBRRRTail(layer, weights, brrr_rows, positions[:m], frequencies, **tail_kwargs)
                    paths[m] = (ref_rows, brrr_rows, reference, tail)
                # One-row probe: re-launches each row alone to read the kernel's routing for that row.
                probe_rows = Scratch.allocate(1, 1, WORLD, device).rows(1, 1)
                probe = PreparedBRRRTail(layer, weights, probe_rows, positions[:1], frequencies, **tail_kwargs)

                def kernel_routing(snapshot, p0, m):
                    rows = torch.zeros((m, route_counts.numel()), dtype=torch.int32)
                    for r in range(m):
                        for name, dst in (("attended", probe_rows.attention.attended), ("residual", probe_rows.attention_hc.residual),
                                          ("post", probe_rows.attention_hc.post), ("comb", probe_rows.attention_hc.comb),
                                          ("attn_mix", probe_rows.attn_mix)):
                            dst[:1].copy_(snapshot[name][r:r + 1])
                        positions[0] = p0[r]
                        route_counts.zero_()
                        torch.cuda.synchronize(device)
                        probe(probe_rows.attention.attended)
                        torch.cuda.synchronize(device)
                        rows[r] = route_counts.cpu()
                    return rows
                session.tune(show_progress=False)
                # FP64 stage references with per-operator bounds (primary check); weights come from the checkpoint
                # bytes the kernel's planes were packed from, routing/mHC parameters from Flare's loaded tensors.
                ref64 = reference64.TailReference(
                    layer=layer, rank=rank, world=WORLD, group=group, device=device, checkpoint=engine.checkpoint,
                    text=config.text, frequencies=frequencies, oa=ext.OA, inter=ext.INTER,
                    gate=weights.experts.router, gate_bias=weights.experts.correction,
                    fn=weights.residual["hc_ffn_fn"], hc_scale=weights.residual["hc_ffn_scale"],
                    hc_base=weights.residual["hc_ffn_base"], norm=weights.residual["ffn_norm.weight"],
                    down_halves=bool(tp3.brrr_split_tail))

                def kernel_inputs(rows, m):
                    # Taken before the launches: the stock path rotates its own attended in place, the kernel reads o as is.
                    return {name: t[:m].clone() for name, t in (
                        ("attended", rows.attention.attended), ("residual", rows.attention_hc.residual),
                        ("post", rows.attention_hc.post), ("comb", rows.attention_hc.comb), ("attn_mix", rows.attn_mix))}

                def launch(m, profile, resid_scale, seed, ref_rows, brrr_rows, reference, tail, tail_positions):
                    p0 = _fill_inputs(ref_rows, m, seed=seed, rank=rank, profile=profile, resid_scale=resid_scale, device=device)
                    p1 = _fill_inputs(brrr_rows, m, seed=seed, rank=rank, profile=profile, resid_scale=resid_scale, device=device)
                    assert torch.equal(p0, p1)
                    positions[:m].copy_(p0)  # the stock reference and the probe read this tensor
                    tail_positions[:m].copy_(p0)
                    snapshot = kernel_inputs(brrr_rows, m)
                    if route_counts is not None:
                        route_counts.zero_()
                    torch.cuda.synchronize(device)
                    reference(positions[:m])
                    tail(brrr_rows.attention.attended)  # same launch sequence on every rank: the kernel all-reduces internally
                    torch.cuda.synchronize(device)
                    dist.barrier(group=group)
                    routes = reference.route_report()
                    kernel_rows = None
                    if route_counts is not None:
                        launch_counts = route_counts.clone()
                        kernel_rows = kernel_routing(snapshot, p0, m)
                        routes.update(reference.reconcile(kernel_rows, launch_counts))
                    got, want = _outputs(brrr_rows, m), _outputs(ref_rows, m)
                    stats = {name: _stats(got[name], want[name]) for name in COMPARED}
                    # Collective (two FP64 all-reduces): every rank evaluates every launch.
                    bounds = ref64.evaluate(inputs={**snapshot, "positions": p0}, outputs=got, kernel_rows=kernel_rows)
                    bad = [f"bound:{name}" for name, report in bounds.items() if not report["ok"]]
                    bad += [name for name, st in stats.items() if not st["finite"]
                            or st["cosine"] < (MLP_TOLERANCE if name == "mlp_output" else TOLERANCE)
                            or (name == "mlp_output" and st["row_min_cosine"] < MLP_ROW_TOLERANCE)]
                    if "routing_error" in routes:
                        bad.append("routing")
                    reports.append((m, profile, routes, bounds, stats, bad))
                    if rank == 0:
                        line = " ".join(f"{name}: cos={st['cosine']:.6f} rowmin={st['row_min_cosine']:.6f} "
                                        f"maxrel={st['max_rel']:.3e} maxabs={st['max_abs']:.3e}" for name, st in stats.items())
                        print(f"[brrr-tail] layer {layer} m={m} {profile}: {routes} | bounds: {reference64.summarize(bounds)} | {line}"
                              + (f" | FAIL {bad}" if bad else ""), flush=True)
                    if bad:
                        details = {name: stats[name] for name in bad if name in stats}
                        details.update({name: bounds[name.removeprefix("bound:")] for name in bad if name.startswith("bound:")})
                        if "routing_error" in routes:
                            details["routing"] = routes["routing_error"]
                        failures.append((rank, m, profile, bad, details))
                    if profile == "uniform" and m > 16:
                        assert routes["max_routes_per_expert"] == m > 16, routes  # > one 16-route chunk per expert
                    if parked and profile == "random" and m == capacity:
                        assert routes["cold_routes"] > 0, f"no cold route hit with hot_rows={weights.experts.hot_rows}: {routes}"

                seed = 1000 * layer
                for m in rows_sweep:
                    ref_rows, brrr_rows, reference, tail = paths[m]
                    for profile, resid_scale in PROFILES:
                        seed += 1
                        launch(m, profile, resid_scale, seed, ref_rows, brrr_rows, reference, tail, positions)
                # Held-out input on a fresh tail: new scratch, new positions tensor and a newly constructed
                # PreparedBRRRTail (fresh extension call with buffers the sweep never used), same bounds.
                heldout_rows = Scratch.allocate(capacity, capacity, WORLD, device).rows(capacity, capacity)
                heldout_positions = torch.zeros((capacity,), dtype=torch.int64, device=device)
                heldout = PreparedBRRRTail(layer, weights, heldout_rows, heldout_positions[:capacity], frequencies, **tail_kwargs)
                ref_rows, _, reference, _ = paths[capacity]
                launch(capacity, "heldout", 1.0, 1000 * layer + 999, ref_rows, heldout_rows, reference, heldout, heldout_positions)
                dist.barrier(group=group)
                iters = int(os.environ.get("FLARE_DS41_BRRR_TEST_PROFILE", "0"))
                if iters > 0 and not expect_failure:
                    _profile_widths(rank=rank, group=group, device=device, ext=ext, layer=layer, rows_sweep=rows_sweep,
                                    paths=paths, positions=positions, route_counts=route_counts, iters=iters)
                variants = dict(group=group, device=device, storage=storage, layer=layer, weights=weights, config=config,
                                frequencies=frequencies, paths=paths, positions=positions, route_counts=route_counts,
                                tp3=tp3)
                if tp3.brrr_split_tail and rank == 0:
                    failures.extend(_split_layout_check(ext, 2 * torch.cuda.get_device_properties(device).multi_processor_count))
                if os.environ.get("FLARE_DS41_BRRR_TEST_EQUIV") == "1" and not expect_failure:
                    failures.extend(_equivalence(rank=rank, rows_sweep=rows_sweep, **variants))
                ab = os.environ.get("FLARE_DS41_BRRR_TEST_AB", "")
                if ab and not expect_failure:
                    _ab_profile(rank=rank, spec=ab, rows_sweep=rows_sweep, **variants)
                gathered = [None] * WORLD
                dist.all_gather_object(gathered, failures, group=group)
                every = [f for part in gathered for f in part]
                if expect_failure:
                    assert every, "negative control (FLARE_DS41_BRRR_BREAK=rope) passed the comparison; the test is not sensitive"
                else:
                    assert not every, f"BRRR tail differs from Flare's path: {every}"
            del paths
    finally:
        torch.cuda.synchronize(device)
        dist.destroy_process_group()


def _snapshot():
    snapshot = os.environ.get("FLARE_DS41_SNAPSHOT")
    if snapshot is None or not (Path(snapshot) / "model.safetensors.index.json").is_file():
        pytest.skip("FLARE_DS41_SNAPSHOT must identify a DeepSeek V4.1 Flash checkpoint (ds41-mini8 is fast)")
    return str(Path(snapshot).resolve())


def _base_env(snapshot, layer):
    text = json.loads((Path(snapshot) / "config.json").read_text())["text_config"]
    return {
        "FLARE_DS41_MINI": "1" if text["num_hidden_layers"] != 40 else None,
        "FLARE_DS41_BRRR_LAYERS": str(layer),
        "FLARE_DS41_BRRR_CKPT": snapshot,
        "FLARE_DS41_BRRR_BREAK": None,
        "FLARE_DS41_GLOBAL_COLD": None,
        "FLARE_DS41_HOST_EXPERT_LAYERS": None,
        "FLARE_DS41_HOT_EXPERTS": None,
        "FLARE_DS41_MERGED_BANK": None,
        "FLARE_DS41_BRRR_PARKED_MAX_ROWS": None,
        "DS41_FUSED_TEST": None,
        # Item-6 kernel switches: off unless the case sets them (CASES "item6").
        "FLARE_DS41_BRRR_SPLIT_TAIL": None,
        "FLARE_DS41_BRRR_FAST_ROUTE": None,
        "FLARE_DS41_BRRR_COUNTER_PARITY": None,
        "FLARE_DS41_BRRR_TEST_EQUIV": None,
        "FLARE_DS41_BRRR_SPIN_TRAP": None,
        # Item-6 A/B timing pass (see _ab_profile) passes through from the caller's environment.
        **{key: os.environ.get(key) for key in ("FLARE_DS41_BRRR_TEST_AB", "FLARE_DS41_BRRR_TEST_AB_ROWS",
                                                 "FLARE_DS41_BRRR_TEST_AB_REPEATS", "FLARE_DS41_BRRR_TEST_AB_ITERS",
                                                 "FLARE_DS41_BRRR_TEST_AB_OUT")},
        # Timing pass and kernel profiling pass through from the caller's environment (see _profile_widths).
        "FLARE_DS41_BRRR_TEST_PROFILE": os.environ.get("FLARE_DS41_BRRR_TEST_PROFILE"),
        "FLARE_DS41_BRRR_TEST_PROFILE_MIX": os.environ.get("FLARE_DS41_BRRR_TEST_PROFILE_MIX"),
        "DS41_FUSED_PROF": os.environ.get("DS41_FUSED_PROF"),
    }


def _parked_env(tmp_path, layer):
    hot = int(os.environ.get("FLARE_DS41_BRRR_TEST_HOT", "64"))
    counts = tmp_path / "counts"
    counts.mkdir(exist_ok=True)
    # Experts 0..hot-1 are the "hottest"; the rest live in pinned host memory behind the merged bank.
    (counts / f"expert-routes-rank0-layer{layer}.json").write_text(json.dumps({"counts": [384 - e for e in range(384)]}))
    return {"FLARE_DS41_HOST_EXPERT_LAYERS": str(layer), "FLARE_DS41_HOT_EXPERTS": str(hot), "FLARE_DS41_MERGED_BANK": "1",
            "FLARE_DS41_EXPERT_COUNTS_DIR": str(counts)}


# Item-6 switch sets (compile-time kernel variants). SPLIT_TAIL 2 keeps the window inside the last chunk; 96 crosses
# chunks at small widths and stays in the lagged block region at wide ones; 1 << 20 splits every down job.
# The item-6 cases run spin-trap builds (FLARE_DS41_BRRR_SPIN_TRAP): a protocol wait that never ends traps after a
# few seconds instead of wedging the GPUs (the timing pass and exports never use them).
TRAP = {"FLARE_DS41_BRRR_SPIN_TRAP": "1"}
ITEM6_ALL = {"FLARE_DS41_BRRR_SPLIT_TAIL": "96", "FLARE_DS41_BRRR_FAST_ROUTE": "1", "FLARE_DS41_BRRR_COUNTER_PARITY": "1", **TRAP}

CASES = {
    "resident-perchunk1": dict(parked=False, per_chunk="1", broken=False),
    "resident-perchunk0": dict(parked=False, per_chunk="0", broken=False),
    "parked-perchunk1": dict(parked=True, per_chunk="1", broken=False),
    "parked-perchunk0": dict(parked=True, per_chunk="0", broken=False),
    "resident-break-rope": dict(parked=False, per_chunk="1", broken=True),
    "resident-split2": dict(parked=False, per_chunk="1", broken=False, item6={"FLARE_DS41_BRRR_SPLIT_TAIL": "2", **TRAP}),
    "resident-splitall": dict(parked=False, per_chunk="1", broken=False, item6={"FLARE_DS41_BRRR_SPLIT_TAIL": str(1 << 20), **TRAP}),
    "resident-fastroute": dict(parked=False, per_chunk="1", broken=False, item6={"FLARE_DS41_BRRR_FAST_ROUTE": "1", **TRAP}),
    "resident-parity": dict(parked=False, per_chunk="1", broken=False, item6={"FLARE_DS41_BRRR_COUNTER_PARITY": "1", **TRAP}),
    "resident-item6": dict(parked=False, per_chunk="1", broken=False, item6=ITEM6_ALL),
    "parked-item6": dict(parked=True, per_chunk="1", broken=False, item6=ITEM6_ALL),
    "resident-item6-break-rope": dict(parked=False, per_chunk="1", broken=True, item6=ITEM6_ALL),
    # Bit-exact check of fast_route / counter_parity against the plain build (split_tail's difference reported only).
    "resident-item6-equivalence": dict(parked=False, per_chunk="1", broken=False, item6={"FLARE_DS41_BRRR_TEST_EQUIV": "1", **TRAP}),
    # parked: hot_map slots, cold entries (the low NCOLD half of fast_route's 64-bit atomic) and the slot shuffle
    "parked-item6-equivalence": dict(parked=True, per_chunk="1", broken=False, item6={"FLARE_DS41_BRRR_TEST_EQUIV": "1", **TRAP}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_brrr_tail_matches_flare_layer_path(tmp_path, case):
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires three explicitly assigned CUDA devices")
    options = CASES[case]
    snapshot = _snapshot()
    layer = int(os.environ.get("FLARE_DS41_BRRR_TEST_PARKED_LAYER" if options["parked"] else "FLARE_DS41_BRRR_TEST_LAYER",
                               "3" if options["parked"] else "2"))
    env = _base_env(snapshot, layer)
    env["FLARE_DS41_FUSED_ATTN_PER_CHUNK"] = options["per_chunk"]
    if options["parked"]:
        env.update(_parked_env(tmp_path, layer))
    if options["broken"]:
        env["FLARE_DS41_BRRR_BREAK"] = "rope"
    env.update(options.get("item6", {}))
    context = mp.spawn(_rank, args=((tmp_path / f"rendezvous-{case}").as_uri(), snapshot, env, layer, options["broken"]),
                       nprocs=WORLD, join=False)
    # A protocol bug hangs the kernel rather than failing it: bound the case (FLARE_DS41_BRRR_TEST_TIMEOUT seconds,
    # default 3600) and kill the ranks so the GPUs come free.
    import time

    deadline = time.monotonic() + float(os.environ.get("FLARE_DS41_BRRR_TEST_TIMEOUT", "3600"))
    while not context.join(timeout=10):
        if time.monotonic() > deadline:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            pytest.fail(f"{case}: ranks did not finish within FLARE_DS41_BRRR_TEST_TIMEOUT; killed (kernel hang?)")

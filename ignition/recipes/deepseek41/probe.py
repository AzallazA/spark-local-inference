# SPDX-License-Identifier: Apache-2.0
"""Debug-only per-layer residual probes for the DeepSeek decode-corruption hunt (flare#193).

Enabled by FLARE_DS41_PROBE=1 (debug-serve only; never export with it). Each captured forward records
four statistics (sum, sum of squares, max |x|, first element) per probed tensor:

  * decode graphs: row 0, into a ring buffer indexed by a device-side decode step counter;
  * prefill graphs: every row, into a prefill table (overwritten by each prefill chunk).

Rank 0 dumps the buffers to /tmp/flare_probe.pt whenever /tmp/flare_probe_req exists.
"""

import os
import threading
import time

import torch

ENABLED = os.environ.get("FLARE_DS41_PROBE", "") not in ("", "0")
RING, LAYERS, STAGES, PREFILL_ROWS = 512, 40, 6, 4096
FULL_LAYERS = (37, 38, 39)  # full-row captures of the MoE input and routed output (bf16)
_STATE = {}


def _state(device):
    key = torch.device(device).index
    if key not in _STATE:
        state = {
            "ring": torch.zeros((RING, LAYERS, STAGES, 4), dtype=torch.float32, device=device),
            "counter": torch.zeros((1,), dtype=torch.int64, device=device),
            "prefill": torch.zeros((PREFILL_ROWS, LAYERS, STAGES, 4), dtype=torch.float32, device=device),
            "prefill_rows": torch.zeros((1,), dtype=torch.int64, device=device),
            "ring_pos": torch.full((RING,), -1, dtype=torch.int64, device=device),
            "prefill_pos": torch.full((PREFILL_ROWS,), -1, dtype=torch.int64, device=device),
            "ring_ids": torch.full((RING, LAYERS, 6), -1, dtype=torch.int32, device=device),
            "ring_w": torch.zeros((RING, LAYERS, 6), dtype=torch.float32, device=device),
            "prefill_ids": torch.full((PREFILL_ROWS, LAYERS, 6), -1, dtype=torch.int32, device=device),
            "prefill_w": torch.zeros((PREFILL_ROWS, LAYERS, 6), dtype=torch.float32, device=device),
            "ring_in": torch.zeros((RING, len(FULL_LAYERS), 5120), dtype=torch.bfloat16, device=device),
            "ring_out": torch.zeros((RING, len(FULL_LAYERS), 5120), dtype=torch.bfloat16, device=device),
            "prefill_in": torch.zeros((PREFILL_ROWS, len(FULL_LAYERS), 5120), dtype=torch.bfloat16, device=device),
            "prefill_out": torch.zeros((PREFILL_ROWS, len(FULL_LAYERS), 5120), dtype=torch.bfloat16, device=device),
        }
        _STATE[key] = state
        if key == 0:
            threading.Thread(target=_dumper, args=(state,), daemon=True).start()
    return _STATE[key]


def _dumper(state):
    while True:
        if os.path.exists("/tmp/flare_probe_req"):
            try:
                torch.save({name: value.detach().cpu() for name, value in state.items()}, "/tmp/flare_probe.pt.tmp")
                os.replace("/tmp/flare_probe.pt.tmp", "/tmp/flare_probe.pt")
            finally:
                os.remove("/tmp/flare_probe_req")
        time.sleep(0.2)


@torch.library.custom_op("flare_debug::probe_step", mutates_args=("counter",))
def _probe_step(counter: torch.Tensor) -> None:
    counter.add_(1)


@_probe_step.register_fake
def _probe_step_fake(counter):
    pass


@torch.library.custom_op("flare_debug::probe_record", mutates_args=("ring", "prefill", "prefill_rows"))
def _probe_record(value: torch.Tensor, ring: torch.Tensor, prefill: torch.Tensor, prefill_rows: torch.Tensor,
                  counter: torch.Tensor, layer: int, stage: int, decode: bool) -> None:
    rows = value.shape[0]
    flat = value.reshape(rows, -1).float()
    stats = torch.stack((flat.sum(1), (flat * flat).sum(1), flat.abs().amax(1), flat[:, 0]), dim=1)
    if decode:
        slot = torch.remainder(counter, RING)
        ring[:, layer, stage].index_copy_(0, slot, stats[:1])
    else:
        count = min(rows, PREFILL_ROWS)
        prefill[:count, layer, stage].copy_(stats[:count])
        prefill_rows.fill_(rows)


@_probe_record.register_fake
def _probe_record_fake(value, ring, prefill, prefill_rows, counter, layer, stage, decode):
    pass


@torch.library.custom_op("flare_debug::probe_positions", mutates_args=("ring_pos", "prefill_pos"))
def _probe_positions(positions: torch.Tensor, ring_pos: torch.Tensor, prefill_pos: torch.Tensor,
                     counter: torch.Tensor, decode: bool) -> None:
    if decode:
        ring_pos.index_copy_(0, torch.remainder(counter, RING), positions[:1].long())
    else:
        count = min(positions.shape[0], PREFILL_ROWS)
        prefill_pos[:count].copy_(positions[:count].long())


@_probe_positions.register_fake
def _probe_positions_fake(positions, ring_pos, prefill_pos, counter, decode):
    pass


def positions(value, mode):
    if ENABLED and mode in ("decode", "prefill"):
        state = _state(value.device)
        _probe_positions(value.reshape(-1), state["ring_pos"], state["prefill_pos"], state["counter"], mode == "decode")


@torch.library.custom_op("flare_debug::probe_router",
                         mutates_args=("ring_ids", "ring_w", "prefill_ids", "prefill_w"))
def _probe_router(ids: torch.Tensor, weights: torch.Tensor, ring_ids: torch.Tensor, ring_w: torch.Tensor,
                  prefill_ids: torch.Tensor, prefill_w: torch.Tensor, counter: torch.Tensor, layer: int,
                  decode: bool) -> None:
    if decode:
        slot = torch.remainder(counter, RING)
        ring_ids[:, layer].index_copy_(0, slot, ids[:1].int())
        ring_w[:, layer].index_copy_(0, slot, weights[:1].float())
    else:
        count = min(ids.shape[0], PREFILL_ROWS)
        prefill_ids[:count, layer].copy_(ids[:count].int())
        prefill_w[:count, layer].copy_(weights[:count].float())


@_probe_router.register_fake
def _probe_router_fake(ids, weights, ring_ids, ring_w, prefill_ids, prefill_w, counter, layer, decode):
    pass


def router(ids, weights, layer, mode):
    if ENABLED and mode in ("decode", "prefill") and ids.shape[-1] == 6:
        state = _state(ids.device)
        _probe_router(ids, weights, state["ring_ids"], state["ring_w"], state["prefill_ids"], state["prefill_w"],
                      state["counter"], layer, mode == "decode")


@torch.library.custom_op("flare_debug::probe_full", mutates_args=("ring", "prefill"))
def _probe_full(value: torch.Tensor, ring: torch.Tensor, prefill: torch.Tensor, counter: torch.Tensor,
                index: int, decode: bool) -> None:
    rows = value.reshape(value.shape[0], -1)
    if decode:
        ring[:, index].index_copy_(0, torch.remainder(counter, RING), rows[:1].to(ring.dtype))
    else:
        count = min(rows.shape[0], PREFILL_ROWS)
        prefill[:count, index].copy_(rows[:count].to(prefill.dtype))


@_probe_full.register_fake
def _probe_full_fake(value, ring, prefill, counter, index, decode):
    pass


def full(value, layer, kind, mode):
    if ENABLED and mode in ("decode", "prefill") and layer in FULL_LAYERS and value.reshape(value.shape[0], -1).shape[1] == 5120:
        state = _state(value.device)
        _probe_full(value, state["ring_" + kind], state["prefill_" + kind], state["counter"],
                    FULL_LAYERS.index(layer), mode == "decode")


def step(device, mode):
    if ENABLED and mode == "decode":
        _probe_step(_state(device)["counter"])


def record(value, layer, stage, mode):
    if ENABLED and mode in ("decode", "prefill"):
        state = _state(value.device)
        _probe_record(value, state["ring"], state["prefill"], state["prefill_rows"], state["counter"],
                      layer, stage, mode == "decode")

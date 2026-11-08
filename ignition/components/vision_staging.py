# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage host-parked weight sets through VRAM slots inside captured graphs.

A vision tower parked in device-visible host memory (``TP3Config.vision_host``) is correct but
slow: every GEMM re-reads its weight tile per row block over PCIe. Copying each block's weights
into a VRAM slot once per image restores resident speed while only the slots stay in VRAM. The
copies run on the arena's auxiliary stream, forked after the main stream's enqueued work (the
slot's previous reader) and joined before the slot's next consumer, following
``ConcurrentProjectionChain``. Both sides are custom operators so compiled and captured graphs
record the fork and join as event dependencies without host synchronization.
"""

import torch

_INTEGER_VIEWS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


@torch.library.custom_op("flare::stage_parked_weights", mutates_args=("destinations",))
def _stage(token: torch.Tensor, sources: list[torch.Tensor], destinations: list[torch.Tensor],
           stream: int, kernel_copy: bool) -> None:
    device = token.device
    current = torch.cuda.current_stream(device)
    if stream == 0:
        auxiliary = current
    else:
        auxiliary = torch.cuda.ExternalStream(stream, device=device)
        auxiliary.wait_stream(current)
    with torch.cuda.stream(auxiliary):
        for source, destination in zip(sources, destinations, strict=True):
            if kernel_copy:
                # Bitwise identity through an elementwise kernel: coalesced sequential loads from the
                # mapped host range, independent of memcpy-kind inference for host-mapped pointers.
                view = _INTEGER_VIEWS[destination.element_size()]
                torch.bitwise_or(source.view(view), 0, out=destination.view(view))
            else:
                destination.copy_(source, non_blocking=True)


@_stage.register_fake
def _fake_stage(token, sources, destinations, stream, kernel_copy):
    pass


@torch.library.custom_op("flare::join_parked_weights", mutates_args=("token", "destinations"))
def _join(token: torch.Tensor, dependency: torch.Tensor, destinations: list[torch.Tensor], stream: int) -> None:
    if stream:
        device = token.device
        torch.cuda.current_stream(device).wait_stream(torch.cuda.ExternalStream(stream, device=device))


@_join.register_fake
def _fake_join(token, dependency, destinations, stream):
    pass


class WeightStaging:
    """VRAM slots for weight sets that share one geometry.

    ``templates`` maps weight names to tensors fixing each slot entry's shape and dtype; ``slots``
    sets are live at once (two double-buffer a sequential block stack: block ``i`` computes from
    slot ``i % 2`` while block ``i + 1`` lands in the other). Slot tensors come from the graph's
    buffer cursor when given, so they share the pool with the other graphs' private buffers and
    are rewritten on every replay; they are scratch, not checkpoint storage.

    ``stage(slot, sources)`` forks the copies onto ``stream`` after the main stream's enqueued
    work, so a slot is overwritten only once its previous reader has finished; ``join(slot,
    dependency)`` makes the main stream wait before the slot's consumers run.

    Eager and captured execution order these by stream order. Under ``torch.compile`` only data
    dependencies order them, so two operands pin the schedule: ``join`` mutates ``token`` and
    ``stage`` reads it, so ``stage(i + 1)`` stays after ``join(i)`` (hoisting it would make
    ``join(i)`` also wait for the new copies); ``join`` reads ``dependency``, the tensor the
    preceding block's last kernel wrote, so ``join(i + 1)`` stays after block ``i``'s compute
    (hoisting it would stall the main stream on block ``i + 1``'s copy before block ``i``). The
    slot tensors themselves carry the write-after-read ordering against their previous consumers.
    """

    def __init__(self, templates, *, slots=None, storage=None, stream=None, buffers=None, kernel_copy=False,
                 token=None):
        """``storage`` supplies preallocated slot dictionaries (for example tensors that alias an earlier
        stage of the graph's scratch) instead of allocating ``slots`` new ones. Sets staged through one
        stream in one graph should share ``token`` so their stages and joins keep program order."""
        if not templates or (storage is None) == (slots is None):
            raise ValueError("weight staging requires weights and either a slot count or preallocated slots")
        self.names = tuple(templates)
        device = next(iter(templates.values())).device

        def zeros(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(tuple(shape), dtype))

        if storage is None:
            if slots < 1:
                raise ValueError("weight staging requires at least one slot")
            self.slots = tuple({name: zeros(tensor.shape, tensor.dtype) for name, tensor in templates.items()}
                               for _ in range(slots))
        else:
            self.slots = tuple(dict(slot) for slot in storage)
            if not self.slots or any(set(slot) != set(self.names) or any(
                    slot[name].shape != templates[name].shape or slot[name].dtype != templates[name].dtype
                    or slot[name].device != device or not slot[name].is_contiguous() for name in self.names)
                    for slot in self.slots):
                raise ValueError("preallocated staging slots must match the staged weights' geometry")
        self.token = zeros((1,), torch.int32) if token is None else token
        self.stream = stream
        self.stream_handle = 0 if stream is None else stream.cuda_stream
        self.kernel_copy = bool(kernel_copy)

    @property
    def nbytes(self):
        return sum(tensor.nbytes for slot in self.slots for tensor in slot.values())

    def slot(self, index):
        return self.slots[index % len(self.slots)]

    def matches(self, sources):
        return all(name in sources and sources[name].shape == template.shape
                   and sources[name].dtype == template.dtype
                   for name, template in self.slots[0].items())

    def stage(self, index, sources):
        slot = self.slot(index)
        _stage(self.token, [sources[name] for name in self.names], [slot[name] for name in self.names],
               self.stream_handle, self.kernel_copy)

    def join(self, index, dependency):
        """``dependency`` is an output of the main-stream work the join must follow."""
        slot = self.slot(index)
        _join(self.token, dependency, [slot[name] for name in self.names], self.stream_handle)

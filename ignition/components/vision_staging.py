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


def tensor_leaves(value):
    """Tensors of a weight tree (nested dataclasses, dicts, lists and tuples) in a fixed order."""
    import dataclasses

    if isinstance(value, torch.Tensor):
        return [value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [leaf for field in dataclasses.fields(value) for leaf in tensor_leaves(getattr(value, field.name))]
    if isinstance(value, dict):
        return [leaf for key in value for leaf in tensor_leaves(value[key])]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in tensor_leaves(item)]
    return []


def map_tensors(value, function):
    """Rebuild a weight tree with ``function(tensor)`` in place of every tensor, in `tensor_leaves` order.

    Frozen dataclasses (b12x packed weights) are rebuilt with `dataclasses.replace`; non-tensor fields
    (feature counts, scale kinds) are kept."""
    import dataclasses

    if isinstance(value, torch.Tensor):
        return function(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {field.name: map_tensors(getattr(value, field.name), function)
                   for field in dataclasses.fields(value) if field.init}
        return dataclasses.replace(value, **changes)
    if isinstance(value, dict):
        return type(value)((key, map_tensors(item, function)) for key, item in value.items())
    if isinstance(value, tuple):
        return type(value)(*(map_tensors(item, function) for item in value)) if hasattr(value, "_fields") \
            else tuple(map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [map_tensors(item, function) for item in value]
    return value


_ALIGNMENT = 256


class PackedLayout:
    """Byte layout of a weight tree's tensors in one flat ``uint8`` buffer.

    Every distinct storage the tree's tensors view contributes the byte span its tensors cover (from a
    256-byte-aligned start, so each tensor keeps its address alignment modulo 256); tensors sharing a
    storage keep sharing it, with their strides and relative offsets. A tree packed into a flat buffer
    (`pack`) and viewed back (`bind`) therefore holds the same bytes under the same tensor geometry, so a
    kernel prepared over the bound tree reads exactly the operands it would read from the original.

    Trees with equal `signature` share a layout: one flat VRAM slot can be bound once and refilled from
    any of their packed copies with a single contiguous copy.
    """

    def __init__(self, tree):
        storages, spans, entries = {}, [], []
        for tensor in tensor_leaves(tree):
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
            size = tensor.element_size()
            first = tensor.storage_offset() * size
            if tensor.numel():
                last = first + (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))) * size
            else:
                last = first
            if key not in storages:
                storages[key] = len(spans)
                spans.append([first, last])
            index = storages[key]
            spans[index][0] = min(spans[index][0], first)
            spans[index][1] = max(spans[index][1], last)
            entries.append((index, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), first))
        offsets, cursor = [], 0
        for span in spans:
            span[0] -= span[0] % _ALIGNMENT
            offsets.append(cursor)
            cursor += -(-(span[1] - span[0]) // _ALIGNMENT) * _ALIGNMENT
        self.spans = tuple((first, last) for first, last in spans)
        self.offsets = tuple(offsets)
        self.entries = tuple((index, dtype, shape, stride, first - self.spans[index][0])
                             for index, dtype, shape, stride, first in entries)
        self.nbytes = cursor

    @property
    def signature(self):
        return (self.nbytes, tuple(last - first for first, last in self.spans), self.offsets, self.entries)

    def pack(self, tree, flat):
        """Copy the covered bytes of ``tree`` (the tree this layout describes) into ``flat``.

        The layout keeps no reference to the tree's storage, so dropping the tree releases it."""
        if flat.dtype != torch.uint8 or flat.ndim != 1 or flat.numel() < self.nbytes:
            raise ValueError("packed weights need a flat uint8 buffer of at least the layout's size")
        if PackedLayout(tree).signature != self.signature:
            raise ValueError("weight tree does not match the packed layout")
        storages = {}
        for tensor in tensor_leaves(tree):
            storages.setdefault((tensor.device, tensor.untyped_storage().data_ptr()), tensor.untyped_storage())
        for (first, last), offset, storage in zip(self.spans, self.offsets, storages.values(), strict=True):
            source = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage, 0, (storage.nbytes(),))
            flat[offset:offset + last - first].copy_(source[first:last])
        return flat

    def bind(self, tree, flat):
        """``tree`` with every tensor viewing ``flat`` at this layout's place (tree must match the layout)."""
        if flat.dtype != torch.uint8 or flat.ndim != 1 or flat.numel() < self.nbytes:
            raise ValueError("packed weights need a flat uint8 buffer of at least the layout's size")
        entries = iter(self.entries)
        flat = flat[:self.nbytes]
        base = flat.storage_offset()
        if base % _ALIGNMENT:
            raise ValueError("a packed weight buffer must start 256-byte aligned")
        typed = {}

        def view(tensor):
            index, dtype, shape, stride, offset = next(entries)
            if (tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())) != (dtype, shape, stride):
                raise ValueError("weight tree does not match the packed layout")
            if dtype not in typed:
                typed[dtype] = flat.view(dtype)
            size = typed[dtype].element_size()
            start = base + self.offsets[index] + offset
            if start % size:
                raise ValueError("packed tensor offset breaks its element alignment")
            return typed[dtype].as_strided(shape, stride, start // size)

        result = map_tensors(tree, view)
        if next(entries, None) is not None:
            raise ValueError("weight tree does not match the packed layout")
        return result


def pack_tree(tree, allocate):
    """Pack a weight tree into ``allocate(nbytes)`` (a flat uint8 tensor, for example device-visible host
    memory); return ``(layout, flat, bound)`` where ``bound`` is the tree viewing ``flat``."""
    layout = PackedLayout(tree)
    flat = allocate(layout.nbytes)
    layout.pack(tree, flat)
    return layout, flat, layout.bind(tree, flat)

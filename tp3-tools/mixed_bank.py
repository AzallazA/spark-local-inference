"""One expert bank in one virtual range: the first `split` experts in VRAM, the rest in pinned host memory (CUDA VMM)."""
import torch
from cuda.bindings import driver as d
from b12x.sequence._shared.disk_table import _tensor_from_pointer

_keep = []


def _ok(result):
    err = result[0] if isinstance(result, tuple) else result
    if err != d.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {err}")
    return result[1] if isinstance(result, tuple) and len(result) == 2 else (result[1:] if isinstance(result, tuple) else None)


def _prop(kind, device):
    p = d.CUmemAllocationProp()
    p.type = d.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    p.location.type = kind
    p.location.id = device if kind == d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE else 0
    return p


def mixed_tensor(shape, dtype, split, device):
    """Tensor whose rows [0, split) are device-backed and rows [split, shape[0]) host-backed."""
    dev = torch.device(device).index
    torch.cuda.init(); torch.empty(1, device=device)  # make the primary context current
    row = torch.Size(shape[1:]).numel() * dtype.itemsize
    total = shape[0] * row
    pd = _prop(d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, dev)
    ph = _prop(d.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, 0)
    flag = d.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    gran = max(_ok(d.cuMemGetAllocationGranularity(pd, flag)), _ok(d.cuMemGetAllocationGranularity(ph, flag)))
    up = lambda n: -(-n // gran) * gran
    dev_bytes = up(split * row)          # the boundary page goes to the device side
    size = up(total)
    host_bytes = max(size - dev_bytes, 0)
    va = _ok(d.cuMemAddressReserve(size, gran, 0, 0))
    handles = []
    for nbytes, prop, offset in ((dev_bytes, pd, 0), (host_bytes, ph, dev_bytes)):
        if nbytes:
            h = _ok(d.cuMemCreate(nbytes, prop, 0))
            _ok(d.cuMemMap(int(va) + offset, nbytes, 0, h, 0))
            handles.append(h)
    access = d.CUmemAccessDesc()
    access.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = dev
    access.flags = d.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _ok(d.cuMemSetAccess(va, size, [access], 1))
    _keep.append((va, size, handles))
    return _tensor_from_pointer(int(va), shape=tuple(shape), dtype=dtype, device=torch.device("cuda", dev), nbytes=total)


def allocator(split, device):
    def host_alloc(name, shape, dtype):
        view = mixed_tensor(shape, dtype, split, device)

        class Region:
            device_view = view
        return Region
    return host_alloc

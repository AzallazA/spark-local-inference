import hashlib
import os
from functools import cache
from pathlib import Path
from types import ModuleType

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

KERNELS = Path(__file__).resolve().parent / "kernels"
SOURCES = {path.stem: path for path in sorted((*KERNELS.glob("*.cu"), *KERNELS.glob("*.cpp")))}
LIBRARIES = {"engram_nvme": ["-luring"]}


@cache
def extension(name: str, defines: tuple[str, ...] = ()) -> ModuleType:
    """Build/load kernel source `name`; `defines` ("NAME=VALUE" strings, passed as -D) select a compile-time variant,
    built and cached under its own module name (no defines = the plain build, same name scheme as before)."""
    source = SOURCES[name]
    digest = hashlib.sha1(b"".join(path.read_bytes() for path in (source, *sorted(KERNELS.glob("*.cuh"))))).hexdigest()[:10]
    cuda = source.suffix == ".cu"
    defines = tuple(defines)
    if defines and not cuda:
        raise ValueError(f"compile-time defines are only supported for CUDA sources, not {source.name}")
    variant = "" if not defines else "_" + hashlib.sha1("\n".join(defines).encode()).hexdigest()[:8]
    return load(
        name=f"{name}_{digest}{variant}",
        sources=[str(source)],
        extra_cflags=None if cuda else ["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", *(["-lineinfo"] if os.environ.get("BRRR_LINEINFO") else []),
                           *(f"-D{d}" for d in defines)] if cuda else None,
        extra_ldflags=LIBRARIES.get(name),
        verbose=False,
    )


def exchange_ipc(ext: ModuleType, nbytes: int, rank: int, group: ProcessGroup) -> list[int]:
    mine = ext.alloc_ipc(nbytes)
    handles: list[list[int]] = [[] for _ in range(ext.RANKS)]
    dist.all_gather_object(handles, mine[1:], group=group)
    return [mine[0] if peer == rank else ext.open_ipc(handles[peer]) for peer in range(ext.RANKS)]

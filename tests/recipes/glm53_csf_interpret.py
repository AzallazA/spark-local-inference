# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Run b12x's CSF scale-gathering Triton kernels on the CPU, for both checkpoint encodings.

`_csf_prepare.py` reads a `CsfSource` in two ways: `expand_source_scales` (the one-time load-time
expansion, `_expand_source`) and `prepare_csf_payload` (the compact payload that the per-layer
scratch decoder reads: `_pack_codes` + `_pack_records`). Both share `_read`, which has a `LEGACY`
branch for the `csf` slab encoding (16-row slabs, global 24-bit exception positions) next to the
`csf_32x2_v1` one. Triton's interpreter (`TRITON_INTERPRET=1`, chosen when the kernels are
decorated, hence the subprocess) runs these exact kernels on host tensors, so the padded GLM
geometry is checked for both encodings without a GPU. The CuTe decoder that consumes the compact
payload is GPU-only; `decode_payload` reads the payload the way the packer lays it out, and the
csf32 GPU test (`test_glm53_weights.py::...[3-2-*]`) anchors that reading.

Also provides synthetic encoders for both encodings (with exceptions) shared by the CPU tests.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ENCODINGS = ("csf_32x2_v1", "csf")


def encode_csf32(scales, exceptions=(), bases=None):
    """Encode [rows, columns] E4M3 bytes as csf_32x2_v1 planes: per-row minimum bases, 4-bit codes.

    Mirrors `decode_csf_scales`: 128-row groups, column groups of four as two pairs, 32-row blocks,
    32 x width codes per block (low nibble first), and per-block sorted exception words
    `(value << 8) | (row % 32 * 2 + column % 2)` indexed by page origins and page-relative prefixes.
    Entries listed in `exceptions` (or not representable as base + code) become exception words.
    `bases` overrides the per-row bases (a checkpoint encoder may choose bases other than the minimum).
    """
    rows, columns = scales.shape
    forced = set(map(tuple, exceptions))
    values = scales.to(torch.int16)
    keep = torch.ones_like(values, dtype=torch.bool)
    for row, column in forced:
        keep[row, column] = False
    if bases is None:
        bases = torch.where(keep, values, 255).min(dim=1).values.clamp_max(240)
    bases = bases.to(torch.int16)
    codes = values - bases[:, None]
    special = ~keep | (codes < 0) | (codes > 15)
    codes = torch.where(special, 0, codes)
    fields, records, prefixes, origins = [], [], [], []
    group = 0
    page_origin = 0
    for row in range(0, rows, 128):
        for column in range(0, columns, 4):
            if group % 32 == 0:
                page_origin = len(records)
                origins.append(2 * page_origin)
                prefixes.append(0)
            for pair in range(column, min(column + 4, columns), 2):
                width = min(2, columns - pair)
                for block_row in range(row, min(row + 128, rows), 32):
                    fields.append(codes[block_row:block_row + 32, pair:pair + width].reshape(-1))
                    tile = special[block_row:block_row + 32, pair:pair + width]
                    for local_row, local_column in tile.nonzero().tolist():
                        value = int(values[block_row + local_row, pair + local_column])
                        records.append((value << 8) | (local_row * 2 + local_column))
                    prefixes.append(len(records) - page_origin)
            group += 1
    fields = torch.cat(fields).to(torch.uint8).numpy()
    packed = torch.from_numpy((fields[0::2] | (fields[1::2] << 4)).astype(np.uint8))
    return (bases.to(torch.uint8), packed, torch.tensor(origins, dtype=torch.int64).to(torch.uint32),
            torch.tensor(prefixes, dtype=torch.int32).to(torch.uint16),
            torch.tensor(records, dtype=torch.int32).to(torch.uint16))


def encode_csf_slab(scales, exceptions=(), bases=None):
    """Encode [rows, columns] E4M3 bytes in the legacy `csf` slab encoding (`decode_csf_slab_scales`).

    Each 16-row slab holds 16 base bytes, then row-major pairs of four-bit offsets (low nibble first);
    exception words are `(value << 24) | (row * columns + column)`, sorted by position. `bases` as
    for `encode_csf32`.
    """
    rows, columns = scales.shape
    values = scales.to(torch.int16)
    keep = torch.ones_like(values, dtype=torch.bool)
    for row, column in exceptions:
        keep[row, column] = False
    if bases is None:
        bases = torch.where(keep, values, 255).min(dim=1).values.clamp_max(240)
    bases = bases.to(torch.int16)
    codes = values - bases[:, None]
    special = ~keep | (codes < 0) | (codes > 15)
    codes = torch.where(special, 0, codes).to(torch.uint8)
    pairs = codes[:, 0::2] | (codes[:, 1::2] << 4)
    fixed = torch.cat((bases.to(torch.uint8).reshape(rows // 16, 16), pairs.reshape(rows // 16, 8 * columns)), dim=1)
    positions = special.reshape(-1).nonzero().reshape(-1)
    words = (values.reshape(-1)[positions].to(torch.int64) << 24) | positions
    return fixed.contiguous(), words.to(torch.uint32)


def encode(encoding, scales, exceptions=(), bases=None):
    return (encode_csf32 if encoding == "csf_32x2_v1" else encode_csf_slab)(scales, exceptions, bases)


def row_bases(encoding, planes):
    """The per-row base plane of an encoded matrix (what the compact payload stores per row)."""
    if encoding == "csf_32x2_v1":
        return planes[0]
    return planes[0][:, :16].reshape(-1)


def _packed_row(row, columns):
    swapped = row // 4 * 4 + row % 2 * 2 + row % 4 // 2
    if columns == 1:
        return swapped // 32 * 32 + swapped % 8 // 2 * 8 + swapped % 2 + swapped % 32 // 8 * 2
    return swapped // 64 * 64 + swapped % 8 * 8 + swapped % 64 // 8


def decode_payload(storage, *, rows, columns, experts, destination):
    """Read a compact NVFP4 `CsfBlockPayload` (destination 0 or 1) back to logical [E, rows, columns] bytes.

    Follows `_pack_codes`/`_pack_records`: a stored base per row slot, 4-bit codes per 128 x 4 group
    in the destination's order, and per-group records (u16 position, u8 value) located by
    absolute page bases plus page-relative u16 prefixes.
    """
    from b12x._lib.quant.csf_blocks import BlockStorageLayout

    layout = BlockStorageLayout(rows, columns, experts, "nvfp4", True)
    storage = storage.cpu()
    bases = storage[layout.bases:layout.bases + experts * rows].view(experts, rows).to(torch.int32)
    pages_u32 = storage[layout.page_bases:].view(-1)[:experts * layout.pages * 4].view(torch.uint32).to(torch.int64)
    offsets_u16 = storage[layout.offsets:].view(-1)[:experts * layout.prefixes * 2].view(torch.uint16).to(torch.int64)
    values_at = int(storage[:8].view(torch.int64)[0])
    total = (values_at - layout.source_records) // 2
    positions = storage[layout.source_records:layout.source_records + 2 * total].view(torch.uint16).to(torch.int64)
    record_values = storage[values_at:values_at + total].to(torch.int32)
    out = torch.zeros((experts, rows, columns), dtype=torch.int32)
    cg = layout.column_groups
    for expert in range(experts):
        for group in range(layout.groups):
            rr, cc = group // cg * 128, group % cg * 4
            nr, nc = min(128, rows - rr), min(4, columns - cc)
            start = layout.codes + expert * layout.code_bytes + (rr * columns + cc * nr) // 2
            packed = storage[start:start + nr * nc // 2].to(torch.int32)
            codes = torch.stack((packed & 15, packed >> 4), 1).reshape(-1)
            i = torch.arange(nr * nc)
            if destination == 0:
                slot = i // nc
                row, col = slot % (nr // 32) * 32 + slot // (nr // 32), i % nc
            elif destination == 1:
                slot = i % nr
                row, col = _packed_row(slot, columns), i // nr
            else:
                raise ValueError("decode_payload reads destinations 0 and 1")
            out[expert, rr + row, cc + col] = bases[expert, rr + slot] + codes
            page = group // 32
            first = int(pages_u32[expert * layout.pages + page])
            index = expert * (layout.groups + layout.pages) + group + page
            lo, hi = first + int(offsets_u16[index]), first + int(offsets_u16[index + 1])
            p = positions[lo:hi]
            if destination == 0:
                prow, pcol = p // 4, p % 4
            else:
                prow, pcol = _packed_row(p % nr, columns), p // nr
            out[expert, rr + prow, cc + pcol] = record_values[lo:hi]
    return out.to(torch.uint8)


def run(cases):
    """Run `cases` through the interpreted b12x kernels in a CPU-only subprocess; returns their outputs.

    Each case is a dict: `encoding`, `components` (one tuple of planes per segment), `rows`,
    `columns`, `selection` (CsfSource fields), and `ops`: a list of
    `("expand", shape, swizzled)` or `("payload", destination, rotation, value_table_or_None)`.
    """
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        source, result = Path(tmp) / "cases.pt", Path(tmp) / "results.pt"
        torch.save(cases, source)
        env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
                   PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))))
        completed = subprocess.run([sys.executable, "-m", "tests.recipes.glm53_csf_interpret", str(source), str(result)],
                                   cwd=root, env=env, capture_output=True, text=True, timeout=1800)
        if completed.returncode:
            raise RuntimeError(f"interpreted CSF kernels failed:\n{completed.stdout}\n{completed.stderr}")
        return torch.load(result)


class _CudaLooking:
    """Delegates to a host `CsfSource` but reports a CUDA device, so b12x's launch guard admits it."""

    def __init__(self, source):
        self._source = source

    def __getattr__(self, name):
        return getattr(self._source, name)

    @property
    def device(self):
        from types import SimpleNamespace

        return SimpleNamespace(type="cuda")


def _main(source_path, result_path):
    import contextlib

    assert os.environ.get("TRITON_INTERPRET") == "1"
    from b12x._lib.quant import _csf_prepare as prepare
    from b12x.loader._csf_source import stage_csf_sources

    original = prepare._arguments
    prepare._arguments = lambda source: original(_CudaLooking(source))
    torch.cuda.device = lambda device: contextlib.nullcontext()
    results = []
    for case in torch.load(source_path):
        source = stage_csf_sources(case["components"], rows=case["rows"], columns=case["columns"],
                                   quantization="nvfp4", device="cpu", encoding=case["encoding"],
                                   **case["selection"])
        prepare.validate_source(source)
        outputs = []
        for op in case["ops"]:
            if op[0] == "expand":
                output = torch.full(op[1], 0xA5, dtype=torch.uint8).view(torch.float8_e4m3fn)
                prepare.expand_source_scales(source, output, swizzled=op[2])
                outputs.append(output.view(torch.uint8))
            else:
                _, destination, rotation, table = op
                payload = prepare.prepare_csf_payload(source, destination=destination, rotation=rotation,
                                                      value_table=table)
                outputs.append(dict(storage=payload.storage, rows=payload.rows, columns=payload.columns,
                                    experts=payload.num_experts, destination=payload.destination))
        results.append(outputs)
    torch.save(results, result_path)


if __name__ == "__main__":
    _main(*sys.argv[1:])

# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Padded NVFP4-CSF scale expansion for the three-rank expert shard, on the CPU, for both encodings.

b12x fills a `CsfSource` with `padded_rows`/`padded_columns` in two ways: the Triton expansion
(`_csf_prepare.py::_expand_source`, used when scales expand at load) reads `padding_base` (0) for
every element outside the selected rectangle, while the compact payload (`_pack_codes`) stores one
base per row read at column 0, so a padded column of a covered row decodes to that row's base; at
A16 that base is lifted through the W4A16 table and clamped to 240. Checkpoints carry the scales in
either `csf_32x2_v1` (FP4-Local) or the legacy `csf` slab encoding (LIL's CSF-QAD revisions), and
`_read` has a separate `LEGACY` branch for the latter. The reference tests pin
`expand_csf_scales_padded` on both encodings; the kernel tests run b12x's own Triton kernels in
Triton's CPU interpreter (`glm53_csf_interpret.py`) over GLM's rank-2 geometry for the a4 payload,
the a16 payload (lifted, rotated) and the load-time expansion, on both encodings, with exceptions
inside, at the edge of, and outside the selected rectangle. The GPU test
`test_glm53_weights.py::test_actual_nvfp4_shards_and_shared_scale_storage[3-2-*]` checks the
CUDA path against the same reference.
"""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.glm53.shards import Shards
from ignition.reference.qwen38_quant import decode_csf_scales, decode_csf_slab_scales, expand_csf_scales_padded
from tests.recipes import glm53_csf_interpret as interpret
from tests.recipes.glm53_csf_interpret import ENCODINGS, encode, row_bases

DEC48ABD = Path(os.environ.get("GLM53_CSF_QAD_CHECKPOINT", "/path/to/ckpt/glm53-csf-qad2-dec48abd"))
TEXT = "model.language_model.layers"


def decode(encoding, planes, rows, columns):
    if encoding == "csf_32x2_v1":
        return decode_csf_scales(*planes, rows=rows, columns=columns)
    return decode_csf_slab_scales(*planes, rows=rows, columns=columns)


def synthetic(rows, columns, *, seed, exceptions=()):
    """Positive E4M3 bytes with a per-row palette of sixteen, plus out-of-palette exception entries."""
    generator = torch.Generator().manual_seed(seed)
    bases = torch.randint(40, 100, (rows, 1), dtype=torch.int16, generator=generator)
    scales = bases + torch.randint(0, 16, (rows, columns), dtype=torch.int16, generator=generator)
    for row, column in exceptions:
        scales[row, column] = bases[row, 0] + 16 + (row + column) % 10
    return scales.to(torch.uint8)


# Exceptions inside the rank-2 rectangle (down columns 96..127; up/gate rows 1536..2047), on its edges,
# and just outside it, so a selection that drops or misplaces a record changes a covered value. The 704-column
# shards (FLARE_GLM53_EXPERT_COLUMNS=704) add edges at down groups 44/88 and up/gate rows 704/1408, which split
# csf_32x2_v1's 128-row x 4-group blocks and the slab encoding's 16-row slabs mid-block.
DOWN_EXCEPTIONS = ((5, 97), (200, 127), (31, 96), (32, 95), (255, 100), (10, 20), (128, 96),
                   (3, 44), (4, 87), (6, 88), (9, 43), (130, 45))
UP_EXCEPTIONS = ((1535, 3), (1536, 0), (2047, 15), (1600, 7), (100, 9), (1663, 4),
                 (703, 2), (704, 5), (1407, 1), (1408, 0), (1409, 11))


@pytest.fixture(scope="module", params=ENCODINGS)
def encoded(request):
    rows, columns = 256, 128  # the down projection's row count scaled down; 2048 / 16 scale columns
    scales = synthetic(rows, columns, seed=1, exceptions=DOWN_EXCEPTIONS)
    planes = encode(request.param, scales, DOWN_EXCEPTIONS)
    assert torch.equal(decode(request.param, planes, rows, columns), scales)
    bases = row_bases(request.param, planes)
    assert bases.shape == (rows,) and int(bases.max()) <= 240
    return scales, bases


def test_rank_two_down_columns_pad_to_zero_or_the_row_base(encoded):
    scales, bases = encoded
    shard = TensorParallelShard(2, 3)
    first, last = shard.padded_partition(2048, alignment=128)
    local, end = Shards.of(3).expert_width, min(last, 2048)
    assert (first, last, local, end) == (1536, 2304, 768, 2048)
    column_range, padded = (first // 16, end // 16), local // 16
    assert column_range == (96, 128) and padded == 48
    triton = expand_csf_scales_padded(scales, bases, row_range=(0, 256), column_range=column_range,
                                      padded_columns=padded, compact=False)
    compact = expand_csf_scales_padded(scales, bases, row_range=(0, 256), column_range=column_range,
                                       padded_columns=padded, compact=True)
    for expanded in (triton, compact):
        assert expanded.shape == (256, 48)
        assert torch.equal(expanded[:, :32], scales[:, 96:128])
    assert not triton[:, 32:].count_nonzero()
    assert torch.equal(compact[:, 32:], bases[:, None].expand(256, 16))
    assert compact[:, 32:].count_nonzero() == 256 * 16  # the base of a covered row is never zero here


def test_rank_two_down_columns_at_704_pad_to_zero_or_the_row_base(encoded):
    """FLARE_GLM53_EXPERT_COLUMNS=704: rank 2 holds groups 88..127 of 128 plus 4 padded groups."""
    scales, bases = encoded
    shards = Shards.of(3, expert_columns=704)
    first, last = shards.routed_columns(2)
    assert (first, last, shards.routed_width, shards.expert_width) == (1408, 2112, 704, 768)
    column_range, padded = (first // 16, 2048 // 16), shards.routed_width // 16
    assert column_range == (88, 128) and padded == 44
    triton = expand_csf_scales_padded(scales, bases, row_range=(0, 256), column_range=column_range,
                                      padded_columns=padded, compact=False)
    compact = expand_csf_scales_padded(scales, bases, row_range=(0, 256), column_range=column_range,
                                       padded_columns=padded, compact=True)
    for expanded in (triton, compact):
        assert expanded.shape == (256, 44) and torch.equal(expanded[:, :40], scales[:, 88:128])
    assert not triton[:, 40:].count_nonzero()
    assert torch.equal(compact[:, 40:], bases[:, None].expand(256, 4))


def test_padded_up_gate_rows_are_zero_on_both_paths(encoded):
    scales, bases = encoded
    # Rows 192..255 of 256 stand in for columns 1536..2047 of 2048 with a 768-wide shard (64 covered + 32 pad).
    for compact in (False, True):
        expanded = expand_csf_scales_padded(scales, bases, row_range=(192, 256), column_range=(0, 128),
                                            padded_rows=96, compact=compact)
        assert expanded.shape == (96, 128)
        assert torch.equal(expanded[:64], scales[192:256])
        assert not expanded[64:].count_nonzero()


def test_exact_partitions_expand_without_padding(encoded):
    scales, bases = encoded
    for world in (1, 2, 4):
        shard = TensorParallelShard(world - 1, world)
        first, last = shard.padded_partition(2048, alignment=128)
        assert shard.partition(2048, alignment=128) == (first, last)
        expanded = expand_csf_scales_padded(scales, bases, row_range=(0, 256), column_range=(first // 16, last // 16),
                                            padded_columns=(last - first) // 16, compact=True)
        assert torch.equal(expanded, scales[:, first // 16:last // 16])


def test_reference_rejects_uncovered_rectangles(encoded):
    scales, bases = encoded
    with pytest.raises(ValueError):
        expand_csf_scales_padded(scales, bases, row_range=(0, 256), column_range=(96, 144), padded_columns=48)
    with pytest.raises(ValueError):
        expand_csf_scales_padded(scales, bases, row_range=(0, 256), column_range=(96, 128), padded_columns=16)


def test_legacy_and_csf32_encodings_share_one_decoded_matrix_and_row_bases():
    """The two encodings of the same scales decode identically and store the same per-row bases."""
    scales = synthetic(256, 128, seed=4, exceptions=DOWN_EXCEPTIONS)
    planes = {encoding: encode(encoding, scales, DOWN_EXCEPTIONS) for encoding in ENCODINGS}
    for encoding, value in planes.items():
        assert torch.equal(decode(encoding, value, 256, 128), scales)
    assert torch.equal(row_bases("csf", planes["csf"]), row_bases("csf_32x2_v1", planes["csf_32x2_v1"]))
    assert planes["csf"][0].shape == (256 // 16, 16 * (1 + 128 // 2)) and planes["csf"][1].dtype == torch.uint32
    assert planes["csf"][1].numel() == planes["csf_32x2_v1"][4].numel() == len(DOWN_EXCEPTIONS)


# ---------------------------------------------------------------------------------------------------
# b12x's Triton kernels under the CPU interpreter: a4 payload, a16 payload, load-time expansion.

def _a16_table(expanded):
    """The W4A16 lift table b12x builds from the expanded (zero-padded) scales of one projection."""
    from b12x.moe._shared.kernels.w4a16.prepare import _nvfp4_compute_scale_factor, _process_nvfp4_packed_scales

    factor = _nvfp4_compute_scale_factor(expanded.view(torch.float8_e4m3fn), torch.bfloat16)
    alphabet = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)[:, None].expand(256, 4).contiguous()
    return _process_nvfp4_packed_scales(alphabet.to(torch.bfloat16), scale_factor=factor).view(torch.uint8)[:, 0].contiguous()


def _lift(compact, table, covered_rows, covered_columns, bases):
    """The a16 compact payload: covered bytes through the table, padded columns the clamped lifted base."""
    lifted = torch.full_like(compact, min(int(table[0]), 240))
    lifted[:covered_rows, :covered_columns] = table[compact[:covered_rows, :covered_columns].long()]
    lifted[:covered_rows, covered_columns:] = table[bases.long()].clamp_max(240)[:, None]
    return lifted


def _projection(encoding, rows, columns, *, seed, exceptions, high_base_row=None):
    scales = synthetic(rows, columns, seed=seed, exceptions=exceptions)
    if high_base_row is not None:
        scales[high_base_row] = 126  # E4M3 448: its lifted base exceeds 240 and must clamp
    planes = encode(encoding, scales, exceptions)
    assert torch.equal(decode(encoding, planes, rows, columns), scales)
    return scales, planes, row_bases(encoding, planes)


def _down_case(encoding, *, column_range, padded_columns, seed=10):
    """Down projections of two experts: [256 hidden rows, 128 groups], columns selected (and padded)."""
    experts = [_projection(encoding, 256, 128, seed=seed + e, exceptions=DOWN_EXCEPTIONS, high_base_row=7 + e)
               for e in range(2)]
    c0, c1 = column_range
    triton = torch.stack([expand_csf_scales_padded(s, b, row_range=(0, 256), column_range=column_range,
                                                   padded_columns=padded_columns) for s, _, b in experts])
    compact = torch.stack([expand_csf_scales_padded(s, b, row_range=(0, 256), column_range=column_range,
                                                    padded_columns=padded_columns, compact=True) for s, _, b in experts])
    table = _a16_table(triton)
    lifted = torch.stack([_lift(compact[e], table, 256, c1 - c0, experts[e][2]) for e in range(2)])
    case = dict(encoding=encoding, components=[p for _, p, _ in experts], rows=256, columns=128,
                selection=dict(column_range=column_range,
                               **({} if padded_columns == c1 - c0 else {"padded_columns": padded_columns})),
                ops=[("expand", (2, 256, padded_columns), True), ("expand", (2, 256, padded_columns), False),
                     ("payload", 0, 0, None), ("payload", 1, 0, table)])
    return case, dict(triton=triton, compact=compact, lifted=lifted, rotation=0, covered=c1 - c0)


def _up_gate_case(encoding, *, row_range, padded_rows, seed=20):
    """Up and gate projections of two experts, joined: [2048 intermediate rows, 16 groups], rows selected."""
    planes, triton, compact, lifted_parts, bases_parts = [], [], [], [], []
    r0, r1 = row_range
    projections = [[_projection(encoding, 2048, 16, seed=seed + 2 * e + p, exceptions=UP_EXCEPTIONS)
                    for p in range(2)] for e in range(2)]  # up, gate per expert
    for expert in projections:
        tri, com = [], []
        for scales, encoded_planes, bases in expert:
            planes.append(encoded_planes)
            tri.append(expand_csf_scales_padded(scales, bases, row_range=row_range, column_range=(0, 16),
                                                padded_rows=padded_rows))
            com.append(expand_csf_scales_padded(scales, bases, row_range=row_range, column_range=(0, 16),
                                                padded_rows=padded_rows, compact=True))
            bases_parts.append(bases[r0:r1])
        triton.append(torch.cat(tri))
        compact.append(torch.cat(com))
    triton, compact = torch.stack(triton), torch.stack(compact)
    table = _a16_table(triton)
    for e in range(2):
        halves = [_lift(compact[e, h * padded_rows:(h + 1) * padded_rows], table, r1 - r0, 16, bases_parts[2 * e + h])
                  for h in range(2)]
        lifted_parts.append(torch.cat(halves))
    rows = 2 * padded_rows
    case = dict(encoding=encoding, components=planes, rows=2048, columns=16,
                selection=dict(join=2, row_range=row_range,
                               **({} if padded_rows == r1 - r0 else {"padded_rows": padded_rows})),
                ops=[("expand", (2, rows, 16), True), ("expand", (2, rows, 16), False),
                     ("payload", 0, 0, None), ("payload", 1, padded_rows, table)])
    return case, dict(triton=triton, compact=compact, lifted=torch.stack(lifted_parts), rotation=padded_rows)


GEOMETRIES = {
    # GLM rank 2 of 3: down columns 1536..2047 of 2048 (groups 96..127) padded to 768 (48 groups).
    "down-rank2-padded": lambda encoding: _down_case(encoding, column_range=(96, 128), padded_columns=48),
    # GLM rank 2 of 3: up/gate rows 1536..2047 padded to 768 each, joined (gate follows up).
    "up_gate-rank2-padded": lambda encoding: _up_gate_case(encoding, row_range=(1536, 2048), padded_rows=768),
    # Rank 0 of 3 is exact: 768 columns / rows from the origin.
    "down-rank0-exact": lambda encoding: _down_case(encoding, column_range=(0, 48), padded_columns=48),
    "up_gate-rank0-exact": lambda encoding: _up_gate_case(encoding, row_range=(0, 768), padded_rows=768),
    # FLARE_GLM53_EXPERT_COLUMNS=704: rank 1 exact (columns 704..1407, groups 44..87), rank 2 columns 1408..2047
    # (groups 88..127) padded to 704 (44 groups: 40 covered + 4 pad).
    "down-rank1-exact-704": lambda encoding: _down_case(encoding, column_range=(44, 88), padded_columns=44),
    "down-rank2-padded-704": lambda encoding: _down_case(encoding, column_range=(88, 128), padded_columns=44),
    "up_gate-rank1-exact-704": lambda encoding: _up_gate_case(encoding, row_range=(704, 1408), padded_rows=704),
    "up_gate-rank2-padded-704": lambda encoding: _up_gate_case(encoding, row_range=(1408, 2048), padded_rows=704),
}


@pytest.fixture(scope="module")
def interpreted():
    """Every geometry on both encodings through the interpreted kernels, in one subprocess."""
    keys = [(name, encoding) for name in GEOMETRIES for encoding in ENCODINGS]
    built = {key: GEOMETRIES[key[0]](key[1]) for key in keys}
    results = interpret.run([built[key][0] for key in keys])
    return {key: (built[key][1], result) for key, result in zip(keys, results, strict=True)}


def _payload(result):
    return interpret.decode_payload(result["storage"], rows=result["rows"], columns=result["columns"],
                                    experts=result["experts"], destination=result["destination"])


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_expanded_scales_pad_to_zero(interpreted, geometry, encoding):
    """Load-time expansion (`expand_expert_scales`): the rectangle at the origin, zero padding, both layouts."""
    from b12x._lib.intrinsics.reference import swizzle_block_scale

    expected, (swizzled, plain, _, _) = interpreted[(geometry, encoding)]
    assert torch.equal(plain, expected["triton"])
    assert torch.equal(swizzled.reshape(-1), swizzle_block_scale(expected["triton"]).reshape(-1))


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_a4_compact_payload_pads_columns_with_the_row_base(interpreted, geometry, encoding):
    """A4 scratch path: covered bytes exact (exceptions included), padded columns the row base, padded rows zero."""
    expected, (_, _, a4, _) = interpreted[(geometry, encoding)]
    assert a4["destination"] == 0
    assert torch.equal(_payload(a4), expected["compact"])


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_a16_compact_payload_lifts_and_clamps_row_bases(interpreted, geometry, encoding):
    """A16 scratch path: MMA-packed rows, gate-first rotation for up/gate, bytes lifted, padded bases <= 240."""
    expected, (_, _, _, a16) = interpreted[(geometry, encoding)]
    assert a16["destination"] == 1
    decoded = _payload(a16)
    assert torch.equal(decoded, expected["lifted"].roll(-expected["rotation"], dims=1))
    if geometry.startswith("down") and "padded" in geometry:
        covered = expected["covered"]
        assert int(decoded[:, :, covered:].max()) <= 240 and int(decoded[0, 7, covered]) == 240


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_both_encodings_prepare_identical_scales(interpreted, geometry):
    """For the same decoded scales the legacy and csf32 sources yield byte-identical expansions and payload values."""
    _, legacy = interpreted[(geometry, "csf")]
    _, csf32 = interpreted[(geometry, "csf_32x2_v1")]
    for index in (0, 1):
        if not torch.equal(legacy[index], csf32[index]):
            # Same seeds produce the same scales, so any difference is an encoding-dependent read.
            pytest.fail(f"{geometry}: expansion {index} differs between encodings")
    for index in (2, 3):
        assert torch.equal(_payload(legacy[index]), _payload(csf32[index]))


# ---------------------------------------------------------------------------------------------------
# One real expert of LIL's CSF-QAD checkpoint (rev dec48abd, legacy `csf`): headers plus three small
# scale tensors per projection (about 0.8 MB), through the same interpreted kernels at full geometry.

def _real_scale_planes(names):
    import json

    from safetensors import safe_open

    index = json.loads((DEC48ABD / "model.safetensors.index.json").read_text())["weight_map"]
    planes = {}
    for name in names:
        fixed, exceptions = (f"{name}.weight_scale.nvfp4_csf_{field}" for field in ("fixed", "exceptions"))
        assert index[fixed] == index[exceptions]
        with safe_open(str(DEC48ABD / index[fixed]), framework="pt") as handle:
            planes[name] = (handle.get_tensor(fixed).contiguous(), handle.get_tensor(exceptions).contiguous())
    return planes


@pytest.mark.skipif(not (DEC48ABD / "config.json").exists(), reason="dec48abd checkpoint not on this host")
def test_dec48abd_legacy_scales_through_the_padded_rank_two_kernels():
    import json

    config = json.loads((DEC48ABD / "config.json").read_text())
    layers = config["quantization_config"]["quantized_layers"]
    assert layers[f"{TEXT}.3.mlp.experts"] == {"group_size": 16, "quant_algo": "NVFP4", "weight_scale_encoding": "csf"}
    prefix = f"{TEXT}.3.mlp.experts.0"
    planes = _real_scale_planes([f"{prefix}.{p}" for p in ("up_proj", "gate_proj", "down_proj")])
    geometry = {"up_proj": (2048, 256), "gate_proj": (2048, 256), "down_proj": (4096, 128)}
    decoded, bases = {}, {}
    for name, (fixed, exceptions) in planes.items():
        rows, columns = geometry[name.rsplit(".", 1)[1]]
        assert fixed.dtype == torch.uint8 and fixed.shape == (rows // 16, 16 * (1 + columns // 2))
        assert exceptions.dtype == torch.uint32 and exceptions.ndim == 1
        decoded[name] = decode_csf_slab_scales(fixed, exceptions, rows=rows, columns=columns)
        bases[name] = row_bases("csf", (fixed, exceptions))
        assert int(bases[name].max()) <= 240  # b12x's validator bound for 4-bit NVFP4 palettes
    up, gate, down = (f"{prefix}.{p}" for p in ("up_proj", "gate_proj", "down_proj"))
    # LIL's encoder does not always pick the row minimum as the base: values below it are exceptions, and the
    # compact payload pads with the stored base, not the minimum (the reference takes the stored plane).
    assert any(bool((decoded[n].min(dim=1).values != bases[n]).any()) for n in decoded)
    down_expected = expand_csf_scales_padded(decoded[down], bases[down], row_range=(0, 4096), column_range=(96, 128),
                                             padded_columns=48)
    down_compact = expand_csf_scales_padded(decoded[down], bases[down], row_range=(0, 4096), column_range=(96, 128),
                                            padded_columns=48, compact=True)
    up_gate = [expand_csf_scales_padded(decoded[n], bases[n], row_range=(1536, 2048), column_range=(0, 256),
                                        padded_rows=768, compact=c) for c in (False, True) for n in (up, gate)]
    up_gate_expected, up_gate_compact = torch.cat(up_gate[:2]), torch.cat(up_gate[2:])
    tables = _a16_table(down_expected[None]), _a16_table(up_gate_expected[None])
    cases = [dict(encoding="csf", components=[planes[down]], rows=4096, columns=128,
                  selection=dict(column_range=(96, 128), padded_columns=48),
                  ops=[("expand", (1, 4096, 48), True), ("payload", 0, 0, None), ("payload", 1, 0, tables[0])]),
             dict(encoding="csf", components=[planes[up], planes[gate]], rows=2048, columns=256,
                  selection=dict(join=2, row_range=(1536, 2048), padded_rows=768),
                  ops=[("expand", (1, 1536, 256), True), ("payload", 0, 0, None), ("payload", 1, 768, tables[1])])]
    from b12x._lib.intrinsics.reference import swizzle_block_scale

    results = interpret.run(cases)
    lifted_down = _lift(down_compact, tables[0], 4096, 32, bases[down])
    lifted_up_gate = torch.cat([_lift(up_gate_compact[h * 768:(h + 1) * 768], tables[1], 512, 256,
                                      bases[name][1536:2048]) for h, name in enumerate((up, gate))])
    (down_expand, down_a4, down_a16), (ug_expand, ug_a4, ug_a16) = results
    assert torch.equal(down_expand.reshape(-1), swizzle_block_scale(down_expected).reshape(-1))
    assert torch.equal(_payload(down_a4)[0], down_compact)
    assert torch.equal(_payload(down_a16)[0], lifted_down)
    assert torch.equal(ug_expand.reshape(-1), swizzle_block_scale(up_gate_expected).reshape(-1))
    assert torch.equal(_payload(ug_a4)[0], up_gate_compact)
    assert torch.equal(_payload(ug_a16)[0], lifted_up_gate.roll(-768, dims=0))


@pytest.mark.parametrize("local,covered", [(768, 512), (704, 640)], ids=["768", "704"])
def test_a16_layout_helpers_round_trip_padded_banks(local, covered):
    """The W4A16 inverse recovers padded zero rows/columns, so the GPU comparison reads logical bytes."""
    from b12x.moe._shared.kernels.w4a16.prepare import _repack_weight

    from tests.recipes.glm53_a16_layout import a16_logical_weights, a16_packed_scales

    torch.manual_seed(2)
    hidden = 256  # GLM rank 2 of 3 (768 or 704 columns) with a narrow hidden width
    w13 = torch.zeros((2, 2 * local, hidden // 2), dtype=torch.uint8)
    w13[:, :covered] = torch.randint(1, 256, (2, covered, hidden // 2), dtype=torch.uint8)
    w13[:, local:local + covered] = torch.randint(1, 256, (2, covered, hidden // 2), dtype=torch.uint8)
    packed = _repack_weight(w13.clone(), size_k=hidden, size_n=2 * local, row_rotation=local)
    assert packed.dtype == torch.int32 and packed.shape == (2, hidden // 16, (2 * local // 64) * 128)
    for index in range(2):
        logical = a16_logical_weights(packed[index], size_k=hidden, size_n=2 * local, row_rotation=local)
        assert torch.equal(logical, w13[index])
        assert not logical[covered:local].count_nonzero() and not logical[local + covered:].count_nonzero()
    w2 = torch.zeros((2, hidden, local // 2), dtype=torch.uint8)
    w2[:, :, :covered // 2] = torch.randint(1, 256, (2, hidden, covered // 2), dtype=torch.uint8)
    packed2 = _repack_weight(w2.clone(), size_k=local, size_n=hidden)
    for index in range(2):
        logical = a16_logical_weights(packed2[index], size_k=local, size_n=hidden)
        assert torch.equal(logical, w2[index]) and not logical[:, covered // 2:].count_nonzero()
    # Permuted scales: zero logical pads stay zero, and the transform is deterministic for the batch.
    scales = torch.zeros((2, 2 * local, hidden // 16), dtype=torch.uint8)
    scales[:, :covered] = torch.randint(64, 120, (2, covered, hidden // 16), dtype=torch.uint8)
    scales[:, local:local + covered] = torch.randint(64, 120, (2, covered, hidden // 16), dtype=torch.uint8)
    packed_scales = a16_packed_scales(scales, torch.ones(2), size_k=hidden, size_n=2 * local,
                                      io_dtype=torch.bfloat16, row_rotation=local)
    assert packed_scales.shape == (2, hidden // 16, 2 * local)  # one E4M3 byte per logical scale, permuted
    assert packed_scales.view(torch.uint8).count_nonzero() == 2 * covered * 2 * (hidden // 16)
    again = a16_packed_scales(scales, torch.ones(2), size_k=hidden, size_n=2 * local, io_dtype=torch.bfloat16,
                              row_rotation=local)
    assert torch.equal(packed_scales.view(torch.uint8), again.view(torch.uint8))


@pytest.mark.parametrize("local", [768, 704])
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_a16_compact_pad_columns_are_lifted_clamped_row_bases(encoding, local):
    """The shared-scratch decoder at A16 fills padded K groups with lifted row bases clamped to 240."""
    from b12x.moe._shared.kernels.w4a16.prepare import _nvfp4_compute_scale_factor, _process_nvfp4_packed_scales

    from tests.recipes.glm53_a16_layout import a16_packed_scales

    # Positive E4M3 scales only, as a checkpoint holds: [256 hidden rows, 128 K groups]; rank 2 of 3 keeps
    # groups 96..127 plus 16 pads. Row 7's base is the E4M3 maximum (448) to exercise the 240 clamp.
    torch.manual_seed(3)
    bases = torch.randint(64, 100, (256,), dtype=torch.int16)
    scales = (bases[:, None] + torch.randint(0, 16, (256, 128), dtype=torch.int16)).to(torch.uint8)
    scales[7] = 126
    planes = encode(encoding, scales)
    assert torch.equal(decode(encoding, planes, 256, 128), scales)
    bases = row_bases(encoding, planes)
    assert int(bases[7]) == 126
    groups, first = local // 16, 2 * local // 16  # rank 2 of 3 starts at column 2 x local
    covered = 128 - first
    logical = expand_csf_scales_padded(scales, bases, row_range=(0, 256), column_range=(first, 128),
                                       padded_columns=groups, compact=True)[None]
    grid = a16_packed_scales(logical, torch.ones(1), size_k=local, size_n=256, io_dtype=torch.bfloat16,
                             pad_groups=(covered, groups)).view(torch.uint8)[0]
    assert grid.shape == (groups, 256)
    factor = _nvfp4_compute_scale_factor(logical.view(torch.float8_e4m3fn), torch.bfloat16)
    alphabet = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)[:, None].expand(256, 4).contiguous()
    table = _process_nvfp4_packed_scales(alphabet.to(torch.bfloat16), scale_factor=factor).view(torch.uint8)[:, 0]
    lifted_bases = table[bases.long()].clamp_max(240)
    for row in range(covered, groups):
        assert sorted(grid[row].tolist()) == sorted(lifted_bases.tolist())  # permuted along N, values intact
    assert int(table[126]) > 240 and int(lifted_bases[7]) == 240
    assert not grid[covered:, :].eq(0).any()
    plain = a16_packed_scales(logical, torch.ones(1), size_k=local, size_n=256, io_dtype=torch.bfloat16).view(torch.uint8)[0]
    assert torch.equal(plain[:covered], grid[:covered]) and (plain[covered:] != grid[covered:]).sum() == groups - covered

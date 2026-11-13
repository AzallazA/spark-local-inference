# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fork-only GLM TP3 decode-speed options: KDA side-chain scratch layout, draft head and eh_proj precision."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.glm53 import kda
from ignition.recipes.glm53.bootstrap import PreparationConfig
from ignition.recipes.glm53.kda import KDAScratch
from ignition.recipes.glm53.recipe import _tp3_override
from ignition.recipes.glm53.shards import Shards


def test_options_default_to_upstream_and_are_validated(monkeypatch):
    config = PreparationConfig(devices=(0, 1, 2))
    assert (config.kda_side_chain, config.draft_head, config.draft_eh_proj) == (False, "target", "bf16")
    PreparationConfig(devices=(0, 1, 2), kda_side_chain=True, draft_head="nvfp4", draft_eh_proj="mxfp8")
    with pytest.raises(TypeError):
        PreparationConfig(devices=(0, 1, 2), kda_side_chain=1)
    for field, value in (("draft_head", "mxfp8"), ("draft_eh_proj", "nvfp4")):
        with pytest.raises(ValueError, match="draft"):
            PreparationConfig(devices=(0, 1, 2), **{field: value})
    monkeypatch.setenv("FLARE_GLM53_DRAFT_HEAD", "NVFP4")
    assert _tp3_override("DRAFT_HEAD", ("target", "nvfp4"), "target") == "nvfp4"
    monkeypatch.setenv("FLARE_GLM53_KDA_SIDE_CHAIN", "2")
    with pytest.raises(ValueError):
        _tp3_override("KDA_SIDE_CHAIN", ("1", "0"), "0")


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("tokens", [1, 5, 65])
def test_side_chain_scratch_is_contiguous_and_the_same_size(world, tokens):
    heads = Shards.of(world).heads
    packed = KDAScratch.allocate(tokens, heads, "cpu", split=False)
    split = KDAScratch.allocate(tokens, heads, "cpu", split=True)
    assert not packed.split and split.split
    width = 3 * heads * 128
    assert split.projected_qkv.shape == (tokens, width) and split.projected_qkv.is_contiguous()
    assert split.projected_input.data_ptr() == split.projected_qkv.data_ptr()
    assert split.raw_beta.shape == (tokens, heads) and split.raw_beta.is_contiguous()
    assert split.forget_input.shape == (tokens, 128) and split.forget_input.is_contiguous()
    # Negative control: the packed layout's q/k/v rows are strided for more than one token.
    assert packed.projected_qkv.is_contiguous() is (tokens == 1)

    def nbytes(scratch):
        seen, total = set(), 0
        for value in vars(scratch).values():
            if isinstance(value, torch.Tensor) and value.untyped_storage().data_ptr() not in seen:
                seen.add(value.untyped_storage().data_ptr())
                total += value.untyped_storage().nbytes()
        return total
    # Same bytes per row as the packed layout, which above one token also owns a copied forget buffer.
    assert nbytes(split) == nbytes(packed) - (0 if tokens == 1 else 128 * 2 * tokens)
    for rows in (1, tokens):
        view = split.rows(rows)
        assert view.split and view.forget_input.data_ptr() == split.forget_input.data_ptr()
        assert view.raw_beta.shape == (rows, heads) and view.projected_qkv.is_contiguous()
    assert packed.rows(1).forget_input.data_ptr() == packed.projected_input[:, -128:].data_ptr()


def test_scratch_follows_the_configured_schedule():
    try:
        kda.configure(side_chain=True)
        assert KDAScratch.allocate(3, 24, "cpu").split
        kda.configure(side_chain=False)
        assert not KDAScratch.allocate(3, 24, "cpu").split
        with pytest.raises(TypeError):
            kda.configure(side_chain="1")
    finally:
        kda.configure(side_chain=False)


def _mxfp8(rows, columns, generator):
    """A packed-MXFP8-shaped weight: E4M3 values, (1, rows, columns / 32) E8M0 scales."""
    values = (torch.randn((rows, columns), generator=generator) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    scales = torch.randint(110, 132, (1, rows, columns // 32), generator=generator, dtype=torch.uint8)
    return SimpleNamespace(weight=SimpleNamespace(values=values, scale_rows=scales),
                           in_features=columns, out_features=rows)


def test_mxfp8_rows_decode_exactly_to_bf16():
    from ignition.components.vocabulary import decode_mxfp8_rows

    weight = _mxfp8(192, 256, torch.Generator().manual_seed(5))
    expected = (weight.weight.values.to(torch.float64).view(192, 8, 32)
                * torch.exp2(weight.weight.scale_rows[0].to(torch.float64) - 127).unsqueeze(-1)).view(192, 256)
    for start, stop in ((0, 192), (64, 128), (191, 192)):
        decoded = decode_mxfp8_rows(weight, start, stop)
        assert decoded.dtype == torch.bfloat16
        torch.testing.assert_close(decoded.to(torch.float64), expected[start:stop], rtol=0, atol=0)
    # Negative control: a scale off by one exponent is caught.
    weight.weight.scale_rows[0, 7, 3] += 1
    assert not torch.equal(decode_mxfp8_rows(weight, 0, 192).to(torch.float64), expected)


def test_nvfp4_head_from_mxfp8_matches_the_bf16_construction(monkeypatch):
    import b12x._lib.intrinsics.reference as reference
    import b12x.gemm.blockscaled as blockscaled

    from ignition.components.vocabulary import decode_mxfp8_rows, nvfp4_draft_head, nvfp4_draft_head_from_mxfp8

    # CPU: capture what would be packed instead of packing.
    monkeypatch.setattr(reference, "swizzle_block_scale", lambda scales: scales.clone())
    monkeypatch.setattr(blockscaled, "pack_weight",
                        lambda values, scales, **kwargs: (values.clone(), scales.clone(), kwargs["global_scale"]))
    weight = _mxfp8(320, 256, torch.Generator().manual_seed(9))
    reference_head = nvfp4_draft_head(decode_mxfp8_rows(weight, 0, 320).contiguous(), rows_per_chunk=128)
    for chunk in (64, 100, 4096):
        candidate = nvfp4_draft_head_from_mxfp8(weight, rows_per_chunk=chunk)
        for actual, expected in zip(candidate, reference_head, strict=True):
            assert torch.equal(actual.view(torch.uint8) if actual.dtype == torch.float8_e4m3fn else actual,
                               expected.view(torch.uint8) if expected.dtype == torch.float8_e4m3fn else expected)
    # Negative control: a different weight gives different codes.
    weight.weight.values[0, :32] = 0
    assert not torch.equal(nvfp4_draft_head_from_mxfp8(weight)[0], reference_head[0])

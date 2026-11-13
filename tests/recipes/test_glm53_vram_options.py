# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fork-only GLM VRAM switches (2026-11-10): FLARE_GLM53_EMBED_HOST parks the token embedding in host memory.

CPU checks cover the switch's parsing, its plumbing from the export environment into PreparationConfig, and
the parking contract. The CUDA check (FLARE_RUN_CUDA_TESTS=1, one GPU, no checkpoint) parks a table through a
pinned host allocation and gathers rows from it on the device:

    FLARE_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \\
        tests/recipes/test_glm53_vram_options.py
"""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.glm53 import recipe as glm_recipe
from ignition.recipes.glm53.bootstrap import PreparationConfig
from ignition.recipes.glm53.weights import park_embedding

_CUDA = os.environ.get("FLARE_RUN_CUDA_TESTS") == "1"


def test_embed_host_defaults_off_and_is_validated(monkeypatch):
    assert PreparationConfig(devices=(0, 1)).embed_host is False
    assert PreparationConfig(devices=(0, 1), embed_host=True).embed_host is True
    with pytest.raises(TypeError, match="embed_host"):
        PreparationConfig(devices=(0, 1), embed_host=1)
    monkeypatch.delenv("FLARE_GLM53_EMBED_HOST", raising=False)
    assert glm_recipe._tp3_override("EMBED_HOST", ("1", "0"), "0") == "0"
    monkeypatch.setenv("FLARE_GLM53_EMBED_HOST", "yes")
    with pytest.raises(ValueError, match="EMBED_HOST"):
        glm_recipe._tp3_override("EMBED_HOST", ("1", "0"), "0")


def _plan(world):
    """The plan fields recipe.prepare reads, at the export script's settings (concurrency 8, chunk 4096)."""
    return SimpleNamespace(
        speculation=SimpleNamespace(widths=(1, 2, 3, 4, 5, 6), fixed_width=None, mtp_only=False), speculative_token_capacity=None,
        collective_tuning="static", collective_ladder=None, prompt_logprobs=False, compile_workers=None,
        autotune_effort="default", tuning_replica=None, native_dense_activations=False, lock_gpu_clocks=False,
        world_size=world, gpus=(SimpleNamespace(capability=(12, 0)),) * world, image_concurrency=None,
        language_model_only=True, concurrency=8, roce_ring=False, roce_chain=False, context_parallel=world != 3,
        stream_overlap=True, recipe=SimpleNamespace(context=1048576), prefill_chunk=4096, minimum_pages=None,
        snapshots=1, costs=SimpleNamespace(state_row_bytes=1 << 20), headroom_bytes=0, reserve_bytes=0,
        checkpoint=None, compile_headroom_bytes=8 << 30)


@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_export_environment_reaches_the_preparation_config(monkeypatch, world, value, expected):
    import ignition.preparation.rank_group as rank_group

    for name in ("EXPERT_ACTIVATION", "EXPAND_SCALES", "EXPERT_COLUMNS", "KDA_SIDE_CHAIN", "DRAFT_HEAD",
                 "DRAFT_EH_PROJ", "EMBED_HOST"):
        monkeypatch.delenv(f"FLARE_GLM53_{name}", raising=False)
    if value is not None:
        monkeypatch.setenv("FLARE_GLM53_EMBED_HOST", value)
    seen = {}
    monkeypatch.setattr(rank_group, "prepare_group",
                        lambda snapshot, config, function, **kwargs: seen.setdefault("config", config))
    glm_recipe.prepare("/nonexistent", _plan(world), compile_forwards=False, kernel_cache=None, progress=None)
    config = seen["config"]
    assert config.embed_host is expected
    # The switch changes nothing else the export prepares.
    assert (config.kda_side_chain, config.draft_head, config.draft_eh_proj, config.expert_columns) == (
        False, "target", "bf16", None)


def _tree(weight):
    embed = torch.nn.Embedding(*weight.shape, device="meta")
    embed.weight = torch.nn.Parameter(weight, requires_grad=False)
    return SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=embed)))


def test_parking_requires_the_loaded_cuda_table():
    tree = _tree(torch.zeros((8, 4), dtype=torch.bfloat16))
    calls = []
    with pytest.raises(ValueError, match="CUDA"):
        park_embedding(tree, lambda *args: calls.append(args))
    assert not calls


class _PinnedHost:
    """Stand-in for PackageStorage.host_allocate: pinned host pages viewed as a CUDA tensor (UVA)."""

    def __init__(self, device):
        self.device, self.regions = device, []

    def __call__(self, name, shape, dtype):
        from b12x.sequence._shared.disk_table import _tensor_from_pointer

        host = torch.empty(shape, dtype=dtype, pin_memory=True)
        view = _tensor_from_pointer(host.data_ptr(), shape=tuple(shape), dtype=dtype, device=self.device,
                                    nbytes=host.nbytes)
        self.regions.append((name, host))
        return SimpleNamespace(host_view=host, device_view=view)


@pytest.mark.skipif(not _CUDA, reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")
def test_parked_table_is_host_resident_bitwise_and_gatherable():
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    # GLM's geometry: 154880 x 4096 BF16, 1210 MiB.
    table = torch.randn((154880, 4096), generator=generator, device=device).to(torch.bfloat16)
    reference = table[torch.tensor([0, 1, 154879, 77440], device=device)].clone()
    tree = _tree(table)
    del table
    allocator = _PinnedHost(device)
    before = torch.cuda.memory_allocated(device)
    parked = park_embedding(tree, allocator)
    weight = tree.model.language_model.embed_tokens.weight
    assert weight.data_ptr() == parked.data_ptr() == allocator.regions[0][1].data_ptr()
    assert allocator.regions[0][0] == "model.language_model.embed_tokens.weight"
    assert weight.device == device and weight.shape == (154880, 4096) and weight.dtype == torch.bfloat16
    # The device table is released: only the parked host view remains referenced.
    assert before - torch.cuda.memory_allocated(device) >= 154880 * 4096 * 2
    ids = torch.tensor([0, 1, 154879, 77440], device=device)
    assert torch.equal(weight[ids], reference)
    # A decode-sized gather through the b12x embedding kernel reads the host rows (DeepSeek's PreparedEmbedding).
    from b12x.sequence import embedding

    output = torch.empty((4, 4096), dtype=torch.bfloat16, device=device)
    plan = embedding.plan(embedding.query_from_call(weight, ids, out=output), device=device)
    embedding.run(weight, ids, plan=plan, out=output)
    torch.cuda.synchronize(device)
    assert torch.equal(output, reference)

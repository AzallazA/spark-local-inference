# SPDX-License-Identifier: Apache-2.0
"""TP3 fork configuration: environment parsing, defaults, validation and the metadata record (CPU only)."""

from pathlib import Path

import pytest

from ignition.recipes.deepseek41.tp3_config import (TP3Config, describe_checkpoint, describe_counts_dir,
                                                    format_layers, parse_layers)

# The release export.sh (DeepSeek-V4.1-Flash Flare-tp3-20261107) environment with its defaults expanded
# (HOT=342, COUNTS=$HERE/expert-order, FLARE_DS41_VISION_HOST=1, FLARE_DS41_VISION_STAGE=1).
RELEASE_ENV = {
    "FLARE_DS41_TP3_GRAPH_GIB": "1",
    "FLARE_DS41_HOST_EXPERT_LAYERS": "20-33",
    "FLARE_DS41_HOT_EXPERTS": "342",
    "FLARE_DS41_MERGED_BANK": "1",
    "FLARE_DS41_TP3_VOCAB_PARALLEL": "1",
    "FLARE_DS41_EXPERT_COUNTS_DIR": "/release/expert-order",
    "HF_HUB_OFFLINE": "1",
    "FLARE_DS41_BRRR_LAYERS": "0-39",
    "FLARE_DS41_BRRR_CKPT": "/m2-2/flare-ckpt/ds41-1017-engram-mxfp4",
    "FLARE_DS41_BRRR_PARKED_MAX_ROWS": "12",
    "FLARE_DS41_EMBED_HOST": "1",
    "FLARE_DS41_VISION_HOST": "1",
    "FLARE_DS41_VISION_STAGE": "1",
    "FLARE_DS41_AR_ROWS": "1024",
}

RELEASE_CONFIG = TP3Config(
    host_expert_layers=frozenset(range(20, 34)), hot_experts=342, merged_bank=True,
    expert_counts_dir=Path("/release/expert-order"), embed_host=True, vision_host=True, vision_stage=True,
    ar_rows=1024, tp3_graph_gib=1.0, tp3_vocab_parallel=True,
    brrr_layers=frozenset(range(40)), brrr_parked_max_rows=12,
    brrr_ckpt=Path("/m2-2/flare-ckpt/ds41-1017-engram-mxfp4"))


def test_release_environment_parses_to_the_expected_config():
    assert TP3Config.from_environment(RELEASE_ENV) == RELEASE_CONFIG


def test_release_environment_does_not_touch_the_unset_switches():
    config = TP3Config.from_environment(RELEASE_ENV)
    assert config.global_cold == 0 and config.vision_stage_copy == "memcpy" and config.joint_headroom_mib is None
    assert config.brrr_dense_native and config.fused_attn_per_chunk
    assert not (config.brrr_native_scales or config.brrr_csf_check or config.brrr_log or config.brrr_sparse
                or config.fused_qkv_norm or config.mini or config.no_engram or config.probe)
    assert config.brrr_break is None


def test_defaults_are_the_bare_upstream_behaviour():
    config = TP3Config.from_environment({})
    assert config == TP3Config()
    assert config.host_expert_layers == frozenset() and config.brrr_layers == frozenset()
    assert config.hot_experts == 0 and config.ar_rows == 0 and config.tp3_graph_gib is None
    assert config.expert_counts_dir is None and config.brrr_ckpt is None
    assert config.brrr_dense_native is True and config.fused_attn_per_chunk is True
    assert config.describe() == "TP3 config: defaults"


def test_empty_values_count_as_unset_and_or_zero_integers():
    env = {key: "" for key in RELEASE_ENV}
    assert TP3Config.from_environment(env) == TP3Config()
    assert TP3Config.from_environment({"FLARE_DS41_HOT_EXPERTS": "", "FLARE_DS41_AR_ROWS": ""}).hot_experts == 0
    # Two switches keep the base parsers' set-but-empty meaning instead (documented on the fields).
    assert TP3Config.from_environment({"FLARE_DS41_BRRR_DENSE_NATIVE": "", "FLARE_DS41_BRRR_LAYERS": "1",
                                       "FLARE_DS41_BRRR_CKPT": "/c"}).brrr_dense_native is False
    assert TP3Config.from_environment({"FLARE_DS41_FUSED_ATTN_PER_CHUNK": ""}).fused_attn_per_chunk is False
    assert TP3Config.from_environment({"FLARE_DS41_VISION_STAGE_COPY": ""}).vision_stage_copy == "memcpy"


def test_layer_specs_accept_ranges_and_comma_lists():
    assert parse_layers("20-22") == frozenset({20, 21, 22})
    assert parse_layers("20,21") == frozenset({20, 21})
    assert parse_layers(" 0-1, 7 ,9-9") == frozenset({0, 1, 7, 9})
    assert parse_layers("") == frozenset()
    assert format_layers({0, 1, 2, 7, 9, 10}) == "0-2,7,9-10"
    assert parse_layers(format_layers(range(20, 34))) == frozenset(range(20, 34))
    config = TP3Config.from_environment({"FLARE_DS41_HOST_EXPERT_LAYERS": "20,22-23"})
    assert config.host_expert_layers == frozenset({20, 22, 23})


def test_flag_semantics_match_the_historical_parsers():
    # "1" flags: only "1" switches them on.
    assert TP3Config.from_environment({"FLARE_DS41_MERGED_BANK": "1"}).merged_bank
    assert not TP3Config.from_environment({"FLARE_DS41_MERGED_BANK": "yes"}).merged_bank
    assert not TP3Config.from_environment({"FLARE_DS41_EMBED_HOST": "0"}).embed_host
    # "not in ('', '0')" switches: any other value is on.
    assert TP3Config.from_environment({"FLARE_DS41_TP3_VOCAB_PARALLEL": "yes"}).tp3_vocab_parallel
    assert not TP3Config.from_environment({"FLARE_DS41_TP3_VOCAB_PARALLEL": "0"}).tp3_vocab_parallel
    assert TP3Config.from_environment({"FLARE_DS41_NO_ENGRAM": "2"}).no_engram
    assert TP3Config.from_environment({"FLARE_DS41_PROBE": "1"}).probe
    # Default-on switches.
    assert not TP3Config.from_environment({"FLARE_DS41_BRRR_DENSE_NATIVE": "0", "FLARE_DS41_BRRR_LAYERS": "1",
                                           "FLARE_DS41_BRRR_CKPT": "/c"}).brrr_dense_native
    assert not TP3Config.from_environment({"FLARE_DS41_FUSED_ATTN_PER_CHUNK": "0"}).fused_attn_per_chunk
    assert not TP3Config.from_environment({"FLARE_DS41_FUSED_ATTN_PER_CHUNK": "x"}).fused_attn_per_chunk  # atoi
    assert TP3Config.from_environment({"FLARE_DS41_FUSED_ATTN_PER_CHUNK": "2"}).fused_attn_per_chunk
    assert TP3Config.from_environment({"FLARE_DS41_FUSED_ATTN_PER_CHUNK": " 1x"}).fused_attn_per_chunk
    assert not TP3Config.from_environment({"FLARE_DS41_BRRR_DENSE_NATIVE": "yes", "FLARE_DS41_BRRR_LAYERS": "1",
                                           "FLARE_DS41_BRRR_CKPT": "/c"}).brrr_dense_native
    # Numbers.
    assert TP3Config.from_environment({"FLARE_DS41_TP3_GRAPH_GIB": "1.5"}).tp3_graph_gib == 1.5
    assert TP3Config.from_environment({"FLARE_DS41_JOINT_HEADROOM_MIB": "256"}).joint_headroom_mib == 256
    assert TP3Config.from_environment({"FLARE_DS41_VISION_HOST": "1", "FLARE_DS41_VISION_STAGE": "1",
                                       "FLARE_DS41_VISION_STAGE_COPY": "kernel"}).vision_stage_copy == "kernel"
    assert TP3Config.from_environment({"FLARE_DS41_BRRR_LAYERS": "3", "FLARE_DS41_BRRR_CKPT": "/c",
                                       "FLARE_DS41_BRRR_BREAK": "rope"}).brrr_break == "rope"


@pytest.mark.parametrize("env,message", [
    ({"FLARE_DS41_BRRR_PARKED_MAX_ROWS": "12"}, "brrr_parked_max_rows needs brrr_layers"),
    ({"FLARE_DS41_BRRR_LAYERS": "0-39"}, "brrr_layers needs brrr_ckpt"),
    ({"FLARE_DS41_VISION_HOST": "1", "FLARE_DS41_VISION_STAGE_COPY": "kernel"}, "vision_stage_copy=kernel needs vision_stage"),
    ({"FLARE_DS41_VISION_HOST": "1", "FLARE_DS41_VISION_STAGE": "1", "FLARE_DS41_VISION_STAGE_COPY": "dma"},
     "vision_stage_copy must be 'memcpy' or 'kernel'"),
    ({"FLARE_DS41_GLOBAL_COLD": "812", "FLARE_DS41_EXPERT_COUNTS_DIR": "/c", "FLARE_DS41_HOST_EXPERT_LAYERS": "20-33"},
     "global_cold is a whole-model placement"),
    ({"FLARE_DS41_GLOBAL_COLD": "812"}, "global_cold needs expert_counts_dir"),
    ({"FLARE_DS41_HOST_EXPERT_LAYERS": "20-33", "FLARE_DS41_HOT_EXPERTS": "342"}, "hot_experts needs expert_counts_dir"),
    ({"FLARE_DS41_BRRR_CSF_CHECK": "1"}, "brrr_csf_check needs brrr_layers"),
    ({"FLARE_DS41_BRRR_NATIVE_SCALES": "1"}, "brrr_native_scales needs brrr_layers"),
    ({"FLARE_DS41_BRRR_BREAK": "rope"}, "brrr_break needs brrr_layers"),
    ({"FLARE_DS41_BRRR_LAYERS": "3", "FLARE_DS41_BRRR_CKPT": "/c", "FLARE_DS41_BRRR_BREAK": "norm"},
     "brrr_break must be unset or 'rope'"),
    ({"FLARE_DS41_BRRR_DENSE_NATIVE": "0"}, "brrr_dense_native=False needs brrr_layers"),
    ({"FLARE_DS41_TP3_GRAPH_GIB": "0"}, "tp3_graph_gib must be positive"),
    ({"FLARE_DS41_HOT_EXPERTS": "-1"}, "hot_experts must be a non-negative integer"),
    ({"FLARE_DS41_BRRR_PROLOGUE": "1"}, "FLARE_DS41_BRRR_PROLOGUE is no longer supported"),
    ({"FLARE_DS41_QB_ATOMIC": "1"}, "FLARE_DS41_QB_ATOMIC is no longer supported"),
    ({"DS41_FUSED_ATTN_PER_CHUNK": "0"}, "DS41_FUSED_ATTN_PER_CHUNK is no longer supported"),
])
def test_validation_errors_are_explicit(env, message):
    with pytest.raises(ValueError, match=message):
        TP3Config.from_environment(env)


def test_vision_stage_implies_vision_host_as_the_base_recipe_did():
    # The release export.sh defaults FLARE_DS41_VISION_STAGE=1; FLARE_DS41_VISION_HOST=0 must not kill the export.
    config = TP3Config.from_environment({"FLARE_DS41_VISION_STAGE": "1", "FLARE_DS41_VISION_HOST": "0"})
    assert config.vision_host and config.vision_stage
    assert TP3Config(vision_stage=True).vision_host is True
    config = TP3Config.from_environment({"FLARE_DS41_VISION_HOST": "0"})
    assert not config.vision_host and not config.vision_stage
    assert TP3Config.from_environment({"FLARE_DS41_VISION_STAGE": "1", "FLARE_DS41_VISION_STAGE_COPY": "kernel"}).vision_stage_copy == "kernel"


def test_planner_peeks_read_one_variable_without_validating_the_rest():
    stale = {"FLARE_DS41_BRRR_PARKED_MAX_ROWS": "12", "DS41_FUSED_ATTN_PER_CHUNK": "0",  # both invalid for from_environment
             "FLARE_DS41_TP3_GRAPH_GIB": "1.5", "FLARE_DS41_TP3_VOCAB_PARALLEL": "1"}
    with pytest.raises(ValueError):
        TP3Config.from_environment(stale)
    assert TP3Config.graph_gib_from_environment(stale) == 1.5
    assert TP3Config.vocab_parallel_from_environment(stale) is True
    assert TP3Config.graph_gib_from_environment({}) is None and TP3Config.graph_gib_from_environment({"FLARE_DS41_TP3_GRAPH_GIB": ""}) is None
    assert TP3Config.graph_gib_from_environment({"FLARE_DS41_TP3_GRAPH_GIB": "lots"}) is None
    assert TP3Config.vocab_parallel_from_environment({"FLARE_DS41_TP3_VOCAB_PARALLEL": "0"}) is False


def test_recipe_import_and_planner_helpers_survive_a_stale_environment(monkeypatch):
    import importlib
    import sys

    from ignition.recipes.deepseek41 import recipe

    monkeypatch.setenv("FLARE_DS41_BRRR_PARKED_MAX_ROWS", "12")  # invalid alone: parked rows without BRRR layers
    monkeypatch.setenv("FLARE_DS41_TP3_GRAPH_GIB", "2")
    monkeypatch.setenv("FLARE_DS41_TP3_VOCAB_PARALLEL", "1")
    assert recipe._graph_bytes(3) == 2 << 30 and recipe._graph_bytes(4) == 8 << 30
    from ignition import planning
    assert planning._tp3_vocabulary_parallel() is True
    module = importlib.reload(sys.modules[recipe.__name__])  # the declaration is rebuilt at import time
    assert module.RECIPE.ranks[3].graph_bytes == 2 << 30
    with pytest.raises(ValueError, match="brrr_parked_max_rows needs brrr_layers"):
        TP3Config.from_environment()


def test_metadata_never_records_absolute_paths(tmp_path):
    counts = tmp_path / "expert-order"
    counts.mkdir()
    for layer in (20, 21):
        (counts / f"expert-routes-rank0-layer{layer}.json").write_text('{"counts": [1, 2]}')
    view = tmp_path / "ds41-view"
    view.mkdir()
    (view / "config.json").write_text('{"model_type": "deepseek_v41"}')
    config = TP3Config(host_expert_layers=frozenset({20}), hot_experts=1, expert_counts_dir=counts,
                       brrr_layers=frozenset({0}), brrr_ckpt=view)
    record = config.to_metadata()
    assert record["expert_counts_dir"].startswith("expert-order#") and record["brrr_ckpt"].startswith("ds41-view#")
    assert str(tmp_path) not in json_text(record) and str(tmp_path) not in config.describe()
    assert len(record["expert_counts_dir"].split("#")[1]) == 12 and len(record["brrr_ckpt"].split("#")[1]) == 12
    # The fingerprint follows the content: a changed counts file or config.json changes it.
    (counts / "expert-routes-rank0-layer21.json").write_text('{"counts": [1, 2, 3]}')
    assert config.to_metadata()["expert_counts_dir"] != record["expert_counts_dir"]
    (view / "config.json").write_text('{"model_type": "other"}')
    assert config.to_metadata()["brrr_ckpt"] != record["brrr_ckpt"]
    assert describe_counts_dir(tmp_path / "absent") == "absent#missing"
    assert describe_checkpoint(tmp_path / "absent") == "absent#missing"
    assert describe_counts_dir(None) is None and describe_checkpoint(None) is None


def json_text(record):
    import json
    return json.dumps(record)


def test_malformed_numbers_and_layers_raise():
    with pytest.raises(ValueError):
        TP3Config.from_environment({"FLARE_DS41_HOT_EXPERTS": "many"})
    with pytest.raises(ValueError):
        TP3Config.from_environment({"FLARE_DS41_HOST_EXPERT_LAYERS": "20-x"})
    with pytest.raises(TypeError):
        TP3Config(merged_bank=1)


def test_metadata_record_is_json_ready_and_complete():
    import json
    from dataclasses import fields

    record = TP3Config.from_environment(RELEASE_ENV).to_metadata()
    assert set(record) == {item.name for item in fields(TP3Config)}
    assert record["host_expert_layers"] == "20-33" and record["brrr_layers"] == "0-39"
    assert record["expert_counts_dir"] == "expert-order#missing" and record["tp3_graph_gib"] == 1.0
    # The release view may exist on the build host, so only the shape is fixed: basename#12-hex-or-missing.
    name, digest = record["brrr_ckpt"].split("#")
    assert name == "ds41-1017-engram-mxfp4" and (digest == "missing" or len(digest) == 12)
    assert "/" not in json_text(record)
    assert record["brrr_parked_max_rows"] == 12 and record["vision_stage"] is True and record["brrr_break"] is None
    json.dumps(record)  # every value serializes
    assert TP3Config().to_metadata()["host_expert_layers"] == ""
    described = TP3Config.from_environment(RELEASE_ENV).describe()
    assert described.startswith("TP3 config: ") and "hot_experts=342" in described and "brrr_sparse" not in described


def test_config_is_frozen_and_hashable():
    config = TP3Config.from_environment(RELEASE_ENV)
    with pytest.raises(AttributeError):
        config.hot_experts = 300
    assert hash(config) == hash(TP3Config.from_environment(dict(RELEASE_ENV)))


def test_preparation_config_carries_the_tp3_config():
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    assert PreparationConfig(max_active=4).tp3 == TP3Config()
    config = PreparationConfig(max_active=4, tp3=RELEASE_CONFIG)
    assert config.tp3 is RELEASE_CONFIG
    with pytest.raises(TypeError):
        PreparationConfig(max_active=4, tp3={"hot_experts": 342})


def test_sidecars_are_written_into_the_cache_entry_directory(tmp_path):
    from types import SimpleNamespace

    from ignition.preparation.model_package import _write_sidecars

    entry = tmp_path / "models" / "ds41" / "gb202-tp3-1"
    entry.mkdir(parents=True)
    model = SimpleNamespace(package_sidecars={"tp3-config.json": {"hot_experts": 342}})
    notes = []
    _write_sidecars(model, SimpleNamespace(entry_path=str(entry)), notes.append)
    assert (entry / "tp3-config.json").read_text() == '{\n  "hot_experts": 342\n}\n'
    assert not (entry.parent / "tp3-config.json").exists()  # a file there would read as a fake package version
    assert notes == [f"wrote {entry / 'tp3-config.json'}"]
    notes.clear()
    _write_sidecars(model, SimpleNamespace(entry_path=None), notes.append)  # no-cache export
    assert notes == ["no cache entry for this export; sidecars not written: tp3-config.json"]
    _write_sidecars(SimpleNamespace(), SimpleNamespace(entry_path=str(entry)), notes.append)  # recipes without sidecars
    assert len(notes) == 1


def test_planner_helpers_read_the_environment_through_the_config(monkeypatch, tmp_path):
    from ignition.recipes.deepseek41.recipe import _graph_bytes, _parked_bytes

    for key in RELEASE_ENV:
        monkeypatch.delenv(key, raising=False)
    assert _graph_bytes(3) == 8 << 30 and _parked_bytes(3) == 0
    assert _graph_bytes(3, RELEASE_CONFIG) == 1 << 30 and _graph_bytes(4, RELEASE_CONFIG) == 8 << 30
    local = -(-2304 // 3)
    per_expert = 2 * local * 5120 // 2 + 5120 * local // 2
    assert _parked_bytes(3, RELEASE_CONFIG) == 14 * (384 - 342) * per_expert
    for key, value in RELEASE_ENV.items():
        monkeypatch.setenv(key, value)
    assert _graph_bytes(3) == 1 << 30 and _parked_bytes(3) == _parked_bytes(3, RELEASE_CONFIG)

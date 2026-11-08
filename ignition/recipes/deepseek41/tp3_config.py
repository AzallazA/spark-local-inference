# SPDX-License-Identifier: Apache-2.0
"""TP3 fork configuration for the DeepSeek V4.1 Flash recipe: one typed, frozen object for every switch.

``TP3Config.from_environment()`` is the single place that reads the ``FLARE_DS41_*`` environment (the release
``export.sh`` sets it); everything else receives a ``TP3Config`` instance. The dataclass defaults are the bare
upstream behaviour (nothing parked, no BRRR, no vision staging), so a ``TP3Config()`` is what a test or a plain
``ignition export`` without the fork's environment gets.

Parsing keeps the historical semantics: an empty string counts as unset, integers default to 0 ("or 0"), layer sets
accept ``"20-33"`` ranges and ``"20,21"`` lists (mixed, comma separated), and flags are ``"1"`` unless noted.
"""

import hashlib
import os
import re
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

ENV_PREFIX = "FLARE_DS41_"

# Switches whose code paths or names were removed (2026-11-08); the full construction (from_environment) rejects
# them so a stale setting fails loudly instead of being ignored.
REMOVED = {
    "FLARE_DS41_BRRR_PROLOGUE": "the BRRR attention prologue kernel measured no gain and was removed",
    "FLARE_DS41_QB_ATOMIC": "the q_b BF16 atomic split-K probe measured no gain and was removed",
    "DS41_FUSED_ATTN_PER_CHUNK": "renamed to FLARE_DS41_FUSED_ATTN_PER_CHUNK (read by TP3Config, not by the kernel)",
}


def _atoi(value: str) -> int:
    """C ``atoi``: the leading integer of the string, 0 when there is none (the kernel's historical parser)."""
    match = re.match(r"\s*[+-]?\d+", value)
    return int(match.group()) if match else 0


def _fingerprint(parts) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def describe_counts_dir(path: Path | None) -> str | None:
    """``<basename>#<sha256[:12] of the sorted route-count file names and sizes>``; never the full path."""
    if path is None:
        return None
    files = sorted(path.glob("expert-routes-rank0-layer*.json")) if path.is_dir() else []
    digest = _fingerprint(f"{f.name}:{f.stat().st_size}" for f in files) if files else "missing"
    return f"{path.name}#{digest}"


def describe_checkpoint(path: Path | None) -> str | None:
    """``<basename>#<sha256[:12] of the view's config.json>``; never the full path."""
    if path is None:
        return None
    config = path / "config.json"
    digest = hashlib.sha256(config.read_bytes()).hexdigest()[:12] if config.is_file() else "missing"
    return f"{path.name}#{digest}"


def parse_layers(spec: str) -> frozenset:
    """``"20-22"`` / ``"20,21"`` / ``"0-3,7"`` -> frozenset of layer indices; empty -> empty set."""
    layers = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        a, _, b = part.partition("-")
        layers.update(range(int(a), int(b or a) + 1))
    return frozenset(layers)


def format_layers(layers) -> str:
    """Compact ``"0-3,7"`` form of a layer set (inverse of ``parse_layers``)."""
    ordered, parts = sorted(layers), []
    while ordered:
        first = last = ordered.pop(0)
        while ordered and ordered[0] == last + 1:
            last = ordered.pop(0)
        parts.append(str(first) if first == last else f"{first}-{last}")
    return ",".join(parts)


@dataclass(frozen=True)
class TP3Config:
    # --- expert placement (weights.py, recipe.py) ---
    host_expert_layers: frozenset = field(default_factory=frozenset)
    """MoE layers whose routed-expert banks live in device-visible host memory (``"20-33"``)."""
    hot_experts: int = 0
    """Hottest experts per parked layer kept in VRAM, by the route counts in ``expert_counts_dir``; 0 = park all."""
    merged_bank: bool = False
    """Parked layers use one mixed VRAM/host bank (hot rows first) instead of separate hot + cold banks."""
    expert_counts_dir: Path | None = None
    """Directory of ``expert-routes-rank0-layer{L}.json`` route counts (vLLM route-monitor format)."""
    global_cold: int = 0
    """N least-routed (layer, expert) pairs across all MoE layers parked in host memory; 0 = off (DROPPED: c8 -2..-7%)."""
    embed_host: bool = False
    """Token embedding read from device-visible host memory (only its tokens' rows are gathered per step)."""
    # --- vision tower (weights.py, bootstrap.py, components/vision_staging.py) ---
    vision_host: bool = False
    """Vision tower parked in device-visible host memory; images ~3 s slower each unless staged. Implied by
    ``vision_stage`` (staging reads from the parked copy), as the pre-config recipe did."""
    vision_stage: bool = False
    """Stage the host-parked tower's weights through VRAM slots per block inside the vision graphs (image 0.11 s vs
    1.70 s). Requesting it parks the tower (``vision_host`` is forced on)."""
    vision_stage_copy: str = "memcpy"
    """Staging copy: ``memcpy`` (cudaMemcpy) or ``kernel`` (elementwise kernel over the mapped host range)."""
    # --- collectives and planning (bootstrap.py, recipe.py, planning.py) ---
    ar_rows: int = 0
    """Rows the shared TP3 all-reduce runtime is sized for (chunked prefill reductions); 0 = a whole prefill chunk."""
    tp3_graph_gib: float | None = None
    """Override of the per-rank captured-graph estimate for world size 3 (GiB); None = the 8 GiB default."""
    tp3_vocab_parallel: bool = False
    """Admit vocabulary-parallel sampling at TP3 (b12x gathers enabled for world 3)."""
    joint_headroom_mib: int | None = None
    """Compile headroom override (MiB) for the --joint-tuning contextual-selection phase only."""
    # --- BRRR fused layer kernel (brrr.py, decoder.py) ---
    brrr_layers: frozenset = field(default_factory=frozenset)
    """Layers whose decode/verify tail (o-proj .. MLP all-reduce) runs in BRRR's fused kernel (``"0-39"``)."""
    brrr_parked_max_rows: int = 0
    """Parked layers use BRRR only in graphs of at most N rows (gate; wide verification keeps Flare's MoE); 0 = ungated."""
    brrr_ckpt: Path | None = None
    """Checkpoint view BRRR reads its staged dense copies and block scales from (required with ``brrr_layers``)."""
    brrr_native_scales: bool = False
    """Load BRRR layers' experts without CSF scale compression (the kernel reads native E8M0 scales)."""
    brrr_dense_native: bool = True
    """BRRR reads Flare's own row-major E4M3 dense values instead of staged checkpoint copies (~35-40 MB/layer).
    Environment: unset = on; any set value other than "1" (including "") = staged copies, as the base recipe did."""
    brrr_csf_check: bool = False
    """Debug: verify Flare's dense bytes and the in-kernel CSF expansion against the checkpoint at register time."""
    brrr_log: bool = False
    """Debug: print BRRR layer registration and fused-norm selection lines."""
    brrr_break: str | None = None
    """Negative control for the layer test: ``"rope"`` reverses the RoPE frequencies the tail inverts (must fail)."""
    fused_attn_per_chunk: bool = True
    """Kernel: one-shot attention exchange posts one flag per chunk (False = legacy single flag; A/B toggle).
    Environment: unset = on; a set value is read like C ``atoi`` ("" and non-numbers = off), as the kernel did."""
    brrr_sparse: bool = False
    """Main-layer decode/verify attention through BRRR's one-launch sparse decode kernel (tabled: measured level)."""
    # --- other opt-in kernels (attention.py) ---
    fused_qkv_norm: bool = False
    """qkv_a's 4-way split-K reduction fused into the MLA norm + MXFP8 encode (tabled: bit-identical, measured level)."""
    # --- debug and test tooling (config.py, decoder.py, probe.py) ---
    mini: bool = False
    """Truncated real-weight checkpoint view (make_mini.py): accept 6..40 layers, no compacted prefill feedback."""
    no_engram: bool = False
    """Debug A/B (flare#193): keep the Engram handshake but discard its contribution to the residual."""
    probe: bool = False
    """Debug-serve only: per-layer residual probes dumped to /tmp/flare_probe.pt (never export with it)."""

    def __post_init__(self):
        object.__setattr__(self, "host_expert_layers", frozenset(self.host_expert_layers))
        object.__setattr__(self, "brrr_layers", frozenset(self.brrr_layers))
        for name in ("expert_counts_dir", "brrr_ckpt"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        for name in ("hot_experts", "global_cold", "ar_rows", "brrr_parked_max_rows"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"TP3 config: {name} must be a non-negative integer, not {value!r}")
        for name in ("merged_bank", "embed_host", "vision_host", "vision_stage", "tp3_vocab_parallel",
                     "brrr_native_scales", "brrr_dense_native", "brrr_csf_check", "brrr_log", "fused_attn_per_chunk",
                     "brrr_sparse", "fused_qkv_norm", "mini", "no_engram", "probe"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"TP3 config: {name} must be a boolean")
        if any(type(layer) is not int or layer < 0 for layer in (*self.host_expert_layers, *self.brrr_layers)):
            raise ValueError("TP3 config: layer sets must hold non-negative layer indices")
        if self.vision_stage_copy not in ("memcpy", "kernel"):
            raise ValueError(f"TP3 config: vision_stage_copy must be 'memcpy' or 'kernel', not {self.vision_stage_copy!r}")
        if self.vision_stage and not self.vision_host:
            object.__setattr__(self, "vision_host", True)  # staging implies parking (the base parked on either switch)
        if self.vision_stage_copy == "kernel" and not self.vision_stage:
            raise ValueError("TP3 config: vision_stage_copy=kernel needs vision_stage")
        if self.global_cold and (self.host_expert_layers or self.hot_experts):
            raise ValueError("TP3 config: global_cold is a whole-model placement; it excludes host_expert_layers and hot_experts")
        if self.global_cold and self.expert_counts_dir is None:
            raise ValueError("TP3 config: global_cold needs expert_counts_dir")
        if self.hot_experts and self.host_expert_layers and self.expert_counts_dir is None:
            raise ValueError("TP3 config: hot_experts needs expert_counts_dir (route counts pick the hot set)")
        if self.brrr_parked_max_rows and not self.brrr_layers:
            raise ValueError("TP3 config: brrr_parked_max_rows needs brrr_layers")
        if self.brrr_layers and self.brrr_ckpt is None:
            raise ValueError("TP3 config: brrr_layers needs brrr_ckpt (the checkpoint view BRRR reads its dense copies from)")
        for name in ("brrr_native_scales", "brrr_csf_check", "brrr_break"):
            if getattr(self, name) and not self.brrr_layers:
                raise ValueError(f"TP3 config: {name} needs brrr_layers")
        if not self.brrr_dense_native and not self.brrr_layers:
            raise ValueError("TP3 config: brrr_dense_native=False needs brrr_layers")
        if self.brrr_break not in (None, "rope"):
            raise ValueError(f"TP3 config: brrr_break must be unset or 'rope', not {self.brrr_break!r}")
        if self.tp3_graph_gib is not None and not self.tp3_graph_gib > 0:
            raise ValueError("TP3 config: tp3_graph_gib must be positive")
        if self.joint_headroom_mib is not None and (type(self.joint_headroom_mib) is not int or self.joint_headroom_mib < 0):
            raise ValueError("TP3 config: joint_headroom_mib must be a non-negative integer")

    # --- environment readers: the shared parsers, used by the full construction and by the two planner-side peeks ---

    @staticmethod
    def _get(env, name, default=""):
        value = env.get(ENV_PREFIX + name, "")
        return default if value == "" else value

    @classmethod
    def graph_gib_from_environment(cls, environ=None) -> float | None:
        """Planner-side peek at FLARE_DS41_TP3_GRAPH_GIB alone (no validation; the recipe declaration is built at
        import time for every ignition command, so this must never raise on an unrelated stale variable)."""
        value = cls._get(os.environ if environ is None else environ, "TP3_GRAPH_GIB")
        try:
            return float(value) if value else None
        except ValueError:
            return None

    @classmethod
    def vocab_parallel_from_environment(cls, environ=None) -> bool:
        """Planner-side peek at FLARE_DS41_TP3_VOCAB_PARALLEL alone (no validation; generic planner code)."""
        return cls._get(os.environ if environ is None else environ, "TP3_VOCAB_PARALLEL") not in ("", "0")

    @classmethod
    def from_environment(cls, environ=None) -> "TP3Config":
        """Build and validate the whole config from ``FLARE_DS41_*`` variables (``environ`` defaults to
        ``os.environ``). Only the recipe's ``prepare`` (and the CUDA tests) call this."""
        env = os.environ if environ is None else environ

        def get(name, default=""):
            return cls._get(env, name, default)

        def flag(name):  # "1" means on; anything else (including unset and "") is off
            return get(name) == "1"

        def truthy(name):  # unset, "" and "0" are off; any other value is on
            return get(name) not in ("", "0")

        def integer(name):
            return int(get(name, "0") or 0)

        for name, reason in REMOVED.items():
            if env.get(name, "") != "":
                raise ValueError(f"{name} is no longer supported: {reason}")
        counts = get("EXPERT_COUNTS_DIR")
        ckpt = get("BRRR_CKPT")
        headroom = get("JOINT_HEADROOM_MIB")
        dense_native = env.get(ENV_PREFIX + "BRRR_DENSE_NATIVE")  # set-but-empty means staged copies (base semantics)
        per_chunk = env.get(ENV_PREFIX + "FUSED_ATTN_PER_CHUNK")  # set-but-empty is atoi("") = 0 (kernel semantics)
        return cls(
            host_expert_layers=parse_layers(get("HOST_EXPERT_LAYERS")),
            hot_experts=integer("HOT_EXPERTS"),
            merged_bank=flag("MERGED_BANK"),
            expert_counts_dir=None if not counts else Path(counts),
            global_cold=integer("GLOBAL_COLD"),
            embed_host=flag("EMBED_HOST"),
            vision_host=flag("VISION_HOST"),
            vision_stage=flag("VISION_STAGE"),
            vision_stage_copy=get("VISION_STAGE_COPY", "memcpy"),
            ar_rows=integer("AR_ROWS"),
            tp3_graph_gib=cls.graph_gib_from_environment(env),
            tp3_vocab_parallel=cls.vocab_parallel_from_environment(env),
            joint_headroom_mib=None if not headroom else int(headroom),
            brrr_layers=parse_layers(get("BRRR_LAYERS")),
            brrr_parked_max_rows=integer("BRRR_PARKED_MAX_ROWS"),
            brrr_ckpt=None if not ckpt else Path(ckpt),
            brrr_native_scales=flag("BRRR_NATIVE_SCALES"),
            brrr_dense_native=dense_native is None or dense_native == "1",
            brrr_csf_check=flag("BRRR_CSF_CHECK"),
            brrr_log=flag("BRRR_LOG"),
            brrr_break=get("BRRR_BREAK") or None,
            fused_attn_per_chunk=per_chunk is None or _atoi(per_chunk) != 0,
            brrr_sparse=flag("BRRR_SPARSE"),
            fused_qkv_norm=flag("FUSED_QKV_NORM"),
            mini=flag("MINI"),
            no_engram=truthy("NO_ENGRAM"),
            probe=truthy("PROBE"),
        )

    def to_metadata(self) -> dict:
        """JSON-ready record of every switch: layer sets as ``"20-33"`` strings; the two paths as basename plus a
        short content fingerprint (never an absolute path, so the record can live in a package or a report)."""
        record = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, frozenset):
                value = format_layers(value)
            elif item.name == "expert_counts_dir":
                value = describe_counts_dir(value)
            elif item.name == "brrr_ckpt":
                value = describe_checkpoint(value)
            record[item.name] = value
        return record

    def describe(self) -> str:
        """One-line summary of the switches that differ from the defaults (for export progress)."""
        record = self.to_metadata()
        changed = [f"{item.name}={record[item.name]!r}" for item in fields(self)
                   if getattr(self, item.name) != (item.default_factory() if item.default is MISSING else item.default)]
        return "TP3 config: " + (", ".join(changed) if changed else "defaults")

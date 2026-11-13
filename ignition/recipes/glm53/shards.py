# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-rank partition widths of the GLM 5.3 Flash geometry, derived once from the rank count.

Every width is the smallest aligned shard whose `world` copies cover the checkpoint axis, so
one, two and four ranks reproduce the exact divisions the recipe always used (64 // world heads,
2048 // world expert columns, 154880 // world vocabulary rows). Three ranks pad the axes that
three does not divide. Heads use shards of eight: every rank holds 24 heads (0-23, 24-47, 48-63 plus
8 zero heads on the last rank), because the b12x sparse-MLA GLM_NEXT decode kernel mis-tiles head
counts that do not divide 64 (22 leaks into inactive rows; 16/24/32 pass,
tests/recipes/test_glm53_sparse_mla_heads_cuda.py) and every rank's compiled forwards and scratch
layouts must share one shape (ignition/preparation/aot.py `_prepare_ranks`, scratch.py `_shared_views`).
2048 expert columns become 2304 (768 per rank; DeepSeek V4.1's 2304-wide experts give the same 768-column shard at TP3, but as an exact
partition, so the zero-padded NVFP4-CSF expert path here has no production precedent and rests on
the GPU equivalence tests) and 154880 vocabulary rows become 154944 (51648 per rank, 64-aligned for
the logit gather, with the 64 extra logits excluded before sampling). The padded tail of every axis
holds zero weights, so checkpoint bytes and scales stay unchanged; the loader zero-fills those tails.

The routed experts may take a narrower shard than the shared expert (`routed_width`, fork-only export
override FLARE_GLM53_EXPERT_COLUMNS=704 in recipe.py). The shared expert's block-FP8 scales cover
128 x 128 blocks, so its rank boundaries must be 128-aligned (768 per rank at three ranks). The routed
NVFP4 experts need only 32-row CSF scale groups (b12x `CsfSource.validate`), W4A16 N64 packed tiles over
the 2 x width gate/up rows and a K64 FC2 tile (`_candidate_tile_fits`); 704 = 11 x 64 is the smallest
width that satisfies them and covers 2048 at three ranks (ranks 0/1 exact, rank 2 holds 640 + 64 zero
columns). Every rank keeps one shape, as for the heads.
"""

from __future__ import annotations

from dataclasses import dataclass

WORLDS = (1, 2, 3, 4)
HEADS = 64
HEAD_DIM = 128
EXPERT_WIDTH = 2048
DENSE_WIDTH = 12288
VOCABULARY = 154880
# Heads per rank are a multiple of eight (see the module note); 64 // world for one, two and four ranks.
HEAD_ALIGNMENT = 8
# Shared-expert columns align to its 128 x 128 block-FP8 scale blocks; routed NVFP4 experts may align to 64.
EXPERT_ALIGNMENT = 128
ROUTED_ALIGNMENT = 64


def padded_width(extent: int, world: int, alignment: int = 1) -> int:
    """Smallest `alignment`-aligned width whose `world` shards cover `extent`."""
    if any(type(value) is not int or value <= 0 for value in (extent, world, alignment)):
        raise ValueError("padded shard widths require positive integer extent, world and alignment")
    return -(-extent // (world * alignment)) * alignment


@dataclass(frozen=True)
class Shards:
    """One rank's share of every partitioned GLM axis."""

    world: int
    # KDA recurrent heads and MLA query/value heads on one rank; the indexer's 32 heads stay replicated.
    heads: int
    # Routed and shared expert intermediate columns on one rank.
    expert_width: int
    # Dense MLP intermediate columns on one rank (layers 0 through 2).
    dense_width: int
    # Vocabulary rows one rank projects before the logit gather.
    vocabulary: int
    # Routed expert intermediate columns on one rank; None means `expert_width` (the upstream shard).
    routed_width: int | None = None

    def __post_init__(self):
        if self.routed_width is None:
            object.__setattr__(self, "routed_width", self.expert_width)

    @classmethod
    def of(cls, world: int, *, expert_columns: int | None = None) -> Shards:
        """Shard widths for `world` ranks; `expert_columns` selects one of `routed_expert_columns(world)`."""
        if world not in WORLDS:
            raise ValueError("GLM tensor parallelism requires one, two, three or four ranks")
        expert = padded_width(EXPERT_WIDTH, world, EXPERT_ALIGNMENT)
        routed = expert if expert_columns is None else expert_columns
        if routed not in routed_expert_columns(world):
            raise ValueError(f"GLM routed experts at {world} ranks take "
                             f"{' or '.join(map(str, sorted(routed_expert_columns(world), reverse=True)))} columns "
                             f"per rank, not {expert_columns!r}")
        return cls(world, padded_width(HEADS, world, HEAD_ALIGNMENT), expert,
                   padded_width(DENSE_WIDTH, world, 128), padded_width(VOCABULARY, world, 64), routed)

    def routed_columns(self, rank: int) -> tuple[int, int]:
        """Rank `rank`'s routed expert columns; the last of three ranks ends past the checkpoint's 2048."""
        if type(rank) is not int or not 0 <= rank < self.world:
            raise ValueError("routed expert columns require a rank within the world")
        return rank * self.routed_width, (rank + 1) * self.routed_width

    def head_range(self, rank: int) -> tuple[int, int]:
        """Rank `rank`'s heads; the last of three ranks ends past the checkpoint's 64 (zero heads)."""
        if type(rank) is not int or not 0 <= rank < self.world:
            raise ValueError("head ranges require a rank within the world")
        return rank * self.heads, (rank + 1) * self.heads

    def checkpoint_heads(self, rank: int) -> int:
        """Checkpoint heads rank `rank` holds: 24, 24 and 16 for three ranks."""
        first, last = self.head_range(rank)
        return min(last, HEADS) - first

    def head_channels(self, rank: int, width: int) -> tuple[int, int]:
        """Rows or columns of a head-major projection with `width` channels per head."""
        first, last = self.head_range(rank)
        return first * width, last * width

    @property
    def kda_channels(self) -> int:
        """Rank-local q, k or v channels of a KDA layer."""
        return self.heads * HEAD_DIM

    @property
    def padded_heads(self) -> int:
        """Heads over every rank, including zero-weight padding heads."""
        return self.heads * self.world

    @property
    def padded_vocabulary(self) -> int:
        """Gathered logit columns over every rank, including the excluded padding logits."""
        return self.vocabulary * self.world


def routed_expert_columns(world: int) -> frozenset[int]:
    """Routed expert widths per rank at `world` ranks: 128-aligned (upstream) and 64-aligned (704 at three)."""
    return frozenset(padded_width(EXPERT_WIDTH, world, alignment) for alignment in (EXPERT_ALIGNMENT, ROUTED_ALIGNMENT))


# Rank-local widths any supported rank count can present to a prepared operator.
HEAD_COUNTS = frozenset(Shards.of(world).heads for world in WORLDS)
EXPERT_WIDTHS = frozenset(Shards.of(world).expert_width for world in WORLDS)
ROUTED_WIDTHS = frozenset(width for world in WORLDS for width in routed_expert_columns(world))

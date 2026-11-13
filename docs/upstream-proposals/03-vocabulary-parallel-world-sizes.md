# 03. Recipe-declared world sizes for vocabulary-parallel sampling

## Problem

`--vocabulary-parallel-sampling` (DSpark drafting with each rank sampling over
its vocabulary shard) is refused at three GPUs: `ignition/planning.py:442` and
`:572` raise "has no TP3 gather", and `:559-563` drop world size 3 from the
recipe's sizes. The reason is in b12x: the column gather runtime
(`ignition/preparation/collectives.py:79-88`) is a `DcpAllToAll`, whose
`SUPPORTED_WORLD_SIZES` on FlashInfer main 6b867fdb
(`comm/pcie/pcie_dcp_a2a.py:42`) is `(2, 4, 8, 9, 10, 12, 16)`. The planner
also hard-codes `recipe.name != "deepseek41"` (`planning.py:433`, `:560`), a
model allowlist in shared code. The fork's published TP3 builds run
vocabulary-parallel DSpark drafting (k = 5) at world 3 and were qualified with
LiveBench 82.2 and a concurrency-8 stress pass through `flare run`.

## What the fork does today (6f9fcba)

- Environment gate `FLARE_DS41_TP3_VOCAB_PARALLEL` (`planning.py:55-59`)
  bypasses the three checks above. Prohibited upstream.
- FlashInfer `tp3-patches/flashinfer-dcp-a2a-world3.patch` adds 3 to
  `SUPPORTED_WORLD_SIZES`; `flashinfer-tuning-world3.patch` adds 3 to the
  `DcpAllToAll*.lse_reduce_scatter` and `all_gather_heads` tuning tables
  (`comm/pcie/_tuning.py:27-28`, `:32-33`). Neither is on main.
- Sampling defaults: `_sampling_defaults` in
  `ignition/recipes/deepseek41/bootstrap.py:92-99` pins `top_p = 1` and
  declares range `{min: 1, max: 1}` when vocabulary-parallel sampling is
  prepared, so requests asking for top-p are clamped by Helios
  (`helios/src/server/config.rs:86-104` loads `sampling_defaults.ranges`)
  instead of invalidating rows.
- Tests: `tests/components/test_vocabulary_cuda.py:129`, `:153` parametrize
  world `[2, 3, 4]` and raise `compile_workers` to `max(2, world)` (`:60`);
  `tests/components/test_vocabulary_sampling_tp3_cuda.py` is the TP2 sampling
  test parametrized over world 2 and 3 (shard width 384 at world 3, `:33`);
  `tests/components/test_vocabulary_gather_stress_cuda.py` replays the gather
  3000 times per world with a per-replay numeric check.
- Copied history proposals for `--history-drafts` on this path (see 05).

## Proposed upstream design

1. `Recipe.vocabulary_parallel_world_sizes: tuple[int, ...] = ()` in
   `ignition/recipes/declaration.py`, validated in `__post_init__` as a subset
   of `tensor_parallel`. DeepSeek V4.1 declares `(3, 4)` once the FlashInfer
   change is installed, `(4,)` before. The planner replaces the name check and
   the `len == 3` checks with: option requires a non-empty set; selected or
   fitted world must belong to it; error text names the declared sizes.
   This is a recipe declaration, the sanctioned home for a setting, and it
   removes a model-name allowlist from `planning.py`.
2. Sampling policy: move the fork's `_sampling_defaults` into the recipe's
   `prepare` path as data, and document in `docs/cli-design.md` (the
   `--vocabulary-parallel-sampling` row) and `docs/model-preparation.md:750`
   that top-p, top-k, and min-p requests are clamped to the defaults, with the
   reason (per-rank proposals must stay exact normalized distributions).
3. Tests: fold the world-3 parameter into `test_vocabulary_cuda.py` and
   `test_vocabulary_sampling_tp_cuda.py` (instead of the fork's copied file),
   and add the stress test with a default iteration count small enough for
   CI (for example 200) and an environment variable only for the count, which
   AGENTS.md permits for tests.

## API and schema changes

Recipe field only. No Protobuf change; `sampling_defaults.ranges` already
exists on the wire.

## Migration and compatibility

Exported packages are unaffected. Planner error messages change wording.
Hosts with four GPUs behave as before.

## What else differs from the TP2/TP4 path

- The gather runtime at world 3 relies on the b12x "generic modulo path" the
  FlashInfer patch enables; TP2/TP4 use the power-of-two path. A b12x-level
  test at world 3 is required before the Flare change.
- Padded vocabulary partition: 129280 / 3 rounds to 43136 columns per rank,
  64-aligned; the last rank holds 43008 real columns and 128 padding columns
  that the sampling kernels must ignore (the TP3 test sets padded logits to
  1000 to catch leaks, `test_vocabulary_sampling_tp3_cuda.py:41-44`).
- `compile_workers=max(2, world)` in tests (`test_vocabulary_cuda.py:60`):
  the fork's commits do not record why two workers were insufficient at world
  3; determine and state the reason before submitting.
- The decode all-reduce at world 3 uses one shared runtime (proposal 04).

## Tests to submit

`test_vocabulary_cuda.py` at world 3 (two tests), `test_vocabulary_sampling_tp_cuda.py`
at world 3, the gather stress test, and a planner unit test (no GPU) that the
option is rejected when the recipe declares no sizes and accepted for a
declared size. The fork has run the world 2 and 3 cases on three GB202 GPUs;
world 4 cannot be run by the fork.

## Open questions

- Is the explicit `{min: 1, max: 1}` range needed? `docs/model-development.md:43`
  says an absent range fixes the parameter at its default; if Helios already
  clamps, the range is redundant and only the default change is needed.
- Whether maintainers want the stress test in the suite or only its
  single-replay form.

## PR plan

FlashInfer first (see 05). Then Flare, two commits:
1. Recipe field, planner checks, planner unit test, documentation.
2. World-3 test parameters and the gather stress test.

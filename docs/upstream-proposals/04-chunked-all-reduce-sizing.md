# 04. One sized PCIe all-reduce runtime for decode and prefill

## Problem

DeepSeek V4.1 preparation builds two reduction runtimes per rank
(`ignition/recipes/deepseek41/bootstrap.py` at upstream 972e28e4, lines
317-319): a PCIe `AllReduce` sized for decode rows and a copy-engine
`DmaAllReduce` sized for a prefill chunk. The fork's comment at
`bootstrap.py:335-338` states that the DMA all-reduce reduce-scatters the
5120-wide hidden state across ranks and therefore needs the world size to
divide 5120, which world size 3 does not. The same comment records a second
constraint: graph capture enters only the main collective's `capture()`
context, so reductions issued through a second runtime would be captured
outside the protocol b12x requires. Sizing the single remaining runtime for a
full 4096-row prefill chunk costs 786 MiB per GPU against 190 MiB at 1024 rows
(fork comment, `bootstrap.py:340-341`); the fork's "GPU room" notes record that
on 96 GB TP3 this VRAM is better spent on resident experts and KV.

## What the fork does today (6f9fcba)

- `bootstrap.py:339-349`: if `5120 % world != 0`, create one `AllReduce` with
  `max_tokens = max(all_reduce_tokens, min(prefill_chunk_size, FLARE_DS41_AR_ROWS or prefill_chunk_size))`
  and use it for both decode and prefill; otherwise the two-runtime form.
  `FLARE_DS41_AR_ROWS` is an environment read, prohibited upstream.
- `ignition/recipes/deepseek41/decoder.py:229-231` and `:275-276` use
  `PreparedChunkedAllReduce` for the attention and MLP reductions so a runtime
  smaller than a prefill chunk still reduces prefill rows in chunks.
- `PreparedChunkedAllReduce` itself (`ignition/components/collectives.py:213-246`)
  is upstream code, unchanged by the fork, and already used by
  `hyperconnection_gate.py:67`, `kimi3/decoder.py:189`, `:208`, `:322`,
  `muse_spark/decoder.py:109`, and `deepseek41/engram.py:202`.
- `ignition/preparation/ladders/gb202_pcie_tp3.json`: an offline tuning ladder
  for three GB202 GPUs, usable with `--collective-tuning runtime`.

## Proposed upstream design

A helper in `ignition/preparation/collectives.py`:

```python
def create_layer_reduction_runtimes(group, device, *, hidden_size, decode_rows,
                                    prefill_rows, max_rows=None, collective_tuning, ipc):
    """Return (decode, prefill) runtimes for row-major hidden-state reductions.

    The copy-engine prefill runtime partitions `hidden_size` columns across ranks and
    requires `hidden_size % world == 0`. Otherwise one PCIe all-reduce, sized for
    `max(decode_rows, min(prefill_rows, max_rows))` rows, serves both; callers reduce
    prefill through PreparedChunkedAllReduce so the smaller capacity still covers a chunk.
    """
```

The condition is explicit in one place and the recipe no longer encodes it.
`max_rows` is a planning parameter: an advanced CLI flag
`--all-reduce-rows N` carried through the plan into the recipe's
`PreparationConfig` and printed in the preparation report, per AGENTS.md
("A setting that only experiments need is an advanced flag carried through the
plan into the recipe's `PreparationConfig`"). Default `None` keeps the full
prefill chunk, matching upstream memory behavior for divisible worlds.

`decoder.py` adopts `PreparedChunkedAllReduce` for the two layer reductions,
as the other recipes already do; for a runtime sized to the full row count it
produces one part and is behaviorally identical.

## API and schema changes

New helper and one advanced flag through `PreparationConfig`; no Protobuf
change. Packages record the plan, so the chosen row capacity is visible in
`flare inspect`.

## Migration and compatibility

Worlds that divide the hidden size keep two runtimes and identical captured
graphs. Exported packages are unaffected; graphs prepared with a smaller
runtime are simply different graphs in new packages.

## Tests

- Pure-Python unit test of the helper's sizing rule (no GPU): divisible world
  returns two distinct runtimes; non-divisible returns one shared runtime
  sized by the rule; `max_rows` clamps. Runtime constructors can be injected
  for the test.
- `tests/components/test_pcie_cuda.py:145-162` already covers
  `PreparedChunkedAllReduce`; add a world-3 parameter and a case where the
  runtime capacity is smaller than the row count so several parts run.
- Export and `flare run` at TP3 with `--all-reduce-rows 1024` and 4096-row
  prefill chunks: compare prefill tokens/s and decode steps/s against the
  two-runtime default on a divisible world. The fork can supply TP3 numbers on
  GB202 only; TP4 comparison must come from a four-GPU host.

## Open questions

- Confirm the divisibility constraint against b12x `DmaAllReduce` source rather
  than the fork comment; also establish what upstream 972e28e4 does at TP3
  today, since the recipe already declares world size 3 (does export fail, or
  does the DMA runtime pad?).
- Prefill cost of chunking at 1024 rows versus one 4096-row reduction has not
  been isolated in the fork's measurements; the fork's decision was driven by
  VRAM, not prefill speed. Measure before claiming neutrality.
- Whether to include `gb202_pcie_tp3.json`: it is measured data for a
  specific topology and only matters with runtime collective tuning.

## PR plan

1. Helper with unit test; DeepSeek bootstrap calls it; decoder uses
   `PreparedChunkedAllReduce` for both layer reductions.
2. `--all-reduce-rows` advanced flag through plan and `PreparationConfig`,
   report line, CLI documentation.
3. Optional: the TP3 ladder file with its measurement conditions in
   `ladders/README.md`.

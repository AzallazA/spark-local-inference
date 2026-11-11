# Exporting the package yourself

> **Fork copy.** This is the export guide as shipped in the dated release folder of the recipes repository
> (https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes). In this checkout the code is already present, so skip
> the `git apply source.patch` step; `export.sh`, `serve.sh`, `make_checkpoint_view.sh`, `check_env.sh` and the
> `expert-order/` counts are in `tp3-tools/release/` (`COUNTS=tp3-tools/release/expert-order`). The release folder's
> copies are regenerated from this tree at tagging time (`docs/TP3-GUIDELINES.md`, B.5).

This guide builds the Hyperflare (formerly Flare; the binary is still `flare`) package in this folder from scratch.
It covers the toolchain, the patched fork,
the checkpoint, the export, and the checks to run before serving. Budget **about 2 hours of mostly unattended time**
and **~900 GB of free disk**.

## 1. Requirements

| | Tested with | Notes |
| --- | --- | --- |
| GPUs | 3 × RTX PRO 6000 Blackwell, 96 GB (SM120), PCIe Gen5 x16 | the export tunes and captures on these exact cards. Tested at **450 W with +75 MHz core / +6000 MHz memory offset**; 350 W measured 10-14% slower. Kernel tuning measures your cards as they run, so export with the clocks and power limit you will serve with |
| Host RAM | 256 GB (tested: DDR5-6000, 8 channels) | export peaks around 160 GB. Serving uses ~148 GB with Engram in RAM, ~59 GB with Engram on disk |
| Disk | ~900 GB free | checkpoints ~412 GB, package ~380 GB, kernel tuning cache 60-80 GB. Tested on PCIe Gen5 NVMe (12 GB/s reads) |
| Driver | 615.71.09 | any driver supporting CUDA 13.1 |
| CUDA toolkit | 13.1 (`/usr/local/cuda-13.1`) | nvcc 12.x is too old for SM120. The export also compiles the BRRR layer kernel with it |
| Rust | 1.97 (rustup) | builds the `flare` binary and the Heliopause extension |
| uv | 0.11 | Hyperflare's Python environment (Python 3.12, managed by uv) |
| Locked memory | `memlock unlimited` for your user | Hyperflare pins host memory; for example `/etc/security/limits.d/90-memlock.conf`: `<user> - memlock unlimited`, then log in again |

```bash
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH
ulimit -l        # must print "unlimited"
```

`check_env.sh` in this folder checks all of the above plus the fork state (step 2), free disk and idle GPUs. It is
read-only and prints one `PASS` / `WARN` / `FAIL` line per requirement; `WARN` lines are steps you have not done yet.
Run it once the clone from step 2 exists and again after each step:

```bash
/path/to/this-folder/check_env.sh <fork checkout>      # exit 1 if anything FAILed; MIN_FREE_GB=<n> if checkpoints live on another drive
```

The exact versions behind every number in this folder (driver, toolkit, torch, FlashInfer, benchmarks) are in
[versions.txt](versions.txt). torch's `cu132` wheel next to the 13.1 toolkit is expected, not a mismatch.

## 2. Build the Hyperflare fork

The fork is upstream NVIDIA Hyperflare at `972e28e4` (the commit predates the rename; the old
`flare-local-inference` URL redirects) plus `source.patch` from this folder.

```bash
git clone https://github.com/NVIDIA/hyperflare-inference.git flare
cd flare
git checkout 972e28e44781d8d0c84d70ae434db92c36076a43
git apply /path/to/this-folder/source.patch           # 164 files (110 new, including brrr/ and tp3-tools/); applies cleanly to 972e28e4

uv sync --locked --extra prepare --extra test         # Python environment in .venv (pulls FlashInfer @ 1f8ed8a3)
uv run --locked --extra prepare python -m ignition.patches   # Hyperflare's torch/Triton patches; re-run after every uv sync
cargo build --release --manifest-path flare/Cargo.toml  # the flare binary: flare/target/release/flare
```

**Apply the FlashInfer patches** to the installed FlashInfer. They patch files inside `.venv`, so re-apply them
after every `uv sync`:

```bash
SP=$(.venv/bin/python -c "import flashinfer, pathlib; print(pathlib.Path(flashinfer.__file__).parents[1])")
for p in /path/to/this-folder/tp3-patches/*.patch; do patch -p1 -d "$SP" < "$p"; done
```

| Patch | Why |
| --- | --- |
| `flashinfer-tiny-decode-swiglu.patch` | the `tiny_decode` W4A8 MoE kernel ignored `swiglu_limit`; without it, outputs corrupt mid-reply at TP3 |
| `flashinfer-dcp-a2a-world3.patch`, `flashinfer-tuning-world3.patch` | PCIe all-to-all and its tuning tables for world size 3 |
| `flashinfer-engram-mxfp4.patch` | packed-FP4 path in the Engram lookup kernel, for NVIDIA's MXFP4 tables |

Check the build:

```bash
flare/target/release/flare --version
.venv/bin/ignition export --help | head -3
```

**Update 20261111 (Hyperflare-tp3).** `source.patch` is generated from fork tag `hyperflare-tp3-20261111`:
graph scratch (`FLARE_DS41_ENGRAM_CHUNK_ROWS=1024` and `FLARE_DS41_SHARE_ENGRAM_RESIDUAL=1`, both `export.sh` defaults:
~465 MiB per GPU of graph scratch freed, KV 2,467,200 → 3,014,656 tokens at cap 24 and 32G prefix cache), the Helios TTFT
guard (`--scheduler-policy interactive`, the new `serve.sh` default, plus the `--scheduler-*` tuning flags and
`--no-mixed-prefill`), the `MAXC` and `PARKED_ROWS` knobs in `export.sh`, BRRR kernel changes (CSF descriptors kept in
registers, `-fno-gnu-unique` builds, three default-off experiments and an opt-in profile pass), and handbook updates.
**Rebuild the `flare` binary** (step 2) and **export again** (step 4); a tuning cache warmed by a 20261108 export is
enough (measured: one 36-minute export, no warm-up needed).

**Update 2026-11-08.1.** `source.patch` is now generated from fork tag `flare-tp3-20261108.1`: the `TP3Config` object
(the `FLARE_DS41_*` variables keep their names and defaults; invalid or removed names fail the export loudly; the
resolved configuration is written as `tp3-config.json` beside `model.safetensors`), the BRRR prologue and QB_ATOMIC
experiment paths removed, the FP64 per-operator layer test, and `tp3-tools/release/` (this folder's scripts and the
expert order, kept in the fork as the single source). The exported package gains ~6K KV tokens from the removed
prologue scratch buffers (2,467,200 at 32G prefix cache). Full c8 qualification of that package: see the README.

**Update 2026-11-08.** `source.patch` now carries the BRRR kernel hygiene pass (cold-first parked jobs, per-chunk attention
flags, a constants contract between kernel and config, a cross-rank agreement check), vision weight staging
(`FLARE_DS41_VISION_STAGE=1`, the new `export.sh` default: 0.11 s per image instead of 1.70 s), one resolution point for
Engram table storage in Helios, `flare run ... --dry-run` (KV capacity without a serve; no GPU memory or CUDA context,
but it needs the NVIDIA driver and the GPUs visible) and three GPU equivalence tests.
It was regenerated from the fork after its history was cleaned for publication (experiment scripts removed, tools
under `tp3-tools/`), and verified to reproduce the fork tree exactly on a clean `972e28e4` checkout. Measurements in
MEASUREMENTS.md ("2026-11-08").

**Update 2026-11-07 (review pass).** `source.patch` now also carries the fixes from a code review of this release. None of
them change the exported package or its numbers; they matter for serving and for other setups:

| Fix | Effect |
| --- | --- |
| Helios counted the TP3 merged expert banks (VRAM prefix + host tail) as all-VRAM in its GPU admission check and in `flare models describe` | the check could reject a package that fits by a few GiB; fixed in all six accounting sites |
| `FLARE_ENGRAM_RAM_TABLES` tables were not counted toward `--memory-limit` host accounting, and the disk row-cache budget was split across tables no longer on disk | correct host accounting; disk tables get the whole cache budget |
| `--history-drafts` inputs were advertised for every DeepSeek export but only consumed under vocabulary-parallel sampling | TP4 exports behave like upstream again (TP3 unchanged) |
| Guards: the BRRR tail checks its attention-output dtype and RoPE table; the parked-layer gate requires an actual host tail; the export fails loudly if b12x ever re-materialises host-parked experts (TP4-style tiling would) | no silent garbage or VRAM over-commit on future changes |
| Three opt-in experiments measured level and left switched off (`FLARE_DS41_BRRR_SPARSE`, `FLARE_DS41_FUSED_QKV_NORM`, `FLARE_DS41_QB_ATOMIC`). The two FlashInfer patches they rely on, `flashinfer-dense-splitk-deferral.patch` and `flashinfer-tuning-route-reuse.patch`, are fork-only and not shipped here; `tp3-patches/` holds only the four the default build needs | default build untouched; see MEASUREMENTS.md |

A `flare` binary built from the updated patch serves the published package as before (checked: same KV count, corruption check clean).

## 3. Build the checkpoint

The package is exported from a **checkpoint view**: DeepSeek's original shards 1-46 plus NVIDIA's shards 47-48,
which hold the two Engram tables in MXFP4 (97 GiB instead of 189 GiB in FP8). Download only the shards you need:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --exclude "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors"          # ~307 GB
hf download nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --include "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors" LICENSE README.md   # ~105 GB
```

Then assemble the view (symlinks only, no copies) with this folder's `config.json` and
`model.safetensors.index.json`. The script checks that every tensor resolves and that the Engram tables are MXFP4:

```bash
/path/to/this-folder/make_checkpoint_view.sh \
  ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
  ~/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-1017-NVFP4/snapshots/<revision> \
  ~/ds41-1017-engram-mxfp4
# expect: "96085 tensors in 48 shards, missing 0; layer-1 Engram table U8 [384006168, 128]"
```

**The directory name becomes the model name.** Hyperflare names the exported model after the checkpoint directory: the
package lands in `~/.cache/flare/packages/models/<directory name>/gb202-tp3-<timestamp>`, `flare models list` shows
it as `<directory name>`, and `flare run <directory name>` serves it. `serve.sh` defaults to
`PACKAGE=ds41-1017-engram-mxfp4`, the name used here; if you name the view differently, serve with
`PACKAGE=<your name>` (or the package path).

The index in this folder is NVIDIA's mapping with a corrected `total_size`. Hyperflare checks that field against the
actual tensor bytes; the uncorrected value makes the export stop at "index total_size does not match".

## 4. Export

From the fork checkout, with GPUs idle (stop any server first). **Export twice on a fresh machine:** with a tuning
cache not yet warmed by this release's code (cold, or from 20261105), tuning at any `HOT` above ~300 runs out of VRAM. A `HOT=300` export first fills the shared cache, and the
second export reuses it:

```bash
CKPT=~/ds41-1017-engram-mxfp4 HOT=300 /path/to/this-folder/export.sh   # warms the tuning cache (and gives the 6.25M-KV variant)
CKPT=~/ds41-1017-engram-mxfp4 /path/to/this-folder/export.sh           # the release build (HOT=342)
```

`DRAFTS` (default 5) must be the same at export and in `serve.sh`: the package is captured for one draft count.

`export.sh` sets these and runs `ignition export`:

| Setting | Value | Meaning |
| --- | --- | --- |
| `FLARE_DS41_HOST_EXPERT_LAYERS` | `20-33` | routed-expert layers that keep part of their experts in host RAM |
| `FLARE_DS41_HOT_EXPERTS` (`HOT`) | `342` | experts per parked layer kept in VRAM (out of 384); the rest live in host RAM |
| `FLARE_DS41_MERGED_BANK` | `1` | one VRAM + host-RAM expert bank per parked layer (one MoE call) |
| `FLARE_DS41_EXPERT_COUNTS_DIR` | `expert-order/` | route counts that decide which experts are hot |
| `FLARE_DS41_BRRR_LAYERS` | `0-39` | layers that use the fused BRRR layer tail (compiled from `brrr/kernels/decoder.cu` at export) |
| `FLARE_DS41_BRRR_CKPT` | `$CKPT` | checkpoint the BRRR tail reads its weight layout from |
| `FLARE_DS41_BRRR_PARKED_MAX_ROWS` (`PARKED_ROWS`) | `12` | parked layers use the BRRR tail only in graphs of at most this many rows (24 = every graph the serve-time verification budget 24 keeps; measured not consistently faster at c8 and slightly slower at 512K, MEASUREMENTS.md) |
| `FLARE_DS41_EMBED_HOST`, `FLARE_DS41_VISION_HOST` | `1`, `1` | token embedding and vision tower in device-visible host RAM. `FLARE_DS41_VISION_HOST=0` in the environment keeps the vision tower in VRAM (pair it with a lower `HOT`; unmeasured, and unnecessary with staging) |
| `FLARE_DS41_VISION_STAGE` | `1` | stage the host-parked vision tower's weights through two VRAM slots per block on a side stream (190 MiB from the activation pool, no KV change). Measured 2026-11-08: cold single image 0.11 s vs 1.70 s, 60-image accuracy unchanged. `0` = direct host-RAM reads. `FLARE_DS41_VISION_STAGE_COPY=kernel` swaps the memcpy for an elementwise copy if a driver mishandles memcpy from mapped host memory |
| `FLARE_DS41_AR_ROWS` | `1024` | size the prefill all-reduce runtime for 1024 rows (~600 MiB/GPU less than 4096) |
| `FLARE_DS41_ENGRAM_CHUNK_ROWS` | `1024` | graph scratch: the Engram projection, all-reduce and mix run in 1024-row chunks (~300 MiB/GPU less scratch); `0` = previous layout |
| `FLARE_DS41_SHARE_ENGRAM_RESIDUAL` | `1` | graph scratch: `engram_residual` lives in the `ffn_hc.residual` buffer (~165 MiB/GPU); `0` = separate buffer |
| `FLARE_DS41_TP3_VOCAB_PARALLEL` | `1` | vocabulary-parallel sampling at TP3 (fixes top_p at 1.0) |
| `FLARE_DS41_TP3_GRAPH_GIB` | `1` | the planner's graph-memory estimate for TP3 |
| `--prefill-chunk-size` (`PREFILL`) | `4096` | prefill chunk; 2048 gives more KV, less prefill speed |
| `--max-concurrency` (`MAXC`) | `8` | request slots the package captures graphs for (8 = the qualified package; 16 costs ~650K KV tokens and needs its own `HOT=300` warm-up) |
| `--mtp-draft-tokens` (`DRAFTS`) | `5` | DSpark drafts, NVFP4 draft head, FP8 Markov head. 7 measured slower (MEASUREMENTS.md) |

### What to expect, and how long

Stage durations from the two exports behind this release (the `[ OK ]` lines in the log), in order. The first export
also compiles the BRRR kernel (~1 min, cached in `.brrr-ext/` afterwards) and, on a fresh machine, tunes every kernel
from scratch.

| Stage (log line) | First export, `HOT=300` | Second export, `HOT=342` | Cumulative, second export |
| --- | --- | --- | --- |
| Load weights | 46 s | 45 s | 0.8 min |
| Allocate model state, construct graph catalog | 26 s | 27 s | 1.2 min |
| Prepare and tune kernels | **20-30 min with a cold cache** (175 s measured with a partly warm one) | 151 s | 3.7 min |
| Compile model forwards | 259 s (cold compile cache) | 41 s | 4.4 min |
| Load and warm compiled forwards | 238 s | 214 s | 8.0 min |
| Prepare serving graphs (capture) | 18 s | 23 s | 8.4 min |
| Materialize weight payloads | 170 s | 166 s | 11.1 min |
| Release preparation resources, assemble package payloads | 111 s | 107 s | 12.9 min |
| Verify and publish 3 ranks + shared data, commit | 410 s | 436 s | 20.2 min |
| **Wall time to `Exported cache entry`** | **~50-60 min expected** (29 min measured with the partly warm cache) | **~30 min** | |

The stage lines add up to about 20 minutes; the rest of the wall time is process start-up, the b12x progress between
stages and the final clean-up. Second-export log, `<home>` = your home directory:

```text
[    OK    ] Load weights.  45.3s
[    OK    ] Prepare and tune kernels.  150.6s
[    OK    ] Compile model forwards.  35/35  40.6s
[    OK    ] Load compiled forwards.  105/105  135.1s
[    OK    ] Materialize weight payloads.  165.5s
[    OK    ] Verify and publish shared data.  104451107600 source bytes verified  115.9s
[    OK    ] Commit package.  <home>/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611070243  0.0s
Exported cache entry: <home>/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611070243
```

When you then serve it (step 5), the server log shows an `Allocate cache` line like this one, recorded for the release
package `gb202-tp3-202611111227` (`HOT=342`, verification row cap 24, `PREFIX_CACHE=32G`; the same line as the first graph scratch package):

```text
[    OK    ] Allocate cache.  7.9 GiB · 3014656 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
[    OK    ] Start server.  http://127.0.0.1:15015 · 1048576 tokens/request  0.3s
```

### Variants: more KV for less speed

The hot-expert count is the main lever. Each hot expert per parked layer moved to host RAM frees ~79 MiB per GPU,
about **92K KV tokens**. Decode gets slower because more experts stream over PCIe.

| `HOT` | KV tokens | Measured (350 W, natural EOS, 32K c1 / c4 / c8) |
| --- | --- | --- |
| **342** | **3,014,656** (graph scratch; 2,467,200 without) | 288 / 634 / 838 tok/s before graph scratch (decode level with it); **this release** at 450 W: 300 / 713 / 950 tok/s |
| 300 | ~6.9M estimated with graph scratch (6,247,680 measured at cap 128 without it) | 269 / 610 / 800 tok/s: 4-7% slower, corruption check clean (before graph scratch) |

> **Disclaimer.** Other values were not measured on this fork. Expect KV ≈ 3,014,656 + (342 − HOT) × 92K tokens at verification row cap 24, and a
> decode cost that grows as more of your traffic routes to cold experts (roughly 1% per 6-10 experts in our
> measurements). Check the `Allocate cache` line and bench your own export before relying on it.

## 5. Check and serve

```bash
flare/target/release/flare models list                      # model ds41-1017-engram-mxfp4 (= the view's directory name), version gb202-tp3-<timestamp>, newest = default
FLARE=flare/target/release/flare /path/to/this-folder/serve.sh              # serves ds41-1017-engram-mxfp4; PACKAGE=<name>@<version> or a path for another one
```

The model name in `flare models list` is the checkpoint view's directory name (step 3); each export adds a version
`gb202-tp3-<timestamp>` and the newest becomes the default. `serve.sh` defaults to `PACKAGE=ds41-1017-engram-mxfp4`,
so with the view name above it needs no `PACKAGE`. Keep `DRAFTS` equal to the export's.

The server is ready when the log shows `Start server. http://127.0.0.1:15015`, about 55 s with the Engram tables
loading into RAM. The `Allocate cache` line reports the KV token count; it should match the table above within a
few thousand tokens (the excerpt in step 4 shows the expected line).

**Less host RAM:** `ENGRAM_RAM=0 ./serve.sh` reads both Engram tables from disk with direct I/O. The server then
uses ~59 GiB of host RAM instead of ~148 GiB, with decode ~4% slower and prefill unchanged on Gen5 NVMe. See the
README for the numbers. Disk mode reads the tables from the exported checkpoint files, so the checkpoint view and the
Hugging Face cache files its symlinks point to must stay at the same absolute paths as at export.

Then run the corruption check, which compares sampled decode tokens against a teacher-forced prefill and must
report `TRUE corruptions 0`, and a benchmark:

```bash
python3 /path/to/this-folder/tools/verify_fix.py 10
cd llm-inference-bench && .venv/bin/python llm_decode_bench.py --port 15015 --model DeepSeek-V4.1-Flash \
  --concurrency 1,4,8 --contexts 32k --coding-peak --output result.json
python3 /path/to/this-folder/tools/bench_steps.py result.json   # steps/s and tokens/step per cell
```

Optional: `tools/llm-bench-flare-steps.patch` (`git apply` inside `llm-inference-bench`, tested on v0.9.1; the
benchmark is MIT-licensed, see its README) makes the benchmark's own summary show the "MTP-normalized decode steps/s"
table for Hyperflare, computed from Hyperflare's draft-round counters.

## 6. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Tuning fails with `OutOfMemoryError ... failed to prepare with configuration` | A tuning cache **not yet warmed by this release's code** (cold, or from 20261105) needs more free VRAM than serving at any `HOT` above ~300. Export with `HOT=300` first, then the target: the tuning cache in `~/.cache/flare/kernels` is shared and the second export reuses it. A new draft count (`DRAFTS=7`) needs its own warm-up export |
| Export stops at `KeyError: 'FLARE_DS41_BRRR_CKPT'` | you ran `ignition export` directly; use `export.sh` or set `FLARE_DS41_BRRR_CKPT` to the checkpoint view |
| BRRR kernel compile fails (`nvcc`, `ninja`) | `CUDA_HOME` must point at CUDA 13.1 (`nvcc` with `sm_120a`); `ninja` comes with `uv sync`. Compiled kernels are cached in `.brrr-ext/` in the checkout |
| `index total_size does not match tensor payload bytes` | use this folder's `model.safetensors.index.json` (step 3) |
| Export dies on `Cannot patch imported module triton.runtime.build` | Hyperflare's torch/Triton patches are missing: run `uv run --locked --extra prepare python -m ignition.patches` after `uv sync` |
| Chat requests return 400 for `thinking` / `clear_thinking` | Hyperflare's thinking toggle is `chat_template_kwargs: {"enable_thinking": true}` (or `false`); vLLM's `thinking` / `clear_thinking` fields are rejected |
| `serve.sh` stops with an invalid-value error for `--scheduler-policy interactive` | the `flare` binary predates this release's `source.patch`: rebuild it (step 2), or serve with `SCHEDULER=balanced` |
| Chat requests return 400 for `reasoning_effort` | accepted values are `low`, `high`, `xhigh`, `max` or an integer from 1 through 100; `"medium"` is rejected |
| The command does nothing and exits 0 | your shell's `env` is not `/usr/bin/env` (uv installs one at `~/.local/bin/env`); the scripts here call `/usr/bin/env` explicitly |
| Corruption check reports true corruptions | `flashinfer-tiny-decode-swiglu.patch` is not applied in `.venv` |
| `serve.sh` fails because `ds41-1017-engram-mxfp4` is not found | the checkpoint view had another directory name, and the model name follows it (step 3). `flare models list` shows the name; serve with `PACKAGE=<name>` or the package path |
| First `flare run` of a new package asks to trust the publisher | packages are signed with your local development key (`~/.config/flare/`); approve it once in an interactive terminal |
| Disk fills up after several exports | remove old versions with `flare models remove <model>@<version>` then `flare models gc` (deleting folders by hand frees nothing: files are shared) |

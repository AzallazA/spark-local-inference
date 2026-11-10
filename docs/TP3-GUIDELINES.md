# TP3 fork handbook: orientation, workflows, rules and the mistakes behind them

For anyone (human or agent) working on DeepSeek-V4.1-Flash on NVIDIA Flare at TP3 (3x RTX PRO 6000 Blackwell
Workstation, 450 W, +75 core / +6000 mem). Part A says where things are and how to run them. Part B gives the
step-by-step for the recurring tasks. Part C is tips for agents. Part D is the rule list, each rule with the
incident that produced it (dates are 2026-11). Read the part you need before starting; read D.1 before any speed
claim, D.3 before any kernel edit, D.4 before any push.

---

## A. Orientation

### A.1 Where things live

| What | Where |
| --- | --- |
| Fork checkout (branch `ds41-tp3-v3`, the one that serves) | `~/flare-up-v3` (`.venv/`, `flare/target/release/flare`, `.brrr-ext/` compiled kernels) |
| Other worktrees of the same repo | `~/flare-up` (old v2, its `serve-best.sh` just execs v3), `~/flare-up-base` (detached baseline), `~/flare-up-stage2` (kernel stage 2), `~/flare-up-vision` (vision staging). `git worktree list` from any of them. |
| Recipe (the Python that defines the model for Ignition) | `ignition/recipes/deepseek41/` (`decoder.py`, `attention.py`, `weights.py`, `brrr.py`, `bootstrap.py`, `fused_qkv_norm.py`) |
| BRRR fused layer tail kernel | `brrr/kernels/decoder.cu` (+ `flare_sparse_decode.cu`, opt-in) |
| Rust serving runtime | `helios/src/` (model/allocation accounting, Engram reader, protection), CLI in `flare/src/cli.rs` |
| FlashInfer (b12x) site-package patches | `tp3-patches/*.patch` (re-apply to `.venv` after every `uv sync`) |
| TP3 test tooling | `tp3-tools/` (`bench_steps.py`, `spec_metrics.py`, `steps_logger.py`, `test_fused_qkv_norm.py`, `test_flare_sparse_decode.py`); Engram precision eval in `tp3-tools/engram-eval/` |
| Release scripts and guides (single source) | `tp3-tools/release/` (`export.sh`, `serve.sh`, `make_checkpoint_view.sh`, `check_env.sh`, `expert-order/`), `docs/tp3/EXPORT.md`, `docs/tp3/ENGRAM-PRECISION.md`; the dated recipes folder carries copies made at tagging time |
| GPU unit tests | `tests/recipes/test_deepseek41_*_cuda.py`; the TP3 ones are `..._brrr_tail_cuda.py` (3 GPUs) and `..._expert_parking_cuda.py` (1 GPU) |
| Quality and consistency checks | `tp3-tools/verify_fix.py` (TRUE corruptions), `tp3-tools/consistency.py` (decode vs prefill logprobs), `~/ds41-tp3/bench.py` (fixed 43-answer set + c4) |
| KV capacity without a serve | `flare run <pkg> <serve flags> --dry-run` (per-rank memory plan + KV tokens + snapshot slots; exact on 5 recorded loads; `--dry-run-free-memory` overrides the assumed free VRAM) |
| Full qualification suite | release folder `qualification/qual_all.sh` (eval tools lavd hotel retrieval longgen eos livebench stress); needs `~/llm-inference-bench`, `~/LiveBench` |
| Checkpoints | full view `/m2-2/flare-ckpt/ds41-1017-engram-mxfp4` (built by `release/make_checkpoint_view.sh`; the release docs' `~/ds41-1017-engram-mxfp4` is an example path, not ours); truncated test models `/m2-2/flare-ckpt/ds41-mini6` (6 layers) and `ds41-mini8` (8 layers, has native scales, use for parking/BRRR tests) |
| Exported packages | `~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-<timestamp>/`; default `...202611070243`; each non-default package is ~4 GB on disk (shared blobs) |
| Server scripts | `~/flare-up-v3/serve-best.sh` (defaults: pkg 070243, K=5, ALPHA=0.8, BUDGET=24, PCM=32G, SCHED=balanced), restart via `/tmp/restart-default.sh` |
| Release folder (public) | `~/rtx-recipes/DeepSeek-V4.1-Flash Flare-tp3-20261107/` (README, EXPORT, MEASUREMENTS, ENGRAM-PRECISION, export.sh, serve.sh, source.patch, qualification/) |
| Upstream proposals | `docs/upstream-proposals/` in the fork (public with the branch, not submitted upstream); mirrored on the `proposals` branch of the private staging repo |
| Scratch / logs / older vLLM-era work | `~/ds41-tp3/` (bench logs, eval_*.json, analyze_trace.py, compose files); `~/engram-eval/` (FP8 vs MXFP4 Engram perplexity eval) |
| Archive of old builds | `/mnt/ai_a` (AI_A), cold storage `/mnt/ai_b` |

### A.2 Server: start, stop, inspect

```bash
# is it up, what is it serving
curl -s localhost:15015/v1/models | grep -o '"id":"[^"]*"'
ps -eo args | grep "release/[f]lare run" | grep -o "gb202[^ ]*\|budget [0-9]*\|alpha [^ ]*\|memory [^ ]*"
curl -s localhost:15015/metrics | awk '/^flare_total_kv_tokens/{print $2}'     # KV pool in tokens

# stop (never pgrep -f a pattern that is in your own command line, see D.2.4)
for p in $(ps -eo pid,args | awk '$2=="./flare/target/release/flare" && $3=="run"{print $1}'); do kill -INT $p; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 2; done

# start the default (background, log in ~/flare-up-v3/serve-best.log); takes ~2 min to answer /v1/models
/tmp/restart-default.sh
# start a candidate package with overrides
cd ~/flare-up-v3 && P=~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-2026110XXXXX BUDGET=24 nohup ./serve-best.sh > cand.log 2>&1 &
```

The server needs ~97 GiB host RAM for the two Engram tables (`FLARE_ENGRAM_RAM_TABLES=0`) plus the parked experts
and prefix cache. Chat requests: thinking is toggled with `chat_template_kwargs: {"enable_thinking": true|false}`.
Flare clamps temperature 0 to 1.0.

### A.3 Benchmarks

```bash
# decode speed, c1/c4/c8 at 32K context, 60 s each; writes JSON
python3 ~/llm-inference-bench/llm_decode_bench.py --port 15015 --model DeepSeek-V4.1-Flash \
    --concurrency 1,4,8 --contexts 32k --duration 60 --output $D/$LABEL-32k-r1.json
# long context c1
... --concurrency 1 --contexts 8k,32k,128k,512k --duration 30 --output $D/$LABEL-longctx.json
# coding peak (high-acceptance speculative workload)
... --concurrency 1 --contexts 8k,32k,128k,512k --coding-peak --output $D/$LABEL-codingpeak.json
# steps/s and acceptance from the server log
.venv/bin/python tp3-tools/spec_metrics.py <serve log>
```

Reference numbers for the default build at 450 W, cap 24, 32G prefix cache: c1/c4/c8 32K ≈ 298/738/959 tok/s,
coding c1 ≈ 456 tok/s. Spread between runs is 2-3%. See D.1 before interpreting anything.

### A.4 Quality checks (cheapest first)

1. `python3 tp3-tools/verify_fix.py 10` on the running server: look only at `TRUE corruptions` (must be 0).
2. `python3 ~/ds41-tp3/bench.py out.json` : 43 known-answer checks (expect 42-43/43), 1,500-token decodes, 32K needle,
   4 concurrent.
3. `.venv/bin/python tp3-tools/consistency.py 96 6 <tokenizer dir>` : decode-vs-prefill logprob agreement; compare against the
   same number from the default build in the same session (it is biased toward b12x rounding).
4. `qualification/qual_all.sh` from the release folder (about 1 h at c8). Needs `FLARE_LOG`, `CKPT`, writes to
   `QUAL_OUT`. LiveBench needs `--livebench-release-option 2024-11-25`. Expected: eval 43/43, tools 60/60, lavd/hotel
   30/30, retrieval pass, longgen 4/4, LiveBench ~82, c8 stress PASS.

### A.5 Export

```bash
cd ~/flare-up-v3 && CKPT=/m2-2/flare-ckpt/ds41-1017-engram-mxfp4 HOT=342 tp3-tools/release/export.sh   # the recipes folder ships the same script
```

Full export ≈ 25-40 min with a warm tuning cache; writes a new `gb202-tp3-<timestamp>` package. Knobs are env vars
(`HOT`, `PREFILL`, `DRAFTS`, `MAXC`, `FLARE_DS41_VISION_HOST`, plus the `FLARE_DS41_*` set in `export.sh`). `MAXC`
sets `--max-concurrency` (default 8; `MAXC=16` = the c16 package, warm its tuning cache first, D.2.2). Graph scratch
is on by default (`FLARE_DS41_ENGRAM_CHUNK_ROWS=1024`, `FLARE_DS41_SHARE_ENGRAM_RESIDUAL=1`; set both to 0 for the
previous layout). The recipe reads
them once, in `TP3Config.from_environment()` (`ignition/recipes/deepseek41/tp3_config.py`), validates them, prints
the non-default switches in the export progress and records every switch in `tp3-config.json` beside the package's
`model.safetensors` (and in each rank's memory report as `tp3_config`; paths appear as basename + fingerprint,
never absolute). Everything downstream receives the `TP3Config` instance; new switches are fields there, not new
`os.environ` reads. `FLARE_DS41_VISION_STAGE=1` implies `FLARE_DS41_VISION_HOST=1` (staging reads from the parked
copy; `VISION_HOST=0` alone keeps the tower in VRAM only when staging is off too). Export and serving cannot overlap
(VRAM). Mini export for kernel/wiring checks: `SNAP=<truncated view> LAYERS=<brrr layers> tp3-tools/mini-export.sh`
(export, serve, `tp3-tools/consistency.py`, stop; ~10 min; refuses to start next to a running server).

### A.6 GPU tests and traces

```bash
# BRRR layer tail vs stock path, 3 GPUs (all widths 1..48, resident+parked, per-chunk on/off, negative control): the 5 base
# cases ~15 min (-k "perchunk or break-rope and not item6"); the item-6 switch cases (-k "item6 or split or fastroute or parity") add ~45-55 min
# CUDA 13.1 must be first on PATH: /usr/bin/nvcc is 12.4 and fails with "Unsupported gpu architecture 'compute_120a'"
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH
cd ~/flare-up-v3 && CUDA_VISIBLE_DEVICES=0,1,2 FLARE_RUN_CUDA_TESTS=1 FLARE_DS41_SNAPSHOT=/m2-2/flare-ckpt/ds41-mini8 \
    .venv/bin/pytest -x -s tests/recipes/test_deepseek41_brrr_tail_cuda.py     # in a worktree, also TORCH_EXTENSIONS_DIR=$PWD/.brrr-ext
# expert parking (resident / split / merged bank) equivalence + byte accounting, 1 GPU
CUDA_VISIBLE_DEVICES=0 FLARE_RUN_CUDA_TESTS=1 FLARE_DS41_SNAPSHOT=/m2-2/flare-ckpt/ds41-mini8 \
    .venv/bin/pytest -x -s tests/recipes/test_deepseek41_expert_parking_cuda.py
# kernel micro-tests
.venv/bin/python tp3-tools/test_fused_qkv_norm.py ; .venv/bin/python tp3-tools/test_flare_sparse_decode.py
# GPU tests that session.tune() then compile_forward() a planned Triton kernel (GLM indexer/MLA, DS41 indexer)
# need the kernel caches pinned as `ignition export` pins them, or Inductor's first kernel repoints
# TRITON_CACHE_DIR and the deferred b12x load fails with "planned Triton program has not been compiled":
export TRITON_CACHE_DIR=~/.cache/flare/kernels/triton B12X_COMPILE_CACHE_DIR=~/.cache/flare/kernels/b12x \
       TORCHINDUCTOR_CACHE_DIR=~/.cache/flare/kernels/inductor
```

Traces: `nsys profile` on the serving process at c1 and c8 (existing captures `serve-node-v3-c1.nsys-rep`,
`serve-node-v3-c8.nsys-rep` in the fork root); summarise per-stream busy intervals, not just kernel sums (D.1.5).

### A.7 Repositories and where to push

| Repository | Where | Purpose and rules |
| --- | --- | --- |
| Upstream Flare | `origin` = https://github.com/NVIDIA/flare-local-inference (read-only for us) | source of truth; our branch `ds41-tp3-v3` sits on top of it (merge-base 972e28e4 as of 2026-11-08; upstream main has moved on, see B.5) |
| Local Gitea (private view) | remote `gitea` on the LAN (address, login and password location: agent memory note `gitea-local`); repos `flare-tp3-fork` and `rtx-recipes` | browse commits, diffs and Markdown; push every commit here (`git push gitea <branch>`); experiment branches welcome (`stage2-wide-rows`, `vision-staging`, `config-object`, `upstream-main`); container `gitea` |
| Public release (GitHub) | https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes (`origin` of `~/rtx-recipes`); private staging copy `RTX-PRO-6000-TP3-Inference-Recipes-staging` (`staging` remote, branch `proposals` holds the proposal drafts) | user-facing release folders (docs, export/serve scripts, source.patch, qualification results); push only with the user's explicit OK, after the hygiene grep (D.4.1) |
| Public fork (GitHub) | remote `public` = https://github.com/AzallazA/spark-local-inference (fork of NVIDIA/spark-local-inference, the renamed flare-local-inference; default branch `ds41-tp3-v3`) | reviewed main line and release tags only (`flare-tp3-YYYYMMDD`, each with a GitHub Release note); push with the user's explicit OK; upstream PRs for `docs/upstream-proposals` open from it; GitHub cannot make a fork private, hence Gitea for the private view |

The fork tree is kept publication-clean at all times: no experiment scripts at the root (they live outside the repo, e.g.
`~/ds41-tp3/fork-archive` and the untracked `serve-best.sh`), no `/home/<user>` paths, no hostnames. `git grep -n
"/home/\|$(hostname)" HEAD` must stay empty before any push.

### A.8 Knob glossary (what each lever does and what we know)

| Knob | Effect / verdict |
| --- | --- |
| `HOT` (hot experts per parked layer, layers 20-33) | 342 = 2.37M KV default; each expert fewer ≈ +92K KV tokens; 300 = 2.99M KV release build, ~4% slower |
| `BUDGET` (`--mtp-max-verification-token-budget`) | 24 adopted: +1-2% at c1/c8 vs 128; also drops 13 speculative graphs (~76 MiB/rank) = +87K KV tokens |
| `PCM` (`--prefix-cache-memory`) | 32G adopted: ~78 warm 127K prompts vs ~37 at 16G; costs ~10 MiB device headroom (-6K KV), host RAM only otherwise |
| `K` drafts | keep 5; coding acceptance is high; k7 was -3-6% at c4/c8 |
| `ALPHA` 0.6/0.8/1.0, scheduler balanced/throughput | within noise; defaults 0.8 / balanced |
| Power 350/450/600 W, core offset | 450 W ≈ +10-12% over 350 W; 600 W no gain; +100/+150 core no gain |
| `FLARE_DS41_BRRR_PARKED_MAX_ROWS=12` | gate for BRRR in parked layers; ungated = c8 +1%, c1 level (tabled) |
| `FLARE_DS41_BRRR_SPARSE`, `FLARE_DS41_FUSED_QKV_NORM` | opt-in experiments, measured level; default off, kept with their kernel tests (`TP3Config.brrr_sparse` / `fused_qkv_norm`) |
| BRRR prologue, `FLARE_DS41_QB_ATOMIC` | measured level; REMOVED 2026-11-08 (`TP3Config.from_environment` rejects the variables) |
| BRRR stage 2 (OTOK/SPAN 48 wide rows, worktree `~/flare-up-stage2`) | REGRESSION: c4 -5%, c1 -3% (11-08, real export); closed |
| Extra PDL trigger at the end of `layer_kernel` | cost ~2% at c8 (11-08 bisect); removed. Only the original trigger remains |
| `FLARE_DS41_VISION_STAGE=1` (vision tower weights staged through VRAM slots; implies `VISION_HOST=1`) | cold image 0.11 s vs 1.70 s (11-08); 190 MiB of activation-pool slots, KV unchanged |
| Engram: `FLARE_ENGRAM_RAM_TABLES=0` vs `--embedding-table-storage disk` | disk = -90 GiB RAM, -3-4% decode; MXFP4 vs FP8 tables: wiki ppl +2.6%, benchmarks within noise |
| `FLARE_DS41_EMBED_HOST=1`, `FLARE_DS41_VISION_HOST=1` | embeddings/vision in host RAM; images ~3 s slower each |
| `FLARE_DS41_AR_ROWS=1024` | chunked all-reduce sizing for prefill |
| `FLARE_DS41_ENGRAM_CHUNK_ROWS=1024`, `FLARE_DS41_SHARE_ENGRAM_RESIDUAL=1` | graph scratch, default on in `export.sh`: Engram projection/reduce/mix in 1024-row chunks (-300 MiB/rank) and `engram_residual` on `ffn_hc.residual` (-165 MiB/rank); KV 2.46M -> 3.01M at HOT 342, decode level |
| `MAXC` (`--max-concurrency` at export) | 8 default; 16 = c16 package: its graphs cost ~650K KV vs c8 (dry run) |
| BRRR item 6: `FLARE_DS41_BRRR_SPLIT_TAIL`, `_FAST_ROUTE`, `_COUNTER_PARITY` | default off, not adopted (11-09): split tail and counter parity slower in the layer A/B; fast route -0.5/-1.9/-2.0 µs/layer at m 1/6/12 but end to end level (c1 1.004x, c4 0.989x, c8 0.998x) |

Ideas already closed (do not re-run without a new angle): FP8 LM head, fused layer L2 prefetch, EP all-reduce over
b12x, CPU-side expert compute, DCP3, dynamic expert cache (0.40% cold routes at H342), route reuse, sparse decode at
FP8, k7 drafts, 8192 prefill chunks, BRRR wide-row restreaming (stage 2), extra PDL trigger in the layer kernel, BRRR item 6
(split tail, fast route, counter parity). Details in `~/.claude/.../memory/` notes and MEASUREMENTS.md.

---

## B. Workflows

### B.1 Speed experiment (settings or build)

1. Write down the hypothesis and the expected gain in tok/s before touching anything.
2. Note the current confounders: power limit, offsets, BUDGET, PCM, K, package id (`ps` line in A.2).
3. Settings change: restart the server with the override. Build change: full export (mini first only for risky
   kernel/wiring edits), then serve the new package with the default's flags.
4. Run alternating A/B in one session: default, candidate, default, candidate, default, candidate (3 each, same
   bench command, same duration). Record tok/s per concurrency and coding.
5. Decide only on the 3-run means. A difference under ~3% at c1/c8 is "level"; stop there and do not look further.
6. If adopted: `verify_fix` (0 TRUE corruptions), then the relevant qualification stages, then update
   `serve-best.sh` comments, MEASUREMENTS.md and the memory note. If not adopted: record the verdict in the experiment's
   docstring or MEASUREMENTS.md so it is not repeated.
7. Restore the default server (`/tmp/restart-default.sh`).

### B.2 Kernel or recipe change

1. Work in a worktree (`git worktree add ~/flare-up-<name> -b <name>`), symlink `.venv`; keep a backup of the file you
   replace (`decoder.cu.pre-stage1` pattern) until the change is committed. Every export or script from the worktree
   needs `PYTHONPATH=<worktree>` (D.2.11) and its own `TORCH_EXTENSIONS_DIR=<worktree>/.brrr-ext`.
2. Compile standalone first (the BRRR extension builds via `torch.utils.cpp_extension`; `CUDA_HOME=/usr/local/cuda-13.1`).
3. Run the layer test (A.6) with the negative control; a per-row mismatch is usually the reference, see D.3.11.
4. Mini export + `tp3-tools/consistency.py` against the default's number.
5. Full export, `verify_fix`, A/B per B.1. Only then commit; then regenerate `source.patch` and the release docs if
   the default changed (B.3).
6. Anything that touched `.venv` site-packages must have a patch file in `tp3-patches/` before the session ends.

### B.3 Release folder update

1. Scripts and guides are edited in the fork (`tp3-tools/release/`, `docs/tp3/`) and copied into the dated folder
   `~/rtx-recipes/DeepSeek-V4.1-Flash Flare-tp3-YYYYMMDD/`; per-release docs (README, MEASUREMENTS, results) are edited
   in the folder. Regenerate `source.patch` from the fork
   (`git diff <upstream base>..HEAD` plus untracked new files) and verify it in a temp worktree of clean upstream:
   apply, then compare `git ls-files -s | md5sum` with the fork.
2. Hygiene pass: `grep -rn "/home/\|~/\|$(hostname)\|gmail" .`; `find . -name __pycache__`; scrub `hostname`/`uname`
   fields from result JSON. Scripts must write to `QUAL_OUT`/run dirs.
3. Every number quoted in README/EXPORT/MEASUREMENTS must point at a log or `results/` file with its power, cap and
   PCM. Package/model name in docs = `ds41-1017-engram-mxfp4`.
4. Commit with the identity configured in the checkout. Push only with the maintainer's explicit OK (each public push).

### B.4 Code review of the fork

1. Give fresh reviewers the diff and the files, not your conclusions. Two passes: first "bugs and wrong
   implementations" (including unadopted opt-in paths), then "what could be implemented better".
2. Verify every finding against the code before fixing; discard the ones that do not reproduce.
3. Fix, rebuild (`cargo build --release` for Helios, re-export for recipe/kernel changes), re-run `verify_fix`.
4. Record verdicts (fixed / no change needed / tabled) in the commit message.

### B.5 Publishing a release

1. Validate the state (B.1/B.2), commit on `ds41-tp3-v3`, push to Gitea.
2. Tag it `flare-tp3-YYYYMMDD` (annotated: settings and measurements in the message), push branch and tag to `public`,
   and write the GitHub Release note (what changed, measured numbers, what was not adopted, link to the recipes folder).
3. The recipes folder is dated by the day the change was made (release convention): a new
   `DeepSeek-V4.1-Flash Flare-tp3-YYYYMMDD` folder per release day, copied from the previous one and edited; earlier
   folders stay as published. Its README describes the changes **since the previous folder** (not since the first
   one): intro list, "Coming from <previous>" migration notes, headline table labelled with the builds it holds; the
   cumulative feature description stays as "What this build contains". *Incident:* the 20261108 folder shipped for an
   hour with the 20261107 intro ("supersedes 20261105, +6-8%"). Regenerate its `source.patch` from the tag (`git diff --binary 972e28e44 <tag>`), verify
   on a clean upstream worktree (tree-identical), cite the tag in the folder docs, add the row to the top-level
   README, run the hygiene grep, commit, push `origin` (GitHub) and `gitea`. The folder and the tag must always agree.

### B.6 Sync from upstream (deliberate, never automatic)

1. `git fetch origin` and read `git log --oneline <merge-base>..origin/main` for anything that touches the DeepSeek
   recipe, Helios memory/Engram/package loading, b12x pins or the CLI. Note it in the status report.
2. While the public fork has no open upstream PRs, rebase: `git rebase origin/main ds41-tp3-v3` in a worktree,
   resolve conflicts, keep commit messages. After PRs are open from the public fork, merge instead
   (`git merge origin/main`) so published hashes stay stable.
3. Re-apply `tp3-patches/` to the venv if `uv.lock` or the FlashInfer pin changed (`uv sync`, `ignition.patches`,
   then the patches).
4. Validate before trusting it: `cargo build --release`, the 3-GPU layer test, a full export, `verify_fix` (0 true
   corruptions), one alternating A/B round vs the current default. Then commit, push to Gitea, and only then update
   the release folder (`source.patch` is regenerated from the new base).

### B.7 Upstream watch (what NVIDIA's branches hold for us, 2026-11-08)

- `astra/ds41-brrr-poc` (10-28, research-only): NVIDIA's own fused layer tail, same stage range as ours, kernel in
  FlashInfer's experimental tree, TP4 all-resident with native scales, selected per layer by a race against the stock
  path after FP64 per-operator bound checks and a held-out input; the stock path won at the one 8-row graph they tried,
  the whole-model export never finished, no tok/s. Our port is further along (BRRR-40 c1 +6.5%, c4 +2.2%, gated parked
  layers at c8). Borrowed: FP64 bounds + held-out check for our layer test; native-replay fixtures are a candidate. A
  contribution of ours would have to live in that shape: kernel in FlashInfer, a Flare component with a CLI flag in the
  signed rebuild manifest, and a per-layer race it must win.
- `fix/vocabulary-parallel-top-p`, `astra/startup-weight-storage-upstream-20261104`: already in our base in merged
  form (the explicit top_p range our bootstrap adds to the server config is now redundant; proposal 03 can drop it).
- `fix/checkpoint-gds-capability-probe`: forces the bounce read mode this host already uses; arrives with the next sync.
- Everything else (GLM, Windows, RoCE, multi-host, scheduler) does not touch our path.

---

## C. Tips for agents

1. **Check the GPUs before using them.** `nvidia-smi --query-gpu=memory.used --format=csv` ; if the server is up you
   have ~600 MiB free and nothing CUDA will start. Ask or coordinate before stopping the server; other agents may be
   benchmarking.
2. **Long jobs run as tracked background jobs** (`run_in_background`), each with its own log file; never a foreground
   command that can exceed the timeout. Chain scripts need `chmod +x` and `set -euo pipefail`.
3. **Save work incrementally.** Agents get killed by rate limits and context resets. Commit-ready files, patch files
   and result JSON on disk survive; a plan in your head does not. Write results to a file as soon as a measurement
   ends.
4. **Do not spawn a duplicate agent or chain** for a task that is already running; send the existing agent a message,
   and before launching a chain script list the running ones with `ps -eo pid,etime,args | grep '[c]hain-'` (never
   `pgrep -f`/`pkill -f` with a pattern that is in your own command line: that kills your shell, 11-08 again). Two
   chains driving the same compose project or server port corrupt each other's results. *Incident (11-08):* a second
   vLLM test chain was launched while the first was still in a restart loop; both brought the same container up and
   down and one of them lost both verify runs.
5. **Worktrees for parallel code work.** Multiple agents editing one checkout corrupt each other's diffs. One worktree
   per task; only `~/flare-up-v3` serves.
6. **Stop and look at the command line before killing processes**: `pgrep -f` matches your own shell. Use the `ps |
   awk` form in A.2.
7. **Read the memory notes first** (`MEMORY.md` in Claude Code's per-project memory folder for this working directory): they hold the
   closed ideas and the measured verdicts, so you do not rerun a dead end.
8. **Report numbers with their context**: package id, power, BUDGET, PCM, concurrency, context length, run count.
   "c1 +4%" alone is not a result.
9. **Docker (vLLM era, still used for comparisons):** one `docker compose` command at a time; prompt logprobs one
   request at a time; MMLU-style evals need 2,048 output tokens.
10. **Dynamo-traced code**: anything in the recipe's forward runs under torch.compile during export. No `data_ptr()`
    comparisons, dict writes or Python-side side effects there; use a custom op (runs at capture time) or a finalizer.
11. **Upstream rules** (AGENTS.md, CONTRIBUTING.md): no env-var-driven behaviour in Flare/Helios/components/recipes,
    no offload policy selected by model identity. Anything meant for a proposal must be a declaration, config value or
    CLI flag. The fork's `FLARE_DS41_*` switches are allowed only because they stay in the recipe + `export.sh`.
12. **Be honest about verdicts.** "Level" is a valid and common result; most fusions here measured level because the
    kernels overlap on side streams. Do not stretch a 1-run +4% into an adoption.
13. **Size KV with the dry run, not a load.** `flare run <pkg> <serve flags> --dry-run` prints each rank's memory plan,
    the KV tokens and snapshot slots in seconds, without a CUDA context, while the server keeps serving. Use it before
    a sweep, when choosing serve flags, and before crediting a build with a KV change (D.1 rule 4). Accuracy:
    exact on the five DeepSeek-V4.1-Flash TP3 loads it was calibrated on; 0.4 % low (TP3) and 1.3 % low (TP2) on
    GLM-5.3-Flash, which was not in the calibration, so read it as a slightly conservative estimate for other models.
    `--dry-run-free-memory` replaces the assumed free memory (driver total minus 606 MiB) when a GPU is shared.
    Dry runs are cheap, so answer KV questions with them before an export: c16 graphs cost ~650K KV against c8, and
    a serve-time verification budget below the export's drops the unused verify graphs.
14. **Hide the GPUs from CPU test runs** while anything serves or benchmarks: prefix `CUDA_VISIBLE_DEVICES=`. Some
    `*_cuda` tests run whenever a GPU is visible, not only with `FLARE_RUN_CUDA_TESTS=1`. *Incident (11-08):* a CPU
    test pass put a CUDA test onto the GPUs for a few seconds in the middle of a benchmark sweep.
15. **Never edit a chain script that is running or waiting;** bash reads scripts as it goes. Kill and relaunch it, or
    write a new file and `mv` it over the old path (new inode) for a script that has not started yet. Gate each
    follow-on chain on a success marker, not a "done" marker. *Incident (11-08):* a comparison chain waited for
    "export done", which the failed TP3 export also wrote, and benchmarked TP2 alone.
16. **A new rank or head count meets kernel limits that tests on other shapes never hit.** Check the qualification
    lists before exporting: b12x's BF16 x MXFP8 batched GEMM takes 8, 16 or 32 heads only; the sparse-MLA kernel
    mis-tiles 22 heads; the shipped TP3 runtime collective ladder stops at 24 MiB (a 4096-row prefill at hidden 4096
    reduces 32 MiB). Check per-layer formats too: a draft layer can be stored differently from the target layers
    (GLM-5.3-Flash FP4-Local: layer 45's `kv_b_proj` is MXFP8, the target layers' are BF16). Each of these failed a
    TP3 export about 11 minutes in; a mini export or a format scan of the index finds them sooner.
17. **Speculation settings trade tokens per step against steps per second.** Read both before choosing. *Example
    (GLM-5.3-Flash TP3, 5 drafts):* adaptive verification off raised tokens per step from 2.23 to 2.72 and cut c1
    steps/s from 112 to 76, i.e. -16 % tok/s; alpha 0.6 reached 2.49 tokens per step at level tok/s.
18. **One Flare binary serves packages from other branches** when their Rust is unchanged: check
    `git diff --stat <base> HEAD -- flare helios heliopause` before building a second binary.
19. **Chain wait-conditions and failure watchers.** Never wait for the default-restart script to exit: it stays
    alive as the server's parent; wait for `/v1/models` instead. Failure watchers must pre-seed the log's existing
    lines (old failures re-trigger otherwise) and match failure words only, not words such as "skipped" that normal
    verdict lines contain.
20. **Profiling Flare** (one process, CUDA graphs, no NVTX ranges): start the server under `nsys launch` with
    collection off, then `nsys start` / `nsys stop` around a steady decode window; trace kernels only, with
    per-node CUDA graph tracing (`--cuda-graph-trace=node`), or the graph shows up as one opaque launch.
21. **Shell environment traps.** `~/.local/bin/env` (uv's sourced-only script) sits first on `PATH` and shadows
    `/usr/bin/env`, so `env VAR=x cmd` silently runs nothing: write `VAR=x cmd` or `/usr/bin/env VAR=x cmd`. The login
    shell also exports `IMG=<another vLLM image>`: a chain variable named `IMG` (`${IMG:-default}`) silently picks it up
    (2026-11-09: the vLLM A/B began pulling the wrong image). Give script variables specific names.
22. **One GPU queue, not parallel chains.** Queue GPU jobs through `/tmp/gpuq-runner.sh` (log `~/bench-results/gpuq.log`):
    an item runs only after `/tmp/gpuq/ready-<name>` exists, which is touched after its prep is reviewed; `done-<name>` is
    written after it ran; moving `ready-*` into `/tmp/gpuq/held/` pauses the queue. Edit the runner only while it is idle
    (kill it, edit, relaunch): bash reads a running script incrementally.
23. **Upstream reproductions need a pristine tree.** The fork venv carries `tp3-patches/` (including the tiny-decode
    clamp), so a bug check on upstream needs its own clone and venv (`uv sync --locked --extra prepare --extra test`).
    For correctness-only exports use `--autotune-effort off --no-mtp` (minutes instead of hours; without MTP every
    decode step is a 1-row launch). DeepSeek-V4 (`deepseek4`) rejects `--language-model-only` (no separable vision
    encoder), and on main 8c2e782 its export needs `engram` dropped from its `LayerWeights` entry in
    `ignition/components/weights.py` (hyperflare#249 notes it).
24. **DeepSeek request options.** The DeepSeek frontends accept `chat_template_kwargs` `enable_thinking`/`thinking_mode`,
    not `thinking` (HTTP 400 "unsupported DeepSeek prompt option").

---

## D. Rules and the incidents behind them

### D.1 Measuring speed

1. **A/B in one session, alternating.** Default, candidate, default, candidate (3 runs each, same power, same serve
   flags, same host prefix cache). Never compare against numbers from another day, another power limit or another
   cap. *Incident:* the ungated parked BRRR build looked 3.6% slower at c1 against the default's earlier runs; the
   alternating A/B showed c1 identical (296 vs 296).
2. **Single runs decide nothing.** Run-to-run spread is 2-3% at c1 and c8 and more in coding. *Incident:* the
   route-reuse quick bench showed c1 +4.7%; three user runs showed level. The first settings sweep crowned alpha 1.0 /
   throughput on one lucky run because the ranking mixed single runs with multi-run averages.
3. **Judge by tok/s, but read steps/s and tokens/step next to it.** A row cap or BRRR gate shifts tokens per step,
   so steps/s alone overstates a change (*incident:* BRRR at c8 showed higher steps/s but lower tok/s). The converse
   also happens: tok/s moves while steps/s does not, which means draft acceptance (sampling at temperature 1) moved,
   not the build (*incident, 11-08:* the config-object package looked 1.5% slower at c8 over 4 rounds at equal or
   higher steps/s; tok/step 15.8-16.8 vs 16.3-17.7 explained all of it). Chain logs print both via
   `tp3-tools/bench_cell.py`; `tp3-tools/bench_steps.py` gives the full per-cell table.
4. **Know the confounders before reading a KV or speed number:** power and offsets; `BUDGET` 24 vs 128 (the cap
   drops 13 speculative graphs, ~76 MiB/rank: 2,373,760 KV at 128 vs 2,461,184 at 24 for the same package);
   `--prefix-cache-memory` (32G costs ~10 MiB device headroom); whether the bench respects EOS. *Incident:* three
   builds were credited with "+58K / +73K / +87K KV" that was entirely the serve flags; the first explanation ("the
   32G prefix cache") was itself wrong and only the KV dry run (`flare run ... --dry-run`) settled it. Run the dry
   run before attributing any KV change to a build.
5. **Know the two background variances before calling a delta real.** Export-to-export (identical code, tuning
   races): 0.5-1% at c8 (11-08: 080909 vs its re-export 081134, 983 vs 988 tok/s over 4 rounds). Session-to-session
   for the same package: ~2% (080909 ran 986-1003 one session, 979-991 the next). A delta inside these bands needs a
   re-export or another session before it means anything; one re-export of the baseline is the cheapest check.
6. **Bisect a regression before dropping a whole change set.** Stage 1 measured -2% at c8; one of its six parts (an
   added PDL trigger) was the entire cost, found with one extra export. The other five were kept.
7. **Trace estimates are upper bounds.** Attention-side kernels overlap on side streams; a kernel that looks like 6%
   of the step may be hidden. Measure per-stream overlap before building a fusion. *Incidents:* BRRR prologue
   (estimated -5%, measured 0), sparse decode (estimated 2-3%, level), fused qkv_a norm (2.6% on paper, level).
8. **`verify_fix` "natural-uncertainty flags" are noise.** One build gave 4, 8, 4, 27, 16, 17 across runs. Only
   `TRUE corruptions` counts; natural-EOS benches and the qualification suite decide quality.
9. **Ignore-EOS bench loops are a bench artifact**, not corruption. Re-check with `--respect-eos` (always 0 loops).

### D.2 Exporting and serving

1. **Mini export first only for risky kernel or wiring changes**; settings and placement changes go straight to a
   full export (working policy). A mini run catches compile/trace/wiring failures, not speed.
2. **Warm the tuning cache before an H342 export.** A cold tuning cache runs out of VRAM at HOT=342; export HOT=300
   first. Needed again after every tuning version or draft-count change, and for a new `MAXC`: the c16 package's
   new verify shapes ran out of memory in tuning at HOT=342 until a HOT=300 c16 export had warmed the cache.
3. **The GPUs are exclusive.** Stop the server before tests, traces or exports and wait for memory.used < 1000 MiB.
4. **Never `pgrep -f`/`pkill -f` a pattern that appears in your own command line.** It killed the working shell
   three times. Use `ps -eo pid,args | awk '$2=="./flare/target/release/flare" && $3=="run"'`, or kill by PID.
5. **`chmod +x` new scripts, including ones generated with `sed ... > new.sh` or a heredoc**, and check the exit code of a
   detached launch (a `Permission denied` exit 126 leaves no log and no process). Run chains as tracked background jobs. **Always `/usr/bin/env`, never bare `env`:** uv
   puts its own `env` shim first on PATH and a command run through it exits 0 having done nothing (the release
   scripts already say so; the 11-08 bisect chain lost a run to it anyway). *Incident:* a confirmation chain "ran" three
   times with an empty log (exit 126) and the launch method was misdiagnosed first.
6. **One compose command at a time.** `down` racing `up` leaves "removal already in progress" and a hung start.
7. **vLLM prompt logprobs one request at a time**; MMLU-style evals need 2,048 output tokens (20% of replies reason
   first; 207 answers were unparsed with an 8-token cap).
8. **No `flare models gc` during an export.** 
9. **Check free space on the package store before every export.** A GLM-5.3-Flash TP3 variant writes about 73 GB of new
   blobs (experts in a new layout share nothing), a DeepSeek re-export about 4 GB. Version files are hardlinks into the
   shared blob store, so `models list` shows almost nothing reclaimable per version; space comes back only through
   `flare models remove` plus `flare models gc` (never during an export). Archive first with `rsync -aH` of the whole
   store to the archive drive, check the archive lists every version, then remove and gc. *Incident (11-08):* the boot
   drive reached 100 % and a GLM export failed while writing its package ("No space left on device", reported by the
   other ranks as a broken pipe); gc of orphans from failed exports freed 93 GB, archiving 19 non-default versions 250 GB.
12. **`/tmp` is a 126 GB tmpfs: it is RAM.** Never point a large or long-running log, download, clone or checkpoint
    at `/tmp`; use `~/ds41-tp3/` or `/m2-2/`. `aria2c --log` writes at debug level unless `--log-level=warn` is given.
    *Incident (11-08):* a 30 GB aria2c debug log in `/tmp` starved the vLLM TP3 image of pinned host RAM (driver
    "out of memory" while pinning the Engram tables, kernel OOM kills, a restart loop) during the .11/.12 check.
13. **A package exported into a separate `--config` package store is signed with that store's development key.**
    Serve it with `--publisher-keys <store>/development-publisher.json`; the first use also asks for explicit key
    approval and the checkpoint license acceptance. Copying the default store's signing key into the other store
    before exporting avoids both.
11. **Exports from a worktree must put the worktree first on the import path.** The shared venv's `ignition` console
    script imports the editable install, which points at `~/flare-up-v3`, whatever the current directory is; pytest
    imports from the cwd, so tests look right while exports silently build the main checkout's code. Run
    `PYTHONPATH=$PWD export.sh` (or `.venv/bin/python -m ignition export`) and log `ignition.__file__` next to the
    package id. *Incident (11-08):* the "stage 2" and "vision staging" packages were stage 1 and plain v3; a
    "compiled cache hit" for a graph that should have changed and a missing progress line were the tells.
9. **Scripts that are running must not be edited in place**; copy to a new name (`sweep2.sh`) and start that.
   **Never switch branches in a repository a script is running from**: `git checkout` of a branch that lacks the
   folder deletes and recreates it, the running process keeps a dead cwd, and every relative path fails from then on.
   Use a worktree per branch (`~/rtx-recipes-proposals` for `proposals`). *Incident (11-08):* a qualification run lost
   hotel-lights, retrieval, long generation and stress to a branch switch made to update a proposal file.
   **A chain's "wait for empty GPUs" loop only knows the processes it stops.** If a manual test leaves a container or
   server on the GPUs, the chain waits forever with no log line (`sleep 2` child, no output). Before launching a chain,
   check `nvidia-smi` and `docker ps`; never run a chain's steps by hand while another chain is queued on the GPUs.
10. **Qualification tooling quirks**: `long_context.py` asserted power ≤ 400 W (now 600) and refused existing output
    dirs; LiveBench needs `--livebench-release-option 2024-11-25`; docker engine crashes restart the container, so
    re-check which model is up after an OOM.

### D.3 Kernel and runtime code

1. **Keyed registries must not use shared pointers.** Scratch arenas hand every projection the same base address and
   sibling graphs reuse offsets. Key by the consumer's output buffer and scope the entry to the graph (arm before the
   producer, consume right after); reset on every non-deferred run. *Incident:* the fused qkv_a norm's split-K
   deferral would have skipped the reduce of any 4-way GEMM in the arena.
2. **Read GPU-produced inputs only after `pdl_wait()`.** *Incident:* sparse decode read per-row lengths before the
   wait; safe only because an unrelated kernel sat in between.
3. **Contract-check kernel constants against the model config at register time** (TOPK, MIX, ROPE, swiglu limit,
   routing scale, Sinkhorn/eps, scratch shapes and strides). A silent upstream scratch change must fail loudly.
4. **Cross-rank decisions must be identical** (which layers use BRRR, at which widths); divergence deadlocks the
   in-kernel all-reduce. Keep the all-gather check in construction code, never in a traced forward.
5. **Assert that parking stayed parking.** b12x's repack is in place for our geometry but would copy to VRAM for a
   ceil-tiled one (TP4). The post-`prepare_weights` check turns silent over-commit into an export error.
6. **Count mixed VRAM/host allocations by their device bytes at every site** (six in Helios; three were missed in the
   first pass, one legacy branch in the second).
7. **No `data_ptr()` or registry work inside Dynamo-traced code**; use a custom op or a binding finalizer.
8. **Every "identical" claim needs a bit-exact test; every equivalence test needs a negative control** shown to fail
   (`FLARE_DS41_BRRR_BREAK=rope`, `TP3Config.brrr_break`). The control's difference must clear the build's own
   run-to-run floor: the BRRR split-tail control (4-22K differing elements) did not clear the plain build's 10-24K
   and is report-only.
9. **Env-var-driven behaviour stays in the recipe and the CLI shim** (upstream rule, D/C.11). In the fork the only
   reader is `TP3Config.from_environment()`; components take the config object (`tp3=`), tests build it from their
   own environment.
10. **Save every site-packages edit as a patch in `tp3-patches/` immediately.** *Incident:* the b12x split-K deferral
    existed only in the venv for a day.
11. **A per-row mismatch in a kernel-vs-reference test is a routing or quantizer effect until proven otherwise.**
    Check which rows differ, the top-k key margins of those rows, and the noise of two independent activation
    quantizers before suspecting the kernel. *Incident:* the layer test's `mlp_output` cosine of 0.95-0.998 cost a
    debugging session; every bad row had a 6th/7th expert key margin ≤ 0.036 (the two mHC/RMSnorm implementations
    round the FFN input differently), and the residual ~5% on matched rows is two e4m3 32-group quantizers. The fix
    was in the test (probe the kernel's routing per row, reconcile near-ties, tolerance 0.995 on `mlp_output`), not in
    the kernel. Swaps chain across rows, so compare per row, not as a multiset.
12. **Rebuild and re-serve after Helios changes**: a Rust edit is not live until `cargo build --release` and a server
    restart; a recipe or kernel edit is not live until a new export.
13. **Several compile-time kernel variants in one process need `-fno-gnu-unique`** (or no function-local static
    guards in inline templates). GCC makes such guards GNU-unique symbols shared by every loaded `.so`, so only the
    first variant set its shared-memory attribute; the others failed with "layer kernel fits 0 CTAs per SM".

### D.4 Release folder

1. **No personal data**: grep for `/home/`, the hostname, the email and `~/`; delete `__pycache__`; scrub
   `hostname`/`uname` from bench JSON. *Incidents:* shipped `.pyc` with a home path, a result file with the machine
   name, a script writing to a literal `~/ds41-tp3/...`.
2. **Scripts write to a run directory, never into shipped results or RESULTS.md.**
3. **Model name = checkpoint directory name** (`ds41-1017-engram-mxfp4`) in docs, `serve.sh` and the view.
4. **`source.patch` must reproduce the fork tree on clean upstream**; verify with a temp worktree and
   `git ls-files -s | md5sum`. Quote the correct file count.
5. **Every number traces to a log or `results/` file**, with power, cap and PCM; say when it came from the pre-rebase
   fork.
6. **Commit with the checkout's configured identity** (agent memory notes hold it). Commit and push only when asked;
   public pushes need an explicit OK; history rewrites need an explicit OK and a force-push warning.

### D.5 Reviews

1. Two passes with fresh reviewers who have not seen the author's conclusions; the second pass found the legacy
   accounting branch and the cross-graph aliasing the first missed.
2. Separate "bug hunt" from "how could this be better".
3. Unadopted experiments stay only behind default-off switches, with the verdict in the docstring, and get reviewed
   like production code (they produced two latent bugs).

### D.6 Quality checks per change (tiered)

Small changes get quick checks before they become the local default; the full suite runs once, on the exact package
being published, before any public release. Every change stays behind its own switch, so a full-suite failure on a
combined build is found by exporting with switches toggled, not by reverting code.

0. **Measure the benefit first.** Before any quality check, show that the change improves what it is for (speed in an
   alternating A/B, KV in a dry run, prefill, host RAM) beyond the variances in D.1, with no loss elsewhere. A change
   that does not pay off stays behind its default-off switch with the verdict recorded, and gets no qualification run.
   Only correctness tests that guard the measurement itself (the layer test, the mini-export consistency check) run
   before the A/B, so a broken build is not timed.
1. **Every change:** its own targeted test (layer test with FP64 bounds and the negative control, or the mini-export
   `consistency.py` check), `verify_fix` with 0 TRUE corruptions, the 43-question eval and the 60 tool-call prompts.
2. **Decode-path changes** (experts, BRRR tail, sampling, drafts): add hotel-lights at c8 (short prompt, long
   generated reasoning; catches drift in the decode path).
3. **Prefill, attention or context changes** (prefill chunking, Engram lookups over the prompt, indexer, KV writes):
   add hotel-lights and lavd at c8 (lavd is a long structured CSV context; it exercises prefill and long-context
   consistency, which hotel-lights does not).
4. **Memory, KV or graph-layout changes:** add the c8 KV stress test, where they fail.
5. **Before a public release:** the full `qual_all.sh` suite (eval, tools, lavd and hotel at c8, retrieval 128K and
   900K, long generation, EOS, LiveBench, stress) on the package being published.
6. **Full suite on a single change before it becomes the default** when it touches sampling or verification math, the
   chat template or tool parsing, or when any quick check flagged it even once.

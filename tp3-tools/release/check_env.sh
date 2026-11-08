#!/bin/bash
# Preflight for building and serving this release. Read-only: it changes nothing and starts nothing on the GPUs.
#   usage: ./check_env.sh <fork checkout dir>    the Flare clone at 972e28e4 (EXPORT.md step 2), before or after source.patch
#   env:   CUDA_HOME     CUDA toolkit root (default /usr/local/cuda-13.1 if present, else the nvcc on PATH)
#          FLARE_CACHE   where packages and the kernel tuning cache go (default ~/.cache/flare)
#          MIN_FREE_GB   free space required on the drive holding FLARE_CACHE (default 900: checkpoints ~412 GB,
#                        package ~380 GB, tuning cache 60-80 GB; lower it if the checkpoints live on another drive)
#          GPUS          GPU indices you will export and serve on (default 0,1,2)
# Prints one PASS / WARN / FAIL line per check and exits 1 if anything FAILed. WARN lines are steps not done yet.
set -uo pipefail
FORK=${1:?usage: check_env.sh <fork checkout dir>}
HERE=$(dirname "$(readlink -f "$0")")
FLARE_CACHE=${FLARE_CACHE:-$HOME/.cache/flare}; MIN_FREE_GB=${MIN_FREE_GB:-900}; GPUS=${GPUS:-0,1,2}
BASE=972e28e44781d8d0c84d70ae434db92c36076a43
fails=0
pass(){ echo "PASS  $*"; }; warn(){ echo "WARN  $*"; }; fail(){ echo "FAIL  $*"; fails=$((fails+1)); }
vge(){ [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }   # vge A B: version A >= B

# --- CUDA toolkit: 13.1+ with nvcc that targets SM120 (sm_120a). The export compiles the BRRR layer kernel with it.
if [ -z "${CUDA_HOME:-}" ] && [ -x /usr/local/cuda-13.1/bin/nvcc ]; then CUDA_HOME=/usr/local/cuda-13.1; fi
NVCC=${CUDA_HOME:+$CUDA_HOME/bin/nvcc}
[ -n "$NVCC" ] && [ -x "$NVCC" ] || NVCC=$(command -v nvcc || true)
if [ -z "$NVCC" ]; then
  fail "nvcc not found: install CUDA 13.1 and export CUDA_HOME=/usr/local/cuda-13.1 PATH=\$CUDA_HOME/bin:\$PATH"
else
  v=$("$NVCC" --version | grep -oE 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)
  if vge "$v" 13.1; then pass "CUDA toolkit $v ($NVCC)"; else fail "CUDA toolkit $v at $NVCC; 13.1 or newer is required (12.x cannot target SM120)"; fi
  if "$NVCC" --list-gpu-code 2>/dev/null | grep -q '^sm_120'; then pass "nvcc accepts sm_120 (sm_120a)"; else fail "nvcc at $NVCC does not list sm_120"; fi
  if [ "${CUDA_HOME:-}" != "$(dirname "$(dirname "$NVCC")")" ]; then
    warn "CUDA_HOME is not set to this toolkit; before exporting: export CUDA_HOME=$(dirname "$(dirname "$NVCC")") PATH=\$CUDA_HOME/bin:\$PATH"
  fi
fi

# --- locked memory: Flare pins host memory (cold experts, Engram tables, prefix cache)
if [ "$(ulimit -l)" = unlimited ]; then pass "memlock unlimited"
else fail "ulimit -l is $(ulimit -l); add '<user> - memlock unlimited' in /etc/security/limits.d/90-memlock.conf and log in again"; fi

# --- toolchain
if command -v rustc >/dev/null; then
  v=$(rustc --version | awk '{print $2}')
  if vge "$v" 1.88; then pass "rustc $v (fork requires >= 1.88)"; else fail "rustc $v; the fork requires 1.88 or newer (rustup update)"; fi
else fail "rustc not found (install rustup)"; fi
if command -v cargo >/dev/null; then pass "cargo $(cargo --version | awk '{print $2}')"; else fail "cargo not found"; fi
if command -v uv >/dev/null; then pass "uv $(uv --version | awk '{print $2}')"; else fail "uv not found (https://docs.astral.sh/uv/)"; fi
if command -v patch >/dev/null; then pass "patch present (applies tp3-patches/)"; else fail "patch not found (install the patch package)"; fi
command -v git >/dev/null || fail "git not found"

# --- fork checkout and source.patch
if ! git -C "$FORK" rev-parse --git-dir >/dev/null 2>&1; then
  fail "$FORK is not a git checkout (EXPORT.md step 2)"
else
  head=$(git -C "$FORK" rev-parse HEAD)
  if [ "$head" = "$BASE" ]; then pass "fork checkout at upstream Flare ${BASE:0:8}"
  elif git -C "$FORK" merge-base --is-ancestor "$BASE" "$head" 2>/dev/null; then warn "fork HEAD ${head:0:8} is ahead of ${BASE:0:8} (fine if you committed source.patch on top of it)"
  else fail "fork HEAD ${head:0:8} is not ${BASE:0:8}: git checkout $BASE (fresh clone if coming from 20261105)"; fi
  if git -C "$FORK" apply --check --reverse "$HERE/source.patch" 2>/dev/null; then pass "source.patch applied"
  elif git -C "$FORK" apply --check "$HERE/source.patch" 2>/dev/null; then warn "source.patch not applied yet (it applies cleanly): git -C $FORK apply $HERE/source.patch"
  else fail "source.patch neither applied nor applicable: wrong base commit or local changes (git -C $FORK status)"; fi
fi

# --- Python environment: uv-managed Python 3.12, FlashInfer patches, ignition.patches
VENV=$FORK/.venv
if [ -x "$VENV/bin/python" ]; then
  pv=$("$VENV/bin/python" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
  case $pv in 3.12.*) pass "fork .venv Python $pv" ;; *) fail ".venv Python is $pv; Flare pins 3.12 (uv sync --locked --extra prepare --extra test)" ;; esac
  SP=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  IMPL=$SP/flashinfer/experimental/b12x/moe/fused_moe/_impl.py
  if [ ! -f "$IMPL" ]; then
    fail "FlashInfer is not installed in .venv (uv sync --locked --extra prepare --extra test)"
  else
    # The swiglu patch adds a guard to the tiny_decode selection in _impl.py; without it replies corrupt mid-way at TP3.
    if grep -q 'tiny_decode implements plain SiLU-GLU only' "$IMPL"; then pass "tiny_decode honors swiglu_limit (flashinfer-tiny-decode-swiglu.patch content present)"
    else fail "flashinfer-tiny-decode-swiglu.patch content missing in $IMPL (replies corrupt at TP3 without it)"; fi
    for p in "$HERE"/tp3-patches/*.patch; do
      b=$(basename "$p")
      if patch -R --dry-run -f -s -p1 -d "$SP" < "$p" >/dev/null 2>&1; then pass "$b applied"
      elif patch --dry-run -f -s -p1 -d "$SP" < "$p" >/dev/null 2>&1; then fail "$b not applied (applies cleanly): patch -p1 -d $SP < $p"
      else fail "$b neither applied nor applicable to the installed FlashInfer (wrong version? re-run uv sync --locked, then re-apply)"; fi
    done
  fi
  # python -m ignition.patches rewrites three files in place; these two added fragments are present once it ran.
  tb=$SP/triton/runtime/build.py; tl=$SP/torch/_inductor/runtime/static_triton_launcher.py
  if [ -f "$tb" ] && [ -f "$tl" ]; then
    if grep -q "calcsize('P')" "$tb" && grep -q '_run_cooperative' "$tl"; then pass "ignition.patches applied (torch Inductor and Triton fixes)"
    else fail "ignition.patches not applied: (cd $FORK && uv run --locked --extra prepare python -m ignition.patches); re-run after every uv sync"; fi
  else fail "torch/triton missing in .venv (uv sync --locked --extra prepare --extra test)"; fi
  if [ -x "$VENV/bin/ignition" ]; then pass "ignition CLI present"; else warn ".venv/bin/ignition missing (uv sync --locked --extra prepare --extra test)"; fi
  if [ -x "$VENV/bin/ninja" ]; then pass "ninja present (BRRR kernel compile)"; else warn ".venv/bin/ninja missing; the BRRR kernel compile needs it (uv sync --locked --extra prepare --extra test)"; fi
  tv=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("torch"))' 2>/dev/null || echo missing)
  fv=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))' 2>/dev/null || echo missing)
  echo "      torch $tv, flashinfer-python $fv (release: 2.14.1+cu132, 0.7.1; see versions.txt). A cu132 torch wheel next to the"
  echo "      13.1 toolkit is expected: the wheel ships its own CUDA libraries, nvcc 13.1 only compiles the BRRR kernel."
else
  warn "no .venv in $FORK yet: uv sync --locked --extra prepare --extra test, then ignition.patches and the FlashInfer patches (EXPORT.md step 2); re-run this check afterwards"
fi
if [ -x "$FORK/flare/target/release/flare" ]; then pass "flare binary built: $FORK/flare/target/release/flare"
else warn "flare binary not built yet: cargo build --release --manifest-path $FORK/flare/Cargo.toml"; fi

# --- disk
d=$FLARE_CACHE; while [ ! -d "$d" ]; do d=$(dirname "$d"); done
free=$(df -BG --output=avail "$d" | tail -1 | tr -dc 0-9)
if [ "$free" -ge "$MIN_FREE_GB" ]; then pass "$free GB free on the drive holding $FLARE_CACHE (need $MIN_FREE_GB)"
else fail "$free GB free on the drive holding $FLARE_CACHE; $MIN_FREE_GB GB needed (set MIN_FREE_GB if the checkpoints are on another drive)"; fi

# --- GPUs: 3 visible SM120 cards, idle. Kernel tuning captures the power limit and clocks in effect during the export.
if ! command -v nvidia-smi >/dev/null; then
  fail "nvidia-smi not found (NVIDIA driver 615.71.09 or any driver supporting CUDA 13.1)"
else
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
  n=0; bad=0
  IFS=, read -ra want <<< "$GPUS"
  while IFS=, read -r idx name cc used plim smmax memmax; do
    idx=${idx// /}; cc=${cc// /}; used=${used// /}; plim=${plim// /}; smmax=${smmax// /}; memmax=${memmax// /}; name=${name# }
    for w in "${want[@]}"; do
      [ "$w" = "$idx" ] || continue
      n=$((n+1))
      [ "$cc" = "12.0" ] || { fail "GPU $idx: compute capability $cc, need 12.0 (SM120, RTX PRO 6000 Blackwell)"; bad=1; }
      [ "$used" -lt 1024 ] || { fail "GPU $idx: $used MiB in use; stop the server and other GPU processes before exporting"; bad=1; }
      echo "      GPU $idx: $name · power limit $plim W · max clocks SM $smmax MHz, memory $memmax MHz · $used MiB used"
    done
  done < <(nvidia-smi --query-gpu=index,name,compute_cap,memory.used,power.limit,clocks.max.sm,clocks.max.memory --format=csv,noheader,nounits)
  if [ "$n" -ne 3 ]; then fail "$n of the GPUs in GPUS=$GPUS are visible; this release needs exactly 3"
  elif [ "$bad" = 0 ]; then pass "3 SM120 GPUs ($GPUS) visible and idle, driver $drv"; fi
  echo "      Export with the power limit and clock offsets you will serve with: the tuning cache records kernels measured"
  echo "      at those settings. The release numbers were taken at 450 W, +75 MHz core, +6000 MHz memory offset; 350 W is 10-14% slower."
fi

echo
if [ "$fails" = 0 ]; then echo "All checks passed."; else echo "$fails check(s) FAILED."; exit 1; fi

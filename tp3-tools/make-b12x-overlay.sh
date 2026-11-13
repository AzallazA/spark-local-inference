#!/bin/bash
# Build the b12x overlay GLM exports use (B12X_OVERLAY in glm53-export-tp3.sh): a copy of the venv's (already
# tp3-patches-patched) b12x plus tp3-patches/overlay/flashinfer-dcp-gather-pull-world3.patch, so the shared venv (also used
# by the DS41 exports) stays unchanged. World-3 pull gather: +7-8 % GLM a16 c1/c4/c8 (2026-11-09, pkg 092215).
#   tp3-tools/make-b12x-overlay.sh [OUT]     default OUT: <worktree>/.b12x-overlay
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
OUT=${1:-$ROOT/.b12x-overlay}
S=$("$ROOT/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
rm -rf "$OUT"; mkdir -p "$OUT/b12x"
cp -a "$S/flashinfer/experimental/b12x" "$OUT/b12x_tree"
find "$OUT/b12x_tree" -name __pycache__ -prune -exec rm -rf {} +
# Shim: the venv's b12x/__init__.py, rooted at the overlay tree.
sed 's#^_root = Path(_flashinfer.origin).parent / "experimental" / "b12x"#_root = Path(__file__).resolve().parent.parent / "b12x_tree"#' \
    "$S/b12x/__init__.py" > "$OUT/b12x/__init__.py"
grep -q 'b12x_tree' "$OUT/b12x/__init__.py"
patch -d "$OUT/b12x_tree" -p4 --forward < "$ROOT/tp3-patches/overlay/flashinfer-dcp-gather-pull-world3.patch"
echo "overlay ready: $OUT"

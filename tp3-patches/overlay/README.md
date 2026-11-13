Patches applied only to the b12x overlay (`tp3-tools/make-b12x-overlay.sh` -> `.b12x-overlay`, used by
`glm53-export-tp3.sh` via `B12X_OVERLAY`), not to the shared `.venv` (which the DS41 exports also use).

- `flashinfer-dcp-gather-pull-world3.patch`: at world 3 the PCIe DCP head gather pulls peers' staged rows instead of
  pushing into peers (payloads under 4 KiB per rank stay push). Switches: `B12X_PCIE_DCP_GATHER_PULL=auto|1|0` (auto =
  world 3 only), `B12X_PCIE_DCP_GATHER_PULL_MIN_BYTES` (4096); both are in the compile key (kernel version 15).
  Applies on top of `../flashinfer-dcp-a2a-world3.patch`. Measured 2026-11-09: head gather 923 -> 63 us at 48 rows;
  GLM a16 c1/c4/c8 +7.0/+8.2/+8.4 % (pkg gb202-tp3-202611092215 vs 082232, 3 alternating rounds); eval 43/43,
  tools 60/60, verify_fix 0, hotel c8 28/0 wrong/2 truncated.
- 2026-11-10 (branch glm53-tp2-update), same patch file: `B12X_PCIE_DCP_GATHER_PULL=1` now pulls at world 2 too (the
  generic pull plane with one peer; payloads under `_MIN_BYTES` keep the upstream TP2 push, moved unchanged into
  `_gather_push_tp2`). `auto` is unchanged (world 3 only), so TP2 exports stay on push unless `PULL=1` is exported.
  CPU check: world-2 push PTX identical to the unpatched venv, world-3 auto PTX identical to the 11-09 overlay; world-2
  pull compiles for both ranks and slot modes. Not measured on GPU yet (glm-tp2 chain stage G).

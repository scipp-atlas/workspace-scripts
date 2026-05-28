#!/bin/bash
# Full workflow: generate both workspace variants, run fits, export HS3 JSON.
#
# Usage:  bash workflow.sh [--seed N]   (default seed: 42)
set -euo pipefail
cd "$(dirname "$0")"
#source setup_local.sh

SEED=42
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

step() { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }

# ── 1. Generate workspaces ──────────────────────────────────────────────────
step "Generating workspaces (seed=$SEED)"
python3 make_workspace.py                         --seed "$SEED"
ok "simple_workspace.root"
python3 make_workspace.py --no-np                 --seed "$SEED"
ok "simple_workspace_nonp.root"
python3 make_workspace.py         --generic-bkg   --seed "$SEED"
ok "simple_workspace_generic.root"
python3 make_workspace.py --no-np --generic-bkg   --seed "$SEED"
ok "simple_workspace_generic_nonp.root"

# ── 2. Run fits ─────────────────────────────────────────────────────────────
step "Fitting: with NP"
bash run_simple_fit.sh simple_workspace.root

step "Fitting: without NP"
bash run_simple_fit.sh simple_workspace_nonp.root

step "Fitting: with NP and with generic dist backgrounds"
bash run_simple_fit.sh simple_workspace_generic.root

step "Fitting: without NP and with generic dist backgrounds"
bash run_simple_fit.sh simple_workspace_generic_nonp.root

# ── 4. Export HS3 ───────────────────────────────────────────────────────────
step "mu scans"
python3 muscan.py                                            --output muscan.json
ok "muscan.json"
python3 muscan.py --input simple_workspace_nonp.root         --output muscan_nonp.json
ok "muscan_nonp.json"
python3 muscan.py --input simple_workspace_generic.root      --output muscan_generic.json
ok "muscan_generic.json"
python3 muscan.py --input simple_workspace_generic_nonp.root --output muscan_generic_nonp.json
ok "muscan_generic_nonp.json"

# ── 5. Export HS3 ───────────────────────────────────────────────────────────
step "Exporting HS3 JSON"
python3 export_hs3.py --input simple_workspace.root      --verify
ok "simple_workspace.json"
python3 export_hs3.py --input simple_workspace_nonp.root --verify
ok "simple_workspace_nonp.json"
python3 export_hs3.py --input simple_workspace_generic.root --verify
ok "simple_workspace_generic.json"
python3 export_hs3.py --input simple_workspace_generic_nonp.root --verify
ok "simple_workspace_generic_nonp.json"

# ── Summary ──────────────────────────────────────────────────────────────────
step "Done"
printf '  Workspaces  : simple_workspace.root  simple_workspace_nonp.root\n'
printf '  Fit results : output_simple/simple_workspace_result.root\n'
printf '                output_simple/simple_workspace_nonp_result.root\n'
printf '  Fit logs    : output_simple/simple_workspace_fit.log\n'
printf '                output_simple/simple_workspace_nonp_fit.log\n'
printf '  HS3 JSON    : simple_workspace.json  simple_workspace_nonp.json\n'


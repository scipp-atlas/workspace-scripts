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

# Each entry will be: <stem>|<make_workspace flags>
VARIANTS=(
    "simple_workspace|"
    "simple_workspace_nonp|--no-np"
    "simple_workspace_generic|--generic-bkg"
    "simple_workspace_generic_nonp|--no-np --generic-bkg"
    "simple_workspace_generic_fixshape|--generic-bkg --fix-bkg-shape"
    "simple_workspace_generic_poly|--generic-bkg --bkg-form poly"
    "simple_workspace_gensig|--generic-sig"
    "simple_workspace_gensig_10_channels|--generic-sig --num-channels 10"
    "simple_workspace_gensig_30_channels|--generic-sig --num-channels 30"
    "simple_workspace_poisson|--constraint poisson"
    "simple_workspace_noconstr|--constraint none"
    "simple_workspace_0-1x_events|--yield-sf 0.1"
    "simple_workspace_10x_events|--yield-sf 10"
    "simple_workspace_100x_events|--yield-sf 100"
    "simple_workspace_1_channels|--num-channels 1"
    "simple_workspace_2_channels|--num-channels 2"
    "simple_workspace_3_channels|--num-channels 3"
    "simple_workspace_4_channels|--num-channels 4"
    "simple_workspace_5_channels|--num-channels 5"
    "simple_workspace_10_channels|--num-channels 10"
    "simple_workspace_30_channels|--num-channels 30"
)

# ── 1. Generate workspaces ──────────────────────────────────────────────────
step "Generating workspaces (seed=$SEED)"
for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r stem flags _constr <<< "$entry"
    python3 make_workspace.py $flags --seed "$SEED" --output "workspaces/${stem}.root"
done

# ── 2. Run fits ─────────────────────────────────────────────────────────────
step "Running fits"
for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r stem flags _constr <<< "$entry"
    # `if` disables set -e here so one failing fit doesn't abort the whole run.
    if bash run_simple_fit.sh "workspaces/${stem}.root"; then
        ok "fit: ${stem}"
    else
        printf '\033[1;31m  ✗ fit failed: %s\033[0m\n' "${stem}"
    fi
done

# ── 3. Plot workspaces (best-fit / minimum-NLL parameters) ───────────────────
# run_simple_fit.sh floats mu_sig and writes the unconditional best fit to
# output_simple/<stem>_result.root; overlay those post-fit parameters.
step "Plotting workspaces (best-fit parameters)"
for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r stem flags _constr <<< "$entry"
    python3 plot_workspace.py "workspaces/${stem}.root" \
        --fit-result "output_simple/${stem}_result.root"
    ok "plots/${stem}_channels.png"
done

# ── 4. mu scans ─────────────────────────────────────────────────────────────
step "mu scans"
for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r stem flags constr <<< "$entry"
    scan="scans/muscan${stem#simple_workspace}.json"  # "" -> muscan.json, "_nonp" -> muscan_nonp.json, ...
    python3 muscan.py --input "workspaces/${stem}.root" --output "$scan"
    ok "$scan"
done

# ── 5. Export HS3 JSON ───────────────────────────────────────────────────────────
step "Exporting HS3 JSON"
  for entry in "${VARIANTS[@]}"; do
      IFS='|' read -r stem flags _constr <<< "$entry"
      python3 export_hs3.py --input "workspaces/${stem}.root" --verify
      ok "workspaces/${stem}.json"
  done

# aux-stripped exports (json-only; reuse the .root files above, paired with the
# standard scans in eval_simple_muscan.py)
python3 export_hs3.py --input workspaces/simple_workspace.root --no-aux-constraints --verify
ok "simple_workspace_noaux.json"
python3 export_hs3.py --input workspaces/simple_workspace_generic.root --no-aux-constraints --verify
ok "simple_workspace_generic_noaux.json"

# ── Summary ──────────────────────────────────────────────────────────────────
step "Done"
printf '  Workspaces  : %d variants (simple_workspace*.root)\n' "${#VARIANTS[@]}"
printf '  Fit results : output_simple/<stem>_result.root\n'
printf '  Fit logs    : output_simple/<stem>_fit.log\n'
printf '  Plots       : plots/<stem>_channels.png  (best-fit overlay)\n'
printf '  mu scans    : muscan*.json\n'
printf '  HS3 JSON    : simple_workspace*.json  (+ *_noaux.json)\n'


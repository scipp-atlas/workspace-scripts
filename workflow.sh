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

# Build a fully-specified workspace name from a set of make_workspace.py flags.
# Every option is spelled out in a fixed order, so comparing two names shows
# exactly which aspects differ. Segments (with make_workspace.py defaults):
#   <N>ch            number of channels                (--num-channels, default 3)
#   bkg<Pdf><Form>   background pdf + form             (RooExp | GenExp | GenPoly)
#   sig<Pdf>         signal pdf                        (Gauss | Generic; --generic-sig)
#   shape<State>     background shape                  (Float | Fixed; --fix-bkg-shape)
#   np<State>        width nuisance parameter          (On | Off; --no-np)
#   constr<Type>     constraint form                   (Gauss | Poisson | None; --constraint)
#   yield<SF>x       yield scale factor                (--yield-sf, default 1)
canonical_stem() {
    local channels=3 bkg_pdf="RooExp" bkg_form="exp" sig_pdf="Gauss"
    local bkg_shape="Float" np="On" constr="gauss" yield_sf="1"

    set -- $1
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-np)         np="Off";            shift ;;
            --generic-bkg)   bkg_pdf="Gen";       shift ;;
            --bkg-form)      bkg_form="$2";       shift 2 ;;
            --generic-sig)   sig_pdf="Generic";   shift ;;
            --fix-bkg-shape) bkg_shape="Fixed";   shift ;;
            --constraint)    constr="$2";         shift 2 ;;
            --yield-sf)      yield_sf="$2";       shift 2 ;;
            --num-channels)  channels="$2";       shift 2 ;;
            *)               shift ;;
        esac
    done

    local bkg
    if [[ $bkg_pdf == RooExp ]]; then
        bkg="bkgRooExp"                                   # RooExponential (form is always exp)
    elif [[ $bkg_form == poly ]]; then
        bkg="bkgGenPoly"                                  # RooGenericPdf, polynomial
    else
        bkg="bkgGenExp"                                   # RooGenericPdf, exp(tau*x)
    fi

    # Capitalized constraint label (avoid ${constr^}; bash 3.2 on macOS lacks it).
    local constr_label
    case $constr in
        gauss)   constr_label="Gauss" ;;
        poisson) constr_label="Poisson" ;;
        none)    constr_label="None" ;;
        *)       constr_label="$constr" ;;
    esac

    printf '%sch_%s_sig%s_shape%s_np%s_constr%s_yield%sx' \
        "$channels" "$bkg" "$sig_pdf" "$bkg_shape" "$np" \
        "$constr_label" "${yield_sf/./p}"
}

# Each entry is the set of make_workspace.py flags; the descriptive name is
# derived from the flags via canonical_stem().
VARIANTS=(
    ""
    "--no-np"
    "--generic-bkg"
    "--no-np --generic-bkg"
    "--generic-bkg --fix-bkg-shape"
    "--generic-bkg --bkg-form poly"
    "--generic-sig"
    "--generic-sig --num-channels 10"
    "--generic-sig --num-channels 30"
    "--constraint poisson"
    "--constraint none"
    "--yield-sf 0.1"
    "--yield-sf 10"
    "--yield-sf 100"
    "--num-channels 1"
    "--num-channels 2"
    "--num-channels 4"
    "--num-channels 5"
    "--num-channels 10"
    "--num-channels 30"

    # ── Multi-axis combinations (each mixes several flags at once) ───────────
    # Both pdfs generic
    "--generic-bkg --generic-sig"
    "--generic-bkg --bkg-form poly --generic-sig"
    "--generic-bkg --generic-sig --no-np"
    "--generic-bkg --generic-sig --num-channels 10"
    "--generic-bkg --bkg-form poly --generic-sig --num-channels 30"
    # Generic signal mixed with other axes
    "--no-np --generic-sig"
    "--generic-sig --fix-bkg-shape"
    "--generic-sig --no-np --num-channels 10"
    "--generic-sig --yield-sf 10 --num-channels 5"
    # Background shape fixed, combined with other axes
    "--fix-bkg-shape"
    "--fix-bkg-shape --no-np"
    "--generic-bkg --bkg-form poly --fix-bkg-shape"
    "--generic-bkg --fix-bkg-shape --num-channels 10"
    # Constraint form mixed with pdf / channel / yield axes (NP kept on)
    "--constraint poisson --generic-bkg"
    "--constraint poisson --num-channels 10"
    "--constraint poisson --yield-sf 10"
    "--constraint none --generic-bkg"
    "--constraint none --generic-sig"
    "--constraint none --num-channels 10"
    # Yield scale mixed with channel / pdf axes
    "--yield-sf 0.1 --num-channels 10"
    "--yield-sf 10 --num-channels 10"
    "--yield-sf 100 --generic-bkg"
    "--yield-sf 10 --no-np"
    # Kitchen sink: every axis pushed off its default at once
    "--generic-bkg --bkg-form poly --generic-sig --fix-bkg-shape --constraint poisson --yield-sf 10 --num-channels 10"
)

# ── 1. Generate workspaces ──────────────────────────────────────────────────
step "Generating workspaces (seed=$SEED)"
for flags in "${VARIANTS[@]}"; do
    stem=$(canonical_stem "$flags")
    python3 make_workspace.py $flags --seed "$SEED" --output "workspaces/${stem}.root"
done

# ── 2. Run fits ─────────────────────────────────────────────────────────────
step "Running fits"
for flags in "${VARIANTS[@]}"; do
    stem=$(canonical_stem "$flags")
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
for flags in "${VARIANTS[@]}"; do
    stem=$(canonical_stem "$flags")
    python3 plot_workspace.py "workspaces/${stem}.root" \
        --fit-result "output_simple/${stem}_result.root"
    ok "plots/${stem}_channels.png"
done

# ── 4. mu scans ─────────────────────────────────────────────────────────────
step "mu scans"
for flags in "${VARIANTS[@]}"; do
    stem=$(canonical_stem "$flags")
    scan="scans/${stem}_muscan.json"
    python3 muscan.py --input "workspaces/${stem}.root" --output "$scan"
    ok "$scan"
done

# ── 5. Export HS3 JSON ───────────────────────────────────────────────────────────
step "Exporting HS3 JSON"
  for flags in "${VARIANTS[@]}"; do
      stem=$(canonical_stem "$flags")
      python3 export_hs3.py --input "workspaces/${stem}.root" --verify
      ok "workspaces/${stem}.json"
  done

# aux-stripped exports (json-only; reuse the .root files above, paired with the
# standard scans in eval_simple_muscan.py)
default_stem=$(canonical_stem "")
generic_stem=$(canonical_stem "--generic-bkg")
# python3 export_hs3.py --input "workspaces/${default_stem}.root" --no-aux-constraints --verify
# ok "${default_stem}_noaux.json"
# python3 export_hs3.py --input "workspaces/${generic_stem}.root" --no-aux-constraints --verify
# ok "${generic_stem}_noaux.json"

# ── Summary ──────────────────────────────────────────────────────────────────
step "Done"
printf '  Workspaces  : %d variants (workspaces/<stem>.root)\n' "${#VARIANTS[@]}"
printf '  Fit results : output_simple/<stem>_result.root\n'
printf '  Fit logs    : output_simple/<stem>_fit.log\n'
printf '  Plots       : plots/<stem>_channels.png  (best-fit overlay)\n'
printf '  mu scans    : scans/<stem>_muscan.json\n'
printf '  HS3 JSON    : workspaces/<stem>.json  (+ <stem>_noaux.json)\n'


#!/bin/bash
# Full workflow over the whole variant matrix: generate each workspace, run an
# unconditional fit, plot the channels, run a mu scan, and export HS3 JSON.
#
# Requires the ROOT/quickFit environment; source setup_local.sh first.
#
# Usage:  bash workflow.sh [--seed N] [--steps LIST]   (default seed: 42)
#
#   --steps LIST   comma-separated subset of stages to run, in the fixed
#                  pipeline order: ws,fit,plot,scan,export (default: all).
#                  e.g. `bash workflow.sh --steps export` re-exports HS3 JSON
#                  from the existing workspaces/*.root without rebuilding
#                  workspaces or re-running fits/scans.
set -euo pipefail
cd "$(dirname "$0")"

SEED=42
STEPS="ws,fit,plot,scan,export"
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)  SEED=$2; shift 2 ;;
        --steps) STEPS=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate step names up front so a typo doesn't silently skip everything.
IFS=',' read -r -a _steps_arr <<< "$STEPS"
for s in "${_steps_arr[@]}"; do
    case $s in
        ws|fit|plot|scan|export) ;;
        *) echo "Unknown step: '$s' (valid: ws,fit,plot,scan,export)"; exit 1 ;;
    esac
done
run_step() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

step() { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }

# Build a fully-specified workspace name from a set of make_workspace.py flags.
# Every option is spelled out in a fixed order, so comparing two names shows
# exactly which aspects differ. Segments (with make_workspace.py defaults):
#   <N>ch            number of channels                (--num-channels, default 3; no cap)
#   bkg<Pdf><Form>   background pdf + form             (RooExp | GenExp | GenPoly)
#   sig<Pdf>         signal pdf                        (Gauss | Generic | DSCB; --generic-sig / --sig-form dscb)
#   shape<State>     background shape                  (Float | Fixed; --fix-bkg-shape)
#   np<State>        width nuisance parameter          (On | Off; --no-np)
#   constr<Type>     constraint form                   (Gauss | Poisson | None; --constraint)
#   yield<SF>x       yield scale factor                (--yield-sf, default 1)
#   systs<M>         yield-systematic NPs, only if M>0 (--num-systs, default 0)
canonical_stem() {
    local channels=3 bkg_pdf="RooExp" bkg_form="exp" sig_pdf="Gauss" sig_form="gauss"
    local bkg_shape="Float" np="On" constr="gauss" yield_sf="1" systs=0

    set -- $1
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-np)         np="Off";            shift ;;
            --generic-bkg)   bkg_pdf="Gen";       shift ;;
            --bkg-form)      bkg_form="${2:?$1 requires a value (check VARIANTS quoting)}";  shift 2 ;;
            --generic-sig)   sig_pdf="Generic";   shift ;;
            --sig-form)      sig_form="${2:?$1 requires a value (check VARIANTS quoting)}";  shift 2 ;;
            --fix-bkg-shape) bkg_shape="Fixed";   shift ;;
            --constraint)    constr="${2:?$1 requires a value (check VARIANTS quoting)}";    shift 2 ;;
            --yield-sf)      yield_sf="${2:?$1 requires a value (check VARIANTS quoting)}";  shift 2 ;;
            --num-channels)  channels="${2:?$1 requires a value (check VARIANTS quoting)}";  shift 2 ;;
            --num-systs)     systs="${2:?$1 requires a value (check VARIANTS quoting)}";     shift 2 ;;
            *)               shift ;;
        esac
    done

    # --sig-form dscb wins over --generic-sig (make_workspace.py ignores the latter)
    if [[ $sig_form == dscb ]]; then
        sig_pdf="DSCB"
    fi

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

    local stem
    stem=$(printf '%sch_%s_sig%s_shape%s_np%s_constr%s_yield%sx' \
        "$channels" "$bkg" "$sig_pdf" "$bkg_shape" "$np" \
        "$constr_label" "${yield_sf/./p}")
    if [[ $systs != 0 ]]; then
        stem+="_systs${systs}"
    fi
    printf '%s' "$stem"
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
    "--sig-form dscb"
    "--sig-form dscb --num-channels 10"
    "--sig-form dscb --no-np"
    "--sig-form dscb --generic-bkg"
    "--constraint poisson"
    
    # number of channels:
    "--num-channels 1"
    "--num-channels 2"
    "--num-channels 4"
    "--num-channels 5"
    "--num-channels 10"
    "--num-channels 15"
    "--num-channels 20"
    "--num-channels 25"
    "--num-channels 30"

    # channel counts beyond the old 30-channel table:
    "--num-channels 50"
    "--num-channels 100"
    "--num-channels 200"

    # yield-systematic NPs (shared across channels):
    "--num-systs 5"
    "--num-systs 20"
    "--num-systs 50"

    # size products approximating real (bbyy-like) workspaces:
    "--num-channels 10 --num-systs 20"
    "--num-channels 50 --num-systs 20"
    "--num-channels 100 --num-systs 50"
    "--num-channels 30 --num-systs 100"

    # number of events:
    "--yield-sf 0.1"
    "--yield-sf 2"
    "--yield-sf 5"
    "--yield-sf 10"
    "--yield-sf 20"
    "--yield-sf 30"
    "--yield-sf 40"
    "--yield-sf 50"
    "--yield-sf 100"
    "--yield-sf 200"
    "--yield-sf 300"
    "--yield-sf 400"
    "--yield-sf 500"
    "--yield-sf 1000"
    "--yield-sf 5000"

    # number of events with 1 channel:
    "--yield-sf 0.1 --num-channels 1"
    "--yield-sf 2 --num-channels 1"
    "--yield-sf 5 --num-channels 1"
    "--yield-sf 10 --num-channels 1"
    "--yield-sf 20 --num-channels 1"
    "--yield-sf 30 --num-channels 1"
    "--yield-sf 40 --num-channels 1"
    "--yield-sf 50 --num-channels 1"
    "--yield-sf 100 --num-channels 1"
    "--yield-sf 200 --num-channels 1"
    "--yield-sf 300 --num-channels 1"
    "--yield-sf 400 --num-channels 1"
    "--yield-sf 500 --num-channels 1"
    "--yield-sf 1000 --num-channels 1"
    "--yield-sf 5000 --num-channels 1"

    # workspaces for np comparison:
    "--no-np"
    "--no-np --yield-sf 10"
    "--no-np --yield-sf 100"
    "--no-np --yield-sf 1000"
    "--no-np --num-channels 10"
    "--no-np --num-channels 30"

    ""


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
    # Yield scale mixed with channel / pdf axes
    "--yield-sf 0.1 --num-channels 10"
    "--yield-sf 10 --num-channels 10"
    "--yield-sf 100 --generic-bkg"
    "--yield-sf 10 --no-np"
    # Kitchen sink: every axis pushed off its default at once
    "--generic-bkg --bkg-form poly --generic-sig --fix-bkg-shape --constraint poisson --yield-sf 10 --num-channels 10"
)

# ── 1. Generate workspaces ──────────────────────────────────────────────────
if run_step ws; then
step "Generating workspaces (seed=$SEED)"
for flags in "${VARIANTS[@]}"; do
    stem=$(canonical_stem "$flags")
    python3 make_workspace.py $flags --seed "$SEED" --output "workspaces/${stem}.root"
done
fi

# ── 2. Run fits ─────────────────────────────────────────────────────────────
if run_step fit; then
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
fi

# ── 3. Plot workspaces (best-fit / minimum-NLL parameters) ───────────────────
# run_simple_fit.sh floats mu_sig and writes the unconditional best fit to
# output_simple/<stem>_result.root; overlay those post-fit parameters.
if run_step plot; then
step "Plotting workspaces (best-fit parameters)"
for flags in "${VARIANTS[@]}"; do
    stem=$(canonical_stem "$flags")
    python3 plot_workspace.py "workspaces/${stem}.root" \
        --fit-result "output_simple/${stem}_result.root"
    ok "plots/${stem}_channels.png"
done
fi

# ── 4. mu scans ─────────────────────────────────────────────────────────────
if run_step scan; then
step "mu scans"
for flags in "${VARIANTS[@]}"; do
    stem=$(canonical_stem "$flags")
    scan="scans/${stem}_muscan.json"
    python3 muscan.py --input "workspaces/${stem}.root" --output "$scan"
    ok "$scan"
done
fi

# ── 5. Export HS3 JSON ───────────────────────────────────────────────────────────
if run_step export; then
step "Exporting HS3 JSON"
  for flags in "${VARIANTS[@]}"; do
      stem=$(canonical_stem "$flags")
      python3 export_hs3.py --input "workspaces/${stem}.root" --verify
      ok "workspaces/${stem}.json"
  done
fi

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


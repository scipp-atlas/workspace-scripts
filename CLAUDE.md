# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Scripts for generating configurable multi-channel RooFit workspaces, running profile-likelihood fits with `quickFit`, scanning the POI (`mu_sig`), and exporting to HS3 JSON for use with `pyhs3`. This is HEP statistics tooling, not a packaged library — the scripts are run directly.

## Environment

Every script depends on a compiled `quickFit` binary and an ATLAS/LCG software environment (`ROOT` with RooFit, `RooJSONFactoryWSTool`). Nothing works without it:

```bash
source setup_local.sh   # run once per shell; sources ATLAS local setup + LCG_108, adds quickFit to PATH/LD_LIBRARY_PATH
```

`setup_local.sh` hardcodes UChicago Analysis Facility paths (`/cvmfs/...`, `/home/mhance/pyhs3/quickFit`). On a machine without this environment the scripts cannot run, and there is no mock/test harness that bypasses ROOT.

## Commands

```bash
bash workflow.sh                 # full pipeline over all variants (seed 42)
bash workflow.sh --seed 7        # reproducible toys with a different seed
bash workflow.sh --steps export  # rerun a stage subset (ws,fit,plot,scan,export) on existing artifacts

python3 make_workspace.py [flags]            # generate one workspace .root
bash run_simple_fit.sh [workspace.root]      # one unconditional quickFit fit
python3 muscan.py --input ws.root --output scan.json   # profile scan over mu_sig
python3 export_hs3.py --input ws.root --verify         # export HS3 JSON + round-trip check
bash test_muscan.sh              # sanity-check muscan against an external real bbyy workspace (edit paths first)
```

There is no build step, linter, or unit-test suite. The closest thing to a test is `export_hs3.py --verify` (round-trip re-import check) and `test_muscan.sh` (runs the scanner against a non-toy workspace).

## Architecture

The four scripts form a pipeline, each consuming the previous stage's `.root` file. `workflow.sh` orchestrates all of them across the variant matrix defined in its `VARIANTS` array.

```
make_workspace.py → <ws>.root → run_simple_fit.sh  → output_simple/<stem>_result.root
                              → muscan.py           → scans/muscan*.json
                              → export_hs3.py       → <stem>.json (HS3)
```

### Hardcoded names are a contract between scripts

The downstream scripts (`muscan.py`, `run_simple_fit.sh`, `export_hs3.py`) inspect the `.root` file at runtime to auto-detect what they need. They rely on naming conventions established in `make_workspace.py`. Changing a name in `make_workspace.py` will silently break detection elsewhere:

- Workspace `combWS`, ModelConfig `ModelConfig`, dataset `combData`, observable `x`, category `index`, POI `mu_sig`.
- Per-channel objects: `tau_<ch>`, `bkg_<ch>`, `sig_<ch>`, `nbkg_<ch>`, `model_<ch>` for channels `ch0`…`ch{N-1}`.
- Constraint PDFs are named `constr_*` (e.g. `constr_alpha_sigma`, `constr_gamma_sigma`). `muscan.py` and `run_simple_fit.sh` find all `constr_*` PDFs and pass them via `--externalConstraint`. `export_hs3.py` detects constraints structurally (a distribution whose `x` is a const global observable, not referenced in any likelihood).
- Yield systematics (`--num-systs M`): shared NPs `alpha_syst<j>` with constraints `constr_alpha_syst<j>`, global observables `nom_alpha_syst<j>`, and per-channel response factors `resp_syst<j>_<ch>` folded into `nsig_tot_<ch>`.
- Background type detection probes `bkg_ch0`'s class (`RooExponential` vs `RooGenericPdf`).

`--num-channels` has no upper limit: the first 30 channels come from the hardcoded `CHANNELS` dict in `make_workspace.py`; beyond that, `get_channels()` extends it with a deterministic per-index formula (seed-independent).

### Critical RooFit constraint: do NOT wrap RooSimultaneous in RooProdPdf

In ROOT 6.30+, wrapping a `RooSimultaneous` in a `RooProdPdf` to attach constraint terms breaks extended-likelihood evaluation. Instead the constraint PDF is imported into the workspace standalone and supplied to `quickFit` via `--externalConstraint`. This is why every fit/scan script auto-detects `constr_*` and passes it explicitly. Preserve this pattern when editing the model.

### export_hs3.py post-processing chain

ROOT's `RooJSONFactoryWSTool.exportJSON` produces HS3 that pyhs3 cannot consume directly. `export_hs3.py` applies an ordered chain of in-place JSON fixes (each individually toggleable with `--no-fix-*`, or all off with `--no-cleanup`):

1. `fix_exponential_functions` — collapse ROOT's `-tau` sign-inversion intermediates (HS3 uses `exp(-c*x)`, RooExponential uses `exp(tau*x)`), flipping signs in domains/parameter_points.
2. `fix_null_axes` / `fix_dataset_axes` — repair empty/malformed axis entries.
3. `fix_analysis_init` — add `init: default_values`.
4. `fix_remove_obs_from_params` — drop non-const observables from parameter sets so they don't overwrite data at eval time.
5. `fix_constraint_pdfs` — wire standalone constraints into the first likelihood under `aux_distributions`/`aux_data` (default; `--no-aux-constraints` uses `distributions`/`data` and writes a `_noaux.json` file).

The single combined likelihood (ROOT's HS3 form of the `RooSimultaneous` joint fit, analysis `sim_pdf_combData`) is kept intact by default so pyhs3 evaluates one joint likelihood over all channels, matching real workspaces. `fix_split_likelihoods` (opt-in via `--split-likelihoods`, debugging only) instead splits it into independent per-channel `L_ch<i>` likelihoods/analyses.

### Output layout (all git-ignored)

`.root` and `.json` files plus `output_*` directories are git-ignored. `workflow.sh` writes workspaces to `workspaces/`, scans to `scans/`, fit logs/results and per-mu-point artifacts to `output_simple/`, and HS3 JSON to the repo root.

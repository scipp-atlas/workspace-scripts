# workspace-scripts

A small, standalone toolkit for generating configurable multi-channel RooFit
workspaces, running profile-likelihood fits and POI scans with `quickFit`,
exporting to [HS3](https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard)
JSON, and **validating** that same statistical model in
[`pyhs3`](https://github.com/scipp-atlas/pyhs3) against the `quickFit` reference.

It began as an informal test suite for pyhs3 — a way to produce simple, fully
controlled workspaces whose likelihood can be compared point-by-point between
the two tools. This repo is the generation + comparison harness; the scripts are
run directly (there is no build step or installed package).

## Two environments

The pipeline spans two software stacks; keep them separate:

| Stage | Scripts | Needs |
|-------|---------|-------|
| **Generate / fit / scan / export** | `make_workspace.py`, `run_simple_fit.sh`, `muscan.py`, `export_hs3.py`, `plot_*.py`, `workflow.sh` | ROOT (RooFit, `RooJSONFactoryWSTool`) + a compiled `quickFit` — see `setup_local.sh` |
| **Validate in pyhs3** | [`pyhs3_eval/`](pyhs3_eval/), `workspace_comparison.sh` | `pyhs3` (with `pytensor`), `numpy`, `matplotlib` |

### ROOT / quickFit setup

On the UChicago Analysis Facility:

```bash
source setup_local.sh   # ATLAS local setup + LCG_108, adds quickFit to PATH/LD_LIBRARY_PATH
```

`setup_local.sh` hardcodes AF paths (`/cvmfs/...`, `/home/mhance/pyhs3/quickFit`);
on a machine without this environment the generation scripts cannot run.

### pyhs3 setup

The `pyhs3_eval/` scripts run in a separate Python environment. Install their
dependencies into a virtualenv or pixi environment via the `pyhs3` extra in
`pyproject.toml`:

```bash
pip install -e ".[pyhs3]"   # pyhs3 (with pytensor), numpy, matplotlib
```

(`pip install -e .` builds nothing — the scripts are run directly — it only
resolves the dependency group.) See [`pyhs3_eval/README.md`](pyhs3_eval/README.md).

## Quick start

```bash
# 1. Generation stage (ROOT/quickFit environment)
source setup_local.sh
bash workflow.sh            # whole variant matrix, random seed 42
bash workflow.sh --seed 7   # reproducible toys with a different seed

# 2. Validation stage (pyhs3 environment)
python3 pyhs3_eval/eval_simple_muscan.py --plot-nll   # single variant
bash workspace_comparison.sh --all                    # batch comparison
```

`workflow.sh` generates every variant under `workspaces/`, runs a fit and a mu
scan on each, plots the channels, and exports HS3 JSON. Fit logs and per-mu
results go to `output_simple/`, scans to `scans/`, plots to `plots/`, and HS3
JSON next to each workspace under `workspaces/`.

## Model

Each workspace is a simultaneous fit across `N` channels (`ch0` … `ch{N-1}`,
default `N=3`, up to 30) with observable `x` in [10, 20].

| Component | Description |
|-----------|-------------|
| Signal | Gaussian at mean = 15, per-channel nominal width ~1, ~7 events/channel at `mu_sig = 1` (or a `RooGenericPdf` with `--generic-sig`) |
| Background | `RooExponential`, or `RooGenericPdf` of exponential or polynomial form, ~23 events/channel |
| POI | `mu_sig` — signal strength, floated in [−5, 10] |
| Unconstrained NPs | `tau_ch*` (bkg shape; held constant with `--fix-bkg-shape`), `nbkg_ch*` (bkg yield) |
| Width NP | Shared signal-width nuisance — `alpha_sigma` (additive, ±10%/σ) or `gamma_sigma` (multiplicative), depending on the constraint form |

### Signal-width nuisance parameter (`--constraint`)

| `--constraint` | NP | Constraint PDF | Effect on width |
|----------------|----|----------------|-----------------|
| `gauss` (default) | `alpha_sigma` | `RooGaussian` (`constr_alpha_sigma`) | `sigma = sigma_nom * (1 + 0.10 * alpha_sigma)` |
| `poisson` | `gamma_sigma` | `RooPoisson` (`constr_gamma_sigma`) | `sigma = sigma_nom * gamma_sigma` |
| `none` | `alpha_sigma` | none (free NP, no aux term) | additive, as in `gauss` |

`--no-np` drops the width NP entirely and fixes the signal width at nominal.

The constraint PDF is supplied to quickFit via `--externalConstraint` (not
wrapped in a `RooProdPdf`, which breaks extended-likelihood evaluation for
`RooSimultaneous` in ROOT 6.30+). The fit/scan scripts auto-detect and pass it.

### Hardcoded names are a contract between scripts

Downstream scripts inspect the `.root` file at runtime and rely on the naming
conventions established in `make_workspace.py`. Renaming an object there will
silently break detection elsewhere:

- Workspace `combWS`, ModelConfig `ModelConfig`, dataset `combData`, observable
  `x`, category `index`, POI `mu_sig`.
- Per-channel objects `tau_<ch>`, `bkg_<ch>`, `sig_<ch>`, `nbkg_<ch>`,
  `model_<ch>`.
- Constraint PDFs are named `constr_*`; the fit/scan scripts pass all of them via
  `--externalConstraint`, and `export_hs3.py` detects them structurally.

### Workspace file naming

`workflow.sh` names every workspace with a **fully-specified canonical stem** —
each `make_workspace.py` option is spelled out in a fixed order, so comparing two
names shows exactly which aspects differ:

```
<N>ch_bkg{RooExp|GenExp|GenPoly}_sig{Gauss|Generic}_shape{Float|Fixed}_np{On|Off}_constr{Gauss|Poisson|None}_yield<F>x
```

For example `3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x` is the
base variant. `pyhs3_eval/` and `workspace_comparison.sh` parse this stem.

When run standalone **without** `--output`, `make_workspace.py` instead derives a
shorter name that encodes only the non-default aspects, e.g.
`simple_workspace.root`, `simple_workspace_nonp.root`, or
`simple_workspace_generic_poly.root`.

## Scripts

### `make_workspace.py`

Generates one workspace variant as a ROOT file.

```bash
python3 make_workspace.py                                 # NP (gauss), RooExponential bkg, 3 channels
python3 make_workspace.py --no-np                         # no width NP
python3 make_workspace.py --generic-bkg                   # RooGenericPdf exponential bkg
python3 make_workspace.py --generic-bkg --bkg-form poly   # RooGenericPdf polynomial bkg
python3 make_workspace.py --generic-sig                   # signal as RooGenericPdf
python3 make_workspace.py --constraint poisson            # Poisson-constrained width NP (gamma_sigma)
python3 make_workspace.py --constraint none               # free width NP, no constraint
python3 make_workspace.py --fix-bkg-shape                 # hold tau_ch constant
python3 make_workspace.py --num-channels 30               # up to 30 channels
python3 make_workspace.py --yield-sf 10                   # scale all yields ×10
python3 make_workspace.py --seed 123 --output my_ws.root
```

| Option | Description |
|--------|-------------|
| `--no-np` | Omit the signal-width nuisance parameter |
| `--generic-bkg` | Use `RooGenericPdf` instead of `RooExponential` for the background |
| `--bkg-form {exp,poly}` | Generic background form (only with `--generic-bkg`); default `exp` |
| `--generic-sig` | Express the signal Gaussian as a `RooGenericPdf` |
| `--fix-bkg-shape` | Hold `tau_ch` constant so the bkg shape is frozen during the scan |
| `--constraint {gauss,poisson,none}` | Constraint form for the width NP (default `gauss`) |
| `--num-channels N` | Number of channels, 1–30 (default 3) |
| `--yield-sf F` | Scale all signal and background yields by `F` (default 1.0) |
| `--seed N` | Random seed for toy generation (default 42) |
| `--output NAME.root` | Output file (default: auto-derived from options) |

### `run_simple_fit.sh`

Runs a single quickFit unconditional fit on a workspace and writes the result to
`output_simple/`. Auto-detects every `constr_*` PDF and passes the matching
`--externalConstraint`.

```bash
bash run_simple_fit.sh                                      # defaults to simple_workspace.root
bash run_simple_fit.sh workspaces/<stem>.root
```

### `muscan.py`

Scans the profile likelihood over a grid of `mu_sig` values (POI fixed at each
point, NPs profiled). Writes JSON with NLL, delta-NLL (`2*(NLL − NLL_min)`), fit
status, and post-fit parameters for every point. Auto-detects the POI, all
`constr_*` PDFs, and the background PDF type.

```bash
python3 muscan.py                                             # default grid 0 → 3 step 0.25
python3 muscan.py --mu-min -1 --mu-max 3 --mu-step 0.1 --output scan.json
python3 muscan.py --mu-vals "-1 0 1 2"                        # explicit list
python3 muscan.py --input workspaces/<stem>.root --output scans/<stem>_muscan.json
```

| Option | Description |
|--------|-------------|
| `--mu-vals "…"` | Explicit space-separated mu values (mutually exclusive with the grid) |
| `--mu-min / --mu-max / --mu-step` | Grid bounds and step (defaults 0.0 / 3.0 / 0.25) |
| `--input` / `--output` | Input workspace / output JSON |
| `--logdir` | Directory for per-mu quickFit logs and result files (default `output_simple`) |
| `--poi` | POI name (default: auto-detected from ModelConfig) |
| `--nll-offset` | Pass `--nllOffset 0` to quickFit (suppresses automatic NLL offsetting) |

### `export_hs3.py`

Exports a workspace to HS3 JSON via `RooJSONFactoryWSTool`, then applies an
ordered chain of in-place fixes so pyhs3 can consume it (collapse ROOT's
exponential sign-inversion intermediates, repair axes, split the combined
likelihood per channel, add `init: default_values`, drop observables from
parameter sets, and wire standalone constraints into the first channel's
likelihood). Each fix is toggleable with `--no-fix-*`, or all off with
`--no-cleanup`.

```bash
python3 export_hs3.py --input workspaces/<stem>.root --verify
python3 export_hs3.py --input workspaces/<stem>.root --no-aux-constraints --verify
```

| Option | Description |
|--------|-------------|
| `--input` / `--ws-name` | Input ROOT file / workspace name (default `combWS`) |
| `--output-stem` | Output stem without extension (default: input name without `.root`) |
| `--verify` | Re-import the exported JSON and check `sim_pdf`, `combData`, `mu_sig` |
| `--aux-constraints` / `--no-aux-constraints` | Constraints under `aux_distributions`/`aux_data` (default, correct HS3 normalisation) or under `distributions`/`data`; `--no-aux-constraints` writes a `<stem>_noaux.json` |

### `plot_workspace.py`

Draws the toy data and model PDF (total, background, signal) for each channel of
a workspace, using the saved `nominal` snapshot or a `--fit-result` overlay. A
quick visual check that a workspace was built and generated as intended.

```bash
python3 plot_workspace.py workspaces/<stem>.root
python3 plot_workspace.py workspaces/<stem>.root --fit-result output_simple/<stem>_result.root
```

### `plot_muscan.py`

Plots `delta_nll = 2*(NLL − NLL_min)` versus the POI from `muscan.py` JSON — a
smooth parabolic well with its minimum at the injected signal strength.

```bash
python3 plot_muscan.py                          # every scans/*.json
python3 plot_muscan.py scans/<stem>_muscan.json
python3 plot_muscan.py scans/*.json --overlay   # all curves on one figure
```

### `workflow.sh`

Orchestrates the full sequence over the whole variant matrix (defined in the
`VARIANTS` array): generate → fit → plot → mu scan → HS3 export, naming every
output with the canonical stem.

```bash
bash workflow.sh
bash workflow.sh --seed 99
```

### `pyhs3_eval/` — pyhs3 validation

Re-evaluates the scans in pyhs3 and compares against quickFit. Runs in the pyhs3
environment. `eval_simple_muscan.py` prints a per-point NLL comparison and
constant-offset statistics; `workspace_comparison.sh` batches it over groups of
workspaces. See [`pyhs3_eval/README.md`](pyhs3_eval/README.md).

```bash
python3 pyhs3_eval/eval_simple_muscan.py --plot-nll
bash workspace_comparison.sh --all        # or --events / --channels
```

### `setup_local.sh` / `test_muscan.sh`

`setup_local.sh` sources the ATLAS local setup and LCG view and adds `quickFit`
to `PATH`/`LD_LIBRARY_PATH` — run it before anything in the generation stage.
`test_muscan.sh` runs `muscan.py` against an external real-analysis (bbyy)
workspace as a non-toy sanity check; edit the paths at the top before running.

## Output layout

All generated artifacts are git-ignored.

```
workspaces/
  <stem>.root              # generated workspaces (workflow.sh)
  <stem>.json              # HS3 export (export_hs3.py)
  <stem>_noaux.json        # HS3 export with --no-aux-constraints
scans/
  <stem>_muscan.json       # per-variant mu scans
output_simple/
  <stem>_fit.log           # quickFit unconditional fit log
  <stem>_result.root       # quickFit unconditional fit result
  log_mu_<tag>.txt         # per-mu-point quickFit log (muscan)
  result_mu_<tag>.root     # per-mu-point quickFit result (muscan)
plots/
  <stem>_channels.png      # per-channel data/model overlay (plot_workspace.py)
pyhs3_eval/
  *.pdf                    # pyhs3-vs-quickFit comparison plots
```

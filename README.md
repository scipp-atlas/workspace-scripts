# workspace_scripts

Scripts for generating configurable multi-channel RooFit workspaces, running
profile-likelihood fits with quickFit, scanning the POI, and exporting results
to [HS3](https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard)
JSON for use with [pyhs3](https://github.com/scikit-hep/pyhs3).


## Prerequisites

You need access to an ATLAS/LCG software environment and a compiled
`quickFit` binary.  On the UChicago Analysis Facility:

```bash
source setup_local.sh
```

This sources the ATLAS local setup and LCG view, then adds `quickFit` to your
`PATH` and `LD_LIBRARY_PATH`.  Run it once per shell session before using any of
the other scripts.


## Quick start

Run the full workflow end-to-end (workspace generation, fits, mu scans,
HS3 export) across all variants:

```bash
source setup_local.sh
bash workflow.sh            # uses random seed 42 by default
bash workflow.sh --seed 7   # reproducible toys with a different seed
```

This generates ~20 workspace variants under `workspaces/`, runs a fit and a mu
scan on each, and exports HS3 JSON.  Fit logs and per-mu result files go into
`output_simple/`; mu-scan JSON goes into `scans/`.


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
`RooSimultaneous` in ROOT 6.30+).  The fit/scan scripts auto-detect and pass it.

### Workspace file naming

When `--output` is not given, `make_workspace.py` derives the file stem from the
options:

```
simple_workspace[_generic[_poly]][_gensig][_poisson|_gauss][_fixshape][_nonp].root
```

For example `simple_workspace_generic_poly.root` (generic polynomial bkg) or
`simple_workspace_nonp.root` (no width NP).


## Scripts

### `make_workspace.py`

Generates a workspace variant as a ROOT file.

```bash
python3 make_workspace.py                              # NP (gauss), RooExponential bkg, 3 channels
python3 make_workspace.py --no-np                      # no width NP
python3 make_workspace.py --generic-bkg                # RooGenericPdf exponential bkg
python3 make_workspace.py --generic-bkg --bkg-form poly  # RooGenericPdf polynomial bkg
python3 make_workspace.py --generic-sig                # signal as RooGenericPdf
python3 make_workspace.py --constraint poisson         # Poisson-constrained width NP (gamma_sigma)
python3 make_workspace.py --constraint none            # free width NP, no constraint
python3 make_workspace.py --fix-bkg-shape              # hold tau_ch/slope_ch constant
python3 make_workspace.py --num-channels 30            # up to 30 channels
python3 make_workspace.py --yield-sf 10                # scale all yields ×10
python3 make_workspace.py --seed 123 --output my_ws.root
```

| Option | Description |
|--------|-------------|
| `--no-np` | Omit the signal-width nuisance parameter |
| `--generic-bkg` | Use `RooGenericPdf` instead of `RooExponential` for the background |
| `--bkg-form {exp,poly}` | Generic background form (only with `--generic-bkg`); default `exp` |
| `--generic-sig` | Express the signal Gaussian as a `RooGenericPdf` |
| `--fix-bkg-shape` | Hold `tau_ch`/`slope_ch` constant so the bkg shape is frozen during the scan |
| `--constraint {gauss,poisson,none}` | Constraint form for the width NP (default `gauss`) |
| `--num-channels N` | Number of channels, 1–30 (default 3) |
| `--yield-sf F` | Scale all signal and background yields by `F` (default 1.0) |
| `--seed N` | Random seed for toy generation (default 42) |
| `--output NAME.root` | Output file (default: auto-derived from options) |

Each run prints the toy events generated per channel and the quick-start
`quickFit` command for the resulting workspace.

### `run_simple_fit.sh`

Runs a single quickFit unconditional fit on a workspace file and writes the
result to `output_simple/`.

```bash
bash run_simple_fit.sh                              # defaults to simple_workspace.root
bash run_simple_fit.sh workspaces/simple_workspace_nonp.root
```

Auto-detects every `constr_*` PDF in the workspace and passes the matching
`--externalConstraint`.  Output is logged to `output_simple/<stem>_fit.log` and
the result ROOT file to `output_simple/<stem>_result.root`.

### `muscan.py`

Scans the profile likelihood over a grid of `mu_sig` values, running quickFit
with the POI fixed at each point.  Writes a JSON file with NLL, delta-NLL
(`2*(NLL − NLL_min)`), fit status, and post-fit parameter values for every point.

```bash
python3 muscan.py                                             # default grid: 0 → 3 in steps of 0.25
python3 muscan.py --mu-min -1 --mu-max 3 --mu-step 0.1 --output scan.json
python3 muscan.py --mu-vals "-1 0 1 2"                        # explicit list
python3 muscan.py --input simple_workspace_nonp.root --output muscan_nonp.json
python3 muscan.py --nll-offset                                # pass --nllOffset 0 to quickFit
```

| Option | Description |
|--------|-------------|
| `--mu-vals "…"` | Explicit space-separated list of mu values (mutually exclusive with the grid) |
| `--mu-min / --mu-max / --mu-step` | Grid bounds and step (defaults 0.0 / 3.0 / 0.25) |
| `--input` | Input workspace (default `simple_workspace.root`) |
| `--output` | Output JSON (default `muscan.json`) |
| `--logdir` | Directory for per-mu quickFit logs and result files (default `output_simple`) |
| `--poi` | POI name (default: auto-detected from ModelConfig) |
| `--nll-offset` | Pass `--nllOffset 0` to quickFit (suppresses automatic NLL offsetting) |

Auto-detects the POI, all `constr_*` constraint PDFs, and the background PDF
type.  Per-mu logs and result files are written to `--logdir`.

### `export_hs3.py`

Exports a RooFit workspace to HS3 JSON using ROOT's `RooJSONFactoryWSTool`,
applying post-processing fixes for pyhs3 compatibility:

- Splits the combined simultaneous likelihood into one per channel.
- Wires standalone constraint PDFs into the first channel's likelihood (under
  `aux_distributions`/`aux_data` by default).
- Cleans up ROOT's sign-inversion intermediates for exponential distributions.
- Fixes null axes and dataset axis entries, and adds `init: default_values`.

```bash
python3 export_hs3.py                                         # reads simple_workspace.root
python3 export_hs3.py --input simple_workspace_nonp.root --verify
python3 export_hs3.py --input my_ws.root --output-stem my_ws --verify
python3 export_hs3.py --input simple_workspace.root --no-aux-constraints --verify
```

| Option | Description |
|--------|-------------|
| `--input` | Input ROOT workspace (default `simple_workspace.root`) |
| `--ws-name` | Workspace name inside the file (default `combWS`) |
| `--output-stem` | Output stem without extension (default: input name without `.root`) |
| `--verify` | Re-import the exported JSON and check `sim_pdf`, `combData`, `mu_sig` |
| `--aux-constraints` / `--no-aux-constraints` | Place constraints under `aux_distributions`/`aux_data` (default, correct HS3 normalisation) or under `distributions`/`data` (old behaviour). `--no-aux-constraints` writes a `<stem>_noaux.json` file |

### `workflow.sh`

Orchestrates the full sequence over all variants defined in the `VARIANTS`
array (NP/no-NP, generic/poly/generic-sig backgrounds and signals, Poisson/no
constraint, yield scale factors, and 1–30 channel counts):

1. Generate each workspace with `make_workspace.py` into `workspaces/`.
2. Run `run_simple_fit.sh` on each.
3. Run `muscan.py` on each, writing JSON to `scans/`.
4. Export HS3 JSON with `export_hs3.py --verify` for each, plus `--no-aux-constraints`
   variants for the two reference workspaces.

```bash
bash workflow.sh
bash workflow.sh --seed 99
```

### `test_muscan.sh`

Runs `muscan.py` against an external real-analysis workspace (the bbyy
non-resonant non-parametric model under `$basedir/$XML`) with `--nll-offset`, as
a sanity check on a non-toy workspace.  Edit the paths at the top before running.

### `setup_local.sh`

Sources the ATLAS local setup and LCG view (`LCG_108 x86_64-el9-gcc14-opt`),
then adds the local `quickFit` build to `PATH` and `LD_LIBRARY_PATH`.  Run this
before anything else.


## Output layout

```
workspaces/
  <stem>.root              # generated workspaces (workflow.sh)
scans/
  muscan<suffix>.json      # per-variant mu scans (workflow.sh)
output_simple/
  <stem>_fit.log           # quickFit unconditional fit log
  <stem>_result.root       # quickFit unconditional fit result
  log_mu_<tag>.txt         # per-mu-point quickFit log (muscan)
  result_mu_<tag>.root     # per-mu-point quickFit result (muscan)
<stem>.json                # HS3 export (export_hs3.py)
<stem>_noaux.json          # HS3 export with --no-aux-constraints
```

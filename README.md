# workspace_scripts

Scripts for generating a simple three-channel RooFit workspace, running
profile-likelihood fits with quickFit, and exporting results to
[HS3](https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard)
JSON for use with [pyhs3](https://github.com/scikit-hep/pyhs3).


## Prerequisites

You need access to an ATLAS/LCG software environment and a compiled
`quickFit` binary.  On the UChicago Analysis Facility:

```bash
source setup_local.sh
```

This sources the ATLAS local setup and adds `quickFit` to your `PATH` and
`LD_LIBRARY_PATH`.  Run it once per shell session before using any of the
other scripts.


## Quick start

Run the full workflow end-to-end (workspace generation, fits, mu scan,
HS3 export):

```bash
source setup_local.sh
bash workflow.sh            # uses random seed 42 by default
bash workflow.sh --seed 7   # reproducible toys with a different seed
```

This produces four workspace variants (see below), runs fits on each,
performs a mu scan, and exports HS3 JSON files.  Fit logs and per-mu
result files go into `output_simple/`.


## Model

Each workspace is a simultaneous fit across three channels (`ch0`, `ch1`,
`ch2`) with observable `x` in [10, 20].

| Component | Description |
|-----------|-------------|
| Signal | Gaussian at mean = 15, width ~1, ~7 events/channel at mu_sig = 1 |
| Background | Exponential or generic PDF, ~23 events/channel |
| POI | `mu_sig` — signal strength, floated in [−5, 10] |
| Unconstrained NPs | `tau_ch{0,1,2}` (exp. decay), `nbkg_ch{0,1,2}` (bkg yield) |
| Constrained NP | `alpha_sigma` — shifts signal width by ±10% per sigma (Gaussian-constrained, NP variant only) |

### Workspace variants

| File | NP | Background |
|------|----|-----------|
| `simple_workspace.root` | yes | `RooExponential` |
| `simple_workspace_nonp.root` | no | `RooExponential` |
| `simple_workspace_generic.root` | yes | `RooGenericPdf` (same exp shape; exports as `generic_dist` in HS3) |
| `simple_workspace_generic_nonp.root` | no | `RooGenericPdf` |


## Scripts

### `make_workspace.py`

Generates one of the four workspace variants as a ROOT file.

```bash
python3 make_workspace.py                          # with NP, RooExponential background
python3 make_workspace.py --no-np                  # without NP
python3 make_workspace.py --generic-bkg            # RooGenericPdf background
python3 make_workspace.py --no-np --generic-bkg    # both
python3 make_workspace.py --seed 123 --output my_ws.root
```

Each run prints the number of toy events generated per channel and the
quick-start `quickFit` command for the resulting workspace.

### `run_simple_fit.sh`

Runs a single quickFit unconditional fit on a workspace file and writes
the result to `output_simple/`.

```bash
bash run_simple_fit.sh                          # defaults to simple_workspace.root
bash run_simple_fit.sh simple_workspace_nonp.root
```

Automatically detects whether the workspace contains `constr_alpha_sigma`
and passes `--externalConstraint` to quickFit when needed.  Fit output is
logged to `output_simple/<stem>_fit.log` and the result ROOT file is
written to `output_simple/<stem>_result.root`.

### `muscan.py`

Scans the profile likelihood over a grid of `mu_sig` values by running
quickFit with the POI fixed at each point.  Writes a JSON file with NLL
values, delta-NLL, fit status, and post-fit parameter values for every
scan point.

```bash
python3 muscan.py                                             # default grid: 0 to 3 in steps of 0.25
python3 muscan.py --mu-min -1 --mu-max 3 --mu-step 0.1 --output scan.json
python3 muscan.py --mu-vals "-1 0 1 2"                       # explicit list
python3 muscan.py --input simple_workspace_nonp.root --output muscan_nonp.json
```

Per-mu quickFit logs and result files are written to `output_simple/`.

### `export_hs3.py`

Exports a RooFit workspace to HS3 JSON using ROOT's
`RooJSONFactoryWSTool`.  Applies several post-processing fixes to make
the output compatible with pyhs3:

- Splits the combined simultaneous likelihood into one per channel.
- Wires standalone constraint PDFs into the first channel's likelihood
  under `aux_distributions` / `aux_data`.
- Cleans up ROOT's sign-inversion intermediates for exponential
  distributions.
- Fixes null axes and dataset axis entries.

```bash
python3 export_hs3.py                                     # reads simple_workspace.root
python3 export_hs3.py --input simple_workspace_nonp.root --verify
python3 export_hs3.py --input my_ws.root --output-stem my_ws --verify
```

`--verify` re-imports the exported JSON and checks that the key
workspace objects (`sim_pdf`, `combData`, `mu_sig`) are correctly
reconstructed.

### `workflow.sh`

Orchestrates the full sequence:

1. Generate all four workspace variants with `make_workspace.py`.
2. Run `run_simple_fit.sh` on each.
3. Run `muscan.py` on each.
4. Export HS3 JSON with `export_hs3.py --verify` for each.

```bash
bash workflow.sh
bash workflow.sh --seed 99
```

### `setup_local.sh`

Sources the ATLAS local setup and LCG view (`LCG_108 x86_64-el9-gcc14-opt`),
then adds the local `quickFit` build to `PATH` and `LD_LIBRARY_PATH`.
Run this before anything else.


## Output layout

```
output_simple/
  <stem>_fit.log           # quickFit unconditional fit log
  <stem>_result.root       # quickFit unconditional fit result
  log_mu_<tag>.txt         # per-mu-point quickFit log (muscan)
  result_mu_<tag>.root     # per-mu-point quickFit result (muscan)
```

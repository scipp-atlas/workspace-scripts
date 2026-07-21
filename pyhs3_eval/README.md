# pyhs3_eval

pyhs3-side validation scripts for the workspaces produced by this repo. They
re-evaluate the profile-likelihood scans in [`pyhs3`](https://github.com/scipp-atlas/pyhs3)
and compare them against the `quickFit` reference values, so the same
statistical model can be checked across both tools.

These scripts run in a **different environment** from the rest of the repo. The
generation/scan/export pipeline (`make_workspace.py`, `muscan.py`,
`export_hs3.py`, ...) needs ROOT + `quickFit` (see `../setup_local.sh`). The
scripts here instead need:

- `pyhs3` (with `pytensor`)
- `numpy`
- `matplotlib`

Install them via the `pyhs3` extra in the repo's `pyproject.toml` (from the
repo root, into a virtualenv or pixi environment):

```bash
pip install -e ".[pyhs3]"
```

## Inputs

Each script consumes the outputs of `workflow.sh`, matched by stem:

- workspace HS3 JSON: `../workspaces/<stem>.json`
- quickFit scan:      `../scans/<stem>_muscan.json`

where `<stem>` is the canonical name from `workflow.sh`, e.g.
`3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x`.

## Scripts

### `eval_simple_muscan.py`

Evaluates the pyhs3 NLL at every scan point and compares it to quickFit. Prints
a per-point table (`qf_nll`, `pyhs3_nll`, `diff`) and constant-offset statistics.
The `diff` should be a constant offset across the scan; `max |residual|` measures
how far it strays from constant — the core agreement metric.

Run from the repository root:

```bash
# single workspace/scan (defaults to the 3ch base variant)
python3 pyhs3_eval/eval_simple_muscan.py

# explicit workspace/scan
python3 pyhs3_eval/eval_simple_muscan.py \
    --workspace workspaces/3ch_bkgGenExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
    --scan      scans/3ch_bkgGenExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x_muscan.json

# compare several workspaces, ranked by flatness, with plots
python3 pyhs3_eval/eval_simple_muscan.py \
    --pair workspaces/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
           scans/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x_muscan.json \
    --pair workspaces/3ch_bkgGenExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
           scans/3ch_bkgGenExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x_muscan.json \
    --plot-nll
```

| Option | Description |
|--------|-------------|
| `--workspace` / `--scan` | Single workspace HS3 JSON and its muscan JSON (defaults to the 3ch base variant) |
| `--pair WS SCAN` | A workspace/scan pair; repeat to compare several and print a ranked constant-offset summary |
| `--plot-nll [PATH]` | Plot pyhs3 vs quickFit ΔNLL curves (one panel per workspace) via `plot_muscan_nll.py` |
| `--plot-resid [PATH]` | Plot mean offset and max residual per workspace via `plot_residuals.py` |
| `--plot-resid-field FIELD…` | Workspace-name field(s) for the residual-plot x labels (default `channels`) |
| `--plot-resid-log-x` / `--plot-resid-log-y` | Log scales on the residual plot |

### `plot_muscan_nll.py`

`plot_nll_curves(results, output_pdf)` — renders one ΔNLL panel per workspace
(quickFit vs pyhs3, both min-shifted to zero). Imported by `eval_simple_muscan.py`.

### `plot_residuals.py`

`plot_residual_and_offset(results, output_pdf, ...)` — plots the mean offset and
max residual across a set of workspaces versus a varied field (channels, yield,
...), parsed from the canonical stem. Imported by `eval_simple_muscan.py`.

## Batch comparison

`../workspace_comparison.sh` drives `eval_simple_muscan.py` over groups of
workspaces (`--all`, `--events`, `--channels`).

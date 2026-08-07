# pyhs3_eval

pyhs3-side validation scripts for the workspaces produced by this repo. They
re-evaluate the profile-likelihood scans in [`pyhs3`](https://github.com/scipp-atlas/pyhs3)
and compare them against the `quickFit` reference values, so the same
statistical model can be checked across both tools.

These scripts run in a **different environment** from the rest of the repo. The
generation/scan/export pipeline (`make_workspace.py`, `muscan.py`,
`export_hs3.py`, ...) needs ROOT + `quickFit` (see `../setup_local.sh`). The
scripts here instead need `pyhs3` (with `pytensor`), `numpy`, and `matplotlib`.

### Recommended: pixi

From the repo root, the committed `../pixi.toml` builds the environment from
conda-forge — including a C++ toolchain and a Python that ships headers, so
pytensor compiles its C ops with no extra system setup:

```bash
pixi install                 # create the environment (writes pixi.lock)
pixi run eval                # single-variant comparison
pixi run compare-all         # or compare-events / compare-channels
pixi shell                   # interactive shell in the environment
```

### Alternative: pip + venv

Install the dependencies from PyPI into a Python ≥ 3.10 virtualenv:

```bash
pip install pyhs3 numpy matplotlib
```

This works, but pytensor JIT-compiles to C at runtime and needs the Python
development headers, which many base interpreters lack — see Troubleshooting.
pixi sidesteps this entirely, which is why it's recommended.

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
| `--analysis REGEX` | Regex fully matched against analysis names to select the likelihoods (default `L_ch\d+`, the split toy channels; for a real workspace pass its combined analysis name, e.g. `CombinedPdf_combData`) |
| `--cache-dir DIR` | Pickle each compiled channel model (written right after it compiles, so an interrupted run keeps finished channels) and reload on later runs; recommended for large real workspaces; delete after changing pyhs3 versions |
| `--pytensor-mode MODE` | Override the pytensor compilation mode (default: `FAST_RUN`). `NO_REWRITES` skips pytensor's graph-rewrite phase entirely and evaluates in pure Python — use it for large real workspaces where `FAST_RUN`/`FAST_COMPILE` pin the CPU for hours during rewriting (evaluation is slower per point, but with ~30 scan points that's seconds, not hours); part of the cache key |
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

## Troubleshooting

**`fatal error: Python.h: No such file or directory`** — pytensor JIT-compiles
its ops to C and needs the Python development headers, which the base
interpreter may lack. The clean fix is to use the **pixi** environment above:
conda-forge's Python ships headers and `cxx-compiler` provides the toolchain, so
compilation just works. If you're stuck on a pip/venv setup, either install the
headers (`sudo dnf install python3.12-devel` on EL9, matching your Python
version), build the venv from an interpreter that bundles headers (a
`uv python`-managed or conda Python), or fall back to pytensor's pure-Python ops
(no compiler needed, but much slower):

```bash
export PYTENSOR_FLAGS="cxx="
```

**`No module named 'pytensor.graph.traversal'` / `pytensor<3.0,>=2.33.0` not
found** — your Python is older than 3.10. pyhs3 and pytensor ≥ 2.33 require
Python ≥ 3.10; build the venv from a newer interpreter (see the repo README).

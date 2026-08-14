#!/usr/bin/env python3
"""
Plot the toy data and model PDF for each channel of a simple workspace.

For every channel in a `make_workspace.py` workspace this draws, on the
observable `x`:

  * the toy dataset for that channel (points with Poisson errors),
  * the full extended model (signal + background), normalised to the data,
  * the background component alone (dashed),
  * the signal component alone (dotted).

If data and model agree — points scattered around the total curve, a bump under
the signal where the Gaussian sits — the workspace was built and generated as
intended.  Parameters are taken from the saved `nominal` snapshot by default
(the values the toys were generated at); pass `--fit-result` to overlay a
post-fit model instead.

Channel objects follow the naming contract from make_workspace.py
(`model_<ch>`, `sig_<ch>`, `bkg_<ch>`, observable `x`, category `index`).

Rendering is done with matplotlib (ROOT is used only to read the workspace and
evaluate the PDFs), which produces clearer, higher-resolution figures than the
RooPlot/TCanvas output.

Usage:
    python3 plot_workspace.py                                  # all workspaces/*.root
    python3 plot_workspace.py workspaces/simple_workspace.root
    python3 plot_workspace.py ws.root -o plots/ws.png --bins 25
    python3 plot_workspace.py ws.root --fit-result output_simple/simple_workspace_result.root
"""

import argparse
import glob
import math
import os
import re
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

COL_MODEL = "C0"  # total S+B model
COL_BKG = "C3"  # background
COL_SIG = "C2"  # signal
COL_DATA = "black"  # toy data points

CURVE_POINTS = 300  # grid resolution for smooth PDF curves


def channel_names(sim) -> list[str]:
    """Return channel labels from the simultaneous PDF's index category,
    sorted by their trailing integer (ch0, ch1, ... ch10) rather than
    lexicographically."""
    cat = sim.indexCat()
    names = []
    for state in cat:
        # ROOT 6.22+ iterates a RooCategory as (name, index) pairs.
        names.append(str(state[0]) if not isinstance(state, str) else str(state))

    def key(n: str):
        m = re.search(r"(\d+)$", n)
        return (int(m.group(1)) if m else 0, n)

    return sorted(names, key=key)


def apply_fit_result(ws, result_file: str) -> bool:
    """Set workspace variables to the post-fit values in a quickFit result
    file. Returns True on success."""
    try:
        f = ROOT.TFile.Open(result_file)
    except OSError:
        f = None
    if not f or f.IsZombie():
        print(f"WARNING: could not open fit result {result_file}; using nominal snapshot")
        return False
    fr = f.Get("fitResult")
    if not fr:
        print(f"WARNING: no 'fitResult' in {result_file}; using nominal snapshot")
        f.Close()
        return False
    for v in fr.floatParsFinal():
        wv = ws.var(v.GetName())
        if wv and not wv.isConstant():
            wv.setVal(v.getVal())
    f.Close()
    return True


def yields(ws, ch: str) -> tuple[float, float]:
    """Return (expected signal, expected background) yields for a channel
    from the current parameter values."""
    nsig = ws.function(f"nsig_tot_{ch}")
    nbkg = ws.function(f"nbkg_tot_{ch}") or ws.var(f"nbkg_{ch}")
    return (nsig.getVal() if nsig else float("nan"), nbkg.getVal() if nbkg else float("nan"))


def data_hist(data, x, ch: str, bins: int, xlo: float, xhi: float):
    """Bin the toy data for one channel; return (centres, counts, errors, n_obs)."""
    data_ch = data.reduce(ROOT.RooFit.Cut(f"index==index::{ch}"))
    vals = np.empty(data_ch.numEntries())
    wts = np.empty(data_ch.numEntries())
    for i in range(data_ch.numEntries()):
        row = data_ch.get(i)
        vals[i] = row.getRealValue("x")
        wts[i] = data_ch.weight()
    counts, edges = np.histogram(vals, bins=bins, range=(xlo, xhi), weights=wts)
    centres = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(counts)
    return centres, counts, errors, float(wts.sum())


def pdf_curve(pdf, x, grid, norm_set, scale: float):
    """Evaluate a normalised PDF over `grid` and scale to events/bin."""
    y = np.empty(len(grid))
    for i, xv in enumerate(grid):
        x.setVal(float(xv))
        y[i] = pdf.getVal(norm_set)
    return y * scale


def plot_channel(ax, ws, x, data, ch: str, bins: int):
    """Draw one channel onto a matplotlib Axes; return the summary info dict."""
    xlo, xhi = x.getMin(), x.getMax()
    bin_width = (xhi - xlo) / bins

    centres, counts, errors, n_obs = data_hist(data, x, ch, bins, xlo, xhi)

    n_sig, n_bkg = yields(ws, ch)
    grid = np.linspace(xlo, xhi, CURVE_POINTS)
    norm_set = ROOT.RooArgSet(x)

    model = ws.pdf(f"model_{ch}")
    bkg = ws.pdf(f"bkg_{ch}")
    sig = ws.pdf(f"sig_{ch}")

    y_model = pdf_curve(model, x, grid, norm_set, (n_sig + n_bkg) * bin_width)
    y_bkg = pdf_curve(bkg, x, grid, norm_set, n_bkg * bin_width)
    y_sig = pdf_curve(sig, x, grid, norm_set, n_sig * bin_width)

    # Reduced chi-square of data vs the total model evaluated at bin centres.
    x_save = x.getVal()
    exp = np.empty(bins)
    edges = np.linspace(xlo, xhi, bins + 1)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    for i, xc in enumerate(bin_centres):
        x.setVal(float(xc))
        exp[i] = model.getVal(norm_set) * (n_sig + n_bkg) * bin_width
    x.setVal(x_save)
    mask = exp > 0
    chi2 = float(np.sum((counts[mask] - exp[mask]) ** 2 / exp[mask]) / bins)

    ax.plot(grid, y_model, color=COL_MODEL, lw=2.0, label="S+B model")
    ax.plot(grid, y_bkg, color=COL_BKG, lw=1.8, ls="--", label=f"background (B={n_bkg:.1f})")
    ax.plot(grid, y_sig, color=COL_SIG, lw=1.8, ls=":", label=f"signal (S={n_sig:.1f})")
    ax.errorbar(
        centres,
        counts,
        yerr=errors,
        fmt="o",
        color=COL_DATA,
        ms=4,
        capsize=2,
        lw=1,
        label="toy data",
    )

    ax.set_title(ch, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("events / bin")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)

    handles, labels = ax.get_legend_handles_labels()
    info_line = f"obs {int(round(n_obs))} / exp {n_sig + n_bkg:.1f}\n$\\chi^2$/bins = {chi2:.2f}"
    leg = ax.legend(handles, labels, fontsize=8, framealpha=0.0, loc="upper right", title=info_line)
    leg.get_title().set_fontsize(8)

    return dict(n_obs=int(round(n_obs)), n_exp=n_sig + n_bkg, n_sig=n_sig, n_bkg=n_bkg, chi2=chi2)


def plot_workspace(
    path: str, out_path: str, *, bins: int, fit_result: str | None, snapshot: str
) -> None:
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        print(f"ERROR: cannot open {path}")
        return
    ws = f.Get("combWS")
    if not ws:
        print(f"ERROR: no 'combWS' workspace in {path}")
        f.Close()
        return
    mc = ws.obj("ModelConfig")
    sim = mc.GetPdf() if mc else ws.pdf("sim_pdf")
    x = ws.var("x")
    data = ws.data("combData")

    if fit_result and apply_fit_result(ws, fit_result):
        param_src = f"post-fit ({os.path.basename(fit_result)})"
    else:
        if ws.loadSnapshot(snapshot):
            param_src = f"snapshot '{snapshot}'"
        else:
            param_src = "as-built parameters"

    chans = channel_names(sim)
    ncols = math.ceil(math.sqrt(len(chans)))
    nrows = math.ceil(len(chans) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    flat = axes.flatten()

    stem = os.path.splitext(os.path.basename(path))[0]
    fig.suptitle(f"{stem}   [{param_src}]", fontsize=13)

    print(f"\n{stem}  [{param_src}]")
    print(f"  {'channel':<8} {'obs':>5} {'exp':>8} {'S':>7} {'B':>7} {'chi2/bins':>10}")
    for i, ch in enumerate(chans):
        info = plot_channel(flat[i], ws, x, data, ch, bins)
        print(
            f"  {ch:<8} {info['n_obs']:>5d} {info['n_exp']:>8.1f} "
            f"{info['n_sig']:>7.1f} {info['n_bkg']:>7.1f} {info['chi2']:>10.2f}"
        )

    for ax in flat[len(chans) :]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  -> {out_path}")
    f.Close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs", nargs="*", help="Workspace .root files (default: workspaces/*.root)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file for a single input, or a directory for "
        "multiple inputs (default: plots/<stem>_channels.png)",
    )
    parser.add_argument(
        "--outdir", default="plots", help="Directory for per-workspace PNGs (default: plots/)"
    )
    parser.add_argument(
        "--bins", type=int, default=25, help="Number of bins for the observable x (default: 25)"
    )
    parser.add_argument(
        "--fit-result",
        default=None,
        help="quickFit result .root file; overlay post-fit model instead of the nominal snapshot",
    )
    parser.add_argument(
        "--snapshot", default="nominal", help="Parameter snapshot to load (default: nominal)"
    )
    args = parser.parse_args()

    paths = args.inputs or sorted(glob.glob("workspaces/*.root"))
    if not paths:
        print("No workspace files found (looked in workspaces/*.root). Pass paths explicitly.")
        sys.exit(1)

    single = len(paths) == 1
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if (
            single
            and args.output
            and not args.output.endswith("/")
            and not os.path.isdir(args.output)
        ):
            out_path = args.output
        else:
            outdir = (
                args.output
                if (args.output and (args.output.endswith("/") or os.path.isdir(args.output)))
                else args.outdir
            )
            out_path = os.path.join(outdir, f"{stem}_channels.png")
        plot_workspace(
            path, out_path, bins=args.bins, fit_result=args.fit_result, snapshot=args.snapshot
        )


if __name__ == "__main__":
    main()

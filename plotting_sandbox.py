#!/usr/bin/env python3
"""
Plot a 2*Delta-NLL scan comparison (quickFit vs pyhs3) from a JSON file like
bbyy_nlls.json, in the style of sample_nll_scan.pdf.

The JSON is a list of scan entries; each entry has:
    mus         POI values scanned
    qf_nlls     quickFit / ROOT NLL at each point   (the reference)
    pyhs3_nlls  pyhs3 NLL at each point
    diffs       residual (pyhs3 - offset) - qf at each point

The figure has two panels sharing the x-axis:
    top     2*Delta-NLL = 2*(NLL - min NLL) for both engines
    bottom  residual (pyhs3 - xRooFit), in units of 1e-3

This script also accepts two-column ".dat" profiles produced by iminuit
(e.g. muWH_iminuit_profile.dat, muZH_iminuit_profile.dat). These hold a single
profile curve, so they render as a single-panel figure in the same style: the
first line is a header naming the POI (e.g. "muWH DeltaNLL"), remaining lines
are "<mu> <DeltaNLL>" pairs. The DeltaNLL column is doubled to 2*Delta-NLL so
the curve and the 2*dNLL=1 (1-sigma) reference line match the JSON plots.

Kept deliberately simple so it is easy to tweak.

Usage:
    python3 plotting_sandbox.py                            # reads bbyy_nlls.json
    python3 plotting_sandbox.py my_nlls.json
    python3 plotting_sandbox.py my_nlls.json -o scan.pdf
    python3 plotting_sandbox.py muWH_iminuit_profile.dat   # single-panel .dat plot
    python3 plotting_sandbox.py muWH_iminuit_profile.dat muZH_iminuit_profile.dat
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def twice_delta_nll(nlls):
    """2*(NLL - min NLL): a well with its minimum sitting at zero."""
    lo = min(nlls)
    return [2.0 * (n - lo) for n in nlls]


def plot_entry(entry, out_path):
    mus = entry["mus"]
    qf = twice_delta_nll(entry["qf_nlls"])
    py = twice_delta_nll(entry["pyhs3_nlls"])
    resid = [d / 1e-3 for d in entry["diffs"]]  # (pyhs3 - xRooFit) / 1e-3

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # --- top panel: the two scans ---
    ax.plot(mus, qf, "o-", color="navy", ms=5, lw=1.4,
            label="ROOT/C++ (quickFit)")
    ax.plot(mus, py, "x--", color="firebrick", ms=6, lw=1.2,
            label="pyhs3 + iminuit")
    ax.axhline(1.0, color="black", ls=":", lw=0.8)          # 2*dNLL = 1
    # ax.text(mus[-1], 1.0, r" $2\Delta$NLL$=1$", va="center",
            # ha="left", fontsize=8, color="grey")
    ax.set_ylabel(r"$2\Delta$NLL", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.grid(True)

    # --- bottom panel: residual ---
    axr.plot(mus, resid, "x--", color="firebrick", ms=6, lw=1.2)
    axr.axhline(0.0, color="grey", lw=0.8)
    axr.set_ylabel(r"(pyhs3 $-$ Root)$/10^{-3}$", fontsize=11)
    axr.set_xlabel(r"$\mu_\mathrm{HH}$", fontsize=11)
    axr.grid(True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def load_dat(path):
    """Read a two-column "<poi> DeltaNLL" .dat profile.

    Returns (poi, mus, dnlls). The first line is a header whose first token
    names the POI (e.g. "muWH"); every remaining line is a "<mu> <DeltaNLL>"
    pair. Blank lines and any non-numeric leading lines are treated as header.
    """
    poi = None
    mus, dnlls = [], []
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            try:
                x = float(tok[0])
            except ValueError:
                # non-numeric first token -> header line naming the POI
                if poi is None:
                    poi = tok[0]
                continue
            mus.append(x)
            dnlls.append(float(tok[1]))
    return poi, mus, dnlls


def poi_axis_label(poi):
    r"""Turn a header token like "muWH" into a label $\mu_\mathrm{WH}$."""
    if poi and poi.lower().startswith("mu") and len(poi) > 2:
        return r"$\mu_\mathrm{%s}$" % poi[2:]
    return poi if poi else r"$\mu$"


def plot_dat(path, out_path):
    """Single-panel 2*Delta-NLL plot for an iminuit .dat profile.

    Mirrors the top panel of plot_entry: same colors, grid, and the
    2*dNLL=1 reference line. The .dat holds only the iminuit curve, so there
    is no ROOT reference and no residual panel.
    """
    poi, mus, dnlls = load_dat(path)
    y = [2.0 * d for d in dnlls]  # DeltaNLL -> 2*Delta-NLL

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mus, y, "x--", color="firebrick", ms=6, lw=1.2,
            label="pyhs3 + iminuit")
    ax.axhline(1.0, color="black", ls=":", lw=0.8)          # 2*dNLL = 1
    ax.set_ylabel(r"$2\Delta$NLL", fontsize=11)
    ax.set_xlabel(poi_axis_label(poi), fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.grid(True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _suffixed(out_path, tag):
    """Insert `tag` before the extension of out_path (foo.pdf -> foo_tag.pdf)."""
    stem, _, ext = out_path.rpartition(".")
    return f"{stem}_{tag}.{ext}" if stem else f"{out_path}_{tag}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", default=["bbyy_nlls.json"],
                   help="Scan file(s): HS3-style .json (default: bbyy_nlls.json) "
                        "or two-column .dat iminuit profiles")
    p.add_argument("-o", "--output", default=None,
                   help="Output figure. Used verbatim for a single figure; "
                        "otherwise names are derived per input "
                        "(default: <stem>.pdf for .dat, nll_scan.pdf for .json)")
    args = p.parse_args()

    single_input = len(args.inputs) == 1

    for inp in args.inputs:
        if inp.lower().endswith(".dat"):
            # One single-panel figure per .dat file.
            if args.output and single_input:
                out = args.output
            else:
                out = os.path.splitext(os.path.basename(inp))[0] + ".pdf"
            plot_dat(inp, out)
            continue

        # Otherwise treat as an HS3-style JSON scan file.
        with open(inp) as fh:
            entries = json.load(fh)

        base = args.output or "nll_scan.pdf"
        if not single_input:
            # Keep multiple JSON inputs from colliding on the same name.
            base = _suffixed(base, os.path.splitext(os.path.basename(inp))[0])

        # One figure per entry; suffix the filename when there is more than one.
        for i, entry in enumerate(entries):
            out = base if len(entries) == 1 else _suffixed(base, str(i))
            plot_entry(entry, out)


if __name__ == "__main__":
    main()

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
    bottom  residual (pyhs3 - xRooFit), in units of 1e-4

Kept deliberately simple so it is easy to tweak.

Usage:
    python3 plotting_sandbox.py                       # reads bbyy_nlls.json
    python3 plotting_sandbox.py my_nlls.json
    python3 plotting_sandbox.py my_nlls.json -o scan.pdf
"""

import argparse
from inspect import AGEN_CLOSED
import json

import matplotlib.pyplot as plt


def twice_delta_nll(nlls):
    """2*(NLL - min NLL): a well with its minimum sitting at zero."""
    lo = min(nlls)
    return [2.0 * (n - lo) for n in nlls]


def plot_entry(entry, out_path):
    mus = entry["mus"]
    qf = twice_delta_nll(entry["qf_nlls"])
    py = twice_delta_nll(entry["pyhs3_nlls"])
    resid = [d / 1e-4 for d in entry["diffs"]]  # (pyhs3 - xRooFit) / 1e-4

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6),
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
    ax.set_ylabel(r"$2\Delta$NLL")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True)

    # --- bottom panel: residual ---
    axr.plot(mus, resid, "x--", color="firebrick", ms=6, lw=1.2)
    axr.axhline(0.0, color="grey", lw=0.8)
    axr.set_ylabel(r"(pyhs3 $-$ xRooFit)$/10^{-4}$", fontsize=9)
    axr.set_xlabel(r"$\mu_\mathrm{sig}$")
    axr.grid(True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", default="bbyy_nlls.json",
                   help="JSON scan file (default: bbyy_nlls.json)")
    p.add_argument("-o", "--output", default="nll_scan.pdf",
                   help="Output figure (default: nll_scan.pdf)")
    args = p.parse_args()

    with open(args.input) as fh:
        entries = json.load(fh)

    # One figure per entry; suffix the filename when there is more than one.
    for i, entry in enumerate(entries):
        out = args.output
        if len(entries) > 1:
            stem, _, ext = args.output.rpartition(".")
            out = f"{stem}_{i}.{ext}" if stem else f"{args.output}_{i}"
        plot_entry(entry, out)


if __name__ == "__main__":
    main()

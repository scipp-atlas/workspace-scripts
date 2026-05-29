#!/usr/bin/env python3
"""
Run a quickFit profile likelihood scan over mu_sig and save results to JSON.

For each mu value the POI is fixed and all NPs are profiled (minimised).
The result file from each fit is read to extract the NLL and post-fit
parameter values with errors.  A delta-NLL column (2*(NLL - NLL_min)) is
appended once all scan points are collected.

Usage:
    python3 muscan.py [options]

    python3 muscan.py --mu-vals "-1 -0.5 0 0.5 1 1.5 2"
    python3 muscan.py --mu-min -1 --mu-max 3 --mu-step 0.25 --output scan.json
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)


# ── helpers ──────────────────────────────────────────────────────────────────

def arange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range robust to floating-point drift."""
    vals, x = [], start
    while x <= stop + step * 1e-9:
        vals.append(round(x, 10))
        x += step
    return vals


def detect_poi(wsfile: str, ws_name: str = "combWS",
               mc_name: str = "ModelConfig") -> str | None:
    """Return the first POI name from the ModelConfig, or None on failure."""
    f = ROOT.TFile(wsfile)
    if f.IsZombie():
        return None
    ws = f.Get(ws_name)
    mc = ws and ws.obj(mc_name)
    poi = None
    if mc:
        pois = mc.GetParametersOfInterest()
        if pois and pois.getSize() > 0:
            poi = pois.first().GetName()
    f.Close()
    return poi


def detect_constraint(wsfile: str, ws_name: str = "combWS",
                      constr_name: str = "constr_alpha_sigma") -> str | None:
    """Return the constraint PDF name if it exists in the workspace, else None."""
    f = ROOT.TFile(wsfile)
    if f.IsZombie():
        return None
    ws = f.Get(ws_name)
    found = ws and ws.pdf(constr_name)
    f.Close()
    return constr_name if found else None


def detect_bkg_type(wsfile: str, ws_name: str = "combWS",
                    probe: str = "bkg_ch0") -> str:
    """Return 'exponential' or 'generic' based on the background PDF class."""
    f = ROOT.TFile(wsfile)
    if f.IsZombie():
        return "unknown"
    ws = f.Get(ws_name)
    pdf = ws and ws.pdf(probe)
    cls = pdf.ClassName() if pdf else ""
    f.Close()
    if cls == "RooExponential":
        return "exponential"
    if cls == "RooGenericPdf":
        return "generic"
    return cls or "unknown"


def run_quickfit(wsfile: str, mu_val: float, result_file: str,
                 poi: str, extra_args: list[str]) -> tuple[bool, str]:
    """Run quickFit with the POI fixed; return (converged, combined log)."""
    cmd = [
        "quickFit",
        "-f", wsfile,
        "-w", "combWS", "-m", "ModelConfig", "-d", "combData",
        "-p", f"{poi}={mu_val}",
        "-o", result_file,
        *extra_args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def read_result(result_file: str) -> dict | None:
    """Read NLL, status, and post-fit parameters from a quickFit result file."""
    f = ROOT.TFile(result_file)
    if f.IsZombie():
        return None

    tree = f.Get("nllscan")
    if not tree or tree.GetEntries() == 0:
        f.Close()
        return None
    tree.GetEntry(0)
    nll    = float(tree.nll)
    status = int(tree.status)

    fr = f.Get("fitResult")
    params = {}
    if fr:
        for v in fr.floatParsFinal():
            params[v.GetName()] = {"value": v.getVal(), "error": v.getError()}

    f.Close()
    return {"nll": nll, "fit_status": status, "parameters": params}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    mu_group = parser.add_mutually_exclusive_group()
    mu_group.add_argument("--mu-vals", type=str,
                          help="Space-separated list of mu values to scan")
    mu_group.add_argument("--mu-min",  type=float, default=0.0,
                          help="Scan start (default: 0.0)")

    parser.add_argument("--mu-max",  type=float, default=3.0,
                        help="Scan end (default: 3.0)")
    parser.add_argument("--mu-step", type=float, default=0.25,
                        help="Scan step size (default: 0.25)")
    parser.add_argument("--input",   default="simple_workspace.root",
                        help="Input ROOT workspace file (default: simple_workspace.root)")
    parser.add_argument("--output",  default="muscan.json",
                        help="Output JSON file (default: muscan.json)")
    parser.add_argument("--logdir",  default="output_simple",
                        help="Directory for per-mu quickFit logs and result files "
                             "(default: output_simple)")
    parser.add_argument("--poi",     default=None,
                        help="Parameter of interest name (default: auto-detect from ModelConfig)")
    parser.add_argument("--nll-offset", action="store_true",
                        help="Pass --nllOffset 0 to quickFit (suppresses automatic NLL offsetting)")
    args = parser.parse_args()

    if args.mu_vals:
        mu_values = [float(v) for v in args.mu_vals.split()]
    else:
        mu_values = arange(args.mu_min, args.mu_max, args.mu_step)

    os.makedirs(args.logdir, exist_ok=True)

    # Auto-detect workspace properties.
    if args.poi is None:
        args.poi = detect_poi(args.input)
        if args.poi is None:
            print("ERROR: could not detect POI from ModelConfig; use --poi to specify it")
            sys.exit(1)
    constraint = detect_constraint(args.input)
    quickfit_extra = ["--externalConstraint", constraint] if constraint else []
    if args.nll_offset:
        quickfit_extra += ["--nllOffset", "0"]
    bkg_type = detect_bkg_type(args.input)

    print(f"Scanning {len(mu_values)} mu values: "
          f"{mu_values[0]:.3g} → {mu_values[-1]:.3g}")
    print(f"Workspace : {args.input}")
    print(f"POI       : {args.poi}")
    print(f"Background: {bkg_type}")
    print(f"Constraint: {constraint or '(none)'}")
    print(f"NLL offset: {'yes (--nllOffset 0)' if args.nll_offset else 'no'}")
    print(f"Logs      : {args.logdir}/")
    print(f"Output    : {args.output}")
    print()

    scan_points = []

    for mu in mu_values:
        mu_tag     = f"{mu:+.4f}".replace("+", "p").replace("-", "m").replace(".", "d")
        result_f   = f"{args.logdir}/result_mu_{mu_tag}.root"
        log_f      = f"{args.logdir}/log_mu_{mu_tag}.txt"

        converged, log_text = run_quickfit(args.input, mu, result_f, args.poi, quickfit_extra)

        # Write per-mu log regardless of convergence
        with open(log_f, "w") as fh:
            fh.write(log_text)

        point = read_result(result_f)
        if point is None:
            print(f"  mu={mu:+.4f}  ERROR: could not read result file")
            scan_points.append({args.poi: mu, "nll": None, "fit_status": -1,
                                 "parameters": {}})
            continue

        # Treat NaN NLL (fit diverged) as a failed point
        if point["nll"] is not None and math.isnan(point["nll"]):
            point["nll"] = None

        point[args.poi] = mu
        scan_points.append(point)

        if point["nll"] is None:
            print(f"  {args.poi}={mu:+.6f}  nll=NaN  FAILED({point['fit_status']})")
        else:
            status_str = "OK" if point["fit_status"] == 0 else f"FAILED({point['fit_status']})"
            print(f"  {args.poi}={mu:+.6f}  nll={point['nll']:+.6f}  {status_str}")

    # Compute delta_nll = 2*(nll - nll_min) across all converged points
    valid_nlls = [p["nll"] for p in scan_points
                  if p["nll"] is not None and not math.isnan(p["nll"])]
    nll_min    = min(valid_nlls) if valid_nlls else None
    for p in scan_points:
        if p["nll"] is not None and nll_min is not None:
            p["delta_nll"] = 2.0 * (p["nll"] - nll_min)
        else:
            p["delta_nll"] = None

    # Sort by POI value for clean output
    scan_points.sort(key=lambda p: p[args.poi])

    output = {
        "metadata": {
            "workspace":  args.input,
            "poi":        args.poi,
            "bkg_type":   bkg_type,
            "constraint": constraint,
            "n_points":   len(scan_points),
            "nll_min":    nll_min,
            "created":    datetime.now(timezone.utc).isoformat(),
        },
        "scan_points": scan_points,
    }

    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nWrote {len(scan_points)} scan points to {args.output}")
    if nll_min is not None:
        print(f"NLL minimum: {nll_min:.6f}")


if __name__ == "__main__":
    main()

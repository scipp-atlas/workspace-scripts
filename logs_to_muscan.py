#!/usr/bin/env python3
"""
Backfill converter: quickFit text logs -> standardized muscan JSON.

Legacy real-workspace scans were run with ad-hoc scripts that left behind only
a ``nlls.txt`` (one ``mu = X    nll = Y`` line per scan point) and per-point
Minuit logs (``log__mu_<mu>.txt``). This tool converts such a directory into
the same ``muscan.json`` schema that ``muscan.py`` writes, so downstream
pyhs3 evaluation (``pyhs3_eval/eval_simple_muscan.py``) works identically for
toy and real workspaces.

New scans should be produced directly with ``muscan.py`` (which reads post-fit
parameters from the quickFit result .root files); this converter exists only
to migrate scans that cannot cheaply be re-run.

NLL convention: values are kept in RooFit's single -log(L), exactly as they
appear in nlls.txt — the same convention muscan.py records.

Pure Python (no ROOT); can run anywhere the log directory is copied to.

Usage:
    python3 logs_to_muscan.py --log-dir output__workspace_FINAL_ISOBUGFIX \\
        --poi mu_HH --output scans/bbyy_muscan.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── nlls.txt ─────────────────────────────────────────────────────────────────

_NLL_LINE_RE = re.compile(r"mu\s*=\s*([-+\d.eE]+)\s+nll\s*=\s*([-+\d.eE]+)")


def parse_nlls_file(path: Path) -> dict[float, float]:
    """Parse nlls.txt into {mu: nll} (RooFit -log L, kept as-is)."""
    result: dict[float, float] = {}
    with path.open() as fh:
        for line in fh:
            m = _NLL_LINE_RE.search(line)
            if m:
                result[float(m.group(1))] = float(m.group(2))
    if not result:
        raise ValueError(f"No mu/nll pairs found in {path}")
    return result


# ── per-point Minuit logs ────────────────────────────────────────────────────

# Parameter lines appear after the "FVAL = ..." line:
#     NAME\t  = VALUE\t +/-  ERROR\t(limited)
# The error column is absent for fixed parameters.
_PARAM_RE = re.compile(r"^(\S+)\t\s+=\s+([-+\d.eE]+)(?:\t\s*\+/-\s+([-+\d.eE]+))?")
_FVAL_RE = re.compile(r"^FVAL\s+=\s+[-+\d.eE]+")
_STATUS_RE = re.compile(r"minimum\s*-\s*status\s*=\s*(-?\d+)", re.IGNORECASE)


def parse_quickfit_log(log_path: Path) -> dict:
    """Extract post-fit parameters and fit status from a quickFit Minuit log.

    Returns {"fit_status": int, "parameters": {name: {value, error}}}.
    If several FVAL blocks appear, later assignments overwrite earlier ones,
    so the final (converged) block wins.
    """
    params: dict[str, dict[str, float]] = {}
    status = -1
    in_param_block = False

    with log_path.open() as fh:
        for line in fh:
            m = _STATUS_RE.search(line)
            if m:
                status = int(m.group(1))
            if _FVAL_RE.match(line):
                in_param_block = True
                continue
            if in_param_block:
                m = _PARAM_RE.match(line)
                if m:
                    params[m.group(1)] = {
                        "value": float(m.group(2)),
                        "error": float(m.group(3)) if m.group(3) is not None else 0.0,
                    }

    return {"fit_status": status, "parameters": params}


def find_log_for_mu(mu: float, log_dir: Path, pattern: str) -> Path | None:
    """Return the log file matching *mu* (exact stem first, then closest)."""
    exact = log_dir / pattern.replace("*", f"{mu:g}")
    if exact.exists():
        return exact

    mu_re = re.compile(re.escape(pattern).replace(r"\*", r"(-?[\d.]+)") + "$")
    candidates: list[tuple[float, Path]] = []
    for p in sorted(log_dir.glob(pattern)):
        m = mu_re.match(p.name)
        if m:
            try:
                candidates.append((float(m.group(1)), p))
            except ValueError:
                continue
    if not candidates:
        return None

    best_mu, best = min(candidates, key=lambda c: abs(c[0] - mu))
    if abs(best_mu - mu) > 1e-6:
        print(f"  WARNING: no log for mu={mu}; nearest is mu={best_mu} ({best.name}) — skipping")
        return None
    return best


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory containing nlls.txt and the per-mu log files",
    )
    parser.add_argument(
        "--poi",
        required=True,
        help="Name of the parameter of interest (e.g. mu_HH); keys the scan points",
    )
    parser.add_argument(
        "--nlls",
        type=Path,
        default=None,
        help="Path to nlls.txt (default: <log-dir>/nlls.txt)",
    )
    parser.add_argument(
        "--log-pattern",
        default="log__mu_*.txt",
        help="Glob for per-mu log files; '*' is the mu value (default: log__mu_*.txt)",
    )
    parser.add_argument(
        "--workspace",
        default="",
        help="Workspace file the scan was run on (recorded in metadata only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <log-dir>/muscan.json)",
    )
    args = parser.parse_args()

    nlls_path = args.nlls or args.log_dir / "nlls.txt"
    output_path = args.output or args.log_dir / "muscan.json"

    print(f"Reading NLLs from {nlls_path}")
    nlls = parse_nlls_file(nlls_path)
    print(f"  {len(nlls)} scan points")

    scan_points = []
    n_missing_logs = 0
    for mu in sorted(nlls):
        point = {args.poi: mu, "nll": nlls[mu], "fit_status": -1, "parameters": {}}
        log_path = find_log_for_mu(mu, args.log_dir, args.log_pattern)
        if log_path is None:
            n_missing_logs += 1
            print(f"  mu={mu:+.6g}  nll={nlls[mu]:+.6f}  NO LOG (no post-fit parameters)")
        else:
            parsed = parse_quickfit_log(log_path)
            point["fit_status"] = parsed["fit_status"]
            point["parameters"] = parsed["parameters"]
            status_str = "OK" if parsed["fit_status"] == 0 else f"status={parsed['fit_status']}"
            print(
                f"  mu={mu:+.6g}  nll={nlls[mu]:+.6f}  "
                f"{len(parsed['parameters'])} params  {status_str}"
            )
        scan_points.append(point)

    if n_missing_logs == len(scan_points):
        print("ERROR: no log files matched any scan point — check --log-dir/--log-pattern")
        sys.exit(1)

    nll_min = min(nlls.values())
    for p in scan_points:
        p["delta_nll"] = 2.0 * (p["nll"] - nll_min)

    output = {
        "metadata": {
            "workspace": args.workspace,
            "poi": args.poi,
            "constraint": None,
            "nll_convention": "-log L (RooFit single negative log-likelihood)",
            "source": f"logs_to_muscan.py backfill from {args.log_dir}",
            "n_points": len(scan_points),
            "nll_min": nll_min,
            "created": datetime.now(timezone.utc).isoformat(),
        },
        "scan_points": scan_points,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nWrote {len(scan_points)} scan points to {output_path}")
    print(f"NLL minimum: {nll_min:.6f}")
    if n_missing_logs:
        print(f"WARNING: {n_missing_logs} point(s) had no log file (empty parameters)")


if __name__ == "__main__":
    main()

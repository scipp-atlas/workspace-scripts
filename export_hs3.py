#!/usr/bin/env python3
"""
Export a RooFit workspace to HS3 (HEP Statistics Serialization Standard) format.

Export a RooFit workspace to HS3 JSON using ROOT's RooJSONFactoryWSTool.
Optionally verifies the round-trip by re-importing the HS3 file and checking
that all key workspace objects are correctly reconstructed.

Usage:
    python3 export_hs3.py [--input simple_workspace.root] [--output-stem simple_workspace]
                          [--verify]

Examples:
    python3 export_hs3.py
    python3 export_hs3.py --input my_ws.root --output-stem my_ws --verify
"""

import argparse
import sys

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)


def load_workspace(root_file: str, ws_name: str = "combWS") -> ROOT.RooWorkspace:
    f = ROOT.TFile(root_file)
    if f.IsZombie():
        sys.exit(f"ERROR: cannot open {root_file!r}")
    ws = f.Get(ws_name)
    if not ws:
        sys.exit(f"ERROR: workspace {ws_name!r} not found in {root_file!r}")
    # Keep the file open; RooWorkspace is owned by the file handle we return
    return ws, f


def cleanup_exponential_functions(doc: dict) -> dict:
    """
    Remove ROOT's sign-inversion intermediates for exponential distributions.

    ROOT exports RooExponential(x, tau) with negative tau as:
      functions:     [{name: "tau_exponential_inverted", type: "generic_function",
                       expression: "-tau"}]
      distributions: [{name: "bkg", type: "exponential_dist", x: "x",
                       c: "tau_exponential_inverted"}]

    because HS3's exponential_dist convention is exp(-c*x) with c > 0, while
    RooExponential uses exp(tau*x) with tau < 0.

    We simplify by replacing the inversion chain with a direct parameter reference,
    flipping the sign of tau everywhere so it becomes a positive decay rate:
      distributions: [{name: "bkg", type: "exponential_dist", x: "x", c: "tau"}]
    with tau > 0 in domains and parameter_points.
    """
    import re

    # Collect all generic_functions that are simple negations: expression = "-<varname>"
    inversions: dict[str, str] = {}  # function_name -> negated_variable_name
    for func in doc.get("functions", []):
        if func.get("type") == "generic_function":
            m = re.match(r"^-(\w+)$", func.get("expression", ""))
            if m:
                inversions[func["name"]] = m.group(1)

    if not inversions:
        return doc

    # Only act on inversions that are referenced as 'c' in an exponential_dist.
    # Replace those c references with the underlying variable name.
    affected_vars: set[str] = set()
    for dist in doc.get("distributions", []):
        if dist.get("type") == "exponential_dist" and dist.get("c") in inversions:
            orig = inversions[dist["c"]]
            dist["c"] = orig
            affected_vars.add(orig)

    if not affected_vars:
        return doc

    # Flip the sign of affected variables in domains (min/max) and parameter_points.
    for domain in doc.get("domains", []):
        for axis in (domain.get("axes") or []):
            if axis["name"] in affected_vars:
                axis["min"], axis["max"] = -axis["max"], -axis["min"]

    for pp in doc.get("parameter_points", []):
        for param in pp.get("parameters", []):
            if param["name"] in affected_vars:
                param["value"] = -param["value"]

    # Remove the inversion generic_function entries.
    removed = {name for name, orig in inversions.items() if orig in affected_vars}
    doc["functions"] = [f for f in doc.get("functions", [])
                        if f["name"] not in removed]

    # Remove their roofit_skip attributes from ROOT_internal.
    attrs = (doc.get("misc", {}).get("ROOT_internal", {}).get("attributes") or {})
    for name in removed:
        attrs.pop(name, None)

    return doc


def cleanup_for_pyhs3(doc: dict, use_aux_distributions: bool = True) -> dict:
    """
    Fixes for pyhs3 compatibility:

    1. Replace null axes in empty domains (e.g. global-observables) with [].
    2. Add min/max to dataset axis entries and drop the ROOT-internal value field.
    3. Split the single combined likelihood into one likelihood per channel, and
       update the analysis to reference them all via "likelihoods".
    4. Add "init": "default_values" to each analysis so pyhs3 knows which
       parameter set carries the const flags.
    5. Remove non-constant observables (e.g. x) from default_values — they must
       not appear in parameter sets or they overwrite the data array at eval time.
       Constant parameters (global observables) are kept even if in a dataset.
    6. Wire standalone constraint PDFs into the first per-channel likelihood.
       Constraint PDFs are those not yet referenced in any likelihood whose 'x'
       field names a constant parameter (the global observable). A single-entry
       dataset is created from default_values and added alongside the constraint.
       When use_aux_distributions=True (default), constraints are placed under
       "aux_distributions"/"aux_data" so tools normalise them correctly.
       When False, constraints are placed under "distributions"/"data" (old
       behaviour, kept for compatibility).
    """
    # ── Fix 1: replace null axes with [] ────────────────────────────────────
    for domain in doc.get("domains", []):
        if domain.get("axes") is None:
            domain["axes"] = []

    # ── Fix 2: add min/max to dataset axes, drop value ──────────────────────
    obs_ranges: dict[str, dict] = {}
    for domain in doc.get("domains", []):
        if domain.get("name") == "default_domain":
            for ax in (domain.get("axes") or []):
                obs_ranges[ax["name"]] = {"min": ax["min"], "max": ax["max"]}

    obs_names: set[str] = set()
    for dataset in doc.get("data", []):
        for ax in (dataset.get("axes") or []):
            obs_names.add(ax["name"])
            if ax["name"] in obs_ranges:
                ax["min"] = obs_ranges[ax["name"]]["min"]
                ax["max"] = obs_ranges[ax["name"]]["max"]
            ax.pop("value", None)

    # ── Fix 5: remove non-const observables from default_values ─────────────
    # Constant global observables are preserved even if they appear in a dataset.
    for pp in doc.get("parameter_points", []):
        if pp.get("name") == "default_values":
            pp["parameters"] = [p for p in pp["parameters"]
                                 if p["name"] not in obs_names or p.get("const", False)]

    # ── Fix 3: split combined likelihood into one per channel ────────────────
    old_to_new: dict[str, list[str]] = {}
    new_likelihoods = []
    for lh in doc.get("likelihoods", []):
        data_list = lh.get("data", [])
        dist_list = lh.get("distributions", [])
        if len(data_list) > 1 and len(data_list) == len(dist_list):
            new_names = []
            for data_name, dist_name in zip(data_list, dist_list):
                suffix   = data_name.split("_", 1)[1] if "_" in data_name else data_name
                lh_name  = f"L_{suffix}"
                new_likelihoods.append({
                    "name":          lh_name,
                    "distributions": [dist_name],
                    "data":          [data_name],
                })
                new_names.append(lh_name)
            old_to_new[lh["name"]] = new_names
        else:
            new_likelihoods.append(lh)
    doc["likelihoods"] = new_likelihoods

    # ── Fix 3 cont. + Fix 4: replace each split analysis with one per channel ─
    new_analyses = []
    for analysis in doc.get("analyses", []):
        old_lh = analysis.get("likelihood")
        if old_lh in old_to_new:
            base = {k: v for k, v in analysis.items() if k != "likelihood"}
            base.setdefault("init", "default_values")   # Fix 4
            for lh_name in old_to_new[old_lh]:
                new_analyses.append({**base, "name": lh_name, "likelihood": lh_name})
        else:
            analysis.setdefault("init", "default_values")  # Fix 4
            new_analyses.append(analysis)
    doc["analyses"] = new_analyses

    # ── Fix 6: attach constraint PDFs to the first per-channel likelihood ────
    # Identify distributions not yet referenced in any likelihood.
    referenced = {d for lh in doc["likelihoods"]
                  for d in (lh.get("distributions") or [])}
    unreferenced = {d["name"] for d in doc.get("distributions", [])} - referenced

    if unreferenced and doc.get("likelihoods"):
        # Build lookup of const parameter values from default_values.
        const_vals: dict[str, float] = {}
        for pp in doc.get("parameter_points", []):
            if pp.get("name") == "default_values":
                for p in pp["parameters"]:
                    if p.get("const"):
                        const_vals[p["name"]] = p["value"]

        # A constraint distribution is one whose 'x' field is a const parameter
        # (i.e. a global observable).
        constr_to_go: dict[str, tuple[str, float]] = {}
        for dist in doc.get("distributions", []):
            if dist["name"] in unreferenced and "x" in dist:
                x_name = dist["x"]
                if x_name in const_vals:
                    constr_to_go[dist["name"]] = (x_name, const_vals[x_name])

        if constr_to_go:
            # Collect unique global observables.
            go_info: dict[str, float] = {v[0]: v[1] for v in constr_to_go.values()}
            go_names = list(go_info.keys())
            go_vals  = [go_info[n] for n in go_names]

            # Create a single-entry global-observable dataset.
            go_dataset = {
                "name": "global_obs_data",
                "type": "unbinned",
                "axes": [{"name": n, "min": v - 5.0, "max": v + 5.0}
                         for n, v in go_info.items()],
                "entries": [go_vals],
            }
            doc.setdefault("data", []).append(go_dataset)

            # Attach each constraint to the first likelihood.
            first_lh = doc["likelihoods"][0]
            if use_aux_distributions:
                dist_key = "aux_distributions"
                data_key = "aux_data"
            else:
                dist_key = "distributions"
                data_key = "data"
            for cname in constr_to_go:
                first_lh.setdefault(dist_key, []).append(cname)
                first_lh.setdefault(data_key, []).append("global_obs_data")

            # Populate the global-observables domain that ROOT left empty.
            for domain in doc.get("domains", []):
                if "global_observables" in domain.get("name", ""):
                    domain["axes"] = [{"name": n, "min": v - 5.0, "max": v + 5.0}
                                      for n, v in go_info.items()]

            # Remove const flag from global observables in default_values.
            # Their value is provided by global_obs_data at evaluation time;
            # const: true would prevent pyhs3 from treating them as observables.
            for pp in doc.get("parameter_points", []):
                if pp.get("name") == "default_values":
                    for p in pp["parameters"]:
                        if p["name"] in go_info:
                            p.pop("const", None)

    return doc


def export_workspace(ws: ROOT.RooWorkspace, stem: str,
                     use_aux_distributions: bool = True) -> str:
    import json as _json
    path = f"{stem}.json"
    ROOT.RooJSONFactoryWSTool(ws).exportJSON(path)
    with open(path) as fh:
        doc = _json.load(fh)
    doc = cleanup_exponential_functions(doc)
    doc = cleanup_for_pyhs3(doc, use_aux_distributions=use_aux_distributions)
    with open(path, "w") as fh:
        _json.dump(doc, fh, indent=2)
    return path


def verify_roundtrip(hs3_file: str, ws_name: str = "combWS") -> bool:
    """Re-import an HS3 file and check that the key objects are present."""
    ws2 = ROOT.RooWorkspace(ws_name)
    tool = ROOT.RooJSONFactoryWSTool(ws2)
    # Suppress harmless duplicate-observable errors that arise when a shared
    # observable ('x') is re-added for each channel during simultaneous PDF import.
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)
    tool.importJSON(hs3_file)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    ok = True
    # ModelConfig is not part of HS3 and is intentionally excluded from checks.
    checks = {
        "sim_pdf (RooSimultaneous)": ws2.pdf("sim_pdf"),
        "combData (RooDataSet)":     ws2.data("combData"),
        "mu_sig (POI)":              ws2.var("mu_sig"),
    }
    print("  Round-trip verification:")
    for label, obj in checks.items():
        status = "OK" if obj else "MISSING"
        print(f"    {label:<35} {status}")
        if not obj:
            ok = False

    # Check event counts per channel
    data = ws2.data("combData")
    if data:
        cat = ws2.cat("index")
        if cat:
            for state in ["ch0", "ch1", "ch2"]:
                cat.setLabel(state)
                n = data.sumEntries(f"index=={cat.getIndex()}")
                print(f"    combData[{state}]: {int(n)} events")

    return ok


def summarise(hs3_file: str) -> None:
    """Print a human-readable summary of the exported HS3 file's top-level keys."""
    import json as _json
    import pathlib

    suffix = pathlib.Path(hs3_file).suffix
    if suffix == ".json":
        with open(hs3_file) as fh:
            doc = _json.load(fh)
        print("  HS3 top-level sections:")
        for key, val in doc.items():
            if isinstance(val, list):
                print(f"    {key}: {len(val)} entries")
            elif isinstance(val, dict):
                print(f"    {key}: {list(val.keys())}")
            else:
                print(f"    {key}: {val!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",       default="simple_workspace.root",
                        help="Input ROOT workspace file (default: simple_workspace.root)")
    parser.add_argument("--ws-name",     default="combWS",
                        help="Workspace name inside the ROOT file (default: combWS)")
    parser.add_argument("--output-stem", default=None,
                        help="Output file stem without extension "
                             "(default: input filename without .root)")
    parser.add_argument("--verify",      action="store_true",
                        help="Re-import the exported file and verify round-trip")
    parser.add_argument("--aux-constraints", dest="aux_constraints",
                        action="store_true", default=True,
                        help="Export constraint PDFs under aux_distributions/aux_data "
                             "(default; correct HS3 normalisation)")
    parser.add_argument("--no-aux-constraints", dest="aux_constraints",
                        action="store_false",
                        help="Export constraint PDFs under distributions/data (old behaviour)")
    args = parser.parse_args()

    stem = args.output_stem
    if stem is None:
        stem = args.input.removesuffix(".root")

    print(f"Loading workspace {args.ws_name!r} from {args.input!r} ...")
    ws, _f = load_workspace(args.input, args.ws_name)  # _f kept alive intentionally

    print("Exporting to HS3 (JSON) ...")
    import os
    path = export_workspace(ws, stem, use_aux_distributions=args.aux_constraints)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Wrote {path}  ({size_kb:.1f} kB)")
    summarise(path)

    if args.verify:
        print("Verifying round-trip ...")
        ok = verify_roundtrip(path, args.ws_name)
        if ok:
            print("  Round-trip: PASS")
        else:
            print("  Round-trip: FAIL — some objects were not reconstructed")
            sys.exit(1)


if __name__ == "__main__":
    main()

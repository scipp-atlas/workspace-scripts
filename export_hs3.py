#!/usr/bin/env python3
"""
Export a RooFit workspace to HS3 (HEP Statistics Serialization Standard) JSON.

Uses ROOT's RooJSONFactoryWSTool to export, then applies an ordered chain of
in-place fixes so the result is consumable by pyhs3 (each fix is individually
toggleable; see the --no-fix-* flags). Optionally verifies the round-trip by
re-importing the HS3 file and checking that all key workspace objects are
correctly reconstructed.

Usage:
    python3 export_hs3.py [--input simple_workspace.root] [--output-stem simple_workspace]
                          [--verify]

Examples:
    python3 export_hs3.py
    python3 export_hs3.py --input my_ws.root --output-stem my_ws --verify
"""

import argparse
import os
import re
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


def fix_exponential_functions(doc: dict) -> dict:
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
    # Replace those c references with the underlying variable name. The negated
    # name may itself be a function rather than a variable (with bkg-shape
    # systs the slope is tau_eff_<ch> = tau_<ch> * resp_bkg_shape_<ch>); then
    # the sign flip must land on the single variable factor of that product —
    # flipping nothing would silently leave c negative (a rising exponential).
    funcs_by_name = {f["name"]: f for f in doc.get("functions", [])}
    param_names = {
        p["name"] for pp in doc.get("parameter_points", []) for p in pp.get("parameters", [])
    }
    affected_vars: set[str] = set()
    for dist in doc.get("distributions", []):
        if dist.get("type") == "exponential_dist" and dist.get("c") in inversions:
            orig = inversions[dist["c"]]
            target = orig
            if orig not in param_names:
                fn = funcs_by_name.get(orig)
                var_factors = (
                    [f for f in fn.get("factors", []) if f in param_names]
                    if fn is not None and fn.get("type") == "product"
                    else []
                )
                if len(var_factors) != 1:
                    # Can't identify a unique variable to flip; keep ROOT's
                    # inversion intermediate for this distribution.
                    continue
                target = var_factors[0]
            dist["c"] = orig
            affected_vars.add(target)

    if not affected_vars:
        return doc

    # Flip the sign of affected variables in domains (min/max) and parameter_points.
    for domain in doc.get("domains", []):
        for axis in domain.get("axes") or []:
            if axis["name"] in affected_vars:
                axis["min"], axis["max"] = -axis["max"], -axis["min"]

    for pp in doc.get("parameter_points", []):
        for param in pp.get("parameters", []):
            if param["name"] in affected_vars:
                param["value"] = -param["value"]

    # Remove the inversion generic_function entries.
    removed = {name for name, orig in inversions.items() if orig in affected_vars}
    doc["functions"] = [f for f in doc.get("functions", []) if f["name"] not in removed]

    # Remove their roofit_skip attributes from ROOT_internal.
    attrs = doc.get("misc", {}).get("ROOT_internal", {}).get("attributes") or {}
    for name in removed:
        attrs.pop(name, None)

    return doc


def fix_null_axes(doc: dict) -> dict:
    """Fix: Replace null axes in empty domains (e.g. global-observables) with []."""
    for domain in doc.get("domains", []):
        if domain.get("axes") is None:
            domain["axes"] = []
    return doc


def fix_dataset_axes(doc: dict) -> dict:
    """Fix: Add min/max to dataset axis entries and drop the ROOT-internal value field."""
    obs_ranges: dict[str, dict] = {}
    for domain in doc.get("domains", []):
        if domain.get("name") == "default_domain":
            for ax in domain.get("axes") or []:
                obs_ranges[ax["name"]] = {"min": ax["min"], "max": ax["max"]}

    for dataset in doc.get("data", []):
        for ax in dataset.get("axes") or []:
            if ax["name"] in obs_ranges:
                ax["min"] = obs_ranges[ax["name"]]["min"]
                ax["max"] = obs_ranges[ax["name"]]["max"]
            ax.pop("value", None)

    return doc


def fix_unique_observables(doc: dict) -> dict:
    """Fix: give each channel of the combined likelihood its own observable name.

    The toy channels all share one observable ('x'), so every per-channel
    dataset in the combined likelihood carries an axis named 'x'. pyhs3 keys
    observed data by axis name within a likelihood, so duplicate names make
    the channels' data collide (its Workspace validation rejects them). Real
    combined workspaces avoid this by construction — each channel has its own
    observable (e.g. myy_ch1) — so this renames the observable to
    '<obs>_<channel>' per channel: in the dataset axes, in every
    distribution/function reachable from the channel's model (both FK
    references and generic-expression strings), and in the domains. The
    shared name is then retired from domains and parameter points. This is a
    pure renaming; NLL values are unchanged."""
    dists = {d["name"]: d for d in doc.get("distributions", [])}
    funcs = {f["name"]: f for f in doc.get("functions", [])}
    nodes = {**funcs, **dists}
    data_by_name = {d["name"]: d for d in doc.get("data", [])}

    token_pats: dict[str, re.Pattern] = {}

    def token_pat(name: str) -> re.Pattern:
        if name not in token_pats:
            token_pats[name] = re.compile(rf"\b{re.escape(name)}\b")
        return token_pats[name]

    def walk_strings(obj, visit):
        """Apply visit(container, key, string) to every string value (not dict
        keys, and not the object's own 'name' field)."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "name":
                    continue
                if isinstance(v, str):
                    visit(obj, k, v)
                else:
                    walk_strings(v, visit)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    visit(obj, i, v)
                else:
                    walk_strings(v, visit)

    def node_mentions(node: dict, obs: str) -> bool:
        pat = token_pat(obs)
        found = []
        walk_strings(node, lambda c, k, s: found.append(True) if pat.search(s) else None)
        return bool(found)

    def rewrite_node(node: dict, obs: str, new: str) -> None:
        pat = token_pat(obs)
        walk_strings(node, lambda c, k, s: c.__setitem__(k, pat.sub(new, s)))

    ident_pat = re.compile(r"[A-Za-z_]\w*")

    def node_refs(node: dict) -> set:
        """Names of other distributions/functions this node references, either
        as a direct FK string or as an identifier inside an expression."""
        refs: set = set()

        def visit(container, key, s):
            if s in nodes:
                refs.add(s)
            else:
                refs.update(t for t in ident_pat.findall(s) if t in nodes)

        walk_strings(node, visit)
        return refs

    renamed: dict[str, set] = {}  # old observable name -> set of new names
    node_axis_owner: dict[tuple, str] = {}  # (node name, old obs) -> new obs

    for lh in doc.get("likelihoods", []):
        data_list = lh.get("data") or []
        dist_list = lh.get("distributions") or []
        if len(data_list) != len(dist_list) or len(data_list) < 2:
            continue

        # Axis names appearing in more than one of this likelihood's datasets.
        counts: dict[str, int] = {}
        for data_name in data_list:
            for ax in data_by_name.get(data_name, {}).get("axes") or []:
                counts[ax["name"]] = counts.get(ax["name"], 0) + 1
        dup_axes = {n for n, c in counts.items() if c > 1}
        if not dup_axes:
            continue

        for i, (dist_name, data_name) in enumerate(zip(dist_list, data_list)):
            m = re.search(r"_(ch\d+)$", data_name) or re.search(r"_(ch\d+)$", dist_name)
            suffix = m.group(1) if m else f"c{i}"
            datum = data_by_name.get(data_name)
            if datum is None:
                continue
            for ax in datum.get("axes") or []:
                old = ax["name"]
                if old not in dup_axes:
                    continue
                new = f"{old}_{suffix}"
                ax["name"] = new
                renamed.setdefault(old, set()).add(new)
                # Rewrite every distribution/function reachable from the
                # channel's model that mentions the old observable.
                seen: set = set()
                queue = [dist_name]
                while queue:
                    name = queue.pop()
                    if name in seen or name not in nodes:
                        continue
                    seen.add(name)
                    node = nodes[name]
                    prev = node_axis_owner.get((name, old))
                    if prev is not None:
                        if prev != new:
                            sys.exit(
                                f"ERROR: fix_unique_observables: {name!r} is shared "
                                f"between channels but depends on observable {old!r} "
                                f"(would need both {prev!r} and {new!r}); rerun with "
                                "--no-fix-unique-observables"
                            )
                    elif node_mentions(node, old):
                        rewrite_node(node, old, new)
                        node_axis_owner[(name, old)] = new
                    queue.extend(node_refs(node))

    # Move each renamed observable's domain entries to the new names and
    # retire the shared name (unless something outside the renamed channel
    # subtrees still uses it).
    def natural_key(name: str):
        m = re.search(r"(\d+)$", name)
        return (int(m.group(1)) if m else -1, name)

    for old, new_names in renamed.items():
        still_used = any(node_mentions(node, old) for node in nodes.values())
        if still_used:
            print(
                f"  WARNING: fix_unique_observables: observable {old!r} is still "
                "referenced outside the combined likelihood; keeping its domain entry"
            )
        for domain in doc.get("domains", []):
            axes = domain.get("axes") or []
            old_entries = [ax for ax in axes if ax["name"] == old]
            if not old_entries:
                continue
            template = old_entries[0]
            axes.extend({**template, "name": n} for n in sorted(new_names, key=natural_key))
            if not still_used:
                domain["axes"] = [ax for ax in axes if ax["name"] != old]
        if not still_used:
            for pp in doc.get("parameter_points", []):
                pp["parameters"] = [
                    p for p in pp.get("parameters", []) if p["name"] != old
                ]

    return doc


def fix_split_likelihoods(doc: dict) -> dict:
    """Optional (--split-likelihoods): split the single combined likelihood into one
    likelihood per channel, and rewrite analyses to reference the new per-channel
    likelihoods.

    OFF by default: ROOT already exports one combined likelihood whose
    distributions/data pairs cover every channel (the HS3 form of the
    RooSimultaneous joint fit), and that is what pyhs3 should evaluate —
    splitting it yields N independent single-channel analyses with no joint
    structure. The split remains available for per-channel debugging only."""
    old_to_new: dict[str, list[str]] = {}
    new_likelihoods = []
    for lh in doc.get("likelihoods", []):
        data_list = lh.get("data", [])
        dist_list = lh.get("distributions", [])
        if len(data_list) >= 1 and len(data_list) == len(dist_list):
            new_names = []
            for data_name, dist_name in zip(data_list, dist_list):
                suffix = data_name.split("_", 1)[1] if "_" in data_name else data_name
                lh_name = f"L_{suffix}"
                new_likelihoods.append(
                    {
                        "name": lh_name,
                        "distributions": [dist_name],
                        "data": [data_name],
                    }
                )
                new_names.append(lh_name)
            old_to_new[lh["name"]] = new_names
        else:
            new_likelihoods.append(lh)
    doc["likelihoods"] = new_likelihoods

    new_analyses = []
    for analysis in doc.get("analyses", []):
        old_lh = analysis.get("likelihood")
        if old_lh in old_to_new:
            base = {k: v for k, v in analysis.items() if k != "likelihood"}
            for lh_name in old_to_new[old_lh]:
                new_analyses.append({**base, "name": lh_name, "likelihood": lh_name})
        else:
            new_analyses.append(analysis)
    doc["analyses"] = new_analyses

    return doc


def fix_analysis_init(doc: dict) -> dict:
    """Fix: Add "init": "default_values" to each analysis so pyhs3 knows which
    parameter set carries the const flags."""
    for analysis in doc.get("analyses", []):
        analysis.setdefault("init", "default_values")
    return doc


def fix_remove_obs_from_params(doc: dict) -> dict:
    """Fix: Remove non-constant observables (e.g. x) from default_values.
    They must not appear in parameter sets or they overwrite the data array at
    eval time. Constant parameters (global observables) are kept even if in a dataset."""
    obs_names: set[str] = set()
    for dataset in doc.get("data", []):
        for ax in dataset.get("axes") or []:
            obs_names.add(ax["name"])

    for pp in doc.get("parameter_points", []):
        if pp.get("name") == "default_values":
            pp["parameters"] = [
                p for p in pp["parameters"] if p["name"] not in obs_names or p.get("const", False)
            ]
    return doc


def fix_constraint_pdfs(doc: dict, use_aux_distributions: bool = True) -> dict:
    """Fix: Wire standalone constraint PDFs into the first likelihood (the
    combined one by default, or the first per-channel one after --split-likelihoods).
    Constraint PDFs are those not yet referenced in any likelihood whose 'x' field
    names a constant parameter (the global observable). A single-entry dataset is
    created from default_values and added alongside the constraint.
    When use_aux_distributions=True (default), constraints are placed under
    "aux_distributions"/"aux_data" so tools normalise them correctly.
    When False, constraints are placed under "distributions"/"data" (old behaviour)."""
    referenced = {d for lh in doc.get("likelihoods", []) for d in (lh.get("distributions") or [])}
    unreferenced = {d["name"] for d in doc.get("distributions", [])} - referenced

    if not unreferenced or not doc.get("likelihoods"):
        return doc

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

    if not constr_to_go:
        return doc

    # Collect unique global observables.
    go_info: dict[str, float] = {v[0]: v[1] for v in constr_to_go.values()}

    # #################################################
    # MH: I don't think this is necessary.
    #
    # # Create a single-entry global-observable dataset.
    # go_dataset = {
    #     "name": "global_obs_data",
    #     "type": "unbinned",
    #     "axes": [{"name": n, "min": v - 5.0, "max": v + 5.0}
    #              for n, v in go_info.items()],
    #     "entries": [[go_info[n] for n in go_info]],
    # }
    # doc.setdefault("data", []).append(go_dataset)

    # Attach each constraint to the first likelihood.
    first_lh = doc["likelihoods"][0]
    dist_key = "aux_distributions" if use_aux_distributions else "distributions"
    data_key = "aux_data" if use_aux_distributions else "data"
    for cname in constr_to_go:
        first_lh.setdefault(dist_key, []).append(cname)
        first_lh.setdefault(data_key, []).append("global_obs_data")

    # Populate the global-observables domain that ROOT left empty.
    for domain in doc.get("domains", []):
        if "global_observables" in domain.get("name", ""):
            domain["axes"] = [
                {"name": n, "min": v - 5.0, "max": v + 5.0} for n, v in go_info.items()
            ]

    # Remove const flag from global observables in default_values.
    # Their value is provided by global_obs_data at evaluation time;
    # const: true would prevent pyhs3 from treating them as observables.
    for pp in doc.get("parameter_points", []):
        if pp.get("name") == "default_values":
            for p in pp["parameters"]:
                if p["name"] in go_info:
                    p.pop("const", None)

    return doc


def export_workspace(
    ws: ROOT.RooWorkspace,
    stem: str,
    use_aux_distributions: bool = True,
    do_fix_exponential: bool = True,
    do_fix_null_axes: bool = True,
    do_fix_dataset_axes: bool = True,
    do_fix_unique_observables: bool = True,
    do_fix_split_likelihoods: bool = False,
    do_fix_analysis_init: bool = True,
    do_fix_remove_obs: bool = True,
    do_fix_constraints: bool = True,
) -> str:
    import json as _json

    path = f"{stem}.json" if use_aux_distributions else f"{stem}_noaux.json"
    ROOT.RooJSONFactoryWSTool(ws).exportJSON(path)
    with open(path) as fh:
        doc = _json.load(fh)
    if do_fix_exponential:
        doc = fix_exponential_functions(doc)
    if do_fix_null_axes:
        doc = fix_null_axes(doc)
    if do_fix_dataset_axes:
        doc = fix_dataset_axes(doc)
    if do_fix_unique_observables:
        doc = fix_unique_observables(doc)
    if do_fix_split_likelihoods:
        doc = fix_split_likelihoods(doc)
    if do_fix_analysis_init:
        doc = fix_analysis_init(doc)
    if do_fix_remove_obs:
        doc = fix_remove_obs_from_params(doc)
    if do_fix_constraints:
        doc = fix_constraint_pdfs(doc, use_aux_distributions=use_aux_distributions)
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
    try:
        tool.importJSON(hs3_file)
    except Exception as e:
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
        print(f"  importJSON raised an exception: {e}")
        return False
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    ok = True
    # ModelConfig is not part of HS3 and is intentionally excluded from checks.
    checks = {
        "sim_pdf (RooSimultaneous)": ws2.pdf("sim_pdf"),
        "combData (RooDataSet)": ws2.data("combData"),
        "mu_sig (POI)": ws2.var("mu_sig"),
    }
    print("  Round-trip verification:")
    for label, obj in checks.items():
        status = "OK" if obj else "MISSING"
        print(f"    {label:<35} {status}")
        if not obj:
            ok = False

    # Check event counts per channel (cap the printout for large workspaces)
    data = ws2.data("combData")
    if data:
        cat = ws2.cat("index")
        if cat:
            # ROOT 6.22+ iterates a RooCategory as (name, index) pairs; the
            # names are C++ std::string proxies, so convert before slicing.
            states = sorted(
                (str(state[0]) if not isinstance(state, str) else str(state) for state in cat),
                key=lambda s: int(s[2:]) if s[2:].isdigit() else 0,
            )
            max_print = 10
            for state in states[:max_print]:
                cat.setLabel(state)
                n = data.sumEntries(f"index=={cat.getIndex()}")
                print(f"    combData[{state}]: {int(n)} events")
            if len(states) > max_print:
                print(f"    ... {len(states) - max_print} more channels")

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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        default="simple_workspace.root",
        help="Input ROOT workspace file (default: simple_workspace.root)",
    )
    parser.add_argument(
        "--ws-name", default="combWS", help="Workspace name inside the ROOT file (default: combWS)"
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Output file stem without extension (default: input filename without .root)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Re-import the exported file and verify round-trip"
    )
    parser.add_argument(
        "--check-json",
        metavar="FILE",
        help="Load an existing HS3 JSON file into ROOT and verify it "
        "(skips export; --ws-name still applies)",
    )
    parser.add_argument(
        "--aux-constraints",
        dest="aux_constraints",
        action="store_true",
        default=True,
        help="Export constraint PDFs under aux_distributions/aux_data "
        "(default; correct HS3 normalisation)",
    )
    parser.add_argument(
        "--no-aux-constraints",
        dest="aux_constraints",
        action="store_false",
        help="Export constraint PDFs under distributions/data (old behaviour)",
    )

    # Individual fix toggles
    fix_group = parser.add_argument_group(
        "cleanup toggles",
        "Disable individual post-export fixes (all enabled by default). "
        "Use --no-cleanup to disable all at once.",
    )
    fix_group.add_argument(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="Disable all fixes; write raw ROOT HS3 output",
    )
    fix_group.add_argument(
        "--no-fix-exponential",
        dest="do_fix_exponential",
        action="store_false",
        default=True,
        help="Skip fix: remove sign-inversion intermediates for RooExponential",
    )
    fix_group.add_argument(
        "--no-fix-null-axes",
        dest="do_fix_null_axes",
        action="store_false",
        default=True,
        help="Skip fix: replace null axes in empty domains with []",
    )
    fix_group.add_argument(
        "--no-fix-dataset-axes",
        dest="do_fix_dataset_axes",
        action="store_false",
        default=True,
        help="Skip fix: add min/max to dataset axes, drop value field",
    )
    fix_group.add_argument(
        "--no-fix-unique-observables",
        dest="do_fix_unique_observables",
        action="store_false",
        default=True,
        help="Skip fix: rename the shared observable to <obs>_<channel> per "
        "channel so the combined likelihood has unique observable axis names "
        "(required by pyhs3)",
    )
    fix_group.add_argument(
        "--split-likelihoods",
        dest="do_fix_split_likelihoods",
        action="store_true",
        default=False,
        help="Split the combined likelihood into independent per-channel "
        "likelihoods/analyses (debugging only; the default keeps ROOT's "
        "single combined likelihood so pyhs3 evaluates the joint fit)",
    )
    fix_group.add_argument(
        "--no-fix-analysis-init",
        dest="do_fix_analysis_init",
        action="store_false",
        default=True,
        help="Skip fix: add init: default_values to each analysis",
    )
    fix_group.add_argument(
        "--no-fix-remove-obs",
        dest="do_fix_remove_obs",
        action="store_false",
        default=True,
        help="Skip fix: remove non-const observables from default_values",
    )
    fix_group.add_argument(
        "--no-fix-constraints",
        dest="do_fix_constraints",
        action="store_false",
        default=True,
        help="Skip fix: wire standalone constraint PDFs into the first likelihood",
    )
    args = parser.parse_args()

    # --no-cleanup disables everything
    if args.no_cleanup:
        args.do_fix_exponential = False
        args.do_fix_null_axes = False
        args.do_fix_dataset_axes = False
        args.do_fix_unique_observables = False
        args.do_fix_split_likelihoods = False
        args.do_fix_analysis_init = False
        args.do_fix_remove_obs = False
        args.do_fix_constraints = False

    if args.check_json:
        summarise(args.check_json)
        print("Verifying JSON loads into ROOT ...")
        ok = verify_roundtrip(args.check_json, args.ws_name)
        if ok:
            print("  Verification: PASS")
        else:
            print("  Verification: FAIL")
            sys.exit(1)
        sys.exit(0)

    stem = args.output_stem
    if stem is None:
        stem = args.input.removesuffix(".root")

    print(f"Loading workspace {args.ws_name!r} from {args.input!r} ...")
    ws, _f = load_workspace(args.input, args.ws_name)  # _f kept alive intentionally

    print("Exporting to HS3 (JSON) ...")
    path = export_workspace(
        ws,
        stem,
        use_aux_distributions=args.aux_constraints,
        do_fix_exponential=args.do_fix_exponential,
        do_fix_null_axes=args.do_fix_null_axes,
        do_fix_dataset_axes=args.do_fix_dataset_axes,
        do_fix_unique_observables=args.do_fix_unique_observables,
        do_fix_split_likelihoods=args.do_fix_split_likelihoods,
        do_fix_analysis_init=args.do_fix_analysis_init,
        do_fix_remove_obs=args.do_fix_remove_obs,
        do_fix_constraints=args.do_fix_constraints,
    )
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

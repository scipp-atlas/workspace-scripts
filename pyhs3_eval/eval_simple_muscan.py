#!/usr/bin/env python3
# ruff: noqa: T201
"""Evaluate the pyhs3 NLL at each scan point of a quickFit muscan.json.

Loads a workspace HS3 JSON (exported by ``export_hs3.py``), builds one pyhs3
``Model`` per matched analysis, and sums the -log(L) contributions. Prints a
comparison table against the quickFit NLL values recorded in the scan JSON
(written by ``muscan.py``, or backfilled from legacy logs by
``logs_to_muscan.py``).

The POI name is read from the scan's ``metadata.poi``, so the same script works
for toy workspaces (``mu_sig``) and real ones (e.g. ``mu_HH``). Which analyses
to evaluate is controlled by ``--analysis``, a regex fully matched against
analysis names: the default ``L_ch\\d+`` picks up the split per-channel toy
likelihoods; for a real workspace pass its combined analysis name, e.g.
``--analysis CombinedPdf_combData``.

The ``diff = pyhs3_nll - qf_nll + N*ln(C)`` column is expected to be a constant
offset across the scan (``N*ln(C)`` corrects the known RooSimultaneous
category-normalization offset; ``N`` = total event weight, ``C`` = number of
channel distributions). How far it deviates from constant (the max absolute
residual about its mean) measures how well pyhs3 reproduces quickFit.

Run from the repository root (the default paths resolve to this repo's own
``workspaces/`` and ``scans/`` directories):

    # single workspace/scan (detailed table; defaults to the 3ch base variant)
    python3 pyhs3_eval/eval_simple_muscan.py
    python3 pyhs3_eval/eval_simple_muscan.py \\
        --workspace workspaces/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \\
        --scan      scans/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x_muscan.json

    # a real workspace with one combined likelihood; cache the compiled model
    python3 pyhs3_eval/eval_simple_muscan.py \\
        --workspace path/to/bbyy.json --scan scans/bbyy_muscan.json \\
        --analysis CombinedPdf_combData --cache-dir pyhs3_eval/cache

    # compare several workspaces, each with its own scan, and rank by flatness:
    python3 pyhs3_eval/eval_simple_muscan.py \\
        --pair workspaces/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \\
               scans/3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x_muscan.json \\
        --pair workspaces/3ch_bkgGenExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \\
               scans/3ch_bkgGenExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x_muscan.json \\
        --plot-nll
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pytensor
import pytensor.tensor as pt
from plot_residuals import FIELDS
from pyhs3 import Workspace
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.scalar.basic import Add as ScalarAdd
from pytensor.scalar.basic import Mul as ScalarMul
from pytensor.tensor.elemwise import Elemwise

_HERE = Path(__file__).resolve().parent
# pyhs3_eval/ lives directly under the workspace-scripts repo root, so the
# workspaces/ and scans/ output directories are one level up.
_WS_SCRIPTS = _HERE.parent
_DEFAULT_STEM = "3ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x"
_DEFAULT_WS = _WS_SCRIPTS / "workspaces" / f"{_DEFAULT_STEM}.json"
_DEFAULT_SCAN = _WS_SCRIPTS / "scans" / f"{_DEFAULT_STEM}_muscan.json"
_NLL_OUTPUT_DEFAULT = _WS_SCRIPTS / f"nll_output.json"

# Sentinel for a bare ``--plot-*`` flag: the output path is then the default
# derived below rather than a user-supplied one.
_PLOT_DEFAULT = object()

# Bumped whenever the baking recipe in ``compile_channel`` changes, so cached
# compiled channels from an older (incompatible) recipe are not reused. v2:
# non-free parameters bake at their JSON default_values, not 0.0. v3/v4:
# NO_REWRITES linker changes (v2 py-linker caches crash at evaluation on
# >32-input Add nodes; v4 splits those nodes instead — see
# ``_split_wide_elemwise`` — because v3's cvm linker spends hours computing
# per-node C-module cache keys on million-node graphs).
_CACHE_SCHEMA = "v4"


def _resolve_mode(mode: str | None):
    """Map the --pytensor-mode string to what pytensor.function accepts.

    ``NO_REWRITES`` skips the graph-rewrite phase entirely (pure-Python linker,
    no optimizer). On large real workspaces FAST_RUN's rewriter — in particular
    constant folding, which C-compiles a thunk per foldable node — can pin the
    CPU for hours; with rewrites off, compilation cost drops to graph traversal
    and each of the ~dozens of scan-point evaluations just walks the
    unoptimized graph in Python. The cvm linker is no alternative at this
    scale: it computes a C-module cache key per node, which on a million-node
    graph is itself hours of silent Python. The py linker needs
    ``_split_wide_elemwise`` first (numpy's 32-operand ufunc limit).
    """
    if mode == "NO_REWRITES":
        from pytensor.compile.mode import Mode

        return Mode(linker="py", optimizer=None)
    return mode


def _split_wide_elemwise(expr, max_args: int = 16):
    """Chunk variadic Add/Mul nodes so none exceeds *max_args* inputs.

    pyhs3 emits e.g. ``pt.add(*terms)`` with hundreds of operands on real
    workspaces (``model.log_prob`` sums every channel and constraint term in
    one node). numpy ufuncs accept at most 32 operands, and the pure-Python
    linker's ``Elemwise.perform`` relies on one existing — evaluation then
    dies with "'Scratchpad' object has no attribute 'ufunc'". Rewrites would
    normally fuse these away, but NO_REWRITES exists precisely to skip them.

    Splitting a flat sum/product into balanced chunks is exact up to float
    summation order. Returns the (cloned) split expression; collect the
    function inputs from the returned graph, not the original.
    """
    fg = FunctionGraph(outputs=[expr], clone=True)
    n_split = 0
    for node in list(fg.toposort()):
        if node not in fg.apply_nodes:
            continue
        op = node.op
        if not (isinstance(op, Elemwise) and isinstance(op.scalar_op, (ScalarAdd, ScalarMul))):
            continue
        if len(node.inputs) <= max_args:
            continue
        variadic = pt.add if isinstance(op.scalar_op, ScalarAdd) else pt.mul
        terms = list(node.inputs)
        while len(terms) > max_args:
            terms = [
                variadic(*chunk) if len(chunk) > 1 else chunk[0]
                for i in range(0, len(terms), max_args)
                for chunk in [terms[i : i + max_args]]
            ]
        fg.replace(node.outputs[0], variadic(*terms) if len(terms) > 1 else terms[0])
        n_split += 1
    if n_split:
        print(f"  split {n_split} wide Add/Mul node(s) for the python linker", flush=True)
    return fg.outputs[0]


def _default_nll_plot_path() -> Path:
    """Default PDF path for the combined NLL-curve plot (next to this script)."""
    return _HERE / "all_nll_comparisons.pdf"


def _default_resid_plot_path() -> Path:
    """Default PDF path for the residual/offset plot (next to this script)."""
    return _HERE / "residual_comparison.pdf"


def category_offset(ws_json: dict, analysis_names: list[str]) -> tuple[float, int, float]:
    """RooSimultaneous category-normalization offset for the matched analyses.

    Returns ``(offset, n_channels, n_events)`` where ``n_channels`` is the total
    number of channel distributions across the matched likelihoods, ``n_events``
    is the total event weight of their datasets (sum of weights when present,
    raw entry count otherwise; aux/constraint data excluded), and
    ``offset = n_events * ln(n_channels)``.
    """
    analyses = {a["name"]: a for a in ws_json["analyses"]}
    likelihoods = {lk["name"]: lk for lk in ws_json["likelihoods"]}
    data = {d["name"]: d for d in ws_json["data"]}

    n_channels = 0
    n_events = 0.0
    for name in analysis_names:
        lik_name = analyses[name].get("likelihood", name)
        lik = likelihoods[lik_name]
        n_channels += len(lik["distributions"])
        for dname in lik["data"]:
            d = data[dname]
            weights = d.get("weights")
            n_events += sum(weights) if weights is not None else len(d["entries"])

    offset = n_events * np.log(n_channels) if n_channels > 0 else 0.0
    return float(offset), n_channels, n_events


def nominal_from_json(ws_json: dict) -> dict[str, float]:
    """Nominal parameter values read straight from the workspace JSON.

    ``model.nominal_params`` is empty for real-workspace analyses that reference
    only ``domains`` (no ``parameter_point``) — e.g. the bbyy ``combData``
    analysis. Baking every non-free parameter to ``0.0`` in that case zeroes
    yields, efficiencies and normalizations, which sends the NLL to ``inf``
    (constant folding then hits a divide-by-zero on the zeroed denominators).

    So read the authoritative defaults straight from ``parameter_points``,
    preferring the ``default_values`` set (the full export RooFit writes) and
    filling any gaps from the remaining sets. The ``const`` flag is ignored
    here: this is only the value used when a parameter is *not* kept symbolic.
    """
    psets = {s["name"]: s for s in ws_json.get("parameter_points", [])}
    order = ["default_values", *(n for n in psets if n != "default_values")]
    nominal: dict[str, float] = {}
    for name in order:
        for p in psets.get(name, {}).get("parameters", []):
            pname, val = p.get("name"), p.get("value")
            if pname is not None and val is not None and pname not in nominal:
                nominal[pname] = float(val)
    return nominal


def compile_channel(
    ws: Workspace,
    analysis,
    mode: str | None,
    extra_free: set[str] | None = None,
    json_nominal: dict[str, float] | None = None,
) -> dict:
    """Build and compile the NLL for one analysis, baking non-free inputs.

    Data arrays and every parameter outside the free set are replaced with
    ``pt.constant`` before compilation (the recipe from the pyhs3 bbyy
    validation scripts): constant folding then collapses most of the graph, so
    pytensor's rewrite phase stays tractable on large real workspaces.

    The free set is ``model.free_params`` plus *extra_free* (the quickFit
    floating set from the scan JSON). The union matters: some real-workspace
    exports mark every parameter const, leaving ``model.free_params`` empty —
    without *extra_free* the whole graph would fold to a constant.

    Non-free parameters are baked at their nominal value. That value comes from
    ``model.nominal_params`` when populated, else from *json_nominal* (the
    workspace JSON defaults; see :func:`nominal_from_json`) — because
    ``model.nominal_params`` is empty for analyses that reference no
    parameter_point, and baking those parameters to ``0.0`` sends the NLL to
    ``inf``.
    """
    t0 = time.perf_counter()
    model = ws.model(analysis, progress=False)
    nll_expr = -model.log_prob
    print(f"  {analysis.name}: model built in {time.perf_counter() - t0:.1f}s", flush=True)
    data_np = {k: np.asarray(v, dtype=np.float64) for k, v in model.data.items()}
    free_names = set(model.free_params) | (extra_free or set())
    # JSON defaults are the base; model.nominal_params (when present) wins.
    nominal = {**(json_nominal or {}), **model.nominal_params}

    all_inputs = [v for v in explicit_graph_inputs([nll_expr]) if v.name is not None]
    print(
        f"  {analysis.name}: {len(all_inputs)} graph inputs; "
        f"model.free_params = {len(model.free_params)}, requested free = {len(free_names)}"
    )

    subs: dict = {}
    free_vars: dict[str, pt.TensorVariable] = {}
    baked_missing: list[str] = []
    for var in all_inputs:
        if var.name in data_np:
            subs[var] = pt.constant(data_np[var.name])
        elif var.name in free_names:
            if var.name not in free_vars:
                free_vars[var.name] = pt.scalar(var.name)
            subs[var] = free_vars[var.name]
        else:
            if var.name not in nominal:
                baked_missing.append(var.name)
            subs[var] = pt.constant(np.float64(nominal.get(var.name, 0.0)))

    # A non-free parameter with no nominal value anywhere is baked to 0.0, which
    # commonly zeroes a yield/normalization and makes the NLL inf. This should
    # not happen once json_nominal covers the workspace, so surface it loudly.
    if baked_missing:
        print(
            f"  WARNING: {len(baked_missing)} non-free parameter(s) had no nominal "
            f"value (baked to 0.0 -- this can make the NLL inf): "
            f"{sorted(baked_missing)[:8]}{' ...' if len(baked_missing) > 8 else ''}"
        )

    t0 = time.perf_counter()
    nll_baked = clone_replace(nll_expr, replace=subs)
    if mode == "NO_REWRITES":
        nll_baked = _split_wide_elemwise(nll_baked)
    free_inputs = [v for v in explicit_graph_inputs([nll_baked]) if v.name is not None]
    input_names = [v.name for v in free_inputs]
    print(
        f"  {analysis.name}: baked in {time.perf_counter() - t0:.1f}s; "
        f"compiling (mode={mode or 'default'}) — the slow phase on big workspaces ...",
        flush=True,
    )
    t0 = time.perf_counter()
    fn = pytensor.function(
        free_inputs, nll_baked, mode=_resolve_mode(mode) or model.mode, on_unused_input="ignore"
    )
    print(
        f"  {analysis.name}: compiled in {time.perf_counter() - t0:.1f}s, "
        f"{len(input_names)} free inputs"
    )

    # A requested-free parameter that is not a compiled input was already baked
    # during pyhs3 model construction (const in the workspace JSON) or does not
    # feed this channel. Setting it per scan point will have NO effect, so an
    # affected POI means a flat scan — surface this loudly.
    missing = sorted((extra_free or set()) - set(input_names))
    if missing:
        print(
            f"  WARNING: {len(missing)} scan parameter(s) are not free in the "
            f"compiled graph (const in the workspace JSON, or not part of this "
            f"channel): {missing[:8]}{' ...' if len(missing) > 8 else ''}"
        )
    if not input_names:
        print(
            "  WARNING: compiled NLL has NO free inputs — it is a constant and "
            "the scan comparison will be meaningless. Check the const flags in "
            "the workspace JSON parameter_points."
        )
    return {
        "name": analysis.name,
        "fn": fn,
        "input_names": input_names,
        "data": data_np,
        "free_params": dict(model.free_params),
    }


def build_channel_models(
    ws_path: Path,
    analysis_pattern: str = r"L_ch\d+",
    cache_dir: Path | None = None,
    mode: str | None = None,
    extra_free: set[str] | None = None,
) -> tuple[list[dict], float]:
    """Load workspace, compile one NLL function per matched analysis.

    Analyses whose name fully matches *analysis_pattern* are compiled (with
    data and non-free parameters baked as constants; see ``compile_channel``).
    Returns ``(channels, offset)`` where each channel is a dict with the
    compiled ``fn``, its ``input_names``, and the model's ``data`` arrays and
    ``free_params``; ``offset`` is the ``N*ln(C)`` category-normalization term.

    With *cache_dir* set, each compiled channel is pickled individually right
    after it compiles, and reloaded on later runs — so a crash or interrupt
    never loses already-compiled channels (important for large real workspaces
    where one compile can take an hour). Delete the cache after changing pyhs3
    versions. *mode* overrides the pytensor compilation mode (e.g.
    ``FAST_COMPILE``) and is part of the cache key.
    """
    print(f"Loading workspace from {ws_path} ...")
    with ws_path.open() as fh:
        ws_json = json.load(fh)

    # Authoritative nominal values for baking non-free parameters, read from the
    # JSON (model.nominal_params is empty for analyses that reference only
    # domains). Extracted before the const-clearing below, which only touches
    # flags, not values.
    json_nominal = nominal_from_json(ws_json)

    # pyhs3 bakes const=true parameters as pt.constant during model
    # construction — no symbolic input then exists, and no compile-time choice
    # can resurrect it. Real-workspace exports sometimes carry const flags on
    # the very parameters quickFit floated (a snapshot artifact), so clear the
    # flag for the scan's floating set before building the models.
    if extra_free:
        n_uncost = 0
        for pset in ws_json.get("parameter_points", []):
            for p in pset.get("parameters", []):
                if p.get("name") in extra_free and p.get("const"):
                    p["const"] = False
                    n_uncost += 1
        if n_uncost:
            print(f"  Cleared const flag on {n_uncost} scan-parameter entries in parameter_points")

    ws = Workspace(**ws_json)
    matched = sorted(
        (a for a in ws.analyses if re.fullmatch(analysis_pattern, a.name)),
        key=lambda a: a.name,
    )
    if not matched:
        available = sorted(a.name for a in ws.analyses)
        msg = f"No analysis matches --analysis '{analysis_pattern}'; available: {available}"
        raise SystemExit(msg)
    print(f"  Matched analyses: {[a.name for a in matched]}")

    mode_tag = f"__{mode}" if mode else ""
    # The free set is part of the compiled function, so it must be part of the
    # cache key — otherwise a cache built against one scan (or against the old
    # free_params-only behaviour) would be silently reused with another.
    free_tag = ""
    if extra_free:
        digest = hashlib.md5(",".join(sorted(extra_free)).encode()).hexdigest()[:8]
        free_tag = f"__f{digest}"

    def channel_cache_file(analysis_name: str) -> Path | None:
        if cache_dir is None:
            return None
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", analysis_name)
        # _CACHE_SCHEMA guards against silently reusing caches whose baked
        # constants were produced by an older, buggy baking recipe.
        return cache_dir / f"{ws_path.stem}__{slug}{mode_tag}{free_tag}__{_CACHE_SCHEMA}.pkl"

    channels: list[dict] = []
    for analysis in matched:
        cache_file = channel_cache_file(analysis.name)
        if cache_file is not None and cache_file.exists():
            print(f"  {analysis.name}: loading from cache {cache_file}", flush=True)
            # Unpickling re-runs the linker: cheap for the py linker
            # (NO_REWRITES), but with the default C/cvm linker it recomputes
            # per-node C-module cache keys and can rival the original compile.
            t0 = time.perf_counter()
            with cache_file.open("rb") as fh:
                channels.append(pickle.load(fh))
            print(f"  {analysis.name}: cache loaded in {time.perf_counter() - t0:.1f}s", flush=True)
            continue

        if cache_file is not None:
            print(f"  {analysis.name}: no cache at {cache_file}; compiling", flush=True)
        channel = compile_channel(ws, analysis, mode, extra_free, json_nominal)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            with cache_file.open("wb") as fh:
                pickle.dump(channel, fh)
            print(
                f"  {analysis.name}: cached to {cache_file} "
                f"in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
        channels.append(channel)

    offset, n_channels, n_events = category_offset(ws_json, [a.name for a in matched])
    print(f"  Offset N*ln(C): N = {n_events:.6g}, C = {n_channels} -> {offset:.6f}")

    return channels, offset


def eval_nll(channels: list[dict], params: dict[str, float]) -> float:
    """Sum -log(L) over all channels at *params*."""
    total = 0.0
    for ch in channels:
        # data arrays (e.g. x) take priority over any scalar fallback in params
        source = {**params, **ch["data"]}
        args = [
            np.asarray(source[n], dtype=np.float64) if n in source else np.float64(0.0)
            for n in ch["input_names"]
        ]
        total += float(np.asarray(ch["fn"](*args)).item())
    return total


def run_scan(
    ws_path: Path,
    scan_path: Path,
    *,
    analysis_pattern: str = r"L_ch\d+",
    cache_dir: Path | None = None,
    mode: str | None = None,
    verbose: bool = True,
) -> dict:
    """Evaluate one workspace against one scan and summarize the diff.

    Returns a dict with the per-point diffs and the constant-offset statistics:
    ``mean_offset`` (the average of the diff column) and ``max_abs_resid``
    (the largest absolute deviation of any point from that mean -- i.e. how
    far the diff strays from being a perfect constant).
    """
    if verbose:
        print(f"Loading scan points from {scan_path} ...")
    with scan_path.open() as fh:
        scan = json.load(fh)

    poi = scan["metadata"].get("poi", "mu_sig")
    qf_nll_min = scan["metadata"]["nll_min"]
    bkg_type = scan["metadata"].get("bkg_type", "")
    points = scan["scan_points"]

    # The quickFit floating set — the POI plus every parameter recorded at any
    # scan point — must stay symbolic in the compiled NLL even if the workspace
    # JSON marks it const (some real-workspace exports do; model.free_params is
    # then empty and blind baking would fold the whole graph to a constant).
    scan_free = {poi}
    for pt_data in points:
        scan_free.update(pt_data["parameters"])

    channels, offset = build_channel_models(
        ws_path, analysis_pattern, cache_dir, mode, extra_free=scan_free
    )

    # Collect nominal free params from all channels (shared params are consistent)
    nominal: dict[str, float] = {}
    for ch in channels:
        nominal.update(ch["free_params"])

    if verbose:
        print(f"  {len(points)} scan points, POI = {poi}, quickFit NLL_min = {qf_nll_min:.6f}\n")
        header = f"{poi:>8}  {'qf_nll':>14}  {'pyhs3_nll':>14}  {'diff':>16}"
        print(header)
        print("-" * len(header))

    mus: list[float] = []
    qf_nlls: list[float] = []
    pyhs3_nlls: list[float] = []
    diffs: list[float] = []
    for pt_data in sorted(points, key=lambda p: p[poi]):
        mu = pt_data[poi]
        qf_nll = pt_data["nll"]
        if qf_nll is None:
            if verbose:
                print(f"{mu:>8.3f}  {'(failed fit -- skipped)':>14}")
            continue

        params: dict[str, float] = dict(nominal)
        params[poi] = mu
        for name, info in pt_data["parameters"].items():
            val = info["value"]
            # muscan.json uses ROOT's negative-tau convention for exponential_dist;
            # HS3/pyhs3 uses positive c — only flip for exponential backgrounds
            if bkg_type == "exponential" and name.startswith("tau_"):
                val = -val
            params[name] = val

        pyhs3_nll = eval_nll(channels, params)
        diff = pyhs3_nll - qf_nll + offset
        mus.append(mu)
        qf_nlls.append(qf_nll)
        pyhs3_nlls.append(pyhs3_nll)
        diffs.append(diff)

        if verbose:
            print(f"{mu:>8.3f}  {qf_nll:>14.6f}  {pyhs3_nll:>14.6f}  {diff:>10.10f}")

    diff_arr = np.asarray(diffs, dtype=np.float64)
    mean_offset = float(diff_arr.mean())
    resid = diff_arr - mean_offset
    max_abs_resid = float(resid[np.abs(resid).argmax()])

    if verbose:
        pyhs3_nll_min = min(pyhs3_nlls)
        print(f"\npyhs3 NLL_min    = {pyhs3_nll_min:.6f}")
        print(f"quickFit NLL_min = {qf_nll_min:.6f}")
        print(f"Difference       = {pyhs3_nll_min - qf_nll_min:.6f}")
        print(f"\nmean offset      = {mean_offset:.6f}")
        print(f"max |residual|   = {max_abs_resid:.3e}  (deviation of diff from constant)")

    return {
        "workspace": ws_path,
        "scan": scan_path,
        "mus": mus,
        "diffs": diffs,
        "mean_offset": mean_offset,
        "max_abs_resid": max_abs_resid,
        "qf_nlls": qf_nlls,
        "pyhs3_nlls": pyhs3_nlls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workspace", type=Path, default=_DEFAULT_WS)
    parser.add_argument("--scan", type=Path, default=_DEFAULT_SCAN)
    parser.add_argument(
        "--analysis",
        default=r"L_ch\d+",
        metavar="REGEX",
        help="Regex fully matched against analysis names to select which "
        "likelihoods to evaluate (default: the split per-channel toy "
        "analyses 'L_ch\\d+'; for a real workspace pass its combined "
        "analysis name, e.g. 'CombinedPdf_combData')",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Cache each compiled channel model here, written right after it "
        "compiles (recommended for large real workspaces; delete the "
        "cache after changing pyhs3 versions)",
    )
    parser.add_argument(
        "--pytensor-mode",
        default=None,
        metavar="MODE",
        help="Override the pytensor compilation mode (default: the model's "
        "mode, FAST_RUN). NO_REWRITES skips the graph-rewrite phase "
        "entirely (pure-Python evaluation) — use this for large real "
        "workspaces where FAST_RUN/FAST_COMPILE pin the CPU for hours "
        "during rewriting. Part of the cache key.",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("WORKSPACE", "SCAN"),
        type=Path,
        help="A workspace/scan pair to evaluate. Repeat to compare several; "
        "when given, --workspace/--scan are ignored and a ranked "
        "constant-offset summary is printed.",
    )
    parser.add_argument(
        "--plot-nll",
        nargs="?",
        const=_PLOT_DEFAULT,
        default=None,
        type=Path,
        metavar="PATH",
        help="Plot the pyhs3 vs quickFit NLL curves into a single PDF, one "
        "panel per workspace. Pass a path to override the default "
        "(all_nll_comparisons.pdf next to this script).",
    )
    parser.add_argument(
        "--output-nll-json",
        nargs="?",
        const=_NLL_OUTPUT_DEFAULT,
        default=None,
        metavar="PATH",
        help="enable outputting resulting nll plot to a .json file at specified location, default: nll-output.json",
    )   
    parser.add_argument(
        "--plot-resid",
        nargs="?",
        const=_PLOT_DEFAULT,
        default=None,
        type=Path,
        metavar="PATH",
        help="Plot curves of the residuals and mean offsets of all evaluated "
        "workspaces (default: residual_comparison.pdf next to this script).",
    )
    parser.add_argument(
        "--plot-resid-field",
        nargs="+",
        default=["channels"],
        choices=FIELDS,
        metavar="FIELD",
        help="workspace-name field(s) to use for the residual-plot x labels",
    )
    parser.add_argument(
        "--plot-resid-log-x",
        action="store_true",
        help="enables log scale for the x-axis in a residual plot",
    )
    parser.add_argument(
        "--plot-resid-log-y",
        action="store_true",
        help="enables log scale for the y-axis in a residual plot",
    )
    args = parser.parse_args()

    # Gather the workspace/scan pairs to evaluate: either the repeated --pair
    # arguments, or the single --workspace/--scan pair.
    if args.pair:
        pairs = args.pair
        summarize = True
    else:
        pairs = [(args.workspace, args.scan)]
        summarize = False

    results = []
    for ws_path, scan_path in pairs:
        if summarize:
            print("=" * 72)
            print(f"WORKSPACE: {ws_path}")
            print(f"SCAN:      {scan_path}")
            print("=" * 72)
        results.append(
            run_scan(
                ws_path,
                scan_path,
                analysis_pattern=args.analysis,
                cache_dir=args.cache_dir,
                mode=args.pytensor_mode,
                verbose=True,
            )
        )

    if args.output_nll_json:
        with open(args.output_nll_json, "w") as f:
            json.dump(results, f, default=str, indent=2)
        
    if args.plot_nll is not None:
        from plot_muscan_nll import plot_nll_curves  # noqa: PLC0415

        out = _default_nll_plot_path() if args.plot_nll is _PLOT_DEFAULT else args.plot_nll
        plot_nll_curves(results, out)

    if args.plot_resid is not None:
        from plot_residuals import plot_residual_and_offset  # noqa: PLC0415

        out = _default_resid_plot_path() if args.plot_resid is _PLOT_DEFAULT else args.plot_resid
        plot_residual_and_offset(
            results,
            out,
            label_field=args.plot_resid_field,
            log_x=args.plot_resid_log_x,
            log_y=args.plot_resid_log_y,
        )

    if not summarize:
        return

    # Rank by how flat the diff is: smaller max |residual| == closer to constant.
    results.sort(key=lambda r: abs(r["max_abs_resid"]))

    name_width = max(len("workspace"), *(len(r["workspace"].name) for r in results))
    off_width, res_width = 14, 11

    header = (
        f"{'workspace':<{name_width}}  {'mean offset':>{off_width}}  {'max |resid|':>{res_width}}"
    )
    print("\n" + "=" * len(header))
    print("CONSTANT-OFFSET SUMMARY  (sorted: flattest diff first)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        name = r["workspace"].name
        print(
            f"{name:<{name_width}}  {r['mean_offset']:>{off_width}.3e}  "
            f"{r['max_abs_resid']:>{res_width}.3e}"
        )


if __name__ == "__main__":
    main()

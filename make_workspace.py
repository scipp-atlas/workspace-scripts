#!/usr/bin/env python3
"""
Generate a simple multi-channel RooFit workspace for quickFit testing.

The number of channels is configurable via --num-channels (default 3, no upper
limit; the first 30 use the hardcoded CHANNELS table, beyond that a deterministic
per-index formula). The examples below use the default three channels ch0, ch1, ch2.

Model (per channel, unbinned, x in [10, 20]):
  Signal   : Gaussian(x, mean=15, sigma_ch), normalization = mu_sig * nsig_ch
           or double-sided Crystal Ball (--sig-form dscb; RooCrystalBall with
              fixed tail parameters, same mean/sigma_ch core)
  Background: RooExponential(x, tau_ch)  [default]
           or RooGenericPdf ~ exp(tau_ch*x)  [--generic-bkg; same shape, generic_dist in HS3]
              normalization = nbkg_ch (free)

Systematic uncertainty (with --np, the default):
  alpha_sigma  : NP varying signal width, Gaussian-constrained to N(0,1)
                 sigma_ch = sigma_nom_ch * (1 + SIGMA_DELTA * alpha_sigma)
                 Shared across all channels (correlated).

Yield systematics (--num-systs M, default 0):
  alpha_syst<j> (j = 0..M-1): shared unit-Gaussian-constrained NPs, each scaling
  every channel's signal yield through a per-channel response factor
  resp_syst<j>_<ch> = 1 + delta(j, ch) * alpha_syst<j>, delta in 3-7%.
  Constraints are named constr_alpha_syst<j> (auto-detected downstream).
  Independent of the width NP flags; alphas sit at 0 during toy generation, so
  datasets are identical to the systs-less workspace for a given seed.

POI  : mu_sig       (signal strength, floated in fit)
NPs  : tau_ch{0,1,2}, nbkg_ch{0,1,2}  (unconstrained)
       alpha_sigma                     (Gaussian constrained, --np only)
       alpha_syst{0..M-1}              (Gaussian constrained, --num-systs M)
Fixed: mean_ch*, sigma_nom_ch*, nsig_ch*

Expected yield per channel at mu_sig=1: ~30 events (7 signal + 23 background)

Usage:
    python3 make_workspace.py [--no-np] [--generic-bkg] [--output NAME.root] [--seed 42]

Fit with quickFit (NP version):
    quickFit -f simple_workspace.root -w combWS -m ModelConfig -d combData \\
             -p mu_sig=1_-5_10 --minos 1 --externalConstraint constr_alpha_sigma

Fit with quickFit (no-NP version):
    quickFit -f simple_workspace_nonp.root -w combWS -m ModelConfig -d combData \\
             -p mu_sig=1_-5_10 --minos 1
"""

import argparse

import ROOT
from itertools import islice

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)


# ─── Channel configuration ──────────────────────────────────────────────────
# tau  : exponential decay constant for exp(tau*x), negative for falling spectrum
# nsig : expected signal events at mu_sig = 1  (CONSTANT in the fit)
# nbkg : initial background yield estimate     (FLOATING in the fit)
# sigma: nominal Gaussian width of the signal peak
CHANNELS = {
    "ch0": {"tau": -0.30, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.00},
    "ch1": {"tau": -0.25, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.90},
    "ch2": {"tau": -0.35, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.10},
    "ch3": {"tau": -0.28, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.95},
    "ch4": {"tau": -0.32, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.05},
    "ch5": {"tau": -0.27, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.85},
    "ch6": {"tau": -0.33, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.15},
    "ch7": {"tau": -0.31, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.00},
    "ch8": {"tau": -0.26, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.92},
    "ch9": {"tau": -0.34, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.08},
    "ch10": {"tau": -0.29, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.97},
    "ch11": {"tau": -0.30, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.03},
    "ch12": {"tau": -0.24, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.88},
    "ch13": {"tau": -0.36, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.12},
    "ch14": {"tau": -0.28, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.94},
    "ch15": {"tau": -0.32, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.06},
    "ch16": {"tau": -0.27, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.91},
    "ch17": {"tau": -0.33, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.09},
    "ch18": {"tau": -0.31, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.99},
    "ch19": {"tau": -0.25, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.01},
    "ch20": {"tau": -0.35, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.87},
    "ch21": {"tau": -0.29, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.13},
    "ch22": {"tau": -0.30, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.96},
    "ch23": {"tau": -0.26, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.04},
    "ch24": {"tau": -0.34, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.89},
    "ch25": {"tau": -0.28, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.11},
    "ch26": {"tau": -0.32, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.93},
    "ch27": {"tau": -0.27, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.07},
    "ch28": {"tau": -0.33, "nsig": 7.0, "nbkg": 23.0, "sigma": 0.98},
    "ch29": {"tau": -0.31, "nsig": 7.0, "nbkg": 23.0, "sigma": 1.02},
}


def _channel_cfg(i: int) -> dict:
    """Deterministic config for channel index i >= 30 (ch0-ch29 stay hardcoded).

    Cycles tau through [-0.36, -0.24] and sigma through [0.85, 1.15] using
    coprime strides so consecutive channels differ, mirroring the spread of
    the hardcoded table. Independent of --seed by construction.
    """
    tau = -0.24 - 0.01 * ((7 * i) % 13)
    sigma = 0.85 + 0.01 * ((11 * i) % 31)
    return {"tau": round(tau, 2), "nsig": 7.0, "nbkg": 23.0, "sigma": round(sigma, 2)}


def get_channels(n: int) -> dict:
    """First n channel configs: hardcoded CHANNELS for i < 30, formula beyond."""
    if n < 1:
        raise SystemExit(f"--num-channels must be >= 1 (got {n})")
    out = dict(islice(CHANNELS.items(), min(n, len(CHANNELS))))
    for i in range(len(out), n):
        out[f"ch{i}"] = _channel_cfg(i)
    return out

# Relative uncertainty on signal width applied by alpha_sigma (±1σ → ±10%)
SIGMA_DELTA = 0.10

# Base relative yield effect of the shared systematics (--num-systs)
SYST_DELTA_BASE = 0.03


def _syst_delta(j: int, i: int) -> float:
    """Relative yield effect of alpha_syst<j> on channel i: 3-7%, varying with
    both indices so no syst is degenerate with mu_sig."""
    return SYST_DELTA_BASE + 0.01 * ((j + 2 * i) % 5)

POLY_SLOPE_INIT = -0.02
POLY_SLOPE_LO = -0.049
POLY_SLOPE_HI = 0.049

# Double-sided crystal ball tail parameters (fixed; --sig-form dscb).
# Mildly asymmetric so a left/right swap in an export round-trip is detectable.
DSCB_ALPHA_L = 1.5
DSCB_N_L = 5.0
DSCB_ALPHA_R = 2.0
DSCB_N_R = 3.0


def build_background(ch, cfg, x, *, generic_bkg, bkg_form, fix_shape, keep):
    """Return (shape_var, bkg_pdf).

    shape_var is named tau_<ch> for all forms so downstream scripts
    (muscan/export/snapshot) find it unchanged.
    """
    if generic_bkg and bkg_form == "poly":
        init, lo, hi = POLY_SLOPE_INIT, POLY_SLOPE_LO, POLY_SLOPE_HI
    else:
        init, lo, hi = cfg["tau"], -2.0, -0.001

    tau = ROOT.RooRealVar(f"tau_{ch}", f"bkg shape ({ch})", init, lo, hi)
    if fix_shape:
        tau.setConstant(True)

    if generic_bkg:
        expr = "1.0 + @1*@0" if bkg_form == "poly" else "exp(@1*@0)"
        bkg = ROOT.RooGenericPdf(
            f"bkg_{ch}", f"background pdf ({ch})", expr, ROOT.RooArgList(x, tau)
        )
    else:
        bkg = ROOT.RooExponential(f"bkg_{ch}", f"background pdf ({ch})", x, tau)

    keep[f"tau_{ch}"] = tau
    keep[f"bkg_{ch}"] = bkg
    return tau, bkg


def build_signal(ch, x, mean, sigma, *, generic_sig, sig_form, keep):
    """Return the signal pdf.

    RooGaussian by default, RooGenericPdf with --generic-sig, or a
    double-sided RooCrystalBall with --sig-form dscb (fixed tail
    parameters, shared symmetric width so the sigma NP still applies).
    The pdf is named sig_<ch> in every case (downstream contract).
    """
    if sig_form == "dscb":
        alpha_l = ROOT.RooRealVar(f"alphaL_{ch}", f"DSCB left tail alpha ({ch})", DSCB_ALPHA_L)
        n_l = ROOT.RooRealVar(f"nL_{ch}", f"DSCB left tail n ({ch})", DSCB_N_L)
        alpha_r = ROOT.RooRealVar(f"alphaR_{ch}", f"DSCB right tail alpha ({ch})", DSCB_ALPHA_R)
        n_r = ROOT.RooRealVar(f"nR_{ch}", f"DSCB right tail n ({ch})", DSCB_N_R)
        for v in (alpha_l, n_l, alpha_r, n_r):
            v.setConstant(True)
            keep[v.GetName()] = v
        sig = ROOT.RooCrystalBall(
            f"sig_{ch}", f"signal pdf ({ch})", x, mean, sigma, alpha_l, n_l, alpha_r, n_r
        )
    elif generic_sig:
        sig = ROOT.RooGenericPdf(
            f"sig_{ch}",
            f"signal pdf ({ch})",
            "exp(-0.5*((@0-@1)/@2)**2)",
            ROOT.RooArgList(x, mean, sigma),
        )
    else:
        sig = ROOT.RooGaussian(f"sig_{ch}", f"signal pdf ({ch})", x, mean, sigma)
    keep[f"sig_{ch}"] = sig
    return sig


def build_width_np(constraint, keep):
    """Build the shared signal-width NP and (optionally) its constraint pdf.

    Returns dict with:
    np        : the floating NP (alpha_sigma or gamma_sigma)
    constr    : the constraint pdf (or None)
    global_ob : the global observable RooRealVar (or None)
    kind      : 'add' (sigma = sigma_nom*(1+delta*alpha)) or
                'mul' (sigma = sigma_nom*gamma)
    """
    if constraint == "poisson":
        gamma = ROOT.RooRealVar("gamma_sigma", "signal width scale NP", 1.0, 0.01, 5.0)
        tau_g = 1.0 / (SIGMA_DELTA**2)
        glob = ROOT.RooRealVar("nom_gamma_sigma", "global obs: poisson count", tau_g)
        tau_c = ROOT.RooRealVar("tau_gamma_sigma", "poisson tau", tau_g)
        glob.setConstant(True)
        tau_c.setConstant(True)
        mean_p = ROOT.RooProduct("mean_gamma_sigma", "tau*gamma", ROOT.RooArgList(tau_c, gamma))
        constr = ROOT.RooPoisson(
            "constr_gamma_sigma", "Poisson constraint on gamma_sigma", glob, mean_p
        )
        constr.setNoRounding(True)
        keep.update(
            {
                "gamma_sigma": gamma,
                "nom_gamma_sigma": glob,
                "tau_gamma_sigma": tau_c,
                "mean_gamma_sigma": mean_p,
                "constr_gamma_sigma": constr,
            }
        )
        return {"np": gamma, "constr": constr, "global_ob": glob, "kind": "mul"}

    # gauss / none -> additive alpha
    alpha = ROOT.RooRealVar("alpha_sigma", "signal width NP", 0.0, -5.0, 5.0)
    keep["alpha_sigma"] = alpha
    if constraint == "gauss":
        glob = ROOT.RooRealVar("nom_alpha_sigma", "global obs: signal width", 0.0)
        sig_c = ROOT.RooRealVar("sigma_constr", "constraint Gaussian sigma", 1.0)
        glob.setConstant(True)
        sig_c.setConstant(True)
        constr = ROOT.RooGaussian(
            "constr_alpha_sigma", "Gaussian constraint on alpha_sigma", glob, alpha, sig_c
        )
        keep.update({"nom_alpha_sigma": glob, "sigma_constr": sig_c, "constr_alpha_sigma": constr})
        return {"np": alpha, "constr": constr, "global_ob": glob, "kind": "add"}

    # constraint == "none": free NP, no aux term
    return {"np": alpha, "constr": None, "global_ob": None, "kind": "add"}


def build_syst_nps(num_systs, keep):
    """Build the shared yield-systematic NPs (--num-systs), unit-Gaussian constrained.

    Returns a list of dicts {"np", "constr", "global_ob"}, one per systematic.
    Constraints are named constr_alpha_syst<j> so muscan.py / run_simple_fit.sh
    auto-detect them; they are imported standalone (never in a RooProdPdf with
    the RooSimultaneous — ROOT 6.30+ contract).
    """
    infos = []
    if num_systs <= 0:
        return infos
    unit = ROOT.RooRealVar("sigma_constr_syst", "syst constraint width", 1.0)
    unit.setConstant(True)
    keep["sigma_constr_syst"] = unit
    for j in range(num_systs):
        alpha = ROOT.RooRealVar(f"alpha_syst{j}", f"yield syst NP {j}", 0.0, -5.0, 5.0)
        glob = ROOT.RooRealVar(f"nom_alpha_syst{j}", f"global obs: yield syst {j}", 0.0)
        glob.setConstant(True)
        constr = ROOT.RooGaussian(
            f"constr_alpha_syst{j}", f"Gaussian constraint on alpha_syst{j}", glob, alpha, unit
        )
        keep.update({alpha.GetName(): alpha, glob.GetName(): glob, constr.GetName(): constr})
        infos.append({"np": alpha, "constr": constr, "global_ob": glob})
    return infos


def build_workspace(
    seed: int = 42,
    with_np: bool = True,
    generic_bkg: bool = False,
    generic_sig: bool = False,
    sig_form: str = "gauss",
    bkg_form: str = "exp",
    fix_shape: bool = False,
    constraint: str = "gauss",
    yield_sf: float = 1.0,
    num_channels: int = 3,
    num_systs: int = 0,
) -> ROOT.RooWorkspace:
    """Build the workspace."""
    ROOT.gRandom.SetSeed(seed)

    channels = get_channels(num_channels)

    # ── Shared observable + channel index ───────────────────────────────────
    x = ROOT.RooRealVar("x", "observable [a.u.]", 10.0, 20.0)
    cat = ROOT.RooCategory("index", "channel index")
    for ch in channels:
        cat.defineType(ch)

    # ── POI ─────────────────────────────────────────────────────────────────
    mu_sig = ROOT.RooRealVar("mu_sig", "signal strength", 1.0, -5.0, 10.0)

    _keep = {}
    np_info = build_width_np(constraint, _keep) if with_np else None
    syst_infos = build_syst_nps(num_systs, _keep)

    # ── Signal-width systematic (only when with_np=True) ────────────────────
    # Convention follows HS3: constraint PDF is Gaussian(x=nom, mean=alpha, sigma).
    # quickFit adds it to the NLL via --externalConstraint (not via RooProdPdf,
    # which breaks extended-likelihood evaluation for RooSimultaneous in ROOT 6.30+).
    # The NP and its constraint are built by build_width_np() above.

    # ── Per-channel PDFs and toy datasets ───────────────────────────────────

    channel_pdfs: dict[str, ROOT.RooAddPdf] = {}
    channel_data: dict[str, ROOT.RooDataSet] = {}
    np_vars: list[ROOT.RooRealVar] = []

    for ch, cfg in channels.items():
        # Background: exp(tau*x)
        tau, bkg = build_background(
            ch, cfg, x, generic_bkg=generic_bkg, bkg_form=bkg_form, fix_shape=fix_shape, keep=_keep
        )

        # Signal: Gaussian with fixed mean
        mean = ROOT.RooRealVar(f"mean_{ch}", f"signal mean ({ch})", 15.0)
        sigma_nom = ROOT.RooRealVar(f"sigma_nom_{ch}", f"nominal signal sigma ({ch})", cfg["sigma"])
        mean.setConstant(True)
        sigma_nom.setConstant(True)

        # Effective width: depends on the NP form
        if with_np and np_info["kind"] == "add":
            # sigma_ch = sigma_nom_ch * (1 + SIGMA_DELTA * alpha_sigma)
            sigma = ROOT.RooFormulaVar(
                f"sigma_{ch}",
                f"signal sigma ({ch})",
                f"@0 * (1.0 + {SIGMA_DELTA} * @1)",
                ROOT.RooArgList(sigma_nom, np_info["np"]),
            )
            _keep[f"sigma_{ch}"] = sigma
        elif with_np and np_info["kind"] == "mul":
            sigma = ROOT.RooProduct(
                f"sigma_{ch}", f"signal sigma ({ch})", ROOT.RooArgList(sigma_nom, np_info["np"])
            )
            _keep[f"sigma_{ch}"] = sigma
        else:
            sigma = sigma_nom  # fixed at nominal width

        sig = build_signal(
            ch, x, mean, sigma, generic_sig=generic_sig, sig_form=sig_form, keep=_keep
        )

        # Nominal signal yield (constant — only mu_sig floats the signal norm)
        nsig_nom = ROOT.RooRealVar(
            f"nsig_{ch}", f"nominal signal yield ({ch})", cfg["nsig"] * yield_sf
        )
        nsig_nom.setConstant(True)

        # Scaled signal yield: mu_sig * nsig_nom * (per-syst response factors)
        factors = ROOT.RooArgList(mu_sig, nsig_nom)
        for j, sinfo in enumerate(syst_infos):
            delta = _syst_delta(j, int(ch[2:]))
            resp = ROOT.RooFormulaVar(
                f"resp_syst{j}_{ch}",
                f"syst {j} response ({ch})",
                f"1.0 + {delta} * @0",
                ROOT.RooArgList(sinfo["np"]),
            )
            _keep[f"resp_syst{j}_{ch}"] = resp
            factors.add(resp)
        nsig_tot = ROOT.RooProduct(f"nsig_tot_{ch}", f"scaled signal yield ({ch})", factors)

        # Floating background yield
        nbkg = ROOT.RooRealVar(
            f"nbkg_{ch}", f"background yield ({ch})", cfg["nbkg"] * yield_sf, 0.0, 500.0 * yield_sf
        )

        # Extended sum PDF
        model = ROOT.RooAddPdf(
            f"model_{ch}",
            f"full model ({ch})",
            ROOT.RooArgList(sig, bkg),
            ROOT.RooArgList(nsig_tot, nbkg),
        )

        channel_pdfs[ch] = model

        # Generate toy data at nominal parameters
        data_ch = model.generate(ROOT.RooArgSet(x), ROOT.RooFit.Extended())
        channel_data[ch] = data_ch
        print(
            f"  {ch}: {data_ch.numEntries():3d} events generated "
            f"(expected {cfg['nsig'] + cfg['nbkg']:.0f})"
        )

        # Unconstrained NPs
        if not fix_shape:
            np_vars.append(tau)
        np_vars.append(nbkg)

        _keep.update(
            {
                f"tau_{ch}": tau,
                f"bkg_{ch}": bkg,
                f"mean_{ch}": mean,
                f"sigma_nom_{ch}": sigma_nom,
                f"sig_{ch}": sig,
                f"nsig_nom_{ch}": nsig_nom,
                f"nsig_tot_{ch}": nsig_tot,
                f"nbkg_{ch}": nbkg,
                f"model_{ch}": model,
            }
        )

    # ── Simultaneous PDF ────────────────────────────────────────────────────
    sim_pdf = ROOT.RooSimultaneous("sim_pdf", "simultaneous model", cat)
    for ch, pdf in channel_pdfs.items():
        sim_pdf.addPdf(pdf, ch)

    # ── Combined dataset ────────────────────────────────────────────────────
    channel_map = ROOT.std.map("std::string, RooDataSet*")()
    for ch, d in channel_data.items():
        channel_map[ch] = d
    combined_data = ROOT.RooDataSet(
        "combData",
        "combined dataset",
        ROOT.RooArgSet(x, cat),
        ROOT.RooFit.Index(cat),
        ROOT.RooFit.Import(channel_map),
    )
    print(f"  Total: {combined_data.numEntries()} events across all channels")

    # ── Workspace ────────────────────────────────────────────────────────────
    ws = ROOT.RooWorkspace("combWS", "Simple three-channel test workspace")
    wsImport = getattr(ws, "import")

    # Import the simultaneous PDF and (optionally) the constraint PDF separately.
    # In ROOT 6.30+, wrapping RooSimultaneous in RooProdPdf breaks extended
    # likelihood evaluation; supply the constraint via ExternalConstraints instead.
    wsImport(sim_pdf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    if with_np and np_info["constr"] is not None:
        wsImport(np_info["constr"], ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    for sinfo in syst_infos:
        wsImport(sinfo["constr"], ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    wsImport(combined_data, ROOT.RooFit.Silence())

    # ── ModelConfig ──────────────────────────────────────────────────────────
    mc = ROOT.RooStats.ModelConfig("ModelConfig", ws)
    mc.SetPdf(ws.pdf("sim_pdf"))
    mc.SetObservables(ROOT.RooArgSet(ws.var("x"), ws.cat("index")))
    mc.SetParametersOfInterest(ROOT.RooArgSet(ws.var("mu_sig")))

    np_argset = ROOT.RooArgSet()
    for v in np_vars:
        ws_var = ws.var(v.GetName())
        if ws_var:
            np_argset.add(ws_var)
    if with_np:
        np_argset.add(ws.var(np_info["np"].GetName()))
    for sinfo in syst_infos:
        np_argset.add(ws.var(sinfo["np"].GetName()))
    mc.SetNuisanceParameters(np_argset)
    glob_argset = ROOT.RooArgSet()
    if with_np and np_info["global_ob"] is not None:
        glob_argset.add(ws.var(np_info["global_ob"].GetName()))
    for sinfo in syst_infos:
        glob_argset.add(ws.var(sinfo["global_ob"].GetName()))
    if glob_argset.getSize() > 0:
        mc.SetGlobalObservables(glob_argset)

    wsImport(mc)

    # ── Save nominal parameter snapshot ──────────────────────────────────────
    snap_vars = ROOT.RooArgSet()
    snap_vars.add(ws.var("mu_sig"))
    if with_np:
        snap_vars.add(ws.var(np_info["np"].GetName()))
        if np_info["global_ob"] is not None:
            snap_vars.add(ws.var(np_info["global_ob"].GetName()))
    for sinfo in syst_infos:
        snap_vars.add(ws.var(sinfo["np"].GetName()))
        snap_vars.add(ws.var(sinfo["global_ob"].GetName()))
    for ch in channels:
        for name in [f"tau_{ch}", f"nbkg_{ch}"]:
            v = ws.var(name)
            if v:
                snap_vars.add(v)
    ws.saveSnapshot("nominal", snap_vars)
    ws._constr_name = (
        np_info["constr"].GetName() if (with_np and np_info["constr"] is not None) else None
    )
    return ws


def _output_stem(
    *,
    with_np: bool,
    generic_bkg: bool,
    generic_sig: bool,
    sig_form: str,
    bkg_form: str,
    fix_shape: bool,
    constraint: str,
    yield_sf: float = 1.0,
    num_channels: int = 3,
    num_systs: int = 0,
) -> str:
    """Derive a default workspace file stem from the build options.

    Only non-default aspects are encoded, so the base build is simply
    ``simple_workspace``. num_channels and yield_sf are included when they
    differ from their defaults so auto-named variants stay unique.
    (workflow.sh instead names files via its own fully-explicit canonical_stem.)
    """
    stem = "simple_workspace"
    if num_channels != 3:
        stem += f"_{num_channels}ch"
    if generic_bkg:
        stem += "_generic"
        if bkg_form == "poly":
            stem += "_poly"
    if sig_form == "dscb":
        stem += "_dscb"
    elif generic_sig:
        stem += "_gensig"
    if with_np and constraint == "poisson":
        stem += "_poisson"
    if with_np and constraint == "gauss":
        stem += "_gauss"
    if fix_shape:
        stem += "_fixshape"
    if not with_np:
        stem += "_nonp"
    if yield_sf != 1.0:
        stem += f"_yield{yield_sf:g}x".replace(".", "p")
    if num_systs:
        stem += f"_systs{num_systs}"
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-np",
        action="store_true",
        help="Omit the signal-width nuisance parameter (alpha_sigma). "
        "Default output name becomes simple_workspace_nonp.root.",
    )
    parser.add_argument(
        "--generic-bkg",
        action="store_true",
        help="Use RooGenericPdf instead of RooExponential for backgrounds "
        "(same exp(tau*x) shape; exports as generic_dist in HS3).",
    )
    parser.add_argument(
        "--bkg-form",
        choices=["exp", "poly"],
        default="exp",
        help="Generic background functional form (only with --generic-bkg)",
    )
    parser.add_argument(
        "--generic-sig", action="store_true", help="express the signal gaussian as a RooGenericPdf"
    )
    parser.add_argument(
        "--sig-form",
        choices=["gauss", "dscb"],
        default="gauss",
        help="Signal shape: Gaussian (default) or double-sided crystal ball "
        "(RooCrystalBall with fixed tail parameters)",
    )
    parser.add_argument(
        "--fix-bkg-shape",
        action="store_true",
        help="Hold tau_ch/slope_ch constant so the bkg normalization is frozen against the mu scan",
    )
    parser.add_argument(
        "--constraint",
        choices=["gauss", "poisson", "none"],
        default="gauss",
        help="Aux/constraint form for the width NP -> gauss poisson or None (free NP)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ROOT file name (default: simple_workspace.root or simple_workspace_nonp.root)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for toy data generation (default: 42)"
    )
    parser.add_argument(
        "--yield-sf", type=float, default=1.0, help="scale all yields by a constant factor"
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=3,
        help="specify the number of channels used (no upper limit; the first 30 "
        "use the hardcoded CHANNELS table, beyond that a deterministic formula)",
    )
    parser.add_argument(
        "--num-systs",
        type=int,
        default=0,
        help="add M shared Gaussian-constrained yield-systematic NPs (alpha_syst<j>), "
        "each scaling every channel's signal yield by a per-channel response factor",
    )
    args = parser.parse_args()

    with_np = not args.no_np
    if args.bkg_form == "poly" and not args.generic_bkg:
        print("Note: --bkg-form poly only applies with --generic-bkg; ignoring.")
    if args.sig_form == "dscb" and args.generic_sig:
        print("Note: --generic-sig only applies to the gaussian signal; ignoring.")

    if args.output is None:
        args.output = (
            _output_stem(
                with_np=with_np,
                generic_bkg=args.generic_bkg,
                generic_sig=args.generic_sig,
                sig_form=args.sig_form,
                bkg_form=args.bkg_form,
                fix_shape=args.fix_bkg_shape,
                constraint=args.constraint,
                yield_sf=args.yield_sf,
                num_channels=args.num_channels,
                num_systs=args.num_systs,
            )
            + ".root"
        )

    bkg_label = "exponential_dist"
    if args.generic_bkg:
        bkg_label = f"generic_dist({args.bkg_form})"
    if args.sig_form == "dscb":
        sig_label = "crystalball_doublesided_dist"
    elif args.generic_sig:
        sig_label = "generic_dist"
    else:
        sig_label = "gaussian_dist"
    np_label = "no NP" if not with_np else f"NP(constraint={args.constraint})"
    shape_label = "fixed-shape" if args.fix_bkg_shape else "floating-shape"
    print(
        f"Building workspace [{np_label}, bkg={bkg_label}, sig={sig_label}, {shape_label}] (seed={args.seed}) ..."
    )
    ws = build_workspace(
        seed=args.seed,
        with_np=with_np,
        generic_bkg=args.generic_bkg,
        generic_sig=args.generic_sig,
        sig_form=args.sig_form,
        bkg_form=args.bkg_form,
        fix_shape=args.fix_bkg_shape,
        constraint=args.constraint,
        yield_sf=args.yield_sf,
        num_channels=args.num_channels,
        num_systs=args.num_systs,
    )

    ws.Print("v")

    ws.writeToFile(args.output, True)
    print(f"\nWorkspace written to: {args.output}")
    print("\nQuick-start fit commands:")
    print(f"  quickFit -f {args.output} -w combWS -m ModelConfig -d combData \\")
    constr_names = []
    if with_np and args.constraint != "none":
        constr_names.append(
            "constr_gamma_sigma" if args.constraint == "poisson" else "constr_alpha_sigma"
        )
    constr_names += [f"constr_alpha_syst{j}" for j in range(args.num_systs)]
    if constr_names:
        print(
            f"           -p mu_sig=1_-5_10 --minos 1 --externalConstraint {','.join(constr_names)} \\"
        )
    else:
        print(f"           -p mu_sig=1_-5_10 --minos 1 \\")
    print(f"           -o result_fit.root")


if __name__ == "__main__":
    main()

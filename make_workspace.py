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

Systematic NP groups (all default 0, freely mixable):
  --num-sig-yield-systs M (alias --num-systs): alpha_syst<j> scaling nsig_tot_<ch>
  --num-sig-width-systs M : alpha_sig_width_syst<j> scaling sigma_<ch>
  --num-bkg-norm-systs  M : alpha_bkg_norm_syst<j> scaling the bkg yield
  --num-bkg-shape-systs M : alpha_bkg_shape_syst<j> scaling the bkg slope tau
  Each group's NPs are shared unit-Gaussian-constrained (constraints named
  constr_<np-name>, auto-detected downstream) and enter each channel through a
  single multiplicative RooStats::HistFactory::FlexibleInterpVar response
  resp_<kind>_<ch> (nominal 1, asymmetric per-syst/per-channel up/down
  variations of 3-7% from _syst_deltas, HistFactory interpolation code from
  --interp-code, default 4). Alphas sit at 0 during toy generation and every
  interp code evaluates to 1 there, so datasets are identical to the
  systs-less workspace for a given seed.
  Note: the pre-FlexibleInterpVar model multiplied independent per-syst
  factors (1 + delta_j*alpha_j); a group response with code 0 combines
  additively (1 + sum_j delta_j*alpha_j), differing at O(delta^2).

POI  : mu_sig       (signal strength, floated in fit)
NPs  : tau_ch{0,1,2}, nbkg_ch{0,1,2}  (unconstrained)
       alpha_sigma                     (Gaussian constrained, --np only)
       alpha_*syst{0..M-1}             (Gaussian constrained, per-group flags)
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

# Base relative effect of the shared systematic groups
SYST_DELTA_BASE = 0.03

# Systematic group kinds; the index offsets the delta pattern so no two groups
# apply identical variations to a channel.
KIND_IDX = {"sig_yield": 0, "sig_width": 1, "bkg_norm": 2, "bkg_shape": 3}


def _syst_np_stem(kind: str) -> str:
    """NP name stem for a group: alpha_syst for sig_yield (legacy contract,
    --num-systs alias), alpha_<kind>_syst otherwise."""
    return "alpha_syst" if kind == "sig_yield" else f"alpha_{kind}_syst"


def _syst_deltas(kind_idx: int, j: int, i: int) -> tuple[float, float]:
    """(delta_down, delta_up) for syst j of a group on channel i, deterministic
    and seed-independent. delta_up is 3-7%, varying with both indices so no
    syst is degenerate with mu_sig (identical to the historical symmetric
    formula for kind_idx=0); delta_down = delta_up times an asymmetry factor
    in 0.8-1.2 with a different stride so it stays non-degenerate."""
    d_up = SYST_DELTA_BASE + 0.01 * ((j + 2 * i + kind_idx) % 5)
    d_dn = d_up * (0.8 + 0.1 * ((j + 3 * i + kind_idx) % 5))
    return d_dn, d_up

POLY_SLOPE_INIT = -0.02
POLY_SLOPE_LO = -0.049
POLY_SLOPE_HI = 0.049

# Double-sided crystal ball tail parameters (fixed; --sig-form dscb).
# Mildly asymmetric so a left/right swap in an export round-trip is detectable.
DSCB_ALPHA_L = 1.5
DSCB_N_L = 5.0
DSCB_ALPHA_R = 2.0
DSCB_N_R = 3.0


def build_background(ch, cfg, x, *, generic_bkg, bkg_form, fix_shape, shape_resp=None, keep):
    """Return (shape_var, bkg_pdf).

    shape_var is named tau_<ch> for all forms so downstream scripts
    (muscan/export/snapshot) find it unchanged. With shape_resp (the bkg-shape
    FlexibleInterpVar response), the pdf slope becomes
    tau_eff_<ch> = tau_<ch> * resp_bkg_shape_<ch>; tau_<ch> itself stays the
    floating (or --fix-bkg-shape fixed) baseline parameter.
    """
    if generic_bkg and bkg_form == "poly":
        init, lo, hi = POLY_SLOPE_INIT, POLY_SLOPE_LO, POLY_SLOPE_HI
    else:
        init, lo, hi = cfg["tau"], -2.0, -0.001

    tau = ROOT.RooRealVar(f"tau_{ch}", f"bkg shape ({ch})", init, lo, hi)
    if fix_shape:
        tau.setConstant(True)

    slope = tau
    if shape_resp is not None:
        slope = ROOT.RooProduct(
            f"tau_eff_{ch}", f"effective bkg shape ({ch})", ROOT.RooArgList(tau, shape_resp)
        )
        keep[f"tau_eff_{ch}"] = slope

    if generic_bkg:
        expr = "1.0 + @1*@0" if bkg_form == "poly" else "exp(@1*@0)"
        bkg = ROOT.RooGenericPdf(
            f"bkg_{ch}", f"background pdf ({ch})", expr, ROOT.RooArgList(x, slope)
        )
    else:
        bkg = ROOT.RooExponential(f"bkg_{ch}", f"background pdf ({ch})", x, slope)

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


def build_syst_group(kind, count, keep):
    """Build one group of shared systematic NPs, unit-Gaussian constrained.

    kind is a key of KIND_IDX; NPs are named per _syst_np_stem (alpha_syst<j>
    for sig_yield — the legacy --num-systs contract — alpha_<kind>_syst<j>
    otherwise). Returns a list of dicts {"np", "constr", "global_ob"}, one per
    systematic. Constraints keep the constr_ prefix so muscan.py /
    run_simple_fit.sh auto-detect them; they are imported standalone (never in
    a RooProdPdf with the RooSimultaneous — ROOT 6.30+ contract). The Gaussian
    arg order (glob, alpha, width) makes the global observable the constraint's
    x, which export_hs3.py's structural detection requires. All groups share
    the const unit width sigma_constr_syst.
    """
    infos = []
    if count <= 0:
        return infos
    unit = keep.get("sigma_constr_syst")
    if unit is None:
        unit = ROOT.RooRealVar("sigma_constr_syst", "syst constraint width", 1.0)
        unit.setConstant(True)
        keep["sigma_constr_syst"] = unit
    stem = _syst_np_stem(kind)
    label = kind.replace("_", " ")
    for j in range(count):
        alpha = ROOT.RooRealVar(f"{stem}{j}", f"{label} syst NP {j}", 0.0, -5.0, 5.0)
        glob = ROOT.RooRealVar(f"nom_{stem}{j}", f"global obs: {label} syst {j}", 0.0)
        glob.setConstant(True)
        constr = ROOT.RooGaussian(
            f"constr_{stem}{j}", f"Gaussian constraint on {stem}{j}", glob, alpha, unit
        )
        keep.update({alpha.GetName(): alpha, glob.GetName(): glob, constr.GetName(): constr})
        infos.append({"np": alpha, "constr": constr, "global_ob": glob})
    return infos


def build_group_response(kind, ch, ch_idx, infos, interp_code, keep):
    """Build the per-channel response factor for one systematic group.

    A single RooStats::HistFactory::FlexibleInterpVar resp_<kind>_<ch> holds
    every NP of the group: nominal 1.0, per-NP low/high = 1 -/+ the asymmetric
    deltas from _syst_deltas, with interp_code applied to all NPs. It is folded
    multiplicatively into the target quantity by the caller. Returns None when
    the group is empty.
    """
    if not infos:
        return None
    nps = ROOT.RooArgList()
    low = ROOT.std.vector("double")()
    high = ROOT.std.vector("double")()
    for j, sinfo in enumerate(infos):
        d_dn, d_up = _syst_deltas(KIND_IDX[kind], j, ch_idx)
        nps.add(sinfo["np"])
        low.push_back(1.0 - d_dn)
        high.push_back(1.0 + d_up)
    resp = ROOT.RooStats.HistFactory.FlexibleInterpVar(
        f"resp_{kind}_{ch}", f"{kind.replace('_', ' ')} response ({ch})", nps, 1.0, low, high
    )
    resp.setAllInterpCodes(interp_code)
    keep[f"resp_{kind}_{ch}"] = resp
    return resp


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
    num_sig_yield_systs: int = 0,
    num_sig_width_systs: int = 0,
    num_bkg_norm_systs: int = 0,
    num_bkg_shape_systs: int = 0,
    interp_code: int = 4,
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
    syst_groups = {
        kind: build_syst_group(kind, count, _keep)
        for kind, count in (
            ("sig_yield", num_sig_yield_systs),
            ("sig_width", num_sig_width_systs),
            ("bkg_norm", num_bkg_norm_systs),
            ("bkg_shape", num_bkg_shape_systs),
        )
    }
    all_syst_infos = [info for infos in syst_groups.values() for info in infos]

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
        ch_idx = int(ch[2:])

        # Per-group FlexibleInterpVar response factors (None for empty groups)
        resp = {
            kind: build_group_response(kind, ch, ch_idx, infos, interp_code, _keep)
            for kind, infos in syst_groups.items()
        }

        # Background: exp(tau*x)
        tau, bkg = build_background(
            ch,
            cfg,
            x,
            generic_bkg=generic_bkg,
            bkg_form=bkg_form,
            fix_shape=fix_shape,
            shape_resp=resp["bkg_shape"],
            keep=_keep,
        )

        # Signal: Gaussian with fixed mean
        mean = ROOT.RooRealVar(f"mean_{ch}", f"signal mean ({ch})", 15.0)
        sigma_nom = ROOT.RooRealVar(f"sigma_nom_{ch}", f"nominal signal sigma ({ch})", cfg["sigma"])
        mean.setConstant(True)
        sigma_nom.setConstant(True)

        # Effective width: depends on the NP form; when sig-width systs are on,
        # the width-NP node becomes sigma_base_<ch> and sigma_<ch> is its
        # product with the group response
        width_resp = resp["sig_width"]
        base_name = f"sigma_base_{ch}" if width_resp is not None else f"sigma_{ch}"
        if with_np and np_info["kind"] == "add":
            # sigma_nom_ch * (1 + SIGMA_DELTA * alpha_sigma)
            sigma_base = ROOT.RooFormulaVar(
                base_name,
                f"signal sigma ({ch})",
                f"@0 * (1.0 + {SIGMA_DELTA} * @1)",
                ROOT.RooArgList(sigma_nom, np_info["np"]),
            )
            _keep[base_name] = sigma_base
        elif with_np and np_info["kind"] == "mul":
            sigma_base = ROOT.RooProduct(
                base_name, f"signal sigma ({ch})", ROOT.RooArgList(sigma_nom, np_info["np"])
            )
            _keep[base_name] = sigma_base
        else:
            sigma_base = sigma_nom  # fixed at nominal width

        if width_resp is not None:
            sigma = ROOT.RooProduct(
                f"sigma_{ch}", f"signal sigma ({ch})", ROOT.RooArgList(sigma_base, width_resp)
            )
            _keep[f"sigma_{ch}"] = sigma
        else:
            sigma = sigma_base

        sig = build_signal(
            ch, x, mean, sigma, generic_sig=generic_sig, sig_form=sig_form, keep=_keep
        )

        # Nominal signal yield (constant — only mu_sig floats the signal norm)
        nsig_nom = ROOT.RooRealVar(
            f"nsig_{ch}", f"nominal signal yield ({ch})", cfg["nsig"] * yield_sf
        )
        nsig_nom.setConstant(True)

        # Scaled signal yield: mu_sig * nsig_nom * (sig-yield group response)
        factors = ROOT.RooArgList(mu_sig, nsig_nom)
        if resp["sig_yield"] is not None:
            factors.add(resp["sig_yield"])
        nsig_tot = ROOT.RooProduct(f"nsig_tot_{ch}", f"scaled signal yield ({ch})", factors)

        # Floating background yield; the bkg-norm group response scales it
        # without constraining the free nbkg_<ch> baseline
        nbkg = ROOT.RooRealVar(
            f"nbkg_{ch}", f"background yield ({ch})", cfg["nbkg"] * yield_sf, 0.0, 500.0 * yield_sf
        )
        if resp["bkg_norm"] is not None:
            nbkg_coef = ROOT.RooProduct(
                f"nbkg_tot_{ch}",
                f"scaled background yield ({ch})",
                ROOT.RooArgList(nbkg, resp["bkg_norm"]),
            )
            _keep[f"nbkg_tot_{ch}"] = nbkg_coef
        else:
            nbkg_coef = nbkg

        # Extended sum PDF
        model = ROOT.RooAddPdf(
            f"model_{ch}",
            f"full model ({ch})",
            ROOT.RooArgList(sig, bkg),
            ROOT.RooArgList(nsig_tot, nbkg_coef),
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
    for sinfo in all_syst_infos:
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
    for sinfo in all_syst_infos:
        np_argset.add(ws.var(sinfo["np"].GetName()))
    mc.SetNuisanceParameters(np_argset)
    glob_argset = ROOT.RooArgSet()
    if with_np and np_info["global_ob"] is not None:
        glob_argset.add(ws.var(np_info["global_ob"].GetName()))
    for sinfo in all_syst_infos:
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
    for sinfo in all_syst_infos:
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
    num_sig_yield_systs: int = 0,
    num_sig_width_systs: int = 0,
    num_bkg_norm_systs: int = 0,
    num_bkg_shape_systs: int = 0,
    interp_code: int = 4,
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
    if num_sig_yield_systs:
        stem += f"_systs{num_sig_yield_systs}"
    if num_sig_width_systs:
        stem += f"_wsysts{num_sig_width_systs}"
    if num_bkg_norm_systs:
        stem += f"_bnsysts{num_bkg_norm_systs}"
    if num_bkg_shape_systs:
        stem += f"_bssysts{num_bkg_shape_systs}"
    any_systs = num_sig_yield_systs or num_sig_width_systs or num_bkg_norm_systs or num_bkg_shape_systs
    if interp_code != 4 and any_systs:
        stem += f"_interp{interp_code}"
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
        "--num-sig-yield-systs",
        "--num-systs",
        dest="num_sig_yield_systs",
        type=int,
        default=0,
        metavar="M",
        help="add M shared Gaussian-constrained signal-yield systematic NPs "
        "(alpha_syst<j>) entering each channel via the FlexibleInterpVar response "
        "resp_sig_yield_<ch> (--num-systs is a backward-compatible alias)",
    )
    parser.add_argument(
        "--num-sig-width-systs",
        type=int,
        default=0,
        metavar="M",
        help="add M shared signal-width systematic NPs (alpha_sig_width_syst<j>) "
        "scaling sigma_<ch> via resp_sig_width_<ch>",
    )
    parser.add_argument(
        "--num-bkg-norm-systs",
        type=int,
        default=0,
        metavar="M",
        help="add M shared background-normalization systematic NPs "
        "(alpha_bkg_norm_syst<j>) scaling the bkg yield via resp_bkg_norm_<ch> "
        "(nbkg_<ch> itself stays free)",
    )
    parser.add_argument(
        "--num-bkg-shape-systs",
        type=int,
        default=0,
        metavar="M",
        help="add M shared background-shape systematic NPs "
        "(alpha_bkg_shape_syst<j>) scaling the slope tau_<ch> via resp_bkg_shape_<ch>",
    )
    parser.add_argument(
        "--interp-code",
        type=int,
        default=4,
        choices=range(0, 5),
        metavar="K",
        help="HistFactory interpolation code for all systematic responses: "
        "0=piecewise linear, 1=piecewise exponential, 2=quadratic interp/linear "
        "extrap, 3=quadratic interp/exp extrap, 4=polynomial interp/exp extrap "
        "(HistFactory default; default here too)",
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
                num_sig_yield_systs=args.num_sig_yield_systs,
                num_sig_width_systs=args.num_sig_width_systs,
                num_bkg_norm_systs=args.num_bkg_norm_systs,
                num_bkg_shape_systs=args.num_bkg_shape_systs,
                interp_code=args.interp_code,
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
        num_sig_yield_systs=args.num_sig_yield_systs,
        num_sig_width_systs=args.num_sig_width_systs,
        num_bkg_norm_systs=args.num_bkg_norm_systs,
        num_bkg_shape_systs=args.num_bkg_shape_systs,
        interp_code=args.interp_code,
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
    for kind, count in (
        ("sig_yield", args.num_sig_yield_systs),
        ("sig_width", args.num_sig_width_systs),
        ("bkg_norm", args.num_bkg_norm_systs),
        ("bkg_shape", args.num_bkg_shape_systs),
    ):
        stem = _syst_np_stem(kind)
        constr_names += [f"constr_{stem}{j}" for j in range(count)]
    if constr_names:
        print(
            f"           -p mu_sig=1_-5_10 --minos 1 --externalConstraint {','.join(constr_names)} \\"
        )
    else:
        print(f"           -p mu_sig=1_-5_10 --minos 1 \\")
    print(f"           -o result_fit.root")


if __name__ == "__main__":
    main()

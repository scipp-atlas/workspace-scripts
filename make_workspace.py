#!/usr/bin/env python3
"""
Generate a simple three-channel RooFit workspace for quickFit testing.

Model (per channel, unbinned, x in [10, 20]):
  Signal   : Gaussian(x, mean=15, sigma_ch), normalization = mu_sig * nsig_ch
  Background: RooExponential(x, tau_ch)  [default]
           or RooGenericPdf ~ exp(tau_ch*x)  [--generic-bkg; same shape, generic_dist in HS3]
              normalization = nbkg_ch (free)

Systematic uncertainty (with --np, the default):
  alpha_sigma  : NP varying signal width, Gaussian-constrained to N(0,1)
                 sigma_ch = sigma_nom_ch * (1 + SIGMA_DELTA * alpha_sigma)
                 Shared across all channels (correlated).

POI  : mu_sig       (signal strength, floated in fit)
NPs  : tau_ch{0,1,2}, nbkg_ch{0,1,2}  (unconstrained)
       alpha_sigma                     (Gaussian constrained, --np only)
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
}

# Relative uncertainty on signal width applied by alpha_sigma (±1σ → ±10%)
SIGMA_DELTA = 0.10


def build_workspace(seed: int = 42, with_np: bool = True,
                    generic_bkg: bool = False) -> ROOT.RooWorkspace:
    ROOT.gRandom.SetSeed(seed)

    # ── Shared observable + channel index ───────────────────────────────────
    x   = ROOT.RooRealVar("x", "observable [a.u.]", 10.0, 20.0)
    cat = ROOT.RooCategory("index", "channel index")
    for ch in CHANNELS:
        cat.defineType(ch)

    # ── POI ─────────────────────────────────────────────────────────────────
    mu_sig = ROOT.RooRealVar("mu_sig", "signal strength", 1.0, -5.0, 10.0)

    # ── Signal-width systematic (only when with_np=True) ────────────────────
    # Convention follows HS3: constraint PDF is Gaussian(x=nom, mean=alpha, sigma).
    # quickFit adds it to the NLL via --externalConstraint (not via RooProdPdf,
    # which breaks extended-likelihood evaluation for RooSimultaneous in ROOT 6.30+).
    if with_np:
        alpha_sigma     = ROOT.RooRealVar("alpha_sigma",     "signal width NP",          0.0, -5.0, 5.0)
        nom_alpha_sigma = ROOT.RooRealVar("nom_alpha_sigma", "global obs: signal width",  0.0)
        nom_alpha_sigma.setConstant(True)
        sigma_constr    = ROOT.RooRealVar("sigma_constr",    "constraint Gaussian sigma", 1.0)
        sigma_constr.setConstant(True)
        constr_alpha_sigma = ROOT.RooGaussian(
            "constr_alpha_sigma", "Gaussian constraint on alpha_sigma",
            nom_alpha_sigma, alpha_sigma, sigma_constr)

    # ── Per-channel PDFs and toy datasets ───────────────────────────────────
    _keep = {}   # keep Python references alive until workspace import is done

    channel_pdfs: dict[str, ROOT.RooAddPdf] = {}
    channel_data: dict[str, ROOT.RooDataSet] = {}
    np_vars:      list[ROOT.RooRealVar]       = []

    for ch, cfg in CHANNELS.items():
        # Background: exp(tau*x)
        tau = ROOT.RooRealVar(f"tau_{ch}", f"exp decay constant ({ch})",
                              cfg["tau"], -2.0, -0.001)
        if generic_bkg:
            bkg = ROOT.RooGenericPdf(f"bkg_{ch}", f"background pdf ({ch})",
                                     "exp(@1*@0)",
                                     ROOT.RooArgList(x, tau))
        else:
            bkg = ROOT.RooExponential(f"bkg_{ch}", f"background pdf ({ch})", x, tau)

        # Signal: Gaussian with fixed mean
        mean      = ROOT.RooRealVar(f"mean_{ch}",      f"signal mean ({ch})",          15.0)
        sigma_nom = ROOT.RooRealVar(f"sigma_nom_{ch}", f"nominal signal sigma ({ch})", cfg["sigma"])
        mean.setConstant(True)
        sigma_nom.setConstant(True)

        if with_np:
            # sigma_ch = sigma_nom_ch * (1 + SIGMA_DELTA * alpha_sigma)
            sigma = ROOT.RooFormulaVar(
                f"sigma_{ch}", f"signal sigma ({ch})",
                f"@0 * (1.0 + {SIGMA_DELTA} * @1)",
                ROOT.RooArgList(sigma_nom, alpha_sigma))
            _keep[f"sigma_{ch}"] = sigma
        else:
            sigma = sigma_nom  # fixed at nominal width

        sig = ROOT.RooGaussian(f"sig_{ch}", f"signal pdf ({ch})", x, mean, sigma)

        # Nominal signal yield (constant — only mu_sig floats the signal norm)
        nsig_nom = ROOT.RooRealVar(f"nsig_{ch}", f"nominal signal yield ({ch})",
                                   cfg["nsig"])
        nsig_nom.setConstant(True)

        # Scaled signal yield: mu_sig * nsig_nom
        nsig_tot = ROOT.RooProduct(f"nsig_tot_{ch}", f"scaled signal yield ({ch})",
                                   ROOT.RooArgList(mu_sig, nsig_nom))

        # Floating background yield
        nbkg = ROOT.RooRealVar(f"nbkg_{ch}", f"background yield ({ch})",
                               cfg["nbkg"], 0.0, 500.0)

        # Extended sum PDF
        model = ROOT.RooAddPdf(f"model_{ch}", f"full model ({ch})",
                               ROOT.RooArgList(sig, bkg),
                               ROOT.RooArgList(nsig_tot, nbkg))

        channel_pdfs[ch] = model

        # Generate toy data at nominal parameters
        data_ch = model.generate(ROOT.RooArgSet(x), ROOT.RooFit.Extended())
        channel_data[ch] = data_ch
        print(f"  {ch}: {data_ch.numEntries():3d} events generated "
              f"(expected {cfg['nsig'] + cfg['nbkg']:.0f})")

        # Unconstrained NPs
        np_vars += [tau, nbkg]

        _keep.update({
            f"tau_{ch}": tau, f"bkg_{ch}": bkg,
            f"mean_{ch}": mean, f"sigma_nom_{ch}": sigma_nom,
            f"sig_{ch}": sig,
            f"nsig_nom_{ch}": nsig_nom, f"nsig_tot_{ch}": nsig_tot,
            f"nbkg_{ch}": nbkg, f"model_{ch}": model,
        })

    # ── Simultaneous PDF ────────────────────────────────────────────────────
    sim_pdf = ROOT.RooSimultaneous("sim_pdf", "simultaneous model", cat)
    for ch, pdf in channel_pdfs.items():
        sim_pdf.addPdf(pdf, ch)

    # ── Combined dataset ────────────────────────────────────────────────────
    combined_data = ROOT.RooDataSet(
        "combData", "combined dataset",
        ROOT.RooArgSet(x, cat),
        ROOT.RooFit.Index(cat),
        ROOT.RooFit.Import("ch0", channel_data["ch0"]),
        ROOT.RooFit.Import("ch1", channel_data["ch1"]),
        ROOT.RooFit.Import("ch2", channel_data["ch2"]),
    )
    print(f"  Total: {combined_data.numEntries()} events across all channels")

    # ── Workspace ────────────────────────────────────────────────────────────
    ws       = ROOT.RooWorkspace("combWS", "Simple three-channel test workspace")
    wsImport = getattr(ws, "import")

    # Import the simultaneous PDF and (optionally) the constraint PDF separately.
    # In ROOT 6.30+, wrapping RooSimultaneous in RooProdPdf breaks extended
    # likelihood evaluation; supply the constraint via ExternalConstraints instead.
    wsImport(sim_pdf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    if with_np:
        wsImport(constr_alpha_sigma, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
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
        np_argset.add(ws.var("alpha_sigma"))
    mc.SetNuisanceParameters(np_argset)
    if with_np:
        mc.SetGlobalObservables(ROOT.RooArgSet(ws.var("nom_alpha_sigma")))

    wsImport(mc)

    # ── Save nominal parameter snapshot ──────────────────────────────────────
    snap_vars = ROOT.RooArgSet()
    snap_vars.add(ws.var("mu_sig"))
    if with_np:
        snap_vars.add(ws.var("alpha_sigma"))
        snap_vars.add(ws.var("nom_alpha_sigma"))
    for ch in CHANNELS:
        for name in [f"tau_{ch}", f"nbkg_{ch}"]:
            v = ws.var(name)
            if v:
                snap_vars.add(v)
    ws.saveSnapshot("nominal", snap_vars)

    return ws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-np", action="store_true",
                        help="Omit the signal-width nuisance parameter (alpha_sigma). "
                             "Default output name becomes simple_workspace_nonp.root.")
    parser.add_argument("--generic-bkg", action="store_true",
                        help="Use RooGenericPdf instead of RooExponential for backgrounds "
                             "(same exp(tau*x) shape; exports as generic_dist in HS3).")
    parser.add_argument("--output", default=None,
                        help="Output ROOT file name "
                             "(default: simple_workspace.root or simple_workspace_nonp.root)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for toy data generation (default: 42)")
    args = parser.parse_args()

    with_np = not args.no_np
    if args.output is None:
        stem = "simple_workspace" + ("_generic" if args.generic_bkg else "")
        args.output = f"{stem}.root" if with_np else f"{stem}_nonp.root"

    bkg_label = "generic_dist" if args.generic_bkg else "exponential_dist"
    label = ("with NP (alpha_sigma)" if with_np else "without NP") + f", bkg={bkg_label}"
    print(f"Building workspace [{label}] (seed={args.seed}) ...")
    ws = build_workspace(seed=args.seed, with_np=with_np, generic_bkg=args.generic_bkg)

    ws.Print("v")

    ws.writeToFile(args.output, True)
    print(f"\nWorkspace written to: {args.output}")
    print("\nQuick-start fit commands:")
    print(f"  quickFit -f {args.output} -w combWS -m ModelConfig -d combData \\")
    if with_np:
        print(f"           -p mu_sig=1_-5_10 --minos 1 --externalConstraint constr_alpha_sigma \\")
    else:
        print(f"           -p mu_sig=1_-5_10 --minos 1 \\")
    print(f"           -o result_fit.root")


if __name__ == "__main__":
    main()

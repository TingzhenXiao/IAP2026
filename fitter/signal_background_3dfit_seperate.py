import os
import json
import argparse
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import zfit

from hepstats.splot import compute_sweights

# These must exist in your repo, same as in teacher's angularfitter.py
import mypdfs


def main():
    parser = argparse.ArgumentParser(
        description="Extract wA0 / wApp / wAS / wAq from mixed signal+background sample."
    )

    parser.add_argument(
        "--signal",
        type=str,
        default="/ceph/submit/data/user/a/anbeck/B2KPiMM_michele/full.root",
        help="Signal ROOT file",
    )
    parser.add_argument(
        "--signal-tree",
        type=str,
        default="B02KstMuMu_Run1_centralQ2E_sig",
        help="Signal tree name",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="../genbkg/background_toy.root",
        help="Background ROOT file",
    )
    parser.add_argument(
        "--background-tree",
        type=str,
        default="background",
        help="Background tree name",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="angular_component_sweights_output",
        help="Output directory",
    )
    parser.add_argument(
        "--nsig",
        type=int,
        default=None,
        help="Optional number of signal events to sample",
    )
    parser.add_argument(
        "--nbkg",
        type=int,
        default=None,
        help="Optional number of background events to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--q2min",
        type=float,
        default=1.1,
        help="Lower q2 cut",
    )
    parser.add_argument(
        "--q2max",
        type=float,
        default=7.0,
        help="Upper q2 cut",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    np.random.seed(args.seed)
    zfit.settings.set_seed(args.seed)

    # -------------------------------------------------
    # 1) Read signal sample
    # -------------------------------------------------
    with uproot.open(args.signal) as f:
        arr_sig = f[args.signal_tree].arrays(
            ["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi"],
            library="np",
        )

    df_sig = pd.DataFrame(
        {
            "B_mass": arr_sig["B_mass"] / 1000.0,   # same convention as your earlier script
            "cosThetaK": arr_sig["cosThetaK"],
            "cosThetaL": arr_sig["cosThetaL"],
            "q2": arr_sig["q2"],
            "mKpi": arr_sig["mKpi"],
        }
    ).dropna()

    df_sig = df_sig[(df_sig["q2"] > args.q2min) & (df_sig["q2"] < args.q2max)].copy()
    df_sig = df_sig[(df_sig["B_mass"] >= 5.170) & (df_sig["B_mass"] <= 5.500)].copy()
    df_sig["is_background"] = 0

    # Optional subsampling
    if args.nsig is not None and args.nsig < len(df_sig):
        df_sig = df_sig.sample(n=args.nsig, random_state=args.seed).reset_index(drop=True)

    # -------------------------------------------------
    # 2) Read background toy sample
    # -------------------------------------------------
    with uproot.open(args.background) as f:
        arr_bkg = f[args.background_tree].arrays(
            ["B_mass", "cosThetaK", "cosThetaL"],
            library="np",
        )

    df_bkg = pd.DataFrame(
        {
            "B_mass": arr_bkg["B_mass"],
            "cosThetaK": arr_bkg["cosThetaK"],
            "cosThetaL": arr_bkg["cosThetaL"],
        }
    ).dropna()

    # background toy usually has no q2/mKpi truth; keep placeholders for output consistency
    df_bkg["q2"] = np.nan
    df_bkg["mKpi"] = np.nan
    df_bkg["is_background"] = 1

    if args.nbkg is not None and args.nbkg < len(df_bkg):
        df_bkg = df_bkg.sample(n=args.nbkg, random_state=args.seed).reset_index(drop=True)

    # -------------------------------------------------
    # 3) Build mixed dataset
    # -------------------------------------------------
    df_mix = pd.concat(
        [
            df_sig[["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi", "is_background"]],
            df_bkg[["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi", "is_background"]],
        ],
        ignore_index=True,
    )

    df_mix = df_mix.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print("Number of signal events:", len(df_sig))
    print("Number of background events:", len(df_bkg))
    print("Number of mixed events:", len(df_mix))

    # -------------------------------------------------
    # 4) Define observables
    # -------------------------------------------------
    mass = zfit.Space("B_mass", limits=(5.170, 5.500))
    cosK = zfit.Space("cosThetaK", limits=(-1.0, 1.0))
    cosL = zfit.Space("cosThetaL", limits=(-1.0, 1.0))
    obs3d = mass * cosK * cosL

    angles = cosK * cosL
    angle_limits = zfit.Space(axes=0, lower=-1, upper=1) * zfit.Space(axes=1, lower=-1, upper=1)

    data = zfit.Data.from_pandas(df_mix[["B_mass", "cosThetaK", "cosThetaL"]], obs=obs3d)

    # -------------------------------------------------
    # 5) Mass PDFs
    # -------------------------------------------------
    mu_sig = zfit.Parameter("mu_sig", 5.28, 5.24, 5.32)
    sigma_sig = zfit.Parameter("sigma_sig", 0.025, 0.005, 0.08)
    lambda_bmass = zfit.Parameter("lambda_bmass", -2.0, -20.0, -1e-4)

    pdf_sig_mass = zfit.pdf.Gauss(obs=mass, mu=mu_sig, sigma=sigma_sig)
    pdf_bkg_mass = zfit.pdf.Exponential(obs=mass, lambda_=lambda_bmass)

    # -------------------------------------------------
    # 6) Angular-amplitude parameters
    # Same structure as teacher's angularfitter.py
    # -------------------------------------------------
    App = zfit.Parameter("App", 0.1670, -1.0, 2.0)
    A0 = zfit.Parameter("A0", 0.50, -1.0, 2.0)
    Aqs = zfit.Parameter("Aqs", 0.01, -10.0, 10.0)
    Aqc = zfit.Parameter("Aqc", 0.01, -10.0, 10.0)

    AfbHS = zfit.Parameter("AfbHS", 0.0, -1.0, 1.0)
    AfbHC = zfit.Parameter("AfbHC", 0.0, -1.0, 1.0)
    AfbLS = zfit.Parameter("AfbLS", 0.0, -1.0, 1.0)
    AfbLC = zfit.Parameter("AfbLC", 0.0, -1.0, 1.0)

    def ASconditions(params):
        return 1.0 - params["A0"] - params["App"] - params["Aqc"] - params["Aqs"]

    AS = zfit.ComposedParameter(
        "AS",
        ASconditions,
        params={"A0": A0, "App": App, "Aqc": Aqc, "Aqs": Aqs},
    )

    # -------------------------------------------------
    # 7) Yields
    # Keep teacher's logic: one total signal yield, then split it
    # -------------------------------------------------
    n_sig_guess = len(df_sig)
    n_bkg_guess = len(df_bkg)

    Nsig = zfit.Parameter("Nsig", float(n_sig_guess), 0.0, float(len(df_mix)))
    Nbkg = zfit.Parameter("Nbkg", float(n_bkg_guess), 0.0, float(len(df_mix)))

    def yieldAS(params):
        return params["Nsig"] * params["AS"]

    def yieldApp(params):
        return params["Nsig"] * params["App"]

    def yieldA0(params):
        return params["Nsig"] * params["A0"]

    def yieldAq(params):
        return params["Nsig"] * (params["Aqc"] + params["Aqs"])

    N_AS = zfit.ComposedParameter(
        "N_AS",
        yieldAS,
        params={"Nsig": Nsig, "AS": AS},
    )
    N_App = zfit.ComposedParameter(
        "N_App",
        yieldApp,
        params={"Nsig": Nsig, "App": App},
    )
    N_A0 = zfit.ComposedParameter(
        "N_A0",
        yieldA0,
        params={"Nsig": Nsig, "A0": A0},
    )
    N_Aq = zfit.ComposedParameter(
        "N_Aq",
        yieldAq,
        params={"Nsig": Nsig, "Aqc": Aqc, "Aqs": Aqs},
    )

    # -------------------------------------------------
    # 8) Angular PDFs from teacher's framework
    # Total angular fit pdf
    # -------------------------------------------------
    fitpdf_ang = mypdfs.my2Dpdf(
        obs=angles,
        App=App,
        A0=A0,
        AS=AS,
        Aqc=Aqc,
        Aqs=Aqs,
        AfbHC=AfbHC,
        AfbHS=AfbHS,
        AfbLC=AfbLC,
        AfbLS=AfbLS,
    )
    fitpdf_ang.register_analytic_integral(func=mypdfs.integral, limits=angle_limits)

    # Component angular PDFs used for sWeights
    pdfS_ang = mypdfs.my2Dpdf_AS(obs=angles)

    pdfApp_ang = mypdfs.my2Dpdf_App(obs=angles)
    pdfApp_ang.register_analytic_integral(func=mypdfs.integral_App, limits=angle_limits)

    pdfA0_ang = mypdfs.my2Dpdf_A0(obs=angles)
    pdfA0_ang.register_analytic_integral(func=mypdfs.integral_A0, limits=angle_limits)

    pdfAq_ang = mypdfs.my2Dpdf_Aq(obs=angles, Aqc=Aqc, Aqs=Aqs)
    pdfAq_ang.register_analytic_integral(func=mypdfs.integral_Aq, limits=angle_limits)

    # -------------------------------------------------
    # 9) Full 3D component PDFs
    # Assume all signal angular components share same mass shape
    # -------------------------------------------------
    pdf_A0_shape = zfit.pdf.ProductPDF([pdf_sig_mass, pdfA0_ang], obs=obs3d)
    pdf_App_shape = zfit.pdf.ProductPDF([pdf_sig_mass, pdfApp_ang], obs=obs3d)
    pdf_AS_shape = zfit.pdf.ProductPDF([pdf_sig_mass, pdfS_ang], obs=obs3d)
    pdf_Aq_shape = zfit.pdf.ProductPDF([pdf_sig_mass, pdfAq_ang], obs=obs3d)

    # Simple background angular model
    a1_cosh = zfit.Parameter("a1_cosh", 0.0, -0.8, 0.8)
    a2_cosh = zfit.Parameter("a2_cosh", 0.0, -0.8, 0.8)
    a1_cosl = zfit.Parameter("a1_cosl", 0.0, -0.8, 0.8)
    a2_cosl = zfit.Parameter("a2_cosl", 0.0, -0.8, 0.8)

    pdf_bkg_k = zfit.pdf.Legendre(obs=cosK, coeffs=[a1_cosh, a2_cosh])
    pdf_bkg_l = zfit.pdf.Legendre(obs=cosL, coeffs=[a1_cosl, a2_cosl])
    pdf_bkg_ang = zfit.pdf.ProductPDF([pdf_bkg_k, pdf_bkg_l], obs=angles)
    pdf_bkg_shape = zfit.pdf.ProductPDF([pdf_bkg_mass, pdf_bkg_ang], obs=obs3d)

    pdf_A0 = pdf_A0_shape.create_extended(N_A0)
    pdf_App = pdf_App_shape.create_extended(N_App)
    pdf_AS = pdf_AS_shape.create_extended(N_AS)
    pdf_Aq = pdf_Aq_shape.create_extended(N_Aq)
    pdf_bkg = pdf_bkg_shape.create_extended(Nbkg)

    model = zfit.pdf.SumPDF([pdf_AS, pdf_App, pdf_A0, pdf_Aq, pdf_bkg])

    # -------------------------------------------------
    # 10) Fit
    # -------------------------------------------------
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    result.hesse()

    print("\n=== Fit result ===")
    print(result)

    # -------------------------------------------------
    # 11) Save fit results
    # -------------------------------------------------
    params_to_save = [
        Nsig, Nbkg, N_AS, N_App, N_A0, N_Aq,
        App, A0, Aqs, Aqc, AS,
        AfbHS, AfbHC, AfbLS, AfbLC,
        mu_sig, sigma_sig, lambda_bmass,
        a1_cosh, a2_cosh, a1_cosl, a2_cosl
    ]

    fit_results = {}
    for p in params_to_save:
        try:
            val = float(zfit.run(p))
        except Exception:
            val = None

        err = None
        if p in result.params and "hesse" in result.params[p]:
            err = float(result.params[p]["hesse"]["error"])

        fit_results[p.name] = {"value": val, "error": err}

    with open(os.path.join(args.outdir, "fit_results_component_model.json"), "w") as f:
        json.dump(fit_results, f, indent=4)

    # -------------------------------------------------
    # 12) Compute sWeights from component model
    # exactly what teacher does conceptually:
    # component PDFs, not just signal/background
    # -------------------------------------------------
    sweights = compute_sweights(model, data)

    print("\nsWeight keys:")
    for k in sweights.keys():
        print("   ", k)

    df_mix["wAS"] = np.asarray(sweights[N_AS])
    df_mix["wApp"] = np.asarray(sweights[N_App])
    df_mix["wA0"] = np.asarray(sweights[N_A0])
    df_mix["wAq"] = np.asarray(sweights[N_Aq])

    # -------------------------------------------------
    # 13) Closure-style plots
    # These are not true A0/App/AS truth labels, because the original
    # full.root sample is not already split event-by-event into A0/App/AS.
    # So here we only plot weighted distributions.
    # If you have separate truth files A0.root / AP.root / AS.root,
    # then compare those against these weighted histograms.
    # -------------------------------------------------
    for var in ["cosThetaK", "cosThetaL"]:
        plt.figure(figsize=(7, 5))
        plt.hist(
            df_mix[var],
            bins=50,
            range=(-1, 1),
            weights=df_mix["wA0"],
            density=True,
            histtype="step",
            label="wA0-weighted",
        )
        plt.hist(
            df_mix[var],
            bins=50,
            range=(-1, 1),
            weights=df_mix["wApp"],
            density=True,
            histtype="step",
            label="wApp-weighted",
        )
        plt.hist(
            df_mix[var],
            bins=50,
            range=(-1, 1),
            weights=df_mix["wAS"],
            density=True,
            histtype="step",
            label="wAS-weighted",
        )
        plt.xlabel(var)
        plt.ylabel("Density")
        plt.title(f"Weighted angular components: {var}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"{var}_weighted_components.png"), dpi=300)
        plt.close()

    # -------------------------------------------------
    # 14) Mass projection in counts
    # -------------------------------------------------
    x_mass = np.linspace(5.170, 5.500, 500)
    dx = x_mass[1] - x_mass[0]

    y_A0 = pdf_sig_mass.pdf(x_mass).numpy() * float(zfit.run(N_A0)) * dx
    y_App = pdf_sig_mass.pdf(x_mass).numpy() * float(zfit.run(N_App)) * dx
    y_AS = pdf_sig_mass.pdf(x_mass).numpy() * float(zfit.run(N_AS)) * dx
    y_Aq = pdf_sig_mass.pdf(x_mass).numpy() * float(zfit.run(N_Aq)) * dx
    y_bkg = pdf_bkg_mass.pdf(x_mass).numpy() * float(zfit.run(Nbkg)) * dx
    y_tot = y_A0 + y_App + y_AS + y_Aq + y_bkg

    nbins = 60
    bin_edges = np.linspace(5.170, 5.500, nbins + 1)

    plt.figure(figsize=(7, 5))
    plt.hist(
        df_mix["B_mass"],
        bins=bin_edges,
        density=False,
        histtype="step",
        label="Mixed data",
    )
    plt.plot(x_mass, y_A0, label="A0 component")
    plt.plot(x_mass, y_App, label="App component")
    plt.plot(x_mass, y_AS, label="AS component")
    plt.plot(x_mass, y_Aq, label="Aq component")
    plt.plot(x_mass, y_bkg, label="Background")
    plt.plot(x_mass, y_tot, label="Total model")
    plt.xlabel("B_mass")
    plt.ylabel("Events / bin")
    plt.title("Component fit: B_mass projection (counts)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "B_mass_projection_components_counts.png"), dpi=300)
    plt.close()

    # -------------------------------------------------
    # 15) Save output with weights
    # Match teacher-style output naming
    # -------------------------------------------------
    outdf = df_mix.copy()
    outdf["wS"] = outdf["wAS"]   # teacher uses wS for AS
    outdf.to_hdf(
        os.path.join(args.outdir, "mixed_sample_with_component_sweights.h5"),
        key="data",
        mode="w",
    )

    print(f"\nSaved results to: {args.outdir}")
    print("  - fit_results_component_model.json")
    print("  - B_mass_projection_components_counts.png")
    print("  - cosThetaK_weighted_components.png")
    print("  - cosThetaL_weighted_components.png")
    print("  - mixed_sample_with_component_sweights.h5")


if __name__ == "__main__":
    main()
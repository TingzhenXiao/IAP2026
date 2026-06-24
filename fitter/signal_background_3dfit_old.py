import os
import json
import argparse
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import zfit
import mypdfs
from sweights import Cow
from sweights import SWeight
from hepstats.splot import compute_sweights

def zfit_pdf3d_to_callable(zpdf, obs_space):
    def wrapped(m, ck, cl):
        m_arr = np.atleast_1d(np.asarray(m, dtype=float))
        ck_arr = np.atleast_1d(np.asarray(ck, dtype=float))
        cl_arr = np.atleast_1d(np.asarray(cl, dtype=float))

        scalar_input = (
            np.ndim(m) == 0
            and np.ndim(ck) == 0
            and np.ndim(cl) == 0
        )

        pts = np.column_stack((m_arr, ck_arr, cl_arr))
        vals = np.asarray(
            zpdf.pdf(pts, norm=obs_space).numpy(),
            dtype=float,
        ).reshape(-1)

        if scalar_input:
            return float(vals[0])

        return vals

    return wrapped

def main():
    parser = argparse.ArgumentParser(
        description="3D signal+background fit in B_mass, cosThetaK, cosThetaL, with sWeight check."
    )
    parser.add_argument(
        "--signal",
        type=str,
        default="/ceph/submit/data/user/a/anbeck/B2KPiMM_michele/full.root",
        help="Input ROOT file containing the signal tree",
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
        help="Input ROOT file containing the background toy tree",
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
        default="signal_background_3dfit_output",
        help="Directory to save plots and fit results",
    )
    parser.add_argument("--nsig", type=int, default=None, help="Number of signal events to sample; use all if omitted")
    parser.add_argument("--nbkg", type=int, default=None, help="Number of background events to sample; use all if omitted")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # -------------------------------------------------
    # 1) Read signal sample
    # -------------------------------------------------
    with uproot.open(args.signal) as f:
        arr_sig = f[args.signal_tree].arrays(
            ["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi"],
            library="np",
        )

    df_sig = pd.DataFrame({
        "B_mass": arr_sig["B_mass"],
        "cosThetaK": arr_sig["cosThetaK"],
        "cosThetaL": arr_sig["cosThetaL"],
        "q2": arr_sig["q2"],
        "mKpi": arr_sig["mKpi"],
    }).dropna()

    df_sig["B_mass"] = df_sig["B_mass"] / 1000.0

    df_sig = df_sig[(df_sig["q2"] > 1.1) & (df_sig["q2"] < 7.0)].copy()
    df_sig = df_sig[(df_sig["mKpi"] < 1.5)].copy()
    df_sig = df_sig[(df_sig["B_mass"] >= 5.170) & (df_sig["B_mass"] <= 5.500)].copy()
    df_sig["is_signal"] = 1

    # -------------------------------------------------
    # 2) Read background toy sample
    # -------------------------------------------------
    with uproot.open(args.background) as f:
        arr_bkg = f[args.background_tree].arrays(
            ["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi"],
            library="np",
        )

    df_bkg = pd.DataFrame({
        "B_mass": arr_bkg["B_mass"],
        "cosThetaK": arr_bkg["cosThetaK"],
        "cosThetaL": arr_bkg["cosThetaL"],
        "q2": arr_bkg["q2"],
        "mKpi": arr_bkg["mKpi"],
    }).dropna()

    df_bkg["is_signal"] = 0

    print("Signal B_mass min/max:", df_sig["B_mass"].min(), df_sig["B_mass"].max())
    print("Background B_mass min/max:", df_bkg["B_mass"].min(), df_bkg["B_mass"].max())

    if args.nsig is not None and args.nsig < len(df_sig):
        df_sig = df_sig.sample(n=args.nsig, random_state=args.seed).reset_index(drop=True)

    if args.nbkg is not None and args.nbkg < len(df_bkg):
        df_bkg = df_bkg.sample(n=args.nbkg, random_state=args.seed).reset_index(drop=True)
    # -------------------------------------------------
    # 3) Sample and build a mixed dataset
    # -------------------------------------------------
    df_sig = df_sig.copy()
    df_bkg = df_bkg.copy()

    df_mix = pd.concat([
        df_sig[["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi", "is_signal"]],
        df_bkg[["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi", "is_signal"]],
    ], ignore_index=True)

    # shuffle mixed sample
    df_mix = df_mix.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print("Number of signal truth events:", len(df_sig))
    print("Number of background truth events:", len(df_bkg))
    print("Number of mixed events:", len(df_mix))
    print(df_mix.head())

    # -------------------------------------------------
    # 4) Define observables
    # -------------------------------------------------
    mass = zfit.Space("B_mass", limits=(5.170, 5.500))
    cosK = zfit.Space("cosThetaK", limits=(-1.0, 1.0))
    cosL = zfit.Space("cosThetaL", limits=(-1.0, 1.0))
    obs = mass * cosK * cosL

    data = zfit.Data.from_pandas(df=df_mix, obs=obs)

    # -------------------------------------------------
    # 5) Define signal mass PDF: DoubleCB with fixed tails
    # -------------------------------------------------
    mu_sig = zfit.Parameter("mu_sig", 5.28329, 5.26, 5.30)
    sigma_sig = zfit.Parameter("sigma_sig", 0.01618, 0.005, 0.05)

    alphal = zfit.Parameter("alphal", 1.641422886178359)
    nl     = zfit.Parameter("nl",     1.9181087305552411)
    alphar = zfit.Parameter("alphar", 2.1093382945592896)
    nr     = zfit.Parameter("nr",     2.6464159750062994)

    alphal.floating = False
    nl.floating = False
    alphar.floating = False
    nr.floating = False

    pdf_sig_mass = zfit.pdf.DoubleCB(
        obs=mass,
        mu=mu_sig,
        sigma=sigma_sig,
        alphal=alphal,
        nl=nl,
        alphar=alphar,
        nr=nr,
    )
    
    alphal.floating = False
    nl.floating = False
    alphar.floating = False
    nr.floating = False

    # Angular space for mypdfs
    angles = cosK * cosL
    limith = zfit.Space(axes=0, lower=-1, upper=1)
    limitl = zfit.Space(axes=1, lower=-1, upper=1)
    limits = limith * limitl

    # Keep only the minimal components
    A0_frac  = zfit.Parameter("A0_frac", 0.50, 0.0, 0.95)
    App_frac = zfit.Parameter("App_frac", 0.20, 0.0, 0.95)

    def ASfrac_func(params):
        return 1.0 - params["A0_frac"] - params["App_frac"]

    AS_frac = zfit.ComposedParameter(
        "AS_frac",
        ASfrac_func,
        params={"A0_frac": A0_frac, "App_frac": App_frac},
    )

    # Pure angular component pdfs from framework
    pdfA0_ang = mypdfs.my2Dpdf_A0(obs=angles)
    pdfA0_ang.register_analytic_integral(func=mypdfs.integral_A0, limits=limits)

    pdfApp_ang = mypdfs.my2Dpdf_App(obs=angles)
    pdfApp_ang.register_analytic_integral(func=mypdfs.integral_App, limits=limits)

    pdfAS_ang = mypdfs.my2Dpdf_AS(obs=angles)

    # Full 3D shapes
    pdf_A0_shape  = zfit.pdf.ProductPDF([pdf_sig_mass, pdfA0_ang], obs=obs)
    pdf_App_shape = zfit.pdf.ProductPDF([pdf_sig_mass, pdfApp_ang], obs=obs)
    pdf_AS_shape  = zfit.pdf.ProductPDF([pdf_sig_mass, pdfAS_ang], obs=obs)

    # -------------------------------------------------
    # 6) Define background PDF
    # -------------------------------------------------
    lambda_bmass = zfit.Parameter("lambda_bmass", -0.1, -2.0, 0.0)

    a1_cosh = zfit.Parameter("a1_cosh", 0.0, -0.5, 0.5)
    a2_cosh = zfit.Parameter("a2_cosh", 0.0, -0.8, 0.8)

    a1_cosl = zfit.Parameter("a1_cosl", 0.0, -0.5, 0.5)
    a2_cosl = zfit.Parameter("a2_cosl", 0.0, -0.8, 0.8)

    pdf_bkg_mass = zfit.pdf.Exponential(obs=mass, lambda_=lambda_bmass)
    pdf_bkg_k = zfit.pdf.Legendre(obs=cosK, coeffs=[a1_cosh, a2_cosh])
    pdf_bkg_l = zfit.pdf.Legendre(obs=cosL, coeffs=[a1_cosl, a2_cosl])

    pdf_bkg_shape = zfit.pdf.ProductPDF([pdf_bkg_mass, pdf_bkg_k, pdf_bkg_l], obs=obs)

    # -------------------------------------------------
    # 7) Extended total model: A0 + App + AS + background
    # -------------------------------------------------
    Nsig = zfit.Parameter("Nsig", len(df_sig), 0.0, len(df_mix))
    Nbkg = zfit.Parameter("Nbkg", len(df_bkg), 0.0, len(df_mix))

    def yield_A0(params):
        return params["Nsig"] * params["A0_frac"]

    def yield_App(params):
        return params["Nsig"] * params["App_frac"]

    def yield_AS(params):
        return params["Nsig"] * params["AS_frac"]

    N_A0 = zfit.ComposedParameter(
        "N_A0",
        yield_A0,
        params={"Nsig": Nsig, "A0_frac": A0_frac},
    )

    N_App = zfit.ComposedParameter(
        "N_App",
        yield_App,
        params={"Nsig": Nsig, "App_frac": App_frac},
    )

    N_AS = zfit.ComposedParameter(
        "N_AS",
        yield_AS,
        params={"Nsig": Nsig, "AS_frac": AS_frac},
    )

    pdf_A0 = pdf_A0_shape.create_extended(N_A0)
    pdf_App = pdf_App_shape.create_extended(N_App)
    pdf_AS = pdf_AS_shape.create_extended(N_AS)
    pdf_bkg = pdf_bkg_shape.create_extended(Nbkg)

    model = zfit.pdf.SumPDF([pdf_A0, pdf_App, pdf_AS, pdf_bkg])

    # -------------------------------------------------
    # 8) Fit
    # -------------------------------------------------
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    result.hesse()

    print("\n=== 3D fit result ===")
    print(result)
    params = [
        Nsig, Nbkg, N_A0, N_App, N_AS,
        mu_sig, sigma_sig, lambda_bmass,
        A0_frac, App_frac, AS_frac,
    ]

    fit_results = {}
    for p in params:
        try:
            val = float(zfit.run(p))
        except Exception:
            val = None

        err = None
        if p in result.params:
            if "hesse" in result.params[p]:
                err = float(result.params[p]["hesse"]["error"])

        fit_results[p.name] = {
            "value": val,
            "error": err,
        }
    
    def make_1d_pdf_callable(zpdf, obs_space):
        def _pdf(x):
            x_arr = np.asarray(x, dtype=float)

            # 标量输入：给 scipy.integrate.quad 用
            if x_arr.ndim == 0:
                val = zpdf.pdf(np.array([x_arr.item()]), norm=obs_space).numpy()
                return float(np.asarray(val).reshape(-1)[0])

            # 数组输入：给事件数组算权重用
            val = zpdf.pdf(x_arr, norm=obs_space).numpy()
            return np.asarray(val, dtype=float).reshape(-1)

        return _pdf

    # -------------------------------------------------
    # 9) Plot B_mass projection as stacked components
    #     with background at the bottom
    # -------------------------------------------------
    x = np.linspace(5.170, 5.500, 500)
    bin_edges = np.linspace(5.170, 5.500, 51)
    bin_width = bin_edges[1] - bin_edges[0]

    y_A0 = pdf_sig_mass.pdf(x).numpy() * float(zfit.run(N_A0)) * bin_width
    y_App = pdf_sig_mass.pdf(x).numpy() * float(zfit.run(N_App)) * bin_width
    y_AS = pdf_sig_mass.pdf(x).numpy() * float(zfit.run(N_AS)) * bin_width
    y_bkg = pdf_bkg_mass.pdf(x).numpy() * float(zfit.run(Nbkg)) * bin_width

    y_total = y_bkg + y_A0 + y_App + y_AS

    plt.figure(figsize=(8, 5))

    plt.hist(
        df_mix["B_mass"],
        bins=bin_edges,
        histtype="step",
        linewidth=1.8,
        label="Mixed data",
    )

    # background at the bottom
    plt.fill_between(
        x,
        0,
        y_bkg,
        alpha=0.5,
        label="Background component",
    )

    plt.fill_between(
        x,
        y_bkg,
        y_bkg + y_A0,
        alpha=0.5,
        label="A0 component",
    )

    plt.fill_between(
        x,
        y_bkg + y_A0,
        y_bkg + y_A0 + y_App,
        alpha=0.5,
        label="App component",
    )

    plt.fill_between(
        x,
        y_bkg + y_A0 + y_App,
        y_total,
        alpha=0.5,
        label="AS component",
    )

    plt.plot(
        x,
        y_total,
        linewidth=2.0,
        label="Total model",
    )

    plt.xlabel("B_mass")
    plt.ylabel("Events / bin")
    plt.title("3D fit: B_mass projection (stacked components)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.outdir, "B_mass_projection_stacked.png"),
        dpi=300,
    )
    plt.close()
    # -------------------------------------------------
    # 10) Compute component sWeights with sweights.SWeight
    #     using the 3 discriminating variables:
    #     B_mass, cosThetaK, cosThetaL
    # -------------------------------------------------
    pdf_A0_call = zfit_pdf3d_to_callable(pdf_A0_shape, obs)
    pdf_App_call = zfit_pdf3d_to_callable(pdf_App_shape, obs)
    pdf_AS_call = zfit_pdf3d_to_callable(pdf_AS_shape, obs)
    pdf_bkg_call = zfit_pdf3d_to_callable(pdf_bkg_shape, obs)

    disc_data = df_mix[["B_mass", "cosThetaK", "cosThetaL"]].to_numpy()

    sw = SWeight(
        data=disc_data,
        pdfs=[pdf_A0_call, pdf_App_call, pdf_AS_call, pdf_bkg_call],
        yields=[
            float(zfit.run(N_A0)),
            float(zfit.run(N_App)),
            float(zfit.run(N_AS)),
            float(zfit.run(Nbkg)),
        ],
        discvarranges=[
            (5.170, 5.500),
            (-1.0, 1.0),
            (-1.0, 1.0),
        ],
        method="summation",
        compnames=["A0", "App", "AS", "bkg"],
        verbose=False,
        checks=False,
    )

    mvals = df_mix["B_mass"].to_numpy()
    ckvals = df_mix["cosThetaK"].to_numpy()
    clvals = df_mix["cosThetaL"].to_numpy()

    df_mix["wA0"] = sw.get_weight(0, mvals, ckvals, clvals)
    df_mix["wApp"] = sw.get_weight(1, mvals, ckvals, clvals)
    df_mix["wAS"] = sw.get_weight(2, mvals, ckvals, clvals)
    df_mix["wBkg"] = sw.get_weight(3, mvals, ckvals, clvals)

    print("\nSWeight component sums:")
    print("sum wA0  =", df_mix["wA0"].sum(),  " target ~", float(zfit.run(N_A0)))
    print("sum wApp =", df_mix["wApp"].sum(), " target ~", float(zfit.run(N_App)))
    print("sum wAS  =", df_mix["wAS"].sum(),  " target ~", float(zfit.run(N_AS)))
    print("sum wBkg =", df_mix["wBkg"].sum(), " target ~", float(zfit.run(Nbkg)))



    # # -------------------------------------------------
    # # 12) Compute COW weights: signal / background in B_mass only
    # # -------------------------------------------------
    # mass_values = df_mix["B_mass"].to_numpy()

    # sig_mass_callable = make_1d_pdf_callable(pdf_sig_mass, mass)
    # bkg_mass_callable = make_1d_pdf_callable(pdf_bkg_mass, mass)

    # cow = Cow(
    #     mrange=(5.170, 5.500),
    #     gs=sig_mass_callable,
    #     gb=bkg_mass_callable,
    #     renorm=True,
    #     verbose=False,
    # )

    # df_mix["wSig_cow"] = cow.get_weight(0, mass_values)
    # df_mix["wBkg_cow"] = cow.get_weight(1, mass_values)

    # print("\nCOW signal/background sums:")
    # print("sum wSig_cow =", df_mix["wSig_cow"].sum(), " target ~", float(zfit.run(Nsig)))
    # print("sum wBkg_cow =", df_mix["wBkg_cow"].sum(), " target ~", float(zfit.run(Nbkg)))

    # -------------------------------------------------
    # 13) Plot component-weighted cosThetaK / cosThetaL
    # -------------------------------------------------
    for var in ["cosThetaK", "cosThetaL"]:
        for weight_name in ["wA0", "wApp", "wAS", "wBkg"]:
            valid = df_mix[var].notna()

            plt.figure(figsize=(7, 5))
            plt.hist(
                df_mix.loc[valid, var],
                bins=50,
                weights=df_mix.loc[valid, weight_name],
                density=True,
                histtype="step",
                label=weight_name,
            )
            plt.xlabel(var)
            plt.ylabel("Density")
            plt.title(f"{var} weighted by {weight_name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(args.outdir, f"{var}_{weight_name}.png"),
                dpi=300,
            )
            plt.close()

    # # -------------------------------------------------
    # # 14) Compare signal/background closure in q2 / mKpi
    # #     using classic mass sWeights and COW
    # # -------------------------------------------------
    # for var in ["q2", "mKpi"]:
    #     valid = df_mix[var].notna()

    #     # classic signal/background sWeights
    #     plt.figure(figsize=(7, 5))

    #     plt.hist(
    #         df_sig[var].dropna(),
    #         bins=50,
    #         density=True,
    #         histtype="step",
    #         label="Truth signal",
    #     )

    #     plt.hist(
    #         df_mix.loc[valid, var],
    #         bins=50,
    #         weights=df_mix.loc[valid, "wSig_cow"],
    #         density=True,
    #         histtype="step",
    #         label="Mixed sample weighted by COW"
    #     )

    #     plt.xlabel(var)
    #     plt.ylabel("Density")
    #     plt.title(f"Classic signal/background sWeight closure: {var}")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(
    #         os.path.join(args.outdir, f"classic_sweight_closure_{var}.png"),
    #         dpi=300,
    #     )
    #     plt.close()

    #     # COW
    #     plt.figure(figsize=(7, 5))

    #     plt.hist(
    #         df_sig[var].dropna(),
    #         bins=50,
    #         density=True,
    #         histtype="step",
    #         label="Truth signal",
    #     )

    #     plt.hist(
    #         df_mix.loc[valid, var],
    #         bins=50,
    #         weights=df_mix.loc[valid, "wSig_cow"],
    #         density=True,
    #         histtype="step",
    #         label="Mixed sample weighted by COW",
    #     )

    #     plt.xlabel(var)
    #     plt.ylabel("Density")
    #     plt.title(f"COW closure check: {var}")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(
    #         os.path.join(args.outdir, f"cow_closure_{var}.png"),
    #         dpi=300,
    #     )
    #     plt.close()

    # -------------------------------------------------
    # 15) Compare component sWeights to reference samples
    # -------------------------------------------------
    datadir = os.environ["DATADIR"]

    reference_mapping = {
        "wA0": "A0.root",
        "wApp": "A1.root",
        "wAS": "AS.root",
    }

    variables = ["mKpi", "q2"]

    for weight_name, rootfile in reference_mapping.items():
        ref_path = f"{datadir}/{rootfile}"
        f = uproot.open(ref_path)
        tree = f[f.keys()[0]]

        ref = tree.arrays(["mKpi", "q2"], library="pd").dropna()
        ref = ref[(ref["q2"] > 1.1) & (ref["q2"] < 7.0)].copy()
        ref = ref[(ref["mKpi"] < 1.5)].copy()

        for var in variables:
            valid = df_mix[var].notna()

            plt.figure(figsize=(7, 5))

            plt.hist(
                ref[var],
                bins=50,
                density=True,
                histtype="step",
                label=f"Reference {rootfile}",
            )

            plt.hist(
                df_mix.loc[valid, var],
                bins=50,
                weights=df_mix.loc[valid, weight_name],
                density=True,
                histtype="step",
                label=f"Mixed sample weighted by {weight_name}",
            )

            plt.xlabel(var)
            plt.ylabel("Density")
            plt.title(f"Reference check: {weight_name} vs {rootfile} in {var}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(args.outdir, f"{weight_name}_{var}_reference_check.png"),
                dpi=300,
            )
            plt.close()

    # -------------------------------------------------
    # 16) Save outputs
    # -------------------------------------------------
    with open(os.path.join(args.outdir, "fit_results_signal_background_3d.json"), "w") as f:
        json.dump(fit_results, f, indent=2)

    print(f"\nSaved results to: {args.outdir}")
    print("  - fit_results_signal_background_3d.json")
    print("  - B_mass_projection_stacked.png")
    print("  - classic_sweight_closure_q2.png")
    print("  - classic_sweight_closure_mKpi.png")
    print("  - cow_closure_q2.png")
    print("  - cow_closure_mKpi.png")
    print("  - cosThetaK_wA0.png")
    print("  - cosThetaK_wApp.png")
    print("  - cosThetaK_wAS.png")
    print("  - cosThetaK_wBkg.png")
    print("  - cosThetaL_wA0.png")
    print("  - cosThetaL_wApp.png")
    print("  - cosThetaL_wAS.png")
    print("  - cosThetaL_wBkg.png")
    print("  - wA0_q2_reference_check.png")
    print("  - wA0_mKpi_reference_check.png")
    print("  - wApp_q2_reference_check.png")
    print("  - wApp_mKpi_reference_check.png")
    print("  - wAS_q2_reference_check.png")
    print("  - wAS_mKpi_reference_check.png")


if __name__ == "__main__":
    main()
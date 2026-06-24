import os
import json
import argparse
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import zfit
import mypdfs

from hepstats.splot import compute_sweights
from sweights import Cow

def project_2d_pdf_to_1d(pdf2d, var, xgrid, angles_obs):
    """
    Numerically project a 2D angular PDF onto one variable by integrating
    over the other variable in [-1, 1].
    """
    ygrid = np.linspace(-1.0, 1.0, 400)
    vals = []

    if var == "cosThetaK":
        for x in xgrid:
            pts = np.column_stack([
                np.full_like(ygrid, x, dtype=float),
                ygrid,
            ])
            pdfvals = pdf2d.pdf(pts, norm=angles_obs).numpy().reshape(-1)
            vals.append(np.trapezoid(pdfvals, ygrid))

    elif var == "cosThetaL":
        for x in xgrid:
            pts = np.column_stack([
                ygrid,
                np.full_like(ygrid, x, dtype=float),
            ])
            pdfvals = pdf2d.pdf(pts, norm=angles_obs).numpy().reshape(-1)
            vals.append(np.trapezoid(pdfvals, ygrid))

    else:
        raise ValueError("var must be 'cosThetaK' or 'cosThetaL'")

    return np.asarray(vals, dtype=float)


def plot_angle_projection_stacked(
    df_mix,
    var,
    outdir,
    angles,
    pdfA0_ang,
    pdfApp_ang,
    pdfAS_ang,
    pdf_bkg_k,
    pdf_bkg_l,
    N_A0,
    N_App,
    N_AS,
    Nbkg,
):
    """
    Plot data and fitted stacked component projection for cosThetaK or cosThetaL.
    """
    xmin, xmax = -1.0, 1.0
    nbins = 100
    x = np.linspace(xmin, xmax, 500)
    bin_edges = np.linspace(xmin, xmax, nbins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    # Signal angular projections from the 2D angular PDFs
    y_A0_shape = project_2d_pdf_to_1d(pdfA0_ang, var, x, angles)
    y_App_shape = project_2d_pdf_to_1d(pdfApp_ang, var, x, angles)
    y_AS_shape = project_2d_pdf_to_1d(pdfAS_ang, var, x, angles)

    y_A0 = y_A0_shape * float(zfit.run(N_A0)) * bin_width
    y_App = y_App_shape * float(zfit.run(N_App)) * bin_width
    y_AS = y_AS_shape * float(zfit.run(N_AS)) * bin_width

    # Background projection
    if var == "cosThetaK":
        y_bkg_shape = pdf_bkg_k.pdf(x).numpy().reshape(-1)
    elif var == "cosThetaL":
        y_bkg_shape = pdf_bkg_l.pdf(x).numpy().reshape(-1)
    else:
        raise ValueError("var must be 'cosThetaK' or 'cosThetaL'")

    y_bkg = y_bkg_shape * float(zfit.run(Nbkg)) * bin_width

    y_total = y_bkg + y_A0 + y_App + y_AS

    plt.figure(figsize=(8, 5))

    plot_data_with_errors(
        df_mix[var],
        bin_edges,
        label="Mixed data",
    )

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

    plt.xlabel(var)
    plt.ylabel("Events / bin")
    plt.title(f"3D fit: {var} projection (stacked components)")
    plt.xlim(xmin, xmax)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"{var}_projection_stacked.png"),
        dpi=300,
    )
    plt.close()
    
def plot_data_with_errors(values, bin_edges, label="Data", zorder=20):
    counts, _ = np.histogram(values, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    xerr = 0.5 * (bin_edges[1:] - bin_edges[:-1])
    yerr = np.sqrt(counts)

    plt.errorbar(
        bin_centers,
        counts,
        yerr=yerr,
        xerr=xerr,
        fmt="o",
        markersize=4,
        capsize=0,
        linewidth=1.2,
        label=label,
        zorder=zorder,
    )

def plot_mass_fit_with_pull(
    values,
    bin_edges,
    components,
    xlabel,
    title,
    outfile,
    data_label="Data",
):
    """
    Plot mass fit with a pull panel.

    Parameters
    ----------
    values : array-like
        Data values to histogram.
    bin_edges : array-like
        Bin edges for histogram.
    components : list of tuples
        Each tuple is (label, yvals), where yvals is the predicted counts
        per bin-center bin for that component.
    xlabel : str
        X-axis label.
    title : str
        Plot title for the top panel.
    outfile : str
        Output file path.
    data_label : str
        Label for data points.
    """
    counts, _ = np.histogram(values, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    xerr = 0.5 * (bin_edges[1:] - bin_edges[:-1])
    yerr = np.sqrt(counts)

    # Total prediction
    y_total = np.zeros_like(bin_centers, dtype=float)
    for _, yvals in components:
        y_total = y_total + np.asarray(yvals, dtype=float)

    # Pull:
    # (Ndata in bin - fit prediction in bin) / sqrt(Ndata)
    pull = np.zeros_like(bin_centers, dtype=float)
    mask = counts > 0
    pull[mask] = (counts[mask] - y_total[mask]) / np.sqrt(counts[mask])

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )
def get_density_and_error(values, bin_edges, weights=None):
    """
    Return histogram density and its uncertainty.

    For unweighted data:
        density_i = n_i / (N * bin_width)
        err_i     = sqrt(n_i) / (N * bin_width)

    For weighted data:
        density_i = sum(w)_i / (sum(w) * bin_width)
        err_i     = sqrt(sum(w^2)_i) / (sum(w) * bin_width)

    This is an approximate uncertainty propagation, sufficient for
    closure-check plots.
    """
    values = np.asarray(values)
    mask = np.isfinite(values)
    values = values[mask]

    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if weights is None:
        counts, _ = np.histogram(values, bins=bin_edges)
        norm = counts.sum()

        density = np.zeros_like(bin_centers, dtype=float)
        error = np.zeros_like(bin_centers, dtype=float)

        if norm > 0:
            density = counts / (norm * bin_widths)
            error = np.sqrt(counts) / (norm * bin_widths)

        return bin_centers, density, error, counts

    weights = np.asarray(weights)
    weights = weights[mask]

    sumw, _ = np.histogram(values, bins=bin_edges, weights=weights)
    sumw2, _ = np.histogram(values, bins=bin_edges, weights=weights ** 2)
    norm = np.sum(weights)

    density = np.zeros_like(bin_centers, dtype=float)
    error = np.zeros_like(bin_centers, dtype=float)

    if norm != 0:
        density = sumw / (norm * bin_widths)
        error = np.sqrt(sumw2) / (abs(norm) * bin_widths)

    return bin_centers, density, error, sumw


def plot_reference_vs_weighted_with_pull(
    ref_values,
    mix_values,
    mix_weights,
    bin_edges,
    xlabel,
    title,
    outfile,
    ref_label="Reference",
    weighted_label="Weighted data",
):
    """
    Plot:
      - top: reference as step histogram, weighted data as points with uncertainty
      - bottom: pull = (weighted - reference) / sigma_combined

    Both are compared as normalized densities.
    """
    bin_centers, ref_density, ref_err, _ = get_density_and_error(
        ref_values,
        bin_edges,
        weights=None,
    )

    _, weighted_density, weighted_err, _ = get_density_and_error(
        mix_values,
        bin_edges,
        weights=mix_weights,
    )

    xerr = 0.5 * np.diff(bin_edges)

    sigma = np.sqrt(ref_err ** 2 + weighted_err ** 2)
    pull = np.zeros_like(bin_centers, dtype=float)
    valid = sigma > 0
    pull[valid] = (weighted_density[valid] - ref_density[valid]) / sigma[valid]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )

    # Top panel
    ax1.step(
        bin_edges[:-1],
        ref_density,
        where="post",
        linewidth=2.0,
        label=ref_label,
    )

    ax1.errorbar(
        bin_centers,
        weighted_density,
        yerr=weighted_err,
        xerr=xerr,
        fmt="o",
        markersize=4,
        capsize=0,
        linewidth=1.2,
        label=weighted_label,
        zorder=20,
    )

    ax1.set_ylabel("Density")
    ax1.set_title(title)
    ax1.legend()

    # Bottom panel
    ax2.axhline(0.0, linewidth=1.2)
    ax2.axhline(3.0, linestyle="--", linewidth=1.0)
    ax2.axhline(-3.0, linestyle="--", linewidth=1.0)

    ax2.errorbar(
        bin_centers,
        pull,
        yerr=np.ones_like(pull),
        xerr=xerr,
        fmt="o",
        markersize=4,
        capsize=0,
        linewidth=1.0,
    )

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Pull")
    ax2.set_ylim(-5, 5)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    # -----------------------
    # Top panel: data + fit
    # -----------------------
    y_stack = np.zeros_like(bin_centers, dtype=float)

    for label, yvals in components:
        yvals = np.asarray(yvals, dtype=float)

        ax1.fill_between(
            bin_centers,
            y_stack,
            y_stack + yvals,
            step="mid",
            alpha=0.5,
            label=label,
        )
        y_stack = y_stack + yvals

    ax1.errorbar(
        bin_centers,
        counts,
        yerr=yerr,
        xerr=xerr,
        fmt="o",
        markersize=4,
        capsize=0,
        linewidth=1.2,
        label=data_label,
        zorder=20,
    )

    ax1.plot(
        bin_centers,
        y_total,
        linewidth=2.0,
        label="Total model",
        zorder=15,
    )

    ax1.set_ylabel("Events / bin")
    ax1.set_title(title)
    ax1.set_ylim(bottom=0)
    ax1.legend()

    # -----------------------
    # Bottom panel: pull
    # -----------------------
    ax2.axhline(0.0, linewidth=1.2)
    ax2.axhline(3.0, linestyle="--", linewidth=1.0)
    ax2.axhline(-3.0, linestyle="--", linewidth=1.0)

    ax2.errorbar(
        bin_centers,
        pull,
        yerr=np.ones_like(pull),
        xerr=xerr,
        fmt="o",
        markersize=4,
        capsize=0,
        linewidth=1.0,
    )

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Pull")
    ax2.set_ylim(-5, 5)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

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

    df_bkg = df_bkg[(df_bkg["q2"] > 1.1) & (df_bkg["q2"] < 7.0)].copy()
    df_bkg = df_bkg[(df_bkg["mKpi"] < 1.5)].copy()
    df_bkg = df_bkg[(df_bkg["B_mass"] >= 5.170) & (df_bkg["B_mass"] <= 5.500)].copy()

    df_bkg["is_signal"] = 0

    print("Signal B_mass min/max:", df_sig["B_mass"].min(), df_sig["B_mass"].max())
    print("Background B_mass min/max:", df_bkg["B_mass"].min(), df_bkg["B_mass"].max())

    if args.nsig is not None:
        df_sig = df_sig.sample(n=args.nsig, random_state=args.seed)

    if args.nbkg is not None:
        df_bkg = df_bkg.sample(n=args.nbkg, random_state=args.seed)
    # -------------------------------------------------
    # 3) Sample and build a mixed dataset
    # -------------------------------------------------

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
    # 5) Define signal-component PDFs: A0, App, AS
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

    # Pure angular component pdfs
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
            val = float(p.numpy())
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

    # -------------------------------------------------
    # 9) Plot B_mass projection as stacked components
    #    with pull panel
    # -------------------------------------------------
    bin_edges = np.linspace(5.170, 5.500, 51)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Predicted counts per bin (using bin centers)
    y_A0 = (
        np.asarray(pdf_sig_mass.pdf(bin_centers).numpy(), dtype=float).reshape(-1)
        * float(N_A0.numpy())
        * bin_width
    )
    y_App = (
        np.asarray(pdf_sig_mass.pdf(bin_centers).numpy(), dtype=float).reshape(-1)
        * float(N_App.numpy())
        * bin_width
    )
    y_AS = (
        np.asarray(pdf_sig_mass.pdf(bin_centers).numpy(), dtype=float).reshape(-1)
        * float(N_AS.numpy())
        * bin_width
    )
    y_bkg = (
        np.asarray(pdf_bkg_mass.pdf(bin_centers).numpy(), dtype=float).reshape(-1)
        * float(Nbkg.numpy())
        * bin_width
    )

    components_mass = [
        ("Background component", y_bkg),
        ("A0 component", y_A0),
        ("App component", y_App),
        ("AS component", y_AS),
    ]

    plot_mass_fit_with_pull(
        values=df_mix["B_mass"],
        bin_edges=bin_edges,
        components=components_mass,
        xlabel="B_mass",
        title="3D fit: B_mass projection (stacked components)",
        outfile=os.path.join(args.outdir, "B_mass_projection_stacked_with_pull.png"),
        data_label="Mixed data",
    )
    # -------------------------------------------------
    # 9a) Signal-only mass projection with pull panel
    # -------------------------------------------------
    y_sig_only = (
        np.asarray(pdf_sig_mass.pdf(bin_centers).numpy(), dtype=float).reshape(-1)
        * float(Nsig.numpy())
        * bin_width
    )

    components_sig_only = [
        ("Signal component", y_sig_only),
    ]

    plot_mass_fit_with_pull(
        values=df_sig["B_mass"],
        bin_edges=bin_edges,
        components=components_sig_only,
        xlabel="B_mass",
        title="Signal-only mass projection with pull",
        outfile=os.path.join(args.outdir, "B_mass_signal_only_with_pull.png"),
        data_label="Signal-only data",
    )
    # -------------------------------------------------
    # 9b) Plot cosThetaK / cosThetaL projections as stacked components
    # -------------------------------------------------
    plot_angle_projection_stacked(
        df_mix=df_mix,
        var="cosThetaK",
        outdir=args.outdir,
        angles=angles,
        pdfA0_ang=pdfA0_ang,
        pdfApp_ang=pdfApp_ang,
        pdfAS_ang=pdfAS_ang,
        pdf_bkg_k=pdf_bkg_k,
        pdf_bkg_l=pdf_bkg_l,
        N_A0=N_A0,
        N_App=N_App,
        N_AS=N_AS,
        Nbkg=Nbkg,
    )

    plot_angle_projection_stacked(
        df_mix=df_mix,
        var="cosThetaL",
        outdir=args.outdir,
        angles=angles,
        pdfA0_ang=pdfA0_ang,
        pdfApp_ang=pdfApp_ang,
        pdfAS_ang=pdfAS_ang,
        pdf_bkg_k=pdf_bkg_k,
        pdf_bkg_l=pdf_bkg_l,
        N_A0=N_A0,
        N_App=N_App,
        N_AS=N_AS,
        Nbkg=Nbkg,
    )

    # -------------------------------------------------
    # 10) Compute sWeights for A0 / App / AS / background
    # -------------------------------------------------
    sweight_model = zfit.pdf.SumPDF([pdf_A0, pdf_App, pdf_AS, pdf_bkg])
    sweights = compute_sweights(sweight_model, data)

    print("\nsWeight keys:")
    print(sweights.keys())

    df_mix["wA0"] = np.asarray(sweights[N_A0])
    df_mix["wApp"] = np.asarray(sweights[N_App])
    df_mix["wAS"] = np.asarray(sweights[N_AS])
    df_mix["wbkg"] = np.asarray(sweights[Nbkg])

    # -------------------------------------------------
    # 11) Plot weighted component shapes
    # -------------------------------------------------
    for var in ["cosThetaK", "cosThetaL"]:
        nbins = 100
        xmin, xmax = -1.0, 1.0
        bin_edges = np.linspace(xmin, xmax, nbins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        # weighted histograms for each component
        h_A0, _ = np.histogram(
            df_mix[var],
            bins=bin_edges,
            weights=df_mix["wA0"],
        )
        h_App, _ = np.histogram(
            df_mix[var],
            bins=bin_edges,
            weights=df_mix["wApp"],
        )
        h_AS, _ = np.histogram(
            df_mix[var],
            bins=bin_edges,
            weights=df_mix["wAS"],
        )

        # approximate variances for weighted histograms
        v_A0, _ = np.histogram(
            df_mix[var],
            bins=bin_edges,
            weights=df_mix["wA0"] ** 2,
        )
        v_App, _ = np.histogram(
            df_mix[var],
            bins=bin_edges,
            weights=df_mix["wApp"] ** 2,
        )
        v_AS, _ = np.histogram(
            df_mix[var],
            bins=bin_edges,
            weights=df_mix["wAS"] ** 2,
        )

        # truth total signal for reference
        h_truth, _ = np.histogram(
            df_mix.loc[df_mix["is_signal"] == 1, var],
            bins=bin_edges,
        )

        # sum of weighted components
        h_sum = h_A0 + h_App + h_AS

        plt.figure(figsize=(8, 5))

        # optional truth total signal line
        plt.step(
            bin_edges[:-1],
            h_truth,
            where="post",
            linewidth=2.0,
            label="Truth total signal",
            zorder=20,
        )

        y_stack = np.zeros(nbins)

        components = [
            ("A0 component", h_A0, v_A0),
            ("App component", h_App, v_App),
            ("AS component", h_AS, v_AS),
        ]

        for label, hvals, hvars in components:
            for k in range(nbins):
                plt.fill_between(
                    bin_edges[k:k+2],
                    y1=y_stack[k],
                    y2=y_stack[k] + hvals[k],
                    alpha=0.5,
                    linewidth=0,
                    zorder=0,
                )

            plt.errorbar(
                bin_centers,
                y_stack + hvals,
                yerr=np.sqrt(hvars),
                xerr=np.full(nbins, bin_width / 2.0),
                fmt=".",
                elinewidth=1,
                label=label,
                zorder=10,
            )

            y_stack = y_stack + hvals

        # sum of weighted components
        plt.step(
            bin_edges[:-1],
            h_sum,
            where="post",
            linewidth=2.0,
            label="Sum of weighted components",
            zorder=15,
        )

        plt.xlabel(var)
        plt.ylabel("Weighted events / bin")
        plt.title(f"sWeighted component shapes: {var}")
        plt.xlim(xmin, xmax)
        plt.ylim(bottom=0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.outdir, f"{var}_sweighted_component_stacked.png"),
            dpi=300,
        )
        plt.close()

    # # -------------------------------------------------
    # # 12) Save mixed dataset with truth label and sWeights
    # # -------------------------------------------------
    # df_mix.to_hdf(
    #     os.path.join(args.outdir, "mixed_sample_with_component_sweights.h5"),
    #     key="data",
    #     mode="w",
    # )

    print(f"\nSaved results to: {args.outdir}")
    # print("  - fit_results_signal_background_3d.json")
    print("  - B_mass_projection_stacked.png")
    print("  - cosThetaK_projection_stacked.png")
    print("  - cosThetaL_projection_stacked.png")
    print("  - cosThetaK_component_stacked.png")
    print("  - cosThetaL_component_stacked.png")
    print("  - mixed_sample_with_component_sweights.h5")
    
    # -------------------------------------------------
    # 13) Compare weighted q2 / mKpi shapes to reference samples
    #     reference = step histogram
    #     weighted data = points with uncertainty
    #     add pull panel
    # -------------------------------------------------
    datadir = os.environ["DATADIR"]

    reference_check_dir = os.path.join(args.outdir, "reference_checks")
    os.makedirs(reference_check_dir, exist_ok=True)

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

        ref = tree.arrays(
            ["mKpi", "q2"],
            library="pd",
        ).dropna()

        ref = ref[(ref["q2"] > 1.1) & (ref["q2"] < 7.0)].copy()
        ref = ref[(ref["mKpi"] < 1.5)].copy()

        for var in variables:
            if var == "mKpi":
                bin_edges = np.linspace(ref[var].min(), ref[var].max(), 51)
            elif var == "q2":
                bin_edges = np.linspace(ref[var].min(), ref[var].max(), 51)
            else:
                raise ValueError(f"Unexpected variable: {var}")

            mix_mask = df_mix[var].notna()

            plot_reference_vs_weighted_with_pull(
                ref_values=ref[var].to_numpy(),
                mix_values=df_mix.loc[mix_mask, var].to_numpy(),
                mix_weights=df_mix.loc[mix_mask, weight_name].to_numpy(),
                bin_edges=bin_edges,
                xlabel=var,
                title=f"Reference check: {weight_name} vs {rootfile} in {var}",
                outfile=os.path.join(
                    reference_check_dir,
                    f"{weight_name}_{var}_reference_check_with_pull.png",
                ),
                ref_label=f"Reference {rootfile}",
                weighted_label=f"Mixed sample weighted by {weight_name}",
            )
    # -------------------------------------------------
    # 14) Compare weighted background shapes to pure background sample
    #     reference = step histogram
    #     weighted data = points with uncertainty
    #     add pull panel
    # -------------------------------------------------
    bkg_variables = ["B_mass", "cosThetaK", "cosThetaL", "mKpi", "q2"]

    for var in bkg_variables:
        if var == "B_mass":
            bin_edges = np.linspace(5.170, 5.500, 51)
        elif var in ["cosThetaK", "cosThetaL"]:
            bin_edges = np.linspace(-1.0, 1.0, 51)
        elif var == "mKpi":
            bin_edges = np.linspace(df_bkg[var].min(), df_bkg[var].max(), 51)
        elif var == "q2":
            bin_edges = np.linspace(df_bkg[var].min(), df_bkg[var].max(), 51)
        else:
            raise ValueError(f"Unexpected variable: {var}")

        ref_mask = df_bkg[var].notna()
        mix_mask = df_mix[var].notna()

        plot_reference_vs_weighted_with_pull(
            ref_values=df_bkg.loc[ref_mask, var].to_numpy(),
            mix_values=df_mix.loc[mix_mask, var].to_numpy(),
            mix_weights=df_mix.loc[mix_mask, "wbkg"].to_numpy(),
            bin_edges=bin_edges,
            xlabel=var,
            title=f"Reference check: wbkg vs background in {var}",
            outfile=os.path.join(
                reference_check_dir,
                f"wbkg_{var}_reference_check_with_pull.png",
            ),
            ref_label=f"Reference background in {var}",
            weighted_label="Mixed sample weighted by wbkg",
        )
    # # -------------------------------------------------
    # # 10) Compute COW weights (Im = 1) for total signal vs background
    # # -------------------------------------------------
    # # For COW, we use B_mass as the discriminant variable.
    # # The total signal component is A0 + App + AS, which all share the same
    # # fitted signal mass pdf in this script.

    # mass_values = df_mix["B_mass"].to_numpy(dtype=float)
    # mass_range = (5.170, 5.500)

    # def sig_mass_pdf_for_cow(m):
    #     m_arr = np.asarray(m, dtype=float)

    #     # Scalar input: used when scipy.integrate.quad evaluates the pdf
    #     if m_arr.ndim == 0:
    #         vals = pdf_sig_mass.pdf(np.array([m_arr.item()]), norm=mass).numpy()
    #         return float(np.asarray(vals, dtype=float).reshape(-1)[0])

    #     # Array input: used when COW evaluates weights for many events at once
    #     vals = pdf_sig_mass.pdf(m_arr, norm=mass).numpy()
    #     return np.asarray(vals, dtype=float).reshape(-1)

    # def bkg_mass_pdf_for_cow(m):
    #     m_arr = np.asarray(m, dtype=float)

    #     # Scalar input: used when scipy.integrate.quad evaluates the pdf
    #     if m_arr.ndim == 0:
    #         vals = pdf_bkg_mass.pdf(np.array([m_arr.item()]), norm=mass).numpy()
    #         return float(np.asarray(vals, dtype=float).reshape(-1)[0])

    #     # Array input: used when COW evaluates weights for many events at once
    #     vals = pdf_bkg_mass.pdf(m_arr, norm=mass).numpy()
    #     return np.asarray(vals, dtype=float).reshape(-1)

    # cw = Cow(
    #     mass_range,
    #     sig_mass_pdf_for_cow,
    #     bkg_mass_pdf_for_cow,
    #     Im=None,
    #     verbose=True,
    # )

    # df_mix["wsig_cow"] = np.asarray(cw.get_weight(0, mass_values), dtype=float)
    # df_mix["wbkg_cow"] = np.asarray(cw.get_weight(1, mass_values), dtype=float)

    # print("\nCOW (Im=1) weight summary:")
    # print("  Sum wsig_cow =", df_mix["wsig_cow"].sum())
    # print("  Sum wbkg_cow =", df_mix["wbkg_cow"].sum())
    # print("  Truth signal yield =", (df_mix["is_signal"] == 1).sum())
    # print("  Truth background yield =", (df_mix["is_signal"] == 0).sum())

    # # -------------------------------------------------
    # # 12) Save mixed dataset with truth label and COW weights
    # # -------------------------------------------------
    # df_mix.to_hdf(
    #     os.path.join(args.outdir, "mixed_sample_with_cow_weights.h5"),
    #     key="data",
    #     mode="w",
    # )

    # # -------------------------------------------------
    # # 13) Compare weighted shapes to truth signal
    # # -------------------------------------------------
    # reference_check_dir = os.path.join(args.outdir, "closure_checks")
    # os.makedirs(reference_check_dir, exist_ok=True)

    # for var in ["B_mass", "cosThetaK", "cosThetaL", "mKpi", "q2"]:
    #     plt.figure(figsize=(7, 5))

    #     plt.hist(
    #         df_mix.loc[df_mix["is_signal"] == 1, var].dropna(),
    #         bins=50,
    #         density=True,
    #         histtype="step",
    #         label="Truth signal",
    #     )

    #     plt.hist(
    #         df_mix[var].dropna(),
    #         bins=50,
    #         weights=df_mix.loc[df_mix[var].notna(), "wsig_cow"],
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
    #         os.path.join(args.outdir, f"{var}_cow_closure_check.png"),
    #         dpi=300,
    #     )
    #     plt.close()

    # # -------------------------------------------------
    # # 14) Compare weighted background shapes to pure background sample
    # # -------------------------------------------------
    # bkg_variables = ["B_mass", "cosThetaK", "cosThetaL", "mKpi", "q2"]

    # for var in bkg_variables:
    #     plt.figure(figsize=(7, 5))

    #     plt.hist(
    #         df_bkg[var].dropna(),
    #         bins=50,
    #         density=True,
    #         histtype="step",
    #         label=f"Reference background in {var}",
    #     )

    #     plt.hist(
    #         df_mix[var].dropna(),
    #         bins=50,
    #         weights=df_mix.loc[df_mix[var].notna(), "wbkg_cow"],
    #         density=True,
    #         histtype="step",
    #         label="Mixed sample weighted by wbkg_cow",
    #     )

    #     plt.xlabel(var)
    #     plt.ylabel("Density")
    #     plt.title(f"COW background closure: {var}")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(
    #         os.path.join(args.outdir, f"wbkg_cow_{var}_closure_check.png"),
    #         dpi=300,
    #     )
    #     plt.close()

    # # -------------------------------------------------
    # # 14b) Build a 3D-informed 1D score for signal vs background
    # #      and run 2-component COW on that score
    # # -------------------------------------------------
    # # Total signal 3D pdf = A0 + App + AS (without background)
    # pdf_sig_total_shape = zfit.pdf.SumPDF(
    #     [
    #         pdf_A0_shape.create_extended(N_A0),
    #         pdf_App_shape.create_extended(N_App),
    #         pdf_AS_shape.create_extended(N_AS),
    #     ]
    # )

    # pts3d = df_mix[["B_mass", "cosThetaK", "cosThetaL"]].to_numpy(dtype=float)

    # sig_vals = np.asarray(
    #     pdf_sig_total_shape.pdf(pts3d, norm=obs).numpy(),
    #     dtype=float,
    # ).reshape(-1)

    # bkg_vals = np.asarray(
    #     pdf_bkg_shape.pdf(pts3d, norm=obs).numpy(),
    #     dtype=float,
    # ).reshape(-1)

    # eps = 1e-12
    # sig_vals = np.clip(sig_vals, eps, None)
    # bkg_vals = np.clip(bkg_vals, eps, None)

    # # likelihood-ratio-like score from the 3D model
    # df_mix["score_3d"] = np.log(sig_vals / bkg_vals)

    # print("\n3D score summary:")
    # print("  min =", df_mix["score_3d"].min())
    # print("  max =", df_mix["score_3d"].max())
    # print("  mean(signal truth) =", df_mix.loc[df_mix["is_signal"] == 1, "score_3d"].mean())
    # print("  mean(background truth) =", df_mix.loc[df_mix["is_signal"] == 0, "score_3d"].mean())

    # # -------------------------------------------------
    # # 14c) Plot truth score distributions
    # # -------------------------------------------------
    # plt.figure(figsize=(7, 5))

    # plt.hist(
    #     df_mix.loc[df_mix["is_signal"] == 1, "score_3d"].dropna(),
    #     bins=100,
    #     density=True,
    #     histtype="step",
    #     label="Truth signal score",
    # )

    # plt.hist(
    #     df_mix.loc[df_mix["is_signal"] == 0, "score_3d"].dropna(),
    #     bins=100,
    #     density=True,
    #     histtype="step",
    #     label="Truth background score",
    # )

    # plt.xlabel("score_3d = log(pdf_sig_total_3d / pdf_bkg_3d)")
    # plt.ylabel("Density")
    # plt.title("Truth score distributions from 3D model")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(
    #     os.path.join(args.outdir, "score_3d_truth_distributions.png"),
    #     dpi=300,
    # )
    # plt.close()

    # # -------------------------------------------------
    # # 14d) Build simple 1D score pdfs from truth histograms
    # #      and run 2-component COW on score_3d
    # # -------------------------------------------------
    # score_sig_truth = df_mix.loc[df_mix["is_signal"] == 1, "score_3d"].to_numpy(dtype=float)
    # score_bkg_truth = df_mix.loc[df_mix["is_signal"] == 0, "score_3d"].to_numpy(dtype=float)
    # score_all = df_mix["score_3d"].to_numpy(dtype=float)

    # score_min = float(np.min(score_all))
    # score_max = float(np.max(score_all))

    # # small padding so the integration range is not exactly on the edge
    # pad = 0.02 * (score_max - score_min) if score_max > score_min else 1.0
    # score_range = (score_min - pad, score_max + pad)

    # nbins_score = 300
    # hist_sig, edges_score = np.histogram(
    #     score_sig_truth,
    #     bins=nbins_score,
    #     range=score_range,
    #     density=True,
    # )
    # hist_bkg, _ = np.histogram(
    #     score_bkg_truth,
    #     bins=nbins_score,
    #     range=score_range,
    #     density=True,
    # )

    # centers_score = 0.5 * (edges_score[:-1] + edges_score[1:])

    # def score_sig_pdf_for_cow(x):
    #     x_arr = np.asarray(x, dtype=float)

    #     vals = np.interp(
    #         x_arr,
    #         centers_score,
    #         hist_sig,
    #         left=0.0,
    #         right=0.0,
    #     )

    #     vals = np.clip(vals, 0.0, None)

    #     if np.ndim(x_arr) == 0:
    #         return float(vals)
    #     return np.asarray(vals, dtype=float)

    # def score_bkg_pdf_for_cow(x):
    #     x_arr = np.asarray(x, dtype=float)

    #     vals = np.interp(
    #         x_arr,
    #         centers_score,
    #         hist_bkg,
    #         left=0.0,
    #         right=0.0,
    #     )

    #     vals = np.clip(vals, 0.0, None)

    #     if np.ndim(x_arr) == 0:
    #         return float(vals)
    #     return np.asarray(vals, dtype=float)

    # cw_score = Cow(
    #     score_range,
    #     score_sig_pdf_for_cow,
    #     score_bkg_pdf_for_cow,
    #     1,
    #     verbose=True,
    # )

    # df_mix["wsig_cow_score3d"] = np.asarray(
    #     cw_score.get_weight(0, score_all),
    #     dtype=float,
    # )
    # df_mix["wbkg_cow_score3d"] = np.asarray(
    #     cw_score.get_weight(1, score_all),
    #     dtype=float,
    # )

    # print("\n3D-score COW (Im=1) weight summary:")
    # print("  Sum wsig_cow_score3d =", df_mix["wsig_cow_score3d"].sum())
    # print("  Sum wbkg_cow_score3d =", df_mix["wbkg_cow_score3d"].sum())
    # print("  Truth signal yield   =", (df_mix["is_signal"] == 1).sum())
    # print("  Truth background yield =", (df_mix["is_signal"] == 0).sum())

    # # -------------------------------------------------
    # # 14e) Closure plots using the 3D-score-based COW
    # # -------------------------------------------------
    # score3d_plot_dir = os.path.join(args.outdir, "cow_score3d_plots")
    # os.makedirs(score3d_plot_dir, exist_ok=True)

    # for var in ["cosThetaK", "cosThetaL", "mKpi", "q2", "B_mass"]:
    #     plt.figure(figsize=(7, 5))

    #     plt.hist(
    #         df_mix.loc[df_mix["is_signal"] == 1, var].dropna(),
    #         bins=50,
    #         density=True,
    #         histtype="step",
    #         label="Truth signal",
    #     )

    #     plt.hist(
    #         df_mix[var].dropna(),
    #         bins=50,
    #         weights=df_mix.loc[df_mix[var].notna(), "wsig_cow_score3d"],
    #         density=True,
    #         histtype="step",
    #         label="Mixed sample weighted by 3D-score COW",
    #     )

    #     plt.xlabel(var)
    #     plt.ylabel("Density")
    #     plt.title(f"3D-score COW closure check: {var}")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(
    #         os.path.join(score3d_plot_dir, f"{var}_cow_score3d_closure_check.png"),
    #         dpi=300,
    #     )
    #     plt.close()

    # # -------------------------------------------------
    # # 14f) Background closure plots using the 3D-score-based COW
    # # -------------------------------------------------
    # for var in ["cosThetaK", "cosThetaL", "mKpi", "q2", "B_mass"]:
    #     plt.figure(figsize=(7, 5))

    #     plt.hist(
    #         df_bkg[var].dropna(),
    #         bins=50,
    #         density=True,
    #         histtype="step",
    #         label=f"Reference background in {var}",
    #     )

    #     plt.hist(
    #         df_mix[var].dropna(),
    #         bins=50,
    #         weights=df_mix.loc[df_mix[var].notna(), "wbkg_cow_score3d"],
    #         density=True,
    #         histtype="step",
    #         label="Mixed sample weighted by 3D-score COW",
    #     )

    #     plt.xlabel(var)
    #     plt.ylabel("Density")
    #     plt.title(f"3D-score background closure: {var}")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(
    #         os.path.join(score3d_plot_dir, f"wbkg_cow_score3d_{var}_closure_check.png"),
    #         dpi=300,
    #     )
    #     plt.close()

    # # -------------------------------------------------
    # # 15) Try 4-component COW (A0 / App / AS / background), Im = 1
    # # -------------------------------------------------
    # # WARNING:
    # # A0, App, and AS share the same B_mass signal shape in this script.
    # # Therefore, with B_mass as the only discriminant variable, the 4-component
    # # COW is a numerical test rather than a truly well-separated decomposition.

    # try:
    #     print("\nTrying 4-component COW with Im=1 ...")

    #     A0_val = float(A0_frac.numpy())
    #     App_val = float(App_frac.numpy())
    #     AS_val = float(AS_frac.numpy())

    #     def A0_mass_pdf_for_cow(m):
    #         m_arr = np.asarray(m, dtype=float)

    #         if m_arr.ndim == 0:
    #             vals = pdf_sig_mass.pdf(np.array([m_arr.item()]), norm=mass).numpy()
    #             return float(A0_val * np.asarray(vals, dtype=float).reshape(-1)[0])

    #         vals = pdf_sig_mass.pdf(m_arr, norm=mass).numpy()
    #         return A0_val * np.asarray(vals, dtype=float).reshape(-1)

    #     def App_mass_pdf_for_cow(m):
    #         m_arr = np.asarray(m, dtype=float)

    #         if m_arr.ndim == 0:
    #             vals = pdf_sig_mass.pdf(np.array([m_arr.item()]), norm=mass).numpy()
    #             return float(App_val * np.asarray(vals, dtype=float).reshape(-1)[0])

    #         vals = pdf_sig_mass.pdf(m_arr, norm=mass).numpy()
    #         return App_val * np.asarray(vals, dtype=float).reshape(-1)

    #     def AS_mass_pdf_for_cow(m):
    #         m_arr = np.asarray(m, dtype=float)

    #         if m_arr.ndim == 0:
    #             vals = pdf_sig_mass.pdf(np.array([m_arr.item()]), norm=mass).numpy()
    #             return float(AS_val * np.asarray(vals, dtype=float).reshape(-1)[0])

    #         vals = pdf_sig_mass.pdf(m_arr, norm=mass).numpy()
    #         return AS_val * np.asarray(vals, dtype=float).reshape(-1)

    #     def bkg4_mass_pdf_for_cow(m):
    #         m_arr = np.asarray(m, dtype=float)

    #         if m_arr.ndim == 0:
    #             vals = pdf_bkg_mass.pdf(np.array([m_arr.item()]), norm=mass).numpy()
    #             return float(np.asarray(vals, dtype=float).reshape(-1)[0])

    #         vals = pdf_bkg_mass.pdf(m_arr, norm=mass).numpy()
    #         return np.asarray(vals, dtype=float).reshape(-1)

    #     cw4 = None

    #     try:
    #         cw4 = Cow(
    #             mass_range,
    #             A0_mass_pdf_for_cow,
    #             App_mass_pdf_for_cow,
    #             AS_mass_pdf_for_cow,
    #             bkg4_mass_pdf_for_cow,
    #             1,
    #             verbose=True,
    #         )
    #     except TypeError as e1:
    #         print("\nFirst 4-component Cow constructor failed:")
    #         print(repr(e1))

    #         try:
    #             cw4 = Cow(
    #                 mass_range,
    #                 [
    #                     A0_mass_pdf_for_cow,
    #                     App_mass_pdf_for_cow,
    #                     AS_mass_pdf_for_cow,
    #                     bkg4_mass_pdf_for_cow,
    #                 ],
    #                 1,
    #                 verbose=True,
    #             )
    #         except Exception as e2:
    #             print("\nSecond 4-component Cow constructor also failed:")
    #             print(repr(e2))
    #             raise

    #     df_mix["wA0_cow"] = np.asarray(cw4.get_weight(0, mass_values), dtype=float)
    #     df_mix["wApp_cow"] = np.asarray(cw4.get_weight(1, mass_values), dtype=float)
    #     df_mix["wAS_cow"] = np.asarray(cw4.get_weight(2, mass_values), dtype=float)
    #     df_mix["wbkg4_cow"] = np.asarray(cw4.get_weight(3, mass_values), dtype=float)

    #     print("\n4-component COW (Im=1) weight summary:")
    #     print("  Sum wA0_cow   =", df_mix["wA0_cow"].sum())
    #     print("  Sum wApp_cow  =", df_mix["wApp_cow"].sum())
    #     print("  Sum wAS_cow   =", df_mix["wAS_cow"].sum())
    #     print("  Sum wbkg4_cow =", df_mix["wbkg4_cow"].sum())
    #     print("  Expected N_A0  =", float(N_A0.numpy()))
    #     print("  Expected N_App =", float(N_App.numpy()))
    #     print("  Expected N_AS  =", float(N_AS.numpy()))
    #     print("  Expected Nbkg  =", float(Nbkg.numpy()))

    #     for wname in ["wA0_cow", "wApp_cow", "wAS_cow", "wbkg4_cow"]:
    #         arr = df_mix[wname].to_numpy()
    #         print(f"\n{wname}:")
    #         print("  min   =", np.min(arr))
    #         print("  max   =", np.max(arr))
    #         print("  mean  =", np.mean(arr))
    #         print("  std   =", np.std(arr))

    #     # -------------------------------------------------
    #     # 16) 4-component COW closure checks for angular variables
    #     # -------------------------------------------------
    #     datadir = os.environ["DATADIR"]

    #     reference_mapping_4c = {
    #         "wA0_cow": "A0.root",
    #         "wApp_cow": "A1.root",
    #         "wAS_cow": "AS.root",
    #     }

    #     for weight_name, rootfile in reference_mapping_4c.items():
    #         ref_path = f"{datadir}/{rootfile}"
    #         f = uproot.open(ref_path)
    #         tree = f[f.keys()[0]]

    #         ref = tree.arrays(
    #             ["cosThetaK", "cosThetaL", "mKpi", "q2"],
    #             library="pd",
    #         ).dropna()

    #         ref = ref[(ref["q2"] > 1.1) & (ref["q2"] < 7.0)].copy()
    #         ref = ref[(ref["mKpi"] < 1.5)].copy()

    #         for var in ["cosThetaK", "cosThetaL", "mKpi", "q2"]:
    #             plt.figure(figsize=(7, 5))

    #             plt.hist(
    #                 ref[var],
    #                 bins=50,
    #                 density=True,
    #                 histtype="step",
    #                 label=f"Reference {rootfile}",
    #             )

    #             plt.hist(
    #                 df_mix[var].dropna(),
    #                 bins=50,
    #                 weights=df_mix.loc[df_mix[var].notna(), weight_name],
    #                 density=True,
    #                 histtype="step",
    #                 label=f"Mixed sample weighted by {weight_name}",
    #             )

    #             plt.xlabel(var)
    #             plt.ylabel("Density")
    #             plt.title(f"4-component COW closure: {weight_name} in {var}")
    #             plt.legend()
    #             plt.tight_layout()
    #             plt.savefig(
    #                 os.path.join(args.outdir, f"{weight_name}_{var}_4comp_cow_check.png"),
    #                 dpi=300,
    #             )
    #             plt.close()

    #     # -------------------------------------------------
    #     # 17) 4-component COW background closure
    #     # -------------------------------------------------
    #     for var in ["B_mass", "cosThetaK", "cosThetaL", "mKpi", "q2"]:
    #         plt.figure(figsize=(7, 5))

    #         plt.hist(
    #             df_bkg[var].dropna(),
    #             bins=50,
    #             density=True,
    #             histtype="step",
    #             label=f"Reference background in {var}",
    #         )

    #         plt.hist(
    #             df_mix[var].dropna(),
    #             bins=50,
    #             weights=df_mix.loc[df_mix[var].notna(), "wbkg4_cow"],
    #             density=True,
    #             histtype="step",
    #             label="Mixed sample weighted by wbkg4_cow",
    #         )

    #         plt.xlabel(var)
    #         plt.ylabel("Density")
    #         plt.title(f"4-component COW background closure: {var}")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.savefig(
    #             os.path.join(args.outdir, f"wbkg4_cow_{var}_closure_check.png"),
    #             dpi=300,
    #         )
    #         plt.close()

    #     df_mix.to_hdf(
    #         os.path.join(args.outdir, "mixed_sample_with_cow_weights.h5"),
    #         key="data",
    #         mode="w",
    #     )

    # except Exception as e:
    #     print("\n4-component COW test failed.")
    #     print("Reason:", repr(e))
    #     print("This is still informative because A0/App/AS share the same B_mass shape in this script,")
    #     print("so multi-component COW with B_mass only is not expected to be well separated.")

    #     df_mix.to_hdf(
    #         os.path.join(args.outdir, "mixed_sample_with_cow_weights.h5"),
    #         key="data",
    #         mode="w",
    #     )

    # print("\nSaved additional 4-component COW outputs.")

    # print(f"\nSaved results to: {args.outdir}")
    # print("  - mixed_sample_with_cow_weights.h5")
    # print("  - cosThetaK_cow_closure_check.png")
    # print("  - cosThetaL_cow_closure_check.png")
    # print("  - mKpi_cow_closure_check.png")
    # print("  - q2_cow_closure_check.png")
    # print("  - wbkg_cow_B_mass_closure_check.png")
    # print("  - wbkg_cow_cosThetaK_closure_check.png")
    # print("  - wbkg_cow_cosThetaL_closure_check.png")
    # print("  - wbkg_cow_mKpi_closure_check.png")
    # print("  - wbkg_cow_q2_closure_check.png")

if __name__ == "__main__":
    main()
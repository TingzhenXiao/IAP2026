import os
import numpy as np
import pandas as pd
import zfit
import matplotlib.pyplot as plt
import mplhep
import uproot
import argparse
mplhep.style.use(mplhep.style.LHCb2)

parser = argparse.ArgumentParser()

parser.add_argument("--data", required=True)
parser.add_argument("--background", default=None)
parser.add_argument("--signal-tree", default="B02KstMuMu_Run1_centralQ2E_sig")
parser.add_argument("--background-tree", default="B02KstMuMu_Run1_centralQ2E")

parser.add_argument("--with-bkg", dest="with_bkg", action="store_true", default=False)
parser.add_argument("--no-bkg", dest="with_bkg", action="store_false")

parser.add_argument("--with-eff", dest="with_eff", action="store_true", default=False)
parser.add_argument("--no-eff", dest="with_eff", action="store_false")

parser.add_argument("--qsq", nargs=2, type=float, default=[1.1, 7.0])
parser.add_argument("--run-minos", action="store_true", default=False)
args = parser.parse_args()
# ============================================================
# Input and output
# ============================================================
case_tag = f"{'bkg' if args.with_bkg else 'nobkg'}_{'eff' if args.with_eff else 'noeff'}"
outdir = f"mass_shape_results_{case_tag}"

os.makedirs(outdir, exist_ok=True)


# ============================================================
# Observable
# ============================================================

mass = zfit.Space("B_mass", limits=(5.17, 5.50))


# ============================================================
# Load signal sample
# ============================================================

if args.with_eff:
    efficiency_input = "/home/submit/xiaot425/IAP2026/efficiency/efficiency_applied_output/signal_with_efficiency.h5"
    df_sig = pd.read_hdf(efficiency_input, key="data").copy()
else:
    with uproot.open(args.data) as f:
        df_sig = f[args.signal_tree].arrays(library="pd")

if df_sig["B_mass"].max() > 100.0:
    df_sig["B_mass"] = df_sig["B_mass"] / 1000.0

df_sig = df_sig[(df_sig["q2"] > args.qsq[0]) & (df_sig["q2"] < args.qsq[1])].copy()
df_sig = df_sig[(df_sig["mKpi"] < 1.5)].copy()
df_sig = df_sig[(df_sig["B_mass"] >= 5.170) & (df_sig["B_mass"] <= 5.500)].copy()
df_sig.dropna(inplace=True)

if args.with_eff:
    if "eff_max" in df_sig.columns:
        eff_max = float(df_sig["eff_max"].iloc[0])
    else:
        eff_max = float(df_sig["efficiency"].max())

    df_sig["fit_weight"] = eff_max / df_sig["efficiency"].to_numpy(dtype=float)

else:
    eff_max = 1.0
    df_sig["fit_weight"] = 1.0

print("Signal events after selection:", len(df_sig))
print("Sum of fit weights:", df_sig["fit_weight"].sum())
print("eff_max:", eff_max)

if args.with_bkg:
    background_path = args.background
    if background_path is None:
        background_path = args.data

    with uproot.open(background_path) as f:
        arr_bkg = f[args.background_tree].arrays(
            ["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi"],
            library="np",
        )

    df_bkg = pd.DataFrame(
        {
            "B_mass": arr_bkg["B_mass"],
            "cosThetaK": arr_bkg["cosThetaK"],
            "cosThetaL": arr_bkg["cosThetaL"],
            "q2": arr_bkg["q2"],
            "mKpi": arr_bkg["mKpi"],
        }
    )

    if df_bkg["B_mass"].max() > 100.0:
        df_bkg["B_mass"] = df_bkg["B_mass"] / 1000.0

    df_bkg = df_bkg[(df_bkg["q2"] > args.qsq[0]) & (df_bkg["q2"] < args.qsq[1])].copy()
    df_bkg = df_bkg[df_bkg["mKpi"] < 1.5].copy()
    df_bkg = df_bkg[(df_bkg["B_mass"] >= 5.170) & (df_bkg["B_mass"] <= 5.500)].copy()
    df_bkg.dropna(inplace=True)

    df_bkg["fit_weight"] = 1.0
else:
    df_bkg = pd.DataFrame(columns=["B_mass", "fit_weight"])
# ============================================================
# zfit data
# ============================================================

data_df = pd.concat([df_sig, df_bkg], ignore_index=True)
data_df = data_df.sample(frac=1.0, random_state=0).reset_index(drop=True)

data = zfit.Data.from_pandas(
    data_df[["B_mass"]],
    obs=mass,
    weights=data_df["fit_weight"].to_numpy(dtype=float),
)


# ============================================================
# Signal mass model: two DoubleCB sharing the same mean
# ============================================================

Nsig = zfit.Parameter("Nsig", df_sig["fit_weight"].sum(), 0.0, 1.0e8)
Nbkg = zfit.Parameter("Nbkg", 0.0, 0.0, 1.0e8)
Nbkg.set_value(df_bkg["fit_weight"].sum())
mu_sig = zfit.Parameter("mu_sig", 5.28329, 5.26, 5.30)

sigma_sig_1 = zfit.Parameter("sigma_sig_1", 0.014, 0.006, 0.025)
sigma_sig_2 = zfit.Parameter("sigma_sig_2", 0.021, 0.012, 0.050)

frac_cb1 = zfit.Parameter("frac_cb1", 0.60, 0.0, 1.0)

alphal_1 = zfit.Parameter("alphal_1", 1.8, 0.8, 5.0)
nl_1 = zfit.Parameter("nl_1", 2.0, 1.0, 20.0)
alphar_1 = zfit.Parameter("alphar_1", 2.4, 0.8, 5.0)
nr_1 = zfit.Parameter("nr_1", 2.0, 1.0, 20.0)

alphal_2 = zfit.Parameter("alphal_2", 1.5, 0.8, 5.0)
nl_2 = zfit.Parameter("nl_2", 8.0, 0.1, 20.0)
alphar_2 = zfit.Parameter("alphar_2", 2.0, 0.8, 5.0)
nr_2 = zfit.Parameter("nr_2", 2.0, 1.0, 20.0)

fitpdf_mass_cb1 = zfit.pdf.DoubleCB(
    obs=mass,
    mu=mu_sig,
    sigma=sigma_sig_1,
    alphal=alphal_1,
    nl=nl_1,
    alphar=alphar_1,
    nr=nr_1,
)

fitpdf_mass_cb2 = zfit.pdf.DoubleCB(
    obs=mass,
    mu=mu_sig,
    sigma=sigma_sig_2,
    alphal=alphal_2,
    nl=nl_2,
    alphar=alphar_2,
    nr=nr_2,
)

fitpdf_mass = zfit.pdf.SumPDF(
    [fitpdf_mass_cb1, fitpdf_mass_cb2],
    fracs=frac_cb1,
)

sigpdf = fitpdf_mass.create_extended(Nsig)

lambda_bkg = zfit.Parameter("lambda_bkg", -0.2, -10.0, 0.0)

fitpdf_bkg_mass = zfit.pdf.Exponential(
    obs=mass,
    lambda_=lambda_bkg,
)

bkgpdf = fitpdf_bkg_mass.create_extended(Nbkg)

if args.with_bkg:
    fitpdf = zfit.pdf.SumPDF([sigpdf, bkgpdf])
else:
    Nbkg.set_value(0.0)
    Nbkg.floating = False
    lambda_bkg.floating = False
    fitpdf = sigpdf


# ============================================================
# Plot function
# ============================================================

def eval_pdf_1d(pdf, x):
    return np.asarray(pdf.pdf(x).numpy(), dtype=float).reshape(-1)


def plot_mass_fit(
    data_df,
    fitpdf_mass,
    fitpdf_bkg_mass,
    Nsig,
    Nbkg,
    with_bkg,
    output_path,
):
    nbins_mass = 100
    xmin_mass, xmax_mass = 5.17, 5.50

    x_mass = np.linspace(xmin_mass, xmax_mass, 1000)

    bin_edges = np.linspace(xmin_mass, xmax_mass, nbins_mass + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    weights = data_df["fit_weight"].to_numpy(dtype=float)
    masses = data_df["B_mass"].to_numpy(dtype=float)

    counts, _ = np.histogram(
        masses,
        bins=bin_edges,
        weights=weights,
    )

    sumw2, _ = np.histogram(
        masses,
        bins=bin_edges,
        weights=weights**2,
    )

    yerr = np.sqrt(sumw2)

    nsig_value = float(Nsig.value())
    nbkg_value = float(Nbkg.value()) if with_bkg else 0.0

    sig_y_centers = (
        eval_pdf_1d(fitpdf_mass, bin_centers)
        * nsig_value
        * bin_width
    )

    sig_y_smooth = (
        eval_pdf_1d(fitpdf_mass, x_mass)
        * nsig_value
        * bin_width
    )

    if with_bkg:
        bkg_y_centers = (
            eval_pdf_1d(fitpdf_bkg_mass, bin_centers)
            * nbkg_value
            * bin_width
        )

        bkg_y_smooth = (
            eval_pdf_1d(fitpdf_bkg_mass, x_mass)
            * nbkg_value
            * bin_width
        )
    else:
        bkg_y_centers = np.zeros_like(sig_y_centers)
        bkg_y_smooth = np.zeros_like(sig_y_smooth)

    total_y_centers = sig_y_centers + bkg_y_centers
    total_y_smooth = sig_y_smooth + bkg_y_smooth

    pull = np.zeros_like(bin_centers, dtype=float)
    mask = yerr > 0
    pull[mask] = (counts[mask] - total_y_centers[mask]) / yerr[mask]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    data_label = "Signal + background MC" if with_bkg else "Signal MC"

    ax1.errorbar(
        bin_centers,
        counts,
        yerr=yerr,
        xerr=np.full_like(bin_centers, bin_width / 2.0),
        fmt="o",
        color="black",
        markersize=4,
        linewidth=1.2,
        label=data_label,
    )

    ax1.plot(
        x_mass,
        total_y_smooth,
        color="black",
        linewidth=2,
        label="Total fit",
    )

    ax1.plot(
        x_mass,
        sig_y_smooth,
        color="blue",
        linewidth=2,
        linestyle="--",
        label="Signal component",
    )

    if with_bkg:
        ax1.plot(
            x_mass,
            bkg_y_smooth,
            color="red",
            linewidth=2,
            linestyle=":",
            label="Background component",
        )

    ax1.set_ylabel(fr"Weighted events / {bin_width:.4f}")
    ax1.legend(loc="best", fontsize=13)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.set_xlim(xmin_mass, xmax_mass)

    ymax = 1.25 * np.nanmax(counts + yerr)
    ax1.set_ylim(0.0, ymax)

    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.axhline(2.0, color="black", linestyle=":", linewidth=1.0)
    ax2.axhline(-2.0, color="black", linestyle=":", linewidth=1.0)

    ax2.bar(
        bin_centers,
        pull,
        width=bin_width,
        align="center",
        color="black",
        linewidth=0,
    )

    ax2.set_xlabel(r"$B$ mass [GeV/$c^2$]")
    ax2.set_ylabel("Pull")
    ax2.set_ylim(-5.0, 5.0)
    ax2.tick_params(axis="both", labelsize=12)

    fig.subplots_adjust(
        hspace=0.08,
        left=0.14,
        right=0.97,
        top=0.97,
        bottom=0.10,
    )

    plt.savefig(output_path)
    plt.close()

    print("Saved mass fit plot to:")
    print(output_path)

# ============================================================
# Run fit
# ============================================================

loss = zfit.loss.ExtendedUnbinnedNLL(
    model=fitpdf,
    data=data,
)

minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(loss)
result.update_params()

print(result)


# ============================================================
# Hesse uncertainties
# ============================================================

try:
    hesse_errors = result.hesse()
except Exception as e:
    print("Hesse failed:")
    print(e)
    hesse_errors = {}


# ============================================================
# MINOS uncertainties
# ============================================================

if args.run_minos:
    try:
        minos_errors_raw = result.errors()

        if isinstance(minos_errors_raw, tuple):
            minos_errors = minos_errors_raw[0]
        else:
            minos_errors = minos_errors_raw

    except Exception as e:
        print("MINOS failed:")
        print(e)
        minos_errors = {}
else:
    minos_errors = {}

# ============================================================
# Save fit result text
# ============================================================

result_txt = os.path.join(outdir, "signal_only_mass_fit_result.txt")

with open(result_txt, "w") as f:
    f.write(str(result))
    f.write("\n\n")
    f.write("Hesse uncertainties:\n")
    f.write(str(hesse_errors))
    f.write("\n\n")
    f.write("MINOS uncertainties:\n")
    f.write(str(minos_errors))
    f.write("\n")

print("Saved fit result to:")
print(result_txt)


# ============================================================
# Save parameter table
# ============================================================

params = [
    Nsig,
    mu_sig,
    sigma_sig_1,
    sigma_sig_2,
    frac_cb1,
    alphal_1,
    nl_1,
    alphar_1,
    nr_1,
    alphal_2,
    nl_2,
    alphar_2,
    nr_2,
]

if args.with_bkg:
    params.extend(
        [
            Nbkg,
            lambda_bkg,
        ]
    )

rows = []

for p in params:
    hesse_error = np.nan
    minos_lower = np.nan
    minos_upper = np.nan

    if p in hesse_errors:
        hesse_error = hesse_errors[p].get("error", np.nan)

    if p in minos_errors:
        minos_lower = minos_errors[p].get("lower", np.nan)
        minos_upper = minos_errors[p].get("upper", np.nan)

    rows.append(
        {
            "name": p.name,
            "value": float(p.value()),
            "hesse_error": hesse_error,
            "minos_lower": minos_lower,
            "minos_upper": minos_upper,
            "at_limit": bool(p.at_limit),
        }
    )

df_params = pd.DataFrame(rows)

params_csv = os.path.join(outdir, "signal_only_mass_fit_parameters.csv")
df_params.to_csv(params_csv, index=False)

print("Saved parameter table to:")
print(params_csv)


# ============================================================
# Save tail parameters to fix later
# ============================================================

tail_params = [
    alphal_1,
    nl_1,
    alphar_1,
    nr_1,
    alphal_2,
    nl_2,
    alphar_2,
    nr_2,
]

tail_txt = os.path.join(outdir, "signal_only_mass_fit_tail_parameters.txt")

with open(tail_txt, "w") as f:
    f.write("Tail parameters to fix later in the signal + background fit:\n")
    for p in tail_params:
        line = f"{p.name} = {float(p.value())}\n"
        f.write(line)
        print(line, end="")

print("Saved tail parameters to:")
print(tail_txt)


# ============================================================
# Save plot
# ============================================================

plot_mass_fit(
    data_df=data_df,
    fitpdf_mass=fitpdf_mass,
    fitpdf_bkg_mass=fitpdf_bkg_mass,
    Nsig=Nsig,
    Nbkg=Nbkg,
    with_bkg=args.with_bkg,
    output_path=os.path.join(outdir, "B_mass_fit.pdf"),
)

# ============================================================
# Final summary
# ============================================================

print("\nFinished 1D signal mass fit.")
print("Outputs saved in:")
print(outdir)
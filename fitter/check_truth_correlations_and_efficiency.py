import os
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import mplhep

import sys
sys.path.append("/home/submit/xiaot425/IAP2026/efficiency")
import efficiency

mplhep.style.use(mplhep.style.LHCb2)

OUTDIR = "truth_checks"
EPS = 1.0e-8


def read_truth_signal():
    with uproot.open("/ceph/submit/data/user/a/anbeck/B2KPiMM_michele/full.root") as f:
        tree = f["B02KstMuMu_Run1_centralQ2E_sig"]
        df = tree.arrays(
            ["cosThetaK", "cosThetaL", "mKpi", "q2", "B_mass"],
            library="pd",
        )

    df = df.dropna().copy()

    df["cosh"] = df["cosThetaK"]
    df["cosl"] = df["cosThetaL"]

    if df["B_mass"].max() > 100.0:
        df["B_mass"] = df["B_mass"] / 1000.0

    df = df[(df["q2"] > 1.1) & (df["q2"] < 7.0)].copy()
    df = df[df["mKpi"] < 1.5].copy()
    df = df[(df["B_mass"] >= 5.170) & (df["B_mass"] <= 5.500)].copy()

    return df.reset_index(drop=True)


def plot_correlation_matrix(corr, output_path):
    names = list(corr.columns)

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(corr.to_numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(names, fontsize=14)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson correlation")

    ax.set_title("Truth signal-only correlations")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_2d(df, xvar, yvar, xlabel, ylabel, output_path, bins=80):
    fig, ax = plt.subplots(figsize=(8, 6))

    h = ax.hist2d(
        df[xvar],
        df[yvar],
        bins=bins,
        cmap="viridis",
    )

    fig.colorbar(h[3], ax=ax, label="Entries")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def check_efficiency_before_clipping(df):
    eff_raw = efficiency.efficiency(
        df["cosh"].to_numpy(dtype=float),
        df["cosl"].to_numpy(dtype=float),
        df["mKpi"].to_numpy(dtype=float),
        df["q2"].to_numpy(dtype=float),
    )

    eff_raw = np.asarray(eff_raw, dtype=float)

    print("\n[Efficiency check before clipping]")
    print("Number of events:", len(eff_raw))
    print("min raw efficiency:", np.nanmin(eff_raw))
    print("max raw efficiency:", np.nanmax(eff_raw))
    print("number NaN:", np.sum(np.isnan(eff_raw)))
    print("number +inf:", np.sum(np.isposinf(eff_raw)))
    print("number -inf:", np.sum(np.isneginf(eff_raw)))
    print("number eff < 1e-8:", np.sum(eff_raw < EPS))
    print("number eff <= 0:", np.sum(eff_raw <= 0.0))

    if np.sum(eff_raw < EPS) == 0:
        print("PASS: no raw efficiency values below 1e-8 before clipping.")
    else:
        print("WARNING: some raw efficiency values are below 1e-8 before clipping.")

    return eff_raw


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    df = read_truth_signal()

    print("Truth signal-only events after cuts:", len(df))

    cols = ["cosh", "cosl", "B_mass", "q2", "mKpi"]
    corr = df[cols].corr(method="pearson")

    print("\n[Correlation matrix]")
    print(corr)

    corr_csv = os.path.join(OUTDIR, "truth_signal_correlations.csv")
    corr_txt = os.path.join(OUTDIR, "truth_signal_correlations.txt")
    corr_pdf = os.path.join(OUTDIR, "truth_signal_correlations.pdf")

    corr.to_csv(corr_csv)

    with open(corr_txt, "w") as f:
        f.write(corr.to_string())

    plot_correlation_matrix(corr, corr_pdf)

    plot_2d(
        df,
        "cosl",
        "q2",
        r"$\cos\theta_\ell$",
        r"$q^2$ [GeV$^2/c^4$]",
        os.path.join(OUTDIR, "truth_2d_cosl_q2.pdf"),
    )

    plot_2d(
        df,
        "mKpi",
        "cosh",
        r"$m(K\pi)$ [GeV/$c^2$]",
        r"$\cos\theta_K$",
        os.path.join(OUTDIR, "truth_2d_mKpi_cosh.pdf"),
    )

    check_efficiency_before_clipping(df)

    print("\nSaved outputs:")
    print(corr_csv)
    print(corr_txt)
    print(corr_pdf)
    print(os.path.join(OUTDIR, "truth_2d_cosl_q2.pdf"))
    print(os.path.join(OUTDIR, "truth_2d_mKpi_cosh.pdf"))


if __name__ == "__main__":
    main()
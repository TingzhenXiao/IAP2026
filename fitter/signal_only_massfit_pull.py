import os
import json
import argparse
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import zfit
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

def plot_projection_with_pull(
    bin_edges,
    bin_centers,
    data_y,
    data_yerr,
    pull,
    xlabel,
    ylabel,
    output_path,
    data_label="Data",
    line_x=None,
    total_y=None,
    total_label="Fit",
    xlim=None,
    ylim_pull=(-5, 5),
    legend_fontsize=12,
    dpi=300,
):
    bin_width = bin_edges[1] - bin_edges[0]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )

    ax1.errorbar(
        bin_centers,
        data_y,
        yerr=data_yerr,
        xerr=np.full_like(bin_centers, bin_width / 2.0),
        fmt="o",
        color="black",
        markersize=4,
        linewidth=1.2,
        label=data_label,
        zorder=10,
    )

    if line_x is not None and total_y is not None:
        ax1.plot(
            line_x,
            total_y,
            color="black",
            linewidth=2,
            label=total_label,
        )

    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=10, width=2)
    ax1.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=6, width=2)
    ax1.minorticks_on()
    ax1.legend(fontsize=legend_fontsize, loc="best", handlelength=1.5)
    

    if xlim is not None:
        ax1.set_xlim(*xlim)

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

    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.set_ylabel("Pull", fontsize=14)
    ax2.set_ylim(*ylim_pull)
    ax2.tick_params(axis="both", which="major", labelsize=12)
    ax2.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=10, width=2)
    ax2.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=6, width=2)
    ax2.minorticks_on()

    fig.subplots_adjust(
        hspace=0.0,
        left=0.14,
        right=0.97,
        top=0.97,
        bottom=0.10,
    )

    plt.savefig(output_path, dpi=dpi)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Fit signal B_mass only with DoubleCB."
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
        "--outdir",
        type=str,
        default="fit_signal_bmass_doublecb_output",
        help="Output directory",
    )
    parser.add_argument(
        "--nmax",
        type=int,
        default=None,
        help="Maximum number of signal events to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------------------------------
    # 1) Read signal sample
    # -------------------------------------------------
    if args.signal.endswith(".h5") or args.signal.endswith(".hdf5"):
        df_sig = pd.read_hdf(args.signal, key="data")
    else:
        with uproot.open(args.signal) as f:
            tree = f[args.signal_tree]
            df_sig = tree.arrays(
                ["B_mass", "q2", "mKpi"],
                library="pd",
            )

    df_sig = df_sig.dropna()

    print("Before conversion B_mass min/max:", df_sig["B_mass"].min(), df_sig["B_mass"].max())
    # Convert MeV -> GeV if needed
    df_sig["B_mass"] = df_sig["B_mass"] / 1000.0

    # Same cuts as your main script
    df_sig = df_sig[(df_sig["q2"] > 1.1) & (df_sig["q2"] < 7.0)].copy()
    df_sig = df_sig[(df_sig["mKpi"] < 1.5)].copy()
    df_sig = df_sig[(df_sig["B_mass"] >= 5.170) & (df_sig["B_mass"] <= 5.500)].copy()

    if args.nmax is not None and args.nmax < len(df_sig):
        df_sig = df_sig.sample(n=args.nmax, random_state=args.seed).reset_index(drop=True)

    print("Number of signal events used:", len(df_sig))
    print("B_mass min/max:", df_sig["B_mass"].min(), df_sig["B_mass"].max())

    # -------------------------------------------------
    # 2) Define observable and data
    # -------------------------------------------------
    mass = zfit.Space("B_mass", limits=(5.170, 5.500))
    data = zfit.Data.from_numpy(
        obs=mass,
        array=df_sig["B_mass"].to_numpy(),
    )

    # -------------------------------------------------
    # 3) Define DoubleCB model
    # -------------------------------------------------
    mu = zfit.Parameter("mu", 5.279, 5.26, 5.30)
    sigma = zfit.Parameter("sigma", 0.025, 0.003, 0.080)

    alphal = zfit.Parameter("alphal", 1.5, 0.2, 5.0)
    nl = zfit.Parameter("nl", 3.0, 0.5, 30.0)

    alphar = zfit.Parameter("alphar", 1.5, 0.2, 5.0)
    nr = zfit.Parameter("nr", 3.0, 0.5, 30.0)

    pdf = zfit.pdf.DoubleCB(
        obs=mass,
        mu=mu,
        sigma=sigma,
        alphal=alphal,
        nl=nl,
        alphar=alphar,
        nr=nr,
    )

    # -------------------------------------------------
    # 4) Fit
    # -------------------------------------------------
    nll = zfit.loss.UnbinnedNLL(model=pdf, data=data)
    minimizer = zfit.minimize.Minuit()

    result = minimizer.minimize(nll)
    result.hesse()

    print("\n=== Signal-only 1D DoubleCB fit result ===")
    print(result)

    # -------------------------------------------------
    # 5) Collect fit results
    # -------------------------------------------------
    params = [mu, sigma, alphal, nl, alphar, nr]
    fit_results = {}

    for p in params:
        value = float(zfit.run(p))
        error = None

        if p in result.params and "hesse" in result.params[p]:
            error = float(result.params[p]["hesse"]["error"])

        fit_results[p.name] = {
            "value": value,
            "error": error,
        }

    print("\n=== Fitted parameters ===")
    for name, info in fit_results.items():
        print(f"{name:8s} = {info['value']:.6f} +/- {info['error']}")

    # -------------------------------------------------
    # 6) Plot data, fitted PDF, and pull
    # -------------------------------------------------
    x = np.linspace(5.170, 5.500, 1000)
    bin_edges = np.linspace(5.170, 5.500, 101)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    counts, _ = np.histogram(df_sig["B_mass"], bins=bin_edges)
    yerr = np.sqrt(counts)

    y_fit_centers = (
        pdf.pdf(bin_centers, norm=mass).numpy()
        * len(df_sig)
        * bin_width
    )

    pull = np.zeros_like(bin_centers)
    mask = counts > 0
    pull[mask] = (counts[mask] - y_fit_centers[mask]) / np.sqrt(counts[mask])

    y_fit_curve = (
        pdf.pdf(x, norm=mass).numpy()
        * len(df_sig)
        * bin_width
    )

    plot_projection_with_pull(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        data_y=counts,
        data_yerr=yerr,
        pull=pull,
        xlabel=r"$B$ mass [GeV/$c^2$]",
        ylabel=fr"Events / {bin_width:.4f} GeV",
        output_path=os.path.join(args.outdir, "signal_bmass_doublecb_fit_with_pull.pdf"),
        data_label="Signal sample",
        line_x=x,
        total_y=y_fit_curve,
        total_label="DoubleCB fit",
        xlim=(5.170, 5.500),
        dpi=300,
    )

    # -------------------------------------------------
    # 7) Save json
    # -------------------------------------------------
    with open(os.path.join(args.outdir, "signal_bmass_doublecb_fit_results.json"), "w") as f:
        json.dump(fit_results, f, indent=2)

    # -------------------------------------------------
    # 8) Print ready-to-copy fixed parameters for main script
    # -------------------------------------------------
    print("\n=== Copy these into your main script if you want to fix the tails ===")
    print(f"alphal_init = {float(zfit.run(alphal))}")
    print(f"nl_init     = {float(zfit.run(nl))}")
    print(f"alphar_init = {float(zfit.run(alphar))}")
    print(f"nr_init     = {float(zfit.run(nr))}")


if __name__ == "__main__":
    main()
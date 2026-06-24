import os
import json
import yaml
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import mplhep
import sys
from sweights.experimental import Cows

import tools

sys.path.append("/home/submit/xiaot425/IAP2026/efficiency")
import efficiency

mplhep.style.use(mplhep.style.LHCb2)

np.random.seed(0)


PDF_BINS = (12, 12, 20)
PLOT_BINS = 50
N_INT = 800000

XRANGES = [
    (-1.0, 1.0),
    (-1.0, 1.0),
    (5.170, 5.500),
]


REFERENCE_MAPPING = {
    "wA0": ("A0.root", "B02KstMuMu_Run1_centralQ2E_sig"),
    "wApp": ("A1.root", "B02KstMuMu_Run1_centralQ2E_sig"),
    "wS": ("AS.root", "B02KstMuMu_Run1_centralQ2E_sig"),
}

REFERENCE_DISPLAY = {
    "wA0": r"$A_0$ reference",
    "wApp": r"$A_{\parallel,\perp}$ reference",
    "wS": r"$A_S$ reference",
}

COMPONENT_DISPLAY = {
    "wA0": r"$A_0$",
    "wApp": r"$A_{\parallel,\perp}$",
    "wS": r"$A_S$",
}


class HistPdf3D:
    def __init__(self, values, weights=None, bins=PDF_BINS, ranges=XRANGES, floor_fraction=1.0e-8):
        self.ranges = ranges
        self.hist, self.edges = np.histogramdd(
            values,
            bins=bins,
            range=ranges,
            weights=weights,
            density=False,
        )

        self.hist = np.asarray(self.hist, dtype=float)
        self.edges = [np.asarray(edge, dtype=float) for edge in self.edges]

        widths = [np.diff(edge) for edge in self.edges]
        volume = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]

        total = np.sum(self.hist)
        if total <= 0.0:
            raise RuntimeError("Cannot build a 3D PDF from an empty histogram.")

        density = self.hist / total / volume

        positive = density[density > 0.0]
        if len(positive) == 0:
            raise RuntimeError("3D PDF has no positive bins.")

        floor = floor_fraction * np.mean(positive)
        density[density <= 0.0] = floor

        norm = np.sum(density * volume)
        density = density / norm

        self.density = density
        self.shape = density.shape

    def __call__(self, x):
        arr = np.asarray(x, dtype=float)

        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError("Expected one 3D point.")
            pts = arr.reshape(1, 3)

        elif arr.ndim == 2:
            if arr.shape[0] == 3:
                pts = arr.T
            elif arr.shape[1] == 3:
                pts = arr
            else:
                raise ValueError(f"Unexpected input shape: {arr.shape}")

        else:
            raise ValueError(f"Unexpected input ndim: {arr.ndim}")

        idx0 = np.searchsorted(self.edges[0], pts[:, 0], side="right") - 1
        idx1 = np.searchsorted(self.edges[1], pts[:, 1], side="right") - 1
        idx2 = np.searchsorted(self.edges[2], pts[:, 2], side="right") - 1

        idx0 = np.clip(idx0, 0, self.shape[0] - 1)
        idx1 = np.clip(idx1, 0, self.shape[1] - 1)
        idx2 = np.clip(idx2, 0, self.shape[2] - 1)

        return np.asarray(self.density[idx0, idx1, idx2], dtype=float)


def prepare_sample(df):
    df = df.copy()

    if "cosl" not in df.columns:
        df["cosl"] = df["cosThetaL"]

    if "cosh" not in df.columns:
        df["cosh"] = df["cosThetaK"]

    if "B_mass" in df.columns and df["B_mass"].max() > 100.0:
        df["B_mass"] = df["B_mass"] / 1000.0

    df = df[(df["q2"] > 1.1) & (df["q2"] < 7.0)].copy()
    df = df[(df["mKpi"] < 1.5)].copy()
    df = df[(df["B_mass"] >= 5.170) & (df["B_mass"] <= 5.500)].copy()
    df.dropna(inplace=True)

    return df


def read_truth_settings(settings_path):
    if settings_path.endswith(".yml"):
        with open(settings_path) as f:
            truth = yaml.load(f, Loader=yaml.FullLoader)
    else:
        with open(settings_path) as f:
            truth = json.load(f)
            for key in truth:
                truth[key] = {"value": truth[key]}

    return truth


def get_truth_value(truth, name, default):
    if name in truth:
        return float(truth[name]["value"])
    return float(default)


def component_fractions_from_truth(truth):
    A0 = get_truth_value(truth, "A0", 0.423)
    App = get_truth_value(truth, "App", 0.167)
    Aqc = get_truth_value(truth, "Aqc", 0.0)
    Aqs = get_truth_value(truth, "Aqs", 0.0)

    AS = 1.0 - A0 - App - Aqc - Aqs

    fractions = {
        "wA0": A0,
        "wApp": App,
        "wS": AS,
    }

    return fractions


def compute_efficiency_values(df):
    return efficiency.efficiency(
        df["cosh"].to_numpy(dtype=float),
        df["cosl"].to_numpy(dtype=float),
        df["mKpi"].to_numpy(dtype=float),
        df["q2"].to_numpy(dtype=float),
    )
    

def projected_efficiency_2d(ref_df_noeff, xvar, yvar, x_edges, y_edges):
    ref_df_noeff = prepare_sample(ref_df_noeff)

    eff = compute_efficiency_values(ref_df_noeff)

    x = ref_df_noeff[xvar].to_numpy(dtype=float)
    y = ref_df_noeff[yvar].to_numpy(dtype=float)

    numerator, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges],
        weights=eff,
    )

    denominator, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges],
    )

    eps_bar = np.zeros_like(numerator, dtype=float)
    mask = denominator > 0
    eps_bar[mask] = numerator[mask] / denominator[mask]

    positive = eps_bar[eps_bar > 0.0]
    if len(positive) == 0:
        raise RuntimeError("2D projected efficiency is empty.")

    floor = np.min(positive)
    eps_bar[eps_bar <= 0.0] = floor

    return eps_bar

def make_eq41_2d_projected_plot(
    datatoy,
    raw_weights,
    ref_df_noeff,
    var,
    xlabel,
    output_path,
    reference_label,
    data_label,
):
    mKpi_edges = np.linspace(0.65, 1.50, PLOT_BINS + 1)
    q2_edges = np.linspace(1.1, 7.0, PLOT_BINS + 1)

    ref_df_noeff = prepare_sample(ref_df_noeff)

    data_mKpi = datatoy["mKpi"].to_numpy(dtype=float)
    data_q2 = datatoy["q2"].to_numpy(dtype=float)

    ref_mKpi = ref_df_noeff["mKpi"].to_numpy(dtype=float)
    ref_q2 = ref_df_noeff["q2"].to_numpy(dtype=float)

    raw_weights = np.asarray(raw_weights, dtype=float)

    h_obs_2d, _, _ = np.histogram2d(
        data_mKpi,
        data_q2,
        bins=[mKpi_edges, q2_edges],
        weights=raw_weights,
    )

    h_obs_var_2d, _, _ = np.histogram2d(
        data_mKpi,
        data_q2,
        bins=[mKpi_edges, q2_edges],
        weights=raw_weights**2,
    )

    eps_bar_2d = projected_efficiency_2d(
        ref_df_noeff=ref_df_noeff,
        xvar="mKpi",
        yvar="q2",
        x_edges=mKpi_edges,
        y_edges=q2_edges,
    )

    h_eq41_2d = h_obs_2d / eps_bar_2d
    err_eq41_2d = np.sqrt(h_obs_var_2d) / eps_bar_2d

    if var == "mKpi":
        bin_edges = mKpi_edges
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        h_eq41 = np.sum(h_eq41_2d, axis=1)
        err_eq41 = np.sqrt(np.sum(err_eq41_2d**2, axis=1))

        h_ref, _ = np.histogram(ref_mKpi, bins=bin_edges)

    elif var == "q2":
        bin_edges = q2_edges
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        h_eq41 = np.sum(h_eq41_2d, axis=0)
        err_eq41 = np.sqrt(np.sum(err_eq41_2d**2, axis=0))

        h_ref, _ = np.histogram(ref_q2, bins=bin_edges)

    else:
        raise ValueError(f"Unknown variable: {var}")

    err_ref = np.sqrt(h_ref.astype(float))

    if h_ref.sum() <= 0 or np.isclose(h_eq41.sum(), 0.0):
        return

    h_ref = h_ref / h_ref.sum()
    err_ref = err_ref / len(ref_df_noeff)

    norm = h_eq41.sum()
    h_eq41 = h_eq41 / norm
    err_eq41 = err_eq41 / abs(norm)

    sigma_pull = np.sqrt(err_ref**2 + err_eq41**2)
    pull = np.zeros_like(bin_centers, dtype=float)

    mask_pull = sigma_pull > 0.0
    pull[mask_pull] = (h_eq41[mask_pull] - h_ref[mask_pull]) / sigma_pull[mask_pull]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )

    ax1.step(
        bin_edges[:-1],
        h_ref,
        where="post",
        linewidth=2,
        color="blue",
        label=reference_label,
    )

    ax1.errorbar(
        bin_centers,
        h_eq41,
        yerr=err_eq41,
        xerr=np.full_like(bin_centers, bin_width / 2.0),
        fmt="o",
        color="black",
        markersize=4,
        linewidth=1.2,
        label=data_label,
    )

    ymax = max(np.nanmax(h_ref), np.nanmax(h_eq41 + err_eq41))
    ax1.set_ylim(0.0, 1.2 * ymax)
    ax1.set_xlim(bin_edges[0], bin_edges[-1])
    ax1.set_ylabel("Normalized entries", fontsize=14)
    ax1.legend(fontsize=12, loc="best")
    ax1.tick_params(axis="both", labelsize=12)

    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.axhline(2.0, color="black", linestyle=":", linewidth=1.0)
    ax2.axhline(-2.0, color="black", linestyle=":", linewidth=1.0)
    ax2.bar(bin_centers, pull, width=bin_width, color="black", linewidth=0)
    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.set_ylabel("Pull", fontsize=14)
    ax2.set_ylim(-5.0, 5.0)
    ax2.tick_params(axis="both", labelsize=12)

    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.10)
    plt.savefig(output_path)
    plt.close()

def main():
    args = tools.parser()

    datadir = os.environ["DATADIR"]
    truth = read_truth_settings(args.settings)
    fractions = component_fractions_from_truth(truth)

    efficiency_input = "/home/submit/xiaot425/IAP2026/efficiency/efficiency_applied_output/signal_with_efficiency.h5"

    if os.path.exists(efficiency_input):
        df_sig_obs = pd.read_hdf(efficiency_input, key="data").copy()
    else:
        raise FileNotFoundError(f"Could not find {efficiency_input}")

    df_sig_obs = prepare_sample(df_sig_obs)
    df_sig_obs["is_signal"] = 1
    df_sig_obs["fit_weight"] = 1.0 / df_sig_obs["efficiency"].to_numpy(dtype=float)

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
    })
    df_bkg = prepare_sample(df_bkg)
    df_bkg["is_signal"] = 0
    df_bkg["efficiency"] = 1.0
    df_bkg["fit_weight"] = 1.0

    data = pd.concat([df_sig_obs, df_bkg], ignore_index=True)
    data = data.sample(frac=1.0, random_state=0).reset_index(drop=True)

    print("Observed signal events:", len(df_sig_obs))
    print("Background events:", len(df_bkg))
    print("Mixed observed events:", len(data))

    ref_noeff = {}
    ref_eff_mean = {}
    obs_pdfs = {}

    for weight_name, (ref_file, ref_tree) in REFERENCE_MAPPING.items():
        ref_path = os.path.join(datadir, ref_file)

        with uproot.open(ref_path) as fref:
            ref_df = fref[ref_tree].arrays(library="pd")

        ref_df = prepare_sample(ref_df)
        eff = compute_efficiency_values(ref_df)
        ref_noeff[weight_name] = ref_df
        ref_eff_mean[weight_name] = float(np.mean(eff))

        values = ref_df[["cosh", "cosl", "B_mass"]].to_numpy(dtype=float)
        obs_pdfs[weight_name] = HistPdf3D(
            values=values,
            weights=eff,
            bins=PDF_BINS,
            ranges=XRANGES,
        )

        print(weight_name, "reference N =", len(ref_df), "mean efficiency =", ref_eff_mean[weight_name])

    bkg_values = df_bkg[["cosh", "cosl", "B_mass"]].to_numpy(dtype=float)
    pdf_bkg = HistPdf3D(
        values=bkg_values,
        weights=None,
        bins=PDF_BINS,
        ranges=XRANGES,
    )

    component_names = ["wA0", "wApp", "wS", "wBkg"]
    pdfs = [
        obs_pdfs["wA0"],
        obs_pdfs["wApp"],
        obs_pdfs["wS"],
        pdf_bkg,
    ]

    signal_props = []
    for name in ["wA0", "wApp", "wS"]:
        signal_props.append(fractions[name] * ref_eff_mean[name])

    signal_props = np.asarray(signal_props, dtype=float)
    signal_props = signal_props / np.sum(signal_props)

    signal_observed_yields = len(df_sig_obs) * signal_props

    yields = [
        signal_observed_yields[0],
        signal_observed_yields[1],
        signal_observed_yields[2],
        float(len(df_bkg)),
    ]

    print("\nObserved-level COW yields used in Eq. 41 test:")
    for name, y in zip(component_names, yields):
        print(name, y)

    pdfs_sig_cow = [
        obs_pdfs["wA0"],
        obs_pdfs["wApp"],
        obs_pdfs["wS"],
    ]

    pdfs_bkg_cow = [
        pdf_bkg,
    ]

    cow = Cows(
        sample=None,
        spdf=pdfs_sig_cow,
        bpdf=pdfs_bkg_cow,
        norm=None,
        range=XRANGES,
        summation=False,
        yields=yields,
        integration_options={
            "n_estimates": 4,
            "n_points": 8192,
        },
    )

    W = cow._wm + np.tril(cow._wm, -1).T
    A = cow._am

    print("\nEq. 41 observed-level COW W matrix from sweights package:")
    print(W)
    print("Eq. 41 observed-level COW A matrix from sweights package:")
    print(A)
    print("Eq. 41 observed-level W condition number:")
    print(np.linalg.cond(W))

    data_cow = data[["cosh", "cosl", "B_mass"]].to_numpy(dtype=float).T

    wA0_obs = cow[0](data_cow)
    wApp_obs = cow[1](data_cow)
    wS_obs = cow[2](data_cow)
    wBkg_obs = cow["b"](data_cow)
    w_sum_obs = wA0_obs + wApp_obs + wS_obs + wBkg_obs

    print("\nRaw observed-level COW sums:")
    print("sum wA0 obs  =", np.sum(wA0_obs), "expected", yields[0])
    print("sum wApp obs =", np.sum(wApp_obs), "expected", yields[1])
    print("sum wS obs   =", np.sum(wS_obs), "expected", yields[2])
    print("sum wBkg obs =", np.sum(wBkg_obs), "expected", yields[3])
    print("sum all obs  =", np.sum(w_sum_obs), "N events", len(data))
    print("event-wise sum mean =", np.mean(w_sum_obs))
    print("event-wise sum min  =", np.min(w_sum_obs))
    print("event-wise sum max  =", np.max(w_sum_obs))

    raw_weights = {
        "wA0": wA0_obs,
        "wApp": wApp_obs,
        "wS": wS_obs,
    }

    outdir = "plots/eq41_strict_observed_level_noq2"
    os.makedirs(outdir, exist_ok=True)

    for weight_name in ["wA0", "wApp", "wS"]:
        make_eq41_2d_projected_plot(
            datatoy=data,
            raw_weights=raw_weights[weight_name],
            ref_df_noeff=ref_noeff[weight_name].copy(),
            var="mKpi",
            xlabel=r"$m(K\pi)$ [GeV/$c^2$]",
            output_path=f"{outdir}/0_{weight_name}_mKpi_eq41_2d_projected.pdf",
            reference_label=REFERENCE_DISPLAY[weight_name],
            data_label=COMPONENT_DISPLAY[weight_name] + r" Eq. 41 2D projected efficiency",
        )

        make_eq41_2d_projected_plot(
            datatoy=data,
            raw_weights=raw_weights[weight_name],
            ref_df_noeff=ref_noeff[weight_name].copy(),
            var="q2",
            xlabel=r"$q^2$ [GeV$^2/c^4$]",
            output_path=f"{outdir}/0_{weight_name}_q2_eq41_2d_projected.pdf",
            reference_label=REFERENCE_DISPLAY[weight_name],
            data_label=COMPONENT_DISPLAY[weight_name] + r" Eq. 41 2D projected efficiency",
        )


    print("\nSaved Eq. 41 strict plots to:")
    print(outdir)


if __name__ == "__main__":
    main()

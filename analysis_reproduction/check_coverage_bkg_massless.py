"""
Make 2D coverage plots in (mKpi, q2) for:
    total, wApp, wA0, wS, wBkg

Example:
python check_coverage_bkg.py \
    --input /work/submit/xiaot425/IAP2026/results_cow_bkg_eff_Ig/coverage_toys/*.h5 \
    --name cow_bkg_eff_Ig \
    --data /home/submit/xiaot425/IAP2026/efficiency/efficiency_applied_output/signal_with_efficiency.h5 \
    --background /home/submit/xiaot425/IAP2026/genbkg/background_toy.root \
    --background-tree background \
    --settings settings/App=0.1670_qsq-1.1-7.0.yml \
    --toy-mean 90000
"""

import argparse
import os
import sys

import hist
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd
import uproot
import yaml

mplhep.style.use(mplhep.style.LHCb2)

sys.path.append("/home/submit/xiaot425/IAP2026/efficiency")
import efficiency


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    nargs="+",
    required=True,
    help="Toy HDF5 files containing mKpi, q2 and COW weights.",
)
parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="Output filename prefix.",
)
parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Efficiency-applied signal HDF5 used to build the toy pool.",
)
parser.add_argument(
    "--background",
    type=str,
    required=True,
    help="Background ROOT file used to build the toy pool.",
)
parser.add_argument(
    "--background-tree",
    type=str,
    default="background",
    help="Tree name in the background ROOT file.",
)
parser.add_argument(
    "--settings",
    type=str,
    required=True,
    help="YAML settings file containing the truth angular coefficients.",
)
parser.add_argument(
    "--datadir",
    type=str,
    default=os.environ.get("DATADIR"),
    help="Directory containing A0.root, A1.root and AS.root.",
)
parser.add_argument(
    "--signal-tree",
    type=str,
    default="B02KstMuMu_Run1_centralQ2E_sig",
    help="Tree name in A0.root, A1.root and AS.root.",
)
parser.add_argument(
    "--toy-mean",
    type=float,
    default=90000.0,
    help="Mean number of unweighted events generated per toy.",
)
parser.add_argument(
    "--nbins",
    type=int,
    default=15,
    help="Number of bins on each axis.",
)
parser.add_argument(
    "--mKpi",
    type=float,
    nargs=2,
    default=[0.746, 1.50],
    help="mKpi range.",
)

parser.add_argument(
    "--q2",
    type=float,
    nargs=2,
    default=[1.1, 19.0],
    help="q2 range.",
)
parser.add_argument(
    "--mass",
    type=float,
    nargs=2,
    default=[5.170, 5.500],
    help="B-mass range in GeV.",
)
parser.add_argument(
    "--app",
    type=float,
    default=0.1670,
    help="Fallback truth value of App if it is not stored in the YAML file.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="plots",
    help="Directory in which the PDF plots are written.",
)

args = parser.parse_args()

# if args.datadir is None:
#     raise RuntimeError(
#         "DATADIR is not set. Set DATADIR or provide --datadir explicitly."
#     )

os.makedirs(args.output_dir, exist_ok=True)

nbins = args.nbins
mih, mah = args.mKpi
miq, maq = args.q2
mib, mab = args.mass


def truth_value(settings, name, default=None):
    if name not in settings:
        if default is None:
            raise KeyError(
                f"Truth parameter {name} is missing from {args.settings}."
            )
        return float(default)

    value = settings[name]

    if isinstance(value, dict):
        if "value" not in value:
            if default is None:
                raise KeyError(
                    f"Truth parameter {name} has no 'value' entry."
                )
            return float(default)

        value = value["value"]

    return float(value)


def filter_dataframe(df):
    df = df.copy()

    if "B_mass" in df.columns:
        if df["B_mass"].max() > 100.0:
            df["B_mass"] = df["B_mass"] / 1000.0

        df = df[
            (df["B_mass"] >= mib)
            & (df["B_mass"] <= mab)
        ].copy()

    df = df[
        (df["mKpi"] >= mih)
        & (df["mKpi"] <= mah)
        & (df["q2"] > miq)
        & (df["q2"] < maq)
    ].copy()

    df = df.query(
        "(0.06 < q2 < 9.0) or "
        "(11.0 < q2 < 12.5) or "
        "(15.0 < q2 < 19.0)"
    ).copy()
    
    df.dropna(subset=["mKpi", "q2"], inplace=True)

    return df


def read_component_reference(path):
    dfs = []

    with uproot.open(path) as root_file:
        for key in root_file.keys():
            obj = root_file[key]

            if not hasattr(obj, "arrays"):
                continue

            try:
                arrays = obj.arrays(
                    ["mB", "mkpi", "q2"],
                    library="pd",
                )
            except Exception:
                continue

            dfs.append(arrays)

    if len(dfs) == 0:
        raise RuntimeError(
            f"No valid trees found in {path}"
        )

    df = pd.concat(
        dfs,
        ignore_index=True,
    )

    df["B_mass"] = (
        df["mB"].to_numpy(dtype=float)
        / 1000.0
    )

    df["mKpi"] = (
        df["mkpi"].to_numpy(dtype=float)
        / 1000.0
    )

    df = df[
        (df["B_mass"] >= mib)
        & (df["B_mass"] <= mab)
        & (df["mKpi"] >= mih)
        & (df["mKpi"] <= mah)
        & (df["q2"] > miq)
        & (df["q2"] < maq)
    ].copy()

    df = df.query(
        "(0.06 < q2 < 9.0) or "
        "(11.0 < q2 < 12.5) or "
        "(15.0 < q2 < 19.0)"
    ).copy()

    df.dropna(
        subset=[
            "B_mass",
            "mKpi",
            "q2",
        ],
        inplace=True,
    )

    return (
        df["mKpi"].to_numpy(dtype=float),
        df["q2"].to_numpy(dtype=float),
    )

def make_empty_2d_hist():
    return hist.Hist(
        hist.axis.Regular(
            nbins,
            mih,
            mah,
            underflow=False,
            overflow=False,
        ),
        hist.axis.Regular(
            nbins,
            miq,
            maq,
            underflow=False,
            overflow=False,
        ),
        storage=hist.storage.Weight(),
    )


def make_scaled_reference_hist(mkpi, q2, target_yield):
    if len(mkpi) == 0:
        raise RuntimeError("Reference sample is empty after selection.")

    href = make_empty_2d_hist()

    event_weight = float(target_yield) / len(mkpi)

    href.fill(
        mkpi,
        q2,
        weight=np.full(len(mkpi), event_weight, dtype=float),
    )

    return href


def save_2d_plot(values, output_path, vmin=None, vmax=None):
    plt.figure()

    image = plt.imshow(
        values,
        cmap="coolwarm",
        origin="lower",
        extent=[miq, maq, mih, mah],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    plt.xlabel(
        r"$q^2$ [GeV$^2/c^4$]",
        ha="right",
        x=1,
    )
    plt.ylabel(
        r"$m(K\pi)$ [GeV/$c^2$]",
        ha="right",
        y=1,
    )
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


print("Reading truth settings...")

with open(args.settings) as settings_file:
    truth = yaml.load(
        settings_file,
        Loader=yaml.FullLoader,
    )

truth_App = truth_value(
    truth,
    "App",
    default=args.app,
)
truth_A0 = truth_value(
    truth,
    "A0",
)
truth_Abeta = truth_value(
    truth,
    "Abeta",
    default=0.0,
)

truth_AS = (
    1.0
    - truth_A0
    - truth_App
    - truth_Abeta
)

print("Truth fractions:")
print("  App   =", truth_App)
print("  A0    =", truth_A0)
print("  Abeta =", truth_Abeta)
print("  AS    =", truth_AS)

if truth_AS <= 0.0:
    raise RuntimeError(
        f"The calculated AS truth fraction is not positive: {truth_AS}"
    )


print("\nReading the efficiency-applied signal pool...")

df_sig = pd.read_hdf(
    args.data,
    key="data",
)
df_sig = filter_dataframe(df_sig)

if "fit_weight" not in df_sig.columns:
    if "efficiency" not in df_sig.columns:
        raise KeyError(
            "The signal HDF5 contains neither fit_weight nor efficiency."
        )

    if "eff_max" in df_sig.columns:
        eff_max_sig = float(df_sig["eff_max"].iloc[0])
    else:
        eff_max_sig = float(df_sig["efficiency"].max())

    df_sig["fit_weight"] = (
        eff_max_sig
        / df_sig["efficiency"].to_numpy(dtype=float)
    )

print("Signal pool events:", len(df_sig))
print("Signal pool weighted sum:", df_sig["fit_weight"].sum())


print("\nReading the background sample...")

with uproot.open(args.background) as root_file:
    background_arrays = root_file[
        args.background_tree
    ].arrays(
        [
            "B_mass",
            "cosThetaK",
            "cosThetaL",
            "mKpi",
            "q2",
        ],
        library="np",
    )

df_bkg_truth = pd.DataFrame(
    {
        "B_mass": background_arrays["B_mass"],
        "cosThetaK": background_arrays["cosThetaK"],
        "cosThetaL": background_arrays["cosThetaL"],
        "mKpi": background_arrays["mKpi"],
        "q2": background_arrays["q2"],
    }
)

df_bkg_truth = filter_dataframe(df_bkg_truth)

df_bkg_truth["cosh"] = df_bkg_truth[
    "cosThetaK"
]
df_bkg_truth["cosl"] = df_bkg_truth[
    "cosThetaL"
]

eff_bkg = efficiency.efficiency(
    df_bkg_truth["cosh"].to_numpy(dtype=float),
    df_bkg_truth["cosl"].to_numpy(dtype=float),
    df_bkg_truth["mKpi"].to_numpy(dtype=float),
    df_bkg_truth["q2"].to_numpy(dtype=float),
)

eff_bkg = np.asarray(
    eff_bkg,
    dtype=float,
)

eff_max_bkg = float(np.max(eff_bkg))

rng = np.random.default_rng(12345)
uniform_random = rng.uniform(
    0.0,
    eff_max_bkg,
    len(df_bkg_truth),
)

accepted_bkg = uniform_random < eff_bkg

df_bkg_observed = df_bkg_truth.loc[
    accepted_bkg
].copy()

df_bkg_observed["efficiency"] = eff_bkg[
    accepted_bkg
]
df_bkg_observed["eff_max"] = eff_max_bkg
df_bkg_observed["fit_weight"] = (
    eff_max_bkg
    / df_bkg_observed[
        "efficiency"
    ].to_numpy(dtype=float)
)

print("Background truth events:", len(df_bkg_truth))
print("Background observed events:", len(df_bkg_observed))
print(
    "Background observed weighted sum:",
    df_bkg_observed["fit_weight"].sum(),
)


pool_size = len(df_sig) + len(df_bkg_observed)

expected_signal_weighted = (
    args.toy_mean
    * df_sig["fit_weight"].sum()
    / pool_size
)
expected_background_weighted = (
    args.toy_mean
    * df_bkg_observed["fit_weight"].sum()
    / pool_size
)

expected_yields = {
    "total": args.toy_mean,
    "wApp": expected_signal_weighted * truth_App,
    "wA0": expected_signal_weighted * truth_A0,
    "wS": expected_signal_weighted * truth_AS,
    "wBkg": expected_background_weighted,
}

print("\nExpected yields per toy:")
for name, value in expected_yields.items():
    print(f"  {name:5s} = {value:.6f}")


print("\nBuilding independent 2D reference histograms...")

reference_hists = {}

# Infer the actual signal/background counts from the first toy.
first_toy = pd.read_hdf(args.input[0], key="data")

if "is_signal" not in first_toy.columns:
    raise KeyError(
        "Cannot infer the signal/background mixture: "
        "'is_signal' is missing from the first toy."
    )

n_sig_toy = int((first_toy["is_signal"] == 1).sum())
n_bkg_toy = int((first_toy["is_signal"] == 0).sum())

if n_sig_toy + n_bkg_toy != len(first_toy):
    raise ValueError(
        "Unexpected values in is_signal: "
        "not all rows are classified as signal (1) or background (0)."
    )

print(
    f"Toy mixture: {n_sig_toy} signal + "
    f"{n_bkg_toy} background = {len(first_toy)} events"
)

# Build the total reference using the actual toy mixture.
reference_hists["total"] = make_empty_2d_hist()

reference_hists["total"].fill(
    df_sig["mKpi"].to_numpy(dtype=float),
    df_sig["q2"].to_numpy(dtype=float),
    weight=np.full(
        len(df_sig),
        n_sig_toy / len(df_sig),
        dtype=float,
    ),
)

reference_hists["total"].fill(
    df_bkg_observed["mKpi"].to_numpy(dtype=float),
    df_bkg_observed["q2"].to_numpy(dtype=float),
    weight=np.full(
        len(df_bkg_observed),
        n_bkg_toy / len(df_bkg_observed),
        dtype=float,
    ),
)

component_files = {
    "wApp": "/ceph/submit/data/user/a/anbeck/B2KPiMM_leon/massless_high_stats/Toyevents_forAnja_pw_par_perp.root",
    "wA0": "/ceph/submit/data/user/a/anbeck/B2KPiMM_leon/massless_high_stats/Toyevents_forAnja_pw_longitudinal.root",
    "wS": "/ceph/submit/data/user/a/anbeck/B2KPiMM_leon/massless_high_stats/Toyevents_forAnja_sw_longitudinal.root",
}

for weight_name, filename in component_files.items():
    print("Reading", filename)

    ref_mkpi, ref_q2 = read_component_reference(
        filename
    )

    reference_hists[weight_name] = (
        make_scaled_reference_hist(
            ref_mkpi,
            ref_q2,
            1.0,
        )
    )

reference_hists["wBkg"] = (
    make_scaled_reference_hist(
        df_bkg_truth["mKpi"].to_numpy(dtype=float),
        df_bkg_truth["q2"].to_numpy(dtype=float),
        1.0,
    )
)

quantities = [
    "total",
    "wApp",
    "wA0",
    "wS",
    "wBkg",
]

required_toy_columns = {
    "total": ["mKpi", "q2"],
    "wApp": ["mKpi", "q2", "wApp", "is_signal", "fit_weight"],
    "wA0": ["mKpi", "q2", "wA0", "is_signal", "fit_weight"],
    "wS": ["mKpi", "q2", "wS", "is_signal", "fit_weight"],
    "wBkg": ["mKpi", "q2", "wBkg", "is_signal", "fit_weight"],
}


for quantity in quantities:
    print("\n" + "=" * 80)
    print("Processing quantity:", quantity)

    href = reference_hists[quantity]
    pulls = []

    hpulls = hist.Hist(
        hist.axis.Regular(
            50,
            -5.0,
            5.0,
            underflow=False,
            overflow=False,
        ),
        storage=hist.storage.Weight(),
    )

    number_of_valid_toys = 0

    for index, filename in enumerate(args.input):
        print(
            "Processing",
            index + 1,
            "/",
            len(args.input),
            filename,
        )

        try:
            data = pd.read_hdf(
                filename,
                key="data",
            )
        except Exception as error:
            print(
                "Warning: failed to read",
                filename,
            )
            print(error)
            continue

        missing_columns = [
            column
            for column in required_toy_columns[
                quantity
            ]
            if column not in data.columns
        ]

        if len(missing_columns) > 0:
            print(
                "Warning: skipping",
                filename,
                "because columns are missing:",
                missing_columns,
            )
            continue

        hi = make_empty_2d_hist()

        if quantity == "total":
            toy_weights = np.ones(
                len(data),
                dtype=float,
            )
        else:
            toy_weights = data[
                quantity
            ].to_numpy(dtype=float)

        hi.fill(
            data["mKpi"].to_numpy(dtype=float),
            data["q2"].to_numpy(dtype=float),
            weight=toy_weights,
        )

        toy_values = np.asarray(
            hi.values(),
            dtype=float,
        )
        toy_variances = np.asarray(
            hi.variances(),
            dtype=float,
        )

        if quantity == "total":
            reference_values = np.asarray(
                href.values(),
                dtype=float,
            )

            if index < 3:
                print(f"\n[Toy {index} | {quantity}]")
                print("Number of events:", len(data))
                print("Toy histogram sum:", np.sum(toy_values))
                print(
                    "Reference histogram sum:",
                    np.sum(reference_values),
                )
                print(
                    "Signal events:",
                    int((data["is_signal"] == 1).sum()),
                )
                print(
                    "Background events:",
                    int((data["is_signal"] == 0).sum()),
                )

        else:
            truth_signal_weighted = data.loc[
                data["is_signal"] == 1,
                "fit_weight",
            ].sum()

            truth_background_weighted = data.loc[
                data["is_signal"] == 0,
                "fit_weight",
            ].sum()

            toy_truth_yields = {
                "wApp": truth_signal_weighted * truth_App,
                "wA0": truth_signal_weighted * truth_A0,
                "wS": truth_signal_weighted * truth_AS,
                "wBkg": truth_background_weighted,
            }

            reference_values = (
                np.asarray(
                    href.values(),
                    dtype=float,
                )
                * toy_truth_yields[quantity]
            )

            if index < 3:
                print(f"\n[Toy {index} | {quantity}]")
                print("Number of events:", len(data))
                print("Toy histogram sum:", np.sum(toy_values))
                print(
                    "Reference histogram sum:",
                    np.sum(reference_values),
                )
                print(
                    "Weighted signal yield:",
                    truth_signal_weighted,
                )
                print(
                    "Weighted background yield:",
                    truth_background_weighted,
                )
                print(
                    "Expected reference yield:",
                    toy_truth_yields[quantity],
                )
                print(
                    "COW extracted yield:",
                    np.sum(toy_weights),
                )
                
        pull = np.full_like(
            toy_values,
            np.nan,
            dtype=float,
        )

        valid_bins = (
            np.isfinite(toy_values)
            & np.isfinite(toy_variances)
            & np.isfinite(reference_values)
            & (toy_variances > 0.0)
        )

        pull[valid_bins] = (
            toy_values[valid_bins]
            - reference_values[valid_bins]
        ) / np.sqrt(
            toy_variances[valid_bins]
        )

        pull[~np.isfinite(pull)] = np.nan

        pulls.append(pull)

        finite_pull_values = pull[
            np.isfinite(pull)
        ]

        hpulls.fill(
            finite_pull_values,
            weight=np.ones(
                len(finite_pull_values),
                dtype=float,
            ),
        )

        number_of_valid_toys += 1

    if number_of_valid_toys == 0:
        print(
            "No valid toys were found for",
            quantity,
        )
        continue

    pulls = np.asarray(
        pulls,
        dtype=float,
    )

    mean_per_bin = np.nanmean(
        pulls,
        axis=0,
    )
    std_per_bin = np.nanstd(
        pulls,
        axis=0,
    )

    finite_mean_pulls = mean_per_bin[
        np.isfinite(mean_per_bin)
    ]

    print(f"\n[{quantity} mean-pull summary]")
    print(
        "  number of valid bins =",
        len(finite_mean_pulls),
    )
    print(
        "  mean   =",
        np.mean(finite_mean_pulls),
    )
    print(
        "  median =",
        np.median(finite_mean_pulls),
    )
    print(
        "  std    =",
        np.std(
            finite_mean_pulls,
            ddof=1,
        ),
    )
    print(
        "  min    =",
        np.min(finite_mean_pulls),
    )
    print(
        "  max    =",
        np.max(finite_mean_pulls),
    )
    print(
        "  frac > 0    =",
        np.mean(
            finite_mean_pulls > 0.0
        ),
    )
    print(
        "  frac > 0.5  =",
        np.mean(
            finite_mean_pulls > 0.5
        ),
    )
    print(
        "  frac > 1.0  =",
        np.mean(
            finite_mean_pulls > 1.0
        ),
    )
    print(
        "  frac < -0.5 =",
        np.mean(
            finite_mean_pulls < -0.5
        ),
    )

    output_prefix = os.path.join(
        args.output_dir,
        f"{args.name}_{quantity}",
    )

    plt.figure()
    mplhep.histplot(
        hpulls,
        histtype="errorbar",
        xerr=True,
        yerr=True,
    )
    plt.xlabel(
        f"Pull {quantity}",
        ha="right",
        x=1,
    )
    plt.ylabel(
        "Entries",
        ha="right",
        y=1,
    )
    plt.tight_layout()
    plt.savefig(
        f"{output_prefix}_pulls.pdf"
    )
    plt.close()

    save_2d_plot(
        mean_per_bin,
        f"{output_prefix}_mean.pdf",
        vmin=-3.0,
        vmax=3.0,
    )

    save_2d_plot(
        std_per_bin,
        f"{output_prefix}_std.pdf",
        vmin=0.0,
        vmax=2.0,
    )

    save_2d_plot(
        href.values(),
        f"{output_prefix}_ref.pdf",
    )

    print(
        "Saved plots with prefix:",
        output_prefix,
    )

print("\nFinished.")
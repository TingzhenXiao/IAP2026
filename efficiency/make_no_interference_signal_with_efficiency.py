import os
import numpy as np
import pandas as pd
import uproot

from efficiency import efficiency


OUTDIR = "efficiency_applied_output"
os.makedirs(OUTDIR, exist_ok=True)

DATADIR = os.environ["DATADIR"]

reference_samples = {
    "A0": ("A0.root", "B02KstMuMu_Run1_centralQ2E_sig"),
    "App": ("A1.root", "B02KstMuMu_Run1_centralQ2E_sig"),
    "AS": ("AS.root", "B02KstMuMu_Run1_centralQ2E_sig"),
}


def read_reference_sample(filename, tree_name, component):

    path = os.path.join(DATADIR, filename)

    with uproot.open(path) as f:
        df = f[tree_name].arrays(
            ["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi"],
            library="pd",
        )

    df["cosh"] = df["cosThetaK"]
    df["cosl"] = df["cosThetaL"]

    if df["B_mass"].max() > 100.0:
        df["B_mass"] = df["B_mass"] / 1000.0

    df = df[(df["q2"] > 1.1) & (df["q2"] < 7.0)].copy()
    df = df[(df["mKpi"] < 1.5)].copy()
    df = df[(df["B_mass"] >= 5.170) & (df["B_mass"] <= 5.500)].copy()
    df.dropna(inplace=True)

    df["component"] = component

    return df


def apply_efficiency(df, rng):

    eff = efficiency(
        df["cosh"].to_numpy(dtype=float),
        df["cosl"].to_numpy(dtype=float),
        df["mKpi"].to_numpy(dtype=float),
        df["q2"].to_numpy(dtype=float),
    )

    eff_max = float(np.max(eff))

    u = rng.uniform(0.0, eff_max, len(df))
    mask = u < eff

    df_eff = df[mask].copy()
    df_eff["efficiency"] = eff[mask]
    df_eff["eff_max"] = eff_max
    df_eff["fit_weight"] = eff_max / df_eff["efficiency"].to_numpy(dtype=float)

    return df_eff


def main():

    rng = np.random.default_rng(0)

    dfs = []

    for component, item in reference_samples.items():
        filename, tree_name = item

        print("Reading", component, filename)

        df = read_reference_sample(filename, tree_name, component)
        df_eff = apply_efficiency(df, rng)

        print(component, "before efficiency:", len(df))
        print(component, "after efficiency:", len(df_eff))
        print(component, "weighted sum:", df_eff["fit_weight"].sum())

        out_single = os.path.join(
            OUTDIR,
            f"{component}_with_efficiency.h5",
        )

        df_eff.to_hdf(out_single, key="data", mode="w")

        dfs.append(df_eff)

    df_nointer = pd.concat(dfs, ignore_index=True)
    df_nointer = df_nointer.sample(frac=1.0, random_state=0).reset_index(drop=True)

    out_combined = os.path.join(
        OUTDIR,
        "nointer_signal_with_efficiency.h5",
    )

    df_nointer.to_hdf(out_combined, key="data", mode="w")

    print("Combined no-interference signal events:", len(df_nointer))
    print("Combined weighted sum:", df_nointer["fit_weight"].sum())
    print("Saved to:", out_combined)


if __name__ == "__main__":
    main()
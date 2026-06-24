import os
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt

from efficiency import efficiency

OUTDIR = "efficiency_applied_output"


def read_input_data():

    with uproot.open("/ceph/submit/data/user/a/anbeck/B2KPiMM_michele/full.root") as f:
        tree = f["B02KstMuMu_Run1_centralQ2E_sig"]
        df = tree.arrays(["cosThetaK", "cosThetaL", "mKpi", "q2", "B_mass"], library="pd")

    df = df[["cosThetaK", "cosThetaL", "mKpi", "q2", "B_mass"]].dropna().reset_index(drop=True)
    return df


def compute_efficiency(df):
    eff = efficiency(
        df["cosThetaK"].to_numpy(),
        df["cosThetaL"].to_numpy(),
        df["mKpi"].to_numpy(),
        df["q2"].to_numpy(),
    )
    eff = np.asarray(eff)
    eff = np.nan_to_num(eff, nan=0.0, posinf=0.0, neginf=0.0)
    return eff


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # =========================
    # Task 4: apply efficiency
    # =========================
    df_original = read_input_data()
    print(f"Original events: {len(df_original)}")

    eff = compute_efficiency(df_original)
    eff_max = eff.max()

    u = rng.uniform(0.0, eff_max, len(df_original))
    mask = u < eff

    df_selected = df_original.loc[mask].copy()
    df_selected["eff_max"] = eff_max
    df_selected["efficiency"] = eff[mask]
    df_selected["fit_weight"] = eff_max / df_selected["efficiency"].to_numpy(dtype=float)

    output_h5 = os.path.join(OUTDIR, "signal_with_efficiency.h5")
    df_selected.to_hdf(output_h5, key="data", mode="w")

    print(f"Selected events: {len(df_selected)}")
    print(f"Acceptance fraction: {mask.mean():.6f}")
    print(f"eff min: {eff.min():.6f}")
    print(f"eff max: {eff.max():.6f}")
    print(f"Saved selected sample to: {output_h5}")
    print("eff min =", df_selected["efficiency"].min())
    print("eff max =", df_selected["efficiency"].max())
    print("sum 1/eff =", np.sum(1.0 / df_selected["efficiency"].to_numpy(dtype=float)))
    print("sum eff_max/eff =", np.sum(eff_max / df_selected["efficiency"].to_numpy(dtype=float)))
    print("ratio =", np.sum(eff_max / df_selected["efficiency"].to_numpy(dtype=float)) / np.sum(1.0 / df_selected["efficiency"].to_numpy(dtype=float)))

    # =========================
    # Task 5: validation
    # =========================
    df_check = pd.read_hdf(output_h5, key="data").copy()

    eff_check = compute_efficiency(df_check)
    df_check["efficiency_recomputed"] = eff_check
    df_check["weight_recomputed"] = eff_max / eff_check

    variables = ["cosThetaK", "cosThetaL", "mKpi", "q2"]
    ranges = {
        "cosThetaK": (-1.0, 1.0),
        "cosThetaL": (-1.0, 1.0),
        "mKpi": (df_original["mKpi"].min(), df_original["mKpi"].max()),
        "q2": (df_original["q2"].min(), df_original["q2"].max()),
    }

    for var in variables:
        plt.figure(figsize=(7, 5))

        # perfect dataset
        plt.hist(
            df_original[var],
            bins=60,
            range=ranges[var],
            density=True,
            histtype="step",
            label="Original perfect sample",
        )

        # After apply efficiency，times 1/efficiency
        plt.hist(
            df_check[var],
            bins=60,
            range=ranges[var],
            weights=df_check["weight_recomputed"],
            density=True,
            histtype="step",
            label="Efficiency-applied sample weighted by 1/eff",
        )

        plt.xlabel(var)
        plt.ylabel("Density")
        plt.title(f"Closure check: {var}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"{var}_closure.png"))
        plt.close()

if __name__ == "__main__":
    main()


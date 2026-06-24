import os
import pandas as pd
import matplotlib.pyplot as plt
import uproot

datadir = os.environ["DATADIR"]
os.makedirs("compare_plots", exist_ok=True)

data = pd.read_hdf("sweights/standard/data_qsq-1.1-7.0/0.h5")

mapping = {
    "wA0": "A0.root",
    "wApp": "A1.root",
    "wS": "AS.root",
}

variables = ["mKpi", "q2"]

q2_low, q2_high = data["q2"].min(), data["q2"].max()
mKpi_low, mKpi_high = data["mKpi"].min(), data["mKpi"].max()

for weight_name, rootfile in mapping.items():
    f = uproot.open(f"{datadir}/{rootfile}")
    tree = f[f.keys()[0]]

    ref = tree.arrays(["mKpi", "q2"], library="pd")

    ref = ref[(ref["mKpi"] > mKpi_low) & (ref["mKpi"] < mKpi_high)]
    ref = ref[(ref["q2"] > q2_low) & (ref["q2"] < q2_high)]

    for v in variables:
        plt.figure()

        plt.hist(
            data[v],
            bins=50,
            weights=data[weight_name],
            histtype="step",
            density=True,
            label=f"{weight_name} weighted",
        )

        plt.hist(
            ref[v],
            bins=50,
            histtype="step",
            density=True,
            label=f"{rootfile} reference",
        )

        plt.xlabel(v)
        plt.ylabel("Normalized entries")
        plt.title(f"{weight_name} vs {rootfile}: {v}")
        plt.legend()
        plt.savefig(f"compare_plots/{weight_name}_vs_{rootfile.replace('.root','')}_{v}.png")
        plt.close()
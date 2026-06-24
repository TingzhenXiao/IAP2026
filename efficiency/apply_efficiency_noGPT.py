import argparse
import matplotlib.pyplot as plt
import uproot
import efficiency
import numpy
import pandas as pd
import os


parser = argparse.ArgumentParser(description="Plot histograms from ROOT file.")
parser.add_argument("--data", type=str, help="Path to the input ROOT file.")
args = parser.parse_args()

# Open the ROOT file
with uproot.open(args.data) as file:
    tree = file["B02KstMuMu_Run1_centralQ2E_sig"]
    data = tree.arrays(library="pd")
    
N = len(data)
cosh = data["cosThetaK"]
cosl = data["cosThetaL"]
mkpi = data["mKpi"]
qsq = data["q2"]

eff = efficiency.efficiency(cosh, cosl, mkpi, qsq)

eff_max = eff.max()

eff_random = numpy.random.uniform(low=0, high=eff_max, size=N)

mask = eff_random < eff

data_new = data[mask].copy()
data_new["efficiency"] = eff[mask]

data_new.to_hdf("signal_with_efficiency.h5", key="data_new", mode="w")

cosh_new = data_new["cosThetaK"]
cosl_new = data_new["cosThetaL"]
mkpi_new = data_new["mKpi"]
qsq_new = data_new["q2"]

os.makedirs("apply_efficiency_plots", exist_ok=True)
eff_new = efficiency.efficiency(cosh_new, cosl_new, mkpi_new, qsq_new)
variables = ["cosThetaK", "cosThetaL", "mKpi", "q2"]

for col in variables:
    plt.figure()
    plt.hist(data[col].dropna(), bins=100, histtype="step", linewidth=1.5, label="original")
    plt.hist(data_new[col].dropna(), bins=100, weights=eff_max / data_new["efficiency"], histtype="step", linewidth=1.5, label="Efficiency correct")
    plt.xlabel(col)
    plt.ylabel("Entries")
    plt.title(f"Distributions of {col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"apply_efficiency_plots/{col}.png")
    plt.close()  

path = "signal_with_efficiency.h5"
df = pd.read_hdf(path, key="data_new")

print("shape:", df.shape)
print("columns:")
print(df.columns.tolist())
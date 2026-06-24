import argparse
import matplotlib.pyplot as plt
import uproot

parser = argparse.ArgumentParser(description="Plot histograms from ROOT file.")
parser.add_argument("--data", type=str, help="Path to the input ROOT file.")
args = parser.parse_args()

# Open the ROOT file
with uproot.open(args.data) as file:
    data = file["B02KstMuMu_Run1_centralQ2E_sig"].arrays(library="pd")

print(data.shape)
print(data.columns)
print(data.head())
print(data.describe())

# Task 1) Inspect the data and identify the different variables. What are their units?
# Task 2) Create one plot per variable with five different distributions:
#           a) the total distribution
#           b,c) the distribution when selecting q2 smaller or larger than 2
#           d,e) the distribution when selecting mKpi smaller or larger than 1.1
#         Each distribution should be represented by a histogram
#         Save the plots as png or pdf files.

sel_q2_low = data["q2"] < 2
sel_q2_high = data["q2"] >= 2
sel_mKpi_low = data["mKpi"] < 1.1
sel_mKpi_high = data["mKpi"] >= 1.1

labels = {
    "cosThetaK": r"$\cos\theta_K$",
    "cosThetaL": r"$\cos\theta_L$",
    "phi": r"$\phi$ [rad]",
    "q2": r"$q^2$ [GeV$^2$/c$^4$]",
    "B_mass": r"$B$ mass [MeV/c$^2$]",
    "mKpi": r"$m_{K\pi}$ [GeV/c$^2$]", 
}

for col in data.columns:
    plt.figure(figsize=(8,6))

    plt.hist(data[col].dropna(), bins=60, histtype="step", linewidth=1.5, label="Total")
    plt.hist(data.loc[sel_q2_low, col].dropna(), bins=60, histtype="step", linewidth=1.5, label=r"$q^2 < 2$")
    plt.hist(data.loc[sel_q2_high, col].dropna(), bins=60, histtype="step", linewidth=1.5, label=r"$q^2 >= 2$")
    plt.hist(data.loc[sel_mKpi_low, col].dropna(), bins=60, histtype="step", linewidth=1.5, label=r"$m_{K\pi} < 1.1$")
    plt.hist(data.loc[sel_mKpi_high, col].dropna(), bins=60, histtype="step", linewidth=1.5, label=r"$m_{K\pi} >= 1.1$")

    plt.xlabel(labels.get(col, col))
    plt.ylabel("Entries")
    plt.title(f"Distributions of {col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{col}.png")
    plt.close()  

    plt.figure(figsize=(7,6))
    plt.hist2d(data["cosThetaK"], data["cosThetaL"], bins=60)
    plt.xlabel(labels["cosThetaK"])
    plt.ylabel(labels["cosThetaL"])
    plt.title(r"2D histogram of $\cos\theta_K$ and $\cos\theta_L$")
    plt.colorbar(label="Entries")
    plt.tight_layout()
    plt.savefig("cosThetaK_vs_cosThetaL.png")
    plt.close()    

    plt.figure(figsize=(7,6))
    plt.hist2d(data["mKpi"]**2, data["q2"], bins=60)
    plt.xlabel(labels["mKpi"])
    plt.ylabel(labels["q2"])
    plt.title(r"2D histogram of $m_{K\pi}$ and $q^2$")
    plt.colorbar(label="Entries")
    plt.tight_layout()
    plt.savefig("mKpi_vs_q2.png")
    plt.close()    

print(data[["cosThetaK", "cosThetaL", "mKpi", "q2"]].describe())



# Task 3) Do your figures have all the required features? (E.g. axis labels, legend, easily distinguishable colors or linestyles, etc.)
# Task 4) Can you interpret the differences between the distributions a-e?
# Task 5) Create a 2D histogram of cosThetaK and cosThetaL. Create another 2D histogram of mKpi and q2.
#         Save both histograms as png or pdf files.
# Task 5a: 2D histogram of cosThetaK and cosThetaL

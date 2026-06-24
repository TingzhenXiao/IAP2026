import os
import argparse

import uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import pandas as pd
mplhep.style.use(mplhep.style.LHCb2)


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input generated background ROOT file")
parser.add_argument("--tree", default="background", help="Tree name")
parser.add_argument("--outdir", default="plots_generated_bkg", help="Output directory")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

with uproot.open(args.input) as f:
    tree = f[args.tree]

    print("Available branches:", tree.keys())

    wanted_branches = [
        "q2",
        "mKpi",
        "B_mass",
        "cosThetaK",
        "cosThetaL",
        "bkg_type",
    ]

    data = {}

    for name in wanted_branches:
        if name in tree.keys():
            data[name] = tree[name].array(library="np")
        else:
            print(f"Warning: branch {name} not found, skip it.")

df = pd.DataFrame(data)

print("Loaded file:", args.input)
print("Columns:", list(df.columns))
print("Number of events:", len(df))

if "bkg_type" in df.columns:
    print("bkg_type counts:")
    print(df["bkg_type"].value_counts().sort_index())

corr_all = np.corrcoef(df["B_mass"], df["q2"])[0, 1]
print("Correlation corr(B_mass, q2), all events =", corr_all)

if "bkg_type" in df.columns:
    for bkg_type in sorted(df["bkg_type"].unique()):
        d = df[df["bkg_type"] == bkg_type]
        corr = np.corrcoef(d["B_mass"], d["q2"])[0, 1]
        print(f"Correlation corr(B_mass, q2), bkg_type={bkg_type} =", corr)


# ============================================================
# 1D histograms
# ============================================================

variables = [
    ("B_mass", r"$B$ mass [GeV/$c^2$]", (5.170, 5.500), 100),
    ("q2", r"$q^2$ [GeV$^2/c^4$]", (1.1, 7.0), 100),
    ("mKpi", r"$m(K\pi)$ [GeV/$c^2$]", (0.65, 1.50), 100),
    ("cosThetaK", r"$\cos\theta_K$", (-1.0, 1.0), 100),
    ("cosThetaL", r"$\cos\theta_\ell$", (-1.0, 1.0), 100),
]

for var, xlabel, xlim, nbins in variables:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        df[var],
        bins=nbins,
        range=xlim,
        histtype="step",
        density=True,
        linewidth=2,
        label="All background",
    )

    if "bkg_type" in df.columns:
        for bkg_type in sorted(df["bkg_type"].unique()):
            d = df[df["bkg_type"] == bkg_type]
            ax.hist(
                d[var],
                bins=nbins,
                range=xlim,
                histtype="step",
                density=True,
                linewidth=2,
                label=f"bkg_type = {bkg_type}",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized entries")
    ax.set_xlim(*xlim)
    ax.legend(fontsize=16)
    fig.tight_layout()

    outname = os.path.join(args.outdir, f"{var}_hist.pdf")
    fig.savefig(outname)
    plt.close(fig)

    print("Saved:", outname)


# ============================================================
# 2D histogram: B_mass vs q2
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

h = ax.hist2d(
    df["q2"],
    df["B_mass"],
    bins=[80, 80],
    range=[[1.1, 7.0], [5.170, 5.500]],
    density=True,
)

cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label("Normalized entries")

ax.set_xlabel(r"$q^2$ [GeV$^2/c^4$]")
ax.set_ylabel(r"$B$ mass [GeV/$c^2$]")
fig.tight_layout()

outname = os.path.join(args.outdir, "Bmass_vs_q2_2D.pdf")
fig.savefig(outname)
plt.close(fig)

print("Saved:", outname)


# ============================================================
# B_mass shape in different q2 bins
# This is the most useful plot to see the correlation.
# If B_mass and q2 are independent, these shapes should be similar.
# If correlated, the B_mass slope changes with q2.
# ============================================================

q2_bins = [
    (1.1, 2.0),
    (2.0, 4.0),
    (4.0, 5.5),
    (5.5, 7.0),
]

fig, ax = plt.subplots(figsize=(8, 6))

for q2_low, q2_high in q2_bins:
    d = df[(df["q2"] > q2_low) & (df["q2"] < q2_high)]

    ax.hist(
        d["B_mass"],
        bins=80,
        range=(5.170, 5.500),
        histtype="step",
        density=True,
        linewidth=2,
        label=fr"${q2_low}<q^2<{q2_high}$",
    )

ax.set_xlabel(r"$B$ mass [GeV/$c^2$]")
ax.set_ylabel("Normalized entries")
ax.set_xlim(5.170, 5.500)
ax.legend(fontsize=14)
fig.tight_layout()

outname = os.path.join(args.outdir, "Bmass_in_q2_bins.pdf")
fig.savefig(outname)
plt.close(fig)

print("Saved:", outname)


# ============================================================
# Optional: plot only correlated component bkg_type = 1
# ============================================================

if "bkg_type" in df.columns:
    d_corr = df[df["bkg_type"] == 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    for q2_low, q2_high in q2_bins:
        d = d_corr[(d_corr["q2"] > q2_low) & (d_corr["q2"] < q2_high)]

        ax.hist(
            d["B_mass"],
            bins=80,
            range=(5.170, 5.500),
            histtype="step",
            density=True,
            linewidth=2,
            label=fr"${q2_low}<q^2<{q2_high}$",
        )

    ax.set_xlabel(r"$B$ mass [GeV/$c^2$]")
    ax.set_ylabel("Normalized entries")
    ax.set_xlim(5.170, 5.500)
    ax.legend(fontsize=14)
    fig.tight_layout()

    outname = os.path.join(args.outdir, "Bmass_in_q2_bins_bkg_type1.pdf")
    fig.savefig(outname)
    plt.close(fig)

    print("Saved:", outname)


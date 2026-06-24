import numpy as np
import pandas as pd
from efficiency import efficiency
import os
import matplotlib.pyplot as plt

# ranges from exploration data
RANGES = {
    "cosh": (-1.0, 1.0),
    "cosl": (-1.0, 1.0),
    "mkpi": (0.634, 1.8),
    "qsq":  (0.100, 12.5),
}

N = 5_000_000

rng = np.random.default_rng(42)

# generate uniform 4D sample
cosh = rng.uniform(*RANGES["cosh"], N)
cosl = rng.uniform(*RANGES["cosl"], N)
mkpi = rng.uniform(*RANGES["mkpi"], N)
qsq  = rng.uniform(*RANGES["qsq"], N)

# evaluate efficiency
eff = efficiency(cosh, cosl, mkpi, qsq)
eff = np.asarray(eff)
eff = np.nan_to_num(eff, nan=0.0, posinf=0.0, neginf=0.0)
eff = np.clip(eff, 0.0, None)

# accept-reject
eff_max = eff.max()
u = rng.uniform(0.0, eff_max, N)
mask = u < eff

# keep accepted points
df = pd.DataFrame({
    "cosThetaK": cosh[mask],
    "cosThetaL": cosl[mask],
    "mKpi": mkpi[mask],
    "q2": qsq[mask],
    "efficiency": eff[mask],
})

df.to_hdf("efficiency_sample.h5", key="data", mode="w")

print(f"Generated points: {N}")
print(f"Accepted points : {mask.sum()}")
print(f"Acceptance frac : {mask.mean():.6f}")
print(f"eff min         : {eff.min():.6f}")
print(f"eff max         : {eff.max():.6f}")

os.makedirs("efficiency_plots", exist_ok=True)

for col in ["cosThetaK", "cosThetaL", "mKpi", "q2"]:
    plt.figure()
    plt.hist(df[col], bins=60, density=True, histtype="step")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.title(f"Efficiency sample: {col}")
    plt.tight_layout()
    plt.savefig(f"efficiency_plots/{col}_1d.png")
    plt.close()

    pairs = [
    ("cosThetaK", "cosThetaL"),
    ("cosThetaK", "mKpi"),
    ("cosThetaK", "q2"),
    ("cosThetaL", "mKpi"),
    ("cosThetaL", "q2"),
    ("mKpi", "q2"),
]

for x, y in pairs:
    plt.figure()
    plt.hist2d(df[x], df[y], bins=60)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Efficiency sample: {x} vs {y}")
    plt.colorbar(label="Counts")
    plt.tight_layout()
    plt.savefig(f"efficiency_plots/{x}_vs_{y}_2d.png")
    plt.close()
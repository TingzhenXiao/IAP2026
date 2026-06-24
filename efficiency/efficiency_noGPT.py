import numpy
import efficiency
import hist
import mplhep
import matplotlib.pyplot as plt  # Plotting library
import os

cosh=numpy.random.uniform(low=-1.0, high=1.0, size=10_000_000)
cosl=numpy.random.uniform(low=-1.0, high=1.0, size=10_000_000)
qsq=numpy.random.uniform(low=1.1, high=7.0, size=10_000_000)
mkpi=numpy.random.uniform(low=0.65, high=1.5, size=10_000_000)

eff = efficiency.efficiency(cosh, cosl, mkpi, qsq)

eff_max = eff.max()

eff_random = numpy.random.uniform(low=0, high=eff_max, size=10_000_000)

mask = eff_random < eff

variable = [("cosh", -1.0, 1.0, cosh[mask]),
            ("cosl", -1.0, 1.0, cosl[mask]),
            ("qsq", 1.1, 7.0, qsq[mask]),
            ("mkpi", 0.65, 1.5, mkpi[mask]),]

os.makedirs("efficiency_plots", exist_ok=True)

for var, x_min, x_max, var_eff in variable:
    plt.figure(figsize=(8,6))
    plt.hist(var_eff, bins=100, histtype="step", linewidth=1.5)
    plt.xlabel(f"{var}")
    plt.ylabel("number of data")
    plt.title(f"efficiency sample {var}")
    plt.tight_layout()
    plt.savefig(f"efficiency_plots/{var}.png")
    plt.close()  

pairs = [("cosh", "cosl"),
         ("cosh", "qsq"),
         ("cosh", "mkpi"),
         ("cosl", "qsq"),
         ("cosl", "mkpi"),
         ("qsq", "mkpi"),]

dict_data = {
    "cosh": cosh[mask],
    "cosl": cosl[mask],
    "qsq": qsq[mask],
    "mkpi": mkpi[mask]
}

for x, y in pairs:
    plt.figure()
    plt.hist2d(dict_data[x], dict_data[y], bins=100)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"efficiency sample {x} vs {y}")
    plt.colorbar(label="Counts")
    plt.tight_layout()
    plt.savefig(f"efficiency_plots/efficiency sample {x} vs {y}.png")


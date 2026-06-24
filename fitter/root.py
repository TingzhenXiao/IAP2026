import mplhep
import matplotlib.pyplot as plt
import pprint

mplhep.style.use(mplhep.style.LHCb2)

keys = [
    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "figure.figsize",
    "figure.autolayout",
    "savefig.bbox",
]

for key in keys:
    print(key, "=", plt.rcParams[key])
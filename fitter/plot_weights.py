import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plot_weights", exist_ok=True)
data = pd.read_hdf("sweights/standard/data_qsq-1.1-7.0/0.h5")

for w in ["wS", "wApp", "wA0", "wAq"]:
    plt.figure()
    plt.hist(data["mKpi"], bins=50, weights=data[w])
    plt.xlabel("mKpi")
    plt.ylabel("Weighted events")
    plt.title(f"{w}-weighted mKpi")
    plt.tight_layout()
    plt.savefig(f"plot_weights/{w}_mKpi.png")
    plt.show()

    plt.figure()
    plt.hist(data["q2"], bins=50, weights=data[w])
    plt.xlabel("q2")
    plt.ylabel("Weighted events")
    plt.title(f"{w}-weighted q2")
    plt.tight_layout()
    plt.savefig(f"plot_weights/{w}_q2.png")
    plt.show()
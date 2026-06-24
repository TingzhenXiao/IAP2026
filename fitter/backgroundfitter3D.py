import os
import json
import argparse
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import zfit

def main():
    parser = argparse.ArgumentParser(description="Fit 3D background PDF to toy sample.")
    parser.add_argument("--data", type=str, default="../genbkg/background_toy.root",
                        help="Input ROOT file")
    parser.add_argument("--tree", type=str, default="background",
                        help="Tree name inside ROOT file")
    parser.add_argument("--outdir", type=str, default="background_fit_output_3d",
                        help="Directory to save plots and fit results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------------
    # 1) Read background toy sample
    # -----------------------------------
    with uproot.open(args.data) as f:
        arr = f[args.tree].arrays(["B_mass", "cosThetaK", "cosThetaL"], library="np")

    df = pd.DataFrame({
        "B_mass": arr["B_mass"],
        "cosThetaK": arr["cosThetaK"],
        "cosThetaL": arr["cosThetaL"],
    }).dropna()

    print("Number of events used in fit:", len(df))
    print(df.head())

    # -----------------------------------
    # 2) Define observables
    # -----------------------------------
    mass = zfit.Space("B_mass", limits=(5.170, 5.500))
    cosK = zfit.Space("cosThetaK", limits=(-1.0, 1.0))
    cosL = zfit.Space("cosThetaL", limits=(-1.0, 1.0))
    obs = mass * cosK * cosL

    data = zfit.Data.from_pandas(obs=obs, df=df)

    # -----------------------------------
    # 3) Define 3D background PDF
    # -----------------------------------
    lambda_bmass = zfit.Parameter("lambda_bmass", -0.1, -2.0, 0.0)

    a1_cosh = zfit.Parameter("a1_cosh", 0.0, -0.5, 0.5)
    a2_cosh = zfit.Parameter("a2_cosh", 0.0, -0.8, 0.8)

    a1_cosl = zfit.Parameter("a1_cosl", 0.0, -0.5, 0.5)
    a2_cosl = zfit.Parameter("a2_cosl", 0.0, -0.8, 0.8)

    pdf_mass = zfit.pdf.Exponential(obs=mass, lambda_=lambda_bmass)
    pdf_cosh = zfit.pdf.Legendre(obs=cosK, coeffs=[a1_cosh, a2_cosh])
    pdf_cosl = zfit.pdf.Legendre(obs=cosL, coeffs=[a1_cosl, a2_cosl])

    pdf_3d = zfit.pdf.ProductPDF([pdf_mass, pdf_cosh, pdf_cosl], obs=obs)

    # -----------------------------------
    # 4) Fit
    # -----------------------------------
    nll = zfit.loss.UnbinnedNLL(model=pdf_3d, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    result.hesse()

    print("\n=== Fit result ===")
    print(result)

    params = [lambda_bmass, a1_cosh, a2_cosh, a1_cosl, a2_cosl]
    for p in params:
        val = result.params[p]["value"]
        err = result.params[p]["hesse"]["error"] if "hesse" in result.params[p] else None
        print(f"{p.name}: {val:.6f} +/- {err}")

    # -----------------------------------
    # 5) Save fit result
    # -----------------------------------
    fit_results = {
        "lambda_bmass": {
            "value": float(result.params[lambda_bmass]["value"]),
            "error": None if "hesse" not in result.params[lambda_bmass] else float(result.params[lambda_bmass]["hesse"]["error"]),
            "truth": -0.2,
        },
        "a1_cosh": {
            "value": float(result.params[a1_cosh]["value"]),
            "error": None if "hesse" not in result.params[a1_cosh] else float(result.params[a1_cosh]["hesse"]["error"]),
            "truth": 0.0,
        },
        "a2_cosh": {
            "value": float(result.params[a2_cosh]["value"]),
            "error": None if "hesse" not in result.params[a2_cosh] else float(result.params[a2_cosh]["hesse"]["error"]),
            "truth": -0.2,
        },
        "a1_cosl": {
            "value": float(result.params[a1_cosl]["value"]),
            "error": None if "hesse" not in result.params[a1_cosl] else float(result.params[a1_cosl]["hesse"]["error"]),
            "truth": 0.0,
        },
        "a2_cosl": {
            "value": float(result.params[a2_cosl]["value"]),
            "error": None if "hesse" not in result.params[a2_cosl] else float(result.params[a2_cosl]["hesse"]["error"]),
            "truth": -0.4,
        },
    }

    with open(os.path.join(args.outdir, "fit_results_3d.json"), "w") as f:
        json.dump(fit_results, f, indent=4)

    # -----------------------------------
    # 6) Plot projections
    # -----------------------------------
    # B_mass
    x_mass = np.linspace(5.170, 5.500, 500)
    y_mass = pdf_mass.pdf(x_mass).numpy()

    plt.figure(figsize=(7, 5))
    plt.hist(df["B_mass"], bins=50, range=(5.170, 5.500), density=True,
             histtype="step", label="Toy data")
    plt.plot(x_mass, y_mass, label="Fit")
    plt.xlabel("B_mass")
    plt.ylabel("Density")
    plt.title("Background fit: B_mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "B_mass_fit.png"), dpi=300)
    plt.close()

    # cosThetaK
    x = np.linspace(-1, 1, 500)
    yK = pdf_cosh.pdf(x).numpy()

    plt.figure(figsize=(7, 5))
    plt.hist(df["cosThetaK"], bins=50, range=(-1, 1), density=True,
             histtype="step", label="Toy data")
    plt.plot(x, yK, label="Fit")
    plt.xlabel("cosThetaK")
    plt.ylabel("Density")
    plt.title("Background fit: cosThetaK")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "cosThetaK_fit.png"), dpi=300)
    plt.close()

    # cosThetaL
    yL = pdf_cosl.pdf(x).numpy()

    plt.figure(figsize=(7, 5))
    plt.hist(df["cosThetaL"], bins=50, range=(-1, 1), density=True,
             histtype="step", label="Toy data")
    plt.plot(x, yL, label="Fit")
    plt.xlabel("cosThetaL")
    plt.ylabel("Density")
    plt.title("Background fit: cosThetaL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "cosThetaL_fit.png"), dpi=300)
    plt.close()

    print(f"\nSaved results to: {args.outdir}")
    print("  - fit_results_3d.json")
    print("  - B_mass_fit.png")
    print("  - cosThetaK_fit.png")
    print("  - cosThetaL_fit.png")


if __name__ == "__main__":
    main()
import glob
import os
import re

import numpy as np
import pandas as pd
import yaml

RESULT_DIR = "results_cow_bkg_eff_Ig_correctBkg_fixAqs_floatMass_wideSigma2_100toys"
SETTINGS_FILE = "settings/App=0.1670_qsq-1.1-7.0.yml"


def toy_index_from_filename(path):
    base = os.path.basename(path)
    match = re.match(r"(\d+)_parameters_with_uncertainties\.csv", base)

    if match is None:
        return None

    return int(match.group(1))


def load_truth(settings_file):
    with open(settings_file) as f:
        truth_raw = yaml.load(f, Loader=yaml.FullLoader)

    truth = {}

    for name, info in truth_raw.items():
        if isinstance(info, dict) and "value" in info:
            truth[name] = float(info["value"])

    return truth


def main():
    truth = load_truth(SETTINGS_FILE)

    paths = sorted(
        glob.glob(os.path.join(RESULT_DIR, "*_parameters_with_uncertainties.csv")),
        key=toy_index_from_filename,
    )

    print("Number of saved toy result files:", len(paths))

    rows = []

    for path in paths:
        toy = toy_index_from_filename(path)
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            rows.append({
                "toy": toy,
                "name": row["name"],
                "value": float(row["value"]),
                "error": row["error"],
                "floating": bool(row["floating"]),
            })

    all_df = pd.DataFrame(rows)
    all_df.to_csv("toy_all_parameters.csv", index=False)

    summary_rows = []

    for name in sorted(all_df["name"].unique()):
        if name not in truth:
            continue

        sub = all_df[all_df["name"] == name].copy()
        truth_value = truth[name]

        sub["error"] = pd.to_numeric(sub["error"], errors="coerce")
        sub["bias"] = sub["value"] - truth_value

        good_error = sub["error"].notna() & np.isfinite(sub["error"]) & (sub["error"] > 0)
        sub_good = sub[good_error].copy()

        if len(sub_good) > 1:
            sub_good["pull"] = (sub_good["value"] - truth_value) / sub_good["error"]
            pull_mean = sub_good["pull"].mean()
            pull_width = sub_good["pull"].std(ddof=1)
        else:
            pull_mean = np.nan
            pull_width = np.nan

        summary_rows.append({
            "name": name,
            "truth": truth_value,
            "ntoys": len(sub),
            "ntoys_with_error": len(sub_good),
            "mean_fit": sub["value"].mean(),
            "std_fit": sub["value"].std(ddof=1),
            "mean_bias": sub["bias"].mean(),
            "pull_mean": pull_mean,
            "pull_width": pull_width,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("toy_summary.csv", index=False)

    print()
    print("Toy summary:")
    print(summary_df.to_string(index=False))

    nbkg = all_df[all_df["name"] == "Nbkg"].copy()

    if len(nbkg) > 0:
        print()
        print("Nbkg summary:")
        print(nbkg["value"].describe())
        print("N toys with Nbkg < 1:", int(np.sum(nbkg["value"] < 1.0)))
        print("N toys with Nbkg < 10:", int(np.sum(nbkg["value"] < 10.0)))
        print("N toys with Nbkg < 100:", int(np.sum(nbkg["value"] < 100.0)))


if __name__ == "__main__":
    main()

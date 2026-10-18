# Analysis reproduction

Code for fitting Leon's simulated B0 → Kπμμ samples and comparing COW extraction with angular moments.

## Setup

Run from `analysis_reproduction/` in the working `sweights-py312` conda environment. Make `tools.py`, `mypdfs.py`, `angularfunctions.py`, `myconstants.py`, and the efficiency module importable. Main packages: zfit, sweights (`experimental.Cows`), hepstats, uproot, pandas/PyTables, NumPy, SciPy, hist, mplhep, Matplotlib, and PyYAML.

```bash
conda activate sweights-py312
LEON=/ceph/submit/data/user/a/anbeck/B2KPiMM_leon
EFF=/ceph/submit/data/user/x/xiaot425/efficiency/efficiency_applied_output
SETTINGS=/home/submit/xiaot425/IAP2026/fitter/settings/truth.yml
COMMON=(--background "$LEON/background.root" --background-tree background
        --settings "$SETTINGS" --use_basis --qsq 1.1 19.0 --mKpi 0.746 1.5
        --with-bkg --with-eff --cow-I g --bkg-series-degree 4 --error-method hesse)
```

Commands use Bash. Update the paths for your machine, including hard-coded efficiency imports, input/reference paths, and output paths in the scripts. With `--with-eff`, an existing hard-coded signal HDF5 takes precedence over `--data`.

## 1. Massless and massive fits

`angularfitter_massless_fraction.py` fits the nominal massless sample and draws fit projections and reference fractions in m(Kπ) and q². `angularfitter_massive.py` does the corresponding massive-sample analysis, with reference plots showing component yields.

```bash
python -u angularfitter_massless_fraction.py \
  --data "$EFF/signal_with_efficiency_nominal_leon_massless.h5" "${COMMON[@]}"

python -u angularfitter_massive.py \
  --data "$EFF/signal_with_efficiency_new_leon.h5" "${COMMON[@]}"
```

These commands fit the complete selected input once. For bootstrap toys, append `--toy --nsig 90000 --ntoys 50` to either command. **`--nsig` means total mixed events; these versions do not enforce a 2:1 signal/background ratio.** Toy composition follows the input pool.

Fit YAMLs, parameter tables, and PDFs are saved under `results_cow_<configuration>/`; plots are made for the first three iterations. Toy event tables are saved as `coverage_toys/<i>.h5`. Full-sample weighted HDF5 paths are printed and currently point into `/work/submit/xiaot425/IAP2026/fitter/sweights/`.

A **toy** is one pseudoexperiment. A **coverage file** is its event table with kinematics, efficiency information, and component weights (`wA0`, `wApp`, `wS`, `wAq`, `wBkg`), saved with HDF5 key `data`. It is an input to uncertainty checks, not a separate dataset or a calculated coverage result. Fit weights are `eff_max / efficiency`, so weighted yields differ from event counts.

## 2. Reference checks: comparison_with_background.py

Reads a weighted HDF5 table and its matching YAML, then compares extracted components and their combinations with truth references in m(Kπ) and q². It does not rerun the fit. Reference normalization uses the fitted signal yield.

The reference plots produced by this script use the correct normalization. First, run `angularfitter_massless_fraction.py` or `angularfitter_massive.py` to generate the fitted results, then use `comparison_with_background.py` to plot the reference comparisons.

Example after running nominal massless toys:

```bash
mkdir -p plots_wide
RESULTS=results_cow_toy_massless_bkg_eff_Ig_mass_degree4_floatAngles_hesse
python -u comparison_with_background.py \
  --input "$RESULTS/coverage_toys/0.h5" --results "$RESULTS/0.yml" \
  --a0 "$LEON/massless_high_stats/Toyevents_forAnja_pw_longitudinal.root" \
  --a1 "$LEON/massless_high_stats/Toyevents_forAnja_pw_par_perp.root" \
  --aS "$LEON/massless_high_stats/Toyevents_forAnja_sw_longitudinal.root" \
  --qsq 1.1 19.0 --mKpi 0.746 1.5 --name massless_toy0
```

PDFs go to `plots_wide/`. For massive checks, use the massive results and `massive_highstats/` references. Component factors are hard-coded for specific selections; review them when changing cuts. This script expects S-wave components and is not used for the P-wave-only moments comparison.

## 3. P-wave-only COW sample

`angularfitter_cow_leon_PwaveOnly.py` fits a P-wave-only signal sample with background and saves the component weights for each toy.

```bash
python -u angularfitter_cow_leon_PwaveOnly.py \
  --data "$EFF/signal_with_efficiency_pw_leon_massless.h5" \
  "${COMMON[@]}" --toy --nsig 90000 --ntoys 50

PW=results_cow_leon_PwaveOnly_fixedFirst60k30k_SB2to1_bkg_eff_Ig_mass_degree4_floatAngles_hesse
```

Each toy contains 60,000 signal events and 30,000 background events, sampled independently from the selected event pools. Different toys use different event samples, while the sample size and signal/background composition remain fixed.

The output `coverage_toys/<i>.h5` contains the event-level kinematics, efficiency information, and COW weights (`wA0`, `wApp`, `wAq`, `wBkg`). The corresponding fit parameters are saved in `<i>.yml` in the parent results directory.

These are resampled pseudoexperiments from existing simulated events, not newly generated decays. The `--nsig` argument must match the total number of events per toy (90,000 for this configuration).

## 4. Mass-only signal weights and angular moments

`make_moment_sweights_massless_PwaveOnly.py` reads the saved COW event tables, performs a one-dimensional B-mass fit, and adds `signal_sweight`. It preserves the event sample so that the moment and COW methods can be compared using the same events.

Edit the script's `input_pattern` and `output_dir` to point to the desired COW toy files and moment output directory. The script has no CLI options.

```bash
python -u make_moment_sweights_massless_PwaveOnly.py
```

`moments.py` compares the P-wave COW fractions with the angular moments (`-S2c` and `4S2s`), using `signal_sweight / efficiency` for the moment calculation. Truth moments are calculated from the high-statistics P-wave ROOT sample.

For a single-toy comparison:

```bash
mkdir -p plots
MOM=/work/submit/xiaot425/IAP2026/results_moments_leon_massless_PwaveOnly_fixedFirst60k30k_mass_fitD4_2to1/coverage_toys

python -u moments.py \
  --input "$PW/coverage_toys/0.h5" \
  --moment-input "$MOM/0.h5" \
  --data "$LEON/massless_high_stats/Toyevents_forAnja_pw.root" \
  --name pw_toy0 \
  --qsq 1.1 19.0 --mKpi 0.746 1.5 \
  --nbins 30 --no-center --error-option per-toy
```

Use matching toy indices for `--input` and `--moment-input`. Both files must represent the same underlying events; matching filenames alone do not guarantee this.

The output is saved as `plots/pw_toy0_q2_moments_scatter.pdf`. Run the command separately for each toy to produce one plot per toy. When processing multiple toys together, the script combines information across toys rather than producing a separate plot for each one.

Use `--center` for a truth-centered uncertainty comparison. `--error-option per-toy` uses the propagated uncertainties, while `--error-option toy-std` uses the toy-to-toy standard deviation and therefore requires multiple independent toys.

## Run 2 and reproducibility

For nominal toys, change `--nsig` to the desired total count; matching the signal/background ratio requires adjusting the sampling code. These scripts do not perform a Wilson-coefficient fit. Their YAMLs currently lack usable off-diagonal covariance entries; save the full covariance separately for a correlated downstream fit. Commands were checked against source but have not been run here against subMIT inputs.

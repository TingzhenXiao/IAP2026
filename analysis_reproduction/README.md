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

## 2. Reference checks: comparison.py

Reads a weighted HDF5 table and its matching YAML, then compares extracted components and their combinations with truth references in m(Kπ) and q². It does not rerun the fit. Reference normalization uses the fitted signal yield.

Example after running nominal massless toys:

```bash
mkdir -p plots_wide
RESULTS=results_cow_toy_massless_bkg_eff_Ig_mass_degree4_floatAngles_hesse
python -u comparison.py \
  --input "$RESULTS/coverage_toys/0.h5" --results "$RESULTS/0.yml" \
  --a0 "$LEON/massless_high_stats/Toyevents_forAnja_pw_longitudinal.root" \
  --a1 "$LEON/massless_high_stats/Toyevents_forAnja_pw_par_perp.root" \
  --aS "$LEON/massless_high_stats/Toyevents_forAnja_sw_longitudinal.root" \
  --qsq 1.1 19.0 --mKpi 0.746 1.5 --name massless_toy0
```

PDFs go to `plots_wide/`. For massive checks, use the massive results and `massive_highstats/` references. Component factors are hard-coded for specific selections; review them when changing cuts. This script expects S-wave components and is not used for the P-wave-only moments comparison.

## 3. P-wave-only COW sample

`angularfitter_cow_leon_PwaveOnly.py` fits an existing massless P-wave-only signal sample plus background and saves component weights. **This version takes the first 60,000 selected signal and 30,000 background events; every iteration reuses the same events.** Use one iteration:

```bash
python -u angularfitter_cow_leon_PwaveOnly.py \
  --data "$EFF/signal_with_efficiency_pw_leon_massless.h5" \
  "${COMMON[@]}" --toy --nsig 90000 --ntoys 1

PW=results_cow_leon_PwaveOnly_fixedFirst60k30k_SB2to1_bkg_eff_Ig_mass_degree4_floatAngles_hesse
```

`$PW/coverage_toys/0.h5` contains P-wave COW weights and no `wS`. This script analyses existing simulation rather than generating new decays. Different sample sizes require editing the fixed counts; increasing `--ntoys` does not produce independent toys.

## 4. Mass-only signal weights and angular moments

`make_moment_sweights_massless_PwaveOnly.py` reads the saved COW events, fits B mass alone, and adds `signal_sweight` using 1D COWs with a degree-4 background expansion. It preserves the events for a paired comparison. Edit its top-level `input_pattern` and `output_dir` if needed; it has no CLI options.

```bash
python -u make_moment_sweights_massless_PwaveOnly.py
```

`moments.py` compares P-wave COW fractions with angular moments (`-S2c`, `4S2s`) using `signal_sweight / efficiency`, and truth moments from the high-statistics P-wave ROOT sample.

```bash
mkdir -p plots
MOM=/work/submit/xiaot425/IAP2026/results_moments_leon_massless_PwaveOnly_fixedFirst60k30k_mass_fitD4_2to1/coverage_toys
python -u moments.py \
  --input "$PW/coverage_toys/0.h5" --moment-input "$MOM/0.h5" \
  --data "$LEON/massless_high_stats/Toyevents_forAnja_pw.root" \
  --name pw_fixedsample --qsq 1.1 19.0 --mKpi 0.746 1.5 \
  --nbins 30 --no-center --error-option per-toy
```

Output: `plots/pw_fixedsample_q2_moments_scatter.pdf`. Use `--center` for a truth-centered uncertainty comparison. The 15 bins are distributed across the allowed q² regions. `toy-std` requires multiple independent toys and is unsuitable for repeated fits of this fixed sample.

## Run 2 and reproducibility

For nominal toys, change `--nsig` to the desired total count; matching the signal/background ratio requires adjusting the sampling code. These scripts do not perform a Wilson-coefficient fit. Their YAMLs currently lack usable off-diagonal covariance entries; save the full covariance separately for a correlated downstream fit. Commands were checked against source but have not been run here against subMIT inputs.

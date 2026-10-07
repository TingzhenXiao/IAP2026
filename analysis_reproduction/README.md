# Analysis reproduction

Fits, COW component extraction, reference checks, and an angular-moments comparison for simulated B0 → Kπμμ samples. These scripts analyse existing Leon samples; they do not generate decays from Wilson coefficients or perform a Wilson-coefficient fit.

This README describes the supplied script versions. In particular, the P-wave-only version is a **fixed-event comparison**, not an independent-toy ensemble.

## Scripts

| Script | Purpose |
| --- | --- |
| `angularfitter_massless_fraction.py` | Fit the nominal massless-muon sample, calculate component COW weights, and draw fit projections and reference-fraction plots. |
| `angularfitter_massive.py` | Fit the nominal massive-muon sample, calculate component COW weights, and draw fit projections and reference-yield checks. |
| `comparison.py` | Read an existing weighted HDF5 sample and its fit YAML; compare extracted components and combinations with separate truth-reference samples in m(Kπ) and q². |
| `angularfitter_cow_leon_PwaveOnly.py` | Fit an existing massless P-wave-only signal sample mixed with background and save event-level COW weights for the moments comparison. |
| `make_moment_sweights_massless_PwaveOnly.py` | Refit the same saved events in B mass only and add a signal weight for the angular-moments method. |
| `moments.py` | Compare P-wave fractions extracted with the multidimensional COW method against angular moments from the same events. |

Use the filenames above in this directory. Uploaded filenames with `(1)` or `(2)` suffixes should be renamed to these names before using the commands.

## Environment and supporting files

The analysis was run on subMIT in the Python 3.12 conda environment `sweights-py312`.

```bash
source /home/submit/xiaot425/miniforge3/etc/profile.d/conda.sh
conda activate sweights-py312
cd /path/to/repository/analysis_reproduction
```

Required Python packages include NumPy, pandas, PyTables (`tables`, for HDF5), PyYAML, uproot, awkward/awkward-pandas as needed by uproot, Matplotlib, mplhep, hist, SciPy, zfit, iminuit, hepstats, and sweights. The installed sweights version must expose `sweights.experimental.Cows`; use the versions from the working environment rather than assuming any version will work. To record that environment, run:

```bash
conda env export --no-builds > environment.yml
python -m pip freeze > requirements-used.txt
```

The fitters also import repository modules `tools.py`, `mypdfs.py`, `angularfunctions.py`, and `myconstants.py`, plus the efficiency module and its dependencies. These must be present and importable. `tools.parser()` provides shared options such as `--data`, `--background`, `--settings`, `--toy`, `--nsig`, `--use_basis`, `--qsq`, and `--mKpi`; preserve the matching version of `tools.py`.

If the supporting modules are in the repository's `fitter/` directory, for example:

```bash
export PYTHONPATH="../fitter:../efficiency:${PYTHONPATH:-}"
```

The scripts currently append `/home/submit/xiaot425/IAP2026/efficiency` to the import path. Update this path when running elsewhere. The efficiency module is required even when the fitters import it without using efficiency corrections.

The examples below use existing subMIT inputs:

```bash
LEON_BASE=/ceph/submit/data/user/a/anbeck/B2KPiMM_leon
EFF_BASE=/ceph/submit/data/user/x/xiaot425/efficiency/efficiency_applied_output
SETTINGS=/home/submit/xiaot425/IAP2026/fitter/settings/truth.yml
```

Replace these paths as necessary. The truth settings and all supporting modules must also be included in the repository for reproducibility. Large ROOT and HDF5 inputs can remain external, but their locations and provenance should be documented.

### Input selection and efficiency

With `--with-eff`, each fitter first tries its own **hard-coded efficiency-applied HDF5 path**. If that file exists, it takes precedence over `--data`. Only if the hard-coded file is absent does the script use an HDF5 path supplied through `--data`. Passing another ROOT file does not override the efficiency-applied sample.

| Fitter | Hard-coded signal HDF5 filename under `EFF_BASE` |
| --- | --- |
| Massless nominal | `signal_with_efficiency_nominal_leon_massless.h5` |
| Massive nominal | `signal_with_efficiency_new_leon.h5` |
| Massless P-wave only | `signal_with_efficiency_pw_leon_massless.h5` |

The HDF5 input uses key `data` and must contain the kinematic and efficiency columns expected by the fitter. With `--no-eff`, the signal is read from the ROOT file passed through `--data` instead. Background is read from ROOT; with efficiency enabled, it is subjected to efficiency acceptance in the script.

Fit weights are `eff_max / efficiency`. Therefore, selected event counts, sums of efficiency weights, and fitted extended yields are different quantities. A 2:1 ratio of selected event counts need not correspond to a 2:1 ratio of weighted yields.

## 1. Nominal massless-muon fit and reference plots

Run a single fit to the complete selected input sample:

```bash
python -u angularfitter_massless_fraction.py \
  --data "$EFF_BASE/signal_with_efficiency_nominal_leon_massless.h5" \
  --background "$LEON_BASE/background.root" \
  --background-tree background \
  --settings "$SETTINGS" \
  --use_basis \
  --qsq 1.1 19.0 \
  --mKpi 0.746 1.5 \
  --with-bkg --with-eff \
  --cow-I g \
  --bkg-series-degree 4 \
  --error-method hesse
```

Without `--toy`, the script performs one fit; “data” in the output name means the full input simulation, not collision data.

For bootstrap pseudoexperiments, add the following options to the same command:

```bash
  --toy --nsig 90000 --ntoys 50
```

Add them before executing the command, not as a separate shell command. Here `--nsig` is the **total number of mixed events** sampled with replacement per pseudoexperiment, despite its name. `--ntoys` defaults to 100 in toy mode; specify it explicitly.

**This supplied version does not force a 2:1 signal-to-background ratio.** It samples from the combined selected signal/background pool, so the average composition follows that pool and the composition fluctuates between toys. Saving a coverage file does not change the ratio.

With the options above and floating angular parameters, results are under:

- Full sample: `results_cow_data_massless_bkg_eff_Ig_mass_degree4_floatAngles_hesse/`
- Toy mode: `results_cow_toy_massless_bkg_eff_Ig_mass_degree4_floatAngles_hesse/`

Outputs include `<i>.yml`, parameter tables, and (for the first three successful iterations) fit-projection PDFs for B mass and the angles, and reference PDFs for m(Kπ) and q². The massless reference-summary function normalizes components **within each bin**: it compares relative fractions, rather than absolute yields. Its normal reference plot includes background in the denominator when background is enabled; its efficiency-reference plot uses only the three supplied signal components. The displayed `n_beta` is outside that denominator.

Truth-reference paths are hard-coded in `reference_mapping`: the massless `pw_longitudinal`, `pw_par_perp`, and `sw_longitudinal` ROOT samples under `massless_high_stats/`. Change this mapping when moving the inputs.

## 2. Nominal massive-muon fit and reference plots

```bash
python -u angularfitter_massive.py \
  --data "$EFF_BASE/signal_with_efficiency_new_leon.h5" \
  --background "$LEON_BASE/background.root" \
  --background-tree background \
  --settings "$SETTINGS" \
  --use_basis \
  --qsq 1.1 19.0 \
  --mKpi 0.746 1.5 \
  --with-bkg --with-eff \
  --cow-I g \
  --bkg-series-degree 4 \
  --error-method hesse
```

For bootstrap pseudoexperiments, add `--toy --nsig 90000 --ntoys 50` to this command. As in the massless nominal script, `--nsig` is the total mixed count, and **no fixed 2:1 ratio is enforced**.

With these options, results are under:

- Full sample: `results_cow_data_massive_bkg_eff_Ig_mass_degree4_floatAngles_hesse/`
- Toy mode: `results_cow_toy_massive_bkg_eff_Ig_mass_degree4_floatAngles_hesse/`

The script produces fit projections and reference checks using the massive samples. Its reference-summary function compares scaled component yields, unlike the per-bin fraction display in the massless script. The massive truth-reference paths are hard-coded in `reference_mapping` under `massive_highstats/`.

Both nominal fitters use `--bkg-series-degree 4` for the COW background mass expansion. This is distinct from the fitted background-shape options `--bkg-degree-mass` (default 2), `--bkg-degree-cosh` (0), `--bkg-degree-cosl` (2), and `--bkg-cross-terms` (off).

## What are toy and coverage files?

A **toy** is one simulated pseudoexperiment used to test the analysis. In the nominal fitters it is a bootstrap sample of existing events, not a new decay simulation generated from a physics model.

A file named `coverage_toys/<i>.h5` is that iteration's event table **after adding component COW weights**. It contains kinematics, `is_signal`, efficiency information, and weights such as `wA0`, `wApp`, `wS`, `wAq`, and `wBkg`. The P-wave-only version omits `wS`. Files are written with HDF5 key `data`.

“Coverage” refers to their intended use in repeated-pseudoexperiment uncertainty checks. These HDF5 files are inputs to such checks; creating them does not itself calculate confidence-interval coverage. They are not an additional independent set of events.

In toy mode, the fitters save these tables under their result directory's `coverage_toys/`. In full-sample mode, weighted HDF5 tables are instead saved under the hard-coded location:

```text
/work/submit/xiaot425/IAP2026/fitter/sweights/<polynomial>/<analysis-name>_mass_degree4/0.h5
```

The exact path is printed by the script. Update the output path for another account or machine. Different runs with the same configuration reuse filenames and can overwrite outputs; retain separate run directories.

## 3. Component and combination checks: comparison.py

This is a plotting/check script; it does not rerun the fit. It reads:

1. The event-level weighted HDF5 table from a fitter.
2. The matching fit-results YAML.
3. Separate longitudinal P-wave (`--a0`), parallel/perpendicular P-wave (`--a1`), and S-wave (`--aS`) truth ROOT samples.

It compares extracted component yields and four combinations of components against reference distributions in m(Kπ) and q², with residual/pull panels. The reference normalization is `factor[component] * fitted_Nsig / len(reference)`, so this check uses the **fitted signal yield**, unlike a reference normalized to each toy's known truth yield. It does not draw a background reference component.

Example using nominal massless toy 0:

```bash
mkdir -p plots_wide
MASSLESS_RESULTS=results_cow_toy_massless_bkg_eff_Ig_mass_degree4_floatAngles_hesse
python -u comparison.py \
  --input "$MASSLESS_RESULTS/coverage_toys/0.h5" \
  --results "$MASSLESS_RESULTS/0.yml" \
  --a0 "$LEON_BASE/massless_high_stats/Toyevents_forAnja_pw_longitudinal.root" \
  --a1 "$LEON_BASE/massless_high_stats/Toyevents_forAnja_pw_par_perp.root" \
  --aS "$LEON_BASE/massless_high_stats/Toyevents_forAnja_sw_longitudinal.root" \
  --qsq 1.1 19.0 --mKpi 0.746 1.5 \
  --name massless_toy0
```

For massive checks, use the matching massive HDF5/YAML pair and replace `massless_high_stats` with `massive_highstats` in all reference paths.

Outputs are `plots_wide/<name>_mKpi_comparison.pdf`, `<name>_q2_comparison.pdf`, and the corresponding `_comparison_check.pdf` files. The code uses 100 bins. With `Abeta` present in the results, component factors are hard-coded for only the q² ranges `[1.1,19]`, `[0.06,19]`, and `[0.1,19]`, and selected using `massless`/`massive` in the S-wave reference path. Recalculate these factors for changed selections; arbitrary q² ranges are not supported in that branch.

This script expects S-wave weights/references and is not the P-wave-only moments comparison.

## 4. P-wave-only weighted sample

`angularfitter_cow_leon_PwaveOnly.py` uses the existing efficiency-applied **P-wave-only signal input**; it does not remove S-wave events from a nominal sample or generate a new P-wave decay simulation. It fits the signal plus background and extracts P-wave component weights.

**The supplied version selects the first 60,000 accepted signal events and first 30,000 accepted background events. In toy mode it copies exactly this same 90,000-event table each iteration, without resampling.** Use one iteration for this fixed-sample comparison:

```bash
python -u angularfitter_cow_leon_PwaveOnly.py \
  --data "$EFF_BASE/signal_with_efficiency_pw_leon_massless.h5" \
  --background "$LEON_BASE/background.root" \
  --background-tree background \
  --settings "$SETTINGS" \
  --toy --nsig 90000 --ntoys 1 \
  --use_basis \
  --qsq 1.1 19.0 --mKpi 0.746 1.5 \
  --with-bkg --with-eff \
  --cow-I g --bkg-series-degree 4 --error-method hesse
```

The output result directory is:

```text
results_cow_leon_PwaveOnly_fixedFirst60k30k_SB2to1_bkg_eff_Ig_mass_degree4_floatAngles_hesse/
```

`coverage_toys/0.h5` contains the same events to use for the mass-only moments weights, with `wA0`, `wApp`, `wAq`, and `wBkg`. Other plots use legacy `plots_degree/` paths; the script also assigns a generic `reference_mass_degree4.pdf` path, so retain the printed paths and do not assume all plots are in the result directory.

Changing `--ntoys` does not produce independent samples in this version. Changing `--nsig` away from 90,000 raises an error. The fixed event counts must be changed in the source to use another size; the unconditional fixed-background selection also makes `--no-bkg` unsuitable for this version.

## 5. Signal weights for the moments method

`make_moment_sweights_massless_PwaveOnly.py` reads the P-wave-only COW HDF5 files from step 4 and preserves their events. It performs a **one-dimensional B-mass fit**, then constructs one-dimensional COW weights using a background mass expansion of degree 4. It adds the column `signal_sweight` and writes a corresponding HDF5 file for every input iteration.

Despite its filename and column name, this version calculates that weight with `sweights.experimental.Cows`; it is not a direct call to ordinary `hepstats.compute_sweights`. Efficiency correction is applied later in `moments.py` as `signal_sweight / efficiency`.

This script has no command-line parser. Before running, inspect/edit its top-level `input_pattern` and `output_dir`. The supplied defaults are:

```python
input_pattern = (
    "./results_cow_leon_PwaveOnly_fixedFirst60k30k_SB2to1_bkg_eff_Ig_mass_degree4_floatAngles_hesse/"
    "coverage_toys/*.h5"
)
output_dir = (
    "/work/submit/xiaot425/IAP2026/"
    "results_moments_leon_massless_PwaveOnly_fixedFirst60k30k_mass_fitD4_2to1/"
    "coverage_toys"
)
```

Then run from `analysis_reproduction/`:

```bash
python -u make_moment_sweights_massless_PwaveOnly.py
```

Numeric filenames are preserved (`0.h5`, etc.). Reusing the exact saved events avoids comparing the two methods on separate random samples.

## 6. Angular moments versus P-wave COW fractions

`moments.py` compares, bin by bin in q²:

- COW fractions `sum(wA0) / sum(wA0 + wApp)` and `sum(wApp) / sum(wA0 + wApp)`.
- Angular-moment estimates of `-S2c` and `4S2s`, using the mass-only signal weights divided by efficiency.
- Truth moments from an independent high-statistics P-wave-only ROOT sample supplied with `--data`.

For the single fixed sample from steps 4 and 5:

```bash
mkdir -p plots
PW_RESULTS=results_cow_leon_PwaveOnly_fixedFirst60k30k_SB2to1_bkg_eff_Ig_mass_degree4_floatAngles_hesse
MOMENT_RESULTS=/work/submit/xiaot425/IAP2026/results_moments_leon_massless_PwaveOnly_fixedFirst60k30k_mass_fitD4_2to1/coverage_toys
python -u moments.py \
  --input "$PW_RESULTS/coverage_toys/0.h5" \
  --moment-input "$MOMENT_RESULTS/0.h5" \
  --data "$LEON_BASE/massless_high_stats/Toyevents_forAnja_pw.root" \
  --name leon_massless_PwaveOnly_fixedsample_15bins \
  --qsq 1.1 19.0 --mKpi 0.746 1.5 \
  --nbins 15 \
  --no-center \
  --error-option per-toy
```

Output: `plots/leon_massless_PwaveOnly_fixedsample_15bins_q2_moments_scatter.pdf`.

Use `--center` instead of `--no-center` to center the displayed estimates at the truth values for an uncertainty comparison; the suffix becomes `_moments_center.pdf`. This changes the displayed central values and should not be used to assess bias.

The current code plots q² only. `--nbins` distributes the requested total across the hard-coded allowed regions `[1.1,9]`, `[11,12.5]`, and `[15,19]`; it is not a request for exact 1 GeV² bins. If changing `--qsq`, also review these hard-coded regions.

For an actual independent ensemble, pass the matching sets of COW and moments files with `*.h5`. With multiple files, the code displays the first COW sample and first valid moments sample as the central values. `per-toy` uses the median propagated COW uncertainty and mean propagated moments uncertainty across the supplied samples. `toy-std` instead uses sample-to-sample standard deviations and adds `_toy_std` to the filename. It requires independent samples and at least two files; repeated fits of the fixed sample are not suitable.

## Adapting to Run 2 statistics

For the nominal massless/massive fitters, `--nsig N` changes the total selected mixed-event count in toy mode. Match both the selected signal count and the background composition to the intended Run 2 selection; changing the total alone does not set the signal-to-background ratio. To enforce specific counts, the current sampling logic must be changed to sample signal and background separately.

For the supplied fixed P-wave-only script, edit `n_sig_fixed` and `n_bkg_fixed` and use their sum as `--nsig`. To obtain independent pseudoexperiments, replace its fixed-copy logic with a documented sampling procedure before running an ensemble.

A Wilson-coefficient fit is a subsequent analysis. These scripts do not supply that fit or the published LHCb Run 2 measurements. In addition, the supplied fitters initialize the covariance array to NaN and write off-diagonal YAML entries as null without populating it. Parameter errors are calculated, but a usable full covariance matrix must be explicitly calculated and saved before relying on these outputs for a correlated downstream fit.

## Reproduction status

The commands and output descriptions above were checked against the supplied source files. They have not been executed against the subMIT data here: the external ROOT/HDF5 inputs, supporting modules, and original conda environment were not supplied. Include these dependencies and record the working environment before treating a fresh clone as self-contained.


# Multicomponent COWs Angular Analysis

This repository performs a three-dimensional extended unbinned maximum-likelihood fit and computes multicomponent COWs (Custom Orthogonal Weights) for an angular analysis of \(B \to K^*\mu^+\mu^-\) candidates.

The fit uses the observables

\[
\left(\cos\theta_K,\ \cos\theta_\ell,\ m_B\right),
\]

represented in the code by:

- `cosh`: \(\cos\theta_K\);
- `cosl`: \(\cos\theta_\ell\);
- `B_mass`: reconstructed \(B\)-candidate mass.

The script supports signal and background samples, efficiency correction, toy studies, different COW normalization choices, parameter fixing and constraints, and diagnostic plots.

## Features

- Three-dimensional signal-plus-background fit using `zfit`
- Extended unbinned negative log-likelihood
- Double Crystal Ball signal mass model
- Two-exponential background mass model
- Two-dimensional signal angular model
- Factorized Legendre background angular model
- Multicomponent COW extraction for:
  - \(A_0\)
  - \(A_{\parallel,\perp}\)
  - \(A_S\)
  - \(A_q\)
  - combinatorial background
- Optional efficiency correction
- Optional background model
- COW normalization choices \(I=g\) and \(I=q\)
- Configurable background mass-series degree
- Bootstrap-style toy studies
- Parameter fixing and Gaussian constraints
- Fit-result, COW-weight, and projection-plot output

## Analysis Workflow

The script performs the following steps:

1. Reads the signal and optional background samples.
2. Applies the analysis selection.
3. Applies efficiency acceptance and inverse-efficiency weights if requested.
4. Combines and shuffles the signal and background samples.
5. Builds the angular and mass PDFs.
6. Performs an extended unbinned maximum-likelihood fit.
7. Builds the multicomponent COW basis.
8. Computes the component weights.
9. Applies the efficiency correction to the COW weights.
10. Saves fit results, toy-study samples, and diagnostic plots.

## Requirements

The script requires Python 3 and the following packages:

```text
numpy
pandas
scipy
matplotlib
mplhep
uproot
zfit
hist
hepstats
sweights
PyYAML
h5py
```

Install the public dependencies with:

```bash
python -m pip install \
    numpy pandas scipy matplotlib mplhep uproot zfit hist \
    hepstats sweights PyYAML h5py
```

The analysis also depends on the following local modules:

```text
myconstants.py
tools.py
mypdfs.py
angularfunctions.py
efficiency.py
```

Their roles are:

| Module | Purpose |
|---|---|
| `myconstants.py` | Analysis constants and shared configuration |
| `tools.py` | Command-line parser and directory utilities |
| `mypdfs.py` | Custom angular PDFs and analytic integrals |
| `angularfunctions.py` | Angular projection functions |
| `efficiency.py` | Multidimensional efficiency parameterization |

The script currently adds a hard-coded efficiency-module path:

```python
sys.path.append("/home/submit/xiaot425/IAP2026/efficiency")
```

Update this path before running in a different environment. A portable alternative is to store `efficiency.py` in the repository and import it directly.

## Input Data

### Signal sample

With efficiency correction enabled, the script first attempts to read:

```text
/ceph/submit/data/user/x/xiaot425/efficiency/efficiency_applied_output/signal_with_efficiency.h5
```

The HDF5 key must be:

```text
data
```

The sample is expected to contain at least:

```text
B_mass
cosThetaK
cosThetaL
q2
mKpi
efficiency
```

If `cosh` and `cosl` are not present, they are created using:

```python
df["cosh"] = df["cosThetaK"]
df["cosl"] = df["cosThetaL"]
```

Without efficiency correction, the signal sample is read from the ROOT file specified by the standard `--data` option. The default signal tree is:

```text
B02KstMuMu_Run1_centralQ2E_sig
```

### Background sample

When background is enabled, the background ROOT file and tree are specified through the standard parser options, for example:

```bash
--background background.root
--background-tree BackgroundTree
```

The background tree must contain:

```text
B_mass
cosThetaK
cosThetaL
q2
mKpi
```

### Mass units

If the largest value of `B_mass` is greater than 100, the script assumes that the mass is stored in MeV and converts it to GeV:

```python
df["B_mass"] = df["B_mass"] / 1000.0
```

## Event Selection

The following selection is applied to both signal and background:

\[
1.1 < q^2 < 7.0,
\]

\[
m_{K\pi} < 1.5,
\]

and

\[
5.170 \leq m_B \leq 5.500~\mathrm{GeV}.
\]

Rows containing missing values are removed after the selection.

## Efficiency Treatment

When efficiency correction is enabled, the event weight is defined as

\[
w_i^{\mathrm{eff}} =
\frac{\varepsilon_{\max}}{\varepsilon_i}.
\]

For the signal sample, the efficiency values are read from the input HDF5 file.

For the background sample, the script:

1. evaluates the efficiency function;
2. applies acceptance-rejection sampling;
3. assigns inverse-efficiency weights to the accepted events.

When efficiency correction is disabled, all efficiencies and fit weights are set to one.

## Command-Line Options

The script adds the following options to those provided by `tools.parser()`:

| Option | Default | Description |
|---|---:|---|
| `--with-bkg` | enabled | Include the background model |
| `--no-bkg` | — | Disable the background model |
| `--with-eff` | enabled | Enable efficiency correction |
| `--no-eff` | — | Disable efficiency correction |
| `--cow-I {g,q}` | `g` | Select the COW normalization function |
| `--bkg-series-degree N` | `4` | Maximum degree of the background mass series |
| `--ntoys N` | `None` | Number of toy experiments |
| `--error-method {hesse,minos}` | `hesse` | Requested parameter-error method |

Because background and efficiency correction are enabled by default, use `--no-bkg` or `--no-eff` to disable them explicitly.

Additional options are defined in `tools.parser()`. Depending on the local implementation, these may include:

```text
--data
--background
--background-tree
--settings
--polynomial
--toy
--nsig
--binned
--fix-to-zero
--fix-to-value
--fix-to-truth
--constrain
```

Run the following command to inspect all available options:

```bash
python <script_name.py> --help
```

## Configuration File

The parameter truth values are read from YAML or JSON.

### YAML example

```yaml
App:
  value: 0.167
  error_lower: -0.010
  error_upper: 0.010

A0:
  value: 0.500
  error_lower: -0.020
  error_upper: 0.020

Aqc:
  value: 0.010
  error_lower: -0.005
  error_upper: 0.005

Aqs:
  value: 0.010
  error_lower: -0.005
  error_upper: 0.005

AfbHS:
  value: 0.0
  error_lower: -0.1
  error_upper: 0.1

AfbHC:
  value: 0.0
  error_lower: -0.1
  error_upper: 0.1

AfbLS:
  value: 0.0
  error_lower: -0.1
  error_upper: 0.1

AfbLC:
  value: 0.0
  error_lower: -0.1
  error_upper: 0.1

Nsig:
  value: 10000
```

The uncertainty fields are required when a Gaussian constraint is applied.

### JSON example

```json
{
  "App": 0.167,
  "A0": 0.500,
  "Aqc": 0.010,
  "Aqs": 0.010,
  "AfbHS": 0.0,
  "AfbHC": 0.0,
  "AfbLS": 0.0,
  "AfbLC": 0.0,
  "Nsig": 10000
}
```

For constraints, YAML is recommended because it can store asymmetric uncertainty fields directly.

## Fit Model

### Signal angular model

The two-dimensional signal angular distribution is implemented by:

```python
mypdfs.my2Dpdf
```

The independent angular parameters are:

```text
App
A0
Aqc
Aqs
AfbHS
AfbHC
AfbLS
AfbLC
```

The S-wave coefficient is a composed parameter:

\[
A_S = 1-A_0-A_{\parallel,\perp}-A_{qc}-A_{qs}.
\]

The corresponding component yields are

\[
N_{AS} = N_{\mathrm{sig}} A_S,
\]

\[
N_{A0} = N_{\mathrm{sig}} A_0,
\]

\[
N_{App} = N_{\mathrm{sig}} A_{\parallel,\perp},
\]

and

\[
N_{Aq} = N_{\mathrm{sig}}(A_{qc}+A_{qs}).
\]

These definitions imply

\[
N_{\mathrm{sig}}
=
N_{AS}+N_{A0}+N_{App}+N_{Aq}.
\]

### Signal mass model

The signal mass model is a weighted sum of two Double Crystal Ball PDFs:

```python
zfit.pdf.DoubleCB
```

The two components share the mean `mu_sig` and have separate width and tail parameters. Their relative fraction is controlled by `frac_cb1`.

### Background mass model

The background mass distribution is a sum of two exponential PDFs with parameters:

```text
lambda_bkg_1
lambda_bkg_2
frac_bkg_exp1
```

### Background angular model

The background angular distribution is modeled as a product of two Legendre PDFs:

```python
fitpdf_bkg_ang = zfit.pdf.ProductPDF(
    [fitpdf_bkg_cosh, fitpdf_bkg_cosl],
    obs=angles,
)
```

The Legendre coefficients are fixed in the current implementation.

### Full extended model

With background enabled, the full model is

\[
F(x)=
N_{\mathrm{sig}}f_{\mathrm{sig}}(x)
+
N_{\mathrm{bkg}}f_{\mathrm{bkg}}(x).
\]

Without background, `Nbkg` is fixed to zero and only the signal PDF is fitted.

## Multicomponent COWs

The COW basis contains the four signal components:

```text
A0
App
AS
Aq
```

When background is enabled, additional background basis functions are included.

### Background mass-series basis

For a maximum degree \(D\), the background basis contains \(D+1\) functions:

\[
g_{b,r}(x)
=
\frac{g_{b,0}(x)u_m^r}{Z_r},
\qquad
r=0,\ldots,D,
\]

where

\[
u_m =
\frac{m_B-5.170}{5.500-5.170}.
\]

Each basis function is numerically normalized using Monte Carlo integration.

For example,

```bash
--bkg-series-degree 4
```

creates five background basis functions with degrees 0 through 4.

### COW normalization

Two normalization choices are available:

```bash
--cow-I g
```

uses the mixture-based normalization provided by the `sweights` package.

```bash
--cow-I q
```

uses an externally constructed histogram-based normalization.

## Running the Analysis

Replace `<script_name.py>` with the actual script filename.

### Signal and background with efficiency correction

```bash
python <script_name.py> \
    --data signal.root \
    --background background.root \
    --background-tree BackgroundTree \
    --settings settings.yml \
    --polynomial nominal \
    --with-bkg \
    --with-eff \
    --cow-I g \
    --bkg-series-degree 4 \
    --error-method hesse
```

### Signal-only fit

```bash
python <script_name.py> \
    --data signal.root \
    --settings settings.yml \
    --polynomial nominal \
    --no-bkg \
    --with-eff \
    --cow-I g
```

### Fit without efficiency correction

```bash
python <script_name.py> \
    --data signal.root \
    --background background.root \
    --background-tree BackgroundTree \
    --settings settings.yml \
    --polynomial nominal \
    --with-bkg \
    --no-eff \
    --cow-I g
```

### COWs with \(I=q\)

```bash
python <script_name.py> \
    --data signal.root \
    --background background.root \
    --background-tree BackgroundTree \
    --settings settings.yml \
    --polynomial nominal \
    --with-bkg \
    --with-eff \
    --cow-I q
```

### Toy study

```bash
python <script_name.py> \
    --data signal.root \
    --background background.root \
    --background-tree BackgroundTree \
    --settings settings.yml \
    --polynomial toy_study \
    --toy \
    --nsig 10000 \
    --ntoys 100 \
    --with-bkg \
    --with-eff \
    --cow-I g \
    --bkg-series-degree 4
```

A short test run can be performed with:

```bash
--ntoys 1
```

## Parameter Control

Fix parameters to zero:

```bash
--fix-to-zero Aqs AfbHS
```

Fix parameters to specified values:

```bash
--fix-to-value A0 0.50 App 0.167
```

Fix parameters to values from the truth file:

```bash
--fix-to-truth A0 App
```

Apply Gaussian constraints:

```bash
--constrain A0 App
```

For constrained parameters, the YAML file must provide `value`, `error_lower`, and `error_upper`.

## Toy Studies

In toy mode, the script:

1. draws the total sample size from a Poisson distribution;
2. samples events with replacement from the combined input dataset;
3. fits each resampled dataset;
4. calculates parameter pulls;
5. stores the fitted values and uncertainties.

The procedure is bootstrap-like: toy events are resampled from the available input events rather than generated directly from the fitted continuous PDF.

For each toy, the signal and background yield truth values are defined by the generated weighted sums:

```python
truth_values["Nsig"] = sum of signal fit weights
truth_values["Nbkg"] = sum of background fit weights
```

## Output

### Fit results

Fit results are written to:

```text
results_cow_<case_tag>/
```

The case tag records:

- background configuration;
- efficiency configuration;
- COW normalization choice;
- background-series degree;
- fixed or floating angular parameters;
- requested error method.

Each fit produces:

```text
<i>.yml
<i>_parameters_with_uncertainties.csv
<i>_parameters_with_uncertainties.txt
```

For example:

```text
results_cow_bkg_eff_Ig_bkgSeriesDegree4_floatAngles_hesse/
├── 0.yml
├── 0_parameters_with_uncertainties.csv
└── 0_parameters_with_uncertainties.txt
```

### Toy COW weights

COW-weighted toy samples are written to:

```text
results_cow_<case_tag>/coverage_toys/<i>.h5
```

The saved columns include:

```text
wA0
wApp
wS
wAq
wBkg
```

The saved weights include the efficiency correction:

\[
w_k^{\mathrm{final}}
=
w_k^{\mathrm{COW}}w^{\mathrm{eff}}.
\]

### Fit projections

Fit projections are written to:

```text
plots_degree/<polynomial>/<name>/fit_projections_cow_<case_tag>/
```

The output includes projections of:

```text
cosh
cosl
B_mass
```

### Reference projections

In non-toy mode, the script can compare the extracted components with reference samples.

Set the reference-data directory using:

```bash
export DATADIR=/path/to/reference/files
```

The directory is expected to contain:

```text
A0.root
A1.root
AS.root
```

Reference plots are written to:

```text
plots_degree/<polynomial>/<name>/reference_cow_multicomp_<case_tag>/
```

## Validation Checks

The script performs basic PDF checks before fitting:

- no non-positive PDF values;
- no `NaN` values;
- no infinite values;
- finite log-PDF values.

After COW extraction, compare:

```text
sum(wA0_final)   with N_A0
sum(wApp_final)  with N_App
sum(wAS_final)   with N_AS
sum(wAq_final)   with N_Aq
sum(wBkg_final)  with Nbkg
```

The total weighted sum should be compatible with the total fitted signal-plus-background yield.

The COW matrix condition number is also printed. A very large condition number may indicate that the component basis is poorly conditioned.

## Known Implementation Notes

### Duplicate `make_q_norm` definition

The script currently defines `make_q_norm` twice. In Python, the second definition replaces the first one.

The later definition does not accept the keyword arguments:

```text
pdfs_cow
yields_cow
smooth_sigma
```

while the \(I=q\) branch calls it with these arguments. As written, this will raise a `TypeError`.

Before using:

```bash
--cow-I q
```

remove or rename the duplicate implementation and retain the intended function.

### Error calculation

The `--error-method` option is stored and added to output names, but the script should explicitly execute the selected uncertainty calculation after minimization.

A typical implementation is conceptually:

```python
if args.error_method == "hesse":
    result.hesse()
else:
    result.errors()
```

The exact API should match the installed `zfit` version.

### Covariance matrix

The covariance matrix is currently initialized with `NaN` values and is not filled from the fit result. Consequently, saved covariance entries are generally `null`.

Populate `covmat` from the fit result before producing a correlation matrix or writing covariance values.

### Fit failure handling

The script prints `Fit not valid.` when a fit is invalid, but in non-toy mode it does not immediately stop. Consider skipping result extraction or raising an exception after an invalid fit.

### Hard-coded paths

The efficiency input and module search paths are currently hard-coded. For portability, replace them with command-line options, environment variables, or repository-relative paths.

### COW code in toy mode

The toy loop currently executes `continue` before reaching the COW calculation. Therefore, the COW-weighted coverage files in the later block are not produced in toy mode as written.

If toy COW outputs are required, move the COW calculation before the toy cleanup and `continue`.

## Recommended Repository Layout

```text
project/
├── README.md
├── fit_script.py
├── myconstants.py
├── tools.py
├── mypdfs.py
├── angularfunctions.py
├── settings.yml
├── requirements.txt
├── efficiency/
│   └── efficiency.py
├── tests/
├── plots_degree/
└── results/
```

Large ROOT and HDF5 files should generally not be committed to Git. Add them to `.gitignore` or store them using an appropriate external data service.

Example `.gitignore` entries:

```gitignore
__pycache__/
*.py[cod]
.ipynb_checkpoints/

*.root
*.h5
*.hdf5

results_cow_*/
plots_degree/

.DS_Store
```

## Reproducibility

The script initializes NumPy and zfit with seed zero:

```python
np.random.seed(0)
zfit.settings.set_seed(0)
```

Each toy then receives a newly generated seed. The efficiency acceptance-rejection step for the background uses a fixed seed of `12345`.

For full reproducibility, record:

- the Git commit hash;
- the Python environment;
- input dataset versions;
- truth/configuration files;
- the complete command used to run the script.

## License

Add the license used by this project, for example:

```text
MIT License
```

or the relevant collaboration/internal-use statement.

## Contact

For questions about this analysis, please contact:

```text
Tingzhen Xiao
Massachusetts Institute of Technology
```

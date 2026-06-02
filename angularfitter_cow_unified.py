import numpy as np  # Numerical library
import yaml  # For reading YAML files
import uproot  # For reading ROOT files
import matplotlib.pyplot as plt  # Plotting library
import zfit  # Fitting library
import hist  # Histogram library
from hepstats.splot import compute_sweights  # For sWeights computation
import json  # For reading JSON files
import argparse
from myconstants import *
import tools  # Some helpful functions
import mypdfs  # Custom pdfs
import angularfunctions as af  # Angular functions
import os
import pandas as pd
from sweights.experimental import Cows
import sys
sys.path.append("/home/submit/xiaot425/IAP2026/efficiency")
import efficiency

# Extra command-line options added in this unified script:
#   --with-bkg / --no-bkg
#   --with-eff / --no-eff
#   --cow-I g / --cow-I q

# Makes nice default plots
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

np.random.seed(0)
zfit.settings.set_seed(0)
zfit.settings.set_verbosity(10)


_cow_arg_parser = argparse.ArgumentParser(add_help=False)
_cow_arg_parser.add_argument("--with-bkg", dest="with_bkg", action="store_true", default=True)
_cow_arg_parser.add_argument("--no-bkg", dest="with_bkg", action="store_false")
_cow_arg_parser.add_argument("--with-eff", dest="with_eff", action="store_true", default=True)
_cow_arg_parser.add_argument("--no-eff", dest="with_eff", action="store_false")
_cow_arg_parser.add_argument("--cow-I", dest="cow_I", choices=["g", "q"], default=os.environ.get("COW_I", "g"))
_cow_cli_args, _remaining_argv = _cow_arg_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

args = tools.parser()
args.with_bkg = bool(_cow_cli_args.with_bkg)
args.with_eff = bool(_cow_cli_args.with_eff)
args.cow_I = str(_cow_cli_args.cow_I).lower()

print("COW configuration:")
print("  with_bkg =", args.with_bkg)
print("  with_eff =", args.with_eff)
print("  cow_I    =", args.cow_I)


if args.toy:
    name = "toy"
else:
    name = "data"


if len(args.fix_to_zero) > 0:
    for n in args.fix_to_zero:
        name += f"_{n}=0"
if len(args.fix_to_value) > 0:
    for n in range(0, len(args.fix_to_value), 2):
        name += f"_{args.fix_to_value[n]}={args.fix_to_value[n+1]}"
if len(args.fix_to_truth) > 0:
    for n in args.fix_to_truth:
        name += f"_{n}"
if len(args.constrain) > 0:
    for n in args.constrain:
        name += f"_{n}=constraint"


if len(args.qsq) == 2:
    name += f"_qsq-{args.qsq[0]}-{args.qsq[1]}"

name += "_withbkg" if args.with_bkg else "_nobkg"
name += "_witheff" if args.with_eff else "_noeff"
name += f"_I{args.cow_I}"

tools.makedirs(args.polynomial, name)

# limits for integrals
limith = zfit.Space(axes=0, lower=-1, upper=1)
limitl = zfit.Space(axes=1, lower=-1, upper=1)
limits = limith * limitl

# create phsp
cosh = zfit.Space('cosh', limits=(-1, 1))
cosl = zfit.Space('cosl', limits=(-1, 1))
angles = cosh * cosl

mass = zfit.Space("B_mass", limits=(5.17, 5.50))
obs = angles * mass

# Read signal sample
if args.with_eff:
    efficiency_input = "/home/submit/xiaot425/IAP2026/efficiency/efficiency_applied_output/signal_with_efficiency.h5"
    if os.path.exists(efficiency_input):
        df_sig = pd.read_hdf(efficiency_input, key="data").copy()
    elif str(args.data).endswith((".h5", ".hdf", ".hdf5")):
        df_sig = pd.read_hdf(args.data, key="data").copy()
    else:
        raise FileNotFoundError(
            "--with-eff requires the efficiency-applied HDF5 signal sample. "
            f"Tried {efficiency_input}."
        )
else:
    signal_tree = getattr(args, "signal_tree", "B02KstMuMu_Run1_centralQ2E_sig")
    with uproot.open(args.data) as f:
        df_sig = f[signal_tree].arrays(
            ["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi"],
            library="pd",
        )

if "cosl" not in df_sig.columns:
    df_sig["cosl"] = df_sig["cosThetaL"]
if "cosh" not in df_sig.columns:
    df_sig["cosh"] = df_sig["cosThetaK"]
if df_sig["B_mass"].max() > 100.0:
    df_sig["B_mass"] = df_sig["B_mass"] / 1000.0

df_sig = df_sig[(df_sig["q2"] > 1.1) & (df_sig["q2"] < 7.0)].copy()
df_sig = df_sig[(df_sig["mKpi"] < 1.5)].copy()
df_sig = df_sig[(df_sig["B_mass"] >= 5.170) & (df_sig["B_mass"] <= 5.500)].copy()
df_sig.dropna(inplace=True)
df_sig["is_signal"] = 1

if args.with_eff:
    if "eff_max" in df_sig.columns:
        eff_max = float(df_sig["eff_max"].iloc[0])
    else:
        eff_max = df_sig["efficiency"].max()
    df_sig["fit_weight"] = eff_max / df_sig["efficiency"].to_numpy(dtype=float)
else:
    df_sig["efficiency"] = 1.0
    df_sig["eff_max"] = 1.0
    df_sig["fit_weight"] = 1.0

print("Signal unweighted events:", len(df_sig))
print("Signal weighted sum:", df_sig["fit_weight"].sum())
print("Mean signal weight:", df_sig["fit_weight"].mean())

# Read background sample
if args.with_bkg:
    with uproot.open(args.background) as f:
        arr_bkg = f[args.background_tree].arrays(
            ["B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi"],
            library="np",
        )

    df_bkg = pd.DataFrame({
        "B_mass": arr_bkg["B_mass"],
        "cosThetaK": arr_bkg["cosThetaK"],
        "cosThetaL": arr_bkg["cosThetaL"],
        "q2": arr_bkg["q2"],
        "mKpi": arr_bkg["mKpi"],
    })

    df_bkg["cosl"] = df_bkg["cosThetaL"]
    df_bkg["cosh"] = df_bkg["cosThetaK"]
    if df_bkg["B_mass"].max() > 100.0:
        df_bkg["B_mass"] = df_bkg["B_mass"] / 1000.0
    df_bkg = df_bkg[(df_bkg["q2"] > 1.1) & (df_bkg["q2"] < 7.0)].copy()
    df_bkg = df_bkg[(df_bkg["mKpi"] < 1.5)].copy()
    df_bkg = df_bkg[(df_bkg["B_mass"] >= 5.170) & (df_bkg["B_mass"] <= 5.500)].copy()
    df_bkg.dropna(inplace=True)

    if args.with_eff:
        eff_bkg = efficiency.efficiency(
            df_bkg["cosh"].to_numpy(dtype=float),
            df_bkg["cosl"].to_numpy(dtype=float),
            df_bkg["mKpi"].to_numpy(dtype=float),
            df_bkg["q2"].to_numpy(dtype=float),
        )
        eff_max_bkg = eff_bkg.max()
        rng = np.random.default_rng(12345)
        u = rng.uniform(0.0, eff_max_bkg, len(df_bkg))
        mask_bkg = u < eff_bkg
        df_bkg = df_bkg.loc[mask_bkg].copy()
        df_bkg["eff_max"] = eff_max_bkg
        df_bkg["efficiency"] = eff_bkg[mask_bkg]
        df_bkg["fit_weight"] = eff_max_bkg / df_bkg["efficiency"].to_numpy(dtype=float)
    else:
        df_bkg["efficiency"] = 1.0
        df_bkg["eff_max"] = 1.0
        df_bkg["fit_weight"] = 1.0

    df_bkg["is_signal"] = 0
else:
    df_bkg = pd.DataFrame(columns=[
        "B_mass", "cosThetaK", "cosThetaL", "q2", "mKpi",
        "cosl", "cosh", "efficiency", "eff_max", "fit_weight", "is_signal",
    ])

# Build mixed dataset
datai = pd.concat([df_sig, df_bkg], ignore_index=True)
datai = datai.sample(frac=1.0, random_state=0).reset_index(drop=True)

print("Number of signal events:", len(df_sig))
print("Number of background events:", len(df_bkg))
print("Number of mixed data points:", len(datai))

# true values
# check if json or yaml
if args.settings.endswith(".yml"):
    with open(args.settings) as f:
        truth = yaml.load(f, Loader=yaml.FullLoader)
else:
    with open(args.settings) as f:
        truth = json.load(f)
        for t in truth:
            truth[t] = {"value": truth[t]}

for zi in args.fix_to_zero:
    truth[zi]["value"] = 0
for i in range(0, len(args.fix_to_value), 2):
    truth[args.fix_to_value[i]]["value"] = float(args.fix_to_value[i+1])

if args.toy:
    ntoys = 100
    nbins = 100
else:
    ntoys = 1
    nbins = 100

# Initialize parameters
App = zfit.Parameter("App", 0.1670, -1.0, 2.0)
A0 = zfit.Parameter("A0", 0.5, -1.0, 2.0)
Aqs = zfit.Parameter("Aqs", 0.01, -10.0, 10.0)
Aqc = zfit.Parameter("Aqc", 0.01, -10.0, 10.0)
AfbHS = zfit.Parameter("AfbHS", 0.0, -1.0, 1.0)
AfbHC = zfit.Parameter("AfbHC", 0.0, -1.0, 1.0)
AfbLS = zfit.Parameter("AfbLS", 0.0, -1.0, 1.0)
AfbLC = zfit.Parameter("AfbLC", 0.0, -1.0, 1.0)

# Set to the true values if provided
if "App" in truth.keys():
    App.set_value(truth["App"]["value"])
if "A0" in truth.keys():
    A0.set_value(truth["A0"]["value"])
if "Aqs" in truth.keys():
    Aqs.set_value(truth["Aqs"]["value"])
    # Aqs.floating = False
if "Aqc" in truth.keys():
    Aqc.set_value(truth["Aqc"]["value"])
if "AfbHS" in truth.keys():
    AfbHS.set_value(truth["AfbHS"]["value"])
if "AfbHC" in truth.keys():
    AfbHC.set_value(truth["AfbHC"]["value"])
if "AfbLS" in truth.keys():
    AfbLS.set_value(truth["AfbLS"]["value"])
if "AfbLC" in truth.keys():
    AfbLC.set_value(truth["AfbLC"]["value"])

def plot_projection_with_pull(
    bin_edges,
    bin_centers,
    data_y,
    data_yerr,
    pull,
    xlabel,
    ylabel,
    output_path,
    data_label="Data",
    reference_y=None,
    reference_label=None,
    line_x=None,
    total_y=None,
    total_label="Fit",
    stack_components=None,
    xlim=None,
    ylim_pull=(-5, 5),
):
    bin_width = bin_edges[1] - bin_edges[0]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )

    if reference_y is not None:
        ax1.step(
            bin_edges[:-1],
            reference_y,
            where="post",
            linewidth=2,
            color="blue",
            label=reference_label,
        )

    if stack_components is not None:
        bottom = np.zeros_like(stack_components[0]["y"], dtype=float)

        for comp in stack_components:
            top = bottom + comp["y"]

            ax1.fill_between(
                comp["x"],
                bottom,
                top,
                color=comp["color"],
                alpha=comp.get("alpha", 0.6),
                label=comp["label"],
                linewidth=0,
                edgecolor=comp.get("edgecolor", "w"),
                hatch=comp.get("hatch", None),
                zorder=0,
            )

            bottom = top

    ax1.errorbar(
        bin_centers,
        data_y,
        yerr=data_yerr,
        xerr=np.full_like(bin_centers, bin_width / 2.0),
        fmt="o",
        color="black",
        markersize=4,
        linewidth=1.2,
        label=data_label,
        zorder=10,
    )
    
    if line_x is not None and total_y is not None:
        ax1.plot(
            line_x,
            total_y,
            color="black",
            linewidth=2,
            label=total_label,
        )

    # Set y-axis to start at 0, or below 0 if there are negative bins
    ymin_candidates = []
    ymax_candidates = []

    if reference_y is not None:
        ymin_candidates.append(np.nanmin(reference_y))
        ymax_candidates.append(np.nanmax(reference_y))

    if data_y is not None:
        ymin_candidates.append(np.nanmin(data_y - data_yerr))
        ymax_candidates.append(np.nanmax(data_y + data_yerr))

    if total_y is not None:
        ymin_candidates.append(np.nanmin(total_y))
        ymax_candidates.append(np.nanmax(total_y))

    if stack_components is not None:
        for comp in stack_components:
            ymin_candidates.append(np.nanmin(comp["y"]))
            ymax_candidates.append(np.nanmax(comp["y"]))

    ymin = min(ymin_candidates) if len(ymin_candidates) > 0 else 0.0
    ymax = max(ymax_candidates) if len(ymax_candidates) > 0 else 1.0

    if ymin >= 0:
        ymin = 0.0
    else:
        ymin = 1.2 * ymin

    ymax = 1.2 * ymax if ymax > 0 else 1.0

    ax1.set_ylim(ymin, ymax)

    ax1.set_ylabel(ylabel)
    ax1.tick_params(axis="both")
    ax1.legend(loc="best", handlelength=1.5)

    if xlim is not None:
        ax1.set_xlim(*xlim)

    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.axhline(2.0, color="black", linestyle=":", linewidth=1.0)
    ax2.axhline(-2.0, color="black", linestyle=":", linewidth=1.0)

    ax2.bar(
        bin_centers,
        pull,
        width=bin_width,
        align="center",
        color="black",
        linewidth=0,
    )

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Pull")
    ax2.set_ylim(*ylim_pull)
    ax2.tick_params(axis="both")

    fig.subplots_adjust(
        hspace=0.0,
        left=0.14,
        right=0.97,
        top=0.97,
        bottom=0.10,
    )

    plt.savefig(output_path)
    plt.close()

def ASconditions(params):
    # The sum of all amplitudes must be 1.
    # This means that AS is not a free parameter.
    return 1-params['A0']-params['App']-params['Aqc']-params['Aqs']


AS = zfit.ComposedParameter("AS", ASconditions,
                            params={'A0': A0, 'App': App, 'Aqc': Aqc, 'Aqs': Aqs})

# total yield
Nsig = zfit.Parameter("Nsig", len(df_sig), 0.0, 1.0e8)
Nbkg = zfit.Parameter("Nbkg", len(df_bkg), 0.0, 1.0e8)

# component yields
def yieldAS(params):
    # S-wave yield
    return params['Nsig']*params['AS']


def yieldApp(params):
    # Perp/parallel yield
    return params['Nsig']*params['App']


def yieldA0(params):
    # 0 yield
    return params['Nsig']*params['A0']


def yieldAq(params):
    # beta-dependent yield
    return params['Nsig']*(params['Aqc']+params['Aqs'])


def yieldP(params):
    # P-wave yield
    return params['Nsig'] - params['N_AS']


# Define the yields as composed parameters based on the total yield
N_AS = zfit.ComposedParameter("N_AS", yieldAS,
                              params={'Nsig': Nsig, 'AS': AS})
N_App = zfit.ComposedParameter("N_App", yieldApp,
                               params={'Nsig': Nsig, 'App': App})
N_A0 = zfit.ComposedParameter("N_A0", yieldA0,
                              params={'Nsig': Nsig, 'A0': A0})
N_Aq = zfit.ComposedParameter("N_Aq", yieldAq,
                              params={'Nsig': Nsig, 'Aqc': Aqc, 'Aqs': Aqs})
N_P = zfit.ComposedParameter("N_P", yieldP,
                             params={'Nsig': Nsig, 'N_AS': N_AS})


# Create the pdf and register the analytic integral
fitpdf_ang = mypdfs.my2Dpdf(
    obs=angles,
    App=App,
    A0=A0,
    AS=AS,
    Aqc=Aqc,
    Aqs=Aqs,
    AfbHC=AfbHC,
    AfbHS=AfbHS,
    AfbLC=AfbLC,
    AfbLS=AfbLS,
)
fitpdf_ang.register_analytic_integral(func=mypdfs.integral, limits=limits)



mu_sig = zfit.Parameter("mu_sig", 5.28315, 5.26, 5.30)

sigma_sig_1 = zfit.Parameter("sigma_sig_1", 0.01412, 0.006, 0.025)
sigma_sig_2 = zfit.Parameter("sigma_sig_2", 0.02134, 0.012, 0.050)

frac_cb1 = zfit.Parameter("frac_cb1", 0.631, 0.0, 1.0)

alphal_1 = zfit.Parameter("alphal_1", 1.411266981340752)
nl_1     = zfit.Parameter("nl_1",     4.764925785834896)
alphar_1 = zfit.Parameter("alphar_1", 2.2832789313804036)
nr_1     = zfit.Parameter("nr_1",     2.5167753219617106)

alphal_2 = zfit.Parameter("alphal_2", 2.0961014043729644)
nl_2     = zfit.Parameter("nl_2",     0.21356971401774544)
alphar_2 = zfit.Parameter("alphar_2", 2.431383498001743)
nr_2     = zfit.Parameter("nr_2",     1.353695093592335)


for p in [alphal_1, nl_1, alphar_1, nr_1, alphal_2, nl_2, alphar_2, nr_2]:
    p.floating = False

fitpdf_mass_cb1 = zfit.pdf.DoubleCB(
    obs=mass,
    mu=mu_sig,
    sigma=sigma_sig_1,
    alphal=alphal_1,
    nl=nl_1,
    alphar=alphar_1,
    nr=nr_1,
)

fitpdf_mass_cb2 = zfit.pdf.DoubleCB(
    obs=mass,
    mu=mu_sig,
    sigma=sigma_sig_2,
    alphal=alphal_2,
    nl=nl_2,
    alphar=alphar_2,
    nr=nr_2,
)

fitpdf_mass = zfit.pdf.SumPDF(
    [fitpdf_mass_cb1, fitpdf_mass_cb2],
    fracs=frac_cb1,
)




lambda_bkg = zfit.Parameter("lambda_bkg", -0.2, -2.0, 0.0)
a1_cosh = zfit.Parameter("a1_cosh", 0.0, -0.5, 0.5)
a2_cosh = zfit.Parameter("a2_cosh", -0.2, -0.8, 0.8)

a1_cosl = zfit.Parameter("a1_cosl", 0.0, -0.5, 0.5)
a2_cosl = zfit.Parameter("a2_cosl", -0.4, -0.8, 0.8)

lambda_bkg.floating = False
a1_cosh.floating = False
a2_cosh.floating = False
a1_cosl.floating = False
a2_cosl.floating = False

fitpdf_bkg_mass = zfit.pdf.Exponential(obs=mass, lambda_=lambda_bkg)
fitpdf_bkg_cosh = zfit.pdf.Legendre(obs=cosh, coeffs=[a1_cosh, a2_cosh])
fitpdf_bkg_cosl = zfit.pdf.Legendre(obs=cosl, coeffs=[a1_cosl, a2_cosl])

fitpdf_bkg_ang = zfit.pdf.ProductPDF([fitpdf_bkg_cosh, fitpdf_bkg_cosl], obs=angles)

sigpdf = zfit.pdf.ProductPDF([fitpdf_ang, fitpdf_mass], obs=obs)
sigpdf = sigpdf.create_extended(Nsig)

bkgpdf = zfit.pdf.ProductPDF([fitpdf_bkg_ang, fitpdf_bkg_mass], obs=obs)
bkgpdf = bkgpdf.create_extended(Nbkg)

if args.with_bkg:
    fitpdf = zfit.pdf.SumPDF([sigpdf, bkgpdf])
else:
    Nbkg.floating = False
    Nbkg.set_value(0.0)
    fitpdf = sigpdf


# Apply constraints or fix parameters if requested
constraints = []

# Loop through all parameters
for p in fitpdf.get_params():
    if p.name in args.fix_to_zero:
        # Set parameter to zero
        p.floating = False
        p.set_value(0)
    if p.name in args.fix_to_value:
        # Set parameter to a specific value
        p.floating = False
        p.set_value(float(args.fix_to_value[args.fix_to_value.index(p.name)+1]))
    if p.name in args.fix_to_truth:
        # Fix parameter to its true value
        p.floating = False
        p.set_value(truth[p.name]["value"])
    if p.name in args.constrain:
        # Constrain parameter to its true value with a Gaussian constraint
        observed = truth[p.name]["value"]
        sigma = max(abs(truth[p.name]["error_lower"]), abs(truth[p.name]["error_upper"]))
        constraints.append(zfit.constraint.GaussianConstraint(p, observation=observed, sigma=sigma))


# create pdfs for sWeights (no asymmetry terms)
pdfS = mypdfs.my2Dpdf_AS(obs=angles)
pdfS = pdfS.create_extended(N_AS)
pdfApp = mypdfs.my2Dpdf_App(obs=angles)
pdfApp.register_analytic_integral(func=mypdfs.integral_App, limits=limits)
pdfApp = pdfApp.create_extended(N_App)
pdfA0 = mypdfs.my2Dpdf_A0(obs=angles)
pdfA0.register_analytic_integral(func=mypdfs.integral_A0, limits=limits)
pdfA0 = pdfA0.create_extended(N_A0)
pdfAq = mypdfs.my2Dpdf_Aq(obs=angles, Aqc=Aqc, Aqs=Aqs)
# Experimental extension:
# Use B_mass to separate signal/background and angles to separate angular components.
# This assumes B_mass is independent of the angular variables for each component.
# The original paper's method is angular-only; B_mass is not part of Eq. (1).
pdfAS_full = zfit.pdf.ProductPDF([fitpdf_mass, pdfS], obs=obs).create_extended(N_AS)
pdfApp_full = zfit.pdf.ProductPDF([fitpdf_mass, pdfApp], obs=obs).create_extended(N_App)
pdfA0_full = zfit.pdf.ProductPDF([fitpdf_mass, pdfA0], obs=obs).create_extended(N_A0)
pdfAq_full = zfit.pdf.ProductPDF([fitpdf_mass, pdfAq], obs=obs).create_extended(N_Aq)
pdfBkg_full = zfit.pdf.ProductPDF([fitpdf_bkg_mass, fitpdf_bkg_ang], obs=obs).create_extended(Nbkg)

pdfAq.register_analytic_integral(func=mypdfs.integral_Aq, limits=limits)
pdfAq = pdfAq.create_extended(N_Aq)
pdfBkg = fitpdf_bkg_ang.create_extended(Nbkg)
pdfsweightslist = []
if not (AS.name in args.fix_to_zero):
    pdfsweightslist.append(pdfS)
if not (App.name in args.fix_to_zero):
    pdfsweightslist.append(pdfApp)
if not (A0.name in args.fix_to_zero):
    pdfsweightslist.append(pdfA0)
if not (Aqs.name in args.fix_to_zero and Aqc.name in args.fix_to_zero):
    pdfsweightslist.append(pdfAq)
if args.with_bkg:
    pdfsweightslist.append(pdfBkg)
pdfsweights = zfit.pdf.SumPDF(pdfsweightslist)
pdfs = {m.get_yield(): m for m in pdfsweights.get_models()}
if args.with_bkg:
    pdfsweights = zfit.pdf.SumPDF([pdfAS_full, pdfApp_full, pdfA0_full, pdfAq_full, pdfBkg_full])
else:
    pdfsweights = zfit.pdf.SumPDF([pdfAS_full, pdfApp_full, pdfA0_full, pdfAq_full])

datadir = os.environ["DATADIR"]

reference_mapping = {
    "wA0": ("A0.root", "B02KstMuMu_Run1_centralQ2E_sig"),
    "wApp": ("A1.root", "B02KstMuMu_Run1_centralQ2E_sig"),
    "wS": ("AS.root", "B02KstMuMu_Run1_centralQ2E_sig"),
}
reference_display = {
    "wA0": r"$A_0$ reference",
    "wApp": r"$A_{\parallel,\perp}$ reference",
    "wS": r"$A_S$ reference",
}
# Select requested number of data points and ranges
if args.toy:
    if len(args.binned) == 2:
        # Toy in bins
        frac = len(datai.query(f"({args.binned[0]}<q2) & (q2<{args.binned[1]})"))/len(datai)
        print("Fraction of data in bin:", frac)
        truth["Nsig"]["value"] = int(args.nsig*frac)
    else:
        # Toy
        truth["Nsig"]["value"] = args.nsig

else:
    if len(args.binned) == 2:
        # Data in bins
        datai.query(f"({args.binned[0]}<q2) &(q2<{args.binned[1]})", inplace=True)
# data = zfit.Data.from_pandas(datai[["cosh", "cosl", "B_mass"]], obs=obs)

data = zfit.Data.from_pandas(datai[["cosh", "cosl", "B_mass"]],obs=obs,weights=datai["fit_weight"].to_numpy())


# Prepare for toys
pulls = {}
for p in fitpdf.get_params():
    if p.floating:
        pulls[p.name] = np.zeros(ntoys)
X = np.linspace(-1, 1, 100)


# Check that the pdf is well defined
assert (np.sum(fitpdf.pdf(data).numpy() <= 0) == 0)
assert (np.sum(np.isnan(fitpdf.pdf(data).numpy())) == 0)
assert (np.sum(np.isinf(fitpdf.pdf(data).numpy())) == 0)
assert (np.sum(np.isnan(np.log(fitpdf.pdf(data).numpy()))) == 0)
assert (np.sum(np.isinf(np.log(fitpdf.pdf(data).numpy()))) == 0)

    
def make_q_norm(df, bins=(10, 10, 15)):
    values = df[["cosh", "cosl", "B_mass"]].to_numpy(dtype=float)

    q_hist_weights = df["fit_weight"].to_numpy(dtype=float)
    q_hist_weights = q_hist_weights**2

    q_range = [
        (-1.0, 1.0),
        (-1.0, 1.0),
        (5.170, 5.500),
    ]

    hist_q, edges = np.histogramdd(
        values,
        bins=bins,
        range=q_range,
        weights=q_hist_weights,
        density=False,
    )

    positive = hist_q[hist_q > 0.0]
    n_empty = np.sum(hist_q <= 0.0)

    print("q(m) histogram empty bins:", n_empty)
    print("q(m) histogram total bins:", hist_q.size)

    if len(positive) == 0:
        raise RuntimeError("q(m) histogram is empty.")

    hist_q = hist_q / np.mean(hist_q)

    def q_norm(x):
        arr = np.asarray(x, dtype=float)

        if arr.ndim == 1:
            pts = arr.reshape(1, -1)
        elif arr.ndim == 2:
            if arr.shape[0] == 3:
                pts = arr.T
            elif arr.shape[1] == 3:
                pts = arr
            else:
                raise ValueError(f"Unexpected q_norm input shape: {arr.shape}")
        else:
            raise ValueError(f"Unexpected q_norm input ndim: {arr.ndim}")

        idx0 = np.searchsorted(edges[0], pts[:, 0], side="right") - 1
        idx1 = np.searchsorted(edges[1], pts[:, 1], side="right") - 1
        idx2 = np.searchsorted(edges[2], pts[:, 2], side="right") - 1

        idx0 = np.clip(idx0, 0, hist_q.shape[0] - 1)
        idx1 = np.clip(idx1, 0, hist_q.shape[1] - 1)
        idx2 = np.clip(idx2, 0, hist_q.shape[2] - 1)

        out = hist_q[idx0, idx1, idx2]
        out = np.asarray(out, dtype=float)

        return out

    return q_norm


def zfit_pdf_to_callable_3d_for_cows(zpdf, obs_space):
    def wrapped(x):
        arr = np.asarray(x, dtype=float)

        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError("Expected one 3D point.")
            pts = arr.reshape(1, 3)

        elif arr.ndim == 2:
            if arr.shape[1] == 3:
                # shape (N, 3)
                pts = arr
            elif arr.shape[0] == 3:
                # shape (3, N)
                pts = arr.T
            else:
                raise ValueError(f"Unexpected shape {arr.shape}")

        else:
            raise ValueError(f"Unexpected ndim {arr.ndim}")

        vals = zpdf.pdf(pts, norm=obs_space).numpy()
        return np.asarray(vals, dtype=float).reshape(-1)

    return wrapped

# def build_manual_cows_3d(pdfs, yields, ranges, n_int=500000, seed=0):
#     rng = np.random.default_rng(seed)

#     ranges = np.asarray(ranges, dtype=float)
#     lows = ranges[:, 0]
#     highs = ranges[:, 1]
#     volume = np.prod(highs - lows)

#     points = rng.uniform(lows, highs, size=(n_int, len(ranges)))
#     points_t = points.T

#     ysum = np.sum(yields)

#     def norm_g(x):
#         total = np.zeros(np.asarray(x).shape[-1], dtype=float)
#         for y, pdf in zip(yields, pdfs):
#             total += (y / ysum) * pdf(x)
#         return total

#     gvals = [pdf(points_t) for pdf in pdfs]
#     norm_vals = norm_g(points_t)

#     # norm_vals = np.asarray(norm_vals, dtype=float)
#     # norm_vals[norm_vals <= 0] = np.nan

#     ncomp = len(pdfs)
#     W = np.zeros((ncomp, ncomp), dtype=float)

#     for i in range(ncomp):
#         for j in range(ncomp):
#             integrand = gvals[i] * gvals[j] / norm_vals
#             W[i, j] = volume * np.nanmean(integrand)

#     A = np.linalg.inv(W)

#     return W, A, norm_g

# def compute_manual_cow_weights(data_cow, pdfs, A, norm_func):
#     norm_vals = norm_func(data_cow)
#     norm_vals = np.asarray(norm_vals, dtype=float)
#     norm_vals[norm_vals <= 0] = np.nan

#     g_data = [pdf(data_cow) for pdf in pdfs]

#     weights = []

#     for i in range(len(pdfs)):
#         wi = np.zeros_like(norm_vals, dtype=float)

#         for k in range(len(pdfs)):
#             wi += A[i, k] * g_data[k] / norm_vals

#         weights.append(wi)

#     return weights
    
def make_cow_reference_plot(
    datatoy,
    weights,
    ref_df,
    var,
    xlabel,
    output_path,
    reference_label="Reference",
    data_label="Weighted mixed sample",
    ref_weights=None,
):
    valid_data = datatoy[var].notna().to_numpy()
    valid_ref = ref_df[var].notna().to_numpy()

    values = datatoy.loc[valid_data, var].to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)[valid_data]

    ref_values = ref_df.loc[valid_ref, var].to_numpy(dtype=float)

    if ref_weights is None:
        ref_w = None
    else:
        ref_w = np.asarray(ref_weights, dtype=float)[valid_ref]

    if var == "mKpi":
        xmin, xmax = 0.65, 1.50
    else:
        xmin, xmax = 1.1, 7.0

    bin_edges = np.linspace(xmin, xmax, 51)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    h_ref, _ = np.histogram(
        ref_values,
        bins=bin_edges,
        weights=ref_w,
    )

    if ref_w is None:
        var_ref, _ = np.histogram(
            ref_values,
            bins=bin_edges,
        )
    else:
        var_ref, _ = np.histogram(
            ref_values,
            bins=bin_edges,
            weights=ref_w**2,
        )

    err_ref = np.sqrt(var_ref.astype(float))

    h_w, _ = np.histogram(
        values,
        bins=bin_edges,
        weights=w,
    )

    var_w, _ = np.histogram(
        values,
        bins=bin_edges,
        weights=w**2,
    )

    err_w = np.sqrt(var_w)

    norm_ref = np.sum(h_ref)
    norm_w = np.sum(h_w)

    if norm_ref <= 0 or np.isclose(norm_w, 0.0):
        return

    h_ref = h_ref / norm_ref
    err_ref = err_ref / abs(norm_ref)

    h_w = h_w / norm_w
    err_w = err_w / abs(norm_w)

    sigma_pull = np.sqrt(err_ref**2 + err_w**2)

    pull = np.zeros_like(bin_centers, dtype=float)
    mask = sigma_pull > 0
    pull[mask] = (h_w[mask] - h_ref[mask]) / sigma_pull[mask]

    plot_projection_with_pull(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        data_y=h_w,
        data_yerr=err_w,
        pull=pull,
        xlabel=xlabel,
        ylabel="Normalized entries",
        output_path=output_path,
        data_label=data_label,
        reference_y=h_ref,
        reference_label=reference_label,
        xlim=(xmin, xmax),
    )

def make_cow_extracted_only_plot(datatoy, weights, var, xlabel, output_path, data_label):
    valid_data = datatoy[var].notna().to_numpy()

    values = datatoy.loc[valid_data, var].to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)[valid_data]

    if var == "mKpi":
        xmin, xmax = 0.65, 1.50
    else:
        xmin, xmax = 1.1, 7.0

    bin_edges = np.linspace(xmin, xmax, 51)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    h_w, _ = np.histogram(values, bins=bin_edges, weights=w)
    var_w, _ = np.histogram(values, bins=bin_edges, weights=w**2)
    err_w = np.sqrt(var_w)

    norm = np.sum(h_w)

    if np.isclose(norm, 0.0):
        return

    h_w = h_w / norm
    err_w = err_w / abs(norm)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        bin_centers,
        h_w,
        yerr=err_w,
        xerr=np.full_like(bin_centers, 0.5 * (bin_edges[1] - bin_edges[0])),
        fmt="o",
        color="black",
        markersize=4,
        linewidth=1.2,
        label=data_label,
    )

    ymin = np.nanmin(h_w - err_w)
    ymax = np.nanmax(h_w + err_w)

    if ymin >= 0:
        ymin = 0.0
    else:
        ymin = 1.2 * ymin

    ymax = 1.2 * ymax if ymax > 0 else 1.0

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized entries")
    ax.legend()
    ax.tick_params(axis="both")

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        top=0.97,
        bottom=0.12,
    )

    plt.savefig(output_path)
    plt.close()


def plot_correlation_matrix(covmat, param_names, output_path):
    covmat = np.asarray(covmat, dtype=float)

    diag = np.diag(covmat)
    sigma = np.sqrt(np.clip(diag, 0.0, None))

    denom = np.outer(sigma, sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        corrmat = np.divide(
            covmat,
            denom,
            out=np.zeros_like(covmat, dtype=float),
            where=denom > 0,
        )

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corrmat, vmin=-1.0, vmax=1.0, cmap="coolwarm")

    ax.set_xticks(np.arange(len(param_names)))
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_yticklabels(param_names)

    for i in range(len(param_names)):
        for j in range(len(param_names)):
            ax.text(
                j,
                i,
                f"{corrmat[i, j]:.2f}",
                ha="center",
                va="center",
                # fontsize=8,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    ax.set_title("Correlation matrix")
    fig.tight_layout()

    plt.savefig(output_path)
    plt.close()

# i = 0
# while i < ntoys:
for i in range(ntoys):
    print("Toy", i)
    seed = np.random.randint(0, 2**32-1)
    zfit.settings.set_seed(seed)
    np.random.seed(seed)

    # create minimizer
    if args.toy:
        minimizer = zfit.minimize.Minuit(strategy=zfit.minimize.DefaultToyStrategy)
    else:  # easier for debugging data
        minimizer = zfit.minimize.Minuit()

    if args.toy:
        nsig_toy = np.random.poisson(len(df_sig))
        nbkg_toy = np.random.poisson(len(df_bkg))

        toy_sig = df_sig.sample(n=nsig_toy, replace=True)
        toy_bkg = df_bkg.sample(n=nbkg_toy, replace=True)

        datatoy = pd.concat([toy_sig, toy_bkg], ignore_index=True)
        datatoy = datatoy.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        if len(args.binned) == 2:
            datatoy.query(f"({args.binned[0]}<q2) &(q2<{args.binned[1]})", inplace=True)

        # data = zfit.Data.from_pandas(datatoy[["cosh", "cosl", "B_mass"]], obs=obs)

        data = zfit.Data.from_pandas(datatoy[["cosh", "cosl", "B_mass"]], obs=obs, weights=datatoy["fit_weight"].to_numpy(),)

        Nsig.set_value(datatoy.query("is_signal == 1")["fit_weight"].sum())
        Nbkg.set_value(datatoy.query("is_signal == 0")["fit_weight"].sum())
    else:
        datatoy = datai.copy()

        if len(args.binned) == 2:
            datatoy.query(f"({args.binned[0]}<q2) &(q2<{args.binned[1]})", inplace=True)

        # data = zfit.Data.from_pandas(datatoy[["cosh", "cosl", "B_mass"]], obs=obs,)

        data = zfit.Data.from_pandas(datatoy[["cosh", "cosl", "B_mass"]], obs=obs, weights=datatoy["fit_weight"].to_numpy(),)

        Nsig.set_value(datatoy.query("is_signal == 1")["fit_weight"].sum())
        Nbkg.set_value(datatoy.query("is_signal == 0")["fit_weight"].sum())

    # Create the loss
    loss = zfit.loss.ExtendedUnbinnedNLL(model=fitpdf, data=data)

    # Add constraints if any
    if len(constraints) > 0:
        loss.add_constraints(constraints)

    # Run the fit
    result = minimizer.minimize(loss)
    result.update_params()
    mass_only = mass

    pdfAS_mass = fitpdf_mass.create_extended(N_AS)
    pdfA0_mass = fitpdf_mass.create_extended(N_A0)
    pdfApp_mass = fitpdf_mass.create_extended(N_App)
    pdfBkg_mass = fitpdf_bkg_mass.create_extended(Nbkg)
    try:
        print(result)
        # result.errors()  # Compute uncertainty
    except Exception as e:
        print(e)
        print("Problem with errors.")
        for p in fitpdf.get_params():
            if p.floating:
                print(p)
                if p.name in truth:
                    p.set_value(truth[p.name]["value"])
                else:
                    print(f"No truth value for {p.name}, keep current value.")
                print(p)
        continue

    print(result)
    result.update_params()

    # # Compute parameter uncertainties from Hesse
    # try:
    #     result.hesse()
    # except Exception as e:
    #     print("Warning: Hesse failed.")
    #     print(e)

    # # Check that the result is valid
    covmat = result.covariance()

    if not np.all(np.isfinite(covmat)):
        print("Warning: covariance matrix contains NaN or inf.")
        posdef = False
    else:
        posdef = np.all(np.linalg.eigvals(covmat) > -1e-8)

    if not result.valid:
        print("Fit not valid.")
        for p in fitpdf.get_params():
            if p.floating:
                if p.name in truth:
                    p.set_value(truth[p.name]["value"])
                else:
                    print(f"No truth value for {p.name}, keep current value.")
        continue

    if not posdef:
        print("Warning: covariance matrix is not positive definite. Continue anyway.")

    cow_I = args.cow_I
    case_tag = f"{'bkg' if args.with_bkg else 'nobkg'}_{'eff' if args.with_eff else 'noeff'}_I{cow_I}"
    outdir_results = f"results_cow_{case_tag}"
    os.makedirs(outdir_results, exist_ok=True)

    corr_param_names = [p.name for p in result.params]
    corr_outname = f"{outdir_results}/{i}_correlation_matrix.pdf"

    try:
        # plot_correlation_matrix(covmat, corr_param_names, corr_outname)
        print("Saved correlation matrix to:")
        print(corr_outname)
    except Exception as e:
        print("Warning: failed to save correlation matrix.")
        print(e)

    # Save the fit results
    paramdict = {}
    pi = 0
    for p in result.params:
        pinfo = result.params[p]

        value = pinfo["value"]
        error = None
        error_upper = None
        error_lower = None

        if "hesse" in pinfo and pinfo["hesse"] is not None:
            error = pinfo["hesse"].get("error", None)

            if error is not None and np.isfinite(error):
                error = float(error)
                error_upper = float(error)
                error_lower = -float(error)

        elif "errors" in pinfo and pinfo["errors"] is not None:
            error_upper = pinfo["errors"].get("upper", None)
            error_lower = pinfo["errors"].get("lower", None)

            if error_upper is not None and np.isfinite(error_upper):
                error_upper = float(error_upper)
            else:
                error_upper = None

            if error_lower is not None and np.isfinite(error_lower):
                error_lower = float(error_lower)
            else:
                error_lower = None

        paramdict[p.name] = {}
        paramdict[p.name]["value"] = float(value)
        paramdict[p.name]["error"] = error
        paramdict[p.name]["error_upper"] = error_upper
        paramdict[p.name]["error_lower"] = error_lower
        paramdict[p.name]["floating"] = bool(p.floating)
        paramdict[p.name]["covariance"] = {}

        qi = 0
        for q in result.params:
            if q == p:
                qi += 1
                continue

            val = covmat[pi][qi]
            paramdict[p.name]["covariance"][q.name] = None if not np.isfinite(val) else float(val)
            qi += 1

        pi += 1

    outname = f"{outdir_results}/{i}.yml"
    with open(outname, 'w') as yaml_file:
        yaml.dump(paramdict, yaml_file, default_flow_style=False)

    table_rows = []

    for pname, pinfo in paramdict.items():
        value = pinfo["value"]
        error = pinfo["error"]

        if error is not None:
            value_pm_error = f"{value:.6g} +/- {error:.3g}"
        else:
            value_pm_error = f"{value:.6g} +/- None"

        table_rows.append({
            "name": pname,
            "value": value,
            "error": error,
            "value_pm_error": value_pm_error,
            "floating": pinfo["floating"],
        })

    table_df = pd.DataFrame(table_rows)

    csv_outname = f"{outdir_results}/{i}_parameters_with_uncertainties.csv"
    table_df.to_csv(csv_outname, index=False)

    txt_outname = f"{outdir_results}/{i}_parameters_with_uncertainties.txt"
    with open(txt_outname, "w") as f:
        f.write(table_df.to_string(index=False))
    # # Compute sWeights

    # try:
    #     sweights = compute_sweights(pdfsweights, data)
    # except Exception as e:
    #     print(e)
    #     print("Problem with massfit sweights.")
    #     continue       

    # # Sanity check
    # diff = Nsig.value()-N_A0.value()-N_App.value()-N_Aq.value()-N_AS.value()
    # assert (np.isclose(diff, 0, atol=1e-2))

    # sApp, sA0, sAS, sAq, sBkg = sweights[N_App], sweights[N_A0], sweights[N_AS], sweights[N_Aq], sweights[Nbkg]

    # --------------------------------------------
    # Multi-component COWs with I = g or I = q
    # A0 / App / AS / Aq / Bkg
    # --------------------------------------------
    try:
        data_cow = datatoy[["cosh", "cosl", "B_mass"]].to_numpy(dtype=float).T

        fA0_3d = zfit_pdf_to_callable_3d_for_cows(pdfA0_full, obs)
        fApp_3d = zfit_pdf_to_callable_3d_for_cows(pdfApp_full, obs)
        fAS_3d = zfit_pdf_to_callable_3d_for_cows(pdfAS_full, obs)
        fAq_3d = zfit_pdf_to_callable_3d_for_cows(pdfAq_full, obs)
        fBkg_3d = zfit_pdf_to_callable_3d_for_cows(pdfBkg_full, obs)

        pdfs_sig_cow = [
            fA0_3d,
            fApp_3d,
            fAS_3d,
            fAq_3d,
        ]

        if args.with_bkg:
            pdfs_bkg_cow = [fBkg_3d]
        else:
            pdfs_bkg_cow = []

        pdfs_cow = pdfs_sig_cow + pdfs_bkg_cow

        yields_cow = [
            float(N_A0.value()),
            float(N_App.value()),
            float(N_AS.value()),
            float(N_Aq.value()),
        ]
        if args.with_bkg:
            yields_cow.append(float(Nbkg.value()))

        ranges_cow = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (5.170, 5.500),
        ]

        cow_I = args.cow_I

        if cow_I not in ["g", "q"]:
            raise ValueError(f"Unknown COW I choice: {cow_I}. Use 'g' or 'q'.")

        if cow_I == "q":
            cow_norm = make_q_norm(datatoy)
            print("Using COWs with I = q from external q_norm.")
        else:
            cow_norm = None
            print("Using COWs with I = g from mixture norm.")

        cow = Cows(
            sample=None,
            spdf=pdfs_sig_cow,
            bpdf=pdfs_bkg_cow,
            norm=cow_norm,
            range=ranges_cow,
            summation=False,
            yields=yields_cow,
            integration_options={
                "n_estimates": 8,
                "n_points": 65536,
            },
        )

        W_cow = cow._wm + np.tril(cow._wm, -1).T
        A_cow = cow._am

        print("COW W matrix from sweights package:")
        print(W_cow)

        print("COW A matrix from sweights package:")
        print(A_cow)

        print("COW W condition number:")
        print(np.linalg.cond(W_cow))

        wA0_cow = cow[0](data_cow)
        wApp_cow = cow[1](data_cow)
        wAS_cow = cow[2](data_cow)
        wAq_cow = cow[3](data_cow)
        if args.with_bkg:
            wBkg_cow = cow["b"](data_cow)
        else:
            wBkg_cow = np.zeros_like(wA0_cow, dtype=float)

        eff_weight = datatoy["fit_weight"].to_numpy(dtype=float)

        wA0_final = wA0_cow * eff_weight
        wApp_final = wApp_cow * eff_weight
        wAS_final = wAS_cow * eff_weight
        wAq_final = wAq_cow * eff_weight
        wBkg_final = wBkg_cow * eff_weight

        w_sum_raw = wA0_cow + wApp_cow + wAS_cow + wAq_cow + wBkg_cow
        w_sum_final = wA0_final + wApp_final + wAS_final + wAq_final + wBkg_final

        print(f"\n[Debug realistic efficiency weighted multi-component COWs, I={cow_I}]")
        print("raw sum wA0  =", np.sum(wA0_cow))
        print("raw sum wApp =", np.sum(wApp_cow))
        print("raw sum wAS  =", np.sum(wAS_cow))
        print("raw sum wAq  =", np.sum(wAq_cow))
        print("raw event-wise sum mean:", np.mean(w_sum_raw))
        print("raw total sum:", np.sum(w_sum_raw))
        print("Unweighted N events:", len(datatoy))

        print("final sum wA0  =", np.sum(wA0_final), "expected N_A0  =", float(N_A0.value()))
        print("final sum wApp =", np.sum(wApp_final), "expected N_App =", float(N_App.value()))
        print("final sum wAS  =", np.sum(wAS_final), "expected N_AS  =", float(N_AS.value()))
        print("final sum wAq  =", np.sum(wAq_final), "expected N_Aq  =", float(N_Aq.value()))
        print("final sum wBkg =", np.sum(wBkg_final), "expected Nbkg =", float(Nbkg.value()))

        print("final event-wise sum mean:", np.mean(w_sum_final))
        print("final total sum:", np.sum(w_sum_final))
        print("N signal:", float(Nsig.value()))
        print("Weighted sum:", datatoy["fit_weight"].sum())

    except Exception:
        import traceback
        traceback.print_exc()
        print("Problem with realistic efficiency weighted signal-only COWs I=g.")
        wA0_cow = None
        wApp_cow = None
        wAS_cow = None
        wAq_cow = None
        wBkg_cow = None

        
    if i < 3 and wA0_cow is not None and wApp_cow is not None and wAS_cow is not None and wBkg_cow is not None:
        outdir_cow = f"plots/{args.polynomial}/{name}/reference_cow_multicomp_{case_tag}"
        os.makedirs(outdir_cow, exist_ok=True)
        
        is_sig = datatoy["is_signal"].to_numpy(dtype=bool)

        print("\n[Debug wBkg COW]")
        print("sum wBkg all         =", np.sum(wBkg_cow))
        print("expected Nbkg        =", float(Nbkg.value()))
        print("sum wBkg true signal =", np.sum(wBkg_cow[is_sig]))
        print("sum wBkg true bkg    =", np.sum(wBkg_cow[~is_sig]))
        print("mean wBkg true signal =", np.mean(wBkg_cow[is_sig]))
        print("mean wBkg true bkg    =", np.mean(wBkg_cow[~is_sig]))

        print("\n[Debug wBkg on reference signal samples]")

        for ref_name, (ref_file, ref_tree) in reference_mapping.items():
            ref_path = os.path.join(datadir, ref_file)

            with uproot.open(ref_path) as fref:
                ref_check = fref[ref_tree].arrays(library="pd")

            ref_check["cosl"] = ref_check["cosThetaL"]
            ref_check["cosh"] = ref_check["cosThetaK"]

            if "B_mass" in ref_check.columns:
                ref_check["B_mass"] = ref_check["B_mass"] / 1000.0
                ref_check = ref_check[
                    (ref_check["B_mass"] >= 5.170)
                    & (ref_check["B_mass"] <= 5.500)
                ].copy()

            ref_check = ref_check[
                (ref_check["q2"] > 1.1)
                & (ref_check["q2"] < 7.0)
                & (ref_check["mKpi"] < 1.5)
            ].copy()

            ref_check.dropna(inplace=True)

            ref_data_cow = ref_check[["cosh", "cosl", "B_mass"]].to_numpy(dtype=float).T

            ref_data_cow = ref_check[["cosh", "cosl", "B_mass"]].to_numpy(dtype=float).T

            wBkg_ref = cow["b"](ref_data_cow)

            print(
                ref_name,
                "sum wBkg =",
                np.sum(wBkg_ref),
                "mean wBkg =",
                np.mean(wBkg_ref),
                "N =",
                len(ref_check),
            )


        effcorr = datatoy["fit_weight"].to_numpy(dtype=float)

        weight_dict_cow = {
            "wA0": wA0_final,
            "wApp": wApp_final,
            "wS": wAS_final,
        }

        for weight_name, (ref_file, ref_tree) in reference_mapping.items():
            ref_path = os.path.join(datadir, ref_file)

            with uproot.open(ref_path) as fref:
                ref_df = fref[ref_tree].arrays(library="pd")

            if "B_mass" in ref_df.columns:
                ref_df["B_mass"] = ref_df["B_mass"] / 1000.0
                ref_df = ref_df[(ref_df["B_mass"] >= 5.170) & (ref_df["B_mass"] <= 5.500)].copy()
            if "mKpi" in ref_df.columns:
                ref_df = ref_df[ref_df["mKpi"] < 1.5].copy()
            if "q2" in ref_df.columns:
                ref_df = ref_df[(ref_df["q2"] > 1.1) & (ref_df["q2"] < 7.0)].copy()

            ref_df.dropna(inplace=True)
            # ref_eff = efficiency.efficiency(
            # ref_df["cosThetaK"].to_numpy(dtype=float),
            # ref_df["cosThetaL"].to_numpy(dtype=float),
            # ref_df["mKpi"].to_numpy(dtype=float),
            # ref_df["q2"].to_numpy(dtype=float),
            # )

            # rng_ref = np.random.default_rng(12345)
            # u_ref = rng_ref.uniform(0.0, ref_eff.max(), len(ref_df))
            # ref_mask = u_ref < ref_eff

            # ref_df = ref_df[ref_mask].copy()
            # ref_df["efficiency"] = ref_eff[ref_mask]
            
            make_cow_reference_plot(
                datatoy,
                weight_dict_cow[weight_name],
                ref_df,
                "mKpi",
                r"$m(K\pi)$ [GeV/$c^2$]",
                f"{outdir_cow}/{i}_{weight_name}_mKpi.pdf",
                reference_label=reference_display[weight_name],
                data_label=f"Weighted mixed sample {weight_name}",
            )

            make_cow_reference_plot(
                datatoy,
                weight_dict_cow[weight_name],
                ref_df,
                "q2",
                r"$q^2$ [GeV$^2/c^4$]",
                f"{outdir_cow}/{i}_{weight_name}_q2.pdf",
                reference_label=reference_display[weight_name],
                data_label=f"Weighted mixed sample {weight_name}",
            )

        make_cow_reference_plot(
            datatoy,
            wBkg_final,
            df_bkg,
            "mKpi",
            r"$m(K\pi)$ [GeV/$c^2$]",
            f"{outdir_cow}/{i}_wBkg_mKpi.pdf",
            reference_label="Background reference",
            data_label="Weighted mixed sample wBkg",
            ref_weights=df_bkg["fit_weight"].to_numpy(dtype=float),
        )

        make_cow_reference_plot(
            datatoy,
            wBkg_final,
            df_bkg,
            "q2",
            r"$q^2$ [GeV$^2/c^4$]",
            f"{outdir_cow}/{i}_wBkg_q2.pdf",
            reference_label="Background reference",
            data_label="Weighted mixed sample wBkg",
            ref_weights=df_bkg["fit_weight"].to_numpy(dtype=float),
        )

        make_cow_extracted_only_plot(
            datatoy,
            wAq_final,
            "mKpi",
            r"$m(K\pi)$ [GeV/$c^2$]",
            f"{outdir_cow}/{i}_wAq_mKpi.pdf",
            data_label=r"$A_q$ weighted mixed sample",
        )

        make_cow_extracted_only_plot(
            datatoy,
            wAq_final,
            "q2",
            r"$q^2$ [GeV$^2/c^4$]",
            f"{outdir_cow}/{i}_wAq_q2.pdf",
            data_label=r"$A_q$ weighted mixed sample",
        )

    # Plot the result
    if i < 3:
        outdir_fit = f"plots/{args.polynomial}/{name}/fit_projections_cow_{case_tag}"
        os.makedirs(outdir_fit, exist_ok=True)
        # Make the same type of plot for costhetah and costhetal
        for v, n, l in zip([cosh, cosl], ["cosh", "cosl"], [r"$\cos(\theta_h)$", r"$\cos(\theta_\ell)$"]):
            xmin = v.limits[0][0][0]
            xmax = v.limits[1][0][0]
            dist = xmax - xmin
            x = np.linspace(xmin, xmax, 1000)

            bin_edges = np.linspace(xmin, xmax, nbins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_width = bin_edges[1] - bin_edges[0]

            w = datatoy["fit_weight"].to_numpy()

            H = hist.Hist(
                hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False),
                storage=hist.storage.Weight(),
            )

            H.fill(values, weight=weights)

            h = H.values()
            err = np.sqrt(H.variances())
            
            y_fit = (
                np.asarray(
                    fitpdf.create_projection_pdf(obs=v).ext_pdf(bin_centers).numpy(),
                    dtype=float,
                ).reshape(-1)
                * bin_width
            )

            pull = np.zeros_like(bin_centers, dtype=float)
            mask_pull = yerr > 0
            pull[mask_pull] = (counts[mask_pull] - y_fit[mask_pull]) / yerr[mask_pull]

            ybkg = (
                np.asarray(
                    fitpdf_bkg_ang.create_projection_pdf(obs=v).pdf(x).numpy(),
                    dtype=float,
                ).reshape(-1)
                * Nbkg.value()
                * dist / nbins
            )

            yAS = (
                np.asarray(
                    pdfs[N_AS].create_projection_pdf(obs=v).ext_pdf(x).numpy(),
                    dtype=float,
                ).reshape(-1)
                * dist / nbins
            )

            yA0 = (
                np.asarray(
                    pdfs[N_A0].create_projection_pdf(obs=v).ext_pdf(x).numpy(),
                    dtype=float,
                ).reshape(-1)
                * dist / nbins
            )

            yApp = (
                np.asarray(
                    pdfs[N_App].create_projection_pdf(obs=v).ext_pdf(x).numpy(),
                    dtype=float,
                ).reshape(-1)
                * dist / nbins
            )

            yAq = (
                np.asarray(
                    pdfs[N_Aq].create_projection_pdf(obs=v).ext_pdf(x).numpy(),
                    dtype=float,
                ).reshape(-1)
                * dist / nbins
            )

            Z = (
                np.asarray(
                    fitpdf.create_projection_pdf(obs=v).ext_pdf(x).numpy(),
                    dtype=float,
                ).reshape(-1)
                * dist / nbins
            )

            if n == "cosh":
                yInt = (
                    af.proj_AfbHC(x, n) * AfbHC.value()
                    + af.proj_AfbHS(x, n) * AfbHS.value()
                ) * Nsig.value() * dist / nbins
            else:
                yInt = (
                    af.proj_AfbLC(x, n) * AfbLC.value()
                    + af.proj_AfbLS(x, n) * AfbLS.value()
                ) * Nsig.value() * dist / nbins

            stack_components = [
                {
                    "x": x,
                    "y": ybkg,
                    "color": "lightgray",
                    "alpha": 0.8,
                    "label": "Background",
                },
                {
                    "x": x,
                    "y": yAS,
                    "color": "gold",
                    "alpha": 0.6,
                    "label": r"$n^S_0=\beta^2(|{A'}_0^L|^2+|{A'}_0^R|^2)$",
                    "hatch": "xx",
                },
                {
                    "x": x,
                    "y": yA0,
                    "color": "navy",
                    "alpha": 0.6,
                    "label": r"$n^P_0=\beta^2(|{A}_0^L|^2+|{A}_0^R|^2)$",
                    "hatch": "//",
                },
                {
                    "x": x,
                    "y": yApp,
                    "color": "dodgerblue",
                    "alpha": 0.6,
                    "label": r"$n^P_1=\beta^2(|{A}_\perp^L|^2+|{A}_\perp^R|^2+|{A}_\parallel^L|^2+|{A}_\parallel^R|^2)$",
                    "hatch": "\\\\",
                },
                {
                    "x": x,
                    "y": yAq,
                    "color": "firebrick",
                    "alpha": 0.6,
                    "label": r"$n_{\beta}$",
                    "hatch": "..",
                },
                {
                    "x": x,
                    "y": yInt,
                    "color": "darkgreen",
                    "alpha": 0.4,
                    "label": "Interference",
                },
            ]

            plot_projection_with_pull(
                bin_edges=bin_edges,
                bin_centers=bin_centers,
                data_y=counts,
                data_yerr=yerr,
                pull=pull,
                xlabel=l,
                ylabel=fr"Data points / {(dist / nbins):.2f}",
                output_path=f"{outdir_fit}/{i}_{n}_with_pull_cow_efficiency_eff.pdf",
                data_label="Data",
                line_x=x,
                total_y=Z,
                total_label="Fit",
                stack_components=stack_components,
                xlim=(xmin, xmax),
            )

        # # Also make weighted plots
        # for vkey, l, u in zip(["mKpi", "q2"], [r"$m(K\pi)$", r"$q^2$"], [r"GeV$/c^2$", r"GeV$^2/c^4$"]):
        #     mi, ma = datatoy[vkey].min(), datatoy[vkey].max()
        #     dist = ma - mi
        #     H = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False))
        #     H.fill(datatoy[vkey])
        #     mplhep.histplot(H, color='black', histtype='errorbar', label='Toy data', xerr=True, yerr=True, marker='.', zorder=20)
        #     nominal = H.values()
        #     y = np.zeros(nbins)  # For stacking the histograms
        #     lists = zip([r"$n^S_0=\beta^2(|{A'}_0^L|^2+|{A'}_0^R|^2)$", r'$n_0^P=\beta^2(|{A}_0^L|^2+|{A}_0^R|^2)$', r'$n_1^P=\beta^2(|{A}_\perp^L|^2+|{A}_\perp^R|^2+|{A}_\parallel^L|^2+|{A}_\parallel^R|^2)$', r'$n_{\beta}$'],
        #                 [sAS, sA0, sApp, sAq],
        #                 ['gold', 'navy', 'dodgerblue', 'firebrick'])
        #     for n, w, c in lists:
        #         if all(w == 0):
        #             continue
        #         # Make a weighted histogram and plot it stacked
        #         H = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
        #         H.fill(datatoy[vkey], weight=w)
        #         hvals = H.values()
        #         for k in range(nbins):
        #             plt.fill_between(H.axes[0].edges[k:k+2], y1=y[k], y2=y[k] + hvals[k],
        #                              color=c, linewidth=0, zorder=0)
        #         # Add errorbars
        #         plt.errorbar(H.axes[0].centers, y+hvals,
        #                      yerr=np.sqrt(H.variances()), xerr=H.axes[0].widths/2,
        #                      fmt='.', elinewidth=1, color=c, label=n)
        #         y += hvals  # Raise the bottom for stacking
        #     plt.legend(handletextpad=0.1, fontsize=24)
        #     plt.axhline(0, color='black', linewidth=1)
        #     plt.xlim(mi, ma)
        #     ylims = plt.ylim()
        #     plt.xlabel(l+f" [{u}]", ha="right", x=1)
        #     plt.ylabel(fr"$\sum$ weights / ({dist/nbins:.2f} {u})", ha="right", y=1)
        #     plt.savefig(f"plots/{args.polynomial}/{name}/{i}_{vkey}_weighted.pdf")
        #     plt.close()

        # -------------------------------------------------
        # Plot B_mass projection with pull
        # -------------------------------------------------
        nbins_mass = 100
        xmin_mass, xmax_mass = 5.17, 5.50
        x_mass = np.linspace(xmin_mass, xmax_mass, 1000)
        bin_edges_mass = np.linspace(xmin_mass, xmax_mass, nbins_mass + 1)
        bin_centers_mass = 0.5 * (bin_edges_mass[:-1] + bin_edges_mass[1:])
        bin_width_mass = bin_edges_mass[1] - bin_edges_mass[0]

        w = datatoy["fit_weight"].to_numpy()
        counts_mass, _ = np.histogram(datatoy["B_mass"],bins=bin_edges_mass,weights=w,)
        sumw2_mass, _ = np.histogram(datatoy["B_mass"],bins=bin_edges_mass,weights=w**2,)
        yerr_mass = np.sqrt(sumw2_mass)

        y_mass = (
            np.asarray(
                fitpdf.create_projection_pdf(obs=mass).ext_pdf(bin_centers_mass).numpy(),
                dtype=float,
            ).reshape(-1)
            * bin_width_mass
        )

        pull_mass = np.zeros_like(bin_centers_mass, dtype=float)
        mask_mass = yerr_mass > 0
        pull_mass[mask_mass] = (counts_mass[mask_mass] - y_mass[mask_mass]) / yerr_mass[mask_mass]

        mass_shape = np.asarray(
            fitpdf_mass.pdf(x_mass).numpy(),
            dtype=float,
        ).reshape(-1)

        bkg_mass_shape = np.asarray(
            fitpdf_bkg_mass.pdf(x_mass).numpy(),
            dtype=float,
        ).reshape(-1)

        ybkg_mass = bkg_mass_shape * Nbkg.value() * bin_width_mass
        yAS_mass = mass_shape * N_AS.value() * bin_width_mass
        yA0_mass = mass_shape * N_A0.value() * bin_width_mass
        yApp_mass = mass_shape * N_App.value() * bin_width_mass
        yAq_mass = mass_shape * N_Aq.value() * bin_width_mass        

        stack_components_mass = [
            {
                "x": x_mass,
                "y": ybkg_mass,
                "color": "lightgray",
                "alpha": 0.8,
                "label": "Background",
            },
            {
                "x": x_mass,
                "y": yAS_mass,
                "color": "gold",
                "alpha": 0.6,
                "label": r"$n^S_0=\beta^2(|{A'}_0^L|^2+|{A'}_0^R|^2)$",
                "hatch": "xx",
            },
            {
                "x": x_mass,
                "y": yA0_mass,
                "color": "navy",
                "alpha": 0.6,
                "label": r"$n^P_0=\beta^2(|{A}_0^L|^2+|{A}_0^R|^2)$",
                "hatch": "//",
            },
            {
                "x": x_mass,
                "y": yApp_mass,
                "color": "dodgerblue",
                "alpha": 0.6,
                "label": r"$n^P_1=\beta^2(|{A}_\perp^L|^2+|{A}_\perp^R|^2+|{A}_\parallel^L|^2+|{A}_\parallel^R|^2)$",
                "hatch": "\\\\",
            },
            {
                "x": x_mass,
                "y": yAq_mass,
                "color": "firebrick",
                "alpha": 0.6,
                "label": r"$n_{\beta}$",
                "hatch": "..",
            },
        ]

        plot_projection_with_pull(
            bin_edges=bin_edges_mass,
            bin_centers=bin_centers_mass,
            data_y=counts_mass,
            data_yerr=yerr_mass,
            pull=pull_mass,
            xlabel=r"$B$ mass [GeV/$c^2$]",
            ylabel=fr"Data points / {bin_width_mass:.4f}",
            output_path=f"{outdir_fit}/{i}_{n}B_mass_with_pull_cow_efficiency_noeff.pdf",
            data_label="Data",
            line_x=x_mass,
            total_y=ybkg_mass + yAS_mass + yA0_mass + yApp_mass + yAq_mass,
            total_label="Fit",
            stack_components=stack_components_mass,
            xlim=(xmin_mass, xmax_mass),
        )
        
    # Save the sWeighted data
    # datas = data.to_pandas()
    # datas['wS'] = sAS
    # datas['wApp'] = sApp
    # datas['wA0'] = sA0
    # datas['wAq'] = sAq
    # datas["wBkg"] = sBkg
    # datas['mKpi'] = datatoy['mKpi'].values
    # datas['q2'] = datatoy['q2'].values
    # datas['cosl'] = datatoy['cosl'].values
    # datas['cosh'] = datatoy['cosh'].values
    # datas.to_hdf(f"sweights/{args.polynomial}/{name}/{i}.h5", key='data', mode='w')
    i += 1


# Plot the pull distributions if this was a toy study
if args.toy:
    mu = zfit.Parameter("mu", 0, -500, 500)
    sig = zfit.Parameter("sig", 1, 0, 100)
    x = zfit.Space('x', (-500, 500))
    gauss = zfit.pdf.Gauss(obs=x, mu=mu, sigma=sig)
    X = np.linspace(-5, 5, num=100)

    minimizer = zfit.minimize.Minuit()

    for k in pulls.keys():
        print("Pulls", k)
        pullsk = pulls[k]
        try:
            res = minimizer.minimize(loss=zfit.loss.UnbinnedNLL(model=gauss, data=zfit.data.Data.from_numpy(obs=x, array=pullsk)))
            result = res.hesse()
            # res.errors()
            print(result)
        except Exception as e:
            print(e)
            print("Problem with pull fit.")
            continue
        # plot data
        f = plt.figure()
        plt.figure(figsize=(f.get_size_inches()[0]/2,f.get_size_inches()[0]/2))
        mplhep.histplot(H, color='black', histtype='errorbar', label='Toy data', xerr=True, yerr=True, marker='.', zorder=20)
        nominal = H.values()        
        plt.plot(X, gauss.pdf(X), label=rf'$\mu={mu.value():.2f}({result[mu]["error"]:.2f})$'+'\n'+rf'$\sigma={sig.value():.2f}({result[sig]["error"]:.2f})$', color='black')
        plt.legend()
        plt.yticks([])
        plt.ylabel("Arbitrary Units")
        plt.xlim(-5, 5)
        plt.xlabel(fr'Pull of {labels[k]}')
        plt.savefig(f'plots/angularfit_2d/{args.polynomial}/{name}/pull_{k}.pdf')
        plt.close()

        mu.set_value(0)
        sig.set_value(1)
print("Unweighted N events:", len(datatoy))
print("Weighted sum:", datatoy["fit_weight"].sum())
print("Nsig + Nbkg:", float(Nsig.value() + Nbkg.value()))
if (
    wA0_cow is not None
    and wApp_cow is not None
    and wAS_cow is not None
    and wAq_cow is not None
    and wBkg_cow is not None
):
    print("COW raw sum all:", np.sum(wA0_cow + wApp_cow + wAS_cow + wAq_cow + wBkg_cow))
    print("COW final sum all:", np.sum(wA0_final + wApp_final + wAS_final + wAq_final + wBkg_final))
else:
    print("COW failed, skip final COW sum print.")

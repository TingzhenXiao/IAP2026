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
from scipy.ndimage import gaussian_filter
sys.path.append("/home/submit/xiaot425/IAP2026/efficiency")
import efficiency
import gc
# Extra command-line options added in this unified script:
#   --with-bkg / --no-bkg
#   --with-eff / --no-eff
#   --cow-I g / --cow-I q

# Makes nice default plots
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

np.random.seed(0)
zfit.settings.set_seed(0)
zfit.settings.set_verbosity(0)


_cow_arg_parser = argparse.ArgumentParser(add_help=False)
_cow_arg_parser.add_argument("--with-bkg", dest="with_bkg", action="store_true", default=True)
_cow_arg_parser.add_argument("--no-bkg", dest="with_bkg", action="store_false")
_cow_arg_parser.add_argument("--with-eff", dest="with_eff", action="store_true", default=True)
_cow_arg_parser.add_argument("--no-eff", dest="with_eff", action="store_false")
_cow_arg_parser.add_argument(
    "--cow-I",
    dest="cow_I",
    choices=["g", "q"],
    default=os.environ.get("COW_I", "g"),
)

_cow_arg_parser.add_argument(
    "--bkg-series-degree",
    dest="bkg_series_degree",
    type=int,
    default=4,
)
_cow_arg_parser.add_argument("--ntoys", dest="ntoys", type=int, default=None)
_cow_arg_parser.add_argument(
    "--error-method",
    dest="error_method",
    choices=["hesse", "minos"],
    default="hesse",
)
_cow_cli_args, _remaining_argv = _cow_arg_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

args = tools.parser()
args.with_bkg = bool(_cow_cli_args.with_bkg)
args.with_eff = bool(_cow_cli_args.with_eff)
args.cow_I = str(_cow_cli_args.cow_I).lower()
args.ntoys = _cow_cli_args.ntoys
args.bkg_series_degree = int(_cow_cli_args.bkg_series_degree)
args.error_method = _cow_cli_args.error_method

print("COW configuration:")
print("  with_bkg =", args.with_bkg)
print("  with_eff =", args.with_eff)
print("  cow_I    =", args.cow_I)
print("  bkg series degree =", args.bkg_series_degree)
print("  error method =", args.error_method)

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
name += f"_bkgSeriesDegree{args.bkg_series_degree}"

angle_param_names = [
    "App",
    "A0",
    "Aqc",
    "Aqs",
    "AfbHS",
    "AfbHC",
    "AfbLS",
    "AfbLC",
]

fixed_angle_params = []

for pname in args.fix_to_truth:
    if pname in angle_param_names:
        fixed_angle_params.append(pname)

for pname in args.fix_to_zero:
    if pname in angle_param_names:
        fixed_angle_params.append(pname)

for j in range(0, len(args.fix_to_value), 2):
    pname = args.fix_to_value[j]
    if pname in angle_param_names:
        fixed_angle_params.append(pname)

fixed_angle_params = sorted(set(fixed_angle_params))

if len(fixed_angle_params) > 0:
    angle_mode_tag = "fixAngles_" + "_".join(fixed_angle_params)
else:
    angle_mode_tag = "floatAngles"

name += f"_{angle_mode_tag}"
name += f"_{args.error_method}"

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
    efficiency_input = "/ceph/submit/data/user/x/xiaot425/efficiency/efficiency_applied_output/signal_with_efficiency.h5"
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

n_sig_total = len(df_sig)
n_bkg_total = len(df_bkg)

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
    pname = args.fix_to_value[i]
    pvalue = float(args.fix_to_value[i + 1])
    
    if pname not in truth:
        truth[pname] = {}

    truth[pname]["value"] = pvalue

if "App" not in truth or not isinstance(truth["App"], dict) or "value" not in truth["App"]:
    truth["App"] = {"value": 0.1670}
    
if args.toy:
    ntoys = 100 if args.ntoys is None else args.ntoys
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
    show_legend=True,
    scientific_y=False,
    show_pull=True,
):
    bin_width = bin_edges[1] - bin_edges[0]

    if show_pull:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )
    else:
        fig, ax1 = plt.subplots(
            1,
            1,
            figsize=(8, 6),
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
            if comp.get("separate", False):
                ax1.fill_between(
                    comp["x"],
                    np.zeros_like(comp["y"]),
                    comp["y"],
                    color=comp["color"],
                    alpha=comp.get("alpha", 0.6),
                    label=comp["label"],
                    linewidth=0,
                    zorder=comp.get("zorder", 20),
                )
                continue

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
                zorder=comp.get("zorder", 0),
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

    # Force the main panel to start from zero
    ymin = min(0.0, ymin)

    if ymin < 0.0:
        ymin = 1.30 * ymin

    ymax = 1.15 * ymax if ymax > 0.0 else 1.0

    ax1.set_ylim(ymin, ymax)

    ax1.set_ylabel(ylabel, fontsize=22)
    if scientific_y:
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax1.yaxis.get_offset_text().set_fontsize(20)
    ax1.tick_params(axis="both", labelsize=20)
    if show_legend:
        ax1.legend(loc="upper right", handlelength=1.5, fontsize=12)

    if xlim is not None:
        ax1.set_xlim(*xlim)

    if show_pull:
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

        ax2.set_xlabel(xlabel, fontsize=22)
        ax2.set_ylabel("Pull", fontsize=22)
        ax2.set_ylim(*ylim_pull)
        ax2.tick_params(axis="both", labelsize=20)

        fig.subplots_adjust(
            hspace=0.08,
            left=0.14,
            right=0.97,
            top=0.97,
            bottom=0.10,
        )
    else:
        ax1.set_xlabel(xlabel, fontsize=22)
        fig.subplots_adjust(
            left=0.16,
            right=0.97,
            top=0.97,
            bottom=0.14,
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
Nsig = zfit.Parameter("Nsig", n_sig_total, 0.0, 1.0e8)
Nbkg = zfit.Parameter("Nbkg", n_bkg_total, 0.0, 1.0e8)

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
sigma_sig_2 = zfit.Parameter("sigma_sig_2", 0.02134, 0.006, 0.050)

frac_cb1 = zfit.Parameter("frac_cb1", 0.73, 0, 1.0)

alphal_1 = zfit.Parameter("alphal_1", 1.4112223895658047)
nl_1     = zfit.Parameter("nl_1",     4.7663298220603485)
alphar_1 = zfit.Parameter("alphar_1", 2.283215192055652)
nr_1     = zfit.Parameter("nr_1",     2.51561783579001)

alphal_2 = zfit.Parameter("alphal_2", 2.0960411025639716)
nl_2     = zfit.Parameter("nl_2",     0.21353297304073857)
alphar_2 = zfit.Parameter("alphar_2", 2.431796143783731)
nr_2     = zfit.Parameter("nr_2",     1.353294122042274)

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


lambda_bkg_1 = zfit.Parameter("lambda_bkg_1", -6.0, -20.0, 10.0)
lambda_bkg_2 = zfit.Parameter("lambda_bkg_2", -1.0, -20.0, 10.0)
frac_bkg_exp1 = zfit.Parameter("frac_bkg_exp1", 0.7, 0.0, 1.0)

fitpdf_bkg_mass_1 = zfit.pdf.Exponential(
    obs=mass,
    lambda_=lambda_bkg_1,
)

fitpdf_bkg_mass_2 = zfit.pdf.Exponential(
    obs=mass,
    lambda_=lambda_bkg_2,
)

fitpdf_bkg_mass = zfit.pdf.SumPDF(
    [fitpdf_bkg_mass_1, fitpdf_bkg_mass_2],
    fracs=frac_bkg_exp1,
)
a1_cosh = zfit.Parameter("a1_cosh", 0.0, -0.5, 0.5)
a2_cosh = zfit.Parameter("a2_cosh", -0.2, -0.8, 0.8)

a1_cosl = zfit.Parameter("a1_cosl", 0.0, -0.5, 0.5)
a2_cosl = zfit.Parameter("a2_cosl", -0.4, -0.8, 0.8)

# lambda_bkg.floating = True
a1_cosh.floating = False
a2_cosh.floating = False
a1_cosl.floating = False
a2_cosl.floating = False

# fitpdf_bkg_mass = zfit.pdf.Exponential(obs=mass, lambda_=lambda_bkg)
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

fit_start_params = [
    p for p in fitpdf.get_params()
    if p.floating
]

print("\nFloating parameters:")
for p in fit_start_params:
    print("  ", p.name)

fit_start_values = {
    p: float(p.value())
    for p in fit_start_params
}

def reset_fit_start_values():
    for p, value in fit_start_values.items():
        if p.floating:
            p.set_value(value)

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
    "wS": r"$n_0^S=\beta^2(|A_0^{\prime L}|^2+|A_0^{\prime R}|^2)$",
    "wA0": r"$n_0^P=\beta^2(|A_0^L|^2+|A_0^R|^2)$",
    "wApp": r"$n_1^P=\beta^2(|A_\perp^L|^2+|A_\perp^R|^2+|A_\parallel^L|^2+|A_\parallel^R|^2)$",
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
        pulls[p.name] = np.full(ntoys, np.nan)

print("\nPull plots will be made for floating parameters with truth values:")
for pname in pulls.keys():
    print("  ", pname)

X = np.linspace(-1, 1, 100)


# Check that the pdf is well defined
assert (np.sum(fitpdf.pdf(data).numpy() <= 0) == 0)
assert (np.sum(np.isnan(fitpdf.pdf(data).numpy())) == 0)
assert (np.sum(np.isinf(fitpdf.pdf(data).numpy())) == 0)
assert (np.sum(np.isnan(np.log(fitpdf.pdf(data).numpy()))) == 0)
assert (np.sum(np.isinf(np.log(fitpdf.pdf(data).numpy()))) == 0)

def make_q_norm(
    df,
    pdfs_cow=None,
    yields_cow=None,
    bins=(6, 6, 10),
    smooth_sigma=0,
):
    print("q/r histogram bins =", bins)
    print("q/r histogram smooth_sigma =", smooth_sigma)
    values = df[["cosh", "cosl", "B_mass"]].to_numpy(dtype=float)
    eff_weight = df["fit_weight"].to_numpy(dtype=float)

    q_range = [
        (-1.0, 1.0),
        (-1.0, 1.0),
        (5.170, 5.500),
    ]

    hist_num, edges = np.histogramdd(
        values,
        bins=bins,
        range=q_range,
        weights=eff_weight**2,
        density=False,
    )

    hist_den, _ = np.histogramdd(
        values,
        bins=bins,
        range=q_range,
        weights=None,
        density=False,
    )

    hist_num = np.asarray(hist_num, dtype=float)
    hist_den = np.asarray(hist_den, dtype=float)

    if smooth_sigma is not None and smooth_sigma > 0.0:
        hist_num = gaussian_filter(hist_num, sigma=smooth_sigma)
        hist_den = gaussian_filter(hist_den, sigma=smooth_sigma)

    positive_den = hist_den[np.isfinite(hist_den) & (hist_den > 0.0)]

    if len(positive_den) == 0:
        raise RuntimeError("r(m) histogram denominator is empty.")

    den_floor = 1.0e-6 * np.mean(positive_den)
    hist_den = np.maximum(hist_den, den_floor)

    r_hist = hist_num / hist_den

    positive_r = r_hist[np.isfinite(r_hist) & (r_hist > 0.0)]

    if len(positive_r) == 0:
        raise RuntimeError("r(m) histogram is empty.")

    r_floor = 1.0e-3 * np.mean(positive_r)
    r_hist = np.maximum(r_hist, r_floor)

    r_hist = r_hist / np.mean(r_hist)

    print("\n[Debug r(m) = E(1/eff^2 | m)]")
    print("r hist min =", np.nanmin(r_hist))
    print("r hist max =", np.nanmax(r_hist))
    print("r hist mean =", np.nanmean(r_hist))
    print("r hist median =", np.nanmedian(r_hist))
    print("r hist zero =", np.sum(r_hist <= 0.0))
    print("r hist nan =", np.sum(~np.isfinite(r_hist)))
    print(
        "r hist percentiles =",
        np.percentile(r_hist, [0, 1, 5, 10, 50, 90, 95, 99, 100]),
    )

    def prepare_points(x):
        arr = np.asarray(x, dtype=float)

        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError("Expected one 3D point.")
            pts = arr.reshape(1, 3)

        elif arr.ndim == 2:
            if arr.shape[1] == 3:
                pts = arr
            elif arr.shape[0] == 3:
                pts = arr.T
            else:
                raise ValueError(f"Unexpected q_norm input shape: {arr.shape}")

        else:
            raise ValueError(f"Unexpected q_norm input ndim: {arr.ndim}")

        return pts

    def r_norm(x):
        pts = prepare_points(x)

        idx0 = np.searchsorted(edges[0], pts[:, 0], side="right") - 1
        idx1 = np.searchsorted(edges[1], pts[:, 1], side="right") - 1
        idx2 = np.searchsorted(edges[2], pts[:, 2], side="right") - 1

        idx0 = np.clip(idx0, 0, r_hist.shape[0] - 1)
        idx1 = np.clip(idx1, 0, r_hist.shape[1] - 1)
        idx2 = np.clip(idx2, 0, r_hist.shape[2] - 1)

        vals = r_hist[idx0, idx1, idx2]
        vals = np.asarray(vals, dtype=float).reshape(-1)

        return vals

    def g_fit_norm(x):
        if pdfs_cow is None or yields_cow is None:
            raise RuntimeError("make_q_norm with ratio method requires pdfs_cow and yields_cow.")

        pts = prepare_points(x).T

        total = None
        ysum = np.sum(yields_cow)

        for y, pdf in zip(yields_cow, pdfs_cow):
            vals = (y / ysum) * np.asarray(pdf(pts), dtype=float)

            if total is None:
                total = np.zeros_like(vals, dtype=float)

            total += vals

        return total

    def q_norm(x):
        return g_fit_norm(x) * r_norm(x)

    q_data = q_norm(values)
    g_data = g_fit_norm(values)
    r_data = r_norm(values)

    print("\n[Debug q_norm = g_fit * r on input data]")
    print("g(data) min =", np.nanmin(g_data))
    print("g(data) max =", np.nanmax(g_data))
    print("g(data) mean =", np.nanmean(g_data))
    print("r(data) min =", np.nanmin(r_data))
    print("r(data) max =", np.nanmax(r_data))
    print("r(data) mean =", np.nanmean(r_data))
    print("q(data) min =", np.nanmin(q_data))
    print("q(data) max =", np.nanmax(q_data))
    print("q(data) mean =", np.nanmean(q_data))
    print("q(data) zero =", np.sum(q_data <= 0.0))
    print("q(data) nan =", np.sum(~np.isfinite(q_data)))

    return q_norm
def make_q_norm(df, bins=(3, 3, 6)):
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

def prepare_cow_points_3d(x):
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("Expected one 3D point.")
        pts = arr.reshape(1, 3)

    elif arr.ndim == 2:
        if arr.shape[0] == 3:
            pts = arr.T
        elif arr.shape[1] == 3:
            pts = arr
        else:
            raise ValueError(f"Unexpected shape {arr.shape}")

    else:
        raise ValueError(f"Unexpected ndim {arr.ndim}")

    return pts

def make_background_mass_series_basis_3d(base_pdf, max_degree=2, n_norm=300000, seed=12345):
    bmass_min = 5.170
    bmass_max = 5.500
    volume = 2.0 * 2.0 * (bmass_max - bmass_min)

    rng = np.random.default_rng(seed)

    norm_cosh = rng.uniform(-1.0, 1.0, n_norm)
    norm_cosl = rng.uniform(-1.0, 1.0, n_norm)
    norm_mass = rng.uniform(bmass_min, bmass_max, n_norm)

    norm_points = np.vstack([norm_cosh, norm_cosl, norm_mass])
    base_norm_values = np.asarray(base_pdf(norm_points), dtype=float).reshape(-1)

    norm_pts = prepare_cow_points_3d(norm_points)

    def mass_monomial(pts, degree):
        bmass_values = pts[:, 2]
        u_m = (bmass_values - bmass_min) / (bmass_max - bmass_min)
        # u_m = np.clip(u_m, 0.0, 1.0)
        return u_m ** degree

    basis_functions = []
    basis_labels = []

    for degree in range(max_degree + 1):
        label = f"bkg_nominal_times_mass_degree{degree}"

        factor_norm_values = mass_monomial(norm_pts, degree)
        integral_estimate = volume * np.mean(base_norm_values * factor_norm_values)

        if not np.isfinite(integral_estimate) or integral_estimate <= 0.0:
            raise RuntimeError(
                f"Bad normalization for {label}: {integral_estimate}"
            )

        def make_basis(deg, norm_value, basis_label):
            def basis(x):
                pts = prepare_cow_points_3d(x)

                base_values = np.asarray(base_pdf(x), dtype=float).reshape(-1)
                factor_values = mass_monomial(pts, deg)

                values = base_values * factor_values / norm_value
                return np.asarray(values, dtype=float).reshape(-1)

            basis.__name__ = basis_label
            return basis

        basis_functions.append(make_basis(degree, integral_estimate, label))
        basis_labels.append(label)

    return basis_functions, basis_labels

def make_background_series_basis_3d(base_pdf, max_degree=1, n_norm=300000, seed=12345):
    """
    Build normalized positive background basis functions

        g_b,r(m) = g_b,0(m) * monomial_r(m) / normalization_r

    where g_b,0(m) is the nominal factorised background PDF.
    This is much more stable than using pure monomials.
    """

    bmass_min = 5.170
    bmass_max = 5.500
    volume = 2.0 * 2.0 * (bmass_max - bmass_min)

    rng = np.random.default_rng(seed)

    norm_cosh = rng.uniform(-1.0, 1.0, n_norm)
    norm_cosl = rng.uniform(-1.0, 1.0, n_norm)
    norm_mass = rng.uniform(bmass_min, bmass_max, n_norm)

    norm_points = np.vstack([norm_cosh, norm_cosl, norm_mass])
    base_norm_values = np.asarray(base_pdf(norm_points), dtype=float)

    basis_functions = []
    basis_labels = []
    degrees = []

    for total_degree in range(max_degree + 1):
        for deg_mass in range(total_degree + 1):
            for deg_cosh in range(total_degree - deg_mass + 1):
                deg_cosl = total_degree - deg_mass - deg_cosh
                degrees.append((deg_mass, deg_cosh, deg_cosl))

    def monomial_values_from_points(pts, deg_mass, deg_cosh, deg_cosl):
        cosh_values = pts[:, 0]
        cosl_values = pts[:, 1]
        bmass_values = pts[:, 2]

        u_h = 0.5 * (cosh_values + 1.0)
        u_l = 0.5 * (cosl_values + 1.0)
        u_m = (bmass_values - bmass_min) / (bmass_max - bmass_min)

        u_h = np.clip(u_h, 0.0, 1.0)
        u_l = np.clip(u_l, 0.0, 1.0)
        u_m = np.clip(u_m, 0.0, 1.0)

        return (u_m ** deg_mass) * (u_h ** deg_cosh) * (u_l ** deg_cosl)

    norm_pts_for_monomial = prepare_cow_points_3d(norm_points)

    def make_one_basis(deg_mass, deg_cosh, deg_cosl):
        label = f"bkg_nominal_times_m{deg_mass}_h{deg_cosh}_l{deg_cosl}"

        factor_norm_values = monomial_values_from_points(
            norm_pts_for_monomial,
            deg_mass,
            deg_cosh,
            deg_cosl,
        )

        integral_estimate = volume * np.mean(base_norm_values * factor_norm_values)

        if not np.isfinite(integral_estimate) or integral_estimate <= 0.0:
            raise RuntimeError(
                f"Bad normalization for {label}: {integral_estimate}"
            )

        def basis(x):
            pts = prepare_cow_points_3d(x)

            base_values = np.asarray(base_pdf(x), dtype=float).reshape(-1)

            factor_values = monomial_values_from_points(
                pts,
                deg_mass,
                deg_cosh,
                deg_cosl,
            )

            values = base_values * factor_values / integral_estimate
            return np.asarray(values, dtype=float).reshape(-1)

        basis.__name__ = label
        return label, basis

    for deg_mass, deg_cosh, deg_cosl in degrees:
        label, basis = make_one_basis(deg_mass, deg_cosh, deg_cosl)
        basis_labels.append(label)
        basis_functions.append(basis)

    return basis_functions, basis_labels

    
def make_background_selected_series_basis_3d(
    base_pdf,
    max_degree_mass=2,
    max_degree_cosh=0,
    max_degree_cosl=2,
    include_cross_terms=False,
    n_norm=300000,
    seed=12345,
):
    bmass_min = 5.170
    bmass_max = 5.500
    volume = 2.0 * 2.0 * (bmass_max - bmass_min)

    rng = np.random.default_rng(seed)

    norm_cosh = rng.uniform(-1.0, 1.0, n_norm)
    norm_cosl = rng.uniform(-1.0, 1.0, n_norm)
    norm_mass = rng.uniform(bmass_min, bmass_max, n_norm)

    norm_points = np.vstack([norm_cosh, norm_cosl, norm_mass])
    norm_pts = prepare_cow_points_3d(norm_points)
    base_norm_values = np.asarray(base_pdf(norm_points), dtype=float).reshape(-1)

    def monomial_values(pts, deg_mass, deg_cosh, deg_cosl):
        cosh_values = pts[:, 0]
        cosl_values = pts[:, 1]
        bmass_values = pts[:, 2]

        u_h = 0.5 * (cosh_values + 1.0)
        u_l = 0.5 * (cosl_values + 1.0)
        u_m = (bmass_values - bmass_min) / (bmass_max - bmass_min)

        u_h = np.clip(u_h, 0.0, 1.0)
        u_l = np.clip(u_l, 0.0, 1.0)
        u_m = np.clip(u_m, 0.0, 1.0)

        return (u_m ** deg_mass) * (u_h ** deg_cosh) * (u_l ** deg_cosl)

    degrees = [(0, 0, 0)]

    for d in range(1, max_degree_mass + 1):
        degrees.append((d, 0, 0))

    for d in range(1, max_degree_cosh + 1):
        degrees.append((0, d, 0))

    for d in range(1, max_degree_cosl + 1):
        degrees.append((0, 0, d))

    if include_cross_terms:
        for dm in range(1, max_degree_mass + 1):
            for dh in range(1, max_degree_cosh + 1):
                degrees.append((dm, dh, 0))

            for dl in range(1, max_degree_cosl + 1):
                degrees.append((dm, 0, dl))

        for dh in range(1, max_degree_cosh + 1):
            for dl in range(1, max_degree_cosl + 1):
                degrees.append((0, dh, dl))

    basis_functions = []
    basis_labels = []

    for deg_mass, deg_cosh, deg_cosl in degrees:
        label = f"bkg_nominal_times_m{deg_mass}_h{deg_cosh}_l{deg_cosl}"

        factor_norm_values = monomial_values(
            norm_pts,
            deg_mass,
            deg_cosh,
            deg_cosl,
        )

        integral_estimate = volume * np.mean(base_norm_values * factor_norm_values)

        if not np.isfinite(integral_estimate) or integral_estimate <= 0.0:
            raise RuntimeError(f"Bad normalization for {label}: {integral_estimate}")

        def make_basis(dm, dh, dl, norm_value, basis_label):
            def basis(x):
                pts = prepare_cow_points_3d(x)
                base_values = np.asarray(base_pdf(x), dtype=float).reshape(-1)

                factor_values = monomial_values(
                    pts,
                    dm,
                    dh,
                    dl,
                )

                values = base_values * factor_values / norm_value
                return np.asarray(values, dtype=float).reshape(-1)

            basis.__name__ = basis_label
            return basis

        basis_functions.append(
            make_basis(
                deg_mass,
                deg_cosh,
                deg_cosl,
                integral_estimate,
                label,
            )
        )
        basis_labels.append(label)

    return basis_functions, basis_labels

    
def make_constant_norm_3d():
    def constant_norm(x):
        pts = prepare_cow_points_3d(x)
        return np.ones(len(pts), dtype=float)

    return constant_norm

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

    ymax = 1.5 * ymax if ymax > 0 else 1.0

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

def _hist_and_error(df, weights, var, bin_edges):
    valid = df[var].notna().to_numpy()
    values = df.loc[valid, var].to_numpy(dtype=float)

    if weights is None:
        w = None
    else:
        w = np.asarray(weights, dtype=float)[valid]

    h, _ = np.histogram(
        values,
        bins=bin_edges,
        weights=w,
    )

    if w is None:
        var_h, _ = np.histogram(
            values,
            bins=bin_edges,
        )
    else:
        var_h, _ = np.histogram(
            values,
            bins=bin_edges,
            weights=w**2,
        )

    err = np.sqrt(np.asarray(var_h, dtype=float))

    return h, err


def make_cow_reference_summary_plot(
    datatoy,
    components,
    variables,
    output_path,
    extra_points=None,
    nbins=50,
    ylim_pull=(-4, 4),
):
    ncomp = len(components)
    nrows = 1 + ncomp

    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(15, 3.8 + 1.05 * ncomp),
        sharex="col",
        gridspec_kw={
            "height_ratios": [3.0] + [0.75] * ncomp,
            "hspace": 0.0,
            "wspace": 0.10,
        },
    )

    if nrows == 2:
        axes = np.asarray(axes).reshape(nrows, 2)

    for j, (var, xlabel, xmin, xmax) in enumerate(variables):
        bin_edges = np.linspace(xmin, xmax, nbins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        ax_main = axes[0, j]

        ymins = []
        ymaxs = []

        for comp in components:
            color = comp["color"]

            h_ref, err_ref = _hist_and_error(
                comp["ref_df"],
                comp.get("ref_weights", None),
                var,
                bin_edges,
            )

            h_w, err_w = _hist_and_error(
                datatoy,
                comp["weights"],
                var,
                bin_edges,
            )

            ref_scale = comp.get("ref_scale", 1.0)
            h_ref = ref_scale * h_ref
            err_ref = abs(ref_scale) * err_ref

            if h_ref is None or h_w is None:
                continue

            comp.setdefault("_cache", {})
            comp["_cache"][var] = {
                "h_ref": h_ref,
                "err_ref": err_ref,
                "h_w": h_w,
                "err_w": err_w,
            }

            ax_main.step(
                bin_edges[:-1],
                h_ref,
                where="post",
                color=color,
                linewidth=2.0,
                label="_nolegend_",
            )

            ax_main.errorbar(
                bin_centers,
                h_w,
                yerr=err_w,
                xerr=np.full_like(bin_centers, 0.5 * bin_width),
                fmt="o",
                color=color,
                markersize=3.0,
                elinewidth=1.0,
                capsize=2.0,
                capthick=1.0,
                linestyle="none",
                label=comp["label"],
            )

            ymins.append(np.nanmin(h_w - err_w))
            ymins.append(np.nanmin(h_ref))
            ymaxs.append(np.nanmax(h_w + err_w))
            ymaxs.append(np.nanmax(h_ref))

        if extra_points is not None:
            for extra in extra_points:
                h_extra, err_extra = _hist_and_error(
                    datatoy,
                    extra["weights"],
                    var,
                    bin_edges,
                )

                if h_extra is None:
                    continue

                ax_main.errorbar(
                    bin_centers,
                    h_extra,
                    yerr=err_extra,
                    xerr=np.full_like(bin_centers, 0.5 * bin_width),
                    fmt="o",
                    color=extra["color"],
                    markersize=3.0,
                    elinewidth=1.0,
                    capsize=2.0,
                    capthick=1.0,
                    linestyle="none",
                    label=extra["label"],
                )

                ymins.append(np.nanmin(h_extra - err_extra))
                ymaxs.append(np.nanmax(h_extra + err_extra))

        if len(ymins) > 0 and len(ymaxs) > 0:
            ymin = min(ymins)
            ymax = max(ymaxs)

            if ymin >= 0.0:
                ymin = 0.0
            else:
                ymin = 1.15 * ymin

            ymax = 1.20 * ymax if ymax > 0.0 else 1.0
            ax_main.set_ylim(ymin, ymax)

        ax_main.set_xlim(xmin, xmax)

        # Main projection panel:
        # keep the y-axis label, but remove y-axis numbers.
        ax_main.tick_params(
            axis="x",
            labelsize=16,
            labelbottom=False,
            direction="in",
            top=True,
            right=True,
        )
        ax_main.tick_params(
            axis="y",
            labelsize=16,
            labelleft=False,
            direction="in",
            top=True,
            right=True,
        )

        if j == 0:
            ax_main.set_ylabel("Value of the coefficient [a.u.]", fontsize=20)

        ax_main.legend(
            loc="best",
            fontsize=16,
            handlelength=1.6,
            frameon=False,
        )

        for k, comp in enumerate(components):
            ax_pull = axes[1 + k, j]
            color = comp["color"]

            cache = comp.get("_cache", {}).get(var, None)

            ax_pull.axhline(
                0.0,
                color="black",
                linewidth=1.0,
            )
            ax_pull.axhline(
                2.0,
                color="black",
                linestyle=":",
                linewidth=0.8,
            )
            ax_pull.axhline(
                -2.0,
                color="black",
                linestyle=":",
                linewidth=0.8,
            )

            if cache is not None:
                h_ref = cache["h_ref"]
                err_ref = cache["err_ref"]
                h_w = cache["h_w"]
                err_w = cache["err_w"]

                sigma_pull = np.sqrt(err_ref**2 + err_w**2)

                pull = np.zeros_like(bin_centers, dtype=float)
                mask = sigma_pull > 0.0
                pull[mask] = (h_w[mask] - h_ref[mask]) / sigma_pull[mask]

                ax_pull.bar(
                    bin_centers,
                    pull,
                    width=bin_width,
                    align="center",
                    color=color,
                    linewidth=0,
                )

            # Only show -2 and 2 on pull panels.
            # Do not show "Pull" ylabel or component labels.
            ax_pull.set_ylim(*ylim_pull)
            ax_pull.set_yticks([-2, 0, 2])
            ax_pull.set_yticklabels(["-2", "0", "2"])

            ax_pull.tick_params(
                axis="both",
                labelsize=14,
                direction="in",
                top=True,
                right=True,
            )

            if k < ncomp - 1:
                ax_pull.tick_params(labelbottom=False)
            else:
                ax_pull.set_xlabel(xlabel, fontsize=20)

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.985,
        bottom=0.075,
        hspace=0.0,
        wspace=0.10,
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

fit_values = {}
fit_uncertainties = {}
i = 0
while i < ntoys:
    print("Toy", i)
    seed = np.random.randint(0, 2**32 - 1)
    zfit.settings.set_seed(seed)
    np.random.seed(seed)

    reset_fit_start_values()

    # Create minimizer.
    if args.toy:
        minimizer = zfit.minimize.Minuit(strategy=zfit.minimize.DefaultToyStrategy)
    else:
        minimizer = zfit.minimize.Minuit()

    if args.toy:
        NN = np.random.poisson(args.nsig)

        datatoy = datai.sample(
            n=NN,
            replace=True,
            random_state=seed,
        )

        if len(args.binned) == 2:
            datatoy.query(
                f"({args.binned[0]}<q2) &(q2<{args.binned[1]})",
                inplace=True,
            )

        print("Toy generated signal =", np.sum(datatoy["is_signal"] == 1))
        print("Toy generated bkg    =", np.sum(datatoy["is_signal"] == 0))

        if args.with_eff:
            print("Toy weighted signal  =", datatoy.query("is_signal == 1")["fit_weight"].sum())
            print("Toy weighted bkg     =", datatoy.query("is_signal == 0")["fit_weight"].sum())
            print("Toy total weighted   =", datatoy["fit_weight"].sum())

        data = zfit.Data.from_pandas(
            datatoy[["cosh", "cosl", "B_mass"]],
            obs=obs,
            weights=datatoy["fit_weight"].to_numpy(),
        )

        Nsig.set_value(
            datatoy.query("is_signal == 1")["fit_weight"].sum()
        )

        if args.with_bkg:
            Nbkg.set_value(
                datatoy.query("is_signal == 0")["fit_weight"].sum()
            )
        else:
            Nbkg.set_value(0.0)

    else:
        datatoy = datai
        data = zfit.Data.from_pandas(
            datatoy[["cosh", "cosl", "B_mass"]],
            obs=obs,
            weights=datatoy["fit_weight"].to_numpy(),
        )

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

    print(result)

    # Check that the fit itself is valid before calculating uncertainties.
    if not result.valid:
        print("Fit not valid.")
        reset_fit_start_values()


    covmat = np.full(
        (len(result.params), len(result.params)),
        np.nan,
        dtype=float,
    )

    cow_I = args.cow_I
    case_tag = (
        f"{'bkg' if args.with_bkg else 'nobkg'}_"
        f"{'eff' if args.with_eff else 'noeff'}_"
        f"I{cow_I}_"
        f"bkgSeriesDegree{args.bkg_series_degree}_"
        f"{angle_mode_tag}_"
        f"{args.error_method}"
    )
    outdir_results = f"results_cow_{case_tag}"
    os.makedirs(outdir_results, exist_ok=True)

    corr_param_names = [p.name for p in result.params]
    corr_outname = f"{outdir_results}/{i}_correlation_matrix.pdf"


    # try:
    #     plot_correlation_matrix(covmat, corr_param_names, corr_outname)
    #     print("Saved correlation matrix to:")
    #     print(corr_outname)
    # except Exception as e:
    #     print("Warning: failed to save correlation matrix.")
    #     print(e)

    # Save the fit results
    paramdict = {}
    pi = 0
    for p in result.params:
        pinfo = result.params[p]

        value = pinfo["value"]
        error = None
        error_upper = None
        error_lower = None

        # First read the Hesse error if available.
        if "hesse" in pinfo and pinfo["hesse"] is not None:
            error = pinfo["hesse"].get("error", None)

            if error is not None and np.isfinite(error):
                error = float(error)
                error_upper = float(error)
                error_lower = -float(error)

        # If MINOS errors are available, overwrite the upper/lower errors.
        # This makes the pull use MINOS instead of Hesse.
        if "errors" in pinfo and pinfo["errors"] is not None:
            minos_upper = pinfo["errors"].get("upper", None)
            minos_lower = pinfo["errors"].get("lower", None)

            if minos_upper is not None and np.isfinite(minos_upper):
                error_upper = float(minos_upper)

            if minos_lower is not None and np.isfinite(minos_lower):
                error_lower = float(minos_lower)

            if error_upper is not None and error_lower is not None:
                error = 0.5 * (abs(error_upper) + abs(error_lower))

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

    if args.toy:
        for pname, parinfo in paramdict.items():
            value = parinfo.get("value", None)
            err_plus = parinfo.get("error_upper", None)
            err_minus = parinfo.get("error_lower", None)

            if value is None:
                continue
    
            if err_plus is None or err_minus is None:
                continue

            err = max(abs(float(err_plus)), abs(float(err_minus)))

            if not np.isfinite(value) or not np.isfinite(err):
                continue

            if pname not in fit_values:
                fit_values[pname] = []
                fit_uncertainties[pname] = []

            fit_values[pname].append(float(value))
            fit_uncertainties[pname].append(float(err))
    if args.toy:
        truth_values = {}

        for tname, tinfo in truth.items():
            if isinstance(tinfo, dict) and "value" in tinfo:
                truth_values[tname] = float(tinfo["value"])
            else:
                truth_values[tname] = float(tinfo)

        truth_values["Nsig"] = float(
            datatoy.query("is_signal == 1")["fit_weight"].sum()
        )

        if args.with_bkg:
            truth_values["Nbkg"] = float(
                datatoy.query("is_signal == 0")["fit_weight"].sum()
            )
        else:
            truth_values["Nbkg"] = 0.0

        for pname in pulls.keys():
            if pname not in paramdict:
                continue

            if pname not in truth_values:
                continue

            value = paramdict[pname]["value"]
            diff = value - truth_values[pname]

            error = paramdict[pname]["error"]
            error_upper = paramdict[pname]["error_upper"]
            error_lower = paramdict[pname]["error_lower"]

            if error_upper is not None and error_lower is not None:
                if diff >= 0:
                    err_for_pull = abs(error_lower)
                else:
                    err_for_pull = abs(error_upper)
            else:
                err_for_pull = error

            if err_for_pull is None or not np.isfinite(err_for_pull) or err_for_pull <= 0:
                pulls[pname][i] = np.nan
            else:
                pulls[pname][i] = diff / err_for_pull
        try:
            del data
            del loss
            del result
            del datatoy
        except Exception:
            pass

        plt.close("all")

        try:
            zfit.run.clear_graph_cache()
        except Exception:
            pass

        gc.collect()

        i += 1
        continue
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

        pdfs_sig_cow = [
            fA0_3d,
            fApp_3d,
            fAS_3d,
            fAq_3d,
        ]

        if args.with_bkg:
            fBkg_3d = zfit_pdf_to_callable_3d_for_cows(pdfBkg_full, obs)

            pdfs_bkg_cow, bkg_basis_labels = make_background_mass_series_basis_3d(
                base_pdf=fBkg_3d,
                max_degree=args.bkg_series_degree,
            )

        else:
            pdfs_bkg_cow = []
            bkg_basis_labels = []

        print("\n[COW background series basis]")
        print("Number of background basis functions:", len(pdfs_bkg_cow))
        for label in bkg_basis_labels:
            print(" ", label)

        pdfs_cow = pdfs_sig_cow + pdfs_bkg_cow

        yields_cow = [
            float(N_A0.value()),
            float(N_App.value()),
            float(N_AS.value()),
            float(N_Aq.value()),
        ]

        if args.with_bkg:
            nbkg_basis = len(pdfs_bkg_cow)

            if nbkg_basis <= 0:
                raise RuntimeError("Background is enabled, but no background basis was built.")

            bkg_basis_yield = float(Nbkg.value()) / nbkg_basis
            yields_cow += [bkg_basis_yield for _ in range(nbkg_basis)]

        ranges_cow = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (5.170, 5.500),
        ]

        cow_I = args.cow_I

        if cow_I not in ["g", "q"]:
            raise ValueError(f"Unknown COW I choice: {cow_I}. Use 'g', or 'q'.")

        if cow_I == "q":
            cow_norm = make_q_norm(
                datatoy,
                pdfs_cow=pdfs_cow,
                yields_cow=yields_cow,
                bins=(1, 1, 1),
                smooth_sigma=1,
            )
            print("Using COWs with I = q from external q_norm.")

            q_vals = np.asarray(cow_norm(data_cow), dtype=float)

            g_vals = np.zeros_like(q_vals, dtype=float)
            ysum = np.sum(yields_cow)

            for y, pdf in zip(yields_cow, pdfs_cow):
                g_vals += (y / ysum) * np.asarray(pdf(data_cow), dtype=float)

            mask_ratio = np.isfinite(q_vals) & np.isfinite(g_vals) & (g_vals > 0.0)

            ratio = q_vals[mask_ratio] / g_vals[mask_ratio]

            print("\n[Debug q_hist / g_fit on data]")
            print("q_vals min =", np.nanmin(q_vals))
            print("q_vals max =", np.nanmax(q_vals))
            print("q_vals mean =", np.nanmean(q_vals))
            print("g_vals min =", np.nanmin(g_vals))
            print("g_vals max =", np.nanmax(g_vals))
            print("g_vals mean =", np.nanmean(g_vals))
            print("ratio min =", np.min(ratio))
            print("ratio max =", np.max(ratio))
            print("ratio mean =", np.mean(ratio))
            print("ratio median =", np.median(ratio))
            print(
                "ratio percentiles =",
                np.percentile(ratio, [0, 1, 5, 10, 50, 90, 95, 99, 100]),
            )

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
            try:
                wBkg_cow = cow["b"](data_cow)
            except Exception:
                bkg_weights = []

                for ibkg in range(len(pdfs_bkg_cow)):
                    bkg_index = len(pdfs_sig_cow) + ibkg
                    bkg_weights.append(cow[bkg_index](data_cow))

                wBkg_cow = np.sum(np.vstack(bkg_weights), axis=0)
        else:
            wBkg_cow = np.zeros_like(wA0_cow, dtype=float)
        print("\n[Debug signal weights on true signal/background]")
        is_sig = datatoy["is_signal"].to_numpy(dtype=bool)

        for label, weight in [
            ("wA0", wA0_cow),
            ("wApp", wApp_cow),
            ("wAS", wAS_cow),
            ("wAq", wAq_cow),
            ("wBkg", wBkg_cow),
        ]:
            print(label)
            print("  sum all          =", np.sum(weight))
            print("  sum true signal  =", np.sum(weight[is_sig]))
            print("  sum true bkg     =", np.sum(weight[~is_sig]))
            print("  mean true signal =", np.mean(weight[is_sig]))
            print("  mean true bkg    =", np.mean(weight[~is_sig]))
            
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


        # Save COW-weighted data for the 2D coverage study.
        if args.toy:
            coverage_outdir = os.path.join(
                outdir_results,
                "coverage_toys",
            )
            os.makedirs(coverage_outdir, exist_ok=True)

            coverage_df = datatoy.copy()

            coverage_df["wA0"] = np.asarray(wA0_final, dtype=float)
            coverage_df["wApp"] = np.asarray(wApp_final, dtype=float)
            coverage_df["wS"] = np.asarray(wAS_final, dtype=float)
            coverage_df["wAq"] = np.asarray(wAq_final, dtype=float)
            coverage_df["wBkg"] = np.asarray(wBkg_final, dtype=float)

            coverage_outname = os.path.join(
                coverage_outdir,
                f"{i}.h5",
            )

            coverage_df.to_hdf(
                coverage_outname,
                key="data",
                mode="w",
            )

            print("Saved COW weights for the 2D coverage study to:")
            print(coverage_outname)

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
        outdir_cow = f"plots_degree/{args.polynomial}/{name}/reference_cow_multicomp_{case_tag}"
        os.makedirs(outdir_cow, exist_ok=True)

        is_sig = datatoy["is_signal"].to_numpy(dtype=bool)

    if not args.toy:
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


        weight_dict_cow = {
            "wA0": wA0_final,
            "wApp": wApp_final,
            "wS": wAS_final,
        }

        reference_colors = {
            "wS": "gold",
            "wA0": "navy",
            "wApp": "dodgerblue",
            "wBkg": "#2E6F40",
            "wAq": "firebrick",
        }

        reference_short_labels = {
            "wS": r"$A_S$",
            "wA0": r"$A_0$",
            "wApp": r"$A_{\parallel,\perp}$",
            "wBkg": r"Bkg",
        }

        reference_components = []

        for weight_name, (ref_file, ref_tree) in reference_mapping.items():
            ref_path = os.path.join(datadir, ref_file)

            with uproot.open(ref_path) as fref:
                ref_df = fref[ref_tree].arrays(library="pd")

            if "cosThetaL" in ref_df.columns:
                ref_df["cosl"] = ref_df["cosThetaL"]

            if "cosThetaK" in ref_df.columns:
                ref_df["cosh"] = ref_df["cosThetaK"]

            if "B_mass" in ref_df.columns:
                if ref_df["B_mass"].max() > 100.0:
                    ref_df["B_mass"] = ref_df["B_mass"] / 1000.0

                ref_df = ref_df[
                    (ref_df["B_mass"] >= 5.170)
                    & (ref_df["B_mass"] <= 5.500)
                ].copy()

            if "mKpi" in ref_df.columns:
                ref_df = ref_df[ref_df["mKpi"] < 1.5].copy()

            if "q2" in ref_df.columns:
                ref_df = ref_df[
                    (ref_df["q2"] > 1.1)
                    & (ref_df["q2"] < 7.0)
                ].copy()

            ref_df.dropna(inplace=True)

            ref_scale = np.sum(weight_dict_cow[weight_name]) / len(ref_df)

            reference_components.append(
                {
                    "name": weight_name,
                    "label": reference_display[weight_name],
                    "short_label": reference_short_labels[weight_name],
                    "weights": weight_dict_cow[weight_name],
                    "ref_df": ref_df,
                    "ref_weights": None,
                    "ref_scale": ref_scale,
                    "color": reference_colors[weight_name],
                }
            )

        if args.with_bkg:
            bkg_ref_weights = df_bkg["fit_weight"].to_numpy(dtype=float)
            bkg_ref_scale = np.sum(wBkg_final) / np.sum(bkg_ref_weights)

            reference_components.append(
                {
                    "name": "wBkg",
                    "label": r"$n_{\rm bkg}$",
                    "short_label": reference_short_labels["wBkg"],
                    "weights": wBkg_final,
                    "ref_df": df_bkg,
                    "ref_weights": bkg_ref_weights,
                    "ref_scale": bkg_ref_scale,
                    "color": reference_colors["wBkg"],
                }
            )

        make_cow_reference_summary_plot(
            datatoy=datatoy,
            components=reference_components,
            variables=[
                ("mKpi", r"$m(K\pi)$ [GeV/$c^2$]", 0.65, 1.50),
                ("q2", r"$q^2$ [GeV$^2/c^4$]", 1.1, 7.0),
            ],
            output_path=(
                f"{outdir_cow}/{i}_all_reference_projections_"
                f"full_dataset_degree{args.bkg_series_degree}.pdf"
            ),
            extra_points=[
                {
                    "label": r"$n_{\beta}$",
                    "weights": wAq_final,
                    "color": reference_colors["wAq"],
                }
            ],
            nbins=100,
            ylim_pull=(-4, 4),
        )

        print("Saved the combined full-data reference projection plot.")
        print(
            f"{outdir_cow}/{i}_all_reference_projections_"
            f"full_dataset_degree{args.bkg_series_degree}.pdf"
        )
        print("Now making the full-data fit projection plots.")

    # Plot the result
    if i < 3:
        outdir_fit = f"plots_degree/{args.polynomial}/{name}/fit_projections_cow_{case_tag}"
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

            values = datatoy[n].to_numpy(dtype=float)
            weights = datatoy["fit_weight"].to_numpy(dtype=float)

            H = hist.Hist(
                hist.axis.Regular(nbins, xmin, xmax, underflow=False, overflow=False),
                storage=hist.storage.Weight(),
            )

            H.fill(values, weight=weights)

            counts = H.values()
            yerr = np.sqrt(H.variances())
            
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
                    "color": "#2E6F40",
                    "alpha": 0.4,
                    "label": "Background",
                    "hatch": "..",
                },
                {
                    "x": x,
                    "y": yAS,
                    "color": "gold",
                    "alpha": 0.6,
                    "label": r"$n_0^S=\beta^2(|A_0^{\prime L}|^2+|A_0^{\prime R}|^2)$",
                    "hatch": "xx",
                },
                {
                    "x": x,
                    "y": yA0,
                    "color": "navy",
                    "alpha": 0.6,
                    "label": r"$n_0^P=\beta^2(|A_0^L|^2+|A_0^R|^2)$",
                    "hatch": "//",
                },
                {
                    "x": x,
                    "y": yApp,
                    "color": "dodgerblue",
                    "alpha": 0.6,
                    "label": r"$n_1^P=\beta^2(|A_\perp^L|^2+|A_\perp^R|^2+|A_\parallel^L|^2+|A_\parallel^R|^2)$",
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
                    "color": "gray",
                    "alpha": 0.4,
                    "label": "Interference",
                    "separate": True,
                    "zorder": 20,
                },
            ]

            plot_projection_with_pull(
                bin_edges=bin_edges,
                bin_centers=bin_centers,
                data_y=counts,
                data_yerr=yerr,
                pull=pull,
                xlabel=l,
                ylabel=fr"$\sum_i \varepsilon_i$ / {(dist / nbins):.2f}",
                output_path=f"{outdir_fit}/{i}_{n}_cow.pdf",
                data_label="Data",
                line_x=x,
                total_y=Z,
                total_label="Fit",
                stack_components=stack_components,
                xlim=(xmin, xmax),
                show_legend=(n == "cosh"),
                scientific_y=True,
                show_pull=False,
            )


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
                "color": "#2E6F40",
                "alpha": 0.8,
                "label": "Background",
                "hatch": "..",
            },
            {
                "x": x_mass,
                "y": yAS_mass,
                "color": "gold",
                "alpha": 0.6,
                "label": r"$n_0^S=\beta^2(|A_0^{\prime L}|^2+|A_0^{\prime R}|^2)$",
                "hatch": "xx",
            },
            {
                "x": x_mass,
                "y": yA0_mass,
                "color": "navy",
                "alpha": 0.6,
                "label": r"$n_0^P=\beta^2(|A_0^L|^2+|A_0^R|^2)$",
                "hatch": "//",
            },
            {
                "x": x_mass,
                "y": yApp_mass,
                "color": "dodgerblue",
                "alpha": 0.6,
                "label": r"$n_1^P=\beta^2(|A_\perp^L|^2+|A_\perp^R|^2+|A_\parallel^L|^2+|A_\parallel^R|^2)$",
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
            ylabel=fr"$\sum_i \varepsilon_i$ / {bin_width_mass:.4f} (GeV/$c^2$)",
            output_path=f"{outdir_fit}/{i}_B_mass_cow.pdf",
            data_label="Data",
            line_x=x_mass,
            total_y=ybkg_mass + yAS_mass + yA0_mass + yApp_mass + yAq_mass,
            total_label="Fit",
            stack_components=stack_components_mass,
            xlim=(xmin_mass, xmax_mass),
            show_legend=True,
            scientific_y=True,
            show_pull=False,
        )

        print("Saved the full-data fit projection plots to:")
        print(outdir_fit)
        
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
    if args.toy:
        try:
            del data
            del loss
            del result
            del datatoy
            del coverage_df
        except Exception:
            pass

        plt.close("all")

        try:
            zfit.run.clear_graph_cache()
        except Exception:
            pass

        gc.collect()

    i += 1

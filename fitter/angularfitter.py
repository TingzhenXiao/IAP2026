import numpy as np  # Numerical library
import yaml  # For reading YAML files
import uproot  # For reading ROOT files
import matplotlib.pyplot as plt  # Plotting library
import zfit  # Fitting library
import hist  # Histogram library
# from hepstats.splot import compute_sweights  # For sWeights computation

def eval_pdf(model, x, params=None, allow_extended=False):
    """Compute pdf of model at a given point x and for given parameters values"""

    if params is None:
        params = {}

    def pdf(model, x):
        ret = model.ext_pdf(x) if model.is_extended and allow_extended else model.pdf(x)

        return np.array(ret)

    for param in model.get_params():
        if param in params:
            value = params[param]["value"]
            param.set_value(value)
    return pdf(model, x)


def compute_sweights(model, x, weights=None):
    if weights is None:
        weights = np.ones(len(x))

    models = model.get_models()
    yields = [m.get_yield() for m in models]

    # p = np.vstack([np.array(m) for m in models]).T
    # Nx = np.array(model.ext_pdf(data))*weights
    p = np.vstack([eval_pdf(m, x)*weights for m in models]).T
    Nx = eval_pdf(model, x, allow_extended=True)*weights
    pN = p / Nx[:, None]

    Vinv = (pN).T.dot(pN)
    V = np.linalg.inv(Vinv)

    sweights = p.dot(V) / Nx[:, None]

    return {y: sweights[:, i] for i, y in enumerate(yields)}

import json  # For reading JSON files
from myconstants import *
import tools  # Some helpful functions
import mypdfs  # Custom pdfs
import angularfunctions as af  # Angular functions

# Makes nice default plots
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

np.random.seed(0)
zfit.settings.set_seed(0)
zfit.settings.set_verbosity(10)


args = tools.parser()


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

tools.makedirs(args.polynomial, name)

# limits for integrals
limith = zfit.Space(axes=0, lower=-1, upper=1)
limitl = zfit.Space(axes=1, lower=-1, upper=1)
limits = limith * limitl

# create phsp
cosh = zfit.Space('cosh', limits=(-1, 1))
cosl = zfit.Space('cosl', limits=(-1, 1))
angles = cosh * cosl

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
    nbins = 30
else:
    ntoys = 1
    nbins = 50

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


def ASconditions(params):
    # The sum of all amplitudes must be 1.
    # This means that AS is not a free parameter.
    return 1-params['A0']-params['App']-params['Aqc']-params['Aqs']


AS = zfit.ComposedParameter("AS", ASconditions,
                            params={'A0': A0, 'App': App, 'Aqc': Aqc, 'Aqs': Aqs})

# total yield
Nsig = zfit.Parameter("Nsig", truth['Nsig']["value"], 0.0, 1.0e8)


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


# Create the pdf and register the analytic integral (this makes the computations faster)
fitpdf = mypdfs.my2Dpdf(obs=angles, App=App, A0=A0, AS=AS, Aqc=Aqc, Aqs=Aqs,
                        AfbHC=AfbHC, AfbHS=AfbHS, AfbLC=AfbLC, AfbLS=AfbLS)
fitpdf.register_analytic_integral(func=mypdfs.integral, limits=limits)

# pdfs are normalized to an integral of 1 but we want extended pdfs
fitpdf = fitpdf.create_extended(Nsig)


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


# create pdfs for sWeights (no asymmetry terms!)
pdfS = mypdfs.my2Dpdf_AS(obs=angles)
pdfS = pdfS.create_extended(N_AS)
pdfApp = mypdfs.my2Dpdf_App(obs=angles)
pdfApp.register_analytic_integral(func=mypdfs.integral_App, limits=limits)
pdfApp = pdfApp.create_extended(N_App)
pdfA0 = mypdfs.my2Dpdf_A0(obs=angles)
pdfA0.register_analytic_integral(func=mypdfs.integral_A0, limits=limits)
pdfA0 = pdfA0.create_extended(N_A0)
pdfAq = mypdfs.my2Dpdf_Aq(obs=angles, Aqc=Aqc, Aqs=Aqs)
pdfAq.register_analytic_integral(func=mypdfs.integral_Aq, limits=limits)
pdfAq = pdfAq.create_extended(N_Aq)

pdfsweightslist = []
if not (AS.name in args.fix_to_zero):
    pdfsweightslist.append(pdfS)
if not (App.name in args.fix_to_zero):
    pdfsweightslist.append(pdfApp)
if not (A0.name in args.fix_to_zero):
    pdfsweightslist.append(pdfA0)
if not (Aqs.name in args.fix_to_zero and Aqc.name in args.fix_to_zero):
    pdfsweightslist.append(pdfAq)
pdfsweights = zfit.pdf.SumPDF(pdfsweightslist)
pdfs = {m.get_yield(): m for m in pdfsweights.get_models()}


# Read the data
with uproot.open(args.data) as f:
    datai = f["B02KstMuMu_Run1_centralQ2E_sig"].arrays(library="pd")
    datai["cosl"] = datai["cosThetaL"]
    datai["cosh"] = datai["cosThetaK"]
    datai = datai.query(f"({args.mKpi[0]}<mKpi) &(mKpi<{args.mKpi[1]})")
    datai.dropna(inplace=True)
    datai = datai.query(f"({args.qsq[0]}<q2) &(q2<{args.qsq[1]})")
    print("Number of data points:", len(datai))


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
    Nsig.set_value(truth["Nsig"]["value"])
else:
    if len(args.binned) == 2:
        # Data in bins
        datai.query(f"({args.binned[0]}<q2) &(q2<{args.binned[1]})", inplace=True)
datai = datai[:100000]
data = zfit.Data.from_pandas(datai, obs=angles)


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


i = 0
while i < ntoys:
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
        NN = np.random.poisson(args.nsig)
        datatoy = datai.sample(n=NN, replace=True)
        if len(args.binned) == 2:
            datatoy.query(f"({args.binned[0]}<q2) &(q2<{args.binned[1]})", inplace=True)
        data = zfit.Data.from_pandas(datatoy[["cosh","cosl"]], obs=angles)
    else:
        datatoy = datai

    # Create the loss
    loss = zfit.loss.ExtendedUnbinnedNLL(model=fitpdf, data=data)

    # Add constraints if any
    if len(constraints) > 0:
        loss.add_constraints(constraints)

    # Run the fit
    result = minimizer.minimize(loss)
    result.update_params()
    try:
        print(result)
        result.errors()  # Compute uncertainty
    except Exception as e:
        print(e)
        print("Problem with errors.")
        for p in fitpdf.get_params():
            if p.floating:
                print(p)
                p.set_value(truth[p.name]["value"])
                # p.randomize()
                print(p)
        continue

    print(result)
    result.update_params()

    # Check that the result is valid
    covmat = result.covariance()
    posdef = np.all(np.linalg.eigvals(covmat) > 0)
    if not result.valid or not posdef:
        print("Fit not valid.")
        for p in fitpdf.get_params():
            if p.floating:
                p.set_value(truth[p.name]["value"])
        continue

    # Save the fit results
    paramdict = {}
    pi = 0
    for p in result.params:
        paramdict[p.name] = {}
        paramdict[p.name]["value"] = result.params[p]["value"]
        paramdict[p.name]["error_upper"] = result.params[p]["errors"]["upper"]
        paramdict[p.name]["error_lower"] = result.params[p]["errors"]["lower"]
        paramdict[p.name]["covariance"] = {}
        qi = 0
        for q in result.params:
            if q == p:
                qi += 1
                continue
            paramdict[p.name]["covariance"][q.name] = float(covmat[pi][qi])
            qi += 1
        pi += 1

    outname = f"plots/{args.polynomial}/{name}/results/{i}.yml"
    with open(outname, 'w') as yaml_file:
        yaml.dump(paramdict, yaml_file, default_flow_style=False)

    # Compute sWeights
    sweights = compute_sweights(pdfsweights, data)
    try:
        sweights = compute_sweights(pdfsweights, data)
    except Exception as e:
        print(e)
        print("Problem with massfit sweights.")
        continue

    # Sanity check
    diff = Nsig.value()-N_A0.value()-N_App.value()-N_Aq.value()-N_AS.value()
    assert (np.isclose(diff, 0, atol=1e-2))

    sApp, sA0, sAS, sAq = sweights[N_App], sweights[N_A0], sweights[N_AS], sweights[N_Aq]

    # Save the pulls for toys
    if args.toy:
        for k in pulls.keys():
            par = result.params[k]
            if k in args.constrain:
                extra = max(abs(truth[k]["error_upper"]), abs(truth[k]["error_lower"]))
            else:
                extra = 0
            if par['value'] < truth[k]["value"]:
                pull = (truth[k]["value"] - par["value"])/np.sqrt(par["errors"]["upper"]**2+extra**2)
            else:
                pull = (truth[k]["value"] - par["value"])/np.sqrt(par["errors"]["lower"]**2+extra**2)
            pulls[k][i] = pull

    # Plot the result
    if i < 3:
        # Make the same type of plot for costhetah and costhetal
        for v, n, l in zip([cosh, cosl], ["cosh", "cosl"], [r"$\cos(\theta_h)$", r"$\cos(\theta_\ell)$"]):
            dist = v.limits[1][0][0] - v.limits[0][0][0]
            x = np.linspace(v.limits[0][0][0], v.limits[1][0][0], 1000)

            # Step 1) Plot the data
            H = hist.Hist(hist.axis.Regular(nbins, v.limits[0][0][0], v.limits[1][0][0], underflow=False, overflow=False))
            H.fill(data[n])
            mplhep.histplot(H, color='black', histtype='errorbar', label='Toy data', xerr=True, yerr=True, marker='.')

            # Step 2) Plot the individual components
            yAS = pdfs[N_AS].create_projection_pdf(obs=v).ext_pdf(x).numpy()*dist/nbins
            plt.fill_between(x, y1=np.zeros(len(yAS)), y2=yAS, facecolor='gold', alpha=0.6, label=r"$n^S_0=\beta^2(|{A'}_0^L|^2+|{A'}_0^R|^2)$", linewidth=0, edgecolor="w", zorder=0, hatch='xx')
            yA0 = pdfs[N_A0].create_projection_pdf(obs=v).ext_pdf(x).numpy()*dist/nbins
            plt.fill_between(x, y1=yAS, y2=yAS+yA0, color='navy', alpha=0.6, label=r"$n^P_0=\beta^2(|{A}_0^L|^2+|{A}_0^R|^2)$", linewidth=0, zorder=0, edgecolor="w", hatch="//")
            yApp = pdfs[N_App].create_projection_pdf(obs=v).ext_pdf(x).numpy()*dist/nbins
            plt.fill_between(x, y1=yAS+yA0, y2=yAS+yA0+yApp, color='dodgerblue', alpha=0.6, label=r"$n^P_1=\beta^2(|{A}_\perp^L|^2+|{A}_\perp^R|^2+|{A}_\parallel^L|^2+|{A}_\parallel^R|^2)$", linewidth=0, zorder=0, edgecolor="w", hatch="\\\\")
            yAq = pdfs[N_Aq].create_projection_pdf(obs=v).ext_pdf(x).numpy()*dist/nbins
            plt.fill_between(x, y1=yAS+yA0+yApp, y2=yAS+yA0+yApp+yAq, color='firebrick', alpha=0.6, label=r"$n_{\beta}$", linewidth=0, zorder=0, edgecolor="w", hatch="..")

            # Step 3) Plot the total fit
            Z = fitpdf.create_projection_pdf(obs=v).ext_pdf(x).numpy()*dist/nbins
            plt.plot(x, Z, label="Fit", color="k")

            # Step 4) Plot the interference terms (= asymmetries)
            if n == "cosh":
                yInt = (af.proj_AfbHC(x, n)*AfbHC.value() + af.proj_AfbHS(x, n)*AfbHS.value())*Nsig.value()*dist/nbins
                plt.fill_between(x, y1=np.zeros(len(yInt)), y2=yInt, color='gray', alpha=0.4, label=r'Interference', linewidth=0, zorder=0)
                plt.legend(loc='upper right', handlelength=1, handleheight=1, fontsize=24)
            else:
                yInt = (af.proj_AfbLC(x, n)*AfbLC.value() + af.proj_AfbLS(x, n)*AfbLS.value())*Nsig.value()*dist/nbins
                plt.fill_between(x, y1=np.zeros(len(yInt)), y2=yInt, color='gray', alpha=0.4, label=r'Interference', linewidth=0, zorder=0)

            # Some plotting settings
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.xlim(v.limits[0][0], v.limits[1][0])
            plt.xlabel(l, ha="right", x=1)
            plt.ylabel(fr"Data points / {(dist/nbins):.2f}",ha="right",y=1)
            plt.savefig(f"plots/{args.polynomial}/{name}/{i}_{n}.pdf")
            plt.close()

        # Also make weighted plots
        for vkey, l, u in zip(["mKpi", "q2"], [r"$m(K\pi)$", r"$q^2$"], [r"GeV$/c^2$", r"GeV$^2/c^4$"]):
            mi, ma = datatoy[vkey].min(), datatoy[vkey].max()
            dist = ma - mi
            H = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False))
            H.fill(datatoy[vkey])
            mplhep.histplot(H, color='black', histtype='errorbar', label='Toy data', xerr=True, yerr=True, marker='.', zorder=20)
            nominal = H.values()
            y = np.zeros(nbins)  # For stacking the histograms
            lists = zip([r"$n^S_0=\beta^2(|{A'}_0^L|^2+|{A'}_0^R|^2)$", r'$n_0^P=\beta^2(|{A}_0^L|^2+|{A}_0^R|^2)$', r'$n_1^P=\beta^2(|{A}_\perp^L|^2+|{A}_\perp^R|^2+|{A}_\parallel^L|^2+|{A}_\parallel^R|^2)$', r'$n_{\beta}$'],
                        [sAS, sA0, sApp, sAq],
                        ['gold', 'navy', 'dodgerblue', 'firebrick'])
            for n, w, c in lists:
                if all(w == 0):
                    continue
                # Make a weighted histogram and plot it stacked
                H = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
                H.fill(datatoy[vkey], weight=w)
                hvals = H.values()
                for k in range(nbins):
                    plt.fill_between(H.axes[0].edges[k:k+2], y1=y[k], y2=y[k] + hvals[k],
                                     color=c, linewidth=0, zorder=0)
                # Add errorbars
                plt.errorbar(H.axes[0].centers, y+hvals,
                             yerr=np.sqrt(H.variances()), xerr=H.axes[0].widths/2,
                             fmt='.', elinewidth=1, color=c, label=n)
                y += hvals  # Raise the bottom for stacking
            plt.legend(handletextpad=0.1, fontsize=24)
            plt.axhline(0, color='black', linewidth=1)
            plt.xlim(mi, ma)
            ylims = plt.ylim()
            plt.xlabel(l+f" [{u}]", ha="right", x=1)
            plt.ylabel(fr"$\sum$ weights / ({dist/nbins:.2f} {u})", ha="right", y=1)
            plt.savefig(f"plots/{args.polynomial}/{name}/{i}_{vkey}_weighted.pdf")
            plt.close()

    # Save the sWeighted data
    datas = data.to_pandas()
    datas['wS'] = sAS
    datas['wApp'] = sApp
    datas['wA0'] = sA0
    datas['wAq'] = sAq
    datas['mKpi'] = datatoy['mKpi'].values
    datas['q2'] = datatoy['q2'].values
    datas['cosl'] = datatoy['cosl'].values
    datas['cosh'] = datatoy['cosh'].values
    datas.to_hdf(f"sweights/{args.polynomial}/{name}/{i}.h5", key='data', mode='w')
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
        mplhep.histplot(zfit.data.Data.from_numpy(obs=x, array=pullsk).to_binned(5000), color='black', histtype='errorbar', xerr=True, yerr=True, density=True)
        plt.plot(X, gauss.pdf(X), label=rf'$\mu={mu.value():.2f}({result[mu]["error"]:.2f})$'+'\n'+rf'$\sigma={sig.value():.2f}({result[sig]["error"]:.2f})$', color='red')
        plt.legend()
        plt.yticks([])
        plt.ylabel("Arbitrary Units")
        plt.xlim(-5, 5)
        plt.xlabel(fr'Pull of {labels[k]}')
        plt.savefig(f'plots/angularfit_2d/{args.polynomial}/{name}/pull_{k}.pdf')
        plt.close()

        mu.set_value(0)
        sig.set_value(1)

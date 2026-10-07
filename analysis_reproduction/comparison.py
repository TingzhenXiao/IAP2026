"""
Usage:
python comparison.py
    --input path/to/sweights/in/hdf5/format.h5
    --a0 /path/to/a0/reference/file.root
    --a1 /path/to/a1/reference/file.root
    --aS /path/to/aS/reference/file.root
    --name test
    --qsq 1.1 7
    --mKpi 0.65 1.5
    --results path/to/results/file.yml
This produces plots called plots/test_mKpi_comparison.pdf and plots/test_q2_comparison.pdf
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str)
parser.add_argument('--a0',type=str)
parser.add_argument('--a1',type=str)
parser.add_argument('--aS',type=str)
parser.add_argument('--name',type=str,help="Output name")
parser.add_argument('--results',type=str,help="Results file")
parser.add_argument('--mKpi',type=float,nargs=2,default=[0.65,1.5])
parser.add_argument('--qsq',type=float,nargs=2,default=[2.0,11.0])
args = parser.parse_args()


import matplotlib.pyplot as plt
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

import hist

import pandas as pd
import numpy as np

labels = {
    'App' : r'$n_1^P$',
    'AS' : r'$n_0^S$',
    'A0' : r'$n_0^P$',
    'alpha' : r'$\alpha$',
    'beta' : r'$\beta$',
    'Nsig' : r'$N_{\rm sig}$',
    'Nbkg' : r'$N_{\rm bkg}$',
    'c' : r'$c$',
    'r' : r'$r$',
    'Aqc' : r'$n_{\beta}^c$',
    'Aqs' : r'$n_{\beta}^s$',
    'Aq' : r'$n_{\beta}$',
    'Cq' : r'$c_{\beta}$',
    'AfbHC' : r'$a_{hc}$',
    'AfbHS' : r'$a_{hs}$',
    'AfbLC' : r'$a_{\ell c}$',
    'AfbLS' : r'$a_{\ell s}$'
}


def get_factors(q2min, q2max):
    if q2min == 1.1 and q2max == 19.0:
        if 'massless' in args.aS:
            print("Using massless factors")
            return {'wS': 0.1877,
                    'wA0': 0.4412,
                    'wApp': 0.3699}
        elif 'massive' in args.aS:
            print("Using massive factors")
            return {'wS': 0.1900,
                    'wA0': 0.4365,
                    'wApp': 0.3709}
    elif q2min == 0.06 and q2max == 19.0:
        if 'massless' in args.aS:
            print("Using massless factors")
            return {'wS': 0.1794,
                    'wA0': 0.4297,
                    'wApp': 0.3948}
        elif 'massive' in args.aS:
            print("Using massive factors")
            return {'wS': 0.1754,
                    'wA0': 0.4207,
                    'wApp': 0.3858}
    elif q2min == 0.1 and q2max == 19.0:
        if 'massless' in args.aS:
            print("Using massless factors")
            return {'wS': 0.1800,
                    'wA0': 0.4313,
                    'wApp': 0.3876}
        elif 'massive' in args.aS:
            print("Using massive factors")
            return {'wS': 0.1774,
                    'wA0': 0.4217,
                    'wApp': 0.3758}
    raise ValueError(f"Unknown q2 range: {q2min}-{q2max}")

nbins = 100

import yaml

with open(args.results) as f:
    results = yaml.load(f, Loader=yaml.FullLoader)

Nsig = results["Nsig"]["value"]

if not ("Aqc" in results):
    results["Aqc"] = {"value": 0}
if not ("Aqs" in results):
    results["Aqs"] = {"value": 0}

data = pd.read_hdf(args.input)

import uproot
def read_data(file):
    with uproot.open(file) as f:
        tree_names = f.keys()
        data = None
        for tree_name in tree_names:
            if data is None:
                data = f[tree_name].arrays(library="pd")
            else:
                data = pd.concat([data, f[tree_name].arrays(library="pd")], ignore_index=True)
        data["cosl"] = data["ctl"]
        data["cosh"] = data["ctk"]
        data['mKpi'] = data['mkpi']/1000.0
        data = data.query(f"(0.06<q2<9.0) or (11.0<q2<12.5) or (15.0<q2<19.0)")
        data = data.query(f"(mKpi>{args.mKpi[0]}) and (mKpi<{args.mKpi[1]})")
        data = data.query(f"(q2>{args.qsq[0]}) and (q2<{args.qsq[1]})")
    return data

dataA0 = read_data(args.a0)
dataA1 = read_data(args.a1)
dataAS = read_data(args.aS)
print(len(data), len(dataA0)/len(data), len(dataA1)/len(data), len(dataAS)/len(data))
truth = {'wS': dataAS, 'wA0': dataA0, 'wApp': dataA1}
if 'Abeta' in results.keys():
    factor = get_factors(args.qsq[0], args.qsq[1]) 
else:
    fs = (1-results["A0"]["value"]-results["App"]["value"]-results["Aqs"]["value"]-results["Aqc"]["value"])
    f1 = results["App"]["value"]
    f0 = results["A0"]["value"]
    factor = {'wS': fs, 'wA0': f0, 'wApp': f1}
print(factor)

for key, label, unit in zip(['mKpi', 'q2'], [r'$m(K\pi)$', r'$q^2$'], [r'GeV/$c^2$', r'GeV$^2/c^4$']):
    mi, ma = data[key].min(), data[key].max()

    fig = plt.figure()
    fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 0.5, 0.5, 0.5], 'hspace':0.0}, sharex=True, figsize=[fig.get_size_inches()[0],fig.get_size_inches()[0]])
    lists = zip([1,2,3,4,4,4],[r"$n^S_0=\beta^2(|{A'}_0^L|^2+|{A'}_0^R|^2)$",r'$n_0^P=\beta^2(|{A}_0^L|^2+|{A}_0^R|^2)$',r'$n_1^P=\beta^2(|{A}_\perp^L|^2+|{A}_\perp^R|^2+|{A}_\parallel^L|^2+|{A}_\parallel^R|^2)$',r'$n_{\beta}$',r'$n_c^q$',r'$n_s^q$'],['wS','wA0','wApp','wAq','wAqc','wAqs'],['gold','navy','dodgerblue','firebrick','firebrick','firebrick'],["xx","//","\\\\","..","..",".."])
    # lists = zip([1,2,3,4,4,4],[r"$n^S_0$",r'$n_0^P$',r'$n_1^P$',r'$n_{\beta}$',r'$n_c^q$',r'$n_s^q$'],['wS','wA0','wApp','wAq','wAqc','wAqs'],['gold','navy','dodgerblue','firebrick','firebrick','firebrick'],["xx","//","\\\\","..","..",".."])
    maximum = 0
    for i,n,w,c,m in lists:
        print(w)
        if w not in data.columns:
            continue
        Hs = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
        Hs.fill(data[key], weight=data[w])
        mplhep.histplot(Hs, histtype="errorbar", label=n, xerr=True, yerr=True, color=c, ax=ax[0], marker='.')
        maximum = max(maximum, Hs.values().max())
        if i<4:
            Ht = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False))
            Ht.fill(truth[w][key], weight=factor[w]*Nsig/len(truth[w]))
            mplhep.histplot(Ht, histtype="step", color=c, ax=ax[0])
            residuals = (Hs.values()-Ht.values())/np.sqrt(Hs.variances()+Ht.values())
            for r in range(len(residuals)):
                color='black'
                ax[i].fill_between([Hs.axes[0].edges[r],Hs.axes[0].edges[r+1]], [0,0], [residuals[r],residuals[r]], color=c, linewidth=0)
            # ax[i].fill_between([mi,ma], [-3.0,-3.0], [3.0,3.0], color=c, alpha=0.3, linewidth=0, zorder=-10)
            ax[i].set_ylim(-3.5,3.5)
            # ax[i].set_ylabel(fr"Pull {n}",ha="right",y=1)
            ax[i].axhline(y=0, color='black', linewidth=1)
            ax[i].axhline(y=2, color='black', linestyle='dotted', linewidth=1)
            ax[i].axhline(y=-2, color='black', linestyle='dotted', linewidth=1)
            ax[i].set_yticks([-2,0,2])
            ax[i].set_yticklabels(["-2","0","2"])
    ax[3].set_xlabel(label+f" [{unit}]",ha="right",x=1)
    # ax[0].set_ylim(,)
    ax[0].axhline(y=0, color='black', linestyle='dotted', linewidth=1)
    if key == 'mKpi':
        ax[0].legend(handletextpad=0.2, fontsize=22, loc='upper right', frameon=False,)
    ax[0].set_ylabel("Value of the combination [a.u.]",ha="right",y=1)
    ax[0].set_xlim(mi,ma)
    ax[0].set_ylim(-maximum*0.05,maximum*1.1)
    ax[0].set_yticklabels([])
    plt.savefig(f"plots_wide/{args.name}_{key}_comparison.pdf")
    plt.close()

    fig = plt.figure()
    fig, ax = plt.subplots(5, 1, gridspec_kw={'height_ratios': [3, 0.5, 0.5, 0.5, 0.5], 'hspace':0.0}, sharex=True, figsize=[fig.get_size_inches()[0],fig.get_size_inches()[0]])

    HfA = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
    HfA.fill(data[key], weight=data['wApp']+4*data['wA0'])
    HfB = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
    HfB.fill(data[key], weight=data['wS']+data['wA0']*3)
    HfC = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
    HfC.fill(data[key], weight=data['wApp']-(4/3)*data['wS'])
    HfD = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
    HfD.fill(data[key], weight=data['wS']+data['wA0']-(1/2)*data['wApp'])
    if 'Abeta' in results.keys():
        HfB.fill(data[key], weight=-(1/2)*data['wAq'])
        HfC.fill(data[key], weight=(2/3)*data['wAq'])
        HfD.fill(data[key], weight=-(1/2)*data['wAq'])
    HtA = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
    HtA.fill(truth['wApp'][key], weight=factor['wApp']*Nsig/len(truth['wApp']))
    HtA.fill(truth['wA0'][key], weight=4*factor['wA0']*Nsig/len(truth['wA0']))
    HtB = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
    HtB.fill(truth['wS'][key], weight=factor['wS']*Nsig/len(truth['wS']))
    HtB.fill(truth['wA0'][key], weight=3*factor['wA0']*Nsig/len(truth['wA0']))
    HtC = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
    HtC.fill(truth['wS'][key], weight=-(4/3)*factor['wS']*Nsig/len(truth['wS']))
    HtC.fill(truth['wApp'][key], weight=factor['wApp']*Nsig/len(truth['wApp']))
    HtD = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
    HtD.fill(truth['wS'][key], weight=factor['wS']*Nsig/len(truth['wS']))
    HtD.fill(truth['wA0'][key], weight=factor['wA0']*Nsig/len(truth['wA0']))
    HtD.fill(truth['wApp'][key], weight=-(1/2)*factor['wApp']*Nsig/len(truth['wApp']))
    
    print("A:", HfA.values().sum(), HtA.values().sum())
    print("B:", HfB.values().sum(), HtB.values().sum())
    print("C:", HfC.values().sum(), HtC.values().sum())
    print("D:", HfD.values().sum(), HtD.values().sum())
    maximum = 0
    idx = 1
    for hf, ht, c, l in zip([HfA,HfB,HfC,HfD],[HtA,HtB,HtC,HtD],['dodgerblue','firebrick','gold','darkblue'],[r"$n_1^P+4n_0^P$",r"$n_0^S+3n_0^P$",r"$|n_1^P-(4/3)n_0^S|$",r"$n_0^S+n_0^P-(1/2)n_1^P$"]):
        hf.values()[hf.values() < 0] = -hf.values()[hf.values() < 0]
        ht.values()[ht.values() < 0] = -ht.values()[ht.values() < 0]
        mplhep.histplot(hf, histtype="errorbar", label=l, xerr=True, yerr=True, color=c, marker='.', ax=ax[0])
        mplhep.histplot(ht, histtype="step", color=c, ax=ax[0])
        maximum = max(maximum, hf.values().max())
        residuals = (hf.values()-ht.values())/np.sqrt(hf.variances()+ht.values())
        for r in range(len(residuals)):
            ax[idx].fill_between([hf.axes[0].edges[r],hf.axes[0].edges[r+1]], [0,0], [residuals[r],residuals[r]], color=c, linewidth=0)
        ax[idx].set_ylim(-4.5,4.5)
        ax[idx].axhline(y=0, color='black', linewidth=1)
        ax[idx].axhline(y=3, color='black', linestyle='dotted', linewidth=1)
        ax[idx].axhline(y=-3, color='black', linestyle='dotted', linewidth=1)
        ax[idx].set_yticks([-3,0,3])
        ax[idx].set_yticklabels(["-3","0","3"])
        idx += 1
    ax[idx-1].set_xlabel(label+f" [{unit}]",ha="right",x=1)
    # ax[0].set_ylim(,)
    ax[0].axhline(y=0, color='black', linestyle='dotted', linewidth=1)
    if key == 'mKpi':
        ax[0].legend(handletextpad=0.2, fontsize=22, loc='upper right', frameon=False,)
    ax[0].set_ylabel("Value of the combination [a.u.]",ha="right",y=1)
    ax[0].set_xlim(mi,ma)
    ax[0].set_ylim(-maximum*0.05,maximum*1.1)
    ax[0].set_yticklabels([])
    plt.savefig(f"plots_wide/{args.name}_{key}_comparison_check.pdf")
    plt.close()
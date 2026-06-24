import numpy as np


def set_K(polynomial, truth, i, j):
    if polynomial == "legendre":
        if i == 0 and j == 0:
            return 0.25
        if i == 0 and j == 1:
            return (1/2)*np.sqrt(3)*truth["alpha"]
        if i == 0 and j == 2:
            return (1/4)*(2*truth["A0"]-truth["App"])
        if i == 1 and j == 0:
            return (3/4)*truth["beta"]
        if i == 1 and j == 1:
            return 0
        if i == 1 and j == 2:
            return -(3/4)*truth["beta"]
        if i == 2 and j == 0:
            return (1/8)*(-2*truth["A0"]+truth["App"]-2*truth["AS"])
        if i == 2 and j == 1:
            return -(1/2)*np.sqrt(3)*truth["alpha"]
        if i == 2 and j == 2:
            return -(1/8)*(4*truth["A0"]+truth["App"])
    elif polynomial == "chebyshev":
        if i == 0 and j == 0:
            return (3/128)*(12*truth["A0"]+9*truth["App"]+8*truth["AS"])
        if i == 0 and j == 1:
            return (3/8)*np.sqrt(3)*truth["alpha"]
        if i == 0 and j == 2:
            return (9/128)*(4*truth["A0"]-3*truth["App"])
        if i == 1 and j == 0:
            return (9/16)*truth["beta"]
        if i == 1 and j == 1:
            return 0
        if i == 1 and j == 2:
            return -(9/16)*truth["beta"]
        if i == 2 and j == 0:
            return -(3/128)*(12*truth["A0"]-3*truth["App"]+8*truth["AS"])
        if i == 2 and j == 1:
            return -(3/8)*np.sqrt(3)*truth["alpha"]
        if i == 2 and j == 2:
            return -(9/128)*(4*truth["A0"]+truth["App"])


def set_A(polynomial, App, AS, alpha, beta, coefficients):
    if polynomial == "legendre":
        _App = -(8/3)*coefficients['K02'].value().numpy() - (8/3)*coefficients['K22'].value().numpy()
        _AS = -2*coefficients['K02'].value().numpy() - 4*coefficients['K20'].value().numpy()
        # _A0 = 1-_App-_AS
        _alpha = -2*coefficients['K21'].value().numpy()/(np.sqrt(3))
        _beta = (4/3)*coefficients['K10'].value().numpy()
        App.set_value(_App)
        AS.set_value(_AS)
        alpha.set_value(_alpha)
        beta.set_value(_beta)
    elif polynomial == "chebyshev":
        _AS = -((8*coefficients['K02'].value().numpy())/3)-(16*coefficients['K20'].value().numpy())/3+(8*coefficients['K22'].value().numpy())/3
        _App = - ((32*coefficients['K02'].value().numpy())/9)-(32*coefficients['K22'].value().numpy())/9
        # _A0 = 1-_App-_AS
        _alpha = -(8*coefficients['K21'].value().numpy())/(3*np.sqrt(3))
        _beta = (16/9)*coefficients['K10'].value().numpy()
        App.set_value(_App)
        AS.set_value(_AS)
        alpha.set_value(_alpha)
        beta.set_value(_beta)


def makedirs(polynomial, name):
    import os
    if not os.path.exists("plots"):
        os.mkdir("plots")
    if not os.path.exists("plots"):
        os.mkdir("plots")
    if not os.path.exists(f"plots/{polynomial}"):
        os.mkdir(f"plots/{polynomial}")
    if not os.path.exists(f"plots/{polynomial}/{name}"):
        os.mkdir(f"plots/{polynomial}/{name}")
    if not os.path.exists(f"plots/{polynomial}/{name}/results"):
        os.mkdir(f"plots/{polynomial}/{name}/results")
    if not os.path.exists("sweights"):
        os.mkdir("sweights")
    if not os.path.exists("sweights"):
        os.mkdir("sweights")
    if not os.path.exists(f"sweights/{polynomial}"):
        os.mkdir(f"sweights/{polynomial}")
    if not os.path.exists(f"sweights/{polynomial}/{name}"):
        os.mkdir(f"sweights/{polynomial}/{name}")


def parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--polynomial', type=str, help="Which basis do you want to use for the fit? Options: standard, legendre, chebyshev", default="standard")
    parser.add_argument('--settings', type=str, help="True values")
    parser.add_argument('--data', type=str, help="Data")
    parser.add_argument('--toy', action=argparse.BooleanOptionalAction, help="Generate toy data? Default: --toy means true and --no-toy means false")
    parser.add_argument('--mKpi', type=float, help="mKpi", nargs='+', default=[0.65,1.5])
    parser.add_argument('--fix_to_zero', type=str, help="coefficients fixed to zero", default=[], nargs='+')
    parser.add_argument('--qsq', type=float, help="Considered q^2 range", default=[], nargs='+')
    parser.add_argument('--binned', type=float, help="Considered q^2 bin", default=[], nargs='+')
    parser.add_argument('--nsig', type=int, help="Number of signal events", default=45000)
    parser.add_argument('--binnedfit', action=argparse.BooleanOptionalAction, help="Do a binned fit?", default=False)
    parser.add_argument('--fix_to_truth', type=str, help="coefficients fixed to their truth", default=[], nargs='+')
    parser.add_argument('--fix_to_value', type=str, help="coefficients fixed to a value", default=[], nargs='+')
    parser.add_argument('--constrain', type=str, help="constrain coefficients", default=[], nargs='+')  
    parser.add_argument("--background",type=str,default="../genbkg/background_toy.root",help="Input ROOT file containing the background toy tree",)
    parser.add_argument("--background-tree",type=str,default="background",help="Background tree name",)
    return parser.parse_args()

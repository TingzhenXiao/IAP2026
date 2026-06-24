import zfit
import argparse
import json
import uproot

parser = argparse.ArgumentParser(description="Generate background events.")
parser.add_argument("--n", type=int, default=1000000, help="Number of background events to generate")
parser.add_argument("--output", type=str, default="background_events.root", help="Output ROOT file name")
parser.add_argument("--params", type=str, default="background_params.json", help="JSON file with background parameters")
args = parser.parse_args()

# Load parameters from JSON file
with open(args.params, 'r') as f:
    params_dict = json.load(f)["parameters"]

# Define the ranges of the variables
qsq = zfit.Space('q2', limits=(1.1, 7.0))
mkpi = zfit.Space('mKpi', limits=(0.65, 1.5))
bmass = zfit.Space('B_mass', limits=(5.170, 5.500))
cosh = zfit.Space('cosThetaK', limits=(-1.0, 1.0))
cosl = zfit.Space('cosThetaL', limits=(-1.0, 1.0))

space = qsq * mkpi * bmass * cosh * cosl

# Create the background PDF components
pdf_qsq = zfit.pdf.Uniform(qsq.limits[0], qsq.limits[1], obs=qsq)
pdf_mkpi = zfit.pdf.Uniform(mkpi.limits[0], mkpi.limits[1], obs=mkpi)

lambda_bmass = zfit.Parameter("lambda_bmass", params_dict["lambda_bmass"])
pdf_bmass = zfit.pdf.Exponential(obs=bmass, lambda_=lambda_bmass)

a1_cosh = zfit.Parameter("a1_cosh", params_dict["a1_cosh"])
a2_cosh = zfit.Parameter("a2_cosh", params_dict["a2_cosh"])
pdf_cosh = zfit.pdf.Legendre(obs=cosh, coeffs=[a1_cosh, a2_cosh])
a1_cosl = zfit.Parameter("a1_cosl", params_dict["a1_cosl"])
a2_cosl = zfit.Parameter("a2_cosl", params_dict["a2_cosl"])
pdf_cosl = zfit.pdf.Legendre(obs=cosl, coeffs=[a1_cosl, a2_cosl])

# Combine into the 5D background PDF
pdf_bkg = zfit.pdf.ProductPDF([pdf_qsq, pdf_mkpi, pdf_bmass, pdf_cosh, pdf_cosl], obs=space)

# Generate background events
n_events = args.n
data_bkg = pdf_bkg.sample(n_events)


# Save to ROOT file
output_file = args.output
with uproot.recreate(output_file) as root_file:
    root_file["background"] = {
        "q2": data_bkg['q2'].numpy(),
        "mKpi": data_bkg['mKpi'].numpy(),
        "B_mass": data_bkg['B_mass'].numpy(),
        "cosThetaK": data_bkg['cosThetaK'].numpy(),
        "cosThetaL": data_bkg['cosThetaL'].numpy(),
    }
print(f"Generated {n_events} background events and saved to {output_file}.")
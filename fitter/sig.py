import numpy as np  # Numerical library
import yaml  # For reading YAML files
import uproot  # For reading ROOT files
import matplotlib.pyplot as plt  # Plotting library
import zfit  # Fitting library
import hist  # Histogram library
from hepstats.splot import compute_sweights  # For sWeights computation
import json  # For reading JSON files
from myconstants import *
import tools  # Some helpful functions
import mypdfs  # Custom pdfs
import angularfunctions as af  # Angular functions
import os
import pandas as pd
from sweights import Cows


df_sig = pd.read_hdf(
    "/home/submit/xiaot425/IAP2026/efficiency/efficiency_applied_output/signal_with_efficiency.h5",
    key="data",
).copy()

print(df_sig.head())
print(df_sig.columns)
print(df_sig.info())
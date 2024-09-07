# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ----
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# Personal libs ---- 
from networks.MAC.mac49 import MAC49
from modules.hierarmerge import Hierarchy
from modules.flatmap import FLATMAP
from modules.colregion import colregion
from various.data_transformations import maps
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
structure = "FLNe"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0.0
opt_score = ["_S"]
save_data = T
version = "49d106"
__nodes__ = 49
__inj__ = 49
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC49(
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    version = version,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = __inj__,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    b = bias
  )
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    NET.A, nlog10, lookup, prob, b=bias
  )
  ## Hierarchy object!! ----
  H = Hierarchy(
    NET, NET.A, R, NET.D,
    __nodes__, linkage, mode, lookup=lookup, index=index
  )

  D12 = H.target_sim_matrix
  D12[D12 != 0] = -2 * np.log(D12[D12 != 0])

  D12p = H.source_sim_matrix
  D12p[D12p != 0] = -2 * np.log(D12p[D12p != 0])

  SLN = NET.SLN[:NET.nodes, :][:, :NET.nodes]

  A = NET.A[:NET.nodes, :][:, :NET.nodes]
  non_zero = A > 0

  flne_label = r"$\log_{10} FLNe$"
  d12_label = r"$D_{1/2}^{-}$"
  tract_label = "Tractography"

  data = pd.DataFrame(
     {
        tract_label : list(NET.D[:NET.nodes, :][:, :NET.nodes][non_zero]) * 1,
        flne_label : list(np.log10(A[non_zero])) * 1,
        "SLN" : list(SLN[non_zero]) * 1,
        d12_label : D12[non_zero]
        # "class" : ["-"] * np.sum(non_zero) + ["+"] * np.sum(non_zero)
     }
  )

  sns.set_style("ticks")
  # sns.set_context("talk")

  cmp = sns.color_palette("deep")

  def regression(x, y):
     import statsmodels.api as sm
     from scipy.stats import zscore

     x2 = np.power(x, 2)
     x2 = zscore(x2)
     x2 = sm.add_constant(x2)

     model = sm.OLS(zscore(y), x2).fit()
     print(model.summary())
     return model.fvalue, model.f_pvalue
  
  data_ftest = pd.DataFrame()

  for ylabel in [d12_label, flne_label, tract_label]:
    f, fp = regression(data["SLN"], data[ylabel])
    data_ftest = pd.concat(
      [
        data_ftest,
        pd.DataFrame(
          {
            "Dependent variable" : [ylabel],
            "F-statistic" : [f],
            "P-value" : [fp]
          }
        )
      ], ignore_index=True
    )

  sns.scatterplot(
    data=data_ftest,
    x="P-value",
    y="F-statistic",
    hue="Dependent variable",
    s=100
  )

  sns.despine()

  ax=plt.gca()
  fig=plt.gcf()
  fig.set_figheight(3)
  fig.set_figwidth(4)
  fig.tight_layout()

  plt.savefig(
     f"{H.plot_path}/sln/models_ftest.svg",
     transparent=True
  )
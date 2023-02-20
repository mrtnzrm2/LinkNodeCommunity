# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
##
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
T = True
F = False
##
from networks.structure import MAC
from modules.nodhierarchy import NODH
from various.network_tools import *

def load_series():
  linkage = "single"
  nlog10 = T
  lookup = F
  cut = T
  mode = "ALPHA"
  nature = "original"
  distance = "MAP3D"
  feature = "TARGET_JACW"
  # Declare global variables DISTBASE ----
  total_nodes = 106
  __inj__ = 57
  __nodes__ = 57
  __version__ = 220830  
  __model__ = "CONSTM"
  __bin__ = 12
  ## Very specific!!! Be careful ----
  if nature == "original":
    __ex_name__ = f"{total_nodes}_{__inj__}"
  else:
    __ex_name__ = f"{total_nodes}_{total_nodes}_{__inj__}"
  if nlog10: __ex_name__ = f"{__ex_name__}_l10"
  if lookup: __ex_name__ = f"{__ex_name__}_lup"
  if cut: __ex_name__ = f"{__ex_name__}_cut"
  data = read_class(
      "../pickle/RAN/distbase/MAC/{}/FLN/{}/{}/BIN_{}/{}/{}/".format(
        __version__,
        distance,
        __model__,
        str(__bin__),
        __ex_name__,
        mode
      ),
      "series_{}_{}".format(linkage, feature)
    )
  return data

def sample_weight_NH(W, cdf, bin_midpoints,  nodes):
  NH_vector = np.zeros(nodes)
  for i in np.arange(nodes):
    deg = (W[:, i] != 0) & (~np.isnan(W[:, i]))
    deg = np.sum(deg)
    values = np.random.rand(deg)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]
    NH_vector[i] = np.max(random_from_cdf) - np.min(random_from_cdf)
  return NH_vector
    
if __name__ == "__main__":
  linkage = "single"
  mode = "ALPHA"
  version = 220830
  nature = "original"
  distance = "MAP3D"
  feature = "NH"
  imputation_model = ""
  direct = "TARGET_FULL"
  nodes=57
  inj=57
  nlog10=T
  lookup=F
  ## Create monkey ----
  NET = MAC(
    linkage, mode, nlog10=nlog10, lookup=lookup,
    version=version, nature=nature, distance=distance,
    dir=direct, feature=feature,
    inj=inj
  )
  W = NET.A.copy()
  W[W != 0] = np.log10(W[W != 0])
  W[W != 0] = W[W != 0] + np.ceil(np.abs(np.nanmin(W[W != 0])))
  HW, bins = np.histogram(W[W !=0].ravel(), bins=500)
  bin_midpoints = bins[:-1] + np.diff(bins)/2
  cdf = np.cumsum(HW)
  cdf = cdf / cdf[-1]
  ##
  data = load_series()
  data = data.data_homoegeity
  data.data.loc[data.data == "1"] = "data"
  data.data.loc[data.data == "0"] = "edr"
  for i in np.arange(1000):
    data = pd.concat(
      [
        data,
        pd.DataFrame(
          {
            "area" : NET.struct_labels[:nodes],
            "TNH" : sample_weight_NH(W, cdf, bin_midpoints, nodes),
            "data" : ["sample"] * nodes
          }
        )
      ],
      ignore_index=True
    )
  data["dataset"] = data.data
  ##
  sns.set_theme()
  nh_data = data.TNH.loc[data.dataset == "data"]
  order = np.argsort(nh_data)
  fig, ax = plt.subplots(1, 1, figsize=(10, 6))
  sns.stripplot(
    data=data.loc[data.dataset != "data"],
    x= "area",
    y="TNH",
    hue="dataset",
    s=3,
    alpha=0.5,
    order=NET.struct_labels[:nodes][order],
    ax=ax
  )
  sns.stripplot(
    data=data.loc[data.dataset == "data"],
    x= "area",
    y="TNH",
    color='black',
    marker="$\circ$",
    s=5,
    order=NET.struct_labels[:nodes][order],
    ax=ax
  )
  plt.xticks(rotation=90)
  fig.tight_layout()
  # Arrange path ----
  plot_path = os.path.join(NET.plot_path, "NT") 
  # Crate path ----
  from pathlib import Path
  Path(
    plot_path
  ).mkdir(exist_ok=True, parents=True)
  # Save plot ----
  plt.savefig(
    os.path.join(
      plot_path, "NT_average.png"
    ),
    dpi = 300
  )


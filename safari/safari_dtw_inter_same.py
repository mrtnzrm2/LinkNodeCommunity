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
from various.network_tools import adj2df

if __name__ == "__main__":
  linkage = "single"
  mode = "ALPHA"
  version = 220830
  nature = "original"
  distance = "MAP3D"
  imputation_model = ""
  direct = "TARGET_FULL"
  nodes=57
  inj=57
  nlog10=T
  lookup=F
  
  ## Creat monkey ----
  feature1 = "dtw_both"
  NETB = MAC(
    linkage, mode, nlog10=nlog10, lookup=F,
    version=version, nature=nature, distance=distance,
    dir=direct, feature=feature1,
    inj=inj
  )
  NETB.A = NETB.A[:nodes, :nodes]
  HB = NODH(NETB, nodes, nlog10=nlog10, lookup=F)
  ##
  feature2 = "dtw_target"
  NETW = MAC(
    linkage, mode, nlog10=nlog10, lookup=lookup,
    version=version, nature=nature, distance=distance,
    dir=direct, feature=feature2,
    inj=inj
  )
  HW = NODH(NETW, nodes, nlog10=nlog10, lookup=lookup)
  ##
  dBOTH = HB.feature_dist
  dTARGET = adj2df(HW.feature_dist)
  ##
  dTARGET = dTARGET.loc[dTARGET.source > dTARGET.target]
  ##
  data = pd.DataFrame(
    {
      "DTW" : np.hstack(
        [dBOTH, dTARGET.weight * 57 / 106]
      ),
      "class" : ["intra"] * len(dBOTH) + ["t_inter"] * len(dTARGET)
    }
  )
  ##
  sns.set_theme()
  fig, ax = plt.subplots(1, 1)
  sns.histplot(
    data=data,
    x="DTW",
    hue="class",
    stat="probability",
    common_norm=False,
    common_bins=False,
    ax=ax
  )
  fig.tight_layout()
  # Arrange path ----
  plot_path = os.path.join(NETW.plot_path, "COMPARE")
  # Crate path ----
  from pathlib import Path
  Path(
    plot_path
  ).mkdir(exist_ok=True, parents=True)
  # Save plot ----
  plt.savefig(
    os.path.join(
      plot_path, "hist_{}_{}.png".format(feature1, feature2)
    ),
    dpi = 300
  )
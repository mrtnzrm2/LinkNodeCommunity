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

if __name__ == "__main__":
  linkage = "single"
  mode = "ALPHA"
  version = 220830
  nature = "original"
  distance = "MAP3D"
  imputation_model = ""
  direct = "SOURCE_FULL"
  nodes=106
  inj=57
  nlog10=T
  lookup=F
  
  ## Creat monkey ----
  feature1 = "paper_source_b"
  NETB = MAC(
    linkage, mode, nlog10=nlog10, lookup=F,
    version=version, nature=nature, distance=distance,
    dir=direct, feature=feature1,
    inj=inj
  )
  HB = NODH(NETB, nodes, nlog10=nlog10, lookup=F)
  ##
  feature2 = "jacw_source"
  NETW = MAC(
    linkage, mode, nlog10=nlog10, lookup=lookup,
    version=version, nature=nature, distance=distance,
    dir=direct, feature=feature2,
    inj=inj
  )
  HW = NODH(NETW, nodes, nlog10=nlog10, lookup=lookup)
  ##
  dBDIST = adj2df(HB.feature_dist)
  dJACWDIST = adj2df(HW.feature_dist)
  ##
  dBDIST = dBDIST.loc[dBDIST.source > dBDIST.target]
  dJACWDIST = dJACWDIST.loc[dJACWDIST.source > dJACWDIST.target]
  ##
  data = pd.DataFrame(
    {
      "bdist" : dBDIST.weight,
      "jacw_dist": dJACWDIST.weight
    }
  )
  ##
  sns.set_theme()
  fig, ax = plt.subplots(1, 1)
  sns.scatterplot(
    data=data,
    x="bdist",
    y="jacw_dist",
    s=3,
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
      plot_path, "{}_{}.png".format(feature1, feature2)
    ),
    dpi = 300
  )
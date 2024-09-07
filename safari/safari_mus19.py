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
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import numpy.typing as npt

from typing import Union
# Personal libs ---- 
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from modules.discovery import discovery_channel
from various.data_transformations import maps
from networks.structure import STR
from various.network_tools import *

def get_ODR_structure(csv_path) -> Union[pd.DataFrame, npt.ArrayLike]:
  file = pd.read_csv(os.path.join(csv_path, "SourceData_Fig3.csv"))
  pathway_var = file["pathway"]
  pathway_var = [(s.split(" to ")[0], s.split(" to ")[1]) for s in pathway_var]
  file["SOURCE_IND"] = [s for s, t in pathway_var]
  file["TARGET_IND"] = [t for s, t in pathway_var] 
  file["SOURCE_IND"] = np.char.lower([s[1:] for s in file["SOURCE_IND"]])
  file["TARGET_IND"] = np.char.lower([s[:-1] for s in file["TARGET_IND"]])

  # print(np.unique(file["SOURCE_IND"]))


  labels_order_paper_DSouzza = np.char.lower(
    ["V1", "LM", "RL", "AL", "A", "PM", "P", "LI", "AM", "POR"]
  )

  file["SOURCE_IND"] = pd.Categorical(file["SOURCE_IND"], labels_order_paper_DSouzza)
  file["TARGET_IND"] = pd.Categorical(file["TARGET_IND"], labels_order_paper_DSouzza)

  return file.pivot("SOURCE_IND", "TARGET_IND", "mean"), labels_order_paper_DSouzza


# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
subject = "MUS"
structure = "FLNe"
mode = "ZERO"
nature = "original"
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0.
alpha = 0.
discovery = "discovery_7"
opt_score = ["_S"]
sln = F

__nodes__ = 7
__inj__ = f"{__nodes__}"
version = f"{__nodes__}"+"d"+"47"
distance = "MAP3D"
save_data = T


# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = STR[f"{subject}{__inj__}"](
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    version = version,
    nature = nature,
    distance = distance,
    inj = __inj__,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    b = bias,
    alpha = alpha,
    discovery = discovery
  )

  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    NET.A, nlog10, lookup, prob, b=bias
  )

  H = Hierarchy(
    NET, NET.A, R, NET.D,
     __nodes__, linkage, mode, lookup=lookup, index=index
  )
  
  labels = NET.struct_labels[:NET.nodes]
  
  SZZ, labels_szz = get_ODR_structure(NET.csv_path)
  labels_szz = list(labels_szz)
  TSIM = pd.DataFrame(H.target_sim_matrix, columns=labels, index=labels)

  labels_szz2 = [l for l in labels if l in labels_szz]

  SZZ = SZZ[labels_szz2].reindex(labels_szz2)
  TSIM = TSIM[labels_szz2].reindex(labels_szz2)
  # print(TSIM[labels_szz2].reindex(labels_szz2))
  # print(SZZ[labels_szz2].reindex(labels_szz2))
  
  ord = SZZ.to_numpy().ravel()
  
  diss = TSIM.to_numpy().ravel()
  diss = -2*np.log(diss)

  isinf = diss == np.Inf

  ord = ord[~isinf]
  diss = diss[~isinf]

  print(np.mean(diss[np.isnan(ord)]))

  sns.regplot(
    x=ord, y=diss,
    lowess=T
  )

  ax = plt.gca()

  ax.set_xlabel("ORD")
  ax.set_ylabel(r"$D^{in}_{1/2}$")

  plt.savefig(f"{NET.plot_path}/sln/information_ord.png", dpi=300)
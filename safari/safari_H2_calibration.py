# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libs ---- 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# from modules.sign.hierarmerge import Hierarchy
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from modules.discovery import discovery_channel
from various.data_transformations import *
from networks.structure import STR
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
subject_id = None
subject = "MAC"
structure = "FLNe"
mode = "ZERO"
distance = "tracto16"
nature = "original"
# imputation_method = "RF2"
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0.
alpha = 0.
discovery = "discovery_7"
opt_score = ["_S"]
save_data = T
__nodes__ = 49
__inj__ = f"{__nodes__}"
version = "220617"
mapping = "trivial" # multiplexed_colnormalized column_normalized_mapping
architecture = "all"
opt_score = ["_S"]
save_data = T

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
R, lookup, _ = trivial_mapping(NET.A)
# R, lookup, _ = column_normalized_mapping(NET.SupraC)
# R, lookup, _ = column_normalized_mapping(NET.InfraC)
# R, lookup, _ = multiplexed_colnormalized_mapping(0, NET.A, np.zeros(NET.A.shape))
H = Hierarchy(
  NET, NET.A, R, NET.D,
  NET.nodes, linkage, mode, lookup=lookup,
)

id_v1pclf = np.where(NET.struct_labels == "v1pclf")[0][0]
id_10= np.where(NET.struct_labels == "10")[0][0]

def sim_H2(u, v, ui, vj):
  n = u.shape[0]
  s = 0

  nu = np.sum(u)
  nv = np.sum(v)

  pu = u / nu
  pv = v / nv

  pu = np.sqrt(pu)
  pv = np.sqrt(pv)

  for i in np.arange(n):
    if i == ui or i == vj: continue
    s += np.power(pu[i] - pv[i], 2.)

  s += np.power(pu[vj] - pv[ui], 2.)
  s += np.power(pu[ui] - pv[vj], 2.)

  return 1 - s/2

def D12(u, v, ui, vj):
  nu = np.sum(u)
  nv = np.sum(v)

  pu = u / nu
  pv = v / nv

  pueff = np.array([p for i, p in enumerate(pu) if i != ui and i != vj])
  pueff = np.hstack([pueff, [pu[ui], pu[vj]]])
  pveff = np.array([p for i, p in enumerate(pv) if i != ui and i != vj])
  pveff = np.hstack([pveff, [pv[vj], pv[ui]]])

  pueff = np.sqrt(pueff)
  pveff = np.sqrt(pveff)

  pp = pueff * pveff
  max_pp = np.max(pp)
  if max_pp == 0: return np.Inf
  else:
    pp = pp / max_pp
    return -2 * (np.log(np.sum(pp)) + np.log(max_pp))

  # pu = np.sqrt(pu)
  # pv = np.sqrt(pv)

  # for i in np.arange(n):
  #   if i == ui or i == vj: continue
  #   s += np.power(pu[i] - pv[i], 2.)

  # s += np.power(pu[vj] - pv[ui], 2.)
  # s += np.power(pu[ui] - pv[vj], 2.)

  # return 1 - s/2

source_simh2 = np.zeros((NET.nodes, NET.nodes))

for i in np.arange(NET.nodes):
  for j in np.arange(i+1, NET.nodes):
    source_simh2[i, j] = D12(NET.A[i, :], NET.A[j, :], i, j)
    source_simh2[j, i] = source_simh2[i, j]

# source_D12 = -2 * np.log(source_simh2)
source_D12 =  source_simh2
source_D12 = adj2df(source_D12)
source_D12 = source_D12.loc[source_D12.source < source_D12.target]
source_D12["tag"] = "py"

print(H.source_sim_matrix)
csource_D12 = -2 * np.log(H.source_sim_matrix)
csource_D12 = adj2df(csource_D12)
csource_D12 = csource_D12.loc[csource_D12.source < csource_D12.target]
csource_D12["tag"] = "c++"

F = pd.concat([csource_D12, source_D12])
F["source_area"] = NET.struct_labels[F["source"]]
F["target_area"] = NET.struct_labels[F["target"]]


# print(F.loc[(F["weight"]> 50) & (F["weight"] < np.Inf)])
# print(F.loc[(F["source_area"] == "v1pclf") & np.isin(F["target_area"], ["f3", "1", "10", "opro", "25"])])

# print(np.max(source_D12[source_D12 < np.Inf]))
# print(np.max(csource_D12[csource_D12 < np.Inf]))

d = pd.DataFrame(
  {
    "py" : source_D12["weight"],
    "c++" : csource_D12["weight"]
  }
)
print(d)

# not_inf = (source_D12["weight"] < np.Inf) & (csource_D12["weight"] < np.Inf)
# v = np.abs(source_D12["weight"].to_numpy() - csource_D12["weight"].to_numpy())

# v = {"val" : v}

# sns.histplot(
#   data=v,
#   x="val"
# )

sns.scatterplot(
  data=d,
  x="py",
  y="c++"
)

plt.show()
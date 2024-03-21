# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# STL ----
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set_theme()
import pandas as pd
import os
import plotly.express as px

from networks.HCP.HCP import HCP
from various.network_tools import read_class, skim_partition

linkage_method = "single"
nlog10 = F
lookup = F
prob = F
cut = F
structure = "Cor"
mode = "ZERO"
nature = "original"
topology = "MIX"
mapping = "signed_trivial"
index  = "Hellinger2"
discovery = "discovery_7"
architecture = "all"
opt_score = ["_S"]
undirected_network = F
undirected_merde = 1
nodetimeseries = "50"
save_data = F

NET = HCP(
    linkage_method, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    nature = nature,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    nodetimeseries = nodetimeseries,
    discovery = discovery,
    undirected = undirected_merde,
    architecture = architecture
  )

H = read_class(
  NET.pickle_path,
  "hanalysis"
)

# Hierarchical association ----
rlabels = cut_tree(H.Z, n_clusters=H.kr["R"].loc[H.kr["score"] == opt_score[0]]).ravel()
rlabels = skim_partition(rlabels)

T = 4799
N = 50
S= 813
# # S= 10

nodetimeseries_path = f"../CSV/HCP/nodetimeseries_{N}"
list_nodetimeseries = os.listdir(nodetimeseries_path)

nodetimeseries_matrix = np.zeros((S, T, N))
for i, file in enumerate(list_nodetimeseries):
  try:
    nodetimeseries_matrix[i, :, :] = pd.read_table(f"../CSV/HCP/nodetimeseries_{N}/{file}", sep=" ").to_numpy()
  except:
    print(file)
    nodetimeseries_matrix[i, :, :] = np.nan
nodetimeseries_matrix = np.nanmean(nodetimeseries_matrix, axis=0).T

# Series per cluster
clusters = np.unique(rlabels)
clusters = np.array([i for i in clusters if i != -1])

# clusters = {
#   "0" : [17, 36, 22, 32],
#   "1" : [45, 46],
#   "2" : [44, 41, 23, 40],
#   "3" : [
#     96, 98,  82, 49, 80,
#     # 62, 56, 90, 71, 59, 51, 75, 92, 74, 91, 64
#   ],
#   "4" : [0, 18, ],
#   "5" : [
#     68, 35, 63, 99, 93, 84, 86,
#     # 42, 83, 73, 60, 81, 61, 97, 94, 79
#   ]
# }

data = pd.DataFrame()

for cls in clusters:
  nodes = np.where(rlabels == cls)[0]
  for n in nodes:
    data = pd.concat(
      [
        data,
        pd.DataFrame(
          {
            "clusters" : [cls] * T,
            "nodetimeserie" : [n] * T,
            "strength" : nodetimeseries_matrix[n, :].ravel(),
            "t" : np.arange(T)
          }
        )
      ],
      ignore_index=T
    )

fig = px.line(
  data,
  x="t", y="strength",
  color="nodetimeserie",
  facet_row="clusters",
  template="plotly_dark",
  # height=1500
)

fig.show()

# #  Arrange path ----
# plot_path = os.path.join(
#   data_.plot_path, "nodetimeseries"
# )
# # Crate path ----
# Path(
#   plot_path
# ).mkdir(exist_ok=True, parents=True)
# # Save plot ----
# # fig.write_html(f"{plot_path}/semiautomatic.html")

# from plotly.io import write_json
# write_json(fig, f"{plot_path}/semiautomatic.json")
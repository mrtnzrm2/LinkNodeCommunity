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
from scipy.cluster.hierarchy import cut_tree
# Personal libs ---- 
from networks.MAC.mac57 import MAC57
from modules.hierarmerge import Hierarchy
from modules.hierarentropy import Hierarchical_Entropy
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
version = "57d106"
__nodes__ = 57
__inj__ = 57
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC57(
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
  
  H = Hierarchy(
    NET, NET.A, R, NET.D,
    __nodes__, linkage, mode, lookup=lookup, index=index
  )
  ## Compute features ----
  H.BH_features_cpp_no_mu()
  ## Compute link entropy ----
  H.link_entropy_cpp("short", cut=cut)
  ## Compute lq arbre de merde ----
  H.la_abre_a_merde_cpp(H.BH[0])
  H.get_h21merge()

  N = NET.nodes
  A = NET.A
  R = A.copy()
  R[R > 0] = np.log10(R[R>0])
  D = NET.D

  rho = np.zeros((2, N-1))
  w = np.zeros((2, N-1))
  d = np.zeros(N-1)

  for i, r in enumerate(np.arange(N-1, 0, -1)):
    rho_ = []
    w_ = []
    d_ = []

    partition = cut_tree(H.Z, n_clusters=r).ravel()
    partition = skim_partition(partition)
    
    unique_clusters = np.unique(partition)
    unique_clusters = unique_clusters[unique_clusters != -1]
    for c in unique_clusters:
      nodes_c = np.where(partition == c)[0]
      rho_.append(adj2Den(A[nodes_c, :][:, nodes_c], dir=True))
      w_.append(adj2barW(R[nodes_c, :][:, nodes_c]))
      d_.append(adj2barW(D[nodes_c, :][:, nodes_c]))

    rho[0, i] = np.mean(rho_)
    w[0, i] = np.mean(w_)
    d[i] = np.mean(d_)

    if unique_clusters.shape[0] > 1:
      rho[1, i] = np.std(rho_)
      w[1, i] = np.std(w_)
    else:
      rho[1, i] = 0
      w[1, i] = 0

  R = np.arange(N-1, 0, -1)

  distance_label = r"$\left<D_{c}\right>$"

  data = pd.DataFrame(
    {
      "Estimate" : list(rho[0]) + list(w[0]),
      "Number of communities" : list(R) * 2,
      distance_label : list(d) * 2,
      "Feature" : [r"$\left<\rho_{c}\right>$"] * (N-1) + [r"$\left<\log_{10} FLNe_{c} \right>$"] * (N-1)
    }
  )

  sns.set_context("talk")
  sns.set_style("white")

  g = sns.FacetGrid(
    data=data,
    col="Feature",
    sharey=False
  )

  g.map_dataframe(
    sns.scatterplot,
    s=30,
    x="Number of communities",
    y="Estimate"
  )

  cmp = sns.color_palette("deep")

  K, RR, TH = get_best_kr_equivalence("_S", H)

  rho_mu = np.round(rho[0, (N-1)-RR], 3)[0]
  rho_sd = np.round(rho[1, (N-1)-RR], 4)[0]

  w_mu = np.round(w[0, (N-1)-RR], 2)[0]
  w_sd = np.round(w[1, (N-1)-RR], 3)[0]

  for i, ax in enumerate(g.axes.flatten()):
    
    if i == 0:
      ax.errorbar(R, rho[0], rho[1], color=cmp[1], linestyle="--", alpha=0.5)
      ax.text(0.6, 0.5, fr"${rho_mu}\pm {rho_sd}$", transform=ax.transAxes)
    elif i == 1:
      ax.errorbar(R, w[0], w[1], color=cmp[1], linestyle="--", alpha=0.5)
      ax.text(0.6, 0.5, rf"${w_mu}\pm {w_sd}$", transform=ax.transAxes)

    ax.axvline(RR, linestyle="--", color="r")

  fig = plt.gcf()
  fig.set_figwidth(12)
  fig.set_figheight(5)
  fig.tight_layout()

  # plt.show()
  path = join(NET.plot_path, "Features", "features_R.svg")
  plt.savefig(path)
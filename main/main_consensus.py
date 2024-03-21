# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# STL ----
from scipy.cluster.hierarchy import cut_tree
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use("dark_background")
# Personal libs ---- 
from modules.hierarmerge import Hierarchy
from modules.hierarentropy import Hierarchical_Entropy
from modules.colregion import colregion
from various.data_transformations import maps
from networks.structure import MAC
from networks.MAC.consensus import CONSENSUS
from various.network_tools import *
from plotting_modules.plotting_naked import Plot_NAKED
from various.clustering_tools import discover_overlap_nodes
import ctools as ct
# Declare global variables ----
MAXIT = 15
linkage_method = "single"
nlog10 = T 
lookup = F
prob = F
cut = F
structure = "FLN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = "RF"
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
discovery = "discovery_7"
SCORE = "_S"
version = "57d106i"
__nodes__ = 106
__inj__ = "57i"
save_data = T

if __name__ == "__main__":
  # Create structure ----
  cons = CONSENSUS(
    MAXIT, __nodes__, __nodes__, linkage_method,
    mode=mode, nlog10=nlog10, lookup=lookup, cut=cut,
    version=version, inj=__inj__, model=imputation_method,
    topology=topology, mapping=mapping, index=index, 
    discovery=discovery, imputation=imputation_method
  )
  if save_data:
    cons.create_pickle_directory()
    cons.create_plot_directory()

    for i in np.arange(MAXIT):
      NET = MAC[f"MAC{__inj__}"](
        linkage_method, mode,
        nlog10 = nlog10,
        structure = structure,
        lookup = lookup,
        version = version,
        nature = nature,
        model = imputation_method,
        distance = distance,
        imputation = imputation_method,
        iteration = i,
        inj = __inj__,
        topology = topology,
        index = index,
        mapping = mapping,
        cut = cut,
        discovery = discovery
      )
      # Transform data for analysis ----
      R, lookup, _ = maps[mapping](
        NET.A, nlog10, lookup, prob, b=0
      )
      cons.R[i, :, :] = R
      print("Compute Hierarchy")
      ## Hierarchy object!! ----
      H = Hierarchy(
        NET, NET.A, R, NET.D,
        __nodes__, linkage_method, mode, lookup=lookup, index=index
      )
      H.BH_features_cpp_no_mu()
      H.la_abre_a_merde_cpp(H.BH[0])
      # Set labels to network ----
      L = colregion(NET, labels_name=f"labels{__inj__}")
      H.set_colregion(L)
      H.delete_dist_matrix()

      # Get best K and R ----
      K, R, _ = get_best_kr_equivalence(SCORE, H)
      cons.best_levels[i, 0] = K[0]
      cons.best_levels[i, 1] = R[0]
      
      for z in np.arange(1, __nodes__):
        node_partition = cut_tree(H.Z, n_clusters=z).ravel().astype(int)
        communities = Counter(node_partition)
        communities = [k for k in communities.keys() if communities[k] > 1]
        for k in communities:
          nodes = np.where(node_partition == k)[0]
          x, y = np.meshgrid(nodes, nodes)
          x = x.ravel()
          y = y.ravel()
          keep = x != y
          x = x[keep]
          y = y[keep]
          cons.hierarchical_association[i, x, y] = H.Z[__nodes__ - 1 - z, 2]
    cons.labels = NET.struct_labels
    cons.colregion = L
    save_class(
      cons, cons.pickle_path,
      "canalysis", on=T
    )
  else:
    cons = read_class(
      cons.pickle_path,
      "canalysis"
    )
  # Consensus analysis ---
  D = np.mean(cons.hierarchical_association, axis=0)
  Z = linkage(squareform(D))
  rmean = int(np.mean(cons.best_levels[:, 1]))

  flne = np.nanmean(cons.R, axis=0)[:, :57]
  min_flne = np.nanmin(flne[flne > 0])

  flne = np.nanmean(cons.R, axis=0)
  np.fill_diagonal(flne, 0)
  flne[flne < min_flne] = 0

  plot_naked = Plot_NAKED(flne, D, cons.labels, cons.plot_path)

  s = Hierarchical_Entropy(Z, cons.nodes, cons.labels[:cons.nodes])
  s.Z2dict("short")
  treeh = s.zdict2newick(s.tree, weighted=T)


  plot_naked.core_dendrogram(Z, [rmean], leaf_font_size=8)
  plot_naked.plot_newick(treeh, cons.colregion, width=7, height=15, fontsize=10, rotation=180)

  # Measure Hellinger2 ---- 
  source_sim = np.zeros((cons.nodes, cons.nodes))
  target_sim = np.zeros((cons.nodes, cons.nodes))

  for i in np.arange(cons.nodes):
    for j in np.arange(i+1, cons.nodes):
      source_sim[i, j] = ct.Hellinger2(flne[i, :], flne[j, :], i, j)
      source_sim[j, i] = source_sim[i, j]
      target_sim[i, j] = ct.Hellinger2(flne[:, i], flne[:, j], i, j)
      target_sim[j, i] = target_sim[i, j]

  rlabels = cut_tree(Z, n_clusters=rmean).ravel()
  rlabels = skim_partition(rlabels)
  A = flne[:__nodes__, :].copy()
  A[A > 0] = -np.log(A[A > 0])
  np.fill_diagonal(A, 0)

  for direction in ["source", "target", "both"]:
    _, data_nocs, noc_sizes, rlabels2  = discover_overlap_nodes(flne, np.sqrt(1-source_sim), np.sqrt(1-target_sim), rlabels, cons.labels[:cons.nodes], direction=direction)
    print("\n\tAreas with predicted overlapping communities:\n",  data_nocs, "\n")
    cover = omega_index_format(rlabels2,  data_nocs, cons.labels[:cons.nodes])
    # Netowk ----
    plot_naked.plot_network_covers(
      A, rlabels2,
      data_nocs, noc_sizes, score=SCORE, direction=direction, cmap_name="hls", figsize=(8,8)
    )
  

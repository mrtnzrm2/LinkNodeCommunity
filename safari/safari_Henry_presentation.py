# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import networkx as nx
# Personal libraries ----
from networks.MAC.mac57 import MAC57
# from networks.MAC.mac29 import MAC29
from networks.toy import TOY
from networks.swapnet import SWAPNET
from various.network_tools import *
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from modules.discovery import discovery_channel
from various.data_transformations import maps
import ctools as ct


def plot_network(NET):
    print("Printing network space")
    sns.set_style("white")
    sns.despine(left=True, bottom=True)

    R = NET.A.copy()[:NET.nodes, :]
    R = -1. / np.log(R)
    edges = np.sum(R > 0)
    # from matplotlib.colors import to_hex
    # Generate graph ----
    G = nx.DiGraph(R)
    r_min = np.min(R[R>0])
    r_max = np.max(R)
    edge_color = [""] * edges
    for i, dat in enumerate(G.edges(data=True)):
      u, v, a = dat
      G[u][v]["kk_weight"] = - (a["weight"] - r_min) / (r_max - r_min) + r_max
      edge_color[i] = "gray"
    pos = nx.kamada_kawai_layout(G, weight="kk_weight")
    ang = 0
    ang = ang * np.pi/ 180
    rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
    pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
    labs = {k : lab for k, lab in zip(G.nodes, NET.labels)}
    
    # sns.set_context("talk")
    plt.figure(figsize=(8,8))
    nx.draw_networkx_labels(G, pos=pos, labels=labs, font_color="white")
    nx.draw_networkx_edges(G, pos=pos, edge_color=edge_color, alpha=0.7, arrowsize=20, connectionstyle="arc3,rad=-0.1", node_size=1000)
    nx.draw_networkx_nodes(G, pos=pos, node_color=sns.color_palette("deep")[0], node_size=1000, alpha=0.8)
    array_pos = np.array([list(pos[v]) for v in pos.keys()])
    plt.xlim(-0.1 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.1)
    plt.ylim(-0.1 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.1)
    plt.box(False)
    plt.savefig(
       "../Presentations/Henry/network_29x29.png", dpi=300
    )

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
mode = "ZERO"
topology = "MIX"
index = "Hellinger2"
discovery = "discovery_7"
opt_score = ["_D"]

properties = {
  "nlog10" : nlog10,
  "lookup" : lookup,
  "prob" : prob,
  "cut" : cut,
  "topology" : topology,
  "index" : index,
  "mode" : mode
}


if __name__ == "__main__":
    
    # A = np.array(
    #    [
    #       [0, 1, 1, 1],
    #       [1, 0, 1, 0],
    #       [0, 0, 0, 0],
    #       [0, 1, 0, 0]
    #    ]
    # )

    A = np.array(
       [
          [0,1,0,0,1,0,0,0],
          [1,0,0,1,1,0,0,0],
          [0,1,0,1,0,1,1,0],
          [1,0,0,0,0,0,1,0],
          [1,0,1,1,0,0,0,0],
          [0,0,0,1,0,0,1,1],
          [0,0,0,0,0,1,0,1],
          [0,0,0,0,0,1,1,0]
       ]
    )

    # A = np.array(
    #    [
    #       [0,1,1],
    #       [0,0,1],
    #       [0,1,0],
    #    ]
    # )

    # A = np.array(
    #    [
    #       [0,0,0,0,0,0,0,0,0,0],
    #       [1,0,1,0,0,0,0,0,0,0],
    #       [1,0,0,1,0,0,0,0,0,0],
    #       [0,0,0,0,1,1,0,0,0,0],
    #       [0,0,0,0,0,1,0,0,0,0],
    #       [0,0,0,0,1,0,0,0,1,0],
    #       [0,0,0,0,0,0,0,0,0,1],
    #       [0,0,0,0,0,0,0,0,0,0],
    #       [0,1,0,0,0,0,1,1,0,0],
    #       [0,0,0,0,0,0,0,0,0,0]
    #     ]
    # )

    # A = np.array(
    #    [
    #       [0, 1, 1, 0, 0, 1, 1],
    #       [0, 0, 1, 0, 0, 0, 0],
    #       [0, 1, 0, 0, 0, 0, 0],
    #       [1, 0, 0, 0, 1, 0, 0],
    #       [1, 0, 0, 1, 0, 0, 0],
    #       [0, 0, 0, 0, 0, 0, 1],
    #       [0, 0, 0, 0, 0, 1, 0]
    #    ]
    # )

    # A = np.array(
    #    [
    #       [0, 1, 1, 0, 0, 1, 1],
    #       [0, 0, 1, 0, 0, 0, 0],
    #       [0, 1, 0, 0, 0, 0, 0],
    #       [1, 0, 0, 0, 1, 0, 0],
    #       [1, 0, 0, 1, 0, 0, 0],
    #       [0, 0, 0, 0, 0, 0, 1],
    #       [0, 0, 0, 0, 0, 1, 0]
    #    ]
    # )

    # A = np.array(
    #    [
    #       [0,1,1,1],
    #       [1,0,1,0],
    #       [1,1,0,0],
    #       [0,0,1,0]
    #    ]
    # )

    # A = np.array(
    #    [
    #       [0, 1, 0],
    #       [1, 0, 1],
    #       [1, 1, 0]
    #    ]
    # )

    # A = np.array(
    #    [
    #       [0,1,1,0,0],
    #       [0,0,1,1,1],
    #       [0,0,0,1,0],
    #       [0,0,0,0,1],
    #       [1,0,0,0,0]
    #    ]
    # )

    # perm = np.random.permutation(A.shape[0])

    # A = A[perm, :][:, perm]

    print(A)

    original_labels = np.arange(1, A.shape[0]+1).astype(str)
    # original_labels = original_labels[perm]

    labels_dict = dict()
    for i in np.arange(A.shape[0]):
      labels_dict[i] = original_labels[i]

    nodes = A.shape[0]

    properties = {
      "version" : "Henry",
      "nlog10" : nlog10,
      "lookup" : lookup,
      "prob" : prob,
      "cut" : cut,
      "topology" : topology,
      "index" : index,
      "mode" : mode
    }

    NET = TOY(A, linkage, **properties)
    NET.create_plot_directory()
    NET.set_labels(original_labels)
    H = Hierarchy(
      NET, NET.A, NET.A, np.zeros(NET.A.shape),
      nodes, linkage, mode, index=index
    )
    ## Compute features ----
    H.BH_features_cpp_no_mu()
    ## Compute link entropy ----
    H.link_entropy_cpp("short", cut=cut)
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    ## Compute node entropy ----
    H.node_entropy_cpp("short", cut=cut)
    # Set labels to network ----
    L = colregion(NET, labels=NET.labels)
    L.get_regions()
    H.set_colregion(L)
    # Entropy ----
    S = Hierarchical_Entropy(H.H, H.leaves, list(range(H.leaves)))
    S.Z2dict("short")
    H.entropy = [
      H.node_entropy, H.node_entropy_H,
      H.link_entropy, H.link_entropy_H
    ]

    # # print(H.BH[0][["K", "D", "S"]])
    # print(H.equivalence)

    direction = "target"
    score = "_S"

    plot_h = Plot_H(NET, H)
    # plot_n = Plot_N(NET, H)
    # print(H.H)
    # S.zdict2newick(S.tree, weighted=F, on=T)
    # plot_h.plot_newick_R(
    #    S.newick, S.total_nodes,
    #    threshold=H.link_entropy[0].shape[0] - np.argmax(H.link_entropy[0]) - 1,
    #    weighted=F, on=T
    # )
    # S.zdict2newick(S.tree, weighted=T, on=T)
    # plot_h.plot_newick_R(S.newick, weighted=T, on=T)

    # from scipy.cluster.hierarchy import dendrogram, linkage
    # dendrogram(H.H)
    # plt.show()

    K, R, TH = get_best_kr_equivalence(score, H)
    k = K[0]
    r = R[0]

    # print(k, r, TH)

    # print(H.H)
    # print(H.equivalence)
    # print(np.sqrt(1 - H.source_sim_matrix))
    # print(np.sqrt(1 - H.target_sim_matrix))

    # # for i in [15, 16, 17, 18]:
    # #   k, r = H.equivalence[i]
    # # print(k, r, H.Z[nodes - r - 1])

    # print(k, r)

    rlabels = get_labels_from_Z(H.Z, r)
    rlabels = skim_partition(rlabels)

    # print(rlabels)

    # NET.overlap, NET.data_nocs, noc_sizes, rlabels2  = discovery_channel[discovery](H, k, rlabels, direction=direction)
    # print("\n\tAreas with predicted overlapping communities:\n",  NET.data_nocs, "\n")
    # cover = omega_index_format(rlabels2,  NET.data_nocs, NET.struct_labels[:NET.nodes])
    # H.set_cover(cover, score, direction)

    # plot_n.plot_network_covers(
    #   k, NET.A[:nodes, :], rlabels2,
    #   NET.data_nocs, noc_sizes, H.colregion.labels[:H.nodes],
    #   score=score, direction="both", cmap_name="deep", on=T, figsize=(6,6),
    #   spring=True
    # )
    # plot_h.plot_network_kk(
    #   H, rlabels2, NET.data_nocs, noc_sizes, H.colregion.labels,
    #   ang=0, score=score, font_size=7, cmap_name="deep", front_edges=True
    # )

    # plot_h.core_dendrogram([r], leaf_font_size=11, on=T) #

    # plot_h.plot_networx(
    #   rlabels, cmap_name="husl", figwidth=6, figheight=5,
    #   labels=labels_dict
    # )

    # plot_h.plot_link_communities(
    #   k, cmap_name="husl", figwidth=6, figheight=5,
    #   labels=labels_dict
    # )

    # for kk in np.arange(6, H.equivalence.shape[0]):
    #   j = H.equivalence[kk, 0]
    #   r = H.equivalence[kk, 1]
    #   print(j, r, H.Z[nodes - r - 1])

    #   rlabels = get_labels_from_Z(H.Z, r)
    #   rlabels = skim_partition(rlabels)

    #   print(rlabels)

    #   # plot_h.core_dendrogram([r], leaf_font_size=11, on=T) #

    #   # plot_h.plot_networx(
    #   #   rlabels, cmap_name="husl", figwidth=6, figheight=5,
    #   #   labels=labels_dict
    #   # )

    #   plot_h.plot_link_communities(
    #     j, cmap_name="husl", figwidth=6, figheight=5,
    #     labels=labels_dict
    #   )

    # Link hierarchies -----

    linkage = "single"
    nlog10 = F
    lookup = F
    prob = F
    cut = F
    subject = "MAC"
    structure = "FLN"
    mode = "ZERO"
    distance = "tracto16"
    nature = "original"
    # imputation_method = "RF2"
    topology = "MIX"
    mapping = "trivial"
    index  = "dist_sim"
    bias = 0.
    alpha = 0.
    discovery = "discovery_7"
    opt_score = ["_S"]
    save_data = T
    __nodes__ = 57
    __inj__ = f"{__nodes__}"
    version = f"{__nodes__}"+"d"+"106"

    NET = MAC57(
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
    # NET.labels = NET.struct_labels
    # plot_network(NET)
    # RAND = SWAPNET(
    #   __inj__,
    #   106,
    #   linkage,
    #   mode, 0,
    #   structure = structure,
    #   version = "57d106",
    #   topology=topology,
    #   nature = nature,
    #   distance = distance,
    #   model = "1k",
    #   mapping=mapping,
    #   index=index,
    #   nlog10 = nlog10, lookup = lookup,
    #   cut=cut, b=bias, discovery=discovery
    # )
    # RAND.C, RAND.A = NET.C, NET.A
    # RAND.D = NET.D
    # RAND.random_one_k(run=T, on_save_csv=F)
    # # Transform data for analysis ----
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
    ## Compute node entropy ----
    H.node_entropy_cpp("short", cut=cut)
    # import various.radialtree as rt
    from scipy.cluster.hierarchy import dendrogram
    # sns.set_style("dark")
    sns.set_context("talk")
    fig, _ = plt.subplots(1, 1)
    with plt.rc_context({'lines.linewidth': 0.5}):
      Z2 = dendrogram(H.H, no_labels=True, color_threshold=0) #0.56839
    fig.set_figwidth(35)
    fig.set_figheight(5)
    plt.gca().set_ylabel("Physical distance [mm]")
    plt.savefig(
       "../Presentations/Henry/H_tracto_model.png", dpi=300
    )

    

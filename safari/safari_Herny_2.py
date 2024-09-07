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
    S = Hierarchical_Entropy(H.Z, H.nodes, list(range(H.nodes)))

    new_label = np.arange(H.nodes, 2*(H.nodes-1)+1)

    Z = np.zeros((H.Z.shape[0], H.Z.shape[1]+3))
    Z[:, :-3] = H.Z
    Z[:, -3] = new_label

    c = 0
    for i in np.arange(1, H.nodes-1):
       if Z[i, 2] > Z[i-1, 2]:
          c += 1
       Z[i, -2] = c

    for i in np.arange(1, H.nodes-1):
        if  Z[H.nodes-1-(i+1), -2] == Z[H.nodes-1-i, -2]:
            Z[H.nodes-1-(i+1), -1] = Z[H.nodes-1-i, -1] + 1

    print(Z[:, :5], "\n", Z[:, -2:])
    S.Z2dict("short")
    H.set_hp()
    # print(H.hp)

  
    nodes = H.nodes
    tree = []

    def insert(tree, k1, k2, r, nodes):
        if isinstance(tree, list):
            if len(tree) == 0 or r in tree:
                if k1 >= nodes or k2 >= nodes:
                  tree.append([k1])
                  tree.append([k2])
                else:
                  tree.append([k1, k2])
            else:
               for t in tree:
                  insert(t, k1, k2, r, nodes)

    def insert2(tree, k1, k2, r, nodes):
        if isinstance(tree, list):
            if len(tree) == 0 or r in tree:
                if k1 >= nodes or k2 >= nodes:
                  tree.append([k1])
                  tree.append([k2])
                else:
                  tree.append(k1)
                  tree.append(k2)
            else:
               for t in tree:
                  insert2(t, k1, k2, r, nodes)
    for i in np.arange(nodes-1):
        k1 = int(Z[nodes-2-i, 0])
        k2 = int(Z[nodes-2-i, 1])
        r = int(Z[nodes-2-i, 4])

        if Z[nodes-2-i, -1] > 0 and i > 0:
          J = np.where(Z[:, -2] == Z[nodes-2-i, -2])[0]
          f = False
          for j in J:
              if j > nodes-2-i and r == Z[j, 1] and Z[j, 0] < nodes:
                r = int(Z[j, 0])
                insert2(tree, k1, k2, r, nodes)
                f = True
                break
          if not f:
            insert(tree, k1, k2, r, nodes)
        else:
          insert(tree, k1, k2, r, nodes)
    
    def remove(tree, r):
       if isinstance(tree, list):
          if r in tree:
             tree.remove(r)
          else:
             for t in tree:
                remove(t, r)
    
    def remove_empty(tree):
        if isinstance(tree, list):
          if [] in tree:
             tree.remove([])
             return True
          else:
             for t in tree:
                remove_empty(t)
        return False

    print(tree)

    for z in -np.sort(-Z[:, 4])[1:]:
       zz = int(z)
       remove(tree, zz)

    print(tree)

    f = True
    while f:
       f = remove_empty(tree)

    print(tree)

    from various.hit import check

    print(check(tree))
    # direction = "target"
    # score = "_S"

    # plot_h = Plot_H(NET, H)
    

    # K, R, TH = get_best_kr_equivalence(score, H)
    # k = K[0]
    # r = R[0]



    # rlabels = get_labels_from_Z(H.Z, r)
    # rlabels = skim_partition(rlabels)


    # plot_h.core_dendrogram([r], leaf_font_size=11, on=F) #

    

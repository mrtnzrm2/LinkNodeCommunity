# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(os.path.abspath(SCRIPT_DIR))
SCRIPT_DIR = os.path.dirname(os.path.abspath(SCRIPT_DIR))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
# Personal libraries ----
from modules.hierarmerge import Hierarchy
from networks.structure import STR
from modules.discovery import discovery_channel
from various.network_tools import *
import matplotlib as mpl
import networkx as nx
from matplotlib.ticker import MultipleLocator

# mpl.rcParams["pdf.fonttype"] = 42
# mpl.rcParams["font.size"] = 20
# sns.set_style("ticks")

def network_cover_plot(NET, H, ax : plt.Axes, cmap="deep", spring=False, scale=1, ang=0):
    RN = NET.A[:NET.nodes, :].copy()
    RN[RN > 0] = -np.log10(RN[RN > 0])
    np.fill_diagonal(RN, 0.)

     # Get best K and R ----
    K, R, _ = get_best_kr_equivalence("_S", H)
    k = K[0]
    r = R[0]

    partition_original = get_labels_from_Z(H.Z, r)
    partition_original = skim_partition(partition_original)

    _, nocs, noc_sizes, partition  = discovery_channel["discovery_7"](
      H, k, partition_original, direction="both", index="Hellinger2"
    )

    color_order = None
    undirected = False

    labels = H.colregion.labels[:H.nodes]

    from scipy.cluster import hierarchy
    import matplotlib.patheffects as path_effects
    # from matplotlib.colors import to_hex
    # Skim partition ----
    unique_clusters_id = np.unique(partition)
    keff = len(unique_clusters_id)
    # Generate all the colors in the color map -----
    if -1 in unique_clusters_id:
      save_colors = sns.color_palette(cmap, keff - 1)
      cmap_heatmap = [[]] * keff
      # cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
      cmap_heatmap[0] = [1., 1., 1.]
      cmap_heatmap[1:] = save_colors
    else:
      save_colors = sns.color_palette(cmap, keff)
      cmap_heatmap = [[]] * (keff+1)
      # cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
      cmap_heatmap[0] = [1., 1., 1.]
      cmap_heatmap[1:] = save_colors
    cmap_heatmap = np.array(cmap_heatmap)
    if isinstance(color_order, np.ndarray):
      cmap_heatmap[1:] = cmap_heatmap[1:][color_order]
    # Assign memberships to nodes ----
    if -1 in unique_clusters_id:
      nodes_memberships = {
        k : {"id" : [0] * keff, "size" : [0] * keff} for k in np.arange(len(partition))
      }
    else:
      nodes_memberships = {
        k : {"id" : [0] * (keff+1), "size" : [0] * (keff+1)} for k in np.arange(len(partition))
      }
    for i, id in enumerate(partition):
      if id == -1: continue
      nodes_memberships[i]["id"][id + 1] = 1
      nodes_memberships[i]["size"][id + 1] = 1
    for i, key in enumerate(nocs.keys()):
      index_key = np.where(labels == key)[0][0]
      for id in nocs[key]:
        if id == -1:
          nodes_memberships[index_key]["id"][0] = 1
          nodes_memberships[index_key]["size"][0] = 1
        else:
          nodes_memberships[index_key]["id"][id + 1] = 1
          nodes_memberships[index_key]["size"][id + 1] = noc_sizes[key][id]
    # Check unassigned ----
    for i in np.arange(NET.nodes):
      if np.sum(np.array(nodes_memberships[i]["id"]) == 1) == 0:
        nodes_memberships[i]["id"][0] = 1
        nodes_memberships[i]["size"][0] = 1
    # Get edges colors ----
    dA = H.dA.copy()
    if not undirected:
        dA["id"] = hierarchy.cut_tree(H.H, k).reshape(-1)
    else:
        dA["id"] = np.tile(hierarchy.cut_tree(H.H, k).reshape(-1), 2)
    minus_one_Dc(dA, undirected)
    aesthetic_ids(dA)
    dA = df2adj(dA, var="id")
    # Generate graph ----
    RN /= np.max(RN)
    G = nx.from_numpy_array(RN, create_using=nx.DiGraph)
    edge_color = [""] * len(G.edges())
    for i, dat in enumerate(G.edges(data=True)):
      u, v, a = dat
      if dA[u, v] == -1: edge_color[i] = cmap_heatmap[0]
      else: edge_color[i] = "#666666"
    
    pos = nx.kamada_kawai_layout(G, weight="weight")
    if spring:
      Rinv = RN.copy()
      Rinv[Rinv != 0] = 1e-2
      Ginv = nx.DiGraph(Rinv)

      pos = nx.spring_layout(Ginv, weight="weight", pos=pos, iterations=4, seed=212)
    ang = ang * np.pi/ 180
    rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
    pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
    labs = {k : lab for k, lab in zip(G.nodes, labels)}

    mu_pos_x = np.mean([k[0] for k in pos.values()])
    mu_pos_y = np.mean([k[1] for k in pos.values()])
    mu_pos = np.array([mu_pos_x, mu_pos_y])

    pos = {k : pos[k] - mu_pos for k in pos.keys()}
    pos = {k : pos[k] * scale for k in pos.keys()}

    ####
    iv1 = np.where(labels == "v1")[0][0]
    pos[iv1] += np.array([0.1, 0.1])
    iv2 = np.where(labels == "v2")[0][0]
    pos[iv2] += np.array([0, -0.1])
    i46 = np.where(labels == "46d")[0][0]
    pos[i46] += np.array([0.05, 0.05])
    i946 = np.where(labels == "9/46d")[0][0]
    pos[i946] += np.array([0, -0.05])
    if4 = np.where(labels == "f4")[0][0]
    pos[if4] += np.array([0.05, 0.05])
    ####

    nx.draw_networkx_edges(
      G, pos=pos,
      edge_color="#666666",
      alpha=0.3, width=0.75, arrowsize=20,
      connectionstyle="arc3,rad=-0.1", arrowstyle="->",
      node_size=200, ax=ax
    )
      
    t = nx.draw_networkx_labels(
      G, pos=pos, labels=labs, font_color="white",
      font_size=6, font_weight="bold", ax=ax
    )
    for key in t.keys():
      t[key].set_path_effects(
      [
        path_effects.Stroke(linewidth=0.75, foreground='gray'),
        path_effects.Normal()
      ]
    )
    
    for node in G.nodes:
      if partition_original[node] == -1:
        wedgecolor = "black"
      else: wedgecolor = "gray"
      a = ax.pie(
        [s for s in nodes_memberships[node]["size"] if s != 0], # s.t. all wedges have equal size
        center=pos[node],
        colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[node]["id"]) if id != 0],
        radius=0.075,
        wedgeprops={"linewidth" : 0.5, "edgecolor": wedgecolor}
      )
      for i in range(len(a[0])):
        a[0][i].set_alpha(0.8)
    array_pos = np.array([list(pos[v]) for v in pos.keys()])
    ax.set_xlim(-0.2 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.2)
    ax.set_ylim(-0.2 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.2)
    ax.set_frame_on(False)

  
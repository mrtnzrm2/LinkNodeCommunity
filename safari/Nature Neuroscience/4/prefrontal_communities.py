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

def prefrontal_network_plot(NET, H, ax : plt.Axes, cmap="deep", spring=False, scale=1, ang=0):
    RN = NET.A[:NET.nodes, :].copy()
    RN[RN > 0] = -np.log10(RN[RN > 0])
    np.fill_diagonal(RN, 0.)

    labels = H.colregion.labels[:H.nodes]

    prefrontal_areas = [
      "24c", "f7", "8b", "9", "9/46v",
      "8m", "45a", "8l", "46d", "9/46d",
      "25", "10", "32"
    ]

    Ipref = np.where(np.isin(labels, prefrontal_areas))[0]
    Tr = {k: v for k, v in zip(Ipref, np.arange(len(prefrontal_areas)))}
    Gpref = RN[:NET.nodes, :][Ipref, :][:, Ipref]

     # Get best K and R ----
    r = 24
    k = get_k_from_equivalence(r, H)

    partition_original = get_labels_from_Z(H.Z, r)
    partition_original = skim_partition(partition_original)

    _, nocs, noc_sizes, partition  = discovery_channel["discovery_7"](
      H, k, partition_original, direction="both", index="Hellinger2"
    )

    color_order = None

    color_prefrontal_palette = ['1fbeb8', '0088c9', '7aa1be']

    def hex_to_rgb(hex):
      return tuple(int(hex[i:i+2], 16) / 255 for i in (0, 2, 4))

    for i in np.arange(len(color_prefrontal_palette)):
      color_prefrontal_palette[i] = hex_to_rgb(color_prefrontal_palette[i])

    partition_original = partition_original[Ipref]
    partition = partition[Ipref]

    nocs = {k: v for k, v in nocs.items() if k in prefrontal_areas}
    noc_sizes = {k: v for k, v in noc_sizes.items() if k in prefrontal_areas}

    import matplotlib.patheffects as path_effects
    # Skim partition ----
    unique_clusters_id = np.unique(partition)
    keff = len(unique_clusters_id)
    # Generate all the colors in the color map -----
    if -1 in unique_clusters_id:
      save_colors = color_prefrontal_palette
      cmap_heatmap = [[]] * keff
      # cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
      cmap_heatmap[0] = [1., 1., 1.]
      cmap_heatmap[1:] = save_colors
    else:
      save_colors = color_prefrontal_palette
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
      if key not in prefrontal_areas: continue
      index_key = np.where(prefrontal_areas == key)[0][0]
      for id in nocs[key]:
        if id == -1:
          nodes_memberships[index_key]["id"][0] = 1
          nodes_memberships[index_key]["size"][0] = 1
        else:
          nodes_memberships[index_key]["id"][id + 1] = 1
          nodes_memberships[index_key]["size"][id + 1] = noc_sizes[key][id]
    # Check unassigned ----
    for i in np.arange(len(prefrontal_areas)):
      if np.sum(np.array(nodes_memberships[i]["id"]) == 1) == 0:
        nodes_memberships[i]["id"][0] = 1
        nodes_memberships[i]["size"][0] = 1

    # Generate graph ----
    RN /= np.max(RN)
    G = nx.from_numpy_array(RN, create_using=nx.DiGraph) 
    pos = nx.kamada_kawai_layout(G, weight="weight")
    if spring:
      Rinv = RN.copy()
      Rinv[Rinv != 0] = 1e-2
      Ginv = nx.DiGraph(Rinv)
      pos = nx.spring_layout(Ginv, weight="weight", pos=pos, iterations=4, seed=212)

    ang = ang * np.pi/ 180
    rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
    pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}

    mu_pos_x = np.mean([k[0] for k in pos.values()])
    mu_pos_y = np.mean([k[1] for k in pos.values()])
    mu_pos = np.array([mu_pos_x, mu_pos_y])

    pos = {k : pos[k] - mu_pos for k in pos.keys()}
    pos = {k : pos[k] * scale for k in pos.keys()}

    # ####
    # iv1 = np.where(labels == "v1")[0][0]
    # pos[iv1] += np.array([0.1, 0.1])
    # iv2 = np.where(labels == "v2")[0][0]
    # pos[iv2] += np.array([0, -0.1])
    i46 = np.where(labels == "46d")[0][0]
    pos[i46] += np.array([0.05, 0.05])
    i946 = np.where(labels == "9/46d")[0][0]
    pos[i946] += np.array([0, -0.05])
    # if4 = np.where(labels == "f4")[0][0]
    # pos[if4] += np.array([0.05, 0.05])
    # ####

    G = nx.from_numpy_array(Gpref, create_using=nx.DiGraph)

    # print({a for a, _ in pos.items() if a in Ipref})
    # print(labels[Ipref])
    # print({prefrontal_areas[Tr[a]] for a, _ in pos.items() if a in Ipref})
    # print({k: lab for k, lab in zip(G.nodes, prefrontal_areas)})

    labs = {Tr[k]: labels[k] for k in pos.keys() if k in Ipref}
    pos = {Tr[k]: v for k, v in pos.items() if k in Ipref}

    nx.draw_networkx_edges(
      G, pos=pos,
      edge_color="#666666",
      alpha=0.5, width=1, arrowsize=20,
      connectionstyle="arc3,rad=-0.1", arrowstyle="->",
      node_size=400, ax=ax
    )
      
    t = nx.draw_networkx_labels(
      G, pos=pos, labels=labs, font_color="white",
      font_size=12, font_weight="bold", ax=ax
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
        wedgeprops={"linewidth" : 1, "edgecolor": wedgecolor}
      )
      for i in range(len(a[0])):
        a[0][i].set_alpha(0.8)

    array_pos = np.array([list(pos[v]) for v in pos.keys()])
    ax.set_xlim(-0.2 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.2)
    ax.set_ylim(-0.2 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.2)
    ax.set_frame_on(False)
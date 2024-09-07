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
import pandas as pd
import seaborn as sns

from networks.structure import STR
from modules.discovery import discovery_channel
from various.network_tools import *

# TODO: not the good version because only the last thing is visible, transform it to be a plotted on a grid!!!

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys
import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import copy

file_path = "/Users/jmarti53/Documents/Projects/LINKPROJECT/Dlink/utils/flatmap_91/"

np.set_printoptions(linewidth=240)

def getAreaIndex(area, labels):
    for index, l in enumerate(labels):
        if(l == area):
            return index
        else:
            if((l == "v1" or l == "v2" or l == "v4") and l in area and len(area) > 3):
                return index
    return -1

print("> processing MAC")

def flatmap_91_plot(NET, H, ax: plt.Axes, cmap="deep", color_order=None):
  import matplotlib.patheffects as path_effects
  # Get best K and R ----
  K, R, _ = get_best_kr_equivalence("_S", H)
  k = K[0]
  r = R[0]

  clustering = get_labels_from_Z(H.Z, r)
  clustering = skim_partition(clustering)
  labels = NET.struct_labels[:NET.nodes]

  print(">> reading data")
  clusterColors = sns.color_palette(cmap)

  print("> reading flatmap data")
  f = open(file_path + "macaque_flatmap_91B.json")
  polygonData = json.load(f)
  f.close()

  f = open(file_path + "label_coord.json")
  data = json.load(f)
  f.close()

  print(">>> preparing data for creating polygons")
  index = []
  name = []
  x = []
  y = []
  for pdindex, pd in enumerate(polygonData):            
      for e in polygonData[pd][0]:          
          index.append(pdindex)
          name.append(pd)      
          x.append(e[0])
          y.append(e[1])

  for index, n in enumerate(name):
      name[index] = n.replace("\"", "")

  totalAreas = len(set(name))
  start = np.zeros(totalAreas)
  end = np.zeros(totalAreas)
  sindex = 0
  eindex = 0
  for index, n in enumerate(name):

      if(index == 0):
          start[sindex] = index
          sindex += 1
      if(index != len(name) - 1):
          if(n != name[index + 1]):
              end[eindex] = index
              eindex += 1
              start[sindex] = index + 1
              sindex += 1
      else:
          end[eindex] = len(name) - 1

  start = start.astype(int)
  end = end.astype(int)

  print("> Covers")
  _, nocs, noc_sizes, partition  = discovery_channel["discovery_7"](
      H, k, clustering, direction="both", index="Hellinger2"
  )

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

  nocs_indeces = [
      getAreaIndex(labels[n].lower(), labels) for n in np.where(clustering == -1)[0]
  ]

  nodes_memberships = {
    getAreaIndex(labels[n].lower(), labels) : {"id" : [0] * (keff+1), "size" : [0] * (keff+1)} for n in nocs_indeces
  }

  for _, key in enumerate(nocs.keys()):
      index_key = getAreaIndex(key, labels)
      if index_key not in nocs_indeces: continue
      for id in nocs[key]:
          if id == -1:
            nodes_memberships[index_key]["id"][0] = 1
            nodes_memberships[index_key]["size"][0] = 1
          else:
            nodes_memberships[index_key]["id"][id + 1] = 1
            nodes_memberships[index_key]["size"][id + 1] = noc_sizes[key][id]

  print("> plotting")

  for index in range(totalAreas):
      px = x[start[index]:(end[index] + 1)]
      py = y[start[index]:(end[index] + 1)]

      vertices = np.column_stack((px, py))

      aindex = getAreaIndex(name[start[index]].lower(), labels)
      if aindex != -1 and clustering[aindex] != -1:
          colorIndex = clustering[aindex]
          color = clusterColors[colorIndex]
      elif aindex != -1 and clustering[aindex] == -1:
          color = tuple((1, 1, 1, 0))
      else:
          color = tuple((.5, .5, .5, .25))
      
      pattern = ''

      if aindex != -1 and clustering[aindex] != -1:
        ax.add_patch(plt.Polygon(vertices, closed=True, edgecolor=[1, 1, 1], linewidth=.5, facecolor=color, hatch=pattern))
      elif aindex != -1 and clustering[aindex] == -1:
        ax.add_patch(plt.Polygon(vertices, closed=True, edgecolor=[0, 0, 0], linewidth=.5, facecolor=color, hatch=pattern))
      else:
        ax.add_patch(plt.Polygon(vertices, closed=True, edgecolor=[1, 1, 1], linewidth=.5, facecolor=color, hatch=pattern))


  for d in data:
      aindex = getAreaIndex(d.lower(), labels)
      if aindex != -1:
        t = ax.text(data[d][0][0], data[d][0][1], d, fontsize=6, weight="bold", color="yellow")
        t.set_path_effects(
        [
          path_effects.Stroke(linewidth=0.5, foreground='k'),
          path_effects.Normal()
        ])
      if d.lower() in nocs.keys():
          wedgecolor = "k"
          ax.pie(
            [s for s in nodes_memberships[aindex]["size"] if s != 0],
            center=np.array([data[d][0][0], data[d][0][1]]),
            colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[aindex]["id"]) if id != 0],
            radius=0.45,
            wedgeprops={"linewidth" : 0.35, "edgecolor": wedgecolor}
          )

  ax.set_aspect("equal")
  ax.autoscale()
  ax.set_axis_off()

  # plt.show()

  print("> done!")


# linkage = "single"
# nlog10 = F
# lookup = F
# prob = T
# cut = F
# subject = "MAC"
# structure = "FLNe"
# mode = "ZERO"
# nature = "original"
# imputation_method = ""
# topology = "MIX"
# discovery = "discovery_7"
# mapping = "trivial"
# index  = "Hellinger2"
# bias = float(0)
# alpha = 0.
# __nodes__ = 40
# __inj__ = 40
# distance = "MAP3D"
# version = f"{__nodes__}" + "d" + "91"
# model_distbase = "M"
# model_swaps = "TWOMX_FULL" # 1K_DENSE

# if __name__ == "__main__":
#     NET = STR[f"{subject}{__inj__}"](
#       linkage, mode,
#       nlog10 = nlog10,
#       structure = structure,
#       lookup = lookup,
#       version = version,
#       nature = nature,
#       model = imputation_method,
#       distance = distance,
#       inj = __inj__,
#       discovery = discovery,
#       topology = topology,
#       index = index,
#       mapping = mapping,
#       cut = cut,
#       b = bias,
#       alpha = alpha
#     )
#     pickle_path = NET.pickle_path

#     H = read_class(pickle_path, "hanalysis")

#     # Get best K and R ----
#     K, R, _ = get_best_kr_equivalence("_S", H)
#     k = K[0]
#     r = R[0]

#     rlabels = get_labels_from_Z(H.Z, r)
#     rlabels = skim_partition(rlabels)

#     labels = NET.struct_labels[:NET.nodes]

#     fig, ax = plt.subplots(1, 1, figsize=(5.35433/2, 6/2))

#     flatmap_91_plot(NET, H, ax)
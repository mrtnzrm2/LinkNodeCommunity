# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libr

import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from os.path import join
from nilearn.plotting import plot_surf
from various.network_tools import read_class, skim_partition, match

from networks.structure import STR

def load_structure():
  linkage = "single"
  nlog10 = T
  lookup = F
  cut = F
  subject = "MAC"
  structure = "FLN"
  mode = "ZERO"
  distance = "MAP3D"
  nature = "original"
  imputation_method = ""
  topology = "MIX"
  discovery = "discovery_7"
  mapping = "trivial"
  index  = "Hellinger2"
  bias = float(0)
  alpha = 0.
  version = "29d91"
  __nodes__ = 29
  __inj__ = 29

  NET = STR[f"{subject}{__inj__}"](
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    version = version,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = __inj__,
    discovery = discovery,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    b = bias,
    alpha = alpha
  )
  return NET, read_class(NET.pickle_path, "hanalysis")
      

NET, H = load_structure()

labels = "Core-Nets_M132LH.label.gii"
borders = "Core-Nets_M132LH.func.gii"
surface = "Core-Nets_M132LH.midthickness.surf.gii"
spec = "Core-Nets_M132LH.spec"
path = "../Workbench/M132LH/"

surface_gii = nib.load(join(path, surface))

labels_gii = nib.load(join(path, labels))
labeltable = labels_gii.labeltable.get_labels_as_dict()
labels = labels_gii.agg_data("NIFTI_INTENT_LABEL")

borders_gii = nib.load(join(path, borders))
borders = borders_gii.agg_data("NIFTI_INTENT_NORMAL")
borders_complete = np.zeros(borders[0].shape[0])
for bd in borders:
  borders_complete += bd
borders_complete[borders_complete > 0] = 1.

## modifying labels ----
max_label = np.max(labels)
uncolor_nodes = np.array([0, 52, 64, 94])
for unc in uncolor_nodes:
  labels[labels == unc] = max_label + 1
labeltable[max_label+1] = "None_M132LH"

labels_copy = labels.copy()
labeltable_rw = dict()


def label_gii2structure(lb, NET):
  if NET.version.startswith("29"):
    if lb == "9_46d":
      return "9/46d"
    elif lb == "9_46v":
      return "9/46v"
    elif lb == "ento":
      return "entorhinal"
    elif lb == "ins":
      return "insula"
    elif lb == "peri":
      return "perirhinal"
    elif lb == "pir":
      return "piriform"
    elif lb == "temporal-pole":
      return "temporal_pole"
    elif lb == "s2":
      return "sii"
    elif lb == "sub":
      return "subiculum"
    elif lb == "pro.":
      return "pro.st."
    elif lb == "tea_m-p":
      return "tea/mp"
    elif lb == "tea_m-a":
      return "tea/ma"
    elif lb == "29_30":
      return "29/30"
    elif lb == "pi":
      return "parainsula"
    else: return lb
  else:
    raise RuntimeError("Unknown version.")
  
for i, lb in enumerate(np.sort(np.unique(labels_copy))):
  labels[labels_copy == lb] = i
  labeltable_rw[i] = label_gii2structure(labeltable[lb].split("_M1")[0].lower(), NET)

labeltable_rw_rev = {v: k for k, v in labeltable_rw.items()}
  
# Relabel by clustering ----
rlabels = skim_partition(H.rlabels["both"]["labels"])

edge_complete_labels = NET.struct_labels[:NET.nodes]
area_labels_dic = {ar : r for ar, r in zip(edge_complete_labels, rlabels)}
masked_labels = [
  k for k, v in labeltable_rw.items() if v not in edge_complete_labels # or area_labels_dic[v] == -1
]

## Split nocs in labels ---- 
covers = H.cover["both"]["_S"]
noc_labels = {lb : [] for lb in edge_complete_labels if area_labels_dic[lb] == -1}
for u, areas in covers.items():
  for v in areas:
    if v in list(noc_labels.keys()):
      noc_labels[v].append(u)

from nilearn.surface import load_surf_mesh
from sklearn.cluster import BisectingKMeans
coords, faces = load_surf_mesh(join(path, surface))

sublabels = np.zeros(labels.shape) * np.nan
for noc, covs in noc_labels.items():
  covs = np.array(covs)
  index_sublabels = np.where(labels == labeltable_rw_rev[noc])[0]
  subcoords = coords[index_sublabels]
  kmeans = BisectingKMeans(n_clusters=len(covs), random_state=0).fit(subcoords)
  klabels = kmeans.labels_
  sublabels[index_sublabels] = covs[klabels]

## Masking structures ----
mask_data = np.zeros(labels.shape)
for mk in masked_labels:
  mask_data[labels == mk] = 1
mask_data[borders_complete == 1] = 0

## hovertext ----
hovertext = [labeltable_rw[lb]+"<extra></extra>" for lb in labels]

## Clustering relabel ----
labels_copy = labels.copy()
for key, lb in labeltable_rw.items():
  if lb in edge_complete_labels:
    r = area_labels_dic[lb]
    if r != -1:
      labels[labels_copy == key] = r
    else:
      labels[labels_copy == key] = 0
  else:
    labels[labels_copy == key] = 0

labels[~np.isnan(sublabels)] = sublabels[~np.isnan(sublabels)]
r = np.unique(labels).shape[0]

fig = plot_surf(
  join(path, surface), surf_map=labels, bg_map=borders_complete,
  engine="plotly", cmap=sns.color_palette("hls", as_cmap=True),
  darkness=None, borders=True, mask=mask_data, symmetric_cmap=False
).figure

fig.data[0]["hovertemplate"] = hovertext
fig.update_layout(
  hovermode="closest",
  hoverlabel={"bgcolor" : "white"},
  paper_bgcolor='rgba(0,0,0,1)'
)

fig.write_html(f"{NET.plot_path}/plotly/surface/{r}.html")
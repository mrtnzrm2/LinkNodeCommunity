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
from networks.toy import TOY
from networks.MAC.mac57 import MAC57
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.hierarentropy import Hierarchical_Entropy
from modules.flatmap import FLATMAP
from modules.colregion import colregion
from various.data_transformations import maps
from modules.discovery import discovery_channel
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
discovery = "discovery_7"
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

  NET_H = read_class(
    NET.pickle_path,
    "hanalysis"
  )

  D = NET.D
  NEURONS = NET.C

  from sklearn.manifold import MDS
  embedding = MDS(n_components=3, dissimilarity="precomputed")
  Xt = embedding.fit_transform(D)

def cartesian_to_spherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,2] = np.arctan2(np.sqrt(xy), xyz[:,2])
    ptsnew[:,1] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

Xt = (Xt - np.mean(Xt, axis=0)) / np.std(Xt, axis=0)

# Coordinate transformation ----
Rz = lambda phi: np.array(
  [
    [np.cos(phi), np.sin(phi), 0],
    [-np.sin(phi), np.cos(phi), 0],
    [0, 0, 1]
  ]
)

Rz2 = lambda phi: np.array(
  [
    [np.cos(phi), -np.sin(phi), 0],
    [np.sin(phi), np.cos(phi), 0],
    [0, 0, 1]
  ]
)

Ry = lambda phi: np.array(
  [
    [np.cos(phi), 0, -np.sin(phi)],
    [0,1, 0],
    [np.sin(phi), 0, np.cos(phi)]
  ]
)

iv1 = np.where(NET.struct_labels == "v1c")[0][0]
V1 = Xt[iv1, :]
i8L = np.where(NET.struct_labels == "8l")[0][0]
a8L = Xt[i8L, :]

u = (a8L - V1)
from scipy.linalg import norm
u /= norm(u)
u = cartesian_to_spherical(u.reshape(1, -1)).ravel()

Tr = Ry(u[2]) @ Rz(u[1])

# print(Tr @ v.reshape(-1, 1))

for i in np.arange(NET.rows):
   Xt[i, :] = (Tr @ Xt[i, :].reshape(-1, 1)).reshape(1, -1)

# Calibrate perpendicular axis

# xymax = -np.Inf
# ixymax = -1
# for i in np.arange(NET.rows):
#    xy = norm(Xt[i, :-1])
#    if xy > xymax:
#       xymax = xy
#       ixymax = i
      
# phi = np.arctan2(Xt[ixymax, 1], Xt[ixymax, 0])
# print(phi)
# uu = (Rz(phi) @ Xt[ixymax, :].reshape(-1, 1)).ravel()
# print(np.arctan2(uu[1], uu[0]))

iv1 = np.where(NET.struct_labels == "10")[0][0]
V1 = Xt[iv1, :]
i8L = np.where(NET.struct_labels == "aip")[0][0]
a8L = Xt[i8L, :]

u = (a8L - V1)
u /= norm(u)
u = cartesian_to_spherical(u.reshape(1, -1)).ravel()

Tr = Rz(u[1])
for i in np.arange(NET.rows):
   Xt[i, :] = (Tr @ Xt[i, :].reshape(-1, 1)).reshape(1, -1)

# raise ValueError("")
# St = cartesian_to_spherical(Xt)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2])

# for i in np.arange(NET.rows):
#    ax.text(Xt[i, 0], Xt[i, 1], Xt[i, 2], NET.struct_labels[i])

# plt.show()

# 2 -----

rmin, rmax = np.Inf, -np.Inf
theta_min, theta_max =  np.Inf, -np.Inf
phi_min, phi_max =  np.Inf, -np.Inf

for i in np.arange(NET.rows):
   for j in np.arange(NET.rows):
      if i == j: continue
      vij = Xt[j, :] - Xt[i, :]
      vij = cartesian_to_spherical(vij.reshape(1, -1)).ravel()
      if vij[0] < rmin: rmin = vij[0]
      if vij[0] > rmax: rmax = vij[0]
      if vij[2] < theta_min: theta_min = vij[2]
      if vij[2] > theta_max: theta_max = vij[2]
      if vij[1] < phi_min: phi_min = vij[1]
      if vij[1] > phi_max: phi_max = vij[1]

bins = 12

R_span = np.linspace(rmin, rmax, bins+1)
R_span[0] -=1
R_span[-1] += 1

Theta_span = np.linspace(theta_min, theta_max, bins+1)
Theta_span[0] -= 1
Theta_span[-1] += 1

Phi_span = np.linspace(phi_min, phi_max, bins+1)
Phi_span[0] -= 1
Phi_span[-1] += 1

P = np.zeros((bins, bins, bins))

def find_index_span(z, boundaries, bins):
    for i in np.arange(bins):
      if (boundaries[i] <= z) and (boundaries[i+1] > z):
         return i
    return -1

connections_dict = {}

for i in np.arange(NET.rows):
   for j in np.arange(NET.rows):
      if i == j: continue
      vij = Xt[j, :] - Xt[i, :]
      vij = cartesian_to_spherical(vij.reshape(1, -1)).ravel()
      ir = find_index_span(vij[0], R_span, bins)
      if ir == -1: raise ValueError("Index can't be negative")
      itheta = find_index_span(vij[2], Theta_span, bins)
      if itheta == -1: raise ValueError("Index can't be negative")
      iphi = find_index_span(vij[1], Phi_span, bins)
      if iphi == -1: raise ValueError("Index can't be negative")
      if (ir, iphi, itheta) not in connections_dict:
        connections_dict[(ir, iphi, itheta)] = [(i, j)]
      else: connections_dict[(ir, iphi, itheta)].append((i, j))
      if j < NET.nodes:
        P[ir, iphi, itheta] += NEURONS[i, j]

P /= np.sum(P)

R_span[0] +=1
R_span[-1] -= 1

Theta_span[0] += 1
Theta_span[-1] -= 1

Phi_span[0] += 1
Phi_span[-1] -= 1

# 3 ----

L = bins * bins * bins
E = np.zeros((L, 4))

e = 0
for i in np.arange(bins):
   for j in np.arange(bins):
      for k in np.arange(bins):
        E[e, 0] = i
        E[e, 1] = j
        E[e, 2] = k
        E[e, 3] = P[i,j,k]
        e += 1

Nmodel = np.zeros((NET.rows, NET.rows))
rng = np.random.default_rng()

M = 0
nonzero = np.sum(NET.C > 0)

while M < nonzero:
  experiment = np.random.choice(E.shape[0], size=1000, p=E[:, -1].ravel())
  for exp in experiment:
     ternary = (int(E[exp, 0]), int(E[exp, 1]), int(E[exp, 2]))
     if ternary in connections_dict:
        len_ternary = len(connections_dict[ternary])
        t = np.random.randint(len_ternary)
        i, j = connections_dict[ternary][t]
        Nmodel[i, j] +=1
        M = np.sum(Nmodel[:,:NET.nodes] > 0)
     if M >= nonzero: break

Pmodel = Nmodel.copy()
Pmodel /= np.sum(Nmodel, axis=0)

# 4 ----
properties = {
  "version" : "3Dfit",
  "nlog10" : nlog10,
  "lookup" : lookup,
  "prob" : prob,
  "cut" : cut,
  "topology" : topology,
  "mapping" : mapping,
  "index" : index,
  "mode" : mode,
}

toy = TOY(Pmodel[:, :NET.nodes], linkage, **properties)
toy.set_labels(NET.struct_labels)
toy.create_plot_directory()

H = Hierarchy(
  toy, Pmodel[:, :NET.nodes], Pmodel[:, :NET.nodes], NET.D,
  __nodes__, linkage, mode, lookup=lookup,
  index=index
)
H.BH_features_cpp_no_mu()
H.la_abre_a_merde_cpp(H.BH[0])
H.get_h21merge()
L = colregion(NET, labels_name=f"labels{__inj__}")
H.set_colregion(L)
H.delete_dist_matrix()

plot_h = Plot_H(toy, H)
plot_n = Plot_N(toy, H)

HS = Hierarchical_Entropy(H.Z, H.nodes, toy.struct_labels[:toy.nodes])
HS.Z2dict("short")

HS.zdict2newick(HS.tree, weighted=T, on=T)
plot_h.plot_newick_R(HS.newick, HS.total_nodes, root_position=1-H.Z[:, 2][-1], weighted=T, on=T)


RN = toy.A[:__nodes__, :].copy()
RN[RN > 0] = -np.log(RN[RN > 0])
np.fill_diagonal(RN, 0.)

RW = toy.A.copy()
RW[RW > 0] = -np.log(RW[RW > 0])
np.fill_diagonal(RW, 0.)

RW10 = toy.A.copy()
RW10[RW10 > 0] = -np.log10(RW10[RW10 > 0])
np.fill_diagonal(RW10, 0.)

plot_n.A_vs_dis(-RW, s=10, on=F, reg=T)
plot_n.projection_probability(Nmodel, "EXPMLE" , bins=bins, on=T)
plot_n.histogram_weight(-RW10, label=r"$\log10(p(i,j))$", suffix="log10_p", on=T)
plot_n.plot_akis(NET.D, s=5, on=T)

for SCORE in opt_score:
  # Get best K and R ----
  K, R, TH = get_best_kr_equivalence(SCORE, H)
  for k, r, th in zip(K, R, TH):
    print(f"Find node partition using {SCORE}")
    print("Best K: {}\nBest R: {}\t Score: {}".format(k, r, SCORE))
    H.set_kr(k, r, score=SCORE)
    rlabels = get_labels_from_Z(H.Z, r)
    rlabels = skim_partition(rlabels)

    print(">>> Single community nodes:")
    print(NET.struct_labels[:NET.nodes][np.where(rlabels == -1)[0]], "\n")

    # Plot H ----
    plot_h.core_dendrogram([r], leaf_font_size=12, on=F) #
    plot_h.heatmap_dendro(r, -RN, on=T, cmap="rocket", score="FLNe", cbar_label=r"$\log_{10}$FLNe", font_size = 12, suffix="perm")
    plot_h.lcmap_dendro(k, r, on=F, font_size = 12) #
    plot_h.threshold_color_map(r, th, index=index, score=SCORE, font_size = 12, on=F)
    
    # Overlap ----
    for direction in ["source", "target", "both"]: # , 
      print("***", direction)
      toy.overlap, toy.data_nocs, noc_sizes, rlabels2  = discovery_channel[discovery](H, k, rlabels, direction=direction, index=index)
      print(">>> Areas with predicted overlapping communities:\n",  toy.data_nocs, "\n")
      cover = omega_index_format(rlabels2,  toy.data_nocs, toy.struct_labels[:toy.nodes])
      omega_index(cover, NET_H.cover[direction]["_S"])
      # cover_art = {}

      plot_n.plot_network_covers(
        k, RN, rlabels2, rlabels,
        toy.data_nocs,
        noc_sizes, H.colregion.labels[:H.nodes], ang=0,
        score=SCORE, direction=direction, spring=F, font_size=10,
        scale=0.5,
        suffix="", cmap_name="hls", not_labels=F, on=F
      )

# 5 ----

# dL = D[:, :NET.nodes].ravel().shape[0]
# mL = D.ravel().shape[0]
# data = {
#    "distances" : list(D[:, :NET.nodes].ravel()) + list(D.ravel()),
#    "FLNe" : list(NET.A.ravel()) + list(Nmodel.ravel()),
#    "model" : ["exp"] * dL + ["model"] * mL
# }

# sns.scatterplot(
#    data=data,
#    x="distances",
#    y="FLNe",
#    hue="model"
# )

# plt.yscale("log")

# plt.show()

# 6 ----

# P /= np.sum(P)

# RPHI = np.sum(P, axis=2)
# RPHI /= np.sum(RPHI)
# RPHI[RPHI == 0] = np.nan

# R_tickvalue = (R_span[1:] + R_span[:-1]) / 2
# R_tickvalue = np.round(R_tickvalue, 3)

# Phi_tickvalue = (Phi_span[1:] + Phi_span[:-1]) / 2
# Phi_tickvalue = np.round(Phi_tickvalue, 3)


# sns.heatmap(np.log(RPHI), cmap="rocket", xticklabels=Phi_tickvalue, yticklabels=R_tickvalue)
# plt.xticks(rotation=90)
# plt.show()

# 7 ----   

# TOTAL_NEURONS = np.sum(NEURONS)
# rx = np.zeros(np.ceil(TOTAL_NEURONS).astype(int)) * np.nan
# thetax = np.zeros(np.ceil(TOTAL_NEURONS).astype(int)) * np.nan
# phix = np.zeros(np.ceil(TOTAL_NEURONS).astype(int)) * np.nan
# e = 0
# for i in np.arange(NET.rows):
#   for j in np.arange(NET.nodes):
#       if NEURONS[i, j] == 0: continue
#       vij = Xt[j, :] - Xt[i, :]
#       vij = cartesian_to_spherical(vij.reshape(1, -1)).ravel()
#       rx[e:(e+int(NEURONS[i,j]))] = vij[0]
#       thetax[e:(e+int(NEURONS[i,j]))] = vij[1]
#       phix[e:(e+int(NEURONS[i,j]))] = vij[2]
#       e += int(NEURONS[i,j])

# rx = rx[~np.isnan(rx)]
# thetax = thetax[~np.isnan(thetax)]
# phix = phix[~np.isnan(phix)]

# fig, ax = plt.subplots(1, 3)

# ax[0].hist(rx, density=True)
# ax[1].hist(thetax, density=True)
# ax[2].hist(phix, density=True)

# ax[0].set_yscale('log')
# plt.show()




  



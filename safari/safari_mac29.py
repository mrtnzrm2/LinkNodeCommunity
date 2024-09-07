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
# Personal libs ---- 
from networks.MAC.mac29 import MAC29
from modules.hierarmerge import Hierarchy
from various.data_transformations import maps
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
structure = "FLNe"
mode = "ZERO"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0.0
opt_score = ["_S"]
save_data = T
version = "29d91"
__nodes__ = 29
__inj__ = 29


def get_structure_new(csv_path) -> pd.DataFrame:
    # Get structure ----
    file = pd.read_csv(f"{csv_path}/Cercor_2012-Table.csv")
    ## Areas to index
    tlabel = np.unique(file["TARGET"]).astype(str)
    inj = tlabel.shape[0]
    print(inj)
    slabel = np.unique(file["SOURCE"])
    total_areas = slabel.shape[0]
    slabel1 = [lab for lab in slabel if lab not in tlabel]
    slabel = np.array(list(tlabel) + slabel1, dtype="<U21").astype(str)
    file["SOURCE_IND"] = match(file["SOURCE"], slabel)
    file["TARGET_IND"] = match(file["TARGET"], slabel)
    ## Average Count
    monkeys = np.unique(file.MONKEY)
    C = []
    tid = np.unique(file.TARGET_IND)
    tmk = {t : [] for t in tid}
    for i, m in enumerate(monkeys):
      Cm = np.zeros((total_areas, inj))
      data_m = file.loc[file.MONKEY == m]
      Cm[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m["NEURONS"]
      C.append(Cm)
      for t in np.unique(data_m.TARGET_IND):
        tmk[t].append(i)
    C = np.array(C)
    C[np.isnan(C)] = 0
    CC = np.sum(C, axis=0)
    c = np.zeros((total_areas, inj))
    for t, mnk in tmk.items(): c[:, t] = np.mean(C[mnk, :, t], axis=0)
    A = CC / np.sum(CC, axis=0)

    slabel = np.char.lower(slabel)
    tlabel = np.char.lower(tlabel)

    original = np.array(["ins", "ento", "pole", "pir", "pi", "sub", "peri"])
    reference = np.array(["insula", "entorhinal", "temporal_pole", "piriform", "parainsula", "subiculum", "perirhinal"])


    slabel[match(original, slabel)] = reference
    # tlabel[match(original, tlabel)] = reference



    # np.savetxt(f"{self.csv_path}/labels.csv", self.struct_labels,  fmt='%s')
    return pd.DataFrame(A, index=slabel, columns=tlabel)

# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC29(
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

  R, lookup, _ = maps[mapping](
    NET.A, nlog10, lookup, prob, b=bias
  )

## Hierarchy object!! ----
#   H = Hierarchy(
#     NET, NET.A, R, NET.D,
#     __nodes__, linkage, mode, lookup=lookup, index=index
#   )

#   D12 = H.target_sim_matrix
#   D12[D12 != 0] = -2 * np.log(D12[D12 != 0])

#   D12p = H.source_sim_matrix
#   D12p[D12p != 0] = -2 * np.log(D12p[D12p != 0])

#   SLN = NET.SLN[:NET.nodes, :][:, :NET.nodes]

#   A = NET.A[:NET.nodes, :][:, :NET.nodes]
#   non_zero = A > 0

#   data = pd.DataFrame(
#      {
#         "SLN" : list(SLN[non_zero]) * 2,
#         "D12" : np.hstack([D12[non_zero], D12p[non_zero]]),
#         "class" : ["-"] * np.sum(non_zero) + ["+"] * np.sum(non_zero)
#      }
#   )

#   data = pd.DataFrame(
#      {
#         "SLN" : list(SLN[non_zero]) * 1,
#         "D12" : D12[non_zero]
#         # "class" : ["-"] * np.sum(non_zero) + ["+"] * np.sum(non_zero)
#      }
#   )

#   sns.set_style("whitegrid")

#   true_bins = 15
#   bins = true_bins - 1
#   min_x = np.min(data["SLN"])
#   max_x = np.max(data["SLN"])
#   x_bin_boundaries = np.linspace(min_x, max_x, bins+1)
#   bin_width = x_bin_boundaries[1]-x_bin_boundaries[0]
#   x_bin_center = x_bin_boundaries[1:] - bin_width
#   x_bin_center = np.hstack([x_bin_center, np.array(x_bin_center[-1] + bin_width)])
#   x_bin_boundaries -= bin_width / 2
#   x_bin_boundaries = np.hstack([x_bin_boundaries, np.array(x_bin_boundaries[-1] + bin_width)])
#   # x_bin_boundaries[-1] += 1e-4
#   y_average_center = np.zeros(x_bin_center.shape[0])
#   y_std_center = np.zeros(x_bin_center.shape[0])

#   for i in np.arange(true_bins):
#      y_average_center[i] = data["D12"].loc[
#           #  (data["class"] == "-") &
#            (data["SLN"] >= x_bin_boundaries[i]) &
#            (data["SLN"] < x_bin_boundaries[i+1])
#         ].mean()
#      y_std_center[i] = data["D12"].loc[
#           #  (data["class"] == "-") &
#            (data["SLN"] >= x_bin_boundaries[i]) &
#            (data["SLN"] < x_bin_boundaries[i+1])
#         ].mean()

#   cmp = sns.color_palette("deep")
#   # plt.errorbar(x_bin_center, y_average_center, y_std_center, color=cmp[1], ecolor=cmp[1], capsize=5, capthick=2, fmt="o", alpha=0.7)

#   sns.scatterplot(
#      data=data,
#      x="SLN",
#      y="D12",
#     #  hue="class",
#      alpha=0.5
#   )

#   plt.plot(x_bin_center, y_average_center, color=cmp[1])
#   plt.scatter(x_bin_center, y_average_center, color=cmp[1])

#   ax=plt.gca()
#   ax.set_xlabel(r"$SLN\left( i,j \right)$")
#   ax.set_ylabel(r"$D_{1/2}^{-}\left(i,j \right)$")

#   plt.savefig(
#      f"{H.plot_path}/sln/scatter_plot_sln_d12.svg",
#      transparent=True
#   )

  A = get_structure_new(NET.csv_path)
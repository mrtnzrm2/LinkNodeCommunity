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
# plt.style.use("dark_background")
from pathlib import Path
# Personal libraries ----
from networks.structure import STR
from various.network_tools import *

# Declare global variables ----
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
imputation_method = ""
topology = "MIX"
discovery = "discovery_7"
mapping = "trivial"
index  = "Hellinger2"
bias = float(0)
alpha = 0.
version = "57"+"d"+"106"
__nodes__ = 57
__inj__ = 57

if __name__ == "__main__":
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

    neighbor_areas = {
        "v1c" : ["v2c", "v2pclf", "v1pcuf", "v2pcuf"],
        "v2c" : ["v1c", "v2pclf", "v2pcuf", "v3c", "v3pcuf", "v3pclf"],
        "v1pcuf" : ["v1c", "v1pclf", "v2pcuf", "v1fpuf", "v1fplf"],
        "v1pclf" : ["v1c", "v2pclf", "v1fplf", "v1pcuf", "v1fpuf", "v2fplf"],
        "v1fpuf" : ["v2fpuf", "v2pcuf", "v1pcuf", "v1fplf", "pro.st."],
        "v2pclf" : ["v1pclf", "v1c", "v2c", "v3pclf", "v3fplf", "v2fplf", "v1fplf"],
        "v1fplf" : ["v1pclf", "v2pclf", "v2fplf", "v1pcuf", "v1fpuf", "pro.st.", "23"],
        "v2fplf" : ["v6", "v3fplf", "v2pclf", "v1fplf", "v1pclf", "23"],
        "v2pcuf" : ["v2c", "v1c", "v1pcuf", "v2fpuf", "v1fpuf", "v3pcuf", "v3fpuf"],
        "v2fpuf" : ["v3fpuf", "v3pcuf", "v2pcuf", "v1fpuf", "th/tf", "subi"],
        "v3fpuf" : ["v4uf", "v3pcuf", "v2fpuf", "th/tf", "v2pcuf"],
        "v3pcuf" : ["v4uf", "v3c", "v2c", "v2pcuf", "v3fpuf", "v2fpuf"],
        "v4uf" : ["teo", "tepv", "th/tf", "v3fpuf", "v3pcuf", "v3c", "v4c"],
        "v4c" : ["teo", "teom", "v4uf", "v3c", "v2c", "v4lf", "v4t"],
        "v4lf" : ["v4c", "v4t", "v3c", "dp", "v3pclf", "pip", "v3a"],
        "v3c" : ["v4c", "v2c", "v3pcuf", "v4uf", "v4lf", "v3pclf"],
        "v3pclf" : ["v4lf", "v3c", "v2c", "v3a", "v3fplf", "v2pclf"],
        "v3fplf" : ["v3a", "v2pclf", "v3pclf", "v2fplf", "v6"],
        "v6" : ["v6a", "v3a", "v3fplf", "v2fplf", "23"],
        "v6a" : ["7m", "v6", "23", "mip", "pip", "v3a", "5"],
        "v3a" : ["pip", "v6a", "v3fplf", "v6", "v3pclf", "v4lf"],
        "v4t" : ["mtp", "mtc", "v4lf", "v4c", "teom", "dp"],
        "subi" : ["th/tf", "ento", "v2fpuf"],
        "th/tf" : ["tepv", "v4uf", "v3fpuf", "v2fpuf", "subi", "ento", "peri", "teav"],
        "tepv" : ["th/tf", "teav", "tead", "tepd", "teo", "v4uf"],
        "tepd" : ["tead", "tea/ma", "tea/mp", "teo", "tepv"],
        "teo" : ["tepd", "tea/mp", "teom", "v4c", "v4uf", "tepv"],
        "teom" : ["tea/mp", "fst", "mtc", "v4t", "v4c", "teo"],
        "mtc" : ["fst", "teom", "v4t", "mtp", "mst"],
        "mtp" : ["mst", "mtc", "v4t", "dp", "v4lf"],
        "dp" : ["mst", "7a", "lip", "pip", "v4lf", "mtp"],
        "pip" : ["lip", "dp", "v4lf", "v3a", "vip", "v6a", "v3pclf"],
        "23" : ["31", "7m", "29/30", "v6", "v2fplf", "v1fplf", "pro.st.", "3", "f1", "f3", "24d", "24b", "24a"],
        "ento" : ["peri", "th/tf", "subi", "pole"],
        "peri" : ["teav", "th/tf", "ento", "pole"],
        "teav" : ["peri", "pole", "tea/ma", "tead", "tepv", "th/tf"],
        "tead" : ["tea/ma", "tea/mp", "tepd", "tepv", "teav"],
        "tea/ma" : ["pole", "stpr", "ipa", "tea/mp", "tepd", "tead", "teav"],
        "tea/mp" : ["ipa", "fst", "teom", "teo", "tepd", "tead", "tea/ma"],
        "fst" : ["teom", "tea/mp", "ipa", "pga", "mst", "mtc"],
        "mst" : ["stpc", "7a", "dp", "mtp", "mtc", "fst", "pga"],
        "7a" : ["7b", "aip", "lip", "dp", "mst", "stpc", "tpt", "7op"],
        "lip" : ["aip", "vip", "pip", "dp", "7a", "7b"],
        "vip" : ["5", "mip", "pip", "lip", "aip"],
        "mip" : ["5", "v6a", "pip", "vip"],
        "7m" : ["5", "2", "1", "3", "23", "31", "v6a"],
        "31" : ["7m", "23"],
        "pro.st." : ["v1fpuf", "v1fplf", "23"],
        "29/30" : ["23"],
        "pole" : ["ento", "pi", "mb", "core", "lb", "pbr", "stpr", "tea/ma", "teav"],
        "stpr" : ["pole", "pbr", "stpi", "pga", "ipa", "tea/ma"],
        "ipa" : ["pga", "fst", "tea/mp", "tea/ma", "stpr"],
        "pga" : ["stpi", "stpc", "mst", "fst", "ipa", "stpr"],
        "stpi" : ["pbr", "pbc", "stpc", "pga", "stpr"],
        "stpc" : ["pbc", "tpt", "mst", "pga", "stpi", "7a"],
        "tpt" : ["7op", "7a", "stpc", "pbc", "lb", "mb", "insula"],
        "7op" : ["sii", "7b", "7a", "tpt", "insula"],
        "7b" : ["2", "aip", "lip", "7a", "7op", "sii"],
        "aip" : ["2", "5", "vip", "lip", "7b"],
        "5" : ["mip", "vip", "aip", "2", "7m", "v6a"],
        "pbr" : ["lb", "pbc", "stpi", "stpr", "pole"],
        "pbc" : ["lb", "tpt", "stpc", "stpi", "pbr"],
        "lb" : ["core", "mb", "tpt", "pbc", "pbr", "pole"],
        "core" : ["mb", "lb", "pole"],
        "mb" : ["insula", "pi", "tpt", "lb", "core", "pole"],
        "pi" : ["insula", "mb", "pole"],
        "insula" : ["sii", "gu", "opro", "pir", "pi", "mb", "7op", "tpt"],
        "sii" : ["7op", "insula", "gu", "prom", "2", "7b"],
        "2" : ["1", "5", "aip", "7b", "sii", "prom", "3", "7m"],
        "1" : ["3", "7m", "2"],
        "3" : ["f1", "23", "7m", "1", "2", "prom", "f5", "f4"],
        "pir" : ["opai", "opro", "insula"],
        "opro" : ["opai", "13", "12", "gu", "insula", "pir"],
        "gu" : ["opro", "12", "prom", "sii", "insula"],
        "prom" : ["f5", "3", "2", "gu", "12"],
        "f5" : ["44", "f4", "3", "prom", "12", "45a"],
        "f4" : ["f5", "44", "8l", "8m", "f2", "f1", "3"],
        "f1" : ["f2", "f3", "23", "3", "f4"],
        "f2" : ["f7", "f6", "f3", "f1", "f4", "8m", "8b"],
        "f3" : ["f6", "24d", "23", "f1", "f2", "f7"],
        "24d" : ["24c", "24b", "23", "f3", "f6"],
        "24b" : ["24c", "24d", "23", "32", "24a"],
        "24a" : ["24b", "32", "25", "23"],
        "opai" : ["13", "opro", "pir"],
        "13" : ["11", "12", "opro", "opai", "14"],
        "11" : ["12", "13", "14", "10"],
        "12" : ["45a", "46v", "11", "13", "opro", "gu", "prom", "f5", "10"],
        "45a" : ["9/46v", "8r", "45b", "44", "12", "f5"],
        "44" : ["45b", "8l", "f4", "f5", "45a"],
        "45b" : ["8l", "44", "45a", "8r"],
        "8l" : ["8r", "8m", "f4", "44", "45b"],
        "8m" : ["8r", "8b", "f2", "f4", "8l"],
        "8r" : ["9/46v", "9/46d", "8m", "8b", "8l", "45b", "45a"],
        "8b" : ["9", "24c", "f6", "f7", "f2", "8m", "8r", "9/46d"],
        "f7" : ["8b", "f6", "f3", "f2"],
        "f6" : ["8b", "24c", "24d", "f3", "f2", "f7"],
        "24c" : ["32", "24b", "24d", "f6", "8b", "9"],
        "46v" : ["10", "46d", "9/46d", "9/46v", "12"],
        "9/46v" : ["46v", "9/46d", "8r", "45a", "12"],
        "9/46d" : ["46d", "9", "8b", "8r", "9/46v", "46v"],
        "46d" : ["46v", "10", "9", "9/46d"],
        "9" : ["10", "32", "24c", "8b", "9/46d", "46d"],
        "10" : ["14", "32", "9", "46d", "46v", "12", "11"],
        "32" : ["14", "25", "24a", "24b", "24c", "9", "10"],
        "14" : ["13", "11", "25", "32", "10"],
        "25" : ["14", "24a", "32"]
    }

    labels = NET.struct_labels
    target = "opro"
    target_id = np.where(labels == target)[0][0]
    
    target_neighbors = neighbor_areas[target]
    target_neighbors_id = []
    for i, t in enumerate(target_neighbors):
        try:
          tid = np.where(labels == t)[0][0]
          if tid >= __inj__: continue
          else: target_neighbors_id.append(tid)
        except:
           raise ValueError("Target neighbor is missing in atlas.")

    target_neighbors_id = np.array(target_neighbors_id)
    neighbor_areas_number = target_neighbors_id.shape[0]

    dist_bins = 20
    target_prob_dist_true = np.zeros(dist_bins)
    target_prob_dist_inferred = np.zeros(dist_bins)

    target_neighbors_prob_dist = np.zeros((dist_bins, neighbor_areas_number))

    D_min = np.nanmin(NET.D[NET.D > 0])
    D_max = np.nanmax(NET.D)

    distance_bin_boundaries = np.linspace(D_min, D_max, dist_bins + 1)
    distance_bin_boundaries[-1] += 1e-6
    delta_D = distance_bin_boundaries[1] - distance_bin_boundaries[0]

    D = NET.D.copy()
    C = NET.CC.copy()
    for i in np.arange(NET.rows):
       if target_id != i:
        d = D[i, target_id]
        c = C[i, target_id]
        for j in np.arange(dist_bins):
            if distance_bin_boundaries[j] <= d and distance_bin_boundaries[j+1] > d:
               target_prob_dist_true[j] += c
               break
            
    target_prob_dist_true /= np.sum(target_prob_dist_true)
    # target_prob_dist_true /= delta_D
    
    for i, tid in enumerate(target_neighbors_id):
       for j in np.arange(NET.rows):
        if tid != j:
          d = D[j, tid]
          c = C[j, tid]
          for k in np.arange(dist_bins):
              if distance_bin_boundaries[k] <= d and distance_bin_boundaries[k+1] > d:
                target_neighbors_prob_dist[k, i] += c
                break
    
    target_neighbors_prob_dist = target_neighbors_prob_dist / np.sum(target_neighbors_prob_dist, axis=0)
    # target_neighbors_prob_dist /= delta_D

    #####


    ## CASE1 : product of probabilities and renormalization

   #  for i in np.arange(neighbor_areas_number):
   #     target_prob_dist_inferred += np.log(target_neighbors_prob_dist[:, i])
   #     target_prob_dist_inferred /= neighbor_areas_number

   #  target_prob_dist_inferred = np.exp(target_prob_dist_inferred)
   #  target_prob_dist_inferred /= np.sum(target_prob_dist_inferred)

    # Case 2: Product sampling

   #  samples = 100000
   #  target_prob_dist_gen = np.ones(samples)
   #  for i in np.arange(neighbor_areas_number):
   #     target_prob_dist_gen *= np.random.choice(
   #        distance_bin_boundaries[:-1] + delta_D/2, samples, p=target_neighbors_prob_dist[:, i]
   #      )
   #  target_prob_dist_gen = np.power(target_prob_dist_gen, 1. / neighbor_areas_number)
   #  target_prob_dist_gen = np.sort(target_prob_dist_gen)
   #  for dt in target_prob_dist_gen:
   #    for k in np.arange(dist_bins):
   #        if distance_bin_boundaries[k] <= dt and distance_bin_boundaries[k+1] > dt:
   #          target_prob_dist_inferred[k] += 1
   #          break

   #  target_prob_dist_inferred /= samples

    ## CASE 3: Random choice

    ### WEIGHTED BY DISTANCE OR UNIFORM SAMPLING
    nearest_neighbors_distaces_to_target = D[target_id, target_neighbors_id]
    weights = nearest_neighbors_distaces_to_target / np.sum(nearest_neighbors_distaces_to_target)
    # weights = np.array([1. / target_neighbors_id.shape[0]] * target_neighbors_id.shape[0])

    number_of_links = 100000
    nearest_neighbor_draw = np.random.choice(
       np.arange(target_neighbors_id.shape[0]), number_of_links, p=weights
    )
    from collections import Counter
    count_draw = Counter(nearest_neighbor_draw)
    for nn, num_nn in count_draw.items():
        didx = np.random.choice(np.arange(dist_bins), num_nn, p=target_neighbors_prob_dist[:, nn])
        count_didx = Counter(didx)
        for xd, vxd in count_didx.items():
            target_prob_dist_inferred[xd] += vxd

    target_prob_dist_inferred /= np.sum(target_prob_dist_inferred)
    
    D12 = [np.sqrt(p * q) for p, q in zip(target_prob_dist_true, target_prob_dist_inferred)]
    D12 = -2 * np.log(np.sum(D12))

    print(f"1/2 Renyi divergence: {D12:.4f}")

    data = pd.DataFrame(
       {
          "distances" : list(np.round(distance_bin_boundaries[:-1] + delta_D/2, 2)) * 2,
          "P" : list(target_prob_dist_true) + list(target_prob_dist_inferred),
          "set" : ["true"] * dist_bins + ["predicrted"] * dist_bins
       }
    )
    
    cmp = sns.color_palette("deep")
    sns.barplot(
      data=data,
      x="distances",
      y="P",
      hue="set",
      alpha=0.8,
    )

    plt.gca().set_yscale('log')
    plt.xticks(rotation = 90)
    plt.show()
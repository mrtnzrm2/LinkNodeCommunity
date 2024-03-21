# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ---- 
from scipy.cluster.hierarchy import cut_tree
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns
# Personal libs ---- 
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from modules.discovery import discovery_channel
from various.data_transformations import maps
from networks.structure import MAC40
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
subject = "MAC"
structure = "FLNe"
mode = "ZERO"
nature = "original"
# imputation_method = "RF2"
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0.
alpha = 0.
discovery = "discovery_7"
opt_score = ["_S"]
save_data = T
__nodes__ = 40
__inj__ = f"{__nodes__}"
version = f"{__nodes__}"+"d"+"91"
distance = "MAP3D"
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC40(
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

  NET.create_pickle_directory()
  NET.create_plot_directory()
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    NET.A, nlog10, lookup, prob, b=bias
  )
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
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
    ## Update entropy ----
    H.entropy = [
      H.node_entropy, H.node_entropy_H,
      H.link_entropy, H.link_entropy_H
    ]
    # Set labels to network ----
    L = colregion(NET, labels_name=f"labels{__nodes__}")
    H.set_colregion(L)
    # Save ----
    H.delete_dist_matrix()
    # save_class(
    #   H, NET.pickle_path,
    #   "hanalysis",
    #   on=T
    # )
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis"
    )

  N = __nodes__

  source = []
  target = []
  h = []
  b = []
  b2 = []
  connection_memberships = []
  connection_classification = []

  K, R, TH = get_best_kr_equivalence("_S", H)
  rlabels = get_labels_from_Z(H.Z, R[0])
  rlabels = skim_partition(rlabels)

  print(">>> Computnig hierarchy differences.")

  unique_labels = np.unique(rlabels)
  unique_labels = unique_labels[unique_labels != -1]
  Z = unique_labels.shape[0]
  membership_matrix = np.arange(Z**2).reshape(Z, Z)

  for direction in ["both"]:
    print("***", direction)
    NET.overlap, NET.data_nocs, noc_sizes, rlabels2  = discovery_channel[discovery](H, K[0], rlabels, direction=direction, index=index)
    cover = omega_index_format(rlabels2,  NET.data_nocs, NET.struct_labels[:NET.nodes])

  def cover_node_2_node_cover(cover : dict, labels):
    node_cover = {k: [] for k in labels}
    for k, vals in cover.items():
      for l in labels:
        if l in vals:
          node_cover[l].append(k)
    
    return node_cover
  
  beta = pd.read_csv(f"{NET.csv_path}/sln_beta_coefficients_40.csv", index_col=1).reindex(NET.struct_labels[:NET.nodes])
  beta = beta["beta"].to_numpy()

  print(cover)
  
  labels = NET.struct_labels

  h2_1merge = np.zeros(N)
  for i in np.arange(N):
    for k in np.arange(N-1, 0, -1):
      partition = cut_tree(H.Z, n_clusters=k).ravel()
      if np.sum(partition == partition[i]) > 1:
        h2i = H.Z[N - k, 2]
        break
    h2_1merge[i] = h2i

  for i in np.arange(N):
    for j in np.arange(N):
      if i == j : continue
      if NET.A[i, j] != 0:
        b.append(norm.cdf(beta[i] - beta[j]))
        b2.append(NET.SLN[i, j])
        h_diff = h2_1merge[i] - h2_1merge[j]
        h.append(h_diff)
        source.append(i)
        target.append(j)

  b = np.array(b)
  b2 = np.array(b2)
  h = np.array(h)
  source = np.array(source)
  target = np.array(target)

  from pathlib import Path
  Path(f"{NET.plot_path}/sln").mkdir(parents=True, exist_ok=True)


  data = pd.DataFrame({
    "eb_dist" : b,
    "b_dist" : b2,
    "h_dist" : h,
    # "source" : total_new_labels[source],
    # "target" : total_new_labels[target]
  })

  # print(data)

  ax = plt.gca()

  sns.scatterplot(
    data=data,
    x="h_dist",
    y="b_dist",
    s=50
  )


  from scipy.stats import pearsonr

  hax = data["h_dist"]
  bax = data["b_dist"]
  r, pval = pearsonr(hax, bax)
  pval = pvalue2asterisks(pval)
  ax.set_title(ax.get_title() + "\n" + fr"$\rho = {r:.2f}$   ({pval})")

  plt.gcf().tight_layout()
  ax.set_xlabel(r"$H^{2}_i - H^{2}_j$")
  # ax.set_xlabel("Estimated "+ r"$SLN\left(i,j\right)$")
  # ax.set_ylabel("Estimated "+ r"$SLN\left(i,j\right)$")
  ax.set_ylabel(r"$SLN\left(i,j\right)$")


  plt.savefig(
    f"{NET.plot_path}/sln/sln_asym_h2_1merging3.svg",
    # f"{NET.plot_path}/sln/esln_asym_h2_1merging3.svg",
    # f"{NET.plot_path}/sln/sln_esln_3.svg",

    transparent=True
  )
  

  plt.close()
  
  
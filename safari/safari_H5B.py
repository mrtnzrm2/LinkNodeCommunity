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
# from modules.sign.hierarmerge import Hierarchy
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from modules.discovery import discovery_channel
from various.data_transformations import maps, multiplexed_colnormalized_mapping, trivial_mapping
from networks.structure import MAC49
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
distance = "tracto16"
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
__nodes__ = 49
__inj__ = f"{__nodes__}"
version = f"{__nodes__}"+"d"+"106"
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC49(
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

  # Transform data for analysis ----
  # R, lookup, _ = multiplexed_colnormalized_mapping(0, NET.SN, NET.IN)
  R, lookup, _ = trivial_mapping(NET.A, nlog10, lookup, prob, b=bias)
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.A, R, NET.D,
      __nodes__, linkage, mode, lookup=lookup,
      # architecture="all"
    )
    ## Compute features ----
    H.BH_features_cpp_no_mu()
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Set labels to network ----
    L = colregion(NET, labels_name=f"labels{__nodes__}")
    H.set_colregion(L)

  N = NET.nodes
  SLN = NET.SLN

  source = []
  target = []
  h = []
  esln = []
  sln = []
  connection_memberships = []

  K, R, _ = get_best_kr_equivalence("_S", H)
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
  
  beta = pd.read_csv(f"{NET.csv_path}/sln_beta_coefficients_49.csv", index_col=1).reindex(NET.struct_labels[:NET.nodes])
  beta = beta["beta"].to_numpy()

  print(cover)
  
  labels = NET.struct_labels

  h2_1merge = np.zeros(N)
  for i in np.arange(N):
    for k in np.arange(N-1, 0, -1):
      partition = cut_tree(H.Z, n_clusters=k).ravel()
      if np.sum(partition == partition[i]) > 1:
        h2i = H.Z[N - k - 1, 2]
        break
    h2_1merge[i] = h2i

  print(cover)

  node_cover = cover_node_2_node_cover(cover, NET.struct_labels[:NET.nodes])
  labels = NET.struct_labels

  for i in np.arange(N):
    for j in np.arange(N):
      if i == j : continue
      if NET.A[i, j] != 0:
        esln.append(norm.cdf(beta[i] - beta[j]))
        sln.append(SLN[i, j])
        h_diff = h2_1merge[j] - h2_1merge[i]
        h.append(h_diff)
        source.append(i)
        target.append(j)

  esln = np.array(esln)
  sln = np.array(sln)
  h = np.array(h)

  h = h - np.min(h)
  h /= np.max(h)

  source = np.array(source)
  target = np.array(target)

  from pathlib import Path
  Path(f"{NET.plot_path}/sln").mkdir(parents=True, exist_ok=True)

  ne = sln.shape[0]

  empirical_sln_label = "Empirical " + r"$SLN(i,j)$"
  sln_bb_label = r"$SLN_{BB}(i,j)$"
  node_community_label = r"$\Delta \hat{S}(i,j)$"

  data = pd.DataFrame({
    empirical_sln_label : list(sln) * 2,
    "value" : list(esln) + list(h),
    "feature" : [sln_bb_label] * ne + [node_community_label] * ne
  })

  # print(data)
  sns.set_style("white")
  sns.set_context("talk")

  fig, axes = plt.subplots(2, 1)

  sns.scatterplot(
    data=data,
    x="value",
    y=empirical_sln_label,
    hue="feature",
    s=50,
    alpha=0.7,
    ax=axes[0]
  )

  # box = axes[0].get_position()
  # axes[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
  # axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

  from scipy.stats import pearsonr

  hax = data["value"].loc[data["feature"] == sln_bb_label]
  bax = data[empirical_sln_label].loc[data["feature"] == sln_bb_label]
  r, pval = pearsonr(hax, bax)
  print(pval)
  pval = pvalue2asterisks(pval)
  axes[0].set_title(r"$\rho_{BB} = $" + f"{r:.2f} ({pval})")
  # ax.set_title(fr"$\rho_{bb}= {r:.2f}$   ({pval})")

  hax = data["value"].loc[data["feature"] == node_community_label]
  bax = data[empirical_sln_label].loc[data["feature"] == node_community_label]
  r, pval = pearsonr(hax, bax)
  print(pval)
  pval = pvalue2asterisks(pval)
  
  axes[0].set_title(axes[0].get_title() + "\t" +r"$\rho_{S} = $" + f"{r:.2f}   ({pval})")

  sns.histplot(
    data=data,
    x="value",
    hue="feature",
    ax=axes[1]
  )

  from scipy.stats import kstest

  r = kstest(esln, h, N=h.shape[0], alternative="two-sided")

  print(r)
  
  axes[1].set_title(f"KS = {pvalue2asterisks(r.pvalue)}")


  fig.set_figwidth(7)
  fig.set_figheight(11)
  fig.tight_layout()

  # plt.gcf().tight_layout()
  # ax.set_xlabel(r"$H^{2}_i - H^{2}_j$")
  # ax.set_ylabel("Estimated "+ r"$SLN\left(i,j\right)$")
  # ax.set_ylabel(r"$SLN\left(i,j\right)$")


  plt.savefig(
    # f"{NET.plot_path}/sln/sln_asym_h2_1merging2.svg",
    f"{NET.plot_path}/sln/sln_and_models_5B.svg",

    transparent=True
  )
  # plt.show()

  # def pvalue2asterisks(pvalue):
  #   if  not np.isnan(pvalue): 
  #     if pvalue > 0.05:
  #       a = ""
  #     elif pvalue <= 0.05 and pvalue > 0.001:
  #       a = "*" 
  #     elif pvalue <= 0.001 and pvalue > 0.0001:
  #       a = "**" 
  #     else:
  #       a = "***"
  #   else:
  #     a = "nan"
  #   return a

  # average_sln_membership = np.zeros((Z,Z))

  # from scipy.stats import ttest_1samp, ttest_ind
  # sln_significance = np.array([""]* Z**2, dtype="<U21").reshape(Z,Z)

  # for zi in np.arange(Z):
  #   for zj in np.arange(Z):
  #     x = data["b_dist"].loc[data["group"] == membership_matrix[zi, zj].astype(int).astype(str)]
  #     average_sln_membership[zi, zj] = np.mean(x)
  #     if zi != zj:
  #       pval = ttest_1samp(x, 0.5).pvalue
  #       sln_significance[zi, zj] = pvalue2asterisks(pval)

  # plt.close()

  # fig, ax = plt.subplots(1,1)

  # annotate_sln = np.array([""]*Z**2, dtype="<U21")
  # for i, (av, pval) in enumerate(zip(average_sln_membership.ravel(), sln_significance.ravel())):
  #   annotate_sln[i] = f"{av:.2f}\n{pval}"

  # annotate_sln = annotate_sln.reshape(Z, Z)

  # import matplotlib

  # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b20000","#cca3ff","#0047AB"])

  # sns.heatmap(
  #   average_sln_membership,
  #   annot=annotate_sln,
  #   fmt="",
  #   cmap=cmap,
  #   alpha=0.7,
  #   center=0.5,
  #   ax=ax
  # )

  # plt.gcf().tight_layout()


  # # plt.savefig(
  # #   f"{NET.plot_path}/sln/average_sln_covers5.svg",
  # #   transparent=True
  # # )

  # plt.close()

  # g = sns.FacetGrid(
  #   data=data,
  #   col="group",
  #   col_wrap=Z
  # )

  # g.map_dataframe(
  #   sns.histplot,
  #   stat="density",
  #   x="b_dist",
  #   common_bins = False,
  #   common_norm = False
  # )

  # plt.gcf().tight_layout()

  # plt.savefig(
  #   f"{NET.plot_path}/sln/sln_hist_cover5.svg",
  #   transparent=True
  # )


 


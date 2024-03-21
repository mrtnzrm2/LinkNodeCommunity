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
from modules.sign.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from modules.colregion import colregion
from modules.hierarentropy import Hierarchical_Entropy
from modules.discovery import discovery_channel
from various.data_transformations import maps, multiplexed_colnormalized_mapping
from networks.structure import MAC49
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
subject = "MAC"
structure = "MULTIPLEX_SI"
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
  
  NET.get_supraneurons()
  NET.get_infraneurons()
  SN = NET.SupraC
  IN = NET.InfraC
  TOTN = SN + IN
  SLN = np.zeros((NET.rows, NET.nodes))
  SLN[TOTN > 0] = SN[TOTN > 0] / TOTN[TOTN > 0]

  # Transform data for analysis ----
  R, lookup, _ = multiplexed_colnormalized_mapping(0, SN.copy(), IN.copy())
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_data:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, NET.A, R, NET.D,
      __nodes__, linkage, mode, lookup=lookup,
      architecture="all"
    )
    ## Compute features ----
    H.BH_features_cpp_no_mu()
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    # Set labels to network ----
    L = colregion(NET, labels_name=f"labels{__nodes__}")
    H.set_colregion(L)

  N = __nodes__

  source = []
  target = []
  h = []
  b = []
  connection_memberships = []

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
  

  print(cover)
  # print(SLN)
  # cover[1] += ["v1fpuf"]

  node_cover = cover_node_2_node_cover(cover, NET.struct_labels[:NET.nodes])
  labels = NET.struct_labels

  h2_1merge = np.zeros(NET.nodes)
  for i in np.arange(NET.nodes):
    for k in np.arange(N-1, 0, -1):
      partition = cut_tree(H.Z, n_clusters=k).ravel()
      if np.sum(partition == partition[i]) > 1:
        h2i = H.Z[N - 1 - k, 2]
        break
    h2_1merge[i] = h2i

  cover_indices = {c: match(l, NET.struct_labels) for c, l in cover.items()}

  for c1, li1 in cover_indices.items():
    for c2, li2 in cover_indices.items():
      for i in li1:
        for j in li2:
          if i == j: continue
          if NET.A[i, j] != 0:
            connection_memberships.append(membership_matrix[c1, c2])
            b.append(SLN[i,j])
            h_diff = h2_1merge[i] - h2_1merge[j]
            h.append(h_diff)
            source.append(i)
            target.append(j)


  b = np.array(b)
  h = np.array(h)
  source = np.array(source)
  target = np.array(target)
  connection_memberships = np.array(connection_memberships).astype(int).astype(str)
  from pathlib import Path
  Path(f"{NET.plot_path}/sln").mkdir(parents=True, exist_ok=True)


  data = pd.DataFrame({
    "b_dist" : b,
    "h_dist" : h,
    "group" : connection_memberships,
  })

  trace_data_numeric = np.diag(membership_matrix).astype(int).astype(str)
  trace_data = data.loc[np.isin(data["group"], trace_data_numeric)]

  for new, old in enumerate(trace_data_numeric):
    trace_data["group"].loc[trace_data["group"] == old] = str(new)

  g=sns.FacetGrid(
    data=trace_data,
    col="group",
    col_wrap=Z // 2,
  )

  g.map_dataframe(
    sns.scatterplot,
    x="h_dist",
    y="b_dist"
  )


  from scipy.stats import pearsonr

  for ax in g.axes.flatten():
    title = ax.get_title().split(" = ")[-1]
    hax = trace_data["h_dist"].loc[trace_data["group"] == title]
    bax = trace_data["b_dist"].loc[trace_data["group"] == title]
    r, pval = pearsonr(hax, bax)
    p_val_trans = -np.floor(-np.log10(pval)).astype(int)
    if p_val_trans == 0:
      ax.set_title(ax.get_title() + "\n" + fr"$\rho = {r:.2f}$ with $p=n.s.$")
    else:
      ax.set_title(ax.get_title() + "\n" + fr"$\rho = {r:.2f}$ with $p<1E{p_val_trans}$")

  g.add_legend()

  plt.gcf().tight_layout()

  plt.savefig(
    f"{NET.plot_path}/sln/sln_asym_h2_1merging5.svg",
    transparent=True
  )

  plt.close()

  def pvalue2asterisks(pvalue):
    if  not np.isnan(pvalue): 
      if pvalue > 0.05:
        a = ""
      elif pvalue <= 0.05 and pvalue > 0.001:
        a = "*" 
      elif pvalue <= 0.001 and pvalue > 0.0001:
        a = "**" 
      else:
        a = "***"
    else:
      a = "nan"
    return a

  average_sln_membership = np.zeros((Z,Z))

  from scipy.stats import ttest_1samp, ttest_ind
  sln_significance = np.array([""]* Z**2, dtype="<U21").reshape(Z,Z)

  for zi in np.arange(Z):
    for zj in np.arange(Z):
      x = data["b_dist"].loc[data["group"] == membership_matrix[zi, zj].astype(int).astype(str)]
      average_sln_membership[zi, zj] = np.mean(x)
      if zi != zj:
        pval = ttest_1samp(x, 0.5).pvalue
        sln_significance[zi, zj] = pvalue2asterisks(pval)

  plt.close()

  fig, ax = plt.subplots(1,1)

  annotate_sln = np.array([""]*Z**2, dtype="<U21")
  for i, (av, pval) in enumerate(zip(average_sln_membership.ravel(), sln_significance.ravel())):
    annotate_sln[i] = f"{av:.2f}\n{pval}"

  annotate_sln = annotate_sln.reshape(Z, Z)

  import matplotlib

  cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b20000","#cca3ff","#0047AB"])

  sns.heatmap(
    average_sln_membership,
    annot=annotate_sln,
    fmt="",
    cmap=cmap,
    alpha=0.7,
    center=0.5,
    ax=ax
  )

  plt.gcf().tight_layout()


  # plt.savefig(
  #   f"{NET.plot_path}/sln/average_sln_covers5.svg",
  #   transparent=True
  # )

  plt.close()

  g = sns.FacetGrid(
    data=data,
    col="group",
    col_wrap=Z
  )

  g.map_dataframe(
    sns.histplot,
    stat="density",
    x="b_dist",
    common_bins = False,
    common_norm = False
  )

  plt.gcf().tight_layout()

  plt.savefig(
    f"{NET.plot_path}/sln/sln_hist_cover5.svg",
    transparent=True
  )


 


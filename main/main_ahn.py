# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
# Boolean aliases ----
T = True
F = False
# Personal libraries ---- 
from networks.structure import MAC
from modules.hierarmerge import Hierarchy
from various.network_tools import *
from plotting_modules.plotting_H import Plot_H

class AhnPlot(Plot_H):
  def __init__(self, Net, Hierarchy, sln=True) -> None:
    super().__init__(Net, Hierarchy, sln)

  def partition_density(self, D, k, on=True):
    if on:
      print("Plot Ahn partition density")
      d = np.zeros((len(D), 2))
      for i, key  in enumerate(D):
        d[i, 0] = key[0]
        d[i, 1] = key[1]
      data = pd.DataFrame(
        {
          "distance_feature": d[:, 0],
          "part_density": d[:, 1]
        }
      )
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data=data,
        x="distance_feature",
        y="part_density",
        ax=ax
      )
      fig.tight_layout()
      # Arrange path ----
      plot_path = os.path.join(self.path, "Feature")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "partition_density_{}.png".format(k)
        ),
        dpi = 300
      )
      plt.close()
    else:
      print("No partition density")
  
  def lcmap_pure(self, Pure, k, on=True, **kwargs):
    if on:
      print("Visualize pure LC memberships!!!")
      if "labels" in kwargs.keys():
        ids = kwargs["labels"]
        I, fq = sort_by_size(ids, self.nodes)
        flag_fq = True
      else:
        I = np.arange(self.nodes, dtype=int)
        fq = {}
        flag_fq = False
      if "name" in kwargs.keys():
        name = kwargs["name"]
      else: name = ""
      if "order" in kwargs.keys():
        I = kwargs["order"]
      # FLN to dataframe and filter FLN = 0 ----
      dFLN = Pure.copy()
      self.minus_one_Dc(dFLN)
      self.aesthetic_ids(dFLN)
      keff = np.unique(
        dFLN["id"].to_numpy()
      ).shape[0]
      # Transform dFLN to Adj ----
      dFLN = df2adj(dFLN, var="id")
      dFLN = dFLN[I, :][:, I]
      dFLN[dFLN == 0] = np.nan
      # dFLN = dFLN.T
      # Configure labels ----
      labels = self.colregion.labels[I]
      rlabels = [
        str(r) for r in self.colregion.regions[
          "AREA"
        ]
      ]
      colors = self.colregion.regions.loc[
        match(
          labels,
          rlabels
        ),
        "COLOR"
      ].to_numpy()
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      fig.set_figwidth(19)
      fig.set_figheight(15 * dFLN.shape[0]/ self.nodes)
      sns.heatmap(
        dFLN,
        xticklabels=labels[:self.nodes],
        yticklabels=labels,
        cmap=sns.color_palette("hls", keff + 1),
        ax = ax
      )
      # Setting labels colors ----
      [t.set_color(i) for i,t in
        zip(
          colors,
          ax.xaxis.get_ticklabels()
        )
      ]
      [t.set_color(i) for i,t in
        zip(
          colors,
          ax.yaxis.get_ticklabels()
        )
      ]
      # Add black lines ----
      if flag_fq:
        c = 0
        for key in fq:
          c += fq[key]
          if c < self.nodes:
            ax.vlines(
              c, ymin=0, ymax=self.nodes,
              colors=["black"]
            )
            ax.hlines(
              c, xmin=0, xmax=self.nodes,
              colors=["black"]
            )
      # Arrange path ----
      plot_path = os.path.join(self.path, "Matrix_pure")
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "k_{}{}.png".format(k, name)
        ),
        dpi = 300
      )
      plt.close()
    else:
      print("No pure logFLN heat map")


# Declare global variables ----
linkage = "single"
nlog10 = T
loopkup = T
mode = "ALPHA"
distance = "MAP3D"
nature = "original"
feature = "COSTARGETAHN"
imputation_method = ""
opt_score = "_maxmu"
save_data = T
save_wsbm = F
version = 220830
__nodes__ = 57
inj = 57
## Ahn's code ----
threshold = 1
dendro_flag = T
# Start main ----
if __name__ == "__main__":
  NET = MAC(
    linkage, mode, nlog10, loopkup,
    version = version,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = inj,
    feature=feature,
    sln=F
  )
  H = Hierarchy(
      NET, NET.A, NET.D,
      __nodes__, linkage, mode,
      nlog10=nlog10, lookup=loopkup,
      feature=feature,
  )
  rlabels = [""]
  # Compute features ----
  H.BH_features_cpp()
  ## Compute lq arbre de merde ----
  H.la_abre_a_merde_cpp(H.BH[0])
  # Get best K and R ----
  k, r = get_best_kr(opt_score, H)
  # Take a look in case of SLN ----
  rlabels = get_labels_from_Z(H.Z, r)
  ## Ahn ----
  # R = H.A.copy()[:inj, :inj]
  # np.fill_diagonal(R, np.nan)
  # R[R != 0] = -np.log10(R[R != 0])
  # R = 0.5 * R + 0.5 * R.T
  # R[R == 0] = np.nanmax(R) + 1
  from Ahn.python.link_clustering_org import HLCD, get_network_from_W_directed
  adj_in, adj_out,edges, ij2wij = get_network_from_W_directed(
    H.R[:inj, :inj], lookup=loopkup
  )
  edge2cid, S_max, D_max,list_D, orig_cid2edge, linkage = HLCD(
    adj_in, adj_out, edges
  ).single_linkage(
    dendro_flag=dendro_flag, w=ij2wij
  )
  # edge2cid, D_max = HLCD(
  #   adj_in, adj_out, edges
  # ).single_linkage(
  #   dendro_flag=dendro_flag, w=ij2wij, threshold=threshold
  # )
  dA = H.A[:inj, :inj].copy()
  # dA = dA + dA.T
  dAid = dA.copy()
  for key in edge2cid.keys():
    i, j = key
    dAid[i, j] = edge2cid[key]
    # dAid[j, i] = edge2cid[key]
  dA = adj2df(dA)
  dA["id"] = dAid.ravel()
  dA = dA.loc[dA.weight != 0]
  plot_h = AhnPlot(NET, H, sln=F)
  plot_h.partition_density(list_D, D_max, on=T)
  plot_h.lcmap_pure(dA, D_max, on=T, labels = rlabels)
  # print(len(linkage))

  #Notes:
  #Algorithm had a restrictive and efficient tree-like structure. It was restrictive because in its core, the
  # algorithm was made for undirected networks (swap, Dc). It also only contemplate source similarities, work only for
  # a edge-complete graph, and computing similarities between out-neighborhoods with present links without being able
  # to add weights to non-links just for similarity reasons. It also was not aware of the potential of using the
  # link communities algorithm from a matrix-like perspective and its topology freedom to choose neighbor links.


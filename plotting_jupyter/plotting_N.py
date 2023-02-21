# Standard libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
# Personal libs ----
from various.network_tools import *
from modules.hierarmerge import Hierarchy

class Plot_N:
  def __init__(self, NET, H : Hierarchy) -> None:
    # From net ----
    self.path = NET.plot_path
    # from Hierarchy ----
    self.nodes = H.nodes
    self.linkage = H.linkage
    self.A = H.A
    self.dA = H.dA
    self.D = H.D
    self.BH = H.BH
    self.R = H.R
    self.index = H.index
    # Data transformed ----
    self.aik = H.source_sim_matrix
    self.aki = H.target_sim_matrix

  def normal(self, mean, std, color="black"):
    from scipy import stats
    x = np.linspace(mean - (4 * std), mean + (4 * std), 200)
    p = stats.norm.pdf(x, mean, std)
    plt.plot(x, p, color, linewidth=2)

  # def SLN_histogram(self,  on=False):
  #   if on:
  #     print("Plot SLN histogram!!!")
  #     df = adj2df(self.sln)
  #     w = adj2df(self.A.copy())
  #     df["w"] = w.weight
  #     df = df.loc[df.w != 0]
  #     # Create figure ----
  #     _, ax = plt.subplots(1, 1)
  #     sns.histplot(
  #       data=df,
  #       x="weight",
  #       ax=ax
  #     )
  #     plt.xlabel("SLN")
  #     # Arrange path ----
  #     plot_path = os.path.join(
  #       self.path, "Features"
  #     )
  #     # Crate path ----
  #     Path(
  #       plot_path
  #     ).mkdir(exist_ok=True, parents=True)
  #     # Save plot ----
  #     plt.savefig(
  #       os.path.join(
  #         plot_path, "sln_histogram.png"
  #       ),
  #       dpi=300
  #     )
  #   else:
  #     print("No SLN histogram")

  def A_vs_dis(self, A, s=1, name="weight", on=True, **kwargs):
    if on:
      print("Plot {} vs dist!!!".format(name))
      # Get data ----
      dA = adj2df(A.copy())
      dD = adj2df(
        self.D.copy()[:, :self.nodes]
      )
      # Get No connections ---
      zeros = dA["weight"] == 0
      # Elllminate zeros ---
      dA = dA.loc[~zeros]
      dD = dD.loc[~zeros]
      # Create data ----
      from pandas import DataFrame
      data = DataFrame(
        {
          "weight" : np.log(dA["weight"]),
          "dist" : dD["weight"]
        }
      )
      # Get slop and p-value ----
      model = linear_fit(
        data["dist"].to_numpy(),
        data["weight"].to_numpy()
      )
      x = np.linspace(
        np.min(self.D[self.D > 0]),
        np.max(self.D),
        100
      )
      # Create figure ----
      _ , ax = plt.subplots(1, 1)
      sns.scatterplot(
        data=data,
        x="dist",
        y="weight",
        s=s
      )
      if "reg" in kwargs.keys():
        # x_st = (x - np.mean(x)) / np.std(x)
        if kwargs["reg"]:
          sns.lineplot(
            x=x,
            y=model.predict(x.reshape(-1, 1)),
            color="r"
          )
      ax.text(
        0.5, 1.05,
        s = "m={:.4f}".format(
          model.coef_[0]
        ),
        ha='center', va='center',
        transform=ax.transAxes
      )
      plt.ylabel("log(FLN)")
      plt.xlabel("dist [mm]")
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "{}_vs_dist.png".format(name)
        ),
        dpi=300
      )
    else:
      print("No {} vs dist".format(name))

  def histogram_dist(self, on=True):
    if on:
      print("Plot distance histogram!!!")
      # Transform FLN to DataFrame ----
      dD = adj2df(
        self.D[:, :self.nodes].copy()
      )
      dD = dD.loc[dD.source > dD.target]
      # Create figure
      fig, ax = plt.subplots(1, 1)
      sns.histplot(
        data=dD,
        x="weight",
        stat="density"
      )
      plt.xlabel("distance [mm]")
      plt.ylabel(r"$q(d)\,(mm^{-1})$")
      fig.tight_layout()
      # self.normal(
      #   dD["weight"].to_numpy().mean(),
      #   dD["weight"].to_numpy().std()
      # )
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "dist_hist.png"
        ),
        dpi=300
      )
    else:
      print("No histogram")

  # def projection_probability_sln(self, bins="auto", on=True):
  #   if on:
  #     print("Projection probability by SLN")
  #     D = self.D.copy()[:, :self.nodes]
  #     supra = self.supra.copy()[:, :self.nodes]
  #     infra = self.infra.copy()[:, :self.nodes]
  #     # Create data ----
  #     data_supra = np.zeros(np.sum(supra))
  #     data_infra = np.zeros(np.sum(infra))
  #     count_supra = 0
  #     count_infra = 0
  #     for u in np.arange(D.shape[0]):
  #       for v in np.arange(D.shape[1]):
  #         if supra[u, v] != 0:
  #           data_supra[count_supra:(count_supra + supra[u, v])] = D[u, v]
  #           count_supra += supra[u, v]
  #         if infra[u, v] != 0:
  #           data_infra[count_infra:(count_infra + infra[u, v])] = D[u, v]
  #           count_infra += infra[u, v]
  #     data = pd.DataFrame(
  #       {
  #         "type" : ["supra"] * len(data_supra) + ["infra"] * len(data_infra),
  #         "distance" : np.hstack(
  #           [data_supra, data_infra]
  #         )
  #       }
  #     )
  #     # Create figures ----
  #     fig, ax = plt.subplots(1, 1)
  #     sns.histplot(
  #       data=data,
  #       x="distance",
  #       hue="type",
  #       stat="density",
  #       bins=bins,
  #       alpha=0.5,
  #       ax=ax
  #     )
  #     ax.set_yscale("log")
  #     fig.tight_layout()
  #     # Arrange path ----
  #     plot_path = os.path.join(
  #       self.path, "Features"
  #     )
  #     # Crate path ----
  #     Path(
  #       plot_path
  #     ).mkdir(exist_ok=True, parents=True)
  #     # Save ----
  #     plt.savefig(
  #       os.path.join(
  #         plot_path, "projection_p_sln.png"
  #       ),
  #       dpi=200
  #     )
          
  #   else:
  #     print("No projection probability by SLN")

  def projection_probability(
    self, C, bins=20, on=True, **kwargs
  ):
    if on:
      print("Plot neuron-distance freq!!!")
      # Distance range ----
      _, x, y = range_and_probs_from_DC(
        self.D, C, self.nodes, bins
      )
      from pandas import DataFrame
      y = np.exp(y)
      _, x_range_model, _, _, model = predicted_D_frequency(
        self.D, C, self.nodes, bins, **kwargs
      )
      y_pred = 0
      y_pred = model.predict(
        x.reshape(-1, 1)
      )
      y_pred = np.exp(y_pred)
      data = DataFrame(
        {
          "dist" : np.round(x, 1).astype(str),
          "prob" : y,
          "pred" : y_pred
        }
      )
      data = data.loc[
        data["prob"] > 0
      ]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      #  Plot data ----
      cmp = sns.color_palette("deep")
      sns.barplot(
        data=data,
        x="dist",
        y="prob",
        color=cmp[0],
        alpha=0.8,
        ax=ax
      )
      sns.scatterplot(
        data=data,
        x="dist",
        y="prob",
        color="black",
        ax=ax
      )
      # Plot prediction ----
      ## Plot pred ----
      label="linear model"
      sns.lineplot(
        data=data,
        linewidth=2,
        x="dist",
        y='pred',
        color="r",
        label=label,
        ax=ax
      )
      ax.text(
          0.5, 1.05,
          s="linear coeff: {:.5f}".format(
            model.coef_[0]
          ),
          ha='center', va='center',
          transform=ax.transAxes
      )
      ax.set_yscale('log')
      ax.legend()
      # Aesthetics ----
      ax.set_xlabel("dist [mm]")
      ax.set_ylabel("p(d) [1/mm]")
      ## Fig size ----
      fig.set_figheight(6)
      fig.set_figwidth(8)
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Features", "projection_p"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      if "reg" in kwargs.keys():
        if kwargs["reg"] == "poly":
          plt.savefig(
            os.path.join(
              plot_path, "poly_bin_{}_deg_{}.png".format(
                bins, kwargs["deg"]
              )
            ),
            dpi=200
          )
        elif kwargs["reg"] == "piece_poly":
          plt.savefig(
            os.path.join(
              plot_path, "piece_poly_bin_{}.png".format(bins)
            ),
            dpi=200
          )
        elif kwargs["reg"] == "gp":
          plt.savefig(
            os.path.join(
              plot_path, "gp_bin_{}.png".format(bins)
            ),
            dpi=200
          )
      else:
        plt.savefig(
          os.path.join(
            plot_path, "bin_{}.png".format(bins)
          ),
          dpi=200
        )
    else:
      print("No histogram")

  def histogram_weight(self, feature="", on=True):
    if on:
      print("Plot weight histogram!!!")
      # Transform FLN to DataFrame ----
      dA = self.dA.copy()
      # Transform FLN to weights ----
      dA["weight"] = np.log(dA["weight"])
      fig, ax = plt.subplots(1, 1)
      sns.histplot(
        data=dA,
        x="weight",
        stat="density"
      )
      plt.xlabel(r"$\log(FLN)$")
      plt.ylabel(r"density $([\log(FLN)]^{-1})$")
      fig.tight_layout()
      # self.normal(
      #   dA["weight"].to_numpy().mean(),
      #   dA["weight"].to_numpy().std()
      # )
      # Arrange path ----
      plot_path = os.path.join(
        self.path,"Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "weight_histo.png"
        ),
        dpi=300
      )
    else:
      print("No histogram")

  def get_color_ids(self, dA, palette="husl"):
    ids = np.sort(
      np.unique(dA["id"].to_numpy())
    )
    n_ids = len(ids)
    cmap = sns.color_palette(palette, n_ids)
    colors = {}
    for i in np.arange(n_ids):
      colors[ids[i]] = cmap[i]
    return colors

  # def ellipse_H(self, A, K, on=True):
  #   if on:
  #     print("Plot hierarchical ellipse!!!")
  #     # Get H ----
  #     H = self.BH[0]
  #     # K loop ----
  #     for k in K:
  #       # From K to N ----
  #       n = H.loc[
  #         H["K"] == k, "NAC"
  #       ].to_numpy().astype(int)
  #       if len(n) < 1: continue
  #       if len(n) > 1: n = np.min(n)
  #       # Get nodes memberships ----
  #       from modules.charts import Charts
  #       CH = Charts(self.Hierarchy).hierarchical_chart(n)
  #       from matplotlib.patches import Ellipse
  #       # Create data ----
  #       from pandas import DataFrame
  #       dA = DataFrame(
  #         {
  #           "x" : A[:, 0],
  #           "y" : A[:, 1],
  #           "id" : CH
  #         }
  #       )
  #       # Create figure ----
  #       fig, axes = plt.subplots(1, 1)
  #       # Plot points ----
  #       sns.scatterplot(
  #         data=dA.sort_values("id"),
  #         x="x", y="y", hue="id",
  #         palette=self.get_color_ids(dA)
  #       )
  #       plt.legend(
  #         bbox_to_anchor=(1.02, 1),
  #         loc='upper left', borderaxespad=0
  #       )
  #       # Draw ellipse ----
  #       axes.add_patch(
  #         Ellipse(
  #           (0, 0), self.net.w, self.net.h,
  #           edgecolor = "b", facecolor = "none"
  #         )
  #       )
  #       # Annotate node label ----
  #       for i in np.arange(A.shape[0]):
  #         plt.text(
  #           A[i, 0] + np.random.uniform(-1.5, 1.5),
  #           A[i, 1] + np.random.uniform(-1.5, 1.5),
  #           i,
  #           fontsize=5
  #         )
  #       # Fix limits ----
  #       plt.xlim([-self.net.w / 2, self.net.w / 2])
  #       plt.ylim([-self.net.h / 2, self.net.h / 2])
  #       # Set size ----
  #       fig.set_figwidth(10)
  #       fig.set_figheight(6)
  #       # Arrange path ----
  #       plot_path = os.path.join(
  #         self.path, "map", "Hierarchy_color"
  #       )
  #       # Crate path ----
  #       Path(
  #         plot_path
  #       ).mkdir(exist_ok=True, parents=True)
  #       # Save plot ----
  #       plt.savefig(
  #         os.path.join(
  #           plot_path, "2d_nac_{}_k_{}.png".format(n[0], k)
  #         ),
  #         dpi=300
  #       )
  #   else:
  #     print("No hierarchical ellipse")

  # def ellipse(self, A, on=True):
  #   if on:
  #     print("Print ellipse!!!")
  #     from matplotlib.patches import Ellipse
  #     # Create figure ----
  #     _, axes = plt.subplots(1, 1)
  #     # Plot points ----
  #     sns.scatterplot(
  #       x = A[:, 0], y = A[:, 1]
  #     )
  #     # Draw ellipse ----
  #     axes.add_patch(
  #       Ellipse(
  #         (0, 0), self.net.w, self.net.h,
  #         edgecolor = "b", facecolor = "none"
  #       )
  #     )
  #     # Annotate node label ----
  #     for i in np.arange(A.shape[0]):
  #       plt.text(
  #         A[i, 0] + np.random.uniform(-1, 1),
  #         A[i, 1] + np.random.uniform(-1, 1),
  #         i,
  #         fontsize=5
  #       )
  #     # Fix limits ----
  #     plt.xlim([-self.net.w / 2, self.net.w / 2])
  #     plt.ylim([-self.net.h / 2, self.net.h / 2])
  #     # Arrange path ----
  #     plot_path = os.path.join(self.path, "map")
  #     # Crate path ----
  #     Path(
  #       plot_path
  #     ).mkdir(exist_ok=True, parents=True)
  #     # Save plot ----
  #     plt.savefig(
  #       os.path.join(
  #         plot_path, "2d.png"
  #       ),
  #       dpi=300
  #     )
  #   else:
  #     print("No ellipse")
  
  # def ellipse_average(self, wsbm, A, K, R, ids, on=True):
  #   if on:
  #     print("Plot average ellipse!!!")
  #     # K loop ----
  #     for k in K:
  #       # Get nodes memberships ----
  #       from modules.charts import Charts
  #       CH = Charts(self.Hierarchy)
  #       CH = CH.average_best_chart(k, R, ids).astype(str)
  #       from matplotlib.patches import Ellipse
  #       # Create data ----
  #       from pandas import DataFrame
  #       dA = DataFrame(
  #         {
  #           "x" : A[:, 0],
  #           "y" : A[:, 1],
  #           "id" : CH
  #         }
  #       )
  #       # Create figure ----
  #       fig, axes = plt.subplots(1, 1)
  #       # Plot points ----
  #       sns.scatterplot(
  #         data=dA.sort_values("id"),
  #         x="x", y="y", hue="id",
  #         palette=self.get_color_ids(dA)
  #       )
  #       plt.legend(
  #         bbox_to_anchor=(1.02, 1),
  #         loc='upper left', borderaxespad=0
  #       )
  #       # Draw ellipse ----
  #       axes.add_patch(
  #         Ellipse(
  #           (0, 0), self.net.w, self.net.h,
  #           edgecolor = "b", facecolor = "none"
  #         )
  #       )
  #       # Annotate node label ----
  #       for i in np.arange(A.shape[0]):
  #         plt.text(
  #           A[i, 0] + np.random.uniform(-1.5, 1.5),
  #           A[i, 1] + np.random.uniform(-1.5, 1.5),
  #           i,
  #           fontsize=5
  #         )
  #       # Fix limits ----
  #       plt.xlim([-self.net.w / 2, self.net.w / 2])
  #       plt.ylim([-self.net.h / 2, self.net.h / 2])
  #       # Set size ----
  #       fig.set_figwidth(10)
  #       fig.set_figheight(6)
  #       # Arrange path ----
  #       plot_path = os.path.join(
  #         self.path, "map", "Average_color", "K_{}".format(k)
  #       )
  #       # Crate path ----
  #       Path(
  #         plot_path
  #       ).mkdir(exist_ok=True, parents=True)
  #       # Save plot ----
  #       plt.savefig(
  #         os.path.join(
  #           plot_path, "2d_R_{}.png".format(R)
  #         ),
  #         dpi=300
  #       )
  #   else:
  #     print("No hierarchical ellipse")
  
  def plot_akis(self, D, s=1, on=True):
    if on:
      from scipy.stats import pearsonr, linregress
      print("Plot similarity plots and distance!!!")
      # Transform data to dataframe ----
      dD = adj2df(
        D.copy()[:self.nodes, :self.nodes]
      )
      daki = self.aki.copy()
      daki[np.isnan(daki)] = np.nanmin(daki) - 1
      daki = adj2df(daki)
      daki = daki.loc[
        daki["source"] < daki["target"], "weight"
      ].to_numpy().ravel()
      daik = self.aik.copy()
      daik[np.isnan(daik)] = np.nanmin(daik) - 1
      daik = adj2df(daik)
      daik = daik.loc[
        daik["source"] < daik["target"], "weight"
      ].to_numpy().ravel()
      # Filter dataframes ----
      dD = dD.loc[
        dD["source"] < dD["target"], "weight"
      ].to_numpy().ravel()
      # Create data ----
      from pandas import DataFrame
      data = DataFrame(
        {
          "dist" : dD,# / np.max(dD["weight"]),
          "target similarity" : daki,
          "source similarity" : daik
        }
      )
      # Create figures ----
      fig, ax = plt.subplots(1, 3)
      sns.scatterplot(
        data=data, x="dist", y="target similarity",
        s=s, ax=ax[0]
      )
      ## Compute stats ----
      data_cor = pearsonr(
        data["dist"], data["target similarity"]
      )
      _, _, data_r2, _, _ = linregress(
        data["dist"], data["target similarity"]
      )
      ax[0].text(
        x = 0.5, y = 1.05,
        s = "{:.5f} {:.5f}".format(
          data_cor[0], data_r2 ** 2
        ),
        ha='center', va='center',
        transform=ax[0].transAxes
      )
      sns.scatterplot(
        data=data, x="dist", y="source similarity",
        s=s, ax=ax[1]
      )
      ## Compute stats ----
      data_cor = pearsonr(
        data["dist"], data["source similarity"]
      )
      _, _, data_r2, _, _ = linregress(
        data["dist"], data["source similarity"]
      )
      ax[1].text(
        x = 0.5, y = 1.05,
        s = "{:.5f} {:.5f}".format(
          data_cor[0], data_r2 ** 2
        ),
        ha='center', va='center',
        transform=ax[1].transAxes
      )
      sns.scatterplot(
        data=data,
        x="source similarity", y="target similarity",
        s=s, ax=ax[2]
      )
      ## Compute stats ----
      data_cor = pearsonr(
        data["source similarity"],
        data["target similarity"]
      )
      _, _, data_r2, _, _ = linregress(
        data["source similarity"], data["target similarity"]
      )
      ax[2].text(
        x = 0.5, y = 1.05,
        s = "{:.5f} {:.5f}".format(
          data_cor[0], data_r2 ** 2
        ),
        ha='center', va='center',
        transform=ax[2].transAxes
      )
      fig.set_figwidth(15)
      fig.tight_layout()
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "similarity_plots_{}.png".format(self.linkage)
        ),
        dpi=300
      )
    else:
      print("No similarities and distance plots")

  # def plot_sln_akis(self, D, s=1, nlog10=True, on=True):
  #   if on:
  #     from scipy.stats import pearsonr, linregress
  #     print("Plot sln similarity plots and distance!!!")
  #     # Transform data to dataframe ----
  #     dD = adj2df(
  #       D.copy()[:self.nodes, :self.nodes]
  #     )
  #     daki = adj2df(self.target_similarity(self.sln, np.min(self.sln), self.nodes))
  #     daik = adj2df(self.source_similarity(self.sln, np.min(self.sln), self.nodes))
  #     # Filter dataframes ----
  #     dD = dD.loc[dD["source"] < dD["target"]]
  #     daik = daik.loc[daik["source"] < daik["target"]]
  #     daki = daki.loc[daki["source"] < daki["target"]]
  #     # Create data ----
  #     from pandas import DataFrame
  #     if nlog10:
  #       data = DataFrame(
  #         {
  #           "dist" : dD["weight"],# / np.max(dD["weight"]),
  #           "sln target similarity" : np.log10(daki["weight"]),
  #           "sln source similarity" : np.log10(daik["weight"])
  #         }
  #       )
  #       data = data.loc[
  #         (data["sln target similarity"] > -np.inf) &
  #         (data["sln source similarity"] > -np.inf)
  #       ]
  #     else:
  #       data = DataFrame(
  #           {
  #             "dist" : dD["weight"],# / np.max(dD["weight"]),
  #             "sln target similarity" : daki["weight"],
  #             "sln source similarity" : daik["weight"]
  #           }
  #         )
  #     # Create figures ----
  #     fig, ax = plt.subplots(1, 3)
  #     sns.scatterplot(
  #       data=data, x="dist", y="sln target similarity",
  #       s=s, ax=ax[0]
  #     )
  #     ## Compute stats ----
  #     data_cor = pearsonr(
  #       data["dist"], data["sln target similarity"]
  #     )
  #     _, _, data_r2, _, _ = linregress(
  #       data["dist"], data["sln target similarity"]
  #     )
  #     ax[0].text(
  #       x = 0.5, y = 1.05,
  #       s = "{:.5f} {:.5f}".format(
  #         data_cor[0], data_r2 ** 2
  #       ),
  #       ha='center', va='center',
  #       transform=ax[0].transAxes
  #     )
  #     sns.scatterplot(
  #       data=data, x="dist", y="sln source similarity",
  #       s=s, ax=ax[1]
  #     )
  #     ## Compute stats ----
  #     data_cor = pearsonr(
  #       data["dist"], data["sln source similarity"]
  #     )
  #     _, _, data_r2, _, _ = linregress(
  #       data["dist"], data["sln source similarity"]
  #     )
  #     ax[1].text(
  #       x = 0.5, y = 1.05,
  #       s = "{:.5f} {:.5f}".format(
  #         data_cor[0], data_r2 ** 2
  #       ),
  #       ha='center', va='center',
  #       transform=ax[1].transAxes
  #     )
  #     sns.scatterplot(
  #       data=data,
  #       x="sln source similarity", y="sln target similarity",
  #       s=s, ax=ax[2]
  #     )
  #     ## Compute stats ----
  #     data_cor = pearsonr(
  #       data["sln source similarity"],
  #       data["sln target similarity"]
  #     )
  #     _, _, data_r2, _, _ = linregress(
  #       data["sln source similarity"], data["sln target similarity"]
  #     )
  #     ax[2].text(
  #       x = 0.5, y = 1.05,
  #       s = "{:.5f} {:.5f}".format(
  #         data_cor[0], data_r2 ** 2
  #       ),
  #       ha='center', va='center',
  #       transform=ax[2].transAxes
  #     )
  #     fig.set_figwidth(15)
  #     fig.tight_layout()
  #     # Arrange path ----
  #     plot_path = os.path.join(
  #       self.path, "Features"
  #     )
  #     # Crate path ----
  #     Path(
  #       plot_path
  #     ).mkdir(exist_ok=True, parents=True)
  #     # Save plot ----
  #     plt.savefig(
  #       os.path.join(
  #         plot_path, "sln_similarity_plots_{}.png".format(self.linkage)
  #       ),
  #       dpi=300
  #     )
  #   else:
  #     print("No sln similarities and distance plots")

  def plot_aki(self, s=1, on=True):
    if on:
      print("Plot aki-aik!!!")
      daki = adj2df(self.aki.copy())
      daik = adj2df(self.aik.copy())
      # Filter dataframes ----
      daik = daik.loc[daik["source"] < daik["target"]]
      daki = daki.loc[daki["source"] < daki["target"]]
      # Create data ----
      from pandas import DataFrame
      data = DataFrame(
        {
          "target similarity" : daki["weight"],
          "source similarity" : daik["weight"]
        }
      )
      # Create figures ----
      fig, _ = plt.subplots(1, 1)
      sns.scatterplot(
        data=data,
        x="source similarity", y="target similarity",
        s=s
      )
      fig.set_figwidth(5)
      fig.tight_layout()
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "sim_plot_{}.png".format(self.linkage)
        ),
        dpi=300
      )
    else:
      print("No aki-aik")
  
  def plot_histoscatter(self, A, partition, on=False, **kwargs):
    if on:
      print("Plot histoscatter for A!!!")
      # Get new partition
      new_partition = skim_partition(partition)
      _, fq = sort_by_size(new_partition, self.nodes)
      print(fq)
      new_partition_2 = np.zeros(len(new_partition), dtype=int)
      c = 0
      for k in fq.keys():
        if k != -1:
          new_partition_2[new_partition == k] = c
          c += 1
        else:
          new_partition_2[new_partition == k] = k
      ##
      dA = adj2df(A)
      dA = dA.loc[dA.weight != 0]
      dA.weight = np.log(dA.weight)
      dA["community"] = "off-diagonal"
      unique_communities = np.unique(new_partition_2)
      # Assign id to edges
      for com in unique_communities:
        if com != -1: name_com = f"cluster_{com}"
        else: name_com = "overlapping_nodes"
        nodes_com = np.where(new_partition_2 == com)[0]
        n_com = len(nodes_com)
        for i in np.arange(n_com):
          for j in np.arange(i+1, n_com):
            dA.community.loc[
              (dA.source == nodes_com[i]) & (dA.target == nodes_com[j])
            ] = name_com
            dA.community.loc[
              (dA.target == nodes_com[i]) & (dA.source == nodes_com[j])
            ] = name_com
      dA = dA.sort_values(by="community", ignore_index=True)
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.histplot(
        data=dA,
        x="weight",
        hue="community",
        element="poly",
        stat="density",
        fill=False,
        common_norm=True,
        alpha=0.5,
        ax=ax
      )
      ax.set_yscale("log")
      sns.move_legend(ax, "upper left", bbox_to_anchor=(1.03, 1))
      fig.tight_layout()
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "histoscatter_fln.png"
        ),
        dpi=300
      )
    else: print("No histoscatter") 

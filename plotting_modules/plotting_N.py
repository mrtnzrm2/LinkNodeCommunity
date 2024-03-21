# Standard libs ----
import networkx as nx
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import numpy.typing as npt
import os
# Personal libs ----
from various.network_tools import *
from modules.hierarmerge import Hierarchy

class Plot_N:
  def __init__(self, NET, H : Hierarchy) -> None:
    # From net ----
    self.path = NET.plot_path
    self.labels = NET.struct_labels
    # from Hierarchy ----
    self.nodes = H.nodes
    self.edges = H.leaves
    self.linkage = H.linkage
    self.A = H.A
    self.dA = H.dA
    self.D = H.D
    self.BH = H.BH
    self.R = H.R
    self.index = H.index
    self.H = H.H
    # Data transformed ----
    self.aik = H.source_sim_matrix
    self.aki = H.target_sim_matrix

  def normal(self, mean, std, color="black"):
    from scipy import stats
    x = np.linspace(mean - (4 * std), mean + (4 * std), 200)
    p = stats.norm.pdf(x, mean, std)
    plt.plot(x, p, color, linewidth=2)


  def A_vs_dis(self, A, s=1, name="weight", on=True, **kwargs):
    if on:
      print("Plot {} vs dist!!!".format(name))
      from various.fit_tools import linear_fit
      # Get data ----
      dA = adj2df(A.copy())
      dD = adj2df(
        self.D.copy()[:, :self.nodes]
      )
      # Get No connections ---
      zeros = dA.weight == 0
      isnan = np.isnan(dA.weight)
      isinf = (dA.weight == np.Inf) | ( dA.weight == -np.Inf)
      # Elllminate zeros ---
      dA = dA.loc[(~zeros) & (~isnan) & (~isinf)]
      dD = dD.loc[(~zeros) & (~isnan) & (~isinf)]
      # Create data ----
      from pandas import DataFrame
      data = DataFrame(
        {
          "weight" : dA["weight"],
          "dist" : dD["weight"]
        }
      )
      # # Get slop and p-value ----
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
      # sns.set_context("talk")
      _ , ax = plt.subplots(1, 1)
      sns.scatterplot(
        data=data,
        x="dist",
        y="weight",
        alpha=0.8,
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

      plt.xlabel("inteareal physical distances [mm]")
      # plt.ylabel(r"$\log(FLNe)$")
      plt.ylabel("weight")
      plt.tight_layout()
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
          plot_path, "{}_vs_dist.svg".format(name)
        ),
        # dpi=300
      )
    else:
      print("No {} vs dist".format(name))

  def histogram_dist(self, on=True):
    if on:
      print("Plot distance histogram!!!")
      sns.set_style("whitegrid")
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
      plt.xlabel("fiber distance [mm]")
      # plt.ylabel(r"$q(d)\,(mm^{-1})$")
      
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
      print(plot_path)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, "dist_hist.png"
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No histogram")

  def projection_probability(
    self, C, model ,bins=20, on=True, **kwargs
  ):
    if on:
      print("Plot neuron-distance freq!!!")
      # Distance range ----
      from pandas import DataFrame
      from various.fit_tools import fitters
      _, x, y = range_and_probs_from_DC(self.D, C, bins)
      _, _, _, _, est = fitters[model](self.D, C, bins, **kwargs)
      y_pred = est.predict(
        x.reshape(-1, 1)
      )
      y_pred = np.exp(y_pred)
      data = DataFrame(
        {
          "dist" : np.round(x, 1).astype(str),
          "prob" : y.ravel(),
          "pred" : y_pred.ravel()
        }
      )
      data = data.loc[
        data["prob"] > 0
      ]
      # Create figure ----
      # sns.set_context("talk")
      sns.set_style("whitegrid")
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
      label="Exponential dist. (MLE)"
      sns.lineplot(
        data=data,
        linewidth=2,
        x="dist",
        y='pred',
        color="r",
        label=label,
        ax=ax
      )
      # ax.text(
      #     0.5, 1.05,
      #     s="linear coeff: {:.5f}".format(
      #       model.coef_[0]
      #     ),
      #     ha='center', va='center',
      #     transform=ax.transAxes
      # )
      ax.set_yscale('log')
      ax.legend()
      # Aesthetics ----
      ax.set_xlabel("interareal tractography distance [mm]")
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
            plot_path, "bin_{}_{}.png".format(bins, model)
          ),
          dpi=200
        )
      plt.close()
    else:
      print("No histogram")

  def histogram_weight(self, A, label="", on=True):
    if on:
      print("Plot weight histogram!!!")
      # Transform FLN to DataFrame ----
      dA = A.copy()
      dA[dA == -np.Inf] = 0
      dA = adj2df(A)
      dA["connection"] = "exist"
      dA.connection.loc[dA.weight == 0] = "~exist"
      # Transform FLN to weights ----

      # sns.set_style("whitegrid")
      # plt.style.use("dark_background")
      # sns.set_context("talk")

      fig, ax = plt.subplots(1, 1)
      sns.histplot(
        data=dA.loc[dA.connection == "exist"],
        x="weight",
        stat="density",
        ax=ax
      )
      plt.xlabel(label)
      # plt.ylabel(r"density $([\log(FLN)]^{-1})$")
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
          plot_path, f"weight_histo{label}.svg", 
        ),
        # dpi=300,
        # transparent=True
      )
      plt.close()
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
  
  def plot_akis(self, D, s=1, on=True):
    if on:
      print("Plot similarity plots and distance!!!")
      # Transform data to dataframe ----
      dD = adj2df(
        D.copy()[:self.nodes, :self.nodes]
      )

      wlabel_src = r"$D_{1/2}^{+}$"
      wlabel_tgt = r"$D_{1/2}^{-}$"

      # daki = 1 - self.aki.copy()
      daki = -2 * np.log(self.aki.copy())
      # daki[daki == 0] = np.nan
      # daki = 1/daki - 1

      daki = adj2df(daki)
      daki = daki.loc[
        daki["source"] < daki["target"], "weight"
      ].to_numpy().ravel()
 
      # daik = 1 - self.aik.copy()
      daik = -2 * np.log(self.aik.copy())
      # daik[daik == 0] = np.nan
      # daik = 1/daik - 1

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
          wlabel_tgt : daki,
          wlabel_src : daik
        }
      )


      data = data.loc[
        (~np.isnan(data[wlabel_tgt])) &
        (~np.isnan(data[wlabel_src])) &
        (np.abs(data[wlabel_src]) < np.inf) &
        (np.abs(data[wlabel_tgt]) < np.inf)
      ]


      # Create figures ----
      fig, ax = plt.subplots(1, 3)
      sns.scatterplot(
        data=data, x="dist", y=wlabel_tgt,
        s=s, ax=ax[0]
      )

      ## Compute stats ----
      # data_cor = pearsonr(
      #   data["dist"], data[wlabel_tgt]
      # )
      # _, _, data_r2, _, _ = linregress(
      #   data["dist"], data[wlabel_tgt]
      # )
      # ax[0].text(
      #   x = 0.5, y = 1.05,
      #   s = "{:.5f} {:.5f}".format(
      #     data_cor[0], data_r2 ** 2
      #   ),
      #   ha='center', va='center',
      #   transform=ax[0].transAxes
      # )
      ax[0].set_ylabel(wlabel_tgt)
      sns.scatterplot(
        data=data, x="dist", y=wlabel_src,
        s=s, ax=ax[1]
      )

      ## Compute stats ----
      # data_cor = pearsonr(
      #   data["dist"], data[wlabel_src]
      # )
      # _, _, data_r2, _, _ = linregress(
      #   data["dist"], data[wlabel_src]
      # )
      # ax[1].text(
      #   x = 0.5, y = 1.05,
      #   s = "{:.5f} {:.5f}".format(
      #     data_cor[0], data_r2 ** 2
      #   ),
      #   ha='center', va='center',
      #   transform=ax[1].transAxes
      # )
      ax[1].set_ylabel(wlabel_src)
      sns.scatterplot(
        data=data,
        x=wlabel_src, y=wlabel_tgt,
        s=s, ax=ax[2]
      )
      
      ## Compute stats ----
      # data_cor = pearsonr(
      #   data[wlabel_src],
      #   data[wlabel_tgt]
      # )
      # _, _, data_r2, _, _ = linregress(
      #   data[wlabel_src], data[wlabel_tgt]
      # )
      # ax[2].text(
      #   x = 0.5, y = 1.05,
      #   s = "{:.5f} {:.5f}".format(
      #     data_cor[0], data_r2 ** 2
      #   ),
      #   ha='center', va='center',
      #   transform=ax[2].transAxes
      # )
      ax[2].set_ylabel(wlabel_tgt)
      ax[2].set_xlabel(wlabel_src)
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
          plot_path, "similarity_plots_{}.svg".format(self.linkage)
        ),
        # dpi=300
      )
      plt.close()
    else:
      print("No similarities and distance plots")
      
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

  def plot_network_kk(self, H : Hierarchy, partition, nocs : dict, sizes, labels, ang=0, score="", undirected=False, cmap_name="hls", figsize=(12,12), on=True):
    if on:
      print("Printing network space")
      unique_clusters_id = np.unique(partition)
      keff = len(unique_clusters_id)
      # Generate all the colors in the color map -----
      if -1 in unique_clusters_id:
        save_colors = sns.color_palette(cmap_name, keff - 1)
        cmap_heatmap = [[]] * keff
        cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
        cmap_heatmap[1:] = save_colors
      else:
        save_colors = sns.color_palette(cmap_name, keff)
        cmap_heatmap = [[]] * (keff+1)
        cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
        cmap_heatmap[1:] = save_colors
      # Assign memberships to nodes ----
      if -1 in unique_clusters_id:
        nodes_memberships = {
          k : {"id" : [0] * keff, "size" : [0] * keff} for k in np.arange(len(partition))
        }
      else:
        nodes_memberships = {
          k : {"id" : [0] * (keff+1), "size" : [0] * (keff+1)} for k in np.arange(len(partition))
        }
      for i, id in enumerate(partition):
        if id == -1: continue
        nodes_memberships[i]["id"][id + 1] = 1
        nodes_memberships[i]["size"][id + 1] = 1
      for i, key in enumerate(nocs.keys()):
        index_key = np.where(labels == key)[0][0]
        for id in nocs[key]:
          if id == -1:
            nodes_memberships[index_key]["id"][0] = 1
            nodes_memberships[index_key]["size"][0] = 1
          else:
            nodes_memberships[index_key]["id"][id + 1] = 1
            nodes_memberships[index_key]["size"][id + 1] = sizes[key][id]
      # Check unassigned ----
      for i in np.arange(self.nodes):
        if np.sum(np.array(nodes_memberships[i]["id"]) == 1) == 0:
          nodes_memberships[i]["id"][0] = 1
          nodes_memberships[i]["size"][0] = 1
      if not undirected:
        G = nx.DiGraph(H.A)
      else:
        G = nx.Graph(H.A, directed=False)
      pos = nx.kamada_kawai_layout(G)
      ang = ang * np.pi/ 180
      rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
      pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
      fig, ax = plt.subplots(1, 1, figsize=figsize)
      if undirected:
        nx.draw_networkx_edges(G, pos=pos, arrows=False)
      else:
        nx.draw_networkx_edges(G, pos=pos)
      nx.draw_networkx_labels(G, pos=pos)
      for node in G.nodes:
        a = plt.pie(
          [s for s in nodes_memberships[node]["size"] if s != 0], # s.t. all wedges have equal size
          center=pos[node],  
          colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[node]["id"]) if id != 0],
          radius=0.08
        )
        for i in range(len(a[0])):
          a[0][i].set_alpha(0.8)
      array_pos = np.array([list(pos[v]) for v in pos.keys()])
      plt.xlim(-0.1 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.1)
      plt.ylim(-0.1 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.1)
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Network"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        os.path.join(
          plot_path, f"kk{score}.png"
        ),
        dpi=300
      )
      plt.close()
  
  def plot_network_covers(
      self, k, R : npt.NDArray, partition, nocs : dict, sizes : dict, labels,
      ang=0, score="", direction="", cmap_name="hls", figsize=(12,12), scale=1.5,
      color_order=None, spring=False , undirected=False, font_size=12, on=True, **kwargs
    ):
    if on:
      print("Printing network space")
      from scipy.cluster import hierarchy
      import matplotlib.patheffects as path_effects
      # from matplotlib.colors import to_hex
      # Skim partition ----
      unique_clusters_id = np.unique(partition)
      keff = len(unique_clusters_id)
      # Generate all the colors in the color map -----
      if -1 in unique_clusters_id:
        save_colors = sns.color_palette(cmap_name, keff - 1)
        cmap_heatmap = [[]] * keff
        cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
        cmap_heatmap[1:] = save_colors
      else:
        save_colors = sns.color_palette(cmap_name, keff)
        cmap_heatmap = [[]] * (keff+1)
        cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
        cmap_heatmap[1:] = save_colors
      cmap_heatmap = np.array(cmap_heatmap)
      if isinstance(color_order, np.ndarray):
        cmap_heatmap[1:] = cmap_heatmap[1:][color_order]
      # Assign memberships to nodes ----
      if -1 in unique_clusters_id:
        nodes_memberships = {
          k : {"id" : [0] * keff, "size" : [0] * keff} for k in np.arange(len(partition))
        }
      else:
        nodes_memberships = {
          k : {"id" : [0] * (keff+1), "size" : [0] * (keff+1)} for k in np.arange(len(partition))
        }
      for i, id in enumerate(partition):
        if id == -1: continue
        nodes_memberships[i]["id"][id + 1] = 1
        nodes_memberships[i]["size"][id + 1] = 1
      for i, key in enumerate(nocs.keys()):
        index_key = np.where(labels == key)[0][0]
        for id in nocs[key]:
          if id == -1:
            nodes_memberships[index_key]["id"][0] = 1
            nodes_memberships[index_key]["size"][0] = 1
          else:
            nodes_memberships[index_key]["id"][id + 1] = 1
            nodes_memberships[index_key]["size"][id + 1] = sizes[key][id]
      # Check unassigned ----
      for i in np.arange(self.nodes):
        if np.sum(np.array(nodes_memberships[i]["id"]) == 1) == 0:
          nodes_memberships[i]["id"][0] = 1
          nodes_memberships[i]["size"][0] = 1
      # Get edges colors ----
      dA = self.dA.copy()
      if not undirected:
         dA["id"] = hierarchy.cut_tree(self.H, k).reshape(-1)
      else:
         dA["id"] = np.tile(hierarchy.cut_tree(self.H, k).reshape(-1), 2)
      minus_one_Dc(dA, undirected)
      aesthetic_ids(dA)
      dA = df2adj(dA, var="id")
      # Generate graph ----
      G = nx.from_numpy_array(R, create_using=nx.DiGraph)
      edge_color = [""] * len(G.edges())
      for i, dat in enumerate(G.edges(data=True)):
        u, v, a = dat
        if dA[u, v] == -1: edge_color[i] = cmap_heatmap[0]
        else: edge_color[i] = "#666666"
      if "coords" not in kwargs.keys():
        # Rsym = (R + R.T) / 2
        # Gpos = nx.from_numpy_array(Rsym, create_using=nx.Graph)
        pos = nx.kamada_kawai_layout(G, weight="weight")
        if spring:
          Ginv = nx.DiGraph(1/R)
          pos = nx.spring_layout(Ginv, weight="weight", pos=pos, iterations=5, seed=212)
      else:
        pos = kwargs["coords"]
      ang = ang * np.pi/ 180
      rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
      pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
      labs = {k : lab for k, lab in zip(G.nodes, labels)}

      mu_pos_x = np.mean([k[0] for k in pos.values()])
      mu_pos_y = np.mean([k[1] for k in pos.values()])
      mu_pos = np.array([mu_pos_x, mu_pos_y])

      pos = {k : pos[k] - mu_pos for k in pos.keys()}
      pos = {k : pos[k] * scale for k in pos.keys()}

      # plt.style.use("dark_background")
      plt.figure(figsize=figsize)
      if "not_edges" not in kwargs.keys():
        nx.draw_networkx_edges(
          G, pos=pos, edge_color=edge_color, alpha=0.5, width=2, arrowsize=10, connectionstyle="arc3,rad=-0.1",
          node_size=1800
        )
      if "modified_labels" not in kwargs.keys():
        t = nx.draw_networkx_labels(G, pos=pos, labels=labs, font_color="white")
        for key in t.keys():
          t[key].set_path_effects(
          [
            path_effects.Stroke(linewidth=1, foreground='black'),
            path_effects.Normal()
          ]
        )
      else:
        t = nx.draw_networkx_labels(G, pos=pos, labels=kwargs["modified_labels"], font_color="white", font_size=font_size)
        for key in t.keys():
          t[key].set_path_effects(
          [
            path_effects.Stroke(linewidth=1, foreground='black'),
            path_effects.Normal()
          ]
        )

      for node in G.nodes:
        a = plt.pie(
          [s for s in nodes_memberships[node]["size"] if s != 0], # s.t. all wedges have equal size
          center=pos[node],  
          colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[node]["id"]) if id != 0],
          radius=0.08
        )
        for i in range(len(a[0])):
          a[0][i].set_alpha(0.8)
      array_pos = np.array([list(pos[v]) for v in pos.keys()])
      plt.xlim(-0.05 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.05)
      plt.ylim(-0.05 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.05)
      # plt.show()
      # Arrange path ----
      plot_path = os.path.join(
        self.path, "Network"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----

      #### Careful: manual modification #### -----
      plt.savefig(
        os.path.join(
          plot_path, f"net_cover_{score}_{direction}.svg"
        ),
        dpi=300, transparent=True
      )
      #### End modification ####
      plt.close()

  def distance_cover_swarm(self, partition, covers : dict, direction="source", index="Hellinger2", on=True):
    if on:
      print("Plot distance cover test!!!")
      unique_clusters_id = np.unique(partition.copy())
      unique_clusters_id = unique_clusters_id[unique_clusters_id != -1]
      unique_clusters_id = np.sort(unique_clusters_id)

      if index == "Hellinger2":  
        if direction == "source":
          dist = 1. - self.aik
        elif direction == "target":
          dist = 1. - self.aki
        elif direction == "both":
          a = np.sqrt(1. - self.aik)
          b = np.sqrt(1. - self.aki)
          dist = np.power(0.5 * (a + b), 2.)
        else:
          raise ValueError("No accepted direction")
      elif index == "dist_sim":
        if direction == "source":
          np.seterr(divide='ignore', invalid='ignore')
          dist = (1. /  self.aik) - 1
        elif direction == "target":
          np.seterr(divide='ignore', invalid='ignore')
          dist = (1. / self.aki) - 1
        elif direction == "both":
          np.seterr(divide='ignore', invalid='ignore')
          a = (1. / self.aik) - 1
          np.seterr(divide='ignore', invalid='ignore')
          b = (1. / self.aki) - 1
          dist = 0.5 * (a + b)
        else:
          raise ValueError("No accepted direction")
      
      # dist = adj2df(dist)
      np.fill_diagonal(dist, np.nan)

      data = pd.DataFrame()
      covs_keys = list(covers.keys())

      if index == "Hellinger2":
        wlabel = r"$H^{2}$"
      elif index == "dist_sim":
        wlabel = r"$D_{phys}$"

      for i in np.arange(len(covs_keys)):
        areas =  match(covers[covs_keys[i]], self.labels)
        keyi = covs_keys[i]
        for j in np.arange(len(covs_keys)):
          areas2 = match(covers[covs_keys[j]], self.labels)
          keyj = covs_keys[j]

          if i == j:
            d = dist[areas, :][:, areas2]
            d = adj2df(d)
            d = d["weight"].loc[d.source > d.target]
          else:
            areas3 = set(areas)
            areas2 = set(areas2)
            inter_areas = areas3.intersection(areas2)
            if len(inter_areas) > 0:
              d = dist[list(areas3 - inter_areas), :][:, list(areas2 - inter_areas)].ravel()
              d2 = dist[list(inter_areas), :][:, list(areas3 - inter_areas)].ravel()
              d3 = dist[list(inter_areas), :][:, list(areas2 - inter_areas)].ravel()
              d = np.hstack([d, d2, d3])

            else:
              d = dist[list(areas3), :][:, list(areas2)].ravel()

          data = pd.concat(
            [
              data,
              pd.DataFrame(
                {
                  "X" : [str(keyi)] * d.shape[0],
                  "Y" : [str(keyj)] * d.shape[0],
                  wlabel : d
                }
              )
            ], ignore_index=True
          )
      
      g = sns.FacetGrid(
        data=data,
        col="X",
        col_wrap=4,
        col_order=unique_clusters_id.astype(str)
      )

      from scipy.stats import ttest_ind

      g.map_dataframe(
        sns.swarmplot,
        y="Y",
        hue="Y",
        palette=sns.color_palette("hls", unique_clusters_id.shape[0]),
        s=2,
        x=wlabel
      )

      ytext = np.linspace(0, 1, unique_clusters_id.shape[0] + 1)
      dtext = (ytext[1] - ytext[0]) / 2
      ytext = ytext[:-1] + dtext
      ytext = -np.sort(-ytext)

      for i, axis in enumerate(g.axes.flat):
        do = data[wlabel].loc[(data.X == str(unique_clusters_id[i])) & (data.Y == str(unique_clusters_id[i]))]

        for j in np.arange(unique_clusters_id.shape[0]):
          if i == j : continue
          da = data[wlabel].loc[(data.X == str(unique_clusters_id[i])) & (data.Y == str(unique_clusters_id[j]))]

          test = ttest_ind(do, da, alternative="less", equal_var=False) 

          if not np.isnan(test.pvalue):
            if test.pvalue > 0.05:
              a = "ns"
            elif test.pvalue <= 0.05 and test.pvalue > 0.001:
              a = "*" 
            elif test.pvalue <= 0.001 and test.pvalue > 0.0001:
              a = "**" 
            else:
              a = "***"
          else:
            a = "nan"

          axis.text(0.05, ytext[j] - 0.03, a, transform=axis.transAxes)

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
          plot_path, f"swarm_{direction}.png"
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No distance_cover")

  def distance_cover_boxplot(self, partition, covers : dict, direction="source", index="Herllinger2", cmap_name="hls", on=True):
    if on:
      print("Plot distance cover box test!!!")
      unique_clusters_id = np.unique(partition.copy())
      unique_clusters_id = unique_clusters_id[unique_clusters_id != -1]
      unique_clusters_id = np.sort(unique_clusters_id)

      cmap_ = sns.color_palette(cmap_name, unique_clusters_id.shape[0])
        
      if index == "Hellinger2" or index == "cos":  
        if direction == "source":
          dist = 1. - self.aik
        elif direction == "target":
          dist = 1. - self.aki
        elif direction == "both":
          a = np.sqrt(1. - self.aik)
          b = np.sqrt(1. - self.aki)
          dist = np.power(0.5 * (a + b), 2.)
        else:
          raise ValueError("No accepted direction")
      elif index == "dist_sim":
        if direction == "source":
          np.seterr(divide='ignore', invalid='ignore')
          dist = (1. /  self.aik) - 1
        elif direction == "target":
          np.seterr(divide='ignore', invalid='ignore')
          dist = (1. / self.aki) - 1
        elif direction == "both":
          np.seterr(divide='ignore', invalid='ignore')
          a = (1. / self.aik) - 1
          np.seterr(divide='ignore', invalid='ignore')
          b = (1. / self.aki) - 1
          dist = 0.5 * (a + b)
        else:
          raise ValueError("No accepted direction")
      
      # dist = adj2df(dist)
      np.fill_diagonal(dist, np.nan)

      data = pd.DataFrame()
      covs_keys = list(covers.keys())

      if index == "Hellinger2":
        wlabel = r"$H^{2}$"
      elif index == "dist_sim":
        wlabel = r"$D_{phys}$"
      elif index == "cos":
        wlabel = "1 - cosine similarity"

      for i in np.arange(len(covs_keys)):
        areas =  match(covers[covs_keys[i]], self.labels)

        d = dist[areas, :][:, areas]
        d = adj2df(d)
        d = d["weight"].loc[d.source > d.target]

        data = pd.concat(
          [
            data,
            pd.DataFrame(
              {
                "cover" : [str(covs_keys[i])] * d.shape[0],
                "set" : ["within"] * d.shape[0],
                wlabel : d
              }
            )
          ], ignore_index=True
        )

        for j in np.arange(len(covs_keys)):
          areas2 = match(covers[covs_keys[j]], self.labels)

          if i == j: continue
          else:
            areas3 = set(areas)
            areas2 = set(areas2)
            inter_areas = areas3.intersection(areas2)
            if len(inter_areas) > 0:
              d = dist[list(areas3 - inter_areas), :][:, list(areas2 - inter_areas)].ravel()
              d2 = dist[list(inter_areas), :][:, list(areas3 - inter_areas)].ravel()
              d3 = dist[list(inter_areas), :][:, list(areas2 - inter_areas)].ravel()
              d = np.hstack([d, d2, d3])

            else:
              d = dist[list(areas3), :][:, list(areas2)].ravel()

          data = pd.concat(
            [
              data,
              pd.DataFrame(
                {
                  "cover" : [str(covs_keys[i])] * d.shape[0],
                  "set" : ["between"] * d.shape[0],
                  wlabel : d
                }
              )
            ], ignore_index=True
          )
      
      data = data.loc[~np.isnan(data[wlabel])]
      plt.style.use("dark_background")
      sns.set_context("talk")
      sns.boxplot(
        data=data,
        x="cover",
        y= wlabel,
        hue="set"
      )

      ax = plt.gca()
      fig = plt.gcf()

      fig.set_figwidth(12)
      fig.set_figheight(9)

      
      from scipy.stats import ttest_ind

      xtext = np.linspace(0, 1, unique_clusters_id.shape[0] + 1)
      dtext = (xtext[1] - xtext[0]) / 2
      xtext = xtext[:-1] + dtext
      xtext = np.sort(xtext)

      for i, c in enumerate(np.unique(data.cover)):
        do = data[wlabel].loc[(data.cover == c) & (data.set == "within")].to_numpy()
        da = data[wlabel].loc[(data.cover == c) & (data.set == "between")].to_numpy()

        test = ttest_ind(do, da, alternative="less", equal_var=False)

        if  not np.isnan(test.pvalue): 
          if test.pvalue > 0.05:
            a = "ns"
          elif test.pvalue <= 0.05 and test.pvalue > 0.001:
            a = "*" 
          elif test.pvalue <= 0.001 and test.pvalue > 0.0001:
            a = "**" 
          else:
            a = "***"
        else:
          a = "nan"
        ax.text(xtext[i] - 0.015, 1.01, a, transform=ax.transAxes)

      ax.set_xticklabels(ax.get_xticklabels(), weight="bold", fontsize=30)
      for i in np.arange(unique_clusters_id.shape[0]):
        ax.get_xticklabels()[i].set_color(cmap_[i]) 

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
          plot_path, f"box_{direction}.png"
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No distance_cover box")

  
  def plot_cover_items(self, cover : dict):

    node_cover = cover_node_2_node_cover(cover, self.labels[:self.nodes])
    nocs = np.array([n for n, val in node_cover.items() if len(val) > 1])
    # Create a directed graph
    G = nx.Graph()

    # Add nodes and edges to the graph based on the dictionary
    for key, values in cover.items():
        G.add_node(f"C{key}")
        G.add_node(values[0])
        G.add_edge(f"C{key}", values[0])
        last_node = values[0]
        for i in np.arange(1, len(values)):
            if values[i] not in G.nodes:
              if np.isin(values[i], nocs):
                val = values[i] + f"-{key}"
                G.add_node(val)
                G.add_edge(last_node, val)
                last_node = val
              else:
                G.add_node(values[i])
                G.add_edge(last_node, values[i])
                last_node = values[i]

    subset_dict = {}
    for key, val in cover.items():
      subset_dict[f"C{key}"] = key
      for v in val:
        if np.isin(v, nocs):
          subset_dict[v+f"-{key}"] = key
        else:
          subset_dict[v] = key

    nx.set_node_attributes(G, subset_dict, 'subset')
    # Create a layout for the nodes
    pos = nx.multipartite_layout(G, align='horizontal')
    G_nodes = np.array(G.nodes)
    node_colors = np.array(["lightblue"] * len(G_nodes), dtype="<U21")

    array_op = lambda x, sx: np.array([x[0]*sx, x[1]])

    pos = {p:array_op(pos[p], -1) for p in pos}

    for i, n in enumerate(G_nodes):
      if "C" in n:
        node_colors[i] = "salmon"

    # k = 0
    # for i, n in enumerate(G_nodes):
    #   if "C" in n:
    #     k = 0
    #   pos[n][1] -= k
    #   k += 1

    # Draw the network plot
    nx.draw(
      G, pos, with_labels=True, node_size=700,
      node_color=node_colors, font_size=7, font_weight='bold'
    )

    # plt.title('Cover memberships')

    # Arrange path ---- 
    plot_path = os.path.join(
      self.path, "sln"
    )
    # Crate path ----
    Path(
      plot_path
    ).mkdir(exist_ok=True, parents=True)
    # Save plot ----
    plt.savefig(
      os.path.join(
        plot_path, f"cover_memberships_net.svg"
      ),
      transparent=True,
      # dpi=300
    )
    plt.close()





# import ctools as ct
#   data_p = np.zeros((57,57))
#   data_n = np.zeros((57,57))

#   for i in np.arange(57):
#     for j in np.arange(i+1, 57):
#       data_p[i, j] = ct.D1_2_4(REF.C[i, :], REF.C[j, :], i, j)
#       data_p[j,i] = data_p[i,j]
#       data_n[i, j] = ct.D1_2_4(REF.C[:, i], REF.C[:, j], i, j)
#       data_n[j, i] = data_n[i,j]

#   data_p[data_p == 0] = np.nan
#   data_n[data_n == 0] = np.nan

#   data_p = 1/data_p - 1
#   data_n = 1/data_n - 1


#   edr_n = H.target_sim_matrix
#   edr_p = H.source_sim_matrix

#   edr_n[edr_n == 0] = np.nan
#   edr_p[edr_p == 0] = np.nan

#   edr_n = 1/edr_n - 1
#   edr_p = 1/edr_p - 1

#   d = D[:57, :57]
#   d = adj2df(d)

#   data_p = adj2df(data_p)
#   data_p["set"] = "data"
#   data_p["dir"] = "source"
#   data_p["distance"] = d["weight"]

#   data_n = adj2df(data_n)
#   data_n["set"] = "data"
#   data_n["dir"] = "target"
#   data_n["distance"] = d["weight"]

#   edr_p = adj2df(edr_p)
#   edr_p["set"] = "edr"
#   edr_p["dir"] = "source"
#   edr_p["distance"] = d["weight"]

#   edr_n = adj2df(edr_n)
#   edr_n["set"] = "edr"
#   edr_n["dir"] = "target"
#   edr_n["distance"] = d["weight"]


#   obj = pd.concat([data_p, data_n, edr_p, edr_n], ignore_index=True)
#   obj = obj.loc[(obj.source > obj.target) & (~np.isnan(obj["weight"]))]

#   import matplotlib.pyplot as plt
#   import seaborn as sns

#   sns.set_style("whitegrid")

#   g = sns.lmplot(
#     data=obj,
#     col="dir",
#     x="distance",
#     y="weight",
#     hue="set",
#     scatter_kws={"alpha": 0.4, "s" : 3},
#     lowess=True
#   )
#   g.set_axis_labels("interareal tractography distances [mm]", r"$D_{1/2}$", )

#   plt.savefig(os.path.join(NET.plot_path, "Features", "edr_data_renyi.png"), dpi=300)
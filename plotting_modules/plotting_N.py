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
      # Elllminate zeros ---
      dA = dA.loc[(~zeros) & (~isnan)]
      dD = dD.loc[(~zeros) & (~isnan)]
      # Create data ----
      from pandas import DataFrame
      data = DataFrame(
        {
          "weight" : dA["weight"],
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
      # plt.ylabel("log(FLN)")
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

  def projection_probability(
    self, C, model ,bins=20, on=True, **kwargs
  ):
    if on:
      print("Plot neuron-distance freq!!!")
      # Distance range ----
      from pandas import DataFrame
      from various.fit_tools import fitters
      _, x, y = range_and_probs_from_DC(self.D, C, bins)
      y = np.exp(y)
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
      label="model"
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
            plot_path, "bin_{}_{}.png".format(bins, model)
          ),
          dpi=200
        )
    else:
      print("No histogram")

  def histogram_weight(self, A, label="", on=True):
    if on:
      print("Plot weight histogram!!!")
      # Transform FLN to DataFrame ----
      dA = A.copy()
      dA = adj2df(A)
      dA["connection"] = "exist"
      dA.connection.loc[dA.weight == 0] = "~exist"
      # Transform FLN to weights ----
      fig, ax = plt.subplots(1, 1)
      sns.histplot(
        data=dA.loc[dA.connection == "exist"],
        x="weight",
        hue = "connection",
        stat="density",
        ax=ax
      )
      # plt.xlabel(r"$\log(FLN)$")
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
          plot_path, f"weight_histo{label}.png"
        ),
        dpi=300
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
      from scipy.stats import pearsonr, linregress
      print("Plot similarity plots and distance!!!")
      # Transform data to dataframe ----
      dD = adj2df(
        D.copy()[:self.nodes, :self.nodes]
      )
      daki = self.aki.copy()
      # print(np.sum(np.isnan(daki)))
      # print(np.nanmin(daki))
      # print(np.sum(daki == np.nanmin(daki)))
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

  def plot_network_kk(self, H : Hierarchy, partition, nocs : dict, labels, ang=0, score="", undirected=False, cmap_name="hls", on=True):
    if on:
      print("Printing network space")
      new_partition = skim_partition(partition)
      unique_clusters_id = np.unique(new_partition)
      keff = len(unique_clusters_id)
      save_colors = sns.color_palette(cmap_name, keff - 1)
      cmap_heatmap = [[]] * keff
      cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
      cmap_heatmap[1:] = save_colors
      # Assign memberships to nodes ----
      nodes_memberships = {k : [0] * keff for k in np.arange(H.nodes)}
      for i, id in enumerate(new_partition):
        if id == -1: continue
        nodes_memberships[i][id + 1] = 1
      for i, key in enumerate(nocs.keys()):
        index_key = np.where(labels == key)[0][0]
        for id in nocs[key]:
          if id == -1:
            nodes_memberships[index_key][0] = 1
          else: nodes_memberships[index_key][id + 1] = 1
      # Check unassigned ----
      for i in np.arange(H.nodes):
        if np.sum(np.array(nodes_memberships[i]) == 0) == keff:
          nodes_memberships[i][0] = 1
      if not undirected:
        G = nx.DiGraph(H.A)
      else:
        G = nx.Graph(H.A, directed=False)
      pos = nx.kamada_kawai_layout(G)
      ang = ang * np.pi/ 180
      rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
      pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
      if undirected:
        nx.draw_networkx_edges(G, pos=pos, arrows=False)
      else:
        nx.draw_networkx_edges(G, pos=pos)
      nx.draw_networkx_labels(G, pos=pos)
      for node in G.nodes:
        plt.pie(
          [1 for id in nodes_memberships[node] if id != 0], # s.t. all wedges have equal size
          center=pos[node], 
          colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[node]) if id != 0],
          radius=0.05
        )
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
  
  def plot_network_covers(self, k, R, partition, nocs : dict, sizes : dict, labels, ang=0, score="", cmap_name="hls", undirected=False, on=True, **kwargs):
    if on:
      print("Printing network space")
      from scipy.cluster import hierarchy
      # from matplotlib.colors import to_hex
      # Skim partition ----
      new_partition = skim_partition(partition)
      unique_clusters_id = np.unique(new_partition)
      keff = len(unique_clusters_id)
      # Generate all the colors in the color map -----
      if -1 in unique_clusters_id:
        save_colors = sns.color_palette(cmap_name, keff - 1)
        cmap_heatmap = [[]] * keff
        cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
        cmap_heatmap[1:] = save_colors
      else:
        save_colors = sns.color_palette(cmap_name, keff)
        cmap_heatmap = [[]] * (keff+1)
        cmap_heatmap[0] = [199/ 255.0, 0, 57/ 255.0]
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
      for i, id in enumerate(new_partition):
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
      for i in np.arange(len(partition)):
        if np.sum(nodes_memberships[i] == 0) == keff:
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
      G = nx.DiGraph(R)
      r_min = np.min(R[R>0])
      r_max = np.max(R)
      edge_color = [""] * self.edges
      for i, dat in enumerate(G.edges(data=True)):
        u, v, a = dat
        if "coords" not in kwargs.keys():
          G[u][v]["kk_weight"] = - (a["weight"] - r_min) / (r_max - r_min) + r_max
        if dA[u, v] == -1: edge_color[i] = cmap_heatmap[0]
        else: edge_color[i] = "gray"
      if "coords" not in kwargs.keys():
        pos = nx.kamada_kawai_layout(G, weight="kk_weight")
      else:
        pos = kwargs["coords"]
      ang = ang * np.pi/ 180
      rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
      pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
      labs = {k : lab for k, lab in zip(G.nodes, labels)}
      plt.figure(figsize=(12,12))
      if "not_edges" not in kwargs.keys():
        nx.draw_networkx_edges(G, pos=pos, edge_color=edge_color, alpha=0.2, arrowsize=20, connectionstyle="arc3,rad=-0.1")
      if "modified_labels" not in kwargs.keys():
        nx.draw_networkx_labels(G, pos=pos, labels=labs)
      else:
        nx.draw_networkx_labels(G, pos=pos, labels=kwargs["modified_labels"])
      for node in G.nodes:
        a = plt.pie(
          [s for s in nodes_memberships[node]["size"] if s != 0], # s.t. all wedges have equal size
          center=pos[node], 
          colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[node]["id"]) if id != 0],
          radius=0.05
        )
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
          plot_path, f"net_cover_{score}.png"
        ),
        dpi=300
      )





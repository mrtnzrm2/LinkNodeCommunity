import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
sns.set_theme()
from scipy.cluster.hierarchy import cut_tree
# Personal libs ----
from networks.structure import MAC
from modules.nodhierarchy import NODH

class PLOT_TREND:
  def __init__(self, NET : MAC, H : NODH) -> None:
    self.net = NET
    self.nodes = H.nodes
    self.inj = H.inj
    self.linkage = H.linkage
    self.W1 = H.W1
    self.W2 = H.W2
    self.D = H.D
    # self.Z = H.Z
    self.feature_dist = H.feature_dist
    ## Labels ----
    self.rlabels = H.struct_labels
    self.clabels = H.struct_labels[:self.nodes]
    ## Plotting directory ----
    self.plot_path = H.plot_path

  def skim_partition(self, partition):
    from collections import Counter
    fq = Counter(partition)
    for i in fq.keys():
      if fq[i] == 1: partition[partition == i] = -1
    new_partition = partition
    ndc = np.unique(partition[partition != -1])
    for i, c in enumerate(ndc):
      new_partition[partition == c] = i
    return new_partition

  def plot_wout_trends(self, areas):
    # Create data -----
    W = self.W1.copy()
    W[W == 0] = np.nan
    D = self.D[:, :W.shape[1]].copy()
    data = pd.DataFrame(
      {
        "areas" : np.repeat(areas, W.shape[1]),
        "dist" : D.ravel(),
        "weight" : W.ravel()
      }
    )
    data = data.loc[~np.isnan(data.weight)]
    # Create figure ----
    order = np.argsort(areas)
    g = sns.FacetGrid(
      data=data,
      col="areas",
      col_wrap=10,
      col_order=areas[order],
      height=2,
      aspect=1
    )
    g.map_dataframe(
      sns.lineplot,
      x="dist",
      y="weight"
    )
    g.add_legend()
    ## Save plot ----
    fname  = f"{self.plot_path}/Trends"
    Path(fname).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      f"{fname}/w_out_trends.png",
      dpi=300
    )
    plt.close()

  def plot_feature_dist_dist(self, r, on=False):
    if on:
      print("Plot feature dist-dist!!")
      nodes = self.feature_dist.shape[1]
      D = self.D[:nodes, :nodes].copy()
      np.fill_diagonal(D, np.nan)
      # Create data ----
      data = pd.DataFrame(
        {
          "source" : np.repeat(np.arange(nodes), nodes),
          "target" : np.tile(np.arange(nodes), nodes),
          "feature_distance" : self.feature_dist.ravel(),
          "dist" : D.ravel()
        }
      )
      data = data.loc[
        (~np.isnan(data.dist)) &
        (data.source > data.target)
      ]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.scatterplot(
        data=data,
        x="dist",
        y="feature_distance",
        s=5,
        ax=ax
      )
      fig.tight_layout()
      ## Save plot ----
      fname  = f"{self.plot_path}/Features"
      Path(fname).mkdir(exist_ok=True, parents=True)
      plt.savefig(
        f"{fname}/feature_dd_{r}.png",
        dpi=300
      )
      plt.close()
    else:
      print("No feature dist-dist")

  def plot_single_area_wout(self, areas, axis=0):
    # Create data -----
    W = self.W1.copy()
    W[W == 0] = np.nan
    D = self.D[:, :W.shape[1]].copy()
    if axis == 0:
      data = pd.DataFrame(
        {
          "areas" : np.repeat(areas, W.shape[1]),
          "dist" : D.ravel(),
          "weight" : W.ravel()
        }
      )
    else:
      data = pd.DataFrame(
        {
          "areas" : np.tile(areas[:W.shape[1]], W.shape[0]),
          "dist" : D.ravel(),
          "weight" : W.ravel()
        }
      )
    data = data.loc[~np.isnan(data.weight)]
    ## Save plot ----
    fname  = f"{self.plot_path}/Single_out_Trend"
    Path(fname).mkdir(exist_ok=True, parents=True)
    for a in areas:
      # Create figure ----
      ax = sns.lineplot(
        data=data.loc[data.areas == a],
        x="dist",
        y="weight"
      )
      ax.set_title(a)
      if "/" in a: a = a.replace("/", "-")
      plt.savefig(
        f"{fname}/{a}.png",
        dpi=300
      )
      plt.close()

  def plot_wtrends_dendrogram(self, R, areas, Z, axis=0, on=False):
    # Merge single communities ----
    if on:
      print("Plot classified trends!!!")
      for r in R:
        partition = cut_tree(Z, r).ravel()
        new_partition = self.skim_partition(partition)
        # Create data -----
        W = self.W1.copy()
        W[W == 0] = np.nan
        D = self.D.copy()[:, :self.inj]
        if axis==0:
          data = pd.DataFrame(
            {
              "areas" :  np.repeat(areas, W.shape[1]),
              "dist" : D.ravel(),
              "weight" : W.ravel()
            }
          )
        else:
          data = pd.DataFrame(
            {
              "areas" :  np.tile(areas[:self.nodes], W.shape[0]),
              "dist" : D.ravel(),
              "weight" : W.ravel()
            }
          )
        data = data.loc[~np.isnan(data.weight)]
        ## By cluster ----
        unique_cluster_id = np.unique(new_partition)
        for id in unique_cluster_id:
          areas_in_cluster = areas[:self.nodes][new_partition == id]
          sub = data.loc[np.isin(data.areas, areas_in_cluster)]
          if len(areas_in_cluster) < 5: ncols = len(areas_in_cluster)
          else: ncols = 5
          # Create figure ----
          order = np.argsort(areas_in_cluster)
          g = sns.FacetGrid(
            data=sub,
            col="areas",
            col_wrap=ncols,
            col_order=areas_in_cluster[order],
            height=2,
            aspect=1
          )
          g.map_dataframe(
            sns.lineplot,
            x="dist",
            y="weight"
          )
          g.add_legend()
          ## Save plot ----
          fname  = f"{self.plot_path}//ByClusters_{r}"
          Path(fname).mkdir(exist_ok=True, parents=True)
          plt.savefig(
            f"{fname}/w_trends_{id}.png", dpi=300
          )
          plt.close()
    else: print("No classified trends")

  def plot_winout_trends(self, areas):
    # Create data -----
    w_sub = self.W1.copy()
    D = self.D.copy()[:w_sub.shape[0], :self.inj]
    # w_sub = w_sub[:w.shape[1], :]
    w_sub[w_sub == 0] = np.nan
    ##
    # d = dist[:w_sub.shape[0], :w_sub.shape[1]]
    d = D[:, :w_sub.shape[1]].copy()
    ##
    a = areas.copy()
    a = a[:w_sub.shape[1]]
    data = pd.DataFrame(
      {
        "set" : ["source"] * len(d[:w_sub.shape[1], :].ravel()),
        "areas" : np.repeat(a, w_sub.shape[1]),
        "dist" : d[:w_sub.shape[1], :].ravel(),
        "weight" : w_sub[:w_sub.shape[1], :].ravel()
      }
    )
    data = pd.concat(
      [
        data,
        pd.DataFrame(
          {
            "set" : ["target"] * len(d.ravel()),
            "areas" : np.tile(a, w_sub.shape[0]),
            "dist" : d.ravel(),
            "weight" : w_sub.ravel()
          }
        )
      ],
      ignore_index=True
    )
    data = data.loc[~np.isnan(data.weight)]
    # Create figure ----
    order = np.argsort(a)
    g = sns.FacetGrid(
      data=data,
      col="areas",
      col_wrap=10,
      col_order=a[order],
      hue="set",
      height=2,
      aspect=1
    )
    g.map_dataframe(
      sns.lineplot,
      x="dist",
      y="weight",
      alpha=0.5
    )
    g.add_legend()
    ## Save plot ----
    fname  = f"{self.plot_path}/Trends"
    Path(fname).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      f"{fname}/winout_trends.png", dpi=300
    )
    plt.close()

  def plot_winout_trends_dendrogram(self, R, areas, Z):
    for r in R:
      partition = cut_tree(Z, r).ravel()
      ## Merge single communities --
      new_partition = self.skim_partition(partition)
      ##
      # Create data -----
      w_sub = self.W1.copy()
      w_sub = w_sub[:w_sub.shape[1], :]
      w_sub[w_sub == 0] = np.nan
      ##
      d = self.D[:w_sub.shape[0], :w_sub.shape[1]].copy()
      ##
      a = areas.copy()
      a = a[:w_sub.shape[1]]
      data = pd.DataFrame(
        {
          "set" : ["source"] * len(d.ravel()),
          "areas" : np.repeat(a, w_sub.shape[0]),
          "dist" : d.ravel(),
          "weight" : w_sub.ravel()
        }
      )
      data = pd.concat(
        [
          data,
          pd.DataFrame(
            {
              "set" : ["target"] * len(d.ravel()),
              "areas" : np.tile(a, w_sub.shape[0]),
              "dist" : d.ravel(),
              "weight" : w_sub.ravel()
            }
          )
        ],
        ignore_index=True
      )
      data = data.loc[~np.isnan(data.weight)]
      ## By cluster ----
      unique_cluster_id = np.unique(new_partition)
      for id in unique_cluster_id:
        areas_in_cluster = areas[new_partition == id]
        subdata = data.loc[np.isin(data.areas, areas_in_cluster)]
        # Create figure ----
        order = np.argsort(areas_in_cluster)
        g = sns.FacetGrid(
          data=subdata,
          col="areas",
          col_wrap=8,
          col_order=areas_in_cluster[order],
          hue="set",
          height=2,
          aspect=1
        )
        g.map_dataframe(
          sns.lineplot,
          x="dist",
          y="weight",
          alpha=0.5
        )
        g.add_legend()
        ## Save plot ----
        fname  = f"{self.plot_path}/Trends"
        Path(fname).mkdir(exist_ok=True, parents=True)
        plt.savefig(
          f"{fname}/winout_cluster_{id}.png",
          dpi=300
        )
        plt.close()

  def plot_feature_distance(self, areas, feature="jacp_source", on=False):
    if on:
      print("Plot feature distance heatmap!!!")
      feature_dist = self.feature_dist.copy()
      ## Create figure ----
      np.fill_diagonal(feature_dist, np.nan)
      data = pd.DataFrame(
        feature_dist, columns=areas[:self.nodes], index=areas[:self.nodes]
      )
      sns.heatmap(
        data=data
      )
      ## Save plot ----
      fname  = f"{self.plot_path}/Features"
      Path(fname).mkdir(exist_ok=True, parents=True)
      plt.savefig(f"{fname}/{feature}_distance.png", dpi=200)
      plt.close()
    else: print("No feature distance heatmap")

  def plot_dendrogram(self, distance, areas, on=False):
    if on:
      print("Plot feature distance heatmap with dendrogram!!!")
      dist = self.feature_dist.copy()
      np.fill_diagonal(dist, np.min(distance))
      data = pd.DataFrame(dist,
        columns=areas[:self.nodes], index=areas[:self.nodes]
      )
      sns.clustermap(
        data, method=self.linkage,
        xticklabels=True, yticklabels=True,
        cmap=sns.cm.rocket_r,
        figsize=(20, 17)
      )
      fname = f"{self.plot_path}/Features"
      Path(fname).mkdir(exist_ok=True, parents=True)
      plt.savefig(
        f"{fname}/heatmap_feature_distance.png",
        dpi=200
      )
      plt.close()
    else: print("No feature distance heatmap with dendrogram")

  def plot_flatmap_220830(self, R, Z, on=False, **kwargs):
    if on:
      print("Plot flatmap!!! 220830")
      from modules.colregion import colregion
      from modules.flatmap import FLATMAP
      fname = f"{self.plot_path}/flatmap"
      Path(fname).mkdir(exist_ok=True, parents=True)
      for r in R:
        partition = cut_tree(Z, r).ravel()
        new_partition = self.skim_partition(partition)
        L = colregion(self.net)
        L.get_regions()
        F = FLATMAP(
          self.net, L.regions, **kwargs
        )
        F.set_para(0, r, new_partition)
        F.plot_flatmap()
    else: print("No flatmap 220830")

  def plot_flatmap_40d91(self, clusters, partition, areas, version=220830, method="single"):
    from python.networks.structure import MAC
    from python.modules.colregion import colregion
    from python.modules.flatmap import FLATMAP
    NET = MAC(
      "average", "no_mode", True, False,
      csv_path="CSV",
      regions_path="CSV/Regions/Table_areas_regions_09_2019.csv",
      plot_path=f"python/DTW/plots/{version}/{method}"
    )
    struct_areas = areas.copy()
    struct_areas[areas == "sub"] = "subi"
    struct_areas[areas == "v1"] = "v1c"
    struct_areas[areas == "v2"] = "v2c"
    struct_areas[areas == "v4"] = "v4c"
    struct_areas[areas == "ins"] = "insula"
    struct_areas[areas == "mt"] = "mtc"
    NET.struct_labels = struct_areas
    ##
    from collections import Counter
    fq = Counter(partition)
    for i in fq.keys():
      if fq[i] == 1: partition[partition == i] = -1
    new_partition = partition
    ndc = np.unique(partition[partition != -1])
    for i, c in enumerate(ndc):
      new_partition[partition == c] = i
    ##
    NET.overlap = np.array(["UNKNOWN"])
    L = colregion(NET)
    L.get_regions()
    F = FLATMAP(
      NET, L.regions
    )
    F.set_para(0, clusters, new_partition)
    F.plot_flatmap(
      flatmap_path="python/utils/flatmap/FlatmapCoordinates_NewSeg_2022.csv"
    )
    data=pd.DataFrame(
      {
        "areas" : NET.struct_labels[:40],
        "id" : new_partition[:40]
      }
    )
    print(data.sort_values(by="id").to_numpy())

  def plot_dtw_both(self, areas, **kwargs):
    # Create data ----
    data = pd.DataFrame(
      {
        "dtw" : self.feature_dist,
        "areas" : areas[:self.W1.shape[1]]
      }
    ).sort_values(by="dtw")
    _, ax = plt.subplots(2, 1, figsize=(18,8))
    sns.histplot(
      data=data,
      x="dtw",
      ax=ax[0]
    )
    sns.scatterplot(
      data=data,
      x="areas",
      y="dtw",
      ax=ax[1]
    )
    ax[1].set_xticklabels(data.areas, rotation=90)
    # Saving ----
    fname = f"{self.plot_path}/Features"
    Path(fname).mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{fname}/dtw_src_tgt.png", dpi=200)

  def core_dendrogram(self, areas, step, Z, **kwargs):
    ## Save ----
    fname = f"{self.plot_path}/Features"
    Path(fname).mkdir(exist_ok=True, parents=True)
    from scipy.cluster import hierarchy
    for stp in step:
      clusters = self.nodes - 1 - stp
      fig, _ = plt.subplots(1, 1)
      hierarchy.dendrogram(
        Z, labels=areas, color_threshold=Z[stp+1, 2],
        leaf_rotation=90, leaf_font_size=10
      )
      fig.set_figwidth(15)
      fig.set_figheight(7)
      plt.savefig(
        f"{fname}/core_dendrogram_{clusters}.png", dpi=200
      )
      plt.close()

  def plot_einout_trends(self, W, D, areas, version=220830):
    inj = W.shape[1]
    dist = D.copy()
    w = W.copy()
    w[w != 0] = 1
    w = w[:inj, :]
    dist = dist[:inj, :inj]
    np.fill_diagonal(w, np.nan)
    np.fill_diagonal(dist, np.nan)
    leaves = len(w.ravel())
    data = pd.DataFrame(
      {
        "direction" : ["source"] * leaves,
        "existence" : w.ravel(),
        "distance" : dist.ravel(),
        "areas" : np.tile(areas[:inj], inj)
      }
    )
    data = pd.concat(
      [
        data,
        pd.DataFrame(
          {
            "direction" : ["target"] * leaves,
            "existence" : w.ravel(),
            "distance" : dist.ravel(),
            "areas" : np.tile(areas[:inj], inj)
          }
        )
      ],
      ignore_index=True
    )
    data = data.loc[~np.isnan(data.existence)]
    order = np.argsort(areas[:inj])
    g = sns.FacetGrid(
      data=data,
      col="areas",
      col_wrap=8,
      col_order=areas[:inj][order],
      hue="direction",
      height=2,
      aspect=1
    )
    g.map_dataframe(
      sns.scatterplot,
      x="distance",
      y="existence",
      s=3,
      alpha=0.5
    )
    g.add_legend()
    plt.savefig(f"python/DTW/plots/{version}/binary_trend_both.png", dpi=400)
    plt.close()

  def plot_ee_trends(self, W, D, areas, version=220830):
    inj = W.shape[1]
    w = W.copy()
    dist = D.copy()
    w[w != 0] = 1
    w = w[:inj, :]
    dist = dist[:inj, :inj]
    np.fill_diagonal(w, np.nan)
    data = pd.DataFrame(
      {
        "areas" : np.repeat(areas[:inj], inj),
        "metric" : np.abs(w.ravel() - w.T.ravel()),
        "distance" : dist.ravel()
      }
    )
    data = data.loc[~np.isnan(data.metric)]
    order = np.argsort(areas[:inj])
    g = sns.FacetGrid(
      data=data,
      col="areas",
      col_wrap=8,
      col_order=areas[:inj][order],
      height=2,
      aspect=1
    )
    g.map_dataframe(
      sns.lineplot,
      x="distance",
      y="metric",
    )
    g.add_legend()
    plt.savefig(f"python/DTW/plots/{version}/binary_eetrend_both.png", dpi=400)
    plt.close()

  def plot_mean_trend_histo(self, R, Z, n=6, axis=0, on=False):
    if on:
      print("Plot mean_trend_histo!!!")
      for r in R:
        W = self.W1.copy()
        W[W == 0] = np.nan
        dist = self.D.copy()[:, :self.inj]
        dist[dist == 0] = np.nan
        ## Merge single communities ----
        partition = cut_tree(Z, r).ravel()
        new_partition = self.skim_partition(partition)
        ##
        unique_clusters_id = np.unique(new_partition)
        num_clusters = len(unique_clusters_id)
        num_nodes_id = np.zeros(num_clusters)
        W_ave = np.zeros((num_clusters, n))
        Dist_ave = np.zeros((num_clusters, n))
        ##
        for i, id in enumerate(unique_clusters_id):
          x = new_partition == id
          num_nodes_id[i] = np.sum(x)
          nodes_index_id = np.where(x)[0]
          count_distance_id = np.zeros(n)
          ##
          subdist = dist[nodes_index_id, :].copy()
          min_dist = np.nanmin(subdist[subdist > 0])
          max_dist = np.nanmax(subdist)
          delta = (max_dist - min_dist) / (n - 1)
          distance_range = np.arange(
            min_dist - (delta /2), max_dist + delta + 0.00001, delta
          )
          Dist_ave[i, :] = distance_range[:-1] + delta / 2
          ##
          for indx in nodes_index_id:
            for j in np.arange(n):
              if axis == 0:
                distance_in_j_bin = (dist[indx, :] > distance_range[j]) & (dist[indx, :] < distance_range[j+1])
                w_temp = np.nanmean(W[indx, distance_in_j_bin])
              elif axis == 1:
                distance_in_j_bin = (dist[:, indx] > distance_range[j]) & (dist[:, indx] < distance_range[j+1])
                w_temp = np.nanmean(W[distance_in_j_bin, indx])
              else: w_temp = np.nan
              if ~np.isnan(w_temp) and w_temp < 0:
                W_ave[i, j] += w_temp
                count_distance_id[j] += 1
                # count_distance_id[j] += np.sum(~np.isnan(W[indx, distance_in_j_bin]))
          W_ave[i, count_distance_id > 0] = W_ave[i, count_distance_id > 0] / count_distance_id[count_distance_id > 0]
        W_ave[W_ave == 0] = np.nan
        ## Create data
        data = pd.DataFrame(
          {
            "dist" : Dist_ave.ravel(),
            "w_ave" : W_ave.ravel(),
            "cluster" : np.repeat(unique_clusters_id, n),
            "num_nodes" : np.repeat(num_nodes_id, n)
          }
        )
        data = data.loc[~np.isnan(data.w_ave)]
        ## Color ----
        cm = sns.color_palette("hls", len(unique_clusters_id))
        color_dic = {}
        size_dic = {}
        for i, id in enumerate(unique_clusters_id[unique_clusters_id != -1]):
          color_dic[id] = cm[i]
          size_dic[id] = 2.5
        color_dic[-1] = (0.5, 0.5, 0.5)
        size_dic[-1] = 2.5
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
        sns.lineplot(
          data=data,
          x="dist",
          y="w_ave",
          hue="cluster",
          size="cluster",
          style="cluster",
          sizes=size_dic,
          palette=color_dic,
          markers=True,
          alpha=0.7,
          ax=ax[0]
        )
        sns.move_legend(ax[0], "lower center", bbox_to_anchor=(.5, 1), ncol=6)
        sns.barplot(
          data=data,
          x="cluster",
          y="num_nodes",
          # hue="cluster",
          palette=color_dic,
          alpha=1,
          dodge=False,
          ax=ax[1]
        ) 
        fig.tight_layout()
        # Save ----
        fname = f"{self.plot_path}/Features"
        Path(fname).mkdir(exist_ok=True, parents=True)
        plt.savefig(
          f"{fname}/average_w_trend_{r}_n_{n}.png", dpi=200
        )
        plt.close()
    else:
      print("No mean_trend_histo")
  
  def plot_mean_trend_hist_area(self, R, Z, n=6, axis=0, on=False):
    if on:
      print("Plot mean_trend_histo w area!!!")
      for r in R:
        W = self.W1.copy()
        W[W == 0] = np.nan
        dist = self.D.copy()[:, :self.inj]
        dist[dist == 0] = np.nan
        ## Merge single communities ----
        partition = cut_tree(Z, r).ravel()
        new_partition = self.skim_partition(partition)
        ##
        unique_clusters_id = np.unique(new_partition)
        num_clusters = len(unique_clusters_id)
        num_nodes_id = np.zeros(num_clusters)
        W_ave = np.zeros((num_clusters, n, self.nodes)) * np.nan
        Dist_ave = np.zeros((num_clusters, n, self.nodes)) * np.nan
        ##
        for i, id in enumerate(unique_clusters_id):
          x = new_partition == id
          num_nodes_id[i] = np.sum(x)
          nodes_index_id = np.where(x)[0]
          ##
          subdist = dist[nodes_index_id, :].copy()
          min_dist = np.nanmin(subdist[subdist > 0])
          max_dist = np.nanmax(subdist)
          delta = (max_dist - min_dist) / (n - 1)
          distance_range = np.arange(
            min_dist - (delta /2), max_dist + delta + 0.00001, delta
          )
          Dist_ave[i, :, :] = np.repeat(distance_range[:-1] + delta / 2, self.nodes).reshape(n, self.nodes)
          ##
          for ni, indx in enumerate(nodes_index_id):
            for j in np.arange(n):
              if axis == 0:
                distance_in_j_bin = (dist[indx, :] > distance_range[j]) & (dist[indx, :] < distance_range[j+1])
                w_temp = np.nanmean(W[indx, distance_in_j_bin])
              elif axis == 1:
                distance_in_j_bin = (dist[:, indx] > distance_range[j]) & (dist[:, indx] < distance_range[j+1])
                w_temp = np.nanmean(W[distance_in_j_bin, indx])
              else: w_temp = np.nan
              if ~np.isnan(w_temp) and w_temp < 0:
                W_ave[i, j, ni] = w_temp
        ## Create data
        data = pd.DataFrame(
          {
            "dist" : Dist_ave.ravel(),
            "w_ave" : W_ave.ravel(),
            "cluster" : np.repeat(np.repeat(unique_clusters_id, n), self.nodes),
            "num_nodes" : np.repeat(np.repeat(num_nodes_id, n), self.nodes)
          }
        )
        data = data.loc[~np.isnan(data.w_ave)]
        ## Color ----
        cm = sns.color_palette("hls", len(unique_clusters_id))
        color_dic = {}
        size_dic = {}
        for i, id in enumerate(unique_clusters_id[unique_clusters_id != -1]):
          color_dic[id] = cm[i]
          size_dic[id] = 2.5
        color_dic[-1] = (0.5, 0.5, 0.5)
        size_dic[-1] = 2.5
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
        sns.lineplot(
          data=data,
          x="dist",
          y="w_ave",
          hue="cluster",
          size="cluster",
          style="cluster",
          sizes=size_dic,
          palette=color_dic,
          markers=True,
          alpha=0.7,
          ax=ax[0]
        )
        sns.move_legend(ax[0], "lower center", bbox_to_anchor=(.5, 1), ncol=6)
        sns.barplot(
          data=data,
          x="cluster",
          y="num_nodes",
          # hue="cluster",
          palette=color_dic,
          alpha=1,
          dodge=False,
          ax=ax[1]
        ) 
        fig.tight_layout()
        # Save ----
        fname = f"{self.plot_path}/Features"
        Path(fname).mkdir(exist_ok=True, parents=True)
        plt.savefig(
          f"{fname}/average_w_trend_area_{r}_n_{n}.png", dpi=200
        )
        plt.close()
    else:
      print("No mean_trend_histo w area")

  def plot_features_scatterplot(self, dtw_dist, jacp_dist, **kwargs):
    # Create data ----
    data = pd.DataFrame(
      {
        "dtw" : dtw_dist.ravel(),
        "jacp" : jacp_dist.ravel()
      }
    )
    data = data.loc[(data.dtw != 0) & (data.jacp != 0)]
    ## Create figure ----
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
      data=data,
      x="dtw",
      y="jacp",
      s=5,
      ax=ax
    )
    fig.tight_layout()
    # Save ----
    fname = "python/DTW/plots/{}/{}/{}/{}/{}".format(
      kwargs["version"], kwargs["nature"], kwargs["model"], kwargs["dir"], kwargs["linkage"]
    )
    Path(fname).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      f"{fname}/two_features.png",
      dpi=200
    )
    plt.close()
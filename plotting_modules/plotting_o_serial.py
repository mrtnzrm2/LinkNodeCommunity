# Standard libs ----
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import networkx as nx
from pathlib import Path
from os.path import join
# Personal libs ----
from various.network_tools import *
from plotting_modules.plotting_serial import PLOT_S

class PLOT_OS(PLOT_S):
  def __init__(self, hrh) -> None:
    super().__init__(hrh)
    self.data_overlap = hrh.data_overlap
    self.association_one = hrh.association_one
    self.Z = hrh.Z
    self.kr = hrh.kr
    self.nodes = hrh.nodes
    self.association_zero = hrh.association_zero
    self.labels = hrh.labels

  def association_heatmap(self, score, direction, on=True, **kwargs):
    if on:
      print(f"Plot association {score} matrix")
      from scipy.cluster.hierarchy import dendrogram, cut_tree
      # Getting the r of the score ----
      r = [a for a in self.kr.R.loc[(self.kr.score == score) & (self.kr.data == "1")]]
      r = r[0]
      # Getting the dendrogram order of Z ----
      one_order = np.array(dendrogram(self.Z, no_plot=True)["ivl"]).astype(int)
      # Preparing the labels of Z at r -----
      one_rlabel = cut_tree(self.Z, n_clusters=r).reshape(-1)
      one_rlabel = skim_partition(one_rlabel)[one_order]
      C = [i+1 for i in np.arange(len(one_rlabel)-1) if one_rlabel[i] != one_rlabel[i+1]]
      D = np.where(one_rlabel == -1)[0] + 1
      C = list(set(C).union(set(list(D))))
      # Preparing matrix ----
      zero_matrix = self.association_zero[direction][score][one_order,:][:, one_order]
      zero_matrix[zero_matrix == 0] = np.nan
      one_labels = self.labels[one_order]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.heatmap(
        zero_matrix,
        xticklabels=one_labels,
        yticklabels=one_labels,
        cmap=sns.color_palette("mako", as_cmap=True),
        ax=ax
      )
      # Add lines denoting communities ----
      for c in C:
        ax.vlines(
          c, ymin=0, ymax=self.nodes,
          linewidth=1.5,
          colors=["#C70039"]
        )
        ax.hlines(
          c, xmin=0, xmax=self.nodes,
          linewidth=1.5,
          colors=["#C70039"]
        )
      if "font_size" in kwargs.keys():
        if kwargs["font_size"] > 0:
          ax.set_xticklabels(
            ax.get_xmajorticklabels(), fontsize = kwargs["font_size"]
          )
          ax.set_yticklabels(
            ax.get_ymajorticklabels(), fontsize = kwargs["font_size"]
          )
      # Configure figure ----
      fig.set_figwidth(11.5)
      fig.set_figheight(9.5)
      # Arrange path ----
      plot_path = join(self.plot_path, f"Features")
      # Crate path ----
      Path(plot_path).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(plot_path, f"association_{score}_{direction}.png"), dpi=300
      )
      plt.close()
    else: print(f"No {score} association matrix")

  def sln_matrix_KS(self, data_sln):
    data_sln_one = data_sln["1"]
    data_sln_zero = np.array(data_sln["shuffle"])

    Z = data_sln_one.shape[0]
    L = data_sln_zero.shape[0]

    sln_zero_mean_noravel = []
    for i in np.arange(L):
      sln_zero_mean_noravel.append(np.sort(data_sln_zero[i, :, :].ravel())) 
    sln_zero_mean_noravel = np.array(sln_zero_mean_noravel)

    sln_zero_mean = np.mean(sln_zero_mean_noravel, axis=0)
    sln_zero_std = np.std(sln_zero_mean_noravel, axis=0)


    data_sln_one = np.sort(data_sln_one.ravel())

    nb_elements = data_sln_one.shape[0]
    elements = np.arange(nb_elements)

    significance_array = np.array([""] * nb_elements, dtype="<U21")

    from scipy.stats import ttest_1samp
    for i in np.arange(Z**2):
        _, pval = ttest_1samp(sln_zero_mean_noravel[:, i], data_sln_one[i])
        pval = pvalue2asterisks(pval)
        significance_array[i] = pval

    

    sns.set(style='whitegrid')

    cmp = sns.color_palette("deep")

    asterisk_height = np.maximum(data_sln_one, sln_zero_mean) + 0.01

    scat_one = plt.scatter(elements, data_sln_one, color=cmp[0], label="loop")
    scat_zero = plt.errorbar(elements, sln_zero_mean, yerr=sln_zero_std, fmt='o', color=cmp[1], ecolor=cmp[1], capsize=5, capthick=2, label="random shuffle")

    plt.gca().legend(handles=[scat_one, scat_zero])

    area_one = np.trapz(elements, data_sln_one)
    area_zero = np.trapz(elements, sln_zero_mean)

    [plt.text(e, h, s, horizontalalignment="center") for e, h, s in zip(elements, asterisk_height, significance_array)]

    frac_area = (area_zero)/area_one * 100
    plt.gca().set_title(f"Area percentage = {frac_area:.2f}% [shuffle/loop]")
    plt.gca().set_ylabel(r"$\left<SLN\right>$")

    fig = plt.gcf()
    fig.set_figwidth(10)
    fig.set_figheight(7)
    
    # Arrange path ----
    plot_path = join(self.plot_path, f"Features")
    # Crate path ----
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    # Save plot ----
    plt.savefig(
      join(plot_path, "sln_shuffle_test.svg"),
      transparent=True,
      # dpi=300
    )
    plt.close()




  def sln_matrix_check(self, data_sln):
    data_sln_one = data_sln["1"]
    data_sln_zero = np.array(data_sln["conf"])

    Z = data_sln_one.shape[0]
    L = data_sln_zero.shape[0]

    membership_matrix = np.arange(Z**2).reshape(Z,Z)

    xlabel = r"$<SLN>$"
    data = pd.DataFrame()
    for i in np.arange(Z):
      for j in np.arange(Z):
        # if i == j:
          data = pd.concat(
            [
              data,
              pd.DataFrame(
                {
                  "group" : [f"{membership_matrix[i,j]}"] * L,
                  xlabel :  data_sln_zero[:, i, j].ravel()
                }
              )
            ], ignore_index=True
          )

    g = sns.FacetGrid(
      data=data,
      col="group",
      col_wrap= Z
    )


    g.map_dataframe(
      sns.histplot,
      x=xlabel
    )

    from scipy.stats import ttest_1samp
    sln_significance = np.array([""]* Z**2, dtype="<U21").reshape(Z,Z)

    for ax in g.axes.flatten():
      title = ax.get_title().split(" = ")[-1]
      title = int(title)
      ix, iy = np.where(membership_matrix == title)
      ix = int(ix[0])
      iy = int(iy[0])

      x = data[xlabel].loc[data["group"]  == f"{title}"].to_numpy()

      if np.sum(np.isnan(x)) > 0:
        x = x[~np.isnan(x)]
        raise RuntimeError("There are nans in the array.")
      
      r, pval = ttest_1samp(x, data_sln_one[ix, iy])

      ax.axvline(data_sln_one[ix, iy], linestyle="--", color="red")

      sln_significance[ix, iy] = pvalue2asterisks(pval)
 
      pval = pvalue2asterisks(pval)
      ax.set_title(f"group = {ix}|{iy}" + "\n" + fr"{pval}")

    # Arrange path ----
    plot_path = join(self.plot_path, f"Features")
    # Crate path ----
    Path(plot_path).mkdir(exist_ok=True, parents=True)
    # Save plot ----
    plt.savefig(
      join(plot_path, "sln_conf_test.svg"),
      transparent=True,
      # dpi=300
    )
    plt.close()

    annotate_sln = np.array([""]*Z**2, dtype="<U21")
    for i, (av, pval) in enumerate(zip(data_sln_one.ravel(), sln_significance.ravel())):
      annotate_sln[i] = f"{av:.2f}\n{pval}"

    annotate_sln = annotate_sln.reshape(Z, Z)

    import matplotlib
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#b20000","#cca3ff","#0047AB"])

    sns.heatmap(
      data_sln_one,
      annot=annotate_sln,
      fmt="", 
      cmap=cmap,
      alpha=0.7,
      center=0.5
    )

    xlabels = plt.gca().get_xticklabels()
    xlabels = [f"C{i.get_text()}" for i in xlabels]
    plt.gca().set_xticklabels(xlabels)

    ylabels = plt.gca().get_yticklabels()
    ylabels = [f"C{i.get_text()}" for i in ylabels]
    plt.gca().set_yticklabels(ylabels)

    plt.gcf().tight_layout()

    plt.savefig(
      f"{plot_path}/sln_conf_clusters.svg",
      transparent=True
    )

    plt.close()

  

  def association_heatmap_zero(self, score, direction, on=True, **kwargs):
    if on:
      print(f"Plot association {score} matrix")
      from scipy.spatial.distance import squareform
      from scipy.cluster.hierarchy import dendrogram, cut_tree, linkage
      # Getting the r of the score ----
      r = [a for a in self.kr.R.loc[(self.kr.score == score) & (self.kr.data != "1")]]
      r = r[0]
      # Getting Z at r -----
      Z = self.association_zero[direction][score]
      Z[Z == 0] = np.nan
      Z = 1 / Z
      Z[np.isnan(Z)] = np.nanmax(Z) + 0.1
      np.fill_diagonal(Z, 0)
      Z = linkage(squareform(Z))
      # Getting the dendrogram order of Z ----
      one_order = np.array(dendrogram(Z, no_plot=True)["ivl"]).astype(int)
      # Preparing the labels of Z at r -----
      one_rlabel = cut_tree(Z, n_clusters=r).reshape(-1)
      one_rlabel = skim_partition(one_rlabel)[one_order]
      C = [i+1 for i in np.arange(len(one_rlabel)-1) if one_rlabel[i] != one_rlabel[i+1]]
      D = np.where(one_rlabel == -1)[0] + 1
      C = list(set(C).union(set(list(D))))
      # Preparing matrix ----
      zero_matrix = self.association_zero[direction][score][one_order,:][:, one_order]
      zero_matrix[zero_matrix == 0] = np.nan
      one_labels = self.labels[one_order]
      # Create figure ----
      fig, ax = plt.subplots(1, 1)
      sns.heatmap(
        zero_matrix,
        xticklabels=one_labels,
        yticklabels=one_labels,
        cmap=sns.color_palette("mako", as_cmap=True),
        ax=ax
      )
      # Add lines denoting communities ----
      for c in C:
        ax.vlines(
          c, ymin=0, ymax=self.nodes,
          linewidth=1.5,
          colors=["#C70039"]
        )
        ax.hlines(
          c, xmin=0, xmax=self.nodes,
          linewidth=1.5,
          colors=["#C70039"]
        )
      if "font_size" in kwargs.keys():
        if kwargs["font_size"] > 0:
          ax.set_xticklabels(
            ax.get_xmajorticklabels(), fontsize = kwargs["font_size"]
          )
          ax.set_yticklabels(
            ax.get_ymajorticklabels(), fontsize = kwargs["font_size"]
          )
      # Configure figure ----
      fig.set_figwidth(11.5)
      fig.set_figheight(9.5)
      # Arrange path ----
      plot_path = join(self.plot_path, f"Features")
      # Crate path ----
      Path(plot_path).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(plot_path, f"association_{score}_{direction}_zero_order.png"), dpi=300
      )
      plt.close()
    else: print(f"No {score} association matrix")

  def cover_flatmap_association(self, nodes, version, regions, score, direction, cmap="hls", on=True, **kwargs):
    if on:
      print(f"Plot flatmap association {score}")
      from modules.flatmap import FLATMAP
      from scipy.cluster.hierarchy import cut_tree, linkage
      from scipy.spatial.distance import squareform
      # Getting Z at r -----
      Z = self.association_zero[direction][score]
      Z[Z == 0] = np.nan
      Z = 1 / Z
      Z[np.isnan(Z)] = np.nanmax(Z) + 0.1
      np.fill_diagonal(Z, 0)
      Z = linkage(squareform(Z))
      # Get partition ----
      mean_r = self.kr["R"].loc[(self.kr.data != "1") & (self.kr.score == score)].mean(skipna=True)
      mean_r = int(mean_r)
      print(f"Mean R:\t{mean_r}")
      node_partition = cut_tree(Z, n_clusters=mean_r).ravel()
      node_partition = skim_partition(node_partition)
      # Create figure ----
      plot_path = join(self.plot_path, f"Features")
      F = FLATMAP(nodes, version, self.labels, regions, plot_path, **kwargs)
      F.set_para("association", mean_r, node_partition)
      F.plot_flatmap(direction=direction, cmap_name=cmap)

    else: print(f"No flatmap association {score}")

  def histogram_overlap_scores(self, on=True, **kwargs):
    if on:
      print("Plot overlap scores histogram!!!")
      print(
        "Mean acc1: {:7f}\nMean acc2: {:.7f}".format(
          self.data_overlap["ACC1"].mean(),
          self.data_overlap["ACC2"].mean()
        )
      )
      # Create figure ----
      if "c" in kwargs.keys():
        if kwargs["c"]:
          self.data_overlap["c"] = [s.replace("_", "") for s in self.data_overlap["c"]]
          fig, ax = plt.subplots(1,2)
          sns.histplot(
            data=self.data_overlap,
            x = "ACC1",
            ax = ax[0],
            hue = "c"
          )
        else:
          fig, ax = plt.subplots(1,2)
          sns.histplot(
            data=self.data_overlap,
            x = "ACC1",
            ax = ax[0]
          )
      else:
        fig, ax = plt.subplots(1,2)
        sns.histplot(
          data=self.data_overlap,
          x = "ACC1",
          ax = ax[0]
        )
      ax[0].text(
        0.5, 1.05,
        "Average: {:.4f}   Std: {:.4f}".format(
          self.data_overlap["ACC1"].mean(),
          self.data_overlap["ACC1"].std()
        ),
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax[0].transAxes
      )
      ax[0].set(xlabel="tp/total")
      # Create figure ACC2 ----
      if "c" in kwargs.keys():
        if kwargs["c"]:
          sns.histplot(
            data=self.data_overlap,
            x = "ACC2",
            ax = ax[1],
            hue = "c"
          )
        else:
          sns.histplot(
            data=self.data_overlap,
            x = "ACC2",
            ax = ax[1]
          )
      else:
        sns.histplot(
          data=self.data_overlap,
          x = "ACC2",
          ax = ax[1]
        )
      ax[1].text(
        0.5, 1.05,
        "Average: {:.4f}    Std: {:.4f}".format(
          self.data_overlap["ACC2"].mean(),
          self.data_overlap["ACC2"].std()
        ),
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax[1].transAxes
      )
      ax[1].set(xlabel="fp/pred")
      # figure custom ---
      fig.set_figwidth(12)
      # Arrange path ----
      plot_path = join(
        self.plot_path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Name ----
      name = ""
      if "name" in kwargs.keys():
        name = kwargs["name"]
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "overlap_scores_{}{}.png".format(
            self.linkage, name
          )
        ),
        dpi=300
      )

  def histogram_overlap(self, score, direction, on=True, **kwargs):
    if on:
      print("Histogram overlap frequency!!!")
      subdata = self.data_overlap.loc[
        (self.data_overlap.score == score) & (self.data_overlap.direction == direction)
      ]
      fig, ax = plt.subplots(1,1)
      sns.histplot(
        data = subdata,
        x = "Areas",
        hue = "data",
        stat = "count",
        common_norm=False,
        ax = ax
      )
      # Rotate axis ----
      plt.xticks(rotation=90)
      # Get areas from data ----
      data_areas = subdata.loc[
        subdata.data == "1",
        "Areas"
       ].to_numpy()
      for i in np.arange(len(data_areas)):
        ax.axvline(data_areas[i], color="red")
      fig.set_figheight(9)
      fig.set_figwidth(15)
      # Arrange path ----
      plot_path = join(
        self.plot_path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "overlap_freq_{}_{}_{}.png".format(
            self.linkage, score, direction
          )
        ),
        dpi=300
      )
      plt.close()
  
  def core_dendrogram(self, Z, R : list, score="_S", cmap_name="hls", leaf_font_size=20, remove_labels=False, figwidth=13, figheight=7, on=True, **kwargs):
    if on:
      print("Plot mean core dendrogram!!!")
      from scipy.cluster import hierarchy
      import matplotlib.colors
      # Create figure ----
      for r in R:
        if r == 1:
          r += 1
        partition = hierarchy.cut_tree(Z, r).ravel()
        new_partition = skim_partition(partition)
        unique_clusters_id = np.unique(new_partition)
        cm = sns.color_palette(cmap_name, len(unique_clusters_id))
        # dlf_col = "#808080"
        dlf_col = "#808080"
        ##
        D_leaf_colors = {}
        for i, _ in enumerate(self.labels):
          if new_partition[i] != -1:
            D_leaf_colors[i] = matplotlib.colors.to_hex(cm[new_partition[i]])
          else: D_leaf_colors[i] = dlf_col
        ##
        link_cols = {}
        for i, i12 in enumerate(Z[:,:2].astype(int)):
          c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x]
            for x in i12)
          link_cols[i+1+len(Z)] = c1 if c1 == c2 else dlf_col
        # plt.style.use("dark_background")
        sns.set_context("talk")
        fig, ax = plt.subplots(1, 1)
        ax.grid(False)
        if not remove_labels:
          hierarchy.dendrogram(Z,
            labels=self.labels,
            color_threshold=Z[len(self.labels) - r, 2],
            link_color_func = lambda k: link_cols[k],
            leaf_rotation=90, leaf_font_size=leaf_font_size, **kwargs
          )
        else:
          hierarchy.dendrogram(Z,
            no_labels=True,
            color_threshold=Z[len(self.labels) - r, 2],
            link_color_func = lambda k: link_cols[k]
          )
        fig.set_figwidth(figwidth)
        fig.set_figheight(figheight)
        plt.ylabel("Height " + r"$(H^{2})$")
        fig.tight_layout()
        sns.despine()
        # Arrange path ----
        plot_path = join(
          self.plot_path, "Features"
        )
        # Crate path ----
        Path(
          plot_path
        ).mkdir(exist_ok=True, parents=True)
        # Save plot ----
        plt.savefig(
          join(
            plot_path, "mean_core_dendrogram_{}.png".format(score)
          ),
          dpi=300
        )
        plt.close()
    else:
      print("No mean core dendrogram")

  def plot_network_covers(self, R : npt.ArrayLike, partition, nocs : dict, sizes : dict, ang=0, score="_S", direction="", cmap_name="hls", figsize=(12,12), spring=False, on=True, **kwargs):
    if on:
      print("Printing network space")
      import matplotlib.patheffects as path_effects
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
        index_key = np.where(self.labels == key)[0][0]
        for id in nocs[key]:
          if id == -1:
            nodes_memberships[index_key]["id"][0] = 1
            nodes_memberships[index_key]["size"][0] = 1
          else:
            nodes_memberships[index_key]["id"][id + 1] = 1
            nodes_memberships[index_key]["size"][id + 1] = sizes[key][id]
      # Check unassigned ----
      for i in np.arange(len(self.labels)):
        if np.sum(np.array(nodes_memberships[i]["id"]) == 1) == 0:
          nodes_memberships[i]["id"][0] = 1
          nodes_memberships[i]["size"][0] = 1
      # Generate graph ----
      G = nx.from_numpy_array(R, create_using=nx.DiGraph)
      if "coords" not in kwargs.keys():
        pos = nx.kamada_kawai_layout(G, weight="weight")
        if spring:
          Rinv = 1 - R
          np.fill_diagonal(Rinv, 0)
          Ginv = nx.DiGraph(Rinv)
          pos = nx.spring_layout(Ginv, weight="weight", pos=pos, iterations=5, seed=212)
      else:
        pos = kwargs["coords"]
      ang = ang * np.pi/ 180
      rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
      pos = {k : np.matmul(rot, pos[k]) for k in pos.keys()}
      labs = {k : lab for k, lab in zip(G.nodes, self.labels)}

      mu_pos_x = np.mean([k[0] for k in pos.values()])
      mu_pos_y = np.mean([k[1] for k in pos.values()])
      mu_pos = np.array([mu_pos_x, mu_pos_y])

      pos = {k : pos[k] - mu_pos for k in pos.keys()}
      pos = {k : pos[k] * 1.5 for k in pos.keys()}
      
      _, ax = plt.subplots(1, 1, figsize=figsize)
      if "not_edges" not in kwargs.keys():
        nx.draw_networkx_edges(
          G, pos=pos, edge_color="#666666", alpha=0.5, width=2, arrowsize=10, connectionstyle="arc3,rad=-0.1",
          node_size=1400, ax=ax
        )
      if "modified_labels" not in kwargs.keys():
        t = nx.draw_networkx_labels(G, pos=pos, labels=labs, font_color="white", ax=ax)
        for key in t.keys():
          t[key].set_path_effects(
          [
            path_effects.Stroke(linewidth=1, foreground='black'),
            path_effects.Normal()
          ]
        )
      else:
        t = nx.draw_networkx_labels(G, pos=pos, labels=kwargs["modified_labels"], font_color="white", ax=ax)
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
      plt.xlim(-0.3 + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + 0.3)
      plt.ylim(-0.1 + np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + 0.1)
      # Arrange path ----
      plot_path = join(
        self.plot_path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "mean_network_{}_{}.png".format(score, direction)
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No network plot.")

  def plot_newick(self, newick : str, colregion, width=8, height=7, fontsize=10, branches_color="#666666", on=True):
    if on:
      print("Plot newick dendrogram!!!")
      from Bio import Phylo
      from io import StringIO
      tree = Phylo.read(StringIO(newick), "newick")
      tree.ladderize()
      
      plt.style.use("dark_background")
      sns.set_context("talk")
      _, ax = plt.subplots(1, 1, figsize=(width, height))
      
      ax.grid(False)
      area = colregion.regions.AREA.to_numpy().astype(str)
      color = colregion.regions.COLOR.to_numpy()
      color_tip = {k: v for k, v in zip(area, color)}
      Phylo.draw(
        tree, axes=ax, label_colors=color_tip,
        do_show=False, fontsize=fontsize, branches_color=branches_color
      )
      ax.set_ylabel("")
      ax.yaxis.set_tick_params(labelleft=False)
      # Arrange path ----
      plot_path = join(
        self.plot_path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "mean_newick_H_dendro.png"
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No newick dendrogram.")
  
  def plot_heatmap_Z(self, D, on=True):
    if on:
      print("Plot heatmapZ!!!")
      DD = D.copy()
      np.fill_diagonal(DD, np.nan)
      _, ax = plt.subplots(1, 1, figsize=(10, 8))
      sns.heatmap(
        DD,
        xticklabels=self.labels,
        yticklabels=self.labels,
        ax=ax
      )

      ax.set_xticklabels(
        ax.get_xmajorticklabels(), fontsize = 8
      )
      ax.set_yticklabels(
        ax.get_ymajorticklabels(), fontsize = 8
      )
      # Arrange path ----
      plot_path = join(
        self.plot_path, "Features"
      )
      # Crate path ----
      Path(
        plot_path
      ).mkdir(exist_ok=True, parents=True)
      # Save plot ----
      plt.savefig(
        join(
          plot_path, "mean_heatmapZ.png"
        ),
        dpi=300
      )
      plt.close()
    else:
      print("No heatmap Z")

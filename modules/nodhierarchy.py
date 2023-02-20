import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree
# My libraries ----
from networks.structure import MAC
from modules.nodanalysis import NODA
from various.network_tools import minus_one_Dc

class NODH(NODA):
  def __init__(self, NET: MAC, n, nlog10=True, lookup=False, **kwargs):
    super().__init__(NET, n, nlog10, lookup, **kwargs)
    self.topology = NET.topology
    self.linkage = NET.linkage
    self.feature_dist = self.select_feature(self.topology)
    self.BH = []
    self.minus_one_Dc = minus_one_Dc


  def hierarchical_clustering(self):
    from scipy.cluster.hierarchy import linkage as link
    self.Z = link(self.feature_dist, self.linkage)
    self.H = self.Z

  def cut_tree(self, r):
    return cut_tree(self.Z, r).ravel()

  def best_partition_similarity(self, save=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path
    sim = self.feature_dist.copy()
    nodes = sim.shape[0]
    np.fill_diagonal(sim, np.nan)
    sim = 1 - sim
    steps = np.arange(nodes - 1, 0, -1)
    part_sim = np.zeros((2, len(steps)))
    part_sim[0, :] = steps
    for i in np.arange(len(steps)):
      partition = cut_tree(self.Z, n_clusters=part_sim[0, i]).ravel()
      for id in np.unique(partition):
        in_group = partition == id
        g_id = np.nansum(in_group)
        if g_id > 1:
          lsim = sim[:, in_group][in_group, :]
          part_sim[1, i] += np.nanmean(lsim) * g_id / nodes # * (g_id * (g_id - 1))/ (nodes * (nodes - 1))
    arg = np.argmax(part_sim[1, :])
    if save:
      data = pd.DataFrame(
        {
          "nclusters" : part_sim[0, :],
          "average similarity" : part_sim[1, :]
        }
      )
      fig, ax = plt.subplots(1, 1)
      sns.lineplot(
        data = data,
        x="nclusters",
        y="average similarity",
        ax=ax
      )
      data_p = pd.DataFrame(
        {
          "nclusters" : [part_sim[0, arg]],
          "average similarity" : [part_sim[1, arg]]
        }
      )
      sns.scatterplot(
        data=data_p,
        x="nclusters",
        y="average similarity",
        color="red",
        s=20,
        ax=ax
      )
      fig.tight_layout()
      # Sve ----
      fname = f"{self.plot_path}/BestSimilarity"
      Path(fname).mkdir(exist_ok=True, parents=True)
      plt.savefig(
        "{}/{}.png".format(
          fname, part_sim[0, arg].astype(int)
        ),
        dpi=200
      )
      plt.close()
    return part_sim[0, arg].astype(int), arg

  def best_partition_X(self, save=False):
      import seaborn as sns
      import matplotlib.pyplot as plt
      from pathlib import Path
      from collections import Counter
      nodes = self.nodes
      steps = np.arange(nodes - 1, 0, -1)
      part_x = np.zeros((2, len(steps)))
      part_x[0, :] = steps
      for i in np.arange(len(steps)):
        partition = cut_tree(self.Z, n_clusters=part_x[0, i]).ravel()
        size_clusters = Counter(partition)
        value_partition = size_clusters.values()
        value_partition = np.array(list(value_partition))
        max_cluster = np.max(value_partition)
        N = np.sum(value_partition) ** 2
        x = [value_partition[i] for i in np.arange(value_partition.shape[0]) if value_partition[i] != max_cluster]
        x = np.array(x) ** 2 / N
        part_x[1, i] = np.sum(x)
      arg = np.argmax(part_x[1, :])
      if save:
        data = pd.DataFrame(
          {
            "nclusters" : part_x[0, :],
            "Susceptability" : part_x[1, :]
          }
        )
        fig, ax = plt.subplots(1, 1)
        sns.lineplot(
          data = data,
          x="nclusters",
          y="Susceptability",
          ax=ax
        )
        data_p = pd.DataFrame(
          {
            "nclusters" : [part_x[0, arg]],
            "Susceptability" : [part_x[1, arg]]
          }
        )
        sns.scatterplot(
          data=data_p,
          x="nclusters",
          y="Susceptability",
          color="red",
          s=20,
          ax=ax 
        )
        fig.tight_layout()
        # Sve ----
        fname = f"{self.plot_path}/BestSimilarity_X "
        Path(fname).mkdir(exist_ok=True, parents=True)
        plt.savefig(
          "{}/{}.png".format(
            fname, part_x[0, arg].astype(int)
          ),
          dpi=200
        )
        plt.close()
      return part_x[0, arg].astype(int), arg
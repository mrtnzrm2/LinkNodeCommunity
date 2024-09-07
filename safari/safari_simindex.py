# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ----
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# Personal libs ---- 
import ctools as ct
from networks.MAC.mac57 import MAC57
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
structure = "FLNe"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0
opt_score = ["_S"]
save_data = T
version = "57d106"
__nodes__ = 57
__inj__ = 57
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC57(
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    version = version,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = __inj__,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    b = bias
  )
  NET.create_plot_directory()

  N = NET.nodes
  A = NET.A
  D = NET.D[:N, :][:, :N]

  dissim_H2 = np.zeros((2, N, N)) # 0 source, 1 target
  dissim_cos = np.zeros((2, N, N))

  for i in np.arange(N):
    for j in np.arange(i+1, N):
      dissim_H2[0, i, j] = 1 - ct.Hellinger2(A[i, :], A[j, :], i, j)
      dissim_H2[1, i, j] = 1 - ct.Hellinger2(A[:, i], A[:, j], i, j)

      dissim_cos[0, i, j] = 1 - ct.cosine_similarity(A[i, :], A[j, :], i, j)
      dissim_cos[1, i, j] = 1 - ct.cosine_similarity(A[:, i], A[:, j], i, j)


      dissim_H2[0, j, i] = 1 - ct.Hellinger2(A[i, :], A[j, :], i, j)
      dissim_H2[1, j, i] = 1 - ct.Hellinger2(A[:, i], A[:, j], i, j)

      dissim_cos[0, j, i] = 1 - ct.cosine_similarity(A[i, :], A[j, :], i, j)
      dissim_cos[1, j, i] = 1 - ct.cosine_similarity(A[:, i], A[:, j], i, j)

  L = dissim_H2[0].ravel().shape[0]

  h2_label = r"$H^{2}$"
  ylabel = "Dissimilarity"
  xlabel = "Interareal tractography distance [mm]"
  cos_label = r"$\cos_{diss}$"

  data = pd.DataFrame(
    {
      ylabel : np.hstack([dissim_H2[0].ravel(), dissim_H2[1].ravel(), dissim_cos[0].ravel(), dissim_cos[1].ravel()]),
      xlabel : np.tile(D.ravel(), 4),
      "Direction" : ["+"] * L + ["-"] * L + ["+"] * L + ["-"] * L,
      "Index" : [h2_label] * 2 * L + [cos_label] * 2 * L
    }
  )

  data = data.loc[data[xlabel] != 0]

  sns.set_style("white")

  g = sns.FacetGrid(
    data=data,
    col="Direction",
    hue="Index"
  )

  g.map_dataframe(
    sns.scatterplot,
    x=xlabel,
    y=ylabel,
    alpha=0.4,
    s=7
  )

  g.add_legend()

  def little_area_trend(x, y, bins=13):
    minx = np.min(x[x>0])
    maxx = np.max(x)
    xbounders = np.linspace(minx, maxx, bins)
    delta = xbounders[1] - xbounders[0]
    xcenters = xbounders[:-1] + delta / 2
    xcenters = np.hstack([[xbounders[0]], xcenters, [xbounders[-1]]])
    xcenters[-1] += 1e-4

    # print(xcenters)

    # print()

    mu = np.zeros(bins)
    error = np.zeros(bins)

    for i in np.arange(bins):

      mu[i] = np.mean(y[(x > xcenters[i]) & (x <= xcenters[i+1])])
      error[i] = np.std(y[(x > xcenters[i]) & (x <= xcenters[i+1])])

    return xbounders, mu, error
    
  cmp = sns.color_palette("deep")
  alpha = 0.2
  mu_s = 30
    
  for i, ax in enumerate(g.axes.flatten()):
    if i == 0:
      d = data[xlabel].loc[(data["Direction"] == "+") & (data["Index"] == h2_label)].to_numpy()
      h2 = data[ylabel].loc[(data["Direction"] == "+") & (data["Index"] == h2_label)].to_numpy()

      dist_b, mu_h2, error_h2 = little_area_trend(d, h2)
      sns.scatterplot(x=dist_b, y=mu_h2, ax=ax, color=cmp[0], s=mu_s, edgecolor="b")
      sns.lineplot(x=dist_b, y=mu_h2, ax=ax, color=cmp[0], linestyle="--")
      ax.fill_between(dist_b, mu_h2 - error_h2, mu_h2 + error_h2, color=cmp[0], alpha=alpha, edgecolor=None)

      d = data[xlabel].loc[(data["Direction"] == "+") & (data["Index"] == cos_label)].to_numpy()
      cos = data[ylabel].loc[(data["Direction"] == "+") & (data["Index"] == cos_label)].to_numpy()

      dist_b, mu_cos, error_cos = little_area_trend(d, cos)
      sns.scatterplot(x=dist_b, y=mu_cos, ax=ax, color=cmp[1], s=mu_s, edgecolor="b")
      sns.lineplot(x=dist_b, y=mu_cos, ax=ax, color=cmp[1], linestyle="--")
      ax.fill_between(dist_b, mu_cos - error_cos, mu_cos + error_cos, color=cmp[1], alpha=alpha, edgecolor=None)

    if i == 1:
      d = data[xlabel].loc[(data["Direction"] == "-") & (data["Index"] == h2_label)].to_numpy()
      h2 = data[ylabel].loc[(data["Direction"] == "-") & (data["Index"] == h2_label)].to_numpy()

      dist_b, mu_h2, error_h2 = little_area_trend(d, h2)
      sns.scatterplot(x=dist_b, y=mu_h2, ax=ax, color=cmp[0], s=mu_s, edgecolor="b")
      sns.lineplot(x=dist_b, y=mu_h2, ax=ax, color=cmp[0], linestyle="--")
      ax.fill_between(dist_b, mu_h2 - error_h2, mu_h2 + error_h2, color=cmp[0], alpha=alpha, edgecolor=None)

      d = data[xlabel].loc[(data["Direction"] == "-") & (data["Index"] == cos_label)].to_numpy()
      cos = data[ylabel].loc[(data["Direction"] == "-") & (data["Index"] == cos_label)].to_numpy()

      dist_b, mu_cos, error_cos = little_area_trend(d, cos)
      sns.scatterplot(x=dist_b, y=mu_cos, ax=ax, color=cmp[1], s=mu_s, edgecolor="b")
      sns.lineplot(x=dist_b, y=mu_cos, ax=ax, color=cmp[1], linestyle="--")
      ax.fill_between(dist_b, mu_cos - error_cos, mu_cos + error_cos, color=cmp[1], alpha=alpha, edgecolor=None)

  fig = plt.gcf()
  fig.set_figwidth(12)
  fig.set_figheight(6)
  fig.tight_layout()

  path = join(NET.plot_path, "Features")
  from pathlib import Path
  Path(path).mkdir(exist_ok=T, parents=T)

  plt.savefig(join(path, "dissimilarities.svg"), transparent=T)
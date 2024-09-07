# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# sns.set_theme()
from various.data_transformations import maps

from networks.MAC.mac40 import MAC40
from networks.MAC.mac57 import MAC57

from various.network_tools import *
from networks.distbase import DISTBASE
from networks.swapnet import SWAPNET  
from various.fit_tools import fitters
import ctools as ct

def simple(x, y):
  return -np.mean(np.abs(x-y))

def model_network(D, C, nodes, bins):
  import statsmodels.api as sm
  from sklearn.preprocessing import StandardScaler
  _, x, y = range_and_probs_from_DC(
    D, C, nodes, bins
  )
  d = x.copy()
  x = x.reshape(-1, 1)
  scaler = StandardScaler(copy=True).fit(x)
  x = scaler.transform(x)
  x = np.hstack([x, np.power(x, 3)])
  x = sm.add_constant(x)
  est = sm.OLS(y, x)
  est = est.fit()
  print(est.summary())
  y_pred = est.predict(x)
  y_pred = np.exp(y_pred)
  y = np.exp(y)
  data = pd.DataFrame(
    {
      "dist" : np.round(d, 1).astype(str),
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
  sns.lineplot(
    data=data,
    linewidth=2,
    x="dist",
    y='pred',
    color="r",
    ax=ax
  )
  ax.set_yscale('log')
  plt.show()

def model_network2(D, C, nodes, bins):
  from sklearn.linear_model import LinearRegression
  from sklearn.preprocessing import StandardScaler
  _, x, y = range_and_probs_from_DC(
    D, C, nodes, bins
  )
  d = x.copy()
  x = x.reshape(-1, 1)
  scaler = StandardScaler(copy=True).fit(x)
  x = scaler.transform(x)
  x = np.hstack([x, np.power(x, 3)])


def EDR_ensemble(NET, trials, bins=12, distance_model="M", fitter="EXPMLE"):

  pars, _, _, _, _ = fitters[fitter](NET.D, NET.CC, bins)
  lb = 1 / pars[2]
  loc = np.min(NET.D[NET.D > 0])
  # if distance_model == "M":
  #   loc = 0
  b = np.max(NET.D)

  D12_edr_tgt = np.zeros((trials, int(NET.nodes * (NET.nodes - 1) / 2)))

  for i in np.arange(trials):

    RAND = DISTBASE(
      __inj__, __nodes__,
      linkage, bins, mode, i,
      structure = structure,
      version = version, model="MAP3D",
      nlog10=nlog10, lookup=F, cut=cut,
      topology=topology, distance=distance,
      mapping=mapping, index=index, b=bias,
      lb=lb,
      discovery=discovery,
      rho = adj2Den(NET.A[:NET.nodes,:][:, :NET.nodes])
    )
    RAND.rows = NET.rows
    # Create network ----
    print("Create random graph")
    RC = RAND.distbase_dict[distance_model](
      NET.D, NET.CC, loc=loc, b=b, run=T, on_save_csv=F
    ) 
    RA = column_normalize(RC)
    # Transform data for analysis ----
    R, _, _ = maps[mapping](
      RA, nlog10, T, prob, b=bias
    )
    e = 0 
    np.fill_diagonal(R, 0)
    for j in np.arange(NET.nodes):
      for k in np.arange(j+1 , NET.nodes):
        D12_edr_tgt[i, e] = -2 * np.log(ct.Hellinger2(R[:, j], R[:, k], j, k))
        # D12_edr_tgt[i, e] = (1 / D12_edr_tgt[i, e]) - 1
        e += 1
  return D12_edr_tgt

def CONF_ensemble(NET, trials):

  D12_edr_tgt = np.zeros((trials, int(NET.nodes * (NET.nodes - 1) / 2)))

  for i in np.arange(trials):

    RAND = SWAPNET(
      __inj__,
      NET.rows,
      linkage,
      mode, i,
      structure = structure,
      version = "40d91",
      topology = topology,
      nature = nature,
      distance = distance,
      model = "1k",
      mapping=mapping,
      index=index,
      nlog10 = F, lookup=F,
      cut=cut, b=bias, dscovery=discovery,
      subject="MAC"
    )
    RAND.C, RAND.A = NET.CC, NET.A
    RAND.D = NET.D

    # Create network ----
    print("Create random graph")
    RAND.random_one_k(run=T, swaps=1000000, on_save_csv=F)
    # Transform data for analysis ----
    R, _, _ = maps[mapping](
      RAND.A, F, F, T, b=bias
    )
    e = 0 
    np.fill_diagonal(R, 0)
    for j in np.arange(NET.nodes):
      for k in np.arange(j+1 , NET.nodes):
        D12_edr_tgt[i, e] = -2 * np.log(ct.Hellinger2(R[:, j], R[:, k], j, k))
        e += 1
  return D12_edr_tgt

def mean_trend(D, data, network : str, bins=12):
  minD = np.min(D[D>0])
  maxD = np.max(D)
  distance_bin = np.linspace(minD, maxD, bins + 1)
  distance_center_bin = (distance_bin[1:] + distance_bin[:-1]) / 2
  mean_d12 = np.zeros(bins)
  sd_d12 = np.zeros(bins)

  for i in np.arange(bins):
    if i != bins - 1:
      mean_d12[i] = data["score"].loc[
        (data["distance"] < distance_bin[i + 1]) &
        (data["distance"] >= distance_bin[i])
      ].mean()
      sd_d12[i] = data["score"].loc[
        (data["distance"] < distance_bin[i + 1]) &
        (data["distance"] >= distance_bin[i])
      ].std()
    else:
      mean_d12[i] = data["score"].loc[
        (data["distance"] >= distance_bin[i])
      ].mean()
      sd_d12[i] = data["score"].loc[
        (data["distance"] >= distance_bin[i])
      ].std()

  
  # mean_d12[0] = data["score"].loc[data["distance"] == minD].mean()
  # mean_d12[-1] = data["score"].loc[data["distance"] == maxD].mean()
  # sd_d12[0] = data["score"].loc[data["distance"] == minD].std()
  # sd_d12[-1] = data["score"].loc[data["distance"] == maxD].std()


  return pd.DataFrame(
    {
      "score" : mean_d12,
      "distance" : distance_center_bin,
      "network" : [network] * bins
    }
  ) , mean_d12, sd_d12, distance_center_bin

# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
structure = "FLNe"
mode = "ZERO"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
discovery = "discovery_7"
bias = 0.
alpha = 0.
opt_score = ["_S"]

__nodes__ = 40
__inj__ = __nodes__
total_nodes = 91
version = f"{__nodes__}d{total_nodes}"

def three_divergences(ax : plt.Axes, distance="MAP3D", mode="ZERO", iterations=500, cmap="deep"):
  NET = MAC40(
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
    b = bias,
    alpha = alpha,
    discovery = discovery
  )

  H_DATA = read_class(
    f"../pickle/MAC/{version}/{structure}/original/{distance}/{__inj__}/SINGLE_{total_nodes}_{__nodes__}/ZERO/MIX_Hellinger2_trivial/b_0.0/",
    "hanalysis"
  )

  bins=12

  trials = iterations

  distance_model = "M"
  fitter = "EXPTRUNC"

  D12_edr_tgt = EDR_ensemble(NET, trials, bins=bins, distance_model=distance_model, fitter=fitter)
  D12_conf_tgt = CONF_ensemble(NET, trials)

  trials, connections = D12_edr_tgt.shape
  D = np.zeros(connections)
  e = 0
  for i in np.arange(NET.nodes):
    for j in np.arange(i + 1, NET.nodes):
      D[e] = NET.D[i, j]
      e += 1

  D = np.tile(D, trials).ravel()

  edr = pd.DataFrame(
    {
      "score" : D12_edr_tgt.ravel(),
      "distance" : D,
      "network" : ["EDR"] * D.shape[0]
    }
  )

  edr_mean, mean_d12, sd_d12, distance_bin_center = mean_trend(D, edr, "EDR", bins=bins)

  conf = pd.DataFrame(
    {
      "score" : D12_conf_tgt.ravel(),
      "distance" : D,
      "network" : ["Configuration"] * D.shape[0]
    }
  )

  conf_mean, conf_mean_d12, conf_sd_d12, _ = mean_trend(D, conf, "Configuration", bins=bins)


  s_data = H_DATA.target_sim_matrix.copy()
  s_data = -2 * np.log(s_data)

  D = H_DATA.D.copy()[:__nodes__, :__nodes__]
  n = D.ravel().shape[0]

  data = pd.DataFrame(
    {
      "score" : s_data.ravel(),
      "distance" : D.ravel(),
      "network" : ["Data"] *  n
    }
  )

  data = data.loc[(data["score"] < np.Inf) & (data["distance"] > 0)]

  data_mean, mean_data_d12, sd_data_d12, _ = mean_trend(data["distance"], data, "Data", bins=bins)

  sns.set_style("ticks")
  sns.set_context("talk")

  cmap = sns.color_palette("deep")

  sns.scatterplot(
    data=data_mean,
    x="distance",
    y="score",
    color=cmap[2],
    style="network",
    markers=["o"],
    alpha=0.8,
    ax=ax,
    legend=T
  )

  sns.scatterplot(
    data=edr_mean,
    x="distance",
    y="score",
    color=cmap[0],
    alpha=0.6,
    style="network",
    markers=["^"],
    ax=ax,
    legend=T
  )

  sns.scatterplot(
    data=conf_mean,
    x="distance",
    y="score",
    color=cmap[1],
    alpha=0.6,
    style="network",
    markers=["*"],
    ax=ax,
    legend=T
  )

  sns.lineplot(
    data=data_mean,
    x="distance",
    y="score",
    color=cmap[2],
    alpha=0.8,
    linewidth=1,
    ax=ax,
    legend=T
  )

  sns.lineplot(
    data=edr_mean,
    x="distance",
    y="score",
    color=cmap[0],
    alpha=0.6,
    linestyle="--",
    linewidth=1,
    ax=ax,
    legend=T
  )

  sns.lineplot(
    data=conf_mean,
    x="distance",
    y="score",
    color=cmap[1],
    alpha=0.6,
    linewidth=1,
    linestyle="-.",
    ax=ax,
    legend=T
  )

  ax.fill_between(
    distance_bin_center,
    mean_data_d12 + 1.96 * sd_data_d12 / 2,
    mean_data_d12 - 1.96 * sd_data_d12 / 2, color=cmap[2],
    alpha=0.4, edgecolor=None
  )

  ax.fill_between(
    distance_bin_center,
    mean_d12 + 1.96 * sd_d12 / 2,
    mean_d12 - 1.96 * sd_d12 / 2, color=cmap[0],
    alpha=0.4, edgecolor=None
  )

  ax.fill_between(
    distance_bin_center,
    conf_mean_d12 + 1.96 * conf_sd_d12 / 2,
    conf_mean_d12 - 1.96 * conf_sd_d12 / 2, color=cmap[1],
    alpha=0.4, edgecolor=None
  )

  plt.ylabel(r"$D_{1/2}^{-}$")
  plt.xlabel('Interareal distance [mm]')

  import matplotlib.lines as mlines

  orange_line = mlines.Line2D( 
      [], [], color=cmap[1], label='Configuration', lw=2, marker="*", alpha=0.6, markeredgewidth=0.1
  )

  green_line = mlines.Line2D( 
      [], [], color=cmap[2], label='Data', lw=2, marker="o", alpha=0.6, markeredgewidth=0.1
  )
  blue_line = mlines.Line2D( 
      [], [], color=cmap[0], label='EDR', lw=2 , marker="^", alpha=0.6, markeredgewidth=0.1
  )

  legend = ax.legend(
    handles=[green_line, blue_line, orange_line],
    bbox_to_anchor=(0.025, 0.8),
    loc="center left",
    fontsize=8.5
  )

  legend.set_title("")
  frame = legend.get_frame()
  frame.set_linewidth(0.75)

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
sns.set_theme()
from various.data_transformations import maps
from networks.MAC.mac57 import MAC57
from various.network_tools import *
from various.similarity_indices import jacp_smooth, jacp
from networks.distbase import DISTBASE
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


def EDR_ensemble(NET, trials):

  bins=12
  distance_model = "EXPMLE"

  _, _, _, _, est = fitters[distance_model](NET.D, NET.C, NET.nodes, bins)
  lb = est.coef_[0]

  D12_edr_tgt = np.zeros((trials, int(NET.nodes * (NET.nodes - 1) / 2)))

  for i in np.arange(trials):

    RAND = DISTBASE(
      __inj__, __nodes__,
      linkage, bins, mode, i,
      structure = structure,
      version = version, model="tracto16",
      nlog10=nlog10, lookup=T, cut=cut,
      topology=topology, distance=distance,
      mapping=mapping, index=index, b=bias,
      lb=lb, discovery=discovery
    )
    RAND.rows = NET.rows
    # Create network ----
    print("Create random graph")
    RC = RAND.distbase_dict[distance_model](
      NET.D, NET.C, run=T, on_save_csv=F
    )
    RA = column_normalize(RC)
    # Transform data for analysis ----
    R, lookup, _ = maps[mapping](
      RA, nlog10, T, prob, b=bias
    )
    e = 0 
    np.fill_diagonal(R, 0)
    for j in np.arange(NET.nodes):
      for k in np.arange(j+1 , NET.nodes):
        D12_edr_tgt[i, e] = ct.D1_2_4(R[:, j], R[:, k], j, k)
        D12_edr_tgt[i, e] = (1 / D12_edr_tgt[i, e]) - 1
        e += 1
  return D12_edr_tgt

def mean_trend(D, data, network : str):
  bins = 12
  minD = np.min(D)
  maxD = np.max(D)
  distance_bin = np.linspace(minD, maxD, bins + 1)
  delta = (distance_bin[1] - distance_bin[0]) / 2
  distance_center_bin = distance_bin[:-1] + delta
  mean_d12 = np.zeros(bins + 2)
  sd_d12 = np.zeros(bins + 2)

  for i in np.arange(bins):
    if i != bins - 1:
      mean_d12[i + 1] = data["score"].loc[
        (data["distance"] < distance_bin[i + 1]) &
        (data["distance"] >= distance_bin[i])
      ].mean()
      sd_d12[i + 1] = data["score"].loc[
        (data["distance"] < distance_bin[i + 1]) &
        (data["distance"] >= distance_bin[i])
      ].std()
    else:
      mean_d12[i + 1] = data["score"].loc[
        (data["distance"] >= distance_bin[i])
      ].mean()
      sd_d12[i + 1] = data["score"].loc[
        (data["distance"] >= distance_bin[i])
      ].std()


  distance_center_bin2 =  np.zeros(bins + 2)
  distance_center_bin2[1:-1] = distance_center_bin
  distance_center_bin2[0] = minD
  distance_center_bin2[-1] = maxD
  
  mean_d12[0] = data["score"].loc[data["distance"] == minD].mean()
  mean_d12[-1] = data["score"].loc[data["distance"] == maxD].mean()

  sd_d12[0] = data["score"].loc[data["distance"] == minD].std()
  sd_d12[-1] = data["score"].loc[data["distance"] == maxD].std()


  return pd.DataFrame(
    {
      "score" : mean_d12[1:-1],
      "distance" : distance_center_bin2[1:-1],
      "network" : [network] * bins
    }
  ) , mean_d12, sd_d12, distance_center_bin2

# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
structure = "LN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
discovery = "discovery_7"
bias = 0.
alpha = 0.
opt_score = ["_S"]
version = "57d106"
__nodes__ = 57
__inj__ = 57

if __name__ == "__main__":
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
    b = bias,
    alpha = alpha,
    discovery = discovery
  )

  H_DATA = read_class(
    "../pickle/MAC/57d106/FLN/original/tracto16/57/SINGLE_106_57_l10/ZERO/MIX_Hellinger2_trivial/b_0.0/discovery_7/",
    "hanalysis"
  )

  trials = 100
  D12_edr_tgt = EDR_ensemble(NET, trials)
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

  edr_mean, mean_d12, sd_d12, distance_bin_center = mean_trend(D, edr, "EDR")

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

  data_mean, mean_data_d12, sd_data_d12, _ = mean_trend(data["distance"], data, "Data")

  # import statsmodels.api as sm


  # X = data["distance"].to_numpy().reshape(-1, 1)
  # X = sm.add_constant(X)
  # data_model = sm.OLS(data["score"].to_numpy(), X).fit()

  # l = np.linspace(np.min(D), np.max(D), 100)
  # data_reg = {
  #   "distance" : l,
  #   "score" : data_model.predict(sm.add_constant(l.reshape(-1, 1))),
  #   "network" : ["Data"] * 100
  # }

  sns.set_style("whitegrid")
  sns.set_context("talk")

  _, ax = plt.subplots(1, 1)

  sns.scatterplot(
    data=data,
    x="distance",
    y="score",
    hue="network",
    alpha=0.9,
    palette={"Data" : sns.color_palette("deep", as_cmap=T)[1]},
    s=10,
    ax=ax
  )

  sns.lineplot(
    data=data_mean,
    x="distance",
    y="score",
    hue="network",
    alpha=0.8,
    palette={"Data" : sns.color_palette("deep", as_cmap=T)[1]},
    ax=ax
  )

  sns.lineplot(
    data=edr_mean,
    x="distance",
    y="score",
    hue="network",
    alpha=0.6,
    markers="o",
    ax=ax
  )

  sns.scatterplot(
    data=data_mean,
    x="distance",
    y="score",
    hue="network",
    alpha=1,
    linewidth=1,
    edgecolors="black",
    palette={"Data" : sns.color_palette("deep", as_cmap=T)[1]},
    s=45,
    ax=ax
  )

  sns.scatterplot(
    data=edr_mean,
    x="distance",
    y="score",
    linewidth=1,
    edgecolors="black",
    hue="network",
    alpha=1,
    s=45,
    ax=ax
  )

  ax.fill_between(
    distance_bin_center,
    mean_data_d12 + sd_data_d12 / 2,
    mean_data_d12 - sd_data_d12 / 2, color=sns.color_palette("deep", as_cmap=T)[1],
    alpha=0.4, edgecolor=None
  )

  ax.fill_between(
    distance_bin_center,
    mean_d12 + sd_d12 / 2,
    mean_d12 - sd_d12 / 2, color=sns.color_palette("deep", as_cmap=T)[0],
    alpha=0.4, edgecolor=None
  )

  plt.ylabel(r"$D_{1/2}^{-}$")
  plt.xlabel('interareal tractography distances [mm]')

  # plt.show()

  plt.gcf().set_figwidth(8.5)
  plt.gcf().set_figheight(6.5)
  

  cortex_letter_path = "../plots/MAC/57d106/FLN/original/tracto16/57/SINGLE_106_57_l10/"


  plt.savefig(
    os.path.join(cortex_letter_path, "cortex_letter/two_divergences2.svg"), dpi=300, transparent=T
  )
  

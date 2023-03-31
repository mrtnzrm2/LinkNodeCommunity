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
from networks.structure import MAC
from various.network_tools import *


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

# Declare global variables ----
__iter__ = 0
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
structure = "FLN"
distance = "MAP3D"
nature = "original"
__mode__ = "ALPHA"
topology = "MIX"
mapping = "R2"
index  = "jacw"
imputation_method = ""
opt_score = ["_maxmu", "_X"]
save_datas = T
# Declare global variables DISTBASE ----
__model__ = "DEN"
total_nodes = 106
__inj__ = 57
__nodes__ = 57
__bin__ = 12
lb = 0.07921125
__version__ = 220830
bias = float(0.3)
## Very specific!!! Be careful ----
if nature == "original":
  __ex_name__ = f"{total_nodes}_{__inj__}"
else:
  __ex_name__ = f"{total_nodes}_{total_nodes}_{__inj__}"
if nlog10: __ex_name__ = f"{__ex_name__}_l10"
if lookup: __ex_name__ = f"{__ex_name__}_lup"
if cut: __ex_name__ = f"{__ex_name__}_cut"

if __name__ == "__main__":
  # MAC network as reference ----
  REF = MAC(
    linkage, __mode__,
    structure = structure,
    nlog10=nlog10, lookup=lookup,
    version = __version__,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = __inj__,
    topology=topology,
    index=index,
    mapping=mapping,
    cut=cut, b=bias
  )
  D = REF.D
  p = lambda a, xm, x: xm * np.power(1 - x, -1/a)
  data = pd.DataFrame(
    {
     "dist" : p(1.00095, np.nanmin(D[D>0])-1e-8, np.random.rand(100000,))
    }
  )
  data = data.loc[data.dist < np.nanmax(D)]
  _, ax= plt.subplots(1, 1)
  sns.histplot(
    data=data,
    x="dist",
    stat="density",
    ax=ax
  )
  ax.set_yscale("log")
  plt.show()
  # model_network(D, REF.C, __nodes__, 12)
  # # Transform data for analysis ----
  # RFLN, lookup, _ = maps[mapping](
  #   REF.A, nlog10, lookup, prob, b=bias
  # )
  # RFLN = RFLN[:, :__nodes__]
  # Create network ----
  # print("Create random graph")
  # Gn = 0
  # G = column_normalize(Gn)
  # # Transform data for analysis ----
  # REDR, lookup, _ = maps[mapping](
  #   G, nlog10, lookup, prob, b=bias
  # )
  # REDR = REDR[:, :__nodes__]
  # ## Draw
  # zeros_fln = RFLN == 0
  # zeros_edr = REDR == 0
  # zeros = zeros_fln | zeros_edr
  # RFLN = RFLN[~zeros]
  # REDR = REDR[~zeros]
  # data = pd.DataFrame(
  #   {
  #     "edr" : REDR.ravel(),
  #     "data" : RFLN.ravel()
  #   }
  # )
  # sns.scatterplot(
  #   data=data,
  #   x="edr",
  #   y="data",
  #   s=5
  # )
  # plt.show()

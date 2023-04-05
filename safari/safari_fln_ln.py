# Insert path ---
import os
import sys
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
from scipy.special import digamma
from scipy.stats import beta, norm
from scipy.optimize import fsolve
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libs ---- 
from various.data_transformations import maps
from networks.structure import MAC
from various.network_tools import *
# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
structure = "FLN"
mode = "ALPHA"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
index  = "jacw"
bias = float(0.3)
opt_score = ["_maxmu", "_X", "_D"]
save_data = F
version = 220830
__nodes__ = 57
__inj__ = 57
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC(
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
    cut = cut,
    b = bias
  )
  # Transform data for analysis ----
  RFLN, lookup, _ = maps["R1"](
    NET.A, T, F, T, b=0.3
  )
  RLN, lookup, _ = maps["R4"](
    NET.C, T, F, F, b=0
  )
  ######
  # FLN = adj2df(NET.A[:, :4])
  # FLN.source = NET.struct_labels[FLN.source]
  # FLN.target = NET.struct_labels[FLN.target]
  # FLN = FLN.loc[FLN.weight != 0]
  # f = FLN.groupby("source")["weight"].mean().sort_values(ascending=False)
  # n = np.unique(FLN.source).shape[0]
  # ## beta
  # al, be, loc, scale = beta.fit(np.log(FLN.weight))
  # pred = np.random.beta(al, be, size=(n, 1000))
  # pred = pred * scale + loc
  # pred = np.exp(pred)
  # for i in np.arange(1000):
  #   order = np.argsort(-pred[:, i])
  #   pred[:, i] = pred[order, i]
  # pred_beta = np.mean(pred, axis=1)
  # ## lognorm
  # mu, sig = norm.fit(np.log(FLN.weight))
  # pred = np.random.normal(loc=mu, scale=sig, size=(n, 1000))
  # pred = np.exp(pred)
  # for i in np.arange(1000):
  #   order = np.argsort(-pred[:, i])
  #   pred[:, i] = pred[order, i]
  # pred_norm = np.mean(pred, axis=1)
  # _, ax = plt.subplots(1, 1)
  # sns.stripplot(
  #   data=FLN,
  #   x="source",
  #   y="weight",
  #   order=f.index.to_numpy(),
  #   hue="target",
  #   ax=ax
  # )
  # data = pd.DataFrame(
  #   {
  #     "weight" : list(pred_beta) + list(pred_norm),
  #     "source" : np.tile(f.index.to_numpy(), 2),
  #     "model" : ["Beta"] * n + ["Lognormal"] * n
  #   }
  # )
  # sns.lineplot(
  #   data=data,
  #   x="source",
  #   y="weight",
  #   hue="model",
  #   ax=ax
  # )
  # ax.set_ylabel("FLNe")
  # ax.set_xlabel("Source areas")
  # plt.xticks(rotation=90)
  # ax.set_yscale("log")
  # plt.show()
  ####
  # FLN_1 = NET.A[:, :]
  # FLN_1 = FLN_1[FLN_1 != 0].ravel()
  # al, be, loc, scale = beta.fit(-np.log(FLN_1))
  # al_norm, be_norm= norm.fit(-np.log(FLN_1))
  # flnmin = np.min(-np.log(FLN_1))
  # flnmax = np.max(-np.log(FLN_1))
  # x = np.linspace(loc, scale, 100)
  # pred_beta = beta.pdf(x, al, be, loc=loc, scale=scale)
  # pred_norm = norm.pdf(x, al_norm, be_norm)
  # data_pred = pd.DataFrame(
  #   {
  #     "Density" : list(pred_beta) + list(pred_norm), "FLN" : list(x) + list(x),
  #     "model" : ["Beta"] * len(pred_beta) + ["Normal"] * len(pred_norm)
  #   }
  # )
  # data = pd.DataFrame({"FLN" : -np.log(FLN_1.ravel())})
  # _, ax = plt.subplots(1, 1)
  # sns.lineplot(
  #   data=data_pred,
  #   x="FLN",
  #   y="Density",
  #   hue="model",
  #   ax=ax
  # )
  # sns.histplot(
  #   data=data,
  #   x="FLN",
  #   stat="density",
  #   color="green",
  #   alpha=0.5,
  #   ax=ax
  # )
  # ax.set_title(r"$\alpha_{1}:$ " + f"{al:.3f} " + r"$\alpha_{2}$: " f"{be:.3f}\t\t" + r"$\mu$: " + f"{al_norm:.3f} " + r"$\sigma$: " + f"{be_norm:.3f}")
  # ax.set_xlabel(r"$-\log(FLN)$")
  # plt.show()
  #########
  # AREAS = adj2df(RFLN)
  # AREAS = AREAS.loc[(AREAS.weight != 0) & (~np.isnan(AREAS.weight))].target.to_numpy().astype(str)
  # zeros = RFLN == 0
  # RFLN = RFLN[~zeros].ravel()
  # RLN = RLN[~zeros].ravel()
  # RFLN = RFLN[~np.isnan(RFLN)]
  # RLN = RLN[~np.isnan(RLN)]
  # data = pd.DataFrame(
  #   {
  #   "FLN" : RFLN,
  #   "LN" : RLN,
  #   "TARGET" : AREAS
  #   }
  # )
  # _, ax = plt.subplots(1, 1)
  # sns.scatterplot(
  #   data=data,
  #   x="FLN",
  #   y="LN",
  #   hue="TARGET",
  #   # scatter_kws={"s" : 2},
  #   # line_kws={"color" : "orange"},
  #   ax=ax
  # )
  # ax.text(0.5, 8, "pearsonr: {:.5f}".format(pearsonr(RFLN, RLN).statistic))
  # X = RFLN.reshape(-1, 1)
  # X = sm.add_constant(X)
  # lm = sm.OLS(RLN, X).fit()
  # a = lm.params[1]
  # b = lm.params[0]
  # ax.set_ylabel(r"$\log(1 + LN)$")
  # ax.text(0.5, 10, r"$LN = a FLN + b$    $a$: " + f"{a:.3f} " + r"$b$: " + f"{b:.3f}")
  # ax.set_xlabel(r"$\log_{10}(FLN) - min_{FLN_{ij}} \log_{10}(FLN_{i,j}) + 0.3$ ")
  # plt.legend(bbox_to_anchor=(1.1, 1.05), ncol=10)
  # plt.show()
  #####
  # _, ax = plt.subplots(1, 1)
  # sns.kdeplot(
  #   data=data,
  #   x="FLN",
  #   hue="TARGET",
  #   multiple="stack",
  #   linewidth=0.5,
  #   ax=ax
  # )
  # ax.set_xlabel(r"$-\log(FLN)$")
  # plt.legend(bbox_to_anchor=(1.1, 1.05), ncol=10)
  # plt.show()
  
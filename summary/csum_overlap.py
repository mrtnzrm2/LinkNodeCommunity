# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Import libraries ----
## Standard libs ----
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
## Personal libs ----
from various.network_tools import *

def draw_heatmap(*args, **kwargs):
  data = kwargs.pop('data')
  d = data.pivot(index=args[1], columns=args[0], values=args[2])
  sns.heatmap(d, **kwargs)

# Declare iter variables ----
topologies = ["TARGET", "SOURCE", "MIX"]
indices = ["jacw", "jacp", "cos", "bsim"]
KAV = [7, 15]
MUT = [0.1, 0.3, 0.5]
MUW = [0.1, 0.5]
OM = [2, 5]
list_of_lists = itertools.product(
  *[topologies, indices, KAV, MUT, MUW, OM]
)
list_of_lists = np.array(list(list_of_lists))
# Constant parameters ---
MAXI = 503
__nodes__ = 128
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
run = T
mapping = "trivial"
__mode__ = "ALPHA"
l10 = ""
lup = ""
_cut = ""
if nlog10: l10 = "_l10"
if lookup: lup = "_lup"
if cut: _cut = "_cut"
opt_score = ["_maxmu", "_X", "_D"]
if __name__ == "__main__":
  # Extract data ----
  THE_DF = pd.DataFrame()
  for topology, index, kav, mut, muw, om in list_of_lists:
    # WDN paramters ----
    opar = {
      "-N" : "{}".format(str(__nodes__)),
      "-k" : f"{kav}.0",
      "-maxk" : "30",
      "-mut" : f"{mut}",
      "-muw" : f"{muw}",
      "-beta" : "2.5",
      "-t1" : "2",
      "-t2" : "1",
      "-nmin" : "2",
      "-nmax" : "10",
      "-on" : "10",
      "-om" : f"{int(om)}"
    }
    data = read_class(
      "../pickle/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/-on_{}/-om_{}/{}/{}{}{}{}/{}/{}".format(
        str(__nodes__),
        opar["-k"], opar["-maxk"],
        opar["-mut"], opar["-muw"],
        opar["-beta"], opar["-t1"], opar["-t2"], opar["-nmin"], opar["-nmax"],
        opar["-on"], opar["-om"], MAXI,
        linkage.upper(), l10, lup, _cut,
        __mode__, f"{topology}_{index}_{mapping}"
      ),
      "series_{}".format(MAXI)
    )
    if isinstance(data, int): continue
    l = data.data["values"].loc[data.data.sim == "NMI"].shape[0]
    THE_DF = pd.concat(
      [
        THE_DF, 
        pd.DataFrame(
          {
            "NMI" : data.data["values"].loc[data.data.sim == "NMI"].to_numpy().astype(float),
            "OMEGA" : data.data["values"].loc[data.data.sim == "OMEGA"].to_numpy().astype(float),
            "TPR":  data.data_overlap.sensitivity.astype(float),
            "FPR":  1 - data.data_overlap.specificity.astype(float),
            "score" : data.data.c.loc[data.data.sim == "NMI"].to_numpy(),
            "topology": [f"{topology}"] * l,
            "index" : [index] * l,
            "kav" : [int(kav)] * l,
            "mut" : [float(mut)] * l,
            "muw" : [float(muw)] * l
          }
        )
      ], ignore_index=T
    )
  # Comparing topology, index, & score for given kav, mut, and muw
  KAV = [7, 15]
  MUT = [0.1, 0.3, 0.5]
  MUW = [0.1, 0.5]
  OM = [2, 5]
  list_of_lists = itertools.product(
    *[KAV, MUT, MUW, OM]
  )
  list_of_lists = np.array(list(list_of_lists))
  for kav, mut, muw, om in list_of_lists:
    print(kav, mut, muw, om)
    #
    # Prepare path ----
    #
    opar = {
      "-N" : "{}".format(str(__nodes__)),
      "-k" : f"{kav}",
      "-maxk" : "30",
      "-mut" : f"{mut}",
      "-muw" : f"{muw}",
      "-beta" : "2.5",
      "-t1" : "2",
      "-t2" : "1",
      "-nmin" : "2",
      "-nmax" : "10",
      "-on" : "10",
      "-om" : f"{int(om)}"
    }
    IM_ROOT =  "../plots/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/-on_{}/-om_{}/{}/{}{}{}{}/{}".format(
        str(__nodes__),
        opar["-k"], opar["-maxk"],
        opar["-mut"], opar["-muw"],
        opar["-beta"], opar["-t1"], opar["-t2"], opar["-nmin"], opar["-nmax"],
        opar["-on"], opar["-om"], MAXI,
        linkage.upper(), l10, lup, _cut,
        __mode__
      )
    Path(IM_ROOT).mkdir(exist_ok=True, parents=True)
    data = THE_DF.loc[
      (THE_DF.kav == int(kav)) &
      (THE_DF.mut == float(mut)) &
      (THE_DF.muw == float(muw))
    ]
    data.NMI.loc[np.isnan(data.NMI)] = 0
    data.TPR.loc[np.isnan(data.TPR)] = 0
    data.FPR.loc[np.isnan(data.FPR)] = 0
    #
     # Prepare data sensitivity & specificity ----
    #
    x = data.groupby(["topology", "index", "score"]).agg({
      "TPR" : ["mean"], "FPR" : [ "mean"]
    }).reset_index()
    x.columns = ["topology", "index", "score", "TPR", "FPR"]
    TPR = x[["topology", "index", "score", "TPR"]]
    TPR.columns = ["topology", "index", "score", "values"]
    TPR["score2"] = ["TPR"] * TPR.shape[0]
    FPR = x[["topology", "index", "score", "FPR"]]
    FPR.columns = ["topology", "index", "score", "values"]
    FPR["score2"] = ["FPR"] * FPR.shape[0]
    TR = pd.concat([TPR, FPR], ignore_index=True)
    if TR.shape[0] > 0:
      g = sns.FacetGrid(
        data=TR,
        row="score2",
        col="score"
      )
      g.map_dataframe(
        draw_heatmap,
        "topology",
        "index",
        "values",
        vmin=0, vmax=1,
        square=T,
        cmap=sns.color_palette("viridis"),
        cbar=T
      )
      # for axes in g.axes.flat:
      #   _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
      plt.tight_layout()
      plt.savefig(
        os.path.join(
          IM_ROOT, "NOCS_facet_score.png"
        ),
        dpi = 300
      )
    #
    # Prepare dat NMI ----
    #
    x = data.groupby(["topology", "index", "score"]).agg({"NMI": ["mean"]}).reset_index()
    x.columns = ["topology", "index", "score", "NMI"]
    if x.shape[0] > 0:
      g = sns.FacetGrid(
        data=x,
        col="score"
      )
      g.map_dataframe(
        draw_heatmap,
        "topology",
        "index",
        "NMI",
        vmin=0, vmax=1,
        square=T,
        cbar=T,
        cmap=sns.color_palette("viridis")
      )
      for axes in g.axes.flat:
          _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
      plt.tight_layout()
      plt.savefig(
        os.path.join(
          IM_ROOT, "NMI_facet_score.png"
        ),
        dpi = 300
      )
    # Prepare dat OMEGA ----
    #
    x = data.groupby(["topology", "index", "score"]).agg({"OMEGA": ["mean"]}).reset_index()
    x.columns = ["topology", "index", "score", "OMEGA"]
    if x.shape[0] > 0:
      g = sns.FacetGrid(
        data=x,
        col="score"
      )
      g.map_dataframe(
        draw_heatmap,
        "topology",
        "index",
        "OMEGA",
        vmin=0, vmax=1,
        square=T,
        cbar=T,
        cmap=sns.color_palette("viridis")
      )
      for axes in g.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
      plt.tight_layout()
      plt.savefig(
        os.path.join(
          IM_ROOT, "OMEGA_facet_score.png"
        ),
        dpi = 300
      )
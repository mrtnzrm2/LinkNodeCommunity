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
from networks_serial.scalehrh import SCALEHRH
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
list_of_lists = itertools.product(
  *[topologies, indices, KAV, MUT, MUW]
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
  for topology, index, kav, mut, muw in list_of_lists:
    # WDN paramters ----
    par = {
      "-N" : "{}".format(str(__nodes__)),
      "-k" : f"{kav}.0",
      "-maxk" : "30",
      "-mut" : f"{mut}",
      "-muw" : f"{muw}",
      "-beta" : "2.5",
      "-t1" : "2",
      "-t2" : "1",
      "-nmin" : "2",
      "-nmax" : "10"
    }
    data = read_class(
      "../pickle/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/{}/{}{}{}{}/{}/{}".format(
        str(__nodes__),
        par["-k"], par["-maxk"],
        par["-mut"], par["-muw"],
        par["-beta"], par["-t1"], par["-t2"],
        par["-nmin"], par["-nmax"], MAXI,
        linkage.upper(), l10, lup, _cut,
        __mode__, f"{topology}_{index}_{mapping}"
      ),
      "series_{}".format(MAXI)
    )
    if not isinstance(data, SCALEHRH): continue
    # print(data.data.shape[0])
    THE_DF = pd.concat(
      [
        THE_DF, 
        pd.DataFrame(
          {
            "NMI" : data.data.NMI.to_numpy().astype(float),
            # "OMEGA" : data.data.values.loc[data.data.sim == "OMEGA"].to_numpy().astype(float), 
            "score" : data.data.c,
            "topology": [f"{topology}"] * data.data.shape[0],
            "index" : [index] * data.data.shape[0],
            "kav" : [int(kav)] * data.data.shape[0],
            "mut" : [float(mut)] * data.data.shape[0],
            "muw" : [float(muw)] * data.data.shape[0]
          }
        )
      ], ignore_index=T
    )
  # Comparing feature, index, & score for given kav, mut, and muw
  KAV = [7, 15]
  MUT = [0.1, 0.3, 0.5]
  MUW = [0.1, 0.5]
  list_of_lists = itertools.product(
    *[KAV, MUT, MUW]
  )
  list_of_lists = np.array(list(list_of_lists))
  for kav, mut, muw in list_of_lists:
    print(kav, mut, muw)
    # Prepare path ----
    par = {
      "-N" : "{}".format(str(__nodes__)),
      "-k" : f"{kav}",
      "-maxk" : "30",
      "-mut" : f"{mut}",
      "-muw" : f"{muw}",
      "-beta" : "2.5",
      "-t1" : "2",
      "-t2" : "1",
      "-nmin" : "2",
      "-nmax" : "10"
    }
    IM_ROOT =  "../plots/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/{}/{}{}{}{}/{}".format(
        str(__nodes__),
        par["-k"], par["-maxk"],
        par["-mut"], par["-muw"],
        par["-beta"], par["-t1"], par["-t2"],
        par["-nmin"], par["-nmax"], MAXI,
        linkage.upper(), l10, lup, _cut,
        __mode__
      )
    # Prepare data ----
    data = THE_DF.loc[
      (THE_DF.kav == int(kav)) &
      (THE_DF.mut == float(mut)) &
      (THE_DF.muw == float(muw))
    ]
    data.NMI.loc[np.isnan(data.NMI)] = 0
    x = data.groupby(["topology", "index", "score"]).agg({"NMI": ["mean"]}).reset_index()
    x.columns = ["topology", "index", "score", "NMI"]
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
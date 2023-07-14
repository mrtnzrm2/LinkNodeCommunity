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
sns.set_theme()
import matplotlib.pyplot as plt
from pathlib import Path
## Personal libs ----
from various.network_tools import *

def draw_heatmap(*args, **kwargs):
  data = kwargs.pop('data')
  d = data.pivot(index=args[1], columns=args[0], values=args[2])
  sns.heatmap(d, **kwargs)

# Declare iter variables ----
number_of_nodes = [100, 150]
topologies = ["MIX"]
indices = [ "D1_2_2"]
MUT = [0.1, 0.3]
NMIN = [5]
NMAX = [25]
list_of_lists = itertools.product(
  *[number_of_nodes, topologies, indices, MUT, NMIN, NMAX]
)
list_of_lists = np.array(list(list_of_lists))
# Constant parameters ---
MAXI = 50
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
run = T
mapping = "trivial"
__mode__ = "BETA"
l10 = ""
lup = ""
_cut = ""
if nlog10: l10 = "_l10"
if lookup: lup = "_lup"
if cut: _cut = "_cut"
opt_score = ["_maxmu", "_X", "_D", "_S"]
if __name__ == "__main__":
  # Extract data ----
  THE_DF = pd.DataFrame()
  for __nodes__, topology, index, mut, nmin, nmax in list_of_lists:
    nmin = int(nmin)
    nmax = int(nmax)
    mut = float(mut)
    if nmax < nmin: continue
    # WDN paramters ----
    par = {
      "-N" : "{}".format(str(__nodes__)),
      "-k" : "7.0",
      "-maxk" : "20",
      "-mut" : f"{mut}",
      "-muw" : "0.01",
      "-beta" : "3",
      "-t1" : "2",
      "-t2" : "1",
      "-nmin" : f"{nmin}",
      "-nmax" : f"{nmax}"
    }
    data = read_class(
      "../pickle/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/{}/{}/{}/{}".format(
        str(__nodes__),
        par["-k"], par["-maxk"],
        par["-mut"], par["-muw"],
        par["-beta"], par["-t1"], par["-t2"],
        par["-nmin"], par["-nmax"], MAXI-1,
        __mode__, __mode__, f"{topology}_{index}_{mapping}"
      ),
      "series_{}".format(MAXI)
    )
    if isinstance(data, int): continue
    # print(data.data.shape[0])
    THE_DF = pd.concat(
      [
        THE_DF, 
        pd.DataFrame(
          {
            "NMI" : data.data.NMI.to_numpy().astype(float),
            "score" : [f"{index}{s}" for s in data.data.c],
            "topology": [f"{topology}"] * data.data.shape[0],
            "mut" : [mut] * data.data.shape[0],
            "size" : [f"{nmin}_{nmax}"] * data.data.shape[0]
          }
        )
      ], ignore_index=T
    )
  # Comparing feature, index, & score for given kav, mut, and muw
  list_of_lists = itertools.product(*[number_of_nodes, topologies])
  for __nodes__, tp in list_of_lists:
    __nodes__ = int(__nodes__)
    print(tp)
    # Prepare path ----
    IM_ROOT =  "../plots/RAN/scalefree/-N_{}/-k_7.0/-maxk_20/{}/{}{}{}{}".format(
      str(__nodes__), MAXI, linkage.upper(), l10, lup, _cut
    )
    # Prepare data ----
    data = THE_DF.loc[(THE_DF.topology == tp)]
    data.NMI.loc[np.isnan(data.NMI)] = 0
    sns.catplot(
      data=data,
      x="NMI",
      y="score",
      col="size",
      hue="mut",
      kind="box"
    )
    Path(IM_ROOT).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      os.path.join(
        IM_ROOT, f"NMI_{tp}_{__mode__}.png"
      ),
      dpi = 300
    )
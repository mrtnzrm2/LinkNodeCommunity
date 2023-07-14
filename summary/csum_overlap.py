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

# Declare iter variables ----
number_of_nodes = [100, 150]
topologies = ["MIX"]
indices = ["D1_2_2"]
MUT = [0.1, 0.3]
NMIN = [5]
NMAX = [25]
ON = [0.1, 0.3]
OM = [2, 3]
list_of_lists = itertools.product(
  *[number_of_nodes, topologies, indices, MUT, NMIN, NMAX, ON, OM]
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
__mode__ = "ALPHA"
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
  for __nodes__, topology, index, mut, nmin, nmax, on, om in list_of_lists:
    __nodes__ = int(__nodes__)
    nmin = int(nmin)
    nmax = int(nmax)
    mut = float(mut)
    on = float(on)
    on = int(__nodes__ * on)
    om = int(om)
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
      "-nmax" : f"{nmax}",
      "-on" : f"{on}",
      "-om" : f"{om}"
    }
    data = read_class(
      "../pickle/RAN/scalefree/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/-on_{}/-om_{}/{}/{}/{}/{}".format(
        str(__nodes__),
        par["-k"], par["-maxk"],
        par["-mut"], par["-muw"],
        par["-beta"], par["-t1"], par["-t2"],
        par["-nmin"], par["-nmax"],
        par["-on"], par["-om"], MAXI-1,
        __mode__, __mode__, f"{topology}_{index}_{mapping}"
      ),
      "series_{}".format(MAXI)
    )
    if isinstance(data, int): continue
    # print(data.data.shape[0])
    nnmi = data.data.loc[data.data.sim == "NMI"].shape[0]
    nomega = data.data.loc[data.data.sim == "NMI"].shape[0]
    THE_DF = pd.concat(
      [
        THE_DF, 
        pd.DataFrame(
          {
            "val" : np.hstack(
              [
                data.data["values"].loc[data.data.sim == "NMI"].to_numpy().astype(float),
                data.data["values"].loc[data.data.sim == "OMEGA"].to_numpy().astype(float)
              ]
            ),
            "sim" : ["NMI"] * nnmi + ["OMEGA"] * nomega,
            "score" : [f"{index}{s}" for s in data.data.c],
            "topology": [f"{topology}"] * (nnmi + nomega),
            "mut" : [mut] * (nnmi + nomega),
            "size" : [f"{nmin}_{nmax}"] * (nnmi + nomega),
            "overlapping" : [f"{on}_{om}"] * (nnmi + nomega)
          }
        )
      ], ignore_index=T
    )
  THE_DF.val.loc[np.isnan(THE_DF.val)] = 0
  list_of_lists = itertools.product(*[number_of_nodes, topologies, ON, OM])
  for __nodes__, tp, on, om in list_of_lists:
    __nodes__ = int(__nodes__)
    on = float(on)
    on = int(__nodes__ * on)
    print(tp, on, om)
    # Prepare path ----
    IM_ROOT =  "../plots/RAN/scalefree/-N_{}/-k_7.0/-maxk_20/{}/{}{}{}{}/{}_{}".format(
      str(__nodes__), MAXI, linkage.upper(), l10, lup, _cut, on, om
    )
    # Prepare data ----
    x = THE_DF.loc[(THE_DF.topology == tp) & (THE_DF.overlapping == f"{on}_{om}")]
    if x.shape[0] == 0: continue
    sns.catplot(
      data=x,
      x="val",
      y="score",
      col="sim",
      # row="sim",
      hue="mut",
      kind="box"
    )
    Path(IM_ROOT).mkdir(exist_ok=True, parents=True)
    plt.savefig(
      os.path.join(
        IM_ROOT, f"NMI_O_{tp}_{__mode__}.png"
      ),
      dpi = 300
    )
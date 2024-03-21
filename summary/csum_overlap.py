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
benchmark = ["WN", "WDN"]
number_of_nodes = [100, 150]
topologies = ["MIX"]
indices = ["Hellinger2"]
MUT = np.linspace(0.1, 0.8, 10)
NMIN = [5]
NMAX = [25]
ON = [0.1]
OM = [2, 3, 4]
discovery = ["discovery_7"]
list_of_lists = itertools.product(
  *[benchmark, number_of_nodes, topologies, indices, MUT, NMIN, NMAX, ON, OM, discovery]
)
list_of_lists = np.array(list(list_of_lists))
# Constant parameters ---
MAXI = 100
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
run = T
mapping = "trivial"
__mode__ = "ZERO"
l10 = ""
lup = ""
_cut = ""
if nlog10: l10 = "_l10"
if lookup: lup = "_lup"
if cut: _cut = "_cut"
opt_score = ["_S"]
if __name__ == "__main__":
  # Extract data ----
  THE_DF = pd.DataFrame()
  for mark, __nodes__, topology, index, mut, nmin, nmax, on, om, dis in list_of_lists:
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
      "-beta" : "2",
      "-t1" : "2",
      "-t2" : "3",
      "-nmin" : f"{nmin}",
      "-nmax" : f"{nmax}",
      "-on" : f"{on}",
      "-om" : f"{om}"
    }
    data = read_class(
      "../pickle/RAN/scalefree/{}/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/-on_{}/-om_{}/{}/{}/{}/{}/{}".format(
        mark,
        str(__nodes__),
        par["-k"], par["-maxk"],
        par["-mut"], par["-muw"],
        par["-beta"], par["-t1"], par["-t2"],
        par["-nmin"], par["-nmax"],
        par["-on"], par["-om"], MAXI-1,
        __mode__, dis, __mode__, f"{topology}_{index}_{mapping}"
      ),
      "series_{}".format(MAXI)
    )

    if isinstance(data, int): continue

    omega_label = r"$\omega$"
    mut_label = r"$\mu_{t}$"
    

    filtered_data = data.data.loc[
      (data.data["c"] == "_D") &
      (data.data["sim"] == "OMEGA") 
    ]

    nrows = filtered_data.shape[0]

    THE_DF = pd.concat(
      [
        THE_DF, 
        pd.DataFrame(
          {
            omega_label : filtered_data["values"].to_numpy().astype(float),
            mut_label : [np.round(mut, 2)] * (nrows),
            "Om" : [f"{om}"] * (nrows),
            "nodes" : [int(__nodes__)] * (nrows),
            "benchmark" : [mark] * (nrows)
          }
        )
      ], ignore_index=T
    )

  # Prepare plot ----
  IM_ROOT =  "../plots/RAN/scalefree/"

  # sns.set_context("talk")
  sns.set_style("whitegrid")
 
  g=sns.relplot(
    data=THE_DF,
    kind="line",
    col="nodes", row="benchmark",
    hue="Om", x=mut_label, y=omega_label,
    alpha=0.8
  )

  par = [("WN", 100), ("WN", 150), ("WDN", 100), ("WDN", 150)]

  for i, ax in enumerate(g.axes.flatten()):
    b, n = par[i]
    subdata = THE_DF.loc[(THE_DF["benchmark"] == b) & (THE_DF["nodes"] == n)]
    subdata = subdata.groupby([mut_label, "Om"])[omega_label].mean().reset_index()
    sns.scatterplot(
      data=subdata,
      x=mut_label, y=omega_label,
      hue="Om",
      s=50,
      ax=ax
    )


  plt.gcf().tight_layout()

  Path(IM_ROOT).mkdir(exist_ok=True, parents=True)
  plt.savefig(
    os.path.join(
      IM_ROOT, "omega.svg"
    ),
    # dpi = 300,
    transparent=T
  )
  plt.close()
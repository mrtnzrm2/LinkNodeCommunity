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
benchmark = ["BN"]
number_of_nodes = [1000]
topologies = ["MIX"]
indices = ["Hellinger2"]
MUT = [0.3]
NMIN = [10]
NMAX = [50]
ON = [0.1]
OM = np.arange(2, 8)
discovery = ["discovery_7"]
list_of_lists = itertools.product(
  *[benchmark, number_of_nodes, topologies, indices, MUT, NMIN, NMAX, ON, OM, discovery]
)
list_of_lists = np.array(list(list_of_lists))
# Constant parameters ---
MAXI = 10
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
      "-k" : "10.0",
      "-maxk" : "50",
      "-mut" : f"{mut}",
      "-t1" : "2",
      "-t2" : "1",
      "-nmin" : f"{nmin}",
      "-nmax" : f"{nmax}",
      "-on" : f"{on}",
      "-om" : f"{om}",
      "-fixed_range" : True
    }
    data = read_class(
      "../pickle/RAN/scalefree/{}/-N_{}/-k_{}/-maxk_{}/-mut_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/-on_{}/-om_{}/-fixed_range_True/{}/{}/{}/{}/{}".format(
        mark,
        str(__nodes__),
        par["-k"], par["-maxk"],
        par["-mut"],
        par["-t1"], par["-t2"],
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
 
  # g=sns.relplot(
  #   data=THE_DF,
  #   kind="line",
  #   col="nodes", row="benchmark",
  #   x="om", y=omega_label,
  #   alpha=0.8
  # )

  # sns.set_context("talk")
  sns.set_style("whitegrid")

  sns.lineplot(
    data=THE_DF,
    x="Om", y=omega_label
  )

  mean_THE_DF = THE_DF.groupby(["Om"])[omega_label].mean().reset_index()

  sns.scatterplot(
    data=mean_THE_DF,
    x="Om", y=omega_label,
    s=50
  )

  plt.gca().set_title("benchmark = BN | nodes = 1000s | " + r"$\mu_{t}=0.3$ " + "| On=10%")
  plt.gcf().tight_layout()

  Path(IM_ROOT).mkdir(exist_ok=True, parents=True)
  plt.savefig(
    os.path.join(
      IM_ROOT, "omega_BN.png"
    ),
    dpi = 300,
    # transparent=T
  )
  plt.close()
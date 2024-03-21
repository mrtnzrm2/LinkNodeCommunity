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
indices = [ "Hellinger2"]
benchmark = ["WN", "WDN"]
MUT = np.linspace(0.1, 0.8, 10)
NMIN = [5]
NMAX = [25]
list_of_lists = itertools.product(
  *[benchmark, number_of_nodes, topologies, indices, MUT, NMIN, NMAX]
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
opt_score = ["_X", "_D", "_S"]
if __name__ == "__main__":
  # Extract data ----
  THE_DF = pd.DataFrame()
  for mark, __nodes__, topology, index, mut, nmin, nmax in list_of_lists:
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
      "-beta" : "2",
      "-t1" : "2",
      "-t2" : "3",
      "-nmin" : f"{nmin}",
      "-nmax" : f"{nmax}"
    }
    data_path = "../pickle/RAN/scalefree/{}/-N_{}/-k_{}/-maxk_{}/-mut_{}/-muw_{}/-beta_{}/-t1_{}/-t2_{}/-nmin_{}/-nmax_{}/{}/{}/{}/{}".format(
        mark,
        str(__nodes__),
        par["-k"], par["-maxk"],
        par["-mut"], par["-muw"],
        par["-beta"], par["-t1"], par["-t2"],
        par["-nmin"], par["-nmax"], MAXI-1,
        __mode__, __mode__, f"{topology}_{index}_{mapping}"
      )
    data = read_class(data_path, "series_{}".format(MAXI))
    if isinstance(data, int):
      print(f">> {data_path}")
      continue
    
    mut_label = r"$\mu_{t}$"
    THE_DF = pd.concat(
      [
        THE_DF, pd.DataFrame({
          "NMI" : data.data.NMI.to_numpy().astype(float),
          "score" : data.data.c,
          mut_label : [np.round(mut,2)] * data.data.shape[0],
          "benchmark" : [mark]* data.data.shape[0],
          "nodes" : [int(__nodes__)] * data.data.shape[0]})
      ], ignore_index=T
    )
  THE_DF["score"] = [s.split("_")[1] for s in THE_DF["score"]]
  THE_DF["score"].loc[THE_DF["score"] == "S"] = r"$S_{L}$"
  # Comparing feature, index, & score for given kav, mut, and muw
  # Prepare path ----
  IM_ROOT =  "../plots/RAN/scalefree/"
  # Prepare data ----
  
  g=sns.relplot(
    data=THE_DF,
    kind="line",
    col="nodes", row="benchmark",
    hue="score", x=mut_label, y="NMI",
    alpha=0.8
  )

  # mean_data = THE_DF.groupby(["benchmark", "nodes", "score", mut_label])["NMI"].mean()
  # mean_data = mean_data.reset_index()

  # for ax, mark, n in zip(g.axes.flat, ["WN", "WN", "WDN", "WDN"], [100, 150, 100, 150]):
  #   # sns.scatterplot(
  #   #   data=mean_data.loc[(mean_data["nodes"] == n) & (mean_data["benchmark"] == mark)],
  #   #   x=mut_label, y="NMI", hue="score", ax=ax, legend=F, s=100
  #   # )
  #   plt.setp(ax.collections, alpha=.6) #for the markers
  #   plt.setp(ax.lines, alpha=.6)       #for the lines
  #   ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

  # plt.gcf().tight_layout()

  Path(IM_ROOT).mkdir(exist_ok=True, parents=True)
  plt.savefig(
    os.path.join( 
      IM_ROOT, "NMI.svg"
    ),
    dpi = 300, transparent=T
  )
  plt.close()
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
topologies = ["TARGET", "SOURCE", "MIX"]
distbases = ["PARETO", "PARETOTRUNC", "EXPMLE", "EXPTRUNC", "LINEAR", "LINEARTRUNC"]
bias = [1e-5, 1e-2, 0.1, 0.3, 0.5]
bins = [12, 50]
list_of_lists = itertools.product(
  *[topologies, distbases, bias, bins]
)
list_of_lists = np.array(list(list_of_lists))
# Constant parameters ---
MAXI = 500
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
run = T
structure = "FLN"
distance = "MAP3D"
nature = "original"
mode = "ALPHA"
mapping = "R2"
index  = "jacw"
imputation_method = ""
opt_score = ["_maxmu", "_X", "_D"]
l10 = ""
lup = ""
_cut = ""
if nlog10: l10 = "_l10"
if lookup: lup = "_lup"
# Basic network parameters ----
total_nodes = 106
__inj__ = 57
__nodes__ = 57
__version__ = 220830
if __name__ == "__main__":
  # Extract data ----
  THE_DF = pd.DataFrame()
  THE_DF_S = pd.DataFrame()
  for topology, model, bias, _bin_ in list_of_lists:
    bias = float(bias)
    _bin_ = int(_bin_)
    data = read_class(
      "../pickle/RAN/distbase/MAC/{}/{}/{}/{}/BIN_{}/{}/{}/{}/{}".format(
        __version__,
        structure,
        distance,
        model,
        str(_bin_),
        f"{linkage.upper()}_{total_nodes}_{__nodes__}{l10}{lup}{_cut}",
        mode,
        f"{topology}_{index}_{mapping}",
        f"b_{bias}"
      ),
      "series_{}".format(MAXI)
    )
    if isinstance(data, int): continue
    THE_DF = pd.concat(
      [
        THE_DF, 
        pd.DataFrame(
          {
            "values" : data.data["values"].to_numpy().astype(float),
            "sim" : data.data.sim,
            "score" : data.data.score,
            "topology": [f"{topology}"] * data.data.shape[0],
            "bins" : [_bin_ - 1] * data.data.shape[0],
            "bias" : [bias] * data.data.shape[0],
            "model" : [model] * data.data.shape[0]
          }
        )
      ], ignore_index=T
    )
    link_entropy = data.link_entropy
    link_entropy = link_entropy.loc[(link_entropy.c == "link_hierarchy") & (link_entropy.dir == "H") & (link_entropy.data == "0")]
    SH = link_entropy.groupby("iter").max()["S"].to_numpy()
    level = link_entropy.groupby("iter").max()["level"].to_numpy()
    THE_DF_S = pd.concat(
      [
        THE_DF_S, 
        pd.DataFrame(
          {
            "val" : np.hstack([SH, level]),
            "stat" : ["S"] * SH.shape[0] + ["level"] * SH.shape[0],
            "topology": [f"{topology}"] * 2 *SH.shape[0],
            "bins" : [_bin_ - 1] * 2 * SH.shape[0],
            "bias" : [bias] * 2 * SH.shape[0],
            "model" : [model] * 2 * SH.shape[0]
          }
        )
      ], ignore_index=T
    )
  # Marginalizing data DF ----
  THE_DF.score = [s.split("_")[1] for s in THE_DF.score]
  THE_DF = THE_DF.loc[~np.isnan(THE_DF["values"])]
  topologies = ["TARGET", "SOURCE", "MIX"]
  bias = [1e-5, 1e-2, 0.1, 0.3, 0.5]
  list_of_lists = itertools.product(*[topologies, bias])
  list_of_lists = np.array(list(list_of_lists))
  for tp, b in list_of_lists:
    b = float(b)
    print(tp, b)
    # Prepare path ----
    IM_ROOT =  "../plots/RAN/distbase/MAC/{}/{}/{}/summary/{}/{}/{}".format(
      __version__,
      structure,
      distance,
      f"{linkage.upper()}_{total_nodes}_{__nodes__}{l10}{lup}{_cut}",
      mode,
      f"{tp}_{index}_{mapping}"
    )
    # Prepare data ----
    sns.catplot(
      data=THE_DF.loc[(THE_DF.topology == tp) & (THE_DF.bias == b)],
      y="values",
      x="model",
      hue="score",
      col="bins",
      row="sim",
      kind="box",
      aspect=1.5
    )
    # plt.tight_layout()
    Path(IM_ROOT).mkdir(parents=True, exist_ok=True)
    plt.savefig(
      os.path.join(
        IM_ROOT, f"sim_value_b_{b}.png"
      ),
      dpi = 300
    )
    plt.close()
  # Marginalizing data S ----
  for b in [1e-5, 1e-2, 0.1, 0.3, 0.5]:
    b = float(b)
    print(b)
    # Prepare path ----
    IM_ROOT =  "../plots/RAN/distbase/MAC/{}/{}/{}/summary/{}/{}".format(
      __version__,
      structure,
      distance,
      f"{linkage.upper()}_{total_nodes}_{__nodes__}{l10}{lup}{_cut}",
      mode
    )
    # Prepare data ----
    sns.catplot(
      data=THE_DF_S.loc[(THE_DF_S.bias == b)],
      y="val",
      x="model",
      hue="topology",
      col="bins",
      row="stat",
      kind="box",
      aspect=1.5,
      sharey=False
    )
    # plt.tight_layout()
    Path(IM_ROOT).mkdir(parents=True, exist_ok=True)
    plt.savefig(
      os.path.join(
        IM_ROOT, f"S_b_{b}.png"
      ),
      dpi = 300
    )
    plt.close()
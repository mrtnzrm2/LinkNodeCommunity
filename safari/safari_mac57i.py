# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ----
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
# Personal libs ---- 
from networks.MAC.mac57 import MAC57
from networks.MAC.mac57i import MAC57i
from various.network_tools import *


# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
structure = "FLN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0
opt_score = ["_D", "_S"]
save_data = T

# Start main ----9
if __name__ == "__main__":
  # Load structure ----
  NET57 = MAC57(
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    version = "57d106",
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = 57,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    b = bias
  )

  NET106i = MAC57i(
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    nature = nature,
    model = imputation_method,
    distance = distance,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    b = bias
  )

  area = ["mip", "stpr", "8m", "v1pclf", "24b", "v2c", "45a", "opro", "teo", "lip"]
  iA = match(area, NET57.struct_labels)

  data = pd.DataFrame()

  for j, i in enumerate(iA):
    data = pd.concat(
      [
        data,
        pd.DataFrame(
          {
            "D" : list(NET57.D[i, :57]) + list(NET106i.D[i, 57:]),
            "FLNe" : list(NET57.A[i, :]) + list(NET106i.A[i, 57:]),
            "cat" : ["GT"] * 57 + ["RF"] * (106-57),
            "area" : [area[j]] * 106
          }
        )
      ],
      ignore_index=True
    )

  data = data.loc[data["FLNe"] > 0]
  data["FLNe"] = np.log10(data["FLNe"]) + 7


  g = sns.FacetGrid(
    data=data,
    col="area",
    hue="cat",
    col_wrap=3
  )

  g.map_dataframe(
    sns.scatterplot,
    x="D",
    y="FLNe",
  )

  g.add_legend()
  g.set_ylabels(r"$\log(FLNe) + 7$")

  plt.show()
# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
#Import libraries ----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
# Personal libraries ----
from networks.structure import STR
from various.network_tools import *

def omega(common_features, ax : plt.Axes, mode="ZERO", iterations=500, cmap="deep"):
  path = "../pickle/RAN/distbase/{}/{}/{}/{}/{}/BIN_{}/{}/{}/{}/MIX_Hellinger2_trivial/discovery_7".format(
     common_features["subject"],
     common_features["version"],
     common_features["structure"],
     common_features["distance"],
     common_features["model_distbase"],
     common_features["bins"],
    "",#  common_features["fitter"],
     common_features["subfolder"],
     mode
  )
  # iterations2=100
  H_EDR = read_class(path, f"series_{iterations}")
  # H_EDR = read_class(path, f"series_{iterations2}")

  path = "../pickle/RAN/swaps/{}/{}/{}/{}/{}/{}/{}/MIX_Hellinger2_trivial/discovery_7".format(
     common_features["subject"],
     common_features["version"],
     common_features["structure"],
    #  common_features["distance"],
     "MAP3D",
     common_features["model_swaps"],
     common_features["subfolder"],
     mode
  )
  H_CONG = read_class(path, f"series_{iterations}")
  edr_data =H_EDR.data
  cong_data = H_CONG.data
  edr_data["score"] = [s.replace("_", "") for s in edr_data["score"]]
  cong_data["score"] = [s.replace("_", "") for s in cong_data["score"]]
  edr_data = edr_data.loc[(edr_data.score == "S") & (edr_data.sim == "OMEGA") & (edr_data.direction == "both")]
  cong_data = cong_data.loc[(cong_data.score == "S") & (cong_data.sim == "OMEGA") & (cong_data.direction == "both")]

  data = pd.DataFrame(
    {
      "omega" : list(edr_data["values"]) + list(cong_data["values"]),
      "model" : ["EDR"] * edr_data.shape[0] + ["Configuration"] * cong_data.shape[0]
    }
  )

  print(data.groupby("model").mean())

  from scipy.stats import ttest_ind, ttest_1samp

  test = ttest_ind(data.omega.loc[data.model == "EDR"], data.omega.loc[data.model == "Configuration"], alternative="greater", equal_var=False)
  test_conf = ttest_1samp(data.omega.loc[data.model == "Configuration"], 0)

  sns.histplot(
    data=data,
    x="omega",
    hue="model",
    stat="density",
    multiple="layer",
    palette=cmap,
    common_bins=False,
    common_norm=False
  )

  if  not np.isnan(test.pvalue): 
    if test.pvalue > 0.05:
      a = "n.s."
    elif test.pvalue <= 0.05 and test.pvalue > 0.001:
      a = "*" 
    elif test.pvalue <= 0.001 and test.pvalue > 0.0001:
      a = "**" 
    else:
      a = "***"
  else:
    a = "nan"

  width_min = ax.get_xbound()[0]
  width_max = ax.get_xbound()[1]


  omega_mean = np.nanmean(data.omega.loc[data.model == "EDR"])
  omega_t = (omega_mean - width_min) / (width_max - width_min)
  ax.text(omega_t, 1.005, f"{omega_mean:.2f}", transform=ax.transAxes, horizontalalignment="center")
  plt.axvline(omega_mean, linestyle="--", color="r", linewidth=1, alpha=0.6)

  if  not np.isnan(test_conf.pvalue): 
    if test_conf.pvalue > 0.05:
      a = "ns"
    elif test_conf.pvalue <= 0.05 and test_conf.pvalue > 0.001:
      a = "*" 
    elif test_conf.pvalue <= 0.001 and test_conf.pvalue > 0.0001:
      a = "**" 
    else:
      a = "***"
  else:
    a = "nan" 

  omega_mean = np.nanmean(data.omega.loc[data.model == "Configuration"])
  omega_t = (omega_mean - width_min) / (width_max - width_min)
  ax.text(omega_t, 1.005, f"{omega_mean:.2f}", transform=ax.transAxes, horizontalalignment="center")
  plt.axvline(omega_mean, linestyle="--", color="r", linewidth=1, alpha=0.6)
# 
  # ax.set_xlabel(r"Omega index $(\omega)$")

  # plt.axvline(0.4770, linestyle="-", color="gray", linewidth=1, alpha=0.6)

  ax.set_xlabel(r"$\omega$")

  legend = ax.get_legend()
  legend.set_title("")

  legend.set_bbox_to_anchor([0.3, 0.65], transform=ax.transAxes)

  for label in legend.get_texts():
    label.set_fontsize(8.5)

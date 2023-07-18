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
from networks.MAC.mac40 import MAC40
from various.network_tools import *

def plot_ratio_area(Area):
  AL = Area
  # Get structure ----
  file = pd.read_csv(f"{NET.csv_path}/corticoconnectiviy database_kennedy-knoblauch-team-1_distances completed.csv")
  ## Areas to index
  tlabel = np.unique(file.TARGET)
  inj = tlabel.shape[0]
  slabel = np.unique(file.SOURCE)
  tareas = slabel.shape[0]
  slabel1 = [lab for lab in slabel if lab not in tlabel]
  slabel = np.array(list(tlabel) + slabel1)
  file["SOURCE_IND"] = match(file.SOURCE, slabel)
  file["TARGET_IND"] = match(file.TARGET, slabel)
  ## Average Count
  monkeys = np.unique(file.MONKEY)
  C = []
  for m in monkeys:
    Cm = np.zeros((tareas, inj)) * np.nan
    data_m = file.loc[file.MONKEY == m]
    Cm[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m.TOTAL
    C.append(Cm)
  C = np.array(C)
  ##
  monkey_target_area = np.unique(file["MONKEY"].loc[file.TARGET == AL])
  ##
  A = match([AL], slabel)[0]
  area_target = C[:, :, A]
  area_target[np.isnan(area_target)] = 0
  TMN = []

  fig, ax = plt.subplots(1, 1)

  for m in np.arange(area_target.shape[0]):
      for i in np.arange(area_target.shape[1]):
        for j in np.arange(i + 1, area_target.shape[1]):
          # if np.minimum(area_target[m, i], area_target[m, j]) == 0: continue
          if monkeys[m] in monkey_target_area:
            # t = (1 + np.maximum(area_target[m, i], area_target[m, j])) / (1 + np.minimum(area_target[m, i], area_target[m, j]))
            t = (1 + area_target[m, i]) / (1 + area_target[m, j])
            # aij = np.sort([slabel[i], slabel[j]])
            aij = f"{slabel[i]}_{slabel[j]}"
            TMN.append([aij, monkeys[m], t])

  TD = pd.DataFrame(TMN, columns=["Pair", "Monkey", "Ratio"])
  mean_td = TD.groupby("Pair")["Ratio"].mean().reindex().sort_values(ascending=False)
  mean_td = pd.DataFrame({
    "Pair" : mean_td.index.to_numpy(),
    "Ratio" : mean_td.to_numpy()
  })
  f = mean_td.Pair.to_numpy()[:5]

  # TD = TD.loc[np.isin(TD.Pair, f)]
  # mean_td = mean_td.loc[np.isin(mean_td.Pair, f)]
  

  sns.lineplot(
    data=mean_td,
    x="Pair",
    y="Ratio",
    color="red",
    alpha=0.6,
    ax=ax
  )

  sns.scatterplot(
    data=TD,
    x="Pair",
    y="Ratio",
    hue="Monkey",
    alpha=0.6,
    s=4,
    ax=ax
  )

  ax.set_ylabel(r"$\frac{1+\max(N_{ik}, N_{jk})}{1+\min(N_{ik}, N_{jk})}$", fontsize=20)
  ax.set_xlabel("sort(i, j)")
  ax.set_yscale("log")
  ax.axes.get_xaxis().set_ticks([])
  ax.set_title(f"Target area k={AL}")# + " | " + r"$\min(N_{ik}, N_{jk})>0$")
  fig.set_figheight(6)
  fig.set_figwidth(10)

  plt.savefig(f"../plots/MAC/40d91/cortex_letter/ratio_{AL}_M_.png", dpi=300)

def plot_ratio_area_B(Area):
  AL = Area
  # Get structure ----
  file = pd.read_csv(f"{NET.csv_path}/corticoconnectiviy database_kennedy-knoblauch-team-1_distances completed.csv")
  ## Areas to index
  tlabel = np.unique(file.TARGET)
  inj = tlabel.shape[0]
  slabel = np.unique(file.SOURCE)
  tareas = slabel.shape[0]
  slabel1 = [lab for lab in slabel if lab not in tlabel]
  slabel = np.array(list(tlabel) + slabel1)
  file["SOURCE_IND"] = match(file.SOURCE, slabel)
  file["TARGET_IND"] = match(file.TARGET, slabel)
  ## Average Count
  monkeys = np.unique(file.MONKEY)
  C = []
  sum_A = np.zeros(np.unique(file.TARGET_IND).shape[0])
  for m in monkeys:
    Cm = np.zeros((tareas, inj)) * np.nan
    data_m = file.loc[file.MONKEY == m]
    Cm[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m.TOTAL
    cm_tgt = np.unique(data_m.TARGET_IND)
    sum_A[cm_tgt] += 1
    C.append(Cm)
  C = np.array(C)
  CC = C.copy()
  CC[np.isnan(CC)] = 0
  CC = np.nansum(CC, axis=0)
  CC = CC / sum_A
  ##
  monkey_target_area = np.unique(file["MONKEY"].loc[file.TARGET == AL])
  ##
  A = match([AL], slabel)[0]
  area_target = C[:, :, A]
  area_target[np.isnan(area_target)] = 0
  TMN = []

  fig, ax = plt.subplots(1, 1)

  for m in np.arange(area_target.shape[0]):
      nxxb = np.nansum(CC[A, :]) / np.nansum(CC[A, :] > 0)
      norm = np.nansum(area_target[m, :])
      if monkeys[m] in monkey_target_area:
        TMN.append([AL, monkeys[m], nxxb / (nxxb + norm), "norm_B"]) 
      for i in np.arange(area_target.shape[1]):
        if monkeys[m] in monkey_target_area:
          t = area_target[m, i] / (norm + nxxb)
          t_ = area_target[m, i] / (norm)
          aij = f"{slabel[i]}"
          TMN.append([aij, monkeys[m], t, "norm_B"])
          TMN.append([aij, monkeys[m], t_, "FLNe"])


  TD = pd.DataFrame(TMN, columns=["SOURCE", "Monkey", "W", "var"])
  TD = TD.loc[TD.W > 0] 
  T1 = TD.loc[TD["var"] == "norm_B"].sort_values("W", ascending=False)
  T2 = TD.loc[TD["var"] == "FLNe"]
  TD = pd.concat([T1, T2], ignore_index=True)
  mean_td = TD.groupby(["SOURCE", "var"])["W"].mean().reset_index()
  mean_td = pd.DataFrame({
    "SOURCE" : mean_td["SOURCE"].to_numpy(),
    "var" : mean_td["var"].to_numpy(),
    "W" : mean_td["W"].to_numpy()
  })

  mean_td1 = mean_td.loc[mean_td["var"] == "norm_B"].sort_values("W", ascending=False)
  mean_td2 = mean_td.loc[mean_td["var"] == "FLNe"]
  mean_td = pd.concat([mean_td1, mean_td2], ignore_index=True)
  # # TD = TD.loc[np.isin(TD.Pair, f)]
  # # mean_td = mean_td.loc[np.isin(mean_td.Pair, f)]

  sns.lineplot(
    data=mean_td,
    x="SOURCE",
    y="W",
    hue="var",
    alpha=0.6,
    ax=ax
  )
  sns.scatterplot(
    data=TD,
    x="SOURCE",
    y="W",
    hue="var",
    alpha=0.6,
    s=12,
    ax=ax
  )
  ax.set_title(AL)
  plt.xticks(rotation=90)
  plt.yscale("log")
  fig.set_figheight(6)
  fig.set_figwidth(10)
  fig.tight_layout()
  plt.savefig(f"../plots/MAC/40d91/cortex_letter/norm_B_{AL}_.png", dpi=300)
  plt.close()

def single_flne_lne():
  AL = "10"
  # Get structure ----
  file = pd.read_csv(f"{NET.csv_path}/corticoconnectiviy database_kennedy-knoblauch-team-1_distances completed.csv")
  ## Areas to index
  tlabel = np.unique(file.TARGET)
  inj = tlabel.shape[0]
  slabel = np.unique(file.SOURCE)
  tareas = slabel.shape[0]
  slabel1 = [lab for lab in slabel if lab not in tlabel]
  slabel = np.array(list(tlabel) + slabel1)
  file["SOURCE_IND"] = match(file.SOURCE, slabel)
  file["TARGET_IND"] = match(file.TARGET, slabel)
  ## Average Count
  monkeys = np.unique(file.MONKEY)
  C = []
  sum_A = np.zeros(np.unique(file.TARGET_IND).shape[0])
  for m in monkeys:
    Cm = np.zeros((tareas, inj)) * np.nan
    data_m = file.loc[file.MONKEY == m]
    Cm[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m.TOTAL
    cm_tgt = np.unique(data_m.TARGET_IND)
    sum_A[cm_tgt] += 1
    C.append(Cm)
  C = np.array(C)
  CC = C.copy()
  CC[np.isnan(CC)] = 0
  CC = np.nansum(CC, axis=0)
  CC = CC / sum_A
  ##
  monkey_target_area = np.unique(file["MONKEY"].loc[file.TARGET == AL])
  ##
  A = match([AL], slabel)[0]
  area_target = C[:, :, A]
  area_target[np.isnan(area_target)] = 0
  TMN = []

  fig, ax = plt.subplots(1, 1)

  for m in np.arange(area_target.shape[0]):
      norm = np.nansum(area_target[m, :])
      for i in np.arange(area_target.shape[1]):
        if monkeys[m] in monkey_target_area:
          t = area_target[m, i]
          t_ = area_target[m, i] / (norm)
          aij = f"{slabel[i]}"
          TMN.append([aij, monkeys[m], t, "LNe"])
          TMN.append([aij, monkeys[m], t_, "FLNe"])


  TD = pd.DataFrame(TMN, columns=["SOURCE", "Monkey", "W", "var"])
  TD = TD.loc[TD.W > 0] 
  T1 = TD.loc[TD["var"] == "LNe"].sort_values("W", ascending=False)
  T2 = TD.loc[TD["var"] == "FLNe"]
  TD = pd.concat([T1, T2], ignore_index=True)
  mean_td = TD.groupby(["SOURCE", "var"])["W"].mean().reset_index()
  mean_td = pd.DataFrame({
    "SOURCE" : mean_td["SOURCE"].to_numpy(),
    "var" : mean_td["var"].to_numpy(),
    "W" : mean_td["W"].to_numpy()
  })

  mean_td1 = mean_td.loc[mean_td["var"] == "LNe"].sort_values("W", ascending=False)
  mean_td2 = mean_td.loc[mean_td["var"] == "FLNe"]
  mean_td = pd.concat([mean_td1, mean_td2], ignore_index=True)
  # # TD = TD.loc[np.isin(TD.Pair, f)]
  # # mean_td = mean_td.loc[np.isin(mean_td.Pair, f)]

  sns.lineplot(
    data=mean_td,
    x="SOURCE",
    y="W",
    hue="var",
    alpha=0.6,
    ax=ax
  )
  sns.scatterplot(
    data=TD,
    x="SOURCE",
    y="W",
    hue="var",
    alpha=0.6,
    s=12,
    ax=ax
  )
  ax.set_title(AL)
  plt.xticks(rotation=90)
  plt.yscale("log")
  fig.set_figheight(6)
  fig.set_figwidth(10)
  fig.tight_layout()
  plt.savefig(f"../plots/MAC/40d91/cortex_letter/FLNe_vs_LNe_.png", dpi=300)
  plt.close()

# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
structure = "FLN"
mode = "ALPHA"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "simple2"
bias = 0
opt_score = ["_maxmu", "_X", "_D"]
save_data = T
version = "40d91"
__nodes__ = 40
__inj__ = 40
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC40(
    linkage, mode,
    nlog10 = nlog10,
    structure = structure,
    lookup = lookup,
    version = version,
    nature = nature,
    model = imputation_method,
    distance = distance,
    inj = __inj__,
    topology = topology,
    index = index,
    mapping = mapping,
    cut = cut,
    b = bias
  )
  
  D = NET.D[:, :__nodes__]
  # Get structure ----
  file = pd.read_csv(f"{NET.csv_path}/corticoconnectiviy database_kennedy-knoblauch-team-1_distances completed.csv")
  ## Areas to index
  tlabel = np.unique(file.TARGET)
  inj = tlabel.shape[0]
  slabel = np.unique(file.SOURCE)
  tareas = slabel.shape[0]
  slabel1 = [lab for lab in slabel if lab not in tlabel]
  slabel = np.array(list(tlabel) + slabel1)
  file["SOURCE_IND"] = match(file.SOURCE, slabel)
  file["TARGET_IND"] = match(file.TARGET, slabel)
  ## Average Count
  monkeys = np.unique(file.MONKEY)
  C = []
  sum_A = np.zeros(np.unique(file.TARGET_IND).shape[0])
  for m in monkeys:
    Cm = np.zeros((tareas, inj)) * np.nan
    data_m = file.loc[file.MONKEY == m]
    Cm[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m.TOTAL
    cm_tgt = np.unique(data_m.TARGET_IND)
    sum_A[cm_tgt] += 1
    C.append(Cm)
  C = np.array(C)
  CC = C.copy()
  CC[np.isnan(CC)] = 0
  CC = np.nansum(CC, axis=0)
  CC = CC / sum_A
  ##
  ##
  TMN = []
  ALS = ["10", "V1"]
  for AL in ALS:
    monkey_target_area = np.unique(file["MONKEY"].loc[file.TARGET == AL])
    A = match([AL], slabel)[0]
    area_target = C[:, :, A]
    D_A = D[:, A]
    area_target[np.isnan(area_target)] = 0
    for m in np.arange(area_target.shape[0]):
        norm = np.nansum(area_target[m, :])
        for i in np.arange(area_target.shape[1]):
          if monkeys[m] in monkey_target_area:
            t = area_target[m, i]
            t_ = area_target[m, i] / (norm)
            aij = f"{slabel[i]}"
            TMN.append([aij, monkeys[m],  D_A[i], t, "ANLNe", AL])
            TMN.append([aij, monkeys[m], D_A[i], t_, "FLNe", AL])


  TD = pd.DataFrame(TMN, columns=["SOURCE AREA", "Monkey", "D", "W", "var", "TARGET"])
  TD = TD.loc[TD.W > 0] 
  T1 = TD.loc[TD["var"] == "ANLNe"].sort_values("D", ascending=True)
  T2 = TD.loc[TD["var"] == "FLNe"]
  TD = pd.concat([T1, T2], ignore_index=True)
  mean_td = TD.groupby(["SOURCE AREA", "TARGET", "var", "D"])["W"].mean().reset_index()
  mean_td = pd.DataFrame({
    "SOURCE AREA" : mean_td["SOURCE AREA"].to_numpy(),
    "TARGET" : mean_td["TARGET"].to_numpy(),
    "var" : mean_td["var"].to_numpy(),
    "W" : mean_td["W"].to_numpy(),
    "D" : mean_td["D"].to_numpy()
  })

  mean_td1 = mean_td.loc[mean_td["var"] == "ANLNe"].sort_values("D", ascending=True)
  mean_td2 = mean_td.loc[mean_td["var"] == "FLNe"]
  mean_td = pd.concat([mean_td1, mean_td2], ignore_index=True)
  
  TD["TARGET AREA/TYPE"] = [f"{i}/{j}" for i, j in zip(TD.TARGET, TD["var"])]
  mean_td["TARGET AREA/TYPE"] = [f"{i}/{j}" for i, j in zip(mean_td.TARGET, mean_td["var"])]

  fig, ax = plt.subplots(1, 1)
  sns.lineplot(
    data=mean_td,
    x="SOURCE AREA",
    y="W",
    hue="TARGET AREA/TYPE",
    alpha=0.6,
    legend=False,
    ax=ax
  )
  sns.scatterplot(
    data=TD,
    x="SOURCE AREA",
    y="W",
    hue="TARGET AREA/TYPE",
    alpha=0.6,
    s=12,
    ax=ax
  )
  plt.xticks(rotation=90)
  plt.yscale("log")
  plt.ylabel("Weights")
  fig.set_figheight(8)
  fig.set_figwidth(16)
  fig.tight_layout()
  plt.savefig(f"../plots/MAC/40d91/cortex_letter/FLNe_vs_ANLNe_.png", dpi=300)
  plt.close()
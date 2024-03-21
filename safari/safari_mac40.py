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

def get_structure_new(csv_path, labels):
    # Get structure ----
    file = pd.read_csv(f"{csv_path}/Neuron_2021_Database.csv")
    ## Areas to index
    tlabel = np.unique(file["TARGET"]).astype(str)
    inj = tlabel.shape[0]
    slabel = np.unique(file["SOURCE"])
    total_areas = slabel.shape[0]
    slabel1 = [lab for lab in slabel if lab not in tlabel]
    slabel = np.array(list(tlabel) + slabel1, dtype="<U21").astype(str)
    file["SOURCE_IND"] = match(file["SOURCE"], slabel)
    file["TARGET_IND"] = match(file["TARGET"], slabel)
    ## Average Count
    monkeys = np.unique(file.MONKEY)
    C = []
    tid = np.unique(file.TARGET_IND)
    tmk = {t : [] for t in tid}
    for i, m in enumerate(monkeys):
      Cm = np.zeros((total_areas, inj))
      data_m = file.loc[file.MONKEY == m]
      Cm[data_m.SOURCE_IND, data_m.TARGET_IND] = data_m["TOTAL"]
      C.append(Cm)
      for t in np.unique(data_m.TARGET_IND):
        tmk[t].append(i)
    C = np.array(C)
    C[np.isnan(C)] = 0
    CC = np.sum(C, axis=0)
    c = np.zeros((total_areas, inj))
    for t, mnk in tmk.items(): c[:, t] = np.mean(C[mnk, :, t], axis=0)
    A = CC / np.sum(CC, axis=0)

    slabel = np.char.lower(slabel)
    tlabel = np.char.lower(tlabel)

    # original = np.array(["ins", "ento", "pole", "pir", "pi", "sub", "peri"])
    # reference = np.array(["insula", "entorhinal", "temporal_pole", "piriform", "parainsula", "subiculum", "perirhinal"])


    # slabel[match(original, slabel)] = reference
    # # tlabel[match(original, tlabel)] = reference



    # # np.savetxt(f"{self.csv_path}/labels.csv", self.struct_labels,  fmt='%s')
    A = pd.DataFrame(A, index=slabel, columns=tlabel)
    return A[labels[:inj]].reindex(labels)

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = T
cut = F
structure = "FLNe"
mode = "ZERO"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0
opt_score = ["_S"]
save_data = T
version = "40d91"
__nodes__ = 40
__inj__ = 40
# Start main ----9
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

  A = get_structure_new(NET.csv_path, NET.struct_labels).to_numpy()
  

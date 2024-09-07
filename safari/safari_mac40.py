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
bias = 0.0
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

  bins = 20

  N = np.sum(NET.CC)
  x = np.zeros(np.ceil(N).astype(int))
  e = 0
  for i in np.arange(91):
     for j in np.arange(40):
        if i == j : continue
        x[e:(e+int(NET.CC[i,j]))] += NET.D[i, j]
        e += int(NET.CC[i,j])
  x = x[x > 0]

  x = np.array(x)


  # sns.histplot(
  #   x=x,
  #   stat="density",
  #   bins=bins
  # )

  # D, X, _ = plt.hist(x, bins=bins, density=T)

  # D = np.log(D)
  # X = (X[1:] + X[:-1]) / 2

  # # D = D[2:-2]
  # # X = X[2:-2]

  # import statsmodels.api as sm

  # X = sm.add_constant(X)
  # ml = sm.OLS(D, X).fit()

  # print(ml.summary())

  # from scipy.stats import expon

  # r = expon.fit(x)
  # print(1/r[1])

  # plt.yscale("log")
  # plt.show()

  # print(np.sum(NET.A[:40, :][:, :40] > 0))

  # A = get_structure_new(NET.csv_path, NET.struct_labels).to_numpy()
  # H = read_class(
  #   NET.pickle_path,
  #   "hanalysis"
  # )
  
  # nodeName = pd.read_table("../MAC3D/nodes.txt", header=None).to_numpy().ravel()
  # labels = NET.struct_labels

  # # print(labels[match(nodeName, labels)])
  # pos = match(nodeName, labels)
  # A = NET.A
  # A[A!=0] = -1/np.log10(A[A!=0])
  # print(A[:, 0])
  # B =A[pos, :][:, pos[:40]]
  # print(B[:, 0])

  # # print(H.hp)

  # from various.hit import EHMI, replicate_hierarchical_partition, flattenator

  # NULL = [[i] for i in range(NET.nodes)]

  # d = sorted(flattenator(H.hp))
  # np.random.shuffle(d)
  # print(replicate_hierarchical_partition(H.hp, d))
  

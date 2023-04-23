# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Allias booleans ----
T = True
F = False
#Import libraries ----
import pandas as pd
import numpy as np
import itertools
# Get workers ----
from crc.serial_distbase import worker_distbase
from crc.serial_overlap import worker_overlap
from crc.serial_scalefree import worker_scalefree
from crc.serial_swaps import worker_swaps
from crc.serial_ER import worker_ER
from crc.serial_HRG import worker_HRG

#Create THE_ARRAY ----
THE_ARRAY = pd.DataFrame()
## distbase ----
worker = ["distbase"]
distbases = ["EXPMLE"]
cut = [F]
topology = ["TARGET", "SOURCE", "MIX"]
bias = [0]
bins = [12]
mode = ["ALPHA", "BETA"]
list_of_lists = itertools.product(
  *[distbases, cut, topology, bias, bins, mode]
)
list_of_lists = np.array(list(list_of_lists))
array_distbase = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "distbase" : list_of_lists[:, 0].astype(str),
    "cut" : list_of_lists[:, 1],
    "topology" : list_of_lists[:, 2].astype(str),
    "bias" : list_of_lists[:, 3].astype(float),
    "bins" : list_of_lists[:, 4].astype(int),
    "mode" : list_of_lists[:, 5]
  }
)
## scalefree -----------------
worker = ["scalefree"]
cut = [F]
number_of_nodes = [1000]
topology = ["SOURCE", "MIX"]
indices = ["jacp",  "bsim"]
kav = [4]
mut = [0.1, 0.3, 0.5]
muw = [0.3]
nmin = [10, 50]
nmax = [20, 100]
list_of_lists = itertools.product(
  *[cut, topology, indices, kav, mut, muw, number_of_nodes, nmin, nmax]
)
list_of_lists = np.array(list(list_of_lists))
array_scalefree = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "cut" : list_of_lists[:, 0],
    "topology" : list_of_lists[:, 1].astype(str),
    "index" : list_of_lists[:, 2].astype(str),
    "kav" : list_of_lists[:, 3].astype(float),
    "mut" : list_of_lists[:, 4].astype(float),
    "muw" : list_of_lists[:, 5].astype(float),
    "number_of_nodes" : list_of_lists[:, 6].astype(int),
    "nmin" : list_of_lists[:, 7].astype(int),
    "nmax" : list_of_lists[:, 8].astype(int)
  }
)
array_scalefree = array_scalefree.loc[
  (array_scalefree.nmax > array_scalefree.nmin)
]
## overlap -----------------
worker = ["overlap"]
cut = [F]
number_of_nodes = [1000, 5000]
topology = ["SOURCE", "MIX"]
indices = ["jacp", "bsim"]
kav = [10]
mut = [0.1, 0.3, 0.5]
muw = [0.3]
on = [0.1, 0.5]
om = [2, 5, 8]
nmin = [10, 50]
nmax = [20, 100]
list_of_lists = itertools.product(
  *[cut, topology, indices, kav, mut, muw, on, om, number_of_nodes, nmin, nmax]
)
list_of_lists = np.array(list(list_of_lists))
array_overlap = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "cut" : list_of_lists[:, 0],
    "topology" : list_of_lists[:, 1].astype(str),
    "index" : list_of_lists[:, 2].astype(str),
    "kav" : list_of_lists[:, 3].astype(float),
    "mut" : list_of_lists[:, 4].astype(float),
    "muw" : list_of_lists[:, 5].astype(float),
    "on" : list_of_lists[:, 6].astype(float),
    "om" : list_of_lists[:, 7].astype(int),
    "number_of_nodes" : list_of_lists[:, 8].astype(int),
    "nmin" : list_of_lists[:, 9].astype(int),
    "nmax" : list_of_lists[:, 10].astype(int)
  }
)
## Overlapping condition -----------------
array_overlap = array_overlap.loc[
  (array_overlap.nmax > array_overlap.nmin)
]
## swaps -----------------
worker = ["swaps"]
cut = [F]
topology = ["MIX"]
bias = [0]
mode = ["ALPHA", "BETA"]
list_of_lists = itertools.product(
  *[cut, topology, bias, mode]
)
list_of_lists = np.array(list(list_of_lists))
array_swaps = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "cut" : list_of_lists[:, 0],
    "topology" : list_of_lists[:, 1].astype(str),
    "bias" : list_of_lists[:, 2].astype(float),
    "mode" : list_of_lists[:, 3]
  }
)
## ER -----------------
worker = ["ER"]
topology = ["MIX"]
index = ["jacp", "bsim"]
list_of_lists = itertools.product(*[topology, index])
list_of_lists = np.array(list(list_of_lists))
array_ER = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "topology" : list_of_lists[:, 0].astype(str),
    "index" : list_of_lists[:, 1].astype(str)
  }
)
## HRG -----------------
worker = ["HRG"]
topology = ["MIX"]
index = ["jacp", "bsim"]
list_of_lists = itertools.product(*[topology, index])
list_of_lists = np.array(list(list_of_lists))
array_HRG = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "topology" : list_of_lists[:, 0].astype(str),
    "index" : list_of_lists[:, 1].astype(str)
  }
)

## Merge arrays -----------------
# THE_ARRAY = pd.concat([THE_ARRAY, array_distbase], ignore_index=True)
# THE_ARRAY = pd.concat([THE_ARRAY, array_scalefree], ignore_index=True)
# THE_ARRAY = pd.concat([THE_ARRAY, array_overlap], ignore_index=True)
# THE_ARRAY = pd.concat([THE_ARRAY, array_swaps], ignore_index=True)
THE_ARRAY = pd.concat([THE_ARRAY, array_ER], ignore_index=True)
THE_ARRAY = pd.concat([THE_ARRAY, array_HRG], ignore_index=True)


def NoGodsNoMaster(number_of_iterations, t):
  # Get array ----
  array = THE_ARRAY.iloc[t - 1]
  # Select worker ----
  if array.loc["worker"] == "distbase":
    number_of_inj = 57
    number_of_nodes = 57
    total_number_nodes = 106
    version = 220830
    nlog10 = T
    lookup = F
    prob = T
    run = T
    mapping = "trivial"
    index = "simple2"
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_distbase(
      number_of_iterations, number_of_inj, number_of_nodes,
      total_number_nodes, version, array.loc["distbase"],
      nlog10, lookup, prob, cut, run, array.loc["topology"],
      mapping, index, float(array.loc["bias"]), int(array.loc["bins"]),
      array.loc["mode"]
    )
  elif array.loc["worker"] == "scalefree":
    nlog10 = F
    lookup = F
    prob = F
    run = T
    maxk = 5
    beta = 3
    t1 = 2
    t2 = 1
    mapping = "trivial"
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_scalefree(
      number_of_iterations, int(array.loc["number_of_nodes"]), nlog10,
      lookup, prob, cut, run, array.loc["topology"],
      mapping, array.loc["index"],
      array.loc["kav"], maxk, array.loc["mut"], array.loc["muw"],
      beta, t1, t2, int(array.loc["nmin"]), int(array.loc["nmax"])
    )
  elif array.loc["worker"] == "overlap":
    nlog10 = F
    lookup = F
    prob = F
    run = T
    maxk = 50
    beta = 3
    t1 = 2
    t2 = 1
    mapping = "trivial"
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_overlap(
      number_of_iterations, int(array.loc["number_of_nodes"]), nlog10,
      lookup, prob, cut, run, array.loc["topology"],
      mapping, array.loc["index"],
      array.loc["kav"], maxk, array.loc["mut"], array.loc["muw"],
      beta, t1, t2, int(array.loc["nmin"]), int(array.loc["nmax"]),
      int(array.loc["number_of_nodes"] * array.loc["on"]), int(array.loc["om"])
    )
  elif array.loc["worker"] == "swaps":
    number_of_inj = 57
    number_of_nodes = 57
    version = 220830
    nlog10 = T
    lookup = F
    prob = T
    run = T
    mapping = "trivial"
    index = "simple2"
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_swaps(
      number_of_iterations, number_of_inj, number_of_nodes,
      version, nlog10, lookup, prob, cut,
      run, array.loc["topology"], mapping, index, array.loc["bias"],
      array.loc["mode"]
    )
  elif array.loc["worker"] == "ER":
    nlog10 = F
    lookup = F
    prob = F
    cut = F
    number_of_nodes = 128
    mapping = "trivial"
    bias = float(0)
    mode = "ALPHA"
    worker_ER(
      number_of_iterations, number_of_nodes, nlog10,
      lookup, prob, cut, array.loc["topology"],
      mapping, array.loc["index"], bias, mode
    )
  elif array.loc["worker"] == "HRG":
    nlog10 = F
    lookup = F
    prob = F
    cut = F
    number_of_nodes = 640
    mapping = "trivial"
    bias = float(0)
    mode = "ALPHA"
    worker_HRG(
      number_of_iterations, number_of_nodes, nlog10,
      lookup, prob, cut, array.loc["topology"],
      mapping, array.loc["index"], bias, mode
    )
  else:
    raise ValueError("Worker does not exists!!!")

if __name__ == "__main__":
  number_of_iterations = int(sys.argv[1])
  t = int(sys.argv[2])
  print(THE_ARRAY.shape)
  from collections import Counter
  print(Counter(THE_ARRAY.worker))
  print(THE_ARRAY.iloc[t - 1])
  NoGodsNoMaster(number_of_iterations, t)
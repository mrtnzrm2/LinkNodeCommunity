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
from crc.serial_ecog import worker_ECoG

## distbase ----
worker = ["distbase"]
distbases = ["EXPMLE"]
cut = [F]
topology = ["MIX"]
bias = [0]
bins = [12]
mode = ["ZERO"]
indices = ["D1_2_4"]
list_of_lists = itertools.product(
  *[distbases, cut, topology, indices, bias, bins, mode]
)
list_of_lists = np.array(list(list_of_lists))
array_distbase = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "distbase" : list_of_lists[:, 0].astype(str),
    "cut" : list_of_lists[:, 1],
    "topology" : list_of_lists[:, 2].astype(str),
    "index" : list_of_lists[:, 3].astype(str),
    "bias" : list_of_lists[:, 4].astype(float),
    "bins" : list_of_lists[:, 5].astype(int),
    "mode" : list_of_lists[:, 6].astype(str)
  }
)

## scalefree -----------------
worker = ["scalefree"]
cut = [F]
number_of_nodes = [100, 150]
topology = ["MIX"]
indices = ["D1_2_4"]
mode = ["ZERO", "ALPHA", "BETA"]
kav = [7]
mut = [0.1, 0.3]
muw = [0.01]
nmin = [5]
nmax = [25]
list_of_lists = itertools.product(
  *[cut, mode, topology, indices, kav, mut, muw, number_of_nodes, nmin, nmax]
)
list_of_lists = np.array(list(list_of_lists))
array_scalefree = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "cut" : list_of_lists[:, 0],
    "mode" : list_of_lists[:, 1].astype(str),
    "topology" : list_of_lists[:, 2].astype(str),
    "index" : list_of_lists[:, 3].astype(str),
    "kav" : list_of_lists[:, 4].astype(float),
    "mut" : list_of_lists[:, 5].astype(float),
    "muw" : list_of_lists[:, 6].astype(float),
    "number_of_nodes" : list_of_lists[:, 7].astype(int),
    "nmin" : list_of_lists[:, 8].astype(int),
    "nmax" : list_of_lists[:, 9].astype(int)
  }
)

## overlap -----------------
worker = ["overlap"]
cut = [F]
number_of_nodes = [100, 150]
topology = ["MIX"]
indices = ["D1_2_4"]
mode = ["ZERO", "ALPHA", "BETA"]
kav = [7]
mut = [0.1, 0.3]
muw = [0.01]
on = [0.1, 0.3]
om = [2, 3]
nmin = [5]
nmax = [25]
list_of_lists = itertools.product(
  *[cut, mode, topology, indices, kav, mut, muw, on, om, number_of_nodes, nmin, nmax]
)
list_of_lists = np.array(list(list_of_lists))
array_overlap = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "cut" : list_of_lists[:, 0],
    "mode" : list_of_lists[:, 1].astype(str),
    "topology" : list_of_lists[:, 2].astype(str),
    "index" : list_of_lists[:, 3].astype(str),
    "kav" : list_of_lists[:, 4].astype(float),
    "mut" : list_of_lists[:, 5].astype(float),
    "muw" : list_of_lists[:, 6].astype(float),
    "on" : list_of_lists[:, 7].astype(float),
    "om" : list_of_lists[:, 8].astype(int),
    "number_of_nodes" : list_of_lists[:, 9].astype(int),
    "nmin" : list_of_lists[:, 10].astype(int),
    "nmax" : list_of_lists[:, 11].astype(int)
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
mode = ["ZERO"]
indices = ["D1_2_4"]
list_of_lists = itertools.product(
  *[cut, topology, indices, bias, mode]
)
list_of_lists = np.array(list(list_of_lists))
array_swaps = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "cut" : list_of_lists[:, 0],
    "topology" : list_of_lists[:, 1].astype(str),
    "index" : list_of_lists[:, 2].astype(str),
    "bias" : list_of_lists[:, 3].astype(float),
    "mode" : list_of_lists[:, 4].astype(str)
  }
)

## ER -----------------
worker = ["ER"]
topology = ["MIX"]
index = ["D1_2_3"]
number_of_nodes = [100, 150]
Rho = [0.2, 0.6]
list_of_lists = itertools.product(*[topology, index, number_of_nodes, Rho])
list_of_lists = np.array(list(list_of_lists))
array_ER = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "topology" : list_of_lists[:, 0].astype(str),
    "index" : list_of_lists[:, 1].astype(str),
    "number_of_nodes" : list_of_lists[:, 2].astype(int),
    "Rho" : list_of_lists[:, 3].astype(float)
  }
)
## HRG -----------------
worker = ["HRG"]
topology = ["MIX"]
index = ["D1_2_3"]
list_of_lists = itertools.product(*[topology, index])
list_of_lists = np.array(list(list_of_lists))
array_HRG = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "topology" : list_of_lists[:, 0].astype(str),
    "index" : list_of_lists[:, 1].astype(str)
  }
)

## ECoG ----
worker = ["ECoG"]
nature = [
  "MK1PreTheta", "MK1PreBeta", "MK1PreHighBeta", "MK1PreGamma",
  "MK2PreTheta", "MK2PreBeta", "MK2PreHighBeta", "MK2PreGamma",
  "MK1PostTheta", "MK1PostBeta", "MK1PostHighBeta", "MK1PostGamma",
  "MK2PostTheta", "MK2PostBeta", "MK2PostHighBeta", "MK2PostGamma"
]
cut = [F]
topology = ["MIX"]
mode = ["ZERO"]
indices = ["D1_2_4"]
list_of_lists = itertools.product(
  *[nature, topology, indices, mode, cut]
)
list_of_lists = np.array(list(list_of_lists))
array_ECoG = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "nature" : list_of_lists[:, 0].astype(str),
    "topology" : list_of_lists[:, 1].astype(str),
    "index" : list_of_lists[:, 2].astype(str),
    "mode" : list_of_lists[:, 3].astype(str),
    "cut" : list_of_lists[:, 4]
  }
)

## Dict arrays -----------------

DARRAY = {
  "distbase" : array_distbase,
  "swaps" : array_swaps,
  "LFR" : array_scalefree,
  "LFRo" : array_overlap,
  "ER" : array_ER,
  "HRG" : array_HRG,
  "ECoG" : array_ECoG
}


def NoGodsNoMaster(number_of_iterations, network, t):
  # Get array ----
  array = DARRAY[network].iloc[t - 1]
  # Select worker ----
  if array.loc["worker"] == "distbase":
    number_of_inj = 57
    number_of_nodes = 57
    total_number_nodes = 106
    version = "57d106"
    nlog10 = T
    lookup = F
    prob = F
    run = T
    mapping = "trivial"
    index = array.loc["index"]
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
    maxk = 20
    beta = 3
    t1 = 2
    t2 = 1
    mapping = "trivial"
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_scalefree(
      number_of_iterations, int(array.loc["number_of_nodes"]), array.loc["mode"], nlog10,
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
    maxk = 20
    beta = 3
    t1 = 2
    t2 = 1
    mapping = "trivial"
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_overlap(
      number_of_iterations, int(array.loc["number_of_nodes"]), array.loc["mode"], nlog10,
      lookup, prob, cut, run, array.loc["topology"],
      mapping, array.loc["index"],
      array.loc["kav"], maxk, array.loc["mut"], array.loc["muw"],
      beta, t1, t2, int(array.loc["nmin"]), int(array.loc["nmax"]),
      int(array.loc["number_of_nodes"] * array.loc["on"]), int(array.loc["om"])
    )
  elif array.loc["worker"] == "swaps":
    number_of_inj = 57
    number_of_nodes = 57
    total_number_nodes = 106
    version = "57d106"
    nlog10 = T
    lookup = F
    prob = F
    run = T
    mapping = "trivial"
    index = array.loc["index"]
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_swaps(
      number_of_iterations, number_of_inj, number_of_nodes, total_number_nodes,
      version, nlog10, lookup, prob, cut,
      run, array.loc["topology"], mapping, index, array.loc["bias"],
      array.loc["mode"]
    )
  elif array.loc["worker"] == "ER":
    nlog10 = F
    lookup = F
    prob = F
    cut = F
    mapping = "trivial"
    bias = float(0)
    mode = "ZERO"
    worker_ER(
      number_of_iterations, int(array.loc["number_of_nodes"]), float(array.loc["Rho"]),
      nlog10, lookup, prob, cut, array.loc["topology"],
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
    mode = "ZERO"
    worker_HRG(
      number_of_iterations, number_of_nodes, nlog10,
      lookup, prob, cut, array.loc["topology"],
      mapping, array.loc["index"], bias, mode
    )
  elif array.loc["worker"] == "ECoG":
    nlog10 = T
    lookup = F
    prob = F
    run = T
    mapping = "trivial"
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_ECoG(
      array.loc["nature"], array.loc["mode"], array.loc["topology"],
      mapping, array.loc["index"], nlog10, lookup, cut
    )
  else:
    raise ValueError("Worker does not exists!!!")

if __name__ == "__main__":
  number_of_iterations = int(sys.argv[1])
  t = int(sys.argv[2])
  network = sys.argv[3]
  ###
  print(DARRAY[network].shape)
  from collections import Counter
  print(Counter(DARRAY[network].worker))
  # for t in np.arange(1, 4):
  print(DARRAY[network].iloc[t - 1])
  NoGodsNoMaster(number_of_iterations, network, t)
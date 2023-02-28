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

#Create THE_ARRAY ----
THE_ARRAY = pd.DataFrame()
## distbase ----
worker = ["distbase"]
distbases = ["DEN", "M"]
cut = [F]
topology = ["TARGET", "SOURCE", "MIX"]
list_of_lists = itertools.product(
  *[distbases, cut, topology]
)
list_of_lists = np.array(list(list_of_lists))
array_distbase = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "distbase" : list_of_lists[:, 0].astype(str),
    "cut" : list_of_lists[:, 1],
    "topology" : list_of_lists[:, 2].astype(str)
  }
)
## scalefree -----------------
worker = ["scalefree"]
cut = [F]
topology = ["TARGET", "SOURCE", "MIX"]
indices = ["jacw", "jacp", "cos", "bsim"]
kav = [7, 15]
mut = [0.1, 0.3, 0.5]
muw = [0.1, 0.5]
list_of_lists = itertools.product(
  *[cut, topology, indices, kav, mut, muw]
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
  }
)
## overlap -----------------
worker = ["overlap"]
cut = [F]
topology = ["TARGET", "SOURCE", "MIX"]
indices = ["jacw", "jacp", "cos", "bsim"]
kav = [7, 15]
mut = [0.1, 0.3, 0.5]
muw = [0.1, 0.5]
om = [2, 5]
list_of_lists = itertools.product(
  *[cut, topology, indices, kav, mut, muw, om]
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
    "om" : list_of_lists[:, 6].astype(int),
  }
)
## Overlapping condition -----------------
array_overlap = array_overlap.loc[
  ~((array_overlap.kav > 7) & (array_overlap.om > 2))
]
## swaps -----------------
worker = ["swaps"]
cut = [F]
topology = ["TARGET", "SOURCE", "MIX"]
list_of_lists = itertools.product(
  *[cut, topology]
)
list_of_lists = np.array(list(list_of_lists))
array_swaps = pd.DataFrame(
  {
    "worker" : worker * list_of_lists.shape[0],
    "cut" : list_of_lists[:, 0],
    "topology" : list_of_lists[:, 1].astype(str)
  }
)
## Merge arrays -----------------
THE_ARRAY = pd.concat([THE_ARRAY, array_distbase], ignore_index=True)
THE_ARRAY = pd.concat([THE_ARRAY, array_scalefree], ignore_index=True)
THE_ARRAY = pd.concat([THE_ARRAY, array_overlap], ignore_index=True)
THE_ARRAY = pd.concat([THE_ARRAY, array_swaps], ignore_index=True)

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
    mapping = "R2"
    index = "jacw"
    bias = 0.3
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_distbase(
      number_of_iterations, number_of_inj, number_of_nodes,
      total_number_nodes, version, array.loc["distbase"],
      nlog10, lookup, prob, cut, run, array.loc["topology"],
      mapping, index, bias
    )
  elif array.loc["worker"] == "scalefree":
    number_of_nodes = 128
    nlog10 = F
    lookup = F
    prob = F
    run = T
    maxk = 30
    beta = 2.5
    t1 = 2
    t2 = 1
    nmin = 2
    nmax = 10
    mapping = "trivial"
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_scalefree(
      number_of_iterations, number_of_nodes, nlog10,
      lookup, prob, cut, run, array.loc["topology"],
      mapping, array.loc["index"],
      array.loc["kav"], maxk, array.loc["mut"], array.loc["muw"],
      beta, t1, t2, nmin, nmax
    )
  elif array.loc["worker"] == "overlap":
    number_of_nodes = 128
    nlog10 = F
    lookup = F
    prob = F
    run = T
    maxk = 30
    beta = 2.5
    t1 = 2
    t2 = 1
    nmin = 2
    nmax = 10
    on = 10
    mapping = "trivial"
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_overlap(
      number_of_iterations, number_of_nodes, nlog10,
      lookup, prob, cut, run, array.loc["topology"],
      mapping, array.loc["index"],
      array.loc["kav"], maxk, array.loc["mut"], array.loc["muw"],
      beta, t1, t2, nmin, nmax, on , int(array.loc["om"])
    )
  elif array.loc["worker"] == "swaps":
    number_of_inj = 57
    number_of_nodes = 57
    version = 220830
    nlog10 = T
    lookup = F
    prob = T
    run = T
    mapping = "R2"
    index = "jacw"
    bias = 0.3
    if array.loc["cut"] == "True": cut = T
    else: cut = F
    worker_swaps(
      number_of_iterations, number_of_inj, number_of_nodes,
      version, nlog10, lookup, prob, cut,
      run, array.loc["topology"], mapping, index, bias
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
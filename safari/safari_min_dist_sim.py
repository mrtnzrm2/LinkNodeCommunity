# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Personal libs ---- 
from modules.simanalysis import Sim
from various.data_transformations import maps
from networks.structure import MAC
from various.network_tools import *

# def mjw(u, v, n):
#   if n > 0:
#     U = u.copy()
#     V = v.copy()
#     zero_u = U == 0
#     zero_v = V == 0
#     U[zero_u | zero_v] = np.nan
#     V[zero_u | zero_v] = np.nan
#     A = np.vstack([U, V])
#     diff = (np.abs(np.nanmin(A, axis=0))) - np.abs(np.nanmax(A, axis=0))
#     return np.nanmax(diff), np.nanmin(diff)
#   else: return np.nan, np.nan

def mjw(u, v, n):
  if n > 0:
    U = u.copy()
    V = v.copy()
    zero_u = U == 0
    zero_v = V == 0
    U[zero_u | zero_v] = np.nan
    V[zero_u | zero_v] = np.nan
    A = np.vstack([U, V])
    diff = np.nansum(np.abs(np.nanmin(A, axis=0)) - np.abs(np.nanmax(A, axis=0))) / n
    return diff
  else: return np.nan

def mjw2(u, v, n):
  if n > 0:
    U = u.copy()
    V = v.copy()
    zero_u = U != 0
    zero_v = V != 0
    U[zero_u & zero_v] = np.nan
    V[zero_u & zero_v] = np.nan
    A = np.vstack([U, V])
    diff = np.nansum(np.abs(np.nanmin(A, axis=0)) - np.abs(np.nanmax(A, axis=0))) / n
    return diff
  else: return np.nan

# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = T
cut = F
mode = "ALPHA"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "R2"
index  = "jacw"
bias = 0
opt_score = ["_maxmu", "_X"]
save_data = T
version = 220830
__nodes__ = 57
__inj__ = 57
# Start main ----
if __name__ == "__main__":
  # Load structure ----
  NET = MAC(
    linkage, mode,
    nlog10 = nlog10,
    lookup =lookup,
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
  # Transform data for analysis ----
  R, lookup, _ = maps[mapping](
    NET.A, nlog10, lookup, prob, b=bias
  )
  sim = Sim(NET.nodes, NET.A, R, NET.D, mode=mode)
  aik = sim.get_aik()
  n_aik = aik.shape[1]
  aki = sim.get_aki()
  n_aki = aki.shape[1]
  # aik & aki
  target_min, target_max = np.Inf, -np.Inf
  for i in np.arange(1, NET.nodes):
    for j in np.arange(i):
      a = mjw2(aki[i, :], aki[j, :], n_aki)
      if a < target_min: target_min = a
      if a > target_max: target_max = a
  
  print(target_max, target_min)
  source_min, source_max = np.Inf, -np.Inf
  for i in np.arange(1, NET.nodes):
    for j in np.arange(i):
      a = mjw2(aik[i, :], aik[j, :], n_aik)
      if a < source_min: source_min = a
      if a > source_max: source_max = a
  
  print(source_max, source_min)


  
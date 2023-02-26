# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Canon ----
from infomap import Infomap
import numpy as np
from os.path import join
# Personal libs ----
from networks.structure import MAC
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from modules.colregion import colregion
from various.network_tools import adj2df
# Boolean aliases ----
T = True
F = False

if __name__ == "__main__":
  inj = 57
  NET = MAC(
    "single", "ALPHA", T, F,
    version=220830, nature="original", model="",
    distance="MAP3D", inj=inj
  )
  L = np.sum(NET.A != 0)
  M = NET.A.shape[0]
  N = NET.A.shape[1]
  total_number_links = (M - 1) * N
  den = L / total_number_links
  print(den)

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
  feature = "MIX_JACW"
  inj = 57
  NET = MAC(
    "single", "ALPHA", T, F,
    version=220830, nature="original", model="",
    distance="MAP3D", inj=inj, feature=feature, cut=T
  )
  # Add Infomap folder ----
  NET.plot_path = join(NET.plot_path, "Infomap")
  H = Hierarchy(
    NET, NET.A, NET.D,
    57, "single", "ALPHA",
    nlog10=T, lookup=F,
    feature=feature, prob=T,
    cut=T,
    # sln=NET.sln
  )
  # Set labels to network ----
  L = colregion(NET)
  H.set_colregion(L)
  # Processing data ----
  H.BH = []
  H.Z = []
  R = H.R
  R[np.isnan(R)] = 0
  R[R != 0] = 1 / R[R != 0]
  R = R[:inj, :inj]
  dR = adj2df(R)
  dR = dR.loc[dR.weight != 0]
  # Create infomap instance ----
  im = Infomap(
    directed=T,
    num_trials=100,
    # prefer_modular_solution=3
  )
  for i in np.arange(dR.shape[0]):
    source = dR["source"].iloc[i]
    target = dR["target"].iloc[i]
    weight = dR["weight"].iloc[i]
    im.add_link(
      int(source), int(target), weight=weight
    )
  im.run()
  partition = np.array(
    list(im.get_modules().values())
  )
  number_of_modules = len(np.unique(partition))
  plot_h = Plot_H(NET, H)
  plot_h.flatmap_labels(
    number_of_modules, partition, on=T, EC=T
  )
  # Study Infomap ----
  # print(im.get_multilevel_modules())

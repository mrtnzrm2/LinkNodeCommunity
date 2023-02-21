# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
#Import libraries ----
import numpy as np
from modules.hierarmerge import Hierarchy
from plotting_modules.plotting_H import Plot_H
from plotting_modules.plotting_N import Plot_N
from networks.ellipticnet import ELLIPTICNET
from modules.WSBM import WSBM
from various.network_tools import column_normalize, save_class, read_class

nodes = 100
__version__ = 0
__model__ = "DENCOUNT"
__mode__ = "BETA"
linkage = "average"
nlog10 = True
save_datas =  False

if __name__ == "__main__":
  # Random seed ----
  np.random.seed(2022)
  # Create EDR network ----
  NET = ELLIPTICNET(
    nodes, __version__, __model__, __mode__
  )
  # Generate random points ----
  ran_pos = NET.throw_nodes_randomly()
  # Create distance matrix ----
  D = NET.distance_matrix(ran_pos, save=False)
  # Create network ----
  print("Create random graph")
  Gn = NET.random_const_net(D, save=False)
  G = column_normalize(Gn)
  # Compute Hierarchy ----
  print("Compute Hierarchy")
  # Save ----
  if save_datas:
    ## Hierarchy object!! ----
    H = Hierarchy(
      NET, G, G, D, nodes, linkage, __mode__
    )
    ## Compute features ----
    H.BH_features_cpp()
    ## Compute lq arbre de merde ----
    H.la_abre_a_merde_cpp(H.BH[0])
    save_class(
      H, NET.pickle_path,
      "hanalysis_{}".format(linkage)
    )
  else:
    H = read_class(
      NET.pickle_path,
      "hanalysis_{}".format(linkage)
    )
  # Plot H ----
  plot_h = Plot_H(NET, H)
  # plot_h.core_dendrogram(on=False)
  # ## Single linkage ----
  # plot_h.heatmap_dendro(on=False)
  # plot_h.lcmap_dendro([1107, 214], on=False)
  # ## Average linkage ----
  # plot_h.lcmap_average(3, 4, labels, on=True)
  # Plot N ----
  plot_n = Plot_N(NET, H)
  plot_n.A_vs_dis(G, on=False)
  plot_n.A_vs_dis(Gn, name="count", on=False)
  plot_n.histogram_weight(on=False)
  plot_n.histogram_dist(on=False)
  plot_n.plot_akis(D, on=False)
  print("End!")
  #@@ Todo:
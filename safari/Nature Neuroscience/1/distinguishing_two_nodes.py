import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def distinguishing_two_nodes_plot():

  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3, 4, 5])
  G.add_edge(1, 2)
  G.add_edge(3, 1)

  pos = {
    1 : np.array([0, 1]),
    2 : np.array([1, 1]),
    3 : np.array([0, -0.1]),
    4 : np.array([0.55, -0.1 + 0.05]),
    5 : np.array([1.1, -0.1 - 0.01])
  }

  node_size=600
  arrowsize = 15


  nx.draw_networkx_edges(
    G, pos=pos, node_size=node_size,
    width=1,
    arrowsize=arrowsize, connectionstyle="arc3, rad=-0.5", arrowstyle="-"
  )

  nx.draw_networkx_nodes(
    G, pos=pos, node_size=node_size, edgecolors=["#DE8344", "#DE8344"] + ["#7EAB55"] * 3,
    linewidths=1,
    node_color=["#FAEEE5", "#FAEEE5"] + ["#EDF3E7"] * 3
  )

  nx.draw_networkx_labels(
    G, pos=pos, labels={1:"A", 2:"B", 3:"C"}, font_size=20
  )

  plt.gca().text(
    0.5, 1.5, r"$H^{2}(A,B)$" + "         " + r"$\mathcal{N}(A,B)$", ha="center", va="center", fontsize=10
  )
  plt.gca().text(
    0.8, 0.45, r"$H^{2}(A,C)$" +  "         " + r"$\mathcal{N}(A,C)$", ha="center", va="center", fontsize=10
  )

  array_pos = np.array([list(pos[v]) for v in pos.keys()])
  xmin, ymin = np.min(array_pos, axis=0)
  xmax, ymax = np.max(array_pos, axis=0)

  return xmin, xmax, ymin, ymax
  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def link_neighbors_plot():

  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3, 4, 5, 6])
  G.add_edge(1, 2)
  G.add_edge(1, 3)
  G.add_edge(5, 4)
  G.add_edge(6, 4)

  pos = {
    1 : np.array([0, 0.5]),
    2 : np.array([0.25, 0.6]),
    3 : np.array([0.25, 0.4]),
    4 : np.array([0, 0.1]),
    5 : np.array([0.25, 0.2]),
    6 : np.array([0.25, 0])
  }

  node_size = 200
  arrowsize = 15

  nx.draw_networkx_edges(
    G, pos=pos, node_size=node_size,
    width=2,
    arrowsize=arrowsize,
    arrowstyle="->"
  )

  nx.draw_networkx_nodes(
    G, pos=pos, node_size=node_size, edgecolors="black",
    linewidths=2,
    node_color="gray"
  )

  array_pos = np.array([list(pos[v]) for v in pos.keys()])
  xmin, ymin = np.min(array_pos, axis=0)
  xmax, ymax = np.max(array_pos, axis=0)

  return xmin, xmax, ymin, ymax
  
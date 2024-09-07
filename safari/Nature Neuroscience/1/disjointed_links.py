import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def disjointed_links_plot():

  G = nx.DiGraph()
  G.add_nodes_from(np.arange(1,15))

  G.add_edge(2, 1)
  G.add_edge(1, 3)
  G.add_edge(4, 5)
  G.add_edge(6, 4)
  G.add_edge(7, 8)
  G.add_edge(9, 10)
  G.add_edge(11, 12)
  G.add_edge(14, 13)


  G2 = nx.DiGraph()
  G2.add_nodes_from([15, 16])
  G2.add_edge(15, 16)
  G2.add_edge(16, 15)


  pos = {
    1 : np.array([0, 1]),
    2 : np.array([0.25, 1.1]),
    3 : np.array([0.25, 0.9]),
    4 : np.array([0.5, 1]),
    5 : np.array([0.75, 1.1]),
    6 : np.array([0.75, 0.9]),
    7 : np.array([0, 0.7]),
    8 : np.array([0.25, 0.7]),
    9 : np.array([0, 0.5]),
    10 : np.array([0.25, 0.5]),
    11 : np.array([0.5, 0.7]),
    12 : np.array([0.75, 0.7]),
    13 : np.array([0.5, 0.5]),
    14 : np.array([0.75, 0.5]),
  }

  pos2 = {
    15 : np.array([0.25-0.1, 0.3]),
    16 : np.array([0.5+0.1, 0.3])
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

  nx.draw_networkx_edges(
    G2, pos=pos2, node_size=node_size,
    width=2,
    arrowsize=15,
    connectionstyle="arc3,rad=-0.3",
    arrowstyle="->"
  )
 
  nx.draw_networkx_nodes(
    G2, pos=pos2, node_size=node_size, edgecolors="black",
    linewidths=2,
    node_color="gray"
  )

  array_pos = np.array([list(pos[v]) for v in pos.keys()])
  array_pos2 = np.array([list(pos2[v]) for v in pos2.keys()])

  xmin1, ymin1 = np.min(array_pos, axis=0)
  xmax1, ymax1 = np.max(array_pos, axis=0)
  xmin2, ymin2 = np.min(array_pos2, axis=0)
  xmax2, ymax2 = np.max(array_pos, axis=0)

  if xmin1 < xmin2: xmin = xmin1
  else: xmin = xmin2
  if xmax1 > xmax2: xmax = xmax1
  else: xmax = xmax2
  if ymin1 < ymin2: ymin = ymin1
  else: ymin = ymin2
  if ymax1 > ymax2: ymax = ymax1
  else: ymax = ymax2

  return xmin, xmax, ymin, ymax
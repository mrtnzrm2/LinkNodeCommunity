import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.patches import ArrowStyle

def directed_neighborhoods_plot():

  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3, 4, 5])

  G.add_edge(3,1)
  G.add_edge(4,1)
  G.add_edge(5,1)
  G.add_edge(3,2)
  G.add_edge(4,2)
  G.add_edge(5,2)

  pos = {
    1 : np.array([-0.25, 0]),
    2 : np.array([0.25, 0]),
    3 : np.array([-0.6, 0.8]),
    4 : np.array([0, 0.85]),
    5 : np.array([0.6, 0.8])
  }

  G2 = nx.DiGraph()
  G2.add_nodes_from([1, 2, 3, 4, 5])

  G2.add_edge(1, 3)
  G2.add_edge(1, 4)
  G2.add_edge(1, 5)
  G2.add_edge(2, 3)
  G2.add_edge(2, 4)
  G2.add_edge(2, 5)

  x2, y2 = 0, -0.6

  pos2 = {
    1 : np.array([-0.25, -0.85 + y2]),
    2 : np.array([0.25, -0.85 + y2]),
    3 : np.array([-0.6, -0.05 + y2]),
    4 : np.array([0, -0.00 + y2]),
    5 : np.array([0.6, -0.05 + y2])
  }

  node_size = 200
  arrowsize = 15


  node_colors = [
    "#F8D849",
    "#F8D849",
    "#B02318",
    "#B02318",
    "#B02318"
  ]

  node_colors2 = [
    "#F8D849",
    "#F8D849",
    "#4E70BE",
    "#4E70BE",
    "#4E70BE"
  ]

 

  nx.draw_networkx_edges(
    G, pos=pos, node_size=node_size,
    width=2,
    arrowsize=arrowsize,
    arrowstyle="->"
  )
  nx.draw_networkx_nodes(
    G, pos=pos, node_size=node_size, edgecolors="black",
    linewidths=2,
    node_color=node_colors
  )

  nx.draw_networkx_edges(
    G2, pos=pos2, node_size=node_size,
    width=2,
    arrowsize=arrowsize,
    arrowstyle="->"
  )
  
  nx.draw_networkx_nodes(
    G2, pos=pos2, node_size=node_size, edgecolors="black",
    linewidths=2,
    node_color=node_colors2
  )

  array_pos = np.array([list(pos[v]) for v in pos.keys()])
  array_pos2 = np.array([list(pos2[v]) for v in pos.keys()])

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
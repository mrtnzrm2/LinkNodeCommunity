import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

  # Plot networkx

node_size = 200
arrowsize = 15

def outsim(xcm, ycm, ax : plt.Axes):
  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3, 4, 5, 6])

  G.add_edge(1, 2)
  G.add_edge(1, 3)
  G.add_edge(4, 2)
  G.add_edge(5, 2)
  G.add_edge(6, 2)
  G.add_edge(4, 3)
  G.add_edge(5, 3)
  G.add_edge(6, 3)

  L = 1.5/3
  

  pos = {
    1 : np.array([xcm - L/2,ycm + 0]),
    2 : np.array([xcm + 0, ycm + L/2]),
    3 : np.array([xcm + 0, ycm - L/2]),
    4 : np.array([xcm + L/2, ycm + L]),
    5 : np.array([xcm + L/2, ycm + 0]),
    6 : np.array([xcm + L/2, ycm - L])
  }

  node_colors = [
    "#55a868",
    "#dd8452",
    "#dd8452",
    "#808080",
    "#808080",
    "#808080"
  ]

  nx.draw_networkx_edges(
    G, pos=pos, node_size=node_size,
    width=1,
    arrowsize=arrowsize,
    arrowstyle="->", ax=ax
  )
  nx.draw_networkx_nodes(
    G, pos=pos, node_size=node_size, edgecolors="black",
    linewidths=1,
    node_color=node_colors, ax=ax
  )

  pos_labels = {
    1 : np.array([xcm - L/2 - L/8,ycm + 0]),
    2 : np.array([xcm + 0 - L/8, ycm + L/2 + L/3.5]),
    3 : np.array([xcm + 0 - L/8, ycm - L/2 - L/3.5])
  }

  nx.draw_networkx_labels(
    G, pos=pos_labels, labels={1:"i", 2:"j", 3:"k"}, font_size=15, ax=ax
  )

  # plt.gcf().tight_layout()
  # plt.show()

def insim(xcm, ycm, ax : plt.Axes):
  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3, 4, 5, 6])

  G.add_edge(2, 1)
  G.add_edge(3, 1)
  G.add_edge(2, 4)
  G.add_edge(2, 5)
  G.add_edge(2, 6)
  G.add_edge(3, 4)
  G.add_edge(3, 5)
  G.add_edge(3, 6)

  L = 1.5/3
  

  pos = {
    1 : np.array([xcm - L/2,ycm + 0]),
    2 : np.array([xcm + 0, ycm + L/2]),
    3 : np.array([xcm + 0, ycm - L/2]),
    4 : np.array([xcm + L/2, ycm + L]),
    5 : np.array([xcm + L/2, ycm + 0]),
    6 : np.array([xcm + L/2, ycm - L])
  }

  node_colors = [
    "#4c72b0",
    "#dd8452",
    "#dd8452",
    "#808080",
    "#808080",
    "#808080"
  ]

  nx.draw_networkx_edges(
    G, pos=pos, node_size=node_size,
    width=1,
    arrowsize=arrowsize,
    arrowstyle="->", ax=ax
  )
  nx.draw_networkx_nodes(
    G, pos=pos, node_size=node_size, edgecolors="black",
    linewidths=1,
    node_color=node_colors, ax=ax
  )

  pos_labels = {
    1 : np.array([xcm - L/2 - L/8,ycm + 0]),
    2 : np.array([xcm + 0 + L/8, ycm + L/2 + L/3.5]),
    3 : np.array([xcm + 0 + L/8, ycm - L/2 - L/3.5])
  }

  nx.draw_networkx_labels(
    G, pos=pos_labels, labels={1:"i", 2:"j", 3:"k"}, font_size=15, ax=ax
  )

  # plt.gcf().tight_layout()
  # plt.show()

def sim_node_neighborhoods_plot(ax : plt.Axes):

  DELTA = 2/3

  # ax = plt.gca()

  xcm = 0
  ycm = 1 * DELTA

  outsim(xcm, ycm, ax)

  xcm = 0
  ycm = -1 * DELTA

  insim(xcm, ycm, ax)

  xmin, xmax = ax.get_xlim()
  ymin, ymax = ax.get_ylim()
  return xmin, xmax, ymin, ymax

# plt.show()
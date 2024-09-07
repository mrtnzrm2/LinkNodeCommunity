import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

  # Plot networkx

node_size = 200
arrowsize = 15


def outlink_neighbors(xcm, ycm, ax : plt.Axes):
  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3])

  G.add_edge(1, 2)
  G.add_edge(1, 3)

  L = 2/3
  

  pos = {
    1 : np.array([xcm - L/2,ycm + 0]),
    2 : np.array([xcm + L/2, ycm + L/2]),
    3 : np.array([xcm + L/2, ycm - L/2])
  }

  node_colors = [
    "#55a868",
    "#dd8452",
    "#dd8452"
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
    1 : np.array([xcm - L/2 - L/4,ycm + 0]),
    2 : np.array([xcm + L/2 + L/4, ycm + L/2]),
    3 : np.array([xcm + L/2 + L/4, ycm - L/2])
  }

  nx.draw_networkx_labels(
    G, pos=pos_labels, labels={1:"i", 2:"j", 3:"k"}, font_size=15, ax=ax
  )

  # plt.show()

def inlink_neighbors(xcm, ycm, ax : plt.Axes):

  L = 2/3
  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3])

  G.add_edge(2, 1)
  G.add_edge(3, 1)
  

  pos = {
    1 : np.array([xcm - L/2,ycm + 0]),
    2 : np.array([xcm + L/2, ycm + L/2]),
    3 : np.array([xcm + L/2, ycm - L/2])
  }

  node_colors = [
    "#4c72b0",
    "#dd8452",
    "#dd8452"
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
    1 : np.array([xcm - L/2 - L/4,ycm + 0]),
    2 : np.array([xcm + L/2 + L/4, ycm + L/2]),
    3 : np.array([xcm + L/2 + L/4, ycm - L/2])
  }

  nx.draw_networkx_labels(
    G, pos=pos_labels, labels={1:"i", 2:"j", 3:"k"}, font_size=15, ax=ax
  )

  # plt.show()


def zigzag(xcm, ycm, ax):

  L = 1/2
  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3])

  G.add_edge(2, 1)
  G.add_edge(1, 3)
  

  pos = {
    1 : np.array([xcm - L/3,ycm + 0]),
    2 : np.array([xcm + L/3, ycm + L/2]),
    3 : np.array([xcm + L/3, ycm - L/2])
  }

  node_colors = [
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

  # plt.show()

def antiparallel(xcm, ycm, ax):

  L = 1/2

  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3, 4])

  G.add_edge(1, 2)
  G.add_edge(4, 3)
  

  pos = {
    1 : np.array([xcm - L/3,ycm + L/2]),
    2 : np.array([xcm + L/3, ycm + L/2]),
    3 : np.array([xcm - L/3, ycm - L/2]),
    4 : np.array([xcm + L/3, ycm - L/2])
  }

  node_colors = [
    "#808080",
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

  # plt.show()

def loop(xcm, ycm, ax):

  L = 1/2

  G = nx.DiGraph()
  G.add_nodes_from([1, 2])

  G.add_edge(1, 2)
  G.add_edge(2, 1)
  

  pos = {
    1 : np.array([xcm - L/3,ycm + 0]),
    2 : np.array([xcm + L/3, ycm + 0])
  }

  node_colors = [
    "#808080",
    "#808080"
  ]

  nx.draw_networkx_edges(
    G, pos=pos, node_size=node_size,
    width=1,
    arrowsize=arrowsize, connectionstyle="arc3,rad=-0.3",
    arrowstyle="->", ax=ax
  )
  nx.draw_networkx_nodes(
    G, pos=pos, node_size=node_size, edgecolors="black",
    linewidths=1,
    node_color=node_colors, ax=ax
  )

  # plt.show()


# ---------

def directed_link_neighborhoods_plot(ax : plt.Axes):

  XLEFT = -1
  DELTA = 1/3
  EPSILON = 0.05


  xcm = XLEFT + DELTA + DELTA/2 - EPSILON
  ycm = 1.5 * DELTA

  outlink_neighbors(xcm, ycm, ax)

  xcm = XLEFT + 4 * DELTA + DELTA/2 + EPSILON
  ycm = 1.5 * DELTA

  inlink_neighbors(xcm, ycm, ax)

  xcm = XLEFT + DELTA
  ycm = -1.5 * DELTA

  zigzag(xcm, ycm, ax)

  xcm = XLEFT + 3 * DELTA
  ycm = -1.5 * DELTA

  antiparallel(xcm, ycm, ax)

  xcm = XLEFT + 5 * DELTA
  ycm = -1.5 * DELTA

  loop(xcm, ycm, ax)

  xmin, xmax = ax.get_xlim()
  ymin, ymax = ax.get_ylim()
  return xmin, xmax, ymin, ymax



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

  # Plot networkx

node_size = 200
arrowsize = 15

def position_around_circle(n, r, d):
  delta = 2*np.pi / n
  theta = d * np.pi / 180. + np.arange(0, 2*np.pi, delta)
  pos = np.zeros((n, 2))
  pos[:, 0] = r * np.cos(theta)
  pos[:, 1] = r * np.sin(theta)
  return pos

def outlinks():
  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3])

  GinvA = nx.DiGraph()
  GinvA.add_nodes_from([1, 2, 3, 4, 5])

  GinvB = nx.DiGraph()
  GinvB.add_nodes_from([1, 2, 3, 4])

  GH = nx.DiGraph()
  GH.add_nodes_from([1, 2, 3])

  # Add edges

  G.add_edge(1, 2)
  G.add_edge(1, 3)

  GinvA.add_edge(1, 2)
  GinvA.add_edge(3, 2)
  GinvA.add_edge(4, 2)
  GinvA.add_edge(5, 2)
  # GinvA.add_edge(6, 2)

  GinvB.add_edge(1, 3)
  GinvB.add_edge(2, 3)
  GinvB.add_edge(4, 3)

  GH.add_edge(1, 2)
  GH.add_edge(1, 3)

  # Define positions

  posG = {
    1 : np.array([0, 0]),
    2 : np.array([0.25/2, 0.4]),
    3 : np.array([0.25/2,-0.4])
  }

  posGinvA = {
    1 : np.array([0.25/2, 0.4]),
    2 : np.array([0.25/2, 0.4]),
    3 : np.array([0.25/2, 0.4]),
    4 : np.array([0.25/2, 0.4]),
    5 : np.array([0.25/2, 0.4]),
    # 6 : np.array([0.25, 0.4])
  }

  pos = position_around_circle(len(posGinvA)-1, 0.05, 45)
  for i, key in enumerate([1, 3, 4, 5]):
    posGinvA[key] = posGinvA[key] + pos[i]
    
  posGinvB = {
    1 : np.array([0.25/2, -0.4]),
    2 : np.array([0.25/2, -0.4]),
    3 : np.array([0.25/2, -0.4]),
    4 : np.array([0.25/2, -0.4]),
  }

  pos = position_around_circle(len(posGinvB)-1, 0.05, 45)
  for i, key in enumerate([1, 2, 4]):
    posGinvB[key] = posGinvB[key] + pos[i]

  posGH = {
    1 : np.array([0.5/2, 0]),
    2 : np.array([0.25/2, 0.4]),
    3 : np.array([0.25/2,-0.4])
  }

  # G
  nx.draw_networkx_edges(
    G, pos=posG, node_size=node_size,
    width=1,
    arrowsize=arrowsize, edge_color="#B02318", arrowstyle="->"
  )
  nx.draw_networkx_edge_labels(
    G, pos=posG, edge_labels={
      (1, 2) : r"$e_{CA}$",
      (1, 3) : r"$e_{CB}$"
    }, font_size=10, label_pos=0.6
  )
  nx.draw_networkx_nodes(
    G, pos=posG, node_size=node_size, edgecolors="black",
    linewidths=1, node_color="white"
  )
  nx.draw_networkx_labels(
    G, pos=posG, labels={1:"C", 2:"A", 3:"B"}, font_size=10
  )

  # GinvA
  
  nx.draw_networkx_nodes(
    GinvA, pos=posGinvA, node_size=0,
    linewidths=1, node_color="none"
  )

  nx.draw_networkx_edges(
    GinvA, pos=posGinvA, node_size=0,
    width=1,
    arrowsize=arrowsize, edge_color="#4E70BE", arrowstyle="->"
  )

  # GinvB
  
  nx.draw_networkx_nodes(
    GinvB, pos=posGinvB, node_size=0,
    linewidths=1, node_color="none"
  )

  nx.draw_networkx_edges(
    GinvB, pos=posGinvB, node_size=0,
    width=1,
    arrowsize=arrowsize, edge_color="#4E70BE", arrowstyle="->"
  )

  # GH

  nx.draw_networkx_edges(
    GH, pos=posGH, node_size=node_size+1200,
    width=1,
    arrowsize=arrowsize, edge_color="k", arrowstyle="-"
  )
  nx.draw_networkx_nodes(
    GH, pos=posGH, node_size=0,
    linewidths=1, node_color="none"
  )
  nx.draw_networkx_labels(
    GH, posGH, labels={1:r"$H^{2}_{-}(A,B)$"}, font_size=9
  )


  xmin, xmax, ymin, ymax = np.Inf, -np.Inf, np.Inf, -np.Inf

  for vec in posG.values():
    if vec[0] < xmin: xmin = vec[0]
    if vec[0] > xmax: xmax = vec[0]
    if vec[1] < ymin: ymin = vec[1]
    if vec[1] > ymax: ymax = vec[1]

  for vec in posGinvA.values():
    if vec[0] < xmin: xmin = vec[0]
    if vec[0] > xmax: xmax = vec[0]
    if vec[1] < ymin: ymin = vec[1]
    if vec[1] > ymax: ymax = vec[1]

  for vec in posGinvB.values():
    if vec[0] < xmin: xmin = vec[0]
    if vec[0] > xmax: xmax = vec[0]
    if vec[1] < ymin: ymin = vec[1]
    if vec[1] > ymax: ymax = vec[1]

  for vec in posGH.values():
    if vec[0] < xmin: xmin = vec[0]
    if vec[0] > xmax: xmax = vec[0]
    if vec[1] < ymin: ymin = vec[1]
    if vec[1] > ymax: ymax = vec[1]
  return xmin, xmax, ymin, ymax

def inlinks():
  G = nx.DiGraph()
  G.add_nodes_from([1, 2, 3])

  GinvA = nx.DiGraph()
  GinvA.add_nodes_from([1, 2, 3, 4, 5])

  GinvB = nx.DiGraph()
  GinvB.add_nodes_from([1, 2, 3, 4])

  GH = nx.DiGraph()
  GH.add_nodes_from([1, 2, 3])

  # Add edges

  G.add_edge(2, 1)
  G.add_edge(3, 1)

  GinvA.add_edge(2, 1)
  GinvA.add_edge(2, 3)
  GinvA.add_edge(2, 4)
  GinvA.add_edge(2, 5)
  # GinvA.add_edge(2, 6)

  GinvB.add_edge(3, 1)
  GinvB.add_edge(3, 2)
  GinvB.add_edge(3, 4)

  GH.add_edge(1, 2)
  GH.add_edge(1, 3)

  # Define positions

  shift = 0

  posG = {
    1 : np.array([shift + 0, 0]),
    2 : np.array([shift + 0.25/2, 0.4]),
    3 : np.array([shift + 0.25/2,-0.4])
  }

  posGinvA = {
    1 : np.array([shift + 0.2/25, 0.4]),
    2 : np.array([shift + 0.25/2, 0.4]),
    3 : np.array([shift + 0.25/2, 0.4]),
    4 : np.array([shift + 0.25/2, 0.4]),
    5 : np.array([shift + 0.25/2, 0.4]),
    # 6 : np.array([shift + 0.25, 0.4])
  }

  pos = position_around_circle(len(posGinvA)-1, 0.05, 45)
  for i, key in enumerate([1, 3, 4, 5]):
    posGinvA[key] = posGinvA[key] + pos[i]

  posGinvB = {
    1 : np.array([shift + 0.25/2, -0.4]),
    2 : np.array([shift + 0.25/2, -0.4]),
    3 : np.array([shift + 0.25/2, -0.4]),
    4 : np.array([shift + 0.25/2, -0.4]),
  }

  pos = position_around_circle(len(posGinvB)-1, 0.05, 45)
  for i, key in enumerate([1, 2, 4]):
    posGinvB[key] = posGinvB[key] + pos[i]

  posGH = {
    1 : np.array([shift + 0.5/2, 0]),
    2 : np.array([shift + 0.25/2, 0.4]),
    3 : np.array([shift + 0.25/2,-0.4])
  }

  # G
  nx.draw_networkx_edges(
    G, pos=posG, node_size=node_size,
    width=1,
    arrowsize=arrowsize, edge_color="#4E70BE", arrowstyle="->"
  )
  nx.draw_networkx_edge_labels(
    G, pos=posG, edge_labels={
      (2, 1) : r"$e_{AC}$",
      (3, 1) : r"$e_{BC}$"
    }, font_size=10, label_pos=0.6
  )
  nx.draw_networkx_nodes(
    G, pos=posG, node_size=node_size, edgecolors="black",
    linewidths=1, node_color="white"
  )
  nx.draw_networkx_labels(
    G, pos=posG, labels={1:"C", 2:"A", 3:"B"}, font_size=10
  )
  

  # GinvA
  
  nx.draw_networkx_nodes(
    GinvA, pos=posGinvA, node_size=0,
    linewidths=1, node_color="none"
  )

  nx.draw_networkx_edges(
    GinvA, pos=posGinvA, node_size=0,
    width=1,
    arrowsize=arrowsize, edge_color="#B02318", arrowstyle="->"
  )

  # GinvB
  
  nx.draw_networkx_nodes(
    GinvB, pos=posGinvB, node_size=0,
    linewidths=1, node_color="none"
  )

  nx.draw_networkx_edges(
    GinvB, pos=posGinvB, node_size=0,
    width=1,
    arrowsize=arrowsize, edge_color="#B02318", arrowstyle="->"
  )

  # GH

  nx.draw_networkx_edges(
    GH, pos=posGH, node_size=node_size+1200,
    width=1,
    arrowsize=arrowsize, edge_color="k", arrowstyle="-"
  )
  nx.draw_networkx_nodes(
    GH, pos=posGH, node_size=0,
    linewidths=1, node_color="none"
  )
  nx.draw_networkx_labels(
    GH, posGH, labels={1:r"$H^{2}_{+}(A,B)$"}, font_size=9
  )

  xmin, xmax, ymin, ymax = np.Inf, -np.Inf, np.Inf, -np.Inf

  for vec in posG.values():
    if vec[0] < xmin: xmin = vec[0]
    if vec[0] > xmax: xmax = vec[0]
    if vec[1] < ymin: ymin = vec[1]
    if vec[1] > ymax: ymax = vec[1]

  for vec in posGinvA.values():
    if vec[0] < xmin: xmin = vec[0]
    if vec[0] > xmax: xmax = vec[0]
    if vec[1] < ymin: ymin = vec[1]
    if vec[1] > ymax: ymax = vec[1]

  for vec in posGinvB.values():
    if vec[0] < xmin: xmin = vec[0]
    if vec[0] > xmax: xmax = vec[0]
    if vec[1] < ymin: ymin = vec[1]
    if vec[1] > ymax: ymax = vec[1]

  for vec in posGH.values():
    if vec[0] < xmin: xmin = vec[0]
    if vec[0] > xmax: xmax = vec[0]
    if vec[1] < ymin: ymin = vec[1]
    if vec[1] > ymax: ymax = vec[1]
  return xmin, xmax, ymin, ymax

def link_similarity_plot(fig, gs, s1, s2):
  ax = fig.add_subplot(gs[2:3, 0:2])
  ax.text(
  0.0, 1, "c", transform=ax.transAxes,
  fontsize=20, va='top', fontfamily='sans-serif', weight="bold", ha="right"
)
  oxmin, oxmax, oymin, oymax = outlinks()
  plt.xlim(oxmin - 0.05, oxmax + 0.05)
  plt.ylim(-0.1 + oymin, oymax+0.1)

  ax = fig.add_subplot(gs[3:4, 0:2])
  ax.text(
  0.0, 1, "d", transform=ax.transAxes,
  fontsize=20, va='top', fontfamily='sans-serif', weight="bold", ha="right"
)
  ixmin, ixmax, iymin, iymax = inlinks()
  plt.xlim(ixmin - 0.05, ixmax + 0.05)
  plt.ylim(-0.1 + iymin, iymax+0.1)
  # plt.gca().set_aspect('equal')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

golden_angle = np.pi * (3 - np.sqrt(5))
golden_angle_degrees = 180 * (3 - np.sqrt(5))

def star(x, y, r, ang):
  G = nx.DiGraph()
  G.add_nodes_from(np.arange(1, 8))
  G.add_edge(1, 2)
  G.add_edge(3, 1)
  G.add_edge(1, 4)
  G.add_edge(5, 1)
  G.add_edge(1, 6)
  G.add_edge(7, 1)

  coords = np.zeros((7, 2))
  coords[:, 0] += x
  coords[:, 1] += y
  theta = ang * np.pi / 180

  for i in np.arange(2, 8):
    coords[i-1, 0] += r * np.cos(theta)
    coords[i-1, 1] += r * np.sin(theta)
    theta += golden_angle

  pos = {}
  for i in np.arange(1, 8):
    pos[i] = coords[i-1]

  node_size = 200
  arrowsize = 15

  edge_colors = [
    "#B02318", "#B02318", "#B02318", "#4E70BE", "#4E70BE", "#4E70BE"
  ]

 
  nx.draw_networkx_edges(
    G, pos=pos, node_size=0,
    width=1, min_target_margin=0,
    arrowsize=arrowsize, edge_color=edge_colors, arrowstyle="->"
  )

  nx.draw_networkx_nodes(
    G, pos=pos, node_size=[node_size] + [0] * 6, edgecolors=["k"] + ["none"] * 6,
    linewidths=1, node_color=["#F8D849"] + ["none"] * 6
  )

  xmin =np.min(coords[:, 0])
  xmax =np.max(coords[:, 0])
  ymin =np.min(coords[:, 1])
  ymax =np.max(coords[:, 1])

  return np.array([xmin, xmax]), np.array([ymin, ymax])

def sim_both_ways_plot():

  iang = 360 / 5

  s1 = star(0, 0.75/2, 0.25, 12)
  s2 = star(-0.5/2, 0, 0.25, 12 + iang)
  s3 = star(0.5/2, 0, 0.25, 12 + 2 * iang)

  sx = np.hstack([s1[0], s2[0], s3[0]])
  sy = np.hstack([s1[1], s2[1], s3[1]])

  xmin = np.min(sx)
  xmax = np.max(sx)
  ymin = np.min(sy)
  ymax = np.max(sy)

  return xmin, xmax, ymin, ymax
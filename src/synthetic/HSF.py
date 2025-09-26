"""
src/LinkNodeCommunity/HSF.py

Module: linknode
Author: Jorge S. Martinez Armas

Overview:
---------
Provides helpers to construct deterministic hierarchical scale-free (HSF)
graphs from a seed adjacency matrix. Core workflow builds labeled edgelists
across replicas, positions nodes with polar rotations, connects replica nodes
back to a central hub, and derives adjacency matrices and community memberships
for downstream analysis and visualization.

Primary Class (HSF):
--------------------
HSF(matrix_seed, pos_seed, central_node, Replicas, L,
    linear_separation_factor=1, exponential_separation_factor=2.5)
- matrix_seed (np.ndarray): Base adjacency matrix used as the seed motif.
- pos_seed (dict[str, np.ndarray]): Initial coordinates for seed nodes; updated
  in place as replicas are generated.
- central_node (str): Label of the hub node anchoring new replica edges.
- Replicas (int): Number of replicas spawned per iteration.
- L (int): Number of hierarchical levels to generate.
- linear_separation_factor (float): Scales per-level radial distances linearly.
- exponential_separation_factor (float): Exponent controlling radial growth
  per level.
  
Notes:
------
- fit() builds the hierarchical edgelist by cloning and labeling replicas and
  attaching them to the central node.
- fit_matrix() reindexes labels to integers, updates stored positions, and
  constructs a symmetric adjacency matrix for analysis.
- Helper methods expose hierarchy-aware membership vectors and 2D layout
  utilities for consistent plotting.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes

from src.LinkNodeCommunity.utils import adjacency_to_edgelist, match

def rotation_matrix_2d(angle_degrees : float) -> np.ndarray:
    """
    Returns a 2D rotation matrix for a given angle in degrees.

    Args:
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        numpy.ndarray: A 2x2 rotation matrix.
    """
    import math
    angle_radians = math.radians(angle_degrees)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a, cos_a]])
    return rotation_matrix

# Class to create the deterministic scale-free hierarchical networks ----
class HSF:
  def __init__(
      self, matrix_seed,
      pos_seed : dict, 
      central_node, Replicas : int, 
      L : int, 
      linear_separation_factor=1, 
      exponential_separation_factor=2.5) -> None:
    
    self.A = matrix_seed
    self.central = central_node
    self.R = Replicas
    self.pos = pos_seed
    self.L = L
    self.alpha = linear_separation_factor
    self.exp_factor = exponential_separation_factor

  def append_labels(self, values, label):
    """Append a label suffix to each element in a sequence."""

    values_series = pd.Series(values, copy=True)
    return values_series.astype(str) + f"_{label}"

  def get_edgelist(self):
    """
    Create an edgelist from the adjacency matrix and label the nodes
    with the required node labeling scheme.
    """

    df = adjacency_to_edgelist(self.A).copy()
    df = df.loc[df.source < df.target].copy()
    df = df.loc[df.weight != 0].copy()
    df["source"] = self.append_labels(df["source"], "0_0")
    df["target"] = self.append_labels(df["target"], "0_0")

    # Set the name of the central node to A_00
    df.loc[df["source"] == "0_0_0", "source"] = "A_0_0"
    df.loc[df["target"] == "0_0_0", "target"] = "A_0_0"

    self.edgelist = df.reset_index(drop=True)

  def pos_angle(self, nodes, pos : dict, ang, l, l2):
    """
    Update the positions of the nodes based on the angle and distance from the center.
    """
    R = rotation_matrix_2d(ang)
    v = R @ np.array([1, 0]) * l
    for i, node in enumerate(nodes):
      #TODO: need to remember the purpose of this if-else clause
      if l2 == 1:
        pos[node] = v + pos[node[:5]]       
      else:
        pos[node] = v + pos[node[:(-4)]]

  def fit(self):
    """
    Create the hierarchical scale-free network by iteratively adding replicas of the seed graph.
    """
    self.get_edgelist()
    edgelist = self.edgelist.copy()

    ang = np.linspace(0, 360, self.R, endpoint=False)
    
    for l in np.arange(1, self.L + 1):
      new_edges = pd.DataFrame()
      for ir, r in enumerate(np.arange(self.R)):
        ed = edgelist.copy()
        ed.source = self.append_labels(ed.source, f"{l}_{r}")
        ed.target = self.append_labels(ed.target, f"{l}_{r}")

        self.pos_angle(np.unique(list(ed.source) + list(ed.target)), self.pos, ang[ir], self.alpha*(l**self.exp_factor), l)
        new_edges = pd.concat([new_edges, ed], ignore_index=True)

      new_edges = pd.concat([new_edges, self.add_edges_to_center(new_edges)], ignore_index=True)
      edgelist = pd.concat([edgelist, new_edges], ignore_index=True)

    self.edgelist = edgelist

  def fit_matrix(self):
    """
    Create the adjacency matrix (self.A) from the edgelist and update node labels and positions.
    """
    # Get unique nodes and sort them
    nodes = np.unique(list(self.edgelist.source) + list(self.edgelist.target))
    nodes = np.sort(nodes)

    self.labels = nodes

    self.get_communities_levels(nodes)
    self.edgelist_letter = self.edgelist.copy()

    ## Change labels in seedpos
    self.pos = {i: self.pos[s] for i, s in enumerate(nodes)}

    self.edgelist.source = match(self.edgelist.source.to_numpy(), nodes)
    self.edgelist.target = match(self.edgelist.target.to_numpy(), nodes)
    self.nodes = len(nodes)
    self.labels = nodes
    self.A = np.zeros((self.nodes, self.nodes))
    self.A[self.edgelist.source, self.edgelist.target] = 1
    self.A = self.A + self.A.T
  
  # TODO: need to remember the purpose of this function
  def get_communities_levels(self, node_labels):
    self.node_communities = np.array([int(s.split("_")[-2]) for s in node_labels])
  
  def add_edges_to_center(self, ed : pd.DataFrame):
    """
    Add new edges to the central node from all newly added nodes in the current iteration.
    """

    added_nodes = np.unique(list(ed.source) + list(ed.target))
    added_nodes = [n for n in added_nodes if self.central[0] != n[0]]
    toCenter = pd.DataFrame(
      {
        "source" : added_nodes,
        "target" : [self.central] * len(added_nodes),
        "weight" : [1] * len(added_nodes)
      }
    )
    return toCenter
  
  def get_fine_grained_memberships(self, node_labels):
    """
    Get fine-grained memberships based on node labels.
    """
    initials = [np.array(s.split("_")) for s in node_labels]
    effective_initials = []

    for i in np.arange(len(node_labels)):
      s = initials[i]
      if s.shape[0] > 3:
        effective_initials.append("_".join(s[3:]) + f"_{str(s.shape[0])}")
      else:
        effective_initials.append("_".join(s[1:]) + f"_{str(s.shape[0])}")

    initials = effective_initials
    initials_mem = np.unique(initials)
    initials_mem = {s : i for i, s in enumerate(initials_mem)}

    node_mem = [initials_mem[i] for i in initials]
    return np.array(node_mem)

  def get_membership_iter(self, node_mem, R, iter : npt.ArrayLike, maxiter=1):
    """
    Recursively refine memberships to achieve desired granularity.
    """
    if iter < maxiter:
      ncom = np.max(node_mem) + 1
      Reff = ncom // (R + 1)
      artificial_mem = np.zeros_like(node_mem)
      for i in np.arange(ncom):
        artificial_mem[node_mem == i] = i % Reff
      
      return self.get_membership_iter(artificial_mem, R, iter + 1, maxiter)
    else:
      return node_mem

  def get_membership_level(self, node_labels, R, level=0):
    """
    Get memberships at a specified hierarchical level.
    """
    i = np.array([0])
    base_mem = self.get_fine_grained_memberships(node_labels)
    return self.get_membership_iter(base_mem, R, i, maxiter=level)

  def graph_network(
      self, 
      G : nx.Graph, 
      node_communities : npt.ArrayLike, 
      labels=None, 
      strokewidth=1, 
      alpha_line=1,
      font_size=10, 
      node_size=40, 
      linewidth=2, 
      wx=1, 
      wy=1,
      ax : Axes | None = None
  ):
    """
    Visualize the network with nodes colored by their community memberships.

    Parameters:
    -----------
    G (nx.Graph): The graph to visualize.
    node_communities (array-like): Community membership for each node.
    labels (dict or list, optional): Node labels to display. If None, no labels are shown.
    strokewidth (int): Width of the stroke around text labels.
    alpha_line (float): Transparency of the edges.
    font_size (int): Font size for node labels.
    node_size (int): Size of the nodes.
    linewidth (int): Width of the edges.
    wx (float): Horizontal padding around the plot.
    wy (float): Vertical padding around the plot.
    ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, uses current axes.
    """

    from matplotlib.colors import to_hex
    import matplotlib.patheffects as path_effects

    if ax is None:
      ax = plt.gca()

    unique_memberships = np.sort(np.unique(node_communities))
    dft_color = to_hex((0.5, 0.5, 0.5))

    if -1 in unique_memberships:
      cm = list(sns.color_palette("hls", len(unique_memberships)-1))
      cm = [dft_color] + cm
    else:
      cm = list(sns.color_palette("hls", len(unique_memberships)))

    cm = {str(u): to_hex(c) for u, c in zip(unique_memberships, cm)}

    nx.draw_networkx_nodes(
      G, pos=self.pos, node_color=[cm[str(u)] for u in node_communities],
      node_size=node_size, ax=ax
    )

    nx.draw_networkx_edges(G, pos=self.pos, width=linewidth, alpha=alpha_line, ax=ax)

    if labels is not None:
      if not isinstance(labels, dict):
        labels = {u: labels[i] for i, u in enumerate(G.nodes)}
      t = nx.draw_networkx_labels(
        G, pos=self.pos,
        labels=labels,
        font_size=font_size,
        font_color="white",
        ax=ax
      )
      for key in t.keys():
          t[key].set_path_effects(
          [
            path_effects.Stroke(linewidth=strokewidth, foreground='gray'),
            path_effects.Normal()
          ]
        )

    array_pos = np.array([list(self.pos[v]) for v in self.pos.keys()])
        
    ax.set_xlim(-wx + np.min(array_pos, axis=0)[0], np.max(array_pos, axis=0)[0] + wx)
    ax.set_ylim(-wy+ np.min(array_pos, axis=0)[1], np.max(array_pos, axis=0)[1] + wy)

    sns.despine(ax=ax, left=True, bottom=True)
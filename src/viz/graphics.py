
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd
import networkx as nx

def graph_network_covers(
      G : nx.DiGraph | nx.Graph,
      node_cover_partition : npt.ArrayLike,
      node_partition : npt.ArrayLike,
      single_node_cover_map : dict,
      single_nodes_cover_scores : dict,
      labels : npt.ArrayLike,
      ang=0, cmap_name="hls", figsize=(12,12), scale=1.5, arrowsize=20,
      color_order=None, spring=False, font_size=12, hide_labels=False, **kwargs
    ):
    """
    Plot the network with pie-chart nodes indicating link community covers.

    Parameters
    ----------
    G : nx.DiGraph | nx.Graph
        The input graph.
    node_cover_partition : npt.ArrayLike
        Hard assignments where unique; otherwise -1 for NOCs.
    node_partition : npt.ArrayLike
        Node partition from the main clustering step.
    single_node_cover_map : dict
        Maps single nodes to their cover nodes.
    single_nodes_cover_scores : dict
        Stores scores for single node covers.
    labels : npt.ArrayLike
        Node labels for visualization.
    ang : float, optional
        Angle for pie chart slices. Default is 0.
    cmap_name : str, optional
        Colormap name for node colors. Default is "hls".
    figsize : tuple, optional
        Figure size for the plot. Default is (12, 12).
    scale : float, optional
        Scale factor for node sizes. Default is 1.5.
    arrowsize : float, optional
        Size of arrowheads in the plot. Default is 20.
    color_order : np.ndarray | None, optional
        Custom color order for nodes. Default is None.
    spring : bool, optional
        Whether to use spring layout for the plot. Default is False.
    undirected : bool, optional
        Whether the graph is undirected. Default is False.
    font_size : int, optional
        Font size for node labels. Default is 12.
    hide_labels : bool, optional
        Whether to show labels on nodes. Default is False.
    **kwargs
        Additional keyword arguments for the plot.
    """
    from scipy.cluster import hierarchy
    import matplotlib.patheffects as path_effects

    N = G.number_of_nodes()

    unique_clusters_id = np.unique(node_cover_partition)
    has_noise = -1 in unique_clusters_id
    keff = len(unique_clusters_id) - (1 if has_noise else 0)

    # Generate color palette
    palette = sns.color_palette(cmap_name, keff)
    if has_noise:
        # First color is white for noise/outliers
        cmap_heatmap = np.vstack([[1., 1., 1.], palette])
    else:
        cmap_heatmap = np.vstack([[1., 1., 1.], palette])

    # Reorder colors if color_order is provided
    if isinstance(color_order, np.ndarray):
        cmap_heatmap[1:] = cmap_heatmap[1:][color_order]


    # Assign memberships to nodes ----
    if has_noise:
        nodes_memberships = {
            k: {"id": [0] * keff, "size": [0] * keff}
            for k in range(N)
        }
    else:
        nodes_memberships = {
            k: {"id": [0] * (keff + 1), "size": [0] * (keff + 1)}
            for k in range(N)
        }

    # Assign cluster memberships to nodes in a more aesthetic and clear way
    for idx, cluster_id in enumerate(node_cover_partition):
        if cluster_id != -1:
            nodes_memberships[idx]["id"][cluster_id + 1] = 1
            nodes_memberships[idx]["size"][cluster_id + 1] = 1

    # Assign cover memberships for single nodes
    for key, clusters in single_node_cover_map.items():
        node_idx = np.where(labels == key)[0][0]
        for cluster_id in clusters:
            if cluster_id == -1:
                nodes_memberships[node_idx]["id"][0] = 1
                nodes_memberships[node_idx]["size"][0] = 1
            else:
                nodes_memberships[node_idx]["id"][cluster_id + 1] = 1
                nodes_memberships[node_idx]["size"][cluster_id + 1] = single_nodes_cover_scores[key][cluster_id]

    # Ensure every node has at least one membership assigned
    for i in range(N):
        if not any(nodes_memberships[i]["id"]):
            nodes_memberships[i]["id"][0] = 1
            nodes_memberships[i]["size"][0] = 1

    # Transform edge weights for layout computation
    G_dist = G.copy()
    for _, _, d in G_dist.edges(data=True):
        w = d.get("weight", 1.0)
        # Avoid log(0) or negative/zero weights
        w = max(w, 1e-10)
        d["distance"] = -np.log(w)
        d["invlogweight"] = -1.0 / np.log(w)

    if spring:
        # Use -1/log(w) as the weight for spring layout
        pos = nx.spring_layout(
            G_dist, 
            weight="invlogweight", 
            iterations=5, 
            seed=212
        )
    else:
        # Use -log(w) as the distance for kamada_kawai_layout
        pos = nx.kamada_kawai_layout(
            G_dist, 
            weight="distance"
        )

    # Rotate positions by the specified angle (in degrees)
    theta = np.deg2rad(ang)
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    pos = {k: rotation_matrix @ pos[k] for k in pos}

    # Assign labels to nodes
    labs = dict(zip(G.nodes, labels))

    # Center positions around the origin
    center = np.mean(np.array(list(pos.values())), axis=0)
    pos = {k: pos[k] - center for k in pos}

    # Scale positions
    pos = {k: pos[k] * scale for k in pos}

#   pos[np.where(labels == "9/46d")[0][0]][0] -= 0.03
#   pos[np.where(labels == "9/46d")[0][0]][1] -= 0.03

#   pos[np.where(labels == "v1")[0][0]][0] += 0.03
#   pos[np.where(labels == "v1")[0][0]][1] -= 0.03

    if hasattr(kwargs, "shifted_positions"):
        for key in kwargs["shifted_positions"].keys():
            i = np.where(labels == key)[0][0]
            pos[i] = kwargs["shifted_positions"][key]

    plt.figure(figsize=figsize)
    # Draw edges with improved aesthetics
    if not getattr(kwargs, "hide_edges", False):
        nx.draw_networkx_edges(
            G, pos=pos,
            edge_color="#888888",
            alpha=0.25, width=2.5, arrowsize=arrowsize,
            connectionstyle="arc3,rad=-0.12",
            node_size=1800, arrowstyle="<|-"
        )

    # Draw node labels with enhanced visibility
    if not hide_labels:
        label_dict = kwargs.get("modified_labels", labs)
        label_color = "yellow" if "modified_labels" in kwargs else "white"
        outline_color = "black" if "modified_labels" in kwargs else "gray"
        t = nx.draw_networkx_labels(
            G, pos=pos, labels=label_dict,
            font_color=label_color, font_size=font_size, font_weight="bold"
        )
        for key in t.keys():
            t[key].set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground=outline_color),
                path_effects.Normal()
            ])

    # Draw pie-chart nodes with subtle shadow and border
    for node in G.nodes:
        if node_partition[node] == -1:
            wedgecolor = "#222222"
            wedgewidth = 4
        else:
            wedgecolor = "#bbbbbb"
            wedgewidth = 2
        sizes = [s for s in nodes_memberships[node]["size"] if s != 0]
        colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[node]["id"]) if id != 0]
        wedges, _ = plt.pie(
            sizes,
            center=pos[node],
            colors=colors,
            radius=0.045,
            wedgeprops={
                "linewidth": wedgewidth,
                "edgecolor": wedgecolor,
                "alpha": 0.97,
                "zorder": 10,
                "antialiased": True
            }
        )
        for w in wedges:
            w.set_alpha(0.97)
            w.set_zorder(11)
            w.set_path_effects([
                path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.15),
                path_effects.Normal()
            ])

    # Adjust plot limits and layout for better framing
    array_pos = np.array([list(pos[v]) for v in pos.keys()])
    pad = 0.08 * np.ptp(array_pos, axis=0).max()
    plt.xlim(np.min(array_pos[:, 0]) - pad, np.max(array_pos[:, 0]) + pad)
    plt.ylim(np.min(array_pos[:, 1]) - pad, np.max(array_pos[:, 1]) + pad)
    plt.axis("off")
    plt.gcf().tight_layout()
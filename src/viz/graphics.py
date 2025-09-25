
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy.typing as npt
import pandas as pd
import networkx as nx

def graph_network_covers(
      G : nx.DiGraph | nx.Graph,
      pos: dict,
      node_cover_partition : npt.ArrayLike,
      node_partition : npt.ArrayLike,
      single_node_cover_map : dict,
      single_nodes_cover_scores : dict,
      labels : npt.ArrayLike,
      ang=0, palette="hls", figsize=(12,12), scale=1.5, arrowsize=20, 
      color_order=None, font_size=12, stroke_linewidth=2, hide_labels=False,
      ax: Axes | None = None, **kwargs
    ):
    """
    Plot the network with pie-chart nodes indicating link community covers.

    Parameters
    ----------
    G : nx.DiGraph | nx.Graph
        The input graph.
    pos : dict
        Node positions for plotting.
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
    palette : str, optional
        Colormap name for node colors. Default is "hls".
    figsize : tuple, optional
        Figure size for the plot. Default is (12, 12).
    scale : float, optional
        Scale factor for node sizes. Default is 1.5.
    arrowsize : float, optional
        Size of arrowheads in the plot. Default is 20.
    color_order : np.ndarray | None, optional
        Custom color order for nodes. Default is None.
    undirected : bool, optional
        Whether the graph is undirected. Default is False.
    font_size : int, optional
        Font size for node labels. Default is 12.
    stroke_linewidth : int, optional
        Line width for text stroke effect. Default is 2.
    hide_labels : bool, optional
        Whether to show labels on nodes. Default is False.
    ax : matplotlib.axes.Axes | None, optional
        Axes to plot on. Default is None (creates new figure).
    **kwargs
        Additional keyword arguments for the plot.
    """
    import matplotlib.patheffects as path_effects

    N = G.number_of_nodes()

    unique_clusters_id = np.unique(node_cover_partition)
    has_noise = -1 in unique_clusters_id.astype(int)
    keff = len(unique_clusters_id)

    # Generate color palette
    cmap = sns.color_palette(palette, keff)
    if has_noise:
        # First color is white for noise/outliers
        cmap_heatmap = np.vstack([[1., 1., 1.], cmap])
    else:
        cmap_heatmap = np.vstack([[1., 1., 1.], cmap])

    # Reorder colors if color_order is provided
    if isinstance(color_order, np.ndarray):
        cmap_heatmap[1:] = cmap_heatmap[1:][color_order]


    # Assign memberships to nodes for pie chart representation
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
        labels_arr = np.atleast_1d(labels)
        node_idx = np.where(labels_arr == key)[0][0]
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

    # Rotate positions by the specified angle (in degrees)
    theta = np.deg2rad(ang)
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    pos = {k: rotation_matrix @ pos[k] for k in pos}

    # Center positions around the origin
    center = np.mean(np.array(list(pos.values())), axis=0)
    pos = {k: pos[k] - center for k in pos}

    # Scale positions
    pos = {k: pos[k] * scale for k in pos}

    if "label_offsets" in kwargs:
        for key in kwargs["label_offsets"].keys():
            i = np.where(labels == key)[0][0]
            pos[i] += kwargs["label_offsets"][key]

    plt.figure(figsize=figsize)

    if ax is not None:
        plt.sca(ax)
    else:
        ax = plt.gca()

    # Draw edges with improved aesthetics
    if not kwargs.get("hide_edges", False):
        nx.draw_networkx_edges(
            G, pos=pos,
            edge_color="#888888",
            alpha=0.25, width=2.5, arrowsize=arrowsize,
            connectionstyle="arc3,rad=-0.12",
            node_size=1800, arrowstyle="<|-",
            ax=ax
        )

    # Draw pie-chart nodes with subtle shadow and border, using ax if provided
    for node in G.nodes():
        if node_partition[int(node)] == -1:
            wedgecolor = "#222222"
            wedgewidth = 4
        else:
            wedgecolor = "#bbbbbb"
            wedgewidth = 2
        sizes = [s for s in nodes_memberships[int(node)]["size"] if s != 0]
        colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[int(node)]["id"]) if id != 0]
        wedges, _ = ax.pie(
            sizes,
            center=pos[int(node)],
            colors=colors,
            radius=0.045,
            wedgeprops={
                "linewidth": wedgewidth,
                "edgecolor": wedgecolor,
                "alpha": 0.97,
                "antialiased": True
            }
        )
        for w in wedges:
            w.set_alpha(0.97)
            w.set_path_effects([
                path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.15),
                path_effects.Normal()
            ])

    # Draw node labels with enhanced visibility
    if not hide_labels:
        if "modified_labels" in kwargs:
            if len(kwargs["modified_labels"]) != N:
                raise ValueError("Length of modified_labels must match number of nodes.")
            label_dict = {i: lbl for i, lbl in enumerate(kwargs["modified_labels"])}
        else:
            label_dict = {n: labels[int(n)] for n in G.nodes()}
        
        label_color = "yellow" if "modified_labels" in kwargs else "white"
        outline_color = "gray" if "modified_labels" in kwargs else "black"
        t = nx.draw_networkx_labels(
            G, pos=pos, labels=label_dict,
            font_color=label_color, font_size=font_size, font_weight="bold", ax=ax
        )
        for key in t.keys():
            t[key].set_path_effects([
                path_effects.Stroke(linewidth=stroke_linewidth, foreground=outline_color),
                path_effects.Normal()
            ])

    # Adjust plot limits and layout for better framing
    array_pos = np.array([list(pos[v]) for v in pos.keys()])
    pad = 0.08 * np.ptp(array_pos, axis=0).max()
    ax.set_xlim(np.min(array_pos[:, 0]) - pad, np.max(array_pos[:, 0]) + pad)
    ax.set_ylim(np.min(array_pos[:, 1]) - pad, np.max(array_pos[:, 1]) + pad)
    ax.axis("off")
    plt.gcf().tight_layout()


def linkcommunity_matrix_map(
    G: nx.DiGraph | nx.Graph,
    H: npt.NDArray,
    Z: npt.NDArray,
    K: int,
    R: int,
    labels: npt.ArrayLike,
    colors: npt.ArrayLike,
    palette="hls",
    remove_labels=False,
    linewidth=1.5,
    font_color=None,
    undirected=False,
    ax: Axes | None = None,
    **kwargs
):
    """
    Plot a heatmap of the adjacency matrix where each cell represents the link community membership,
    with rows and columns reordered according to the node dendrogram. Community boundaries are marked
    with lines, and node labels are colored.

    Parameters
    ----------
    G : nx.DiGraph | nx.Graph
        The input graph.
    H : np.ndarray
        Linkage matrix for links (edges).
    Z : np.ndarray
        Linkage matrix for nodes.
    K : int
        Number of link communities.
    R : int
        Number of node communities.
    labels : array-like
        Node labels.
    colors : array-like
        Colors for node labels.
    palette : str, optional
        Seaborn color palette for link communities. Default is "hls".
    remove_labels : bool, optional
        If True, hides axis labels. Default is False.
    linewidth : float, optional
        Width of community boundary lines. Default is 1.5.
    font_color : str or None, optional
        Color for axis tick labels. If None, uses colors argument.
    undirected : bool, optional
        If True, treats the graph as undirected. Default is False.
    ax : matplotlib.axes.Axes | None, optional
        The axes to plot on. If None, creates a new figure and axes. Default is None.
    **kwargs
        Additional keyword arguments (e.g., fontsize).
    """
    from src.LinkNodeCommunity.utils import (
        fast_cut_tree,
        linear_partition,
        linkcommunity_collapsed_partition,
        linkcommunity_linear_partition,
        edgelist_to_adjacency, edgelist_from_graph
    )
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.ticker as ticker

    sns.set_style("white", rc={"axes.grid": False})

    # Copy and assign link community memberships

    edgelist = edgelist_from_graph(G)

    if not undirected:
        edgelist["id"] = fast_cut_tree(H, n_clusters=K)
    else:
        edgelist["id"] = np.tile(fast_cut_tree(H, n_clusters=K), 2)

    edgelist["source_label"] = labels[edgelist["source"].astype(int)]
    edgelist["target_label"] = labels[edgelist["target"].astype(int)]

    linkcommunity_collapsed_partition(edgelist, undirected)
    linkcommunity_linear_partition(edgelist, offset=1)

    number_of_link_communities = np.unique(edgelist.id).shape[0]

    # Build adjacency matrix with link community IDs
    A = edgelist_to_adjacency(edgelist, weight="id").astype(float)
    N = A.shape[0]

    # Get node ordering from dendrogram
    den_order = np.array(dendrogram(Z, no_plot=True)["ivl"]).astype(int)
    memberships = fast_cut_tree(Z, R).ravel()
    memberships = linear_partition(memberships)[den_order]

    # Find community boundaries for lines
    C = [i + 1 for i in range(len(memberships) - 1) if memberships[i] != memberships[i + 1]]
    D = np.where(memberships == -1)[0] + 1
    C = sorted(set(C).union(D))

    # Reorder adjacency and labels
    A = A[den_order, :][:, den_order]
    A[A == 0] = np.nan
    A[A > 0] = A[A > 0] - 1

    labels_copy = labels.copy()[den_order]
    colors_copy = np.array(colors)[den_order]

    # Create figure and axes

    if ax is not None:
        plt.sca(ax)
    else:
        fig, ax = plt.subplots(figsize=(10, 10))

    if ax.figure is None:
        fig = plt.gcf()
    else:
        fig = ax.figure

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    # Prepare color palette for link communities
    if -1 in A:
        save_colors = sns.color_palette(palette, number_of_link_communities - 1)
        cmap = [[]] * number_of_link_communities
        cmap[0] = [178 / 255, 178 / 255, 178 / 255]
        cmap[1:] = save_colors
    else:
        cmap = sns.color_palette(palette, number_of_link_communities)

    # Plot heatmap
    if not remove_labels:
        sns.heatmap(
            A,
            cmap=cmap,
            xticklabels=labels_copy,
            yticklabels=labels_copy,
            linecolor="gray",
            linewidths=0.5,
            ax=ax,
            cbar_ax=cbar_ax,
            cbar_kws={"label": "Link Community Membership"},
        )

        # Improved tick label aesthetics
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0.5, N, 2)))
        ax.xaxis.set_ticklabels([l for i, l in enumerate(labels_copy) if i % 2 == 0], rotation=90)
        colors1 = [c for i, c in enumerate(colors_copy) if i % 2 == 0]
        [t.set_color(c) for c, t in zip(colors1, ax.xaxis.get_ticklabels())]

        ax2 = ax.twiny()
        ax2.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1.5, N, 2) / N))
        ax2.xaxis.set_ticklabels([l for i, l in enumerate(labels_copy) if i % 2 == 1], rotation=90)
        colors2 = [c for i, c in enumerate(colors_copy) if i % 2 == 1]
        [t.set_color(c) for c, t in zip(colors2, ax2.xaxis.get_ticklabels())]

        ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0.5, N, 2)))
        ax.yaxis.set_ticklabels([l for i, l in enumerate(labels_copy) if i % 2 == 0])
        [t.set_color(c) for c, t in zip(colors1, ax.yaxis.get_ticklabels())]

        ax3 = ax.twinx()
        ax3.set_aspect('auto')
        ax3.yaxis.set_inverted(True)
        ax3.yaxis.set_major_locator(ticker.FixedLocator(np.arange(1.5, N, 2) / N))
        ax3.yaxis.set_ticklabels([l for i, l in enumerate(labels_copy) if i % 2 == 1])
        [t.set_color(c) for c, t in zip(colors2, ax3.yaxis.get_ticklabels())]

        # Set font color if provided
        if font_color:
            [t.set_color(font_color) for t in ax.xaxis.get_ticklabels()]
            [t.set_color(font_color) for t in ax.yaxis.get_ticklabels()]

        # Set font size if provided
        fontsize = kwargs.get("fontsize", 14)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=fontsize)
        ax2.set_xticklabels(ax2.get_xmajorticklabels(), fontsize=fontsize)
        ax3.set_yticklabels(ax3.get_ymajorticklabels(), fontsize=fontsize)

        ax.set_ylabel("Source", fontdict={"fontsize": 20, "weight": "bold"})
        ax.set_xlabel("Target", fontdict={"fontsize": 20, "weight": "bold"})
    else:
        sns.heatmap(
            A,
            cmap=cmap,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
            square=True,
        )

    # Draw community boundary lines
    for c in C:
        ax.vlines(
            c, ymin=0, ymax=N,
            linewidth=linewidth,
            colors="black",
            zorder=10,
        )
        ax.hlines(
            c, xmin=0, xmax=N,
            linewidth=linewidth,
            colors="black",
            zorder=10,
        )
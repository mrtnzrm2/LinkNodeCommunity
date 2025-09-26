
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
      labels : npt.ArrayLike | None = None,
      ang=0,
      scale=1.5,
      palette="hls",
      pie_radius=0.045,
      pie_alpha=0.97,
      single_node_color="#888888",
      wedgewidth : float | None = None,
      edge_color="#888888",
      alpha_line=1,
      edge_linewidth=1,
      connectionstyle="arc3,rad=-0.12",
      arrowstyle="<|-",
      arrowsize=20, 
      draw_arrows: bool | None = None,
      font_size=12,
      stroke_linewidth=2,
      color_order=None,
      hide_labels=False,
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
   labels : npt.ArrayLike | None, optional
        Labels for nodes. If None, uses default integer labels. Default is None.
    ang : float, optional
        Angle for pie chart slices. Default is 0.
    scale : float, optional
        Scale factor for node sizes. Default is 1.5.
    palette : str, optional
        Colormap name for node colors. Default is "hls".
    pie_radius : float, optional
        Radius of the pie charts. Default is 0.045.
    pie_alpha : float, optional
        Transparency level of the pie charts. Default is 0.97.
    wedgewidth : float | None, optional
        Width of the wedges in the pie charts. Default is None.
    edge_color : str, optional
        Color of the edges in the plot. Default is "#000000".
    alpha_line : float, optional
        Transparency level of the edges. Default is 0.25.
    connectionstyle : str, optional
        Style for connections between nodes. Default is "arc3,rad=-0.12".
    arrowstyle : str, optional
        Style for arrows in the plot. Default is "<|-".
    arrowsize : float, optional
        Size of arrowheads in the plot. Default is 20.
    draw_arrows : bool | None, optional
        If None, arrowheads are drawn when the graph is directed; set True or False to override.
    font_size : int, optional
        Font size for node labels. Default is 12.
    stroke_linewidth : int, optional
        Line width for text stroke effect. Default is 2.
    color_order : np.ndarray | None, optional
        Custom color order for nodes. Default is None.
    hide_labels : bool, optional
        Whether to show labels on nodes. Default is False.
    ax : matplotlib.axes.Axes | None, optional
        Axes to plot on. Default is None (creates new figure).
    **kwargs
        Additional keyword arguments for the plot.
    """
    import matplotlib.patheffects as path_effects

    from src.LinkNodeCommunity.utils import hex_to_rgb

    if labels is None:
        labels = np.arange(G.number_of_nodes())
    else:
        assert len(labels) == G.number_of_nodes(), f"labels must have {G.number_of_nodes()} elements, got {len(labels)}"
        labels = np.asarray(labels)

    N = G.number_of_nodes()

    unique_clusters_id = np.unique(node_cover_partition)
    has_noise = -1 in unique_clusters_id.astype(int)
    number_non_single_communities = len(unique_clusters_id)

    # Generate color palette
    cmap = sns.color_palette(palette, number_non_single_communities)
    if has_noise:
        # First color is white for noise/outliers
        cmap_heatmap = np.vstack([hex_to_rgb(single_node_color), cmap])
    else:
        cmap_heatmap = np.vstack([hex_to_rgb(single_node_color), cmap])

    # Reorder colors if color_order is provided
    if isinstance(color_order, np.ndarray):
        cmap_heatmap[1:] = cmap_heatmap[1:][color_order]


    # Assign memberships to nodes for pie chart representation
    if has_noise:
        nodes_memberships = {
            k: {"id": [0] * number_non_single_communities, "size": [0] * number_non_single_communities}
            for k in range(N)
        }
    else:
        nodes_memberships = {
            k: {"id": [0] * (number_non_single_communities + 1), "size": [0] * (number_non_single_communities + 1)}
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

    if ax is None:
        ax = plt.gca()

    # Draw edges with improved aesthetics
    draw_arrows_flag = draw_arrows
    if draw_arrows_flag is None:
        draw_arrows_flag = kwargs.get("arrows")
    if draw_arrows_flag is None:
        draw_arrows_flag = G.is_directed()

    if not kwargs.get("hide_edges", False):
        edge_kwargs = dict(
            pos=pos,
            edge_color=edge_color,
            alpha=alpha_line,
            width=edge_linewidth,
            node_size=1800,
            arrows=draw_arrows_flag,
            ax=ax
        )
        if draw_arrows_flag:
            edge_kwargs.update({
                "arrowsize": arrowsize,
                "connectionstyle": connectionstyle,
                "arrowstyle": arrowstyle
            })
        nx.draw_networkx_edges(G, **edge_kwargs)

    # Draw pie-chart nodes with subtle shadow and border, using ax if provided
    for node in G.nodes():
        if node_partition[int(node)] == -1:
            wedgecolor = "#222222"
            wedgewidth = 4 if wedgewidth is None else wedgewidth
        else:
            wedgecolor = "#bbbbbb"
            wedgewidth = 2 if wedgewidth is None else wedgewidth
        sizes = [s for s in nodes_memberships[int(node)]["size"] if s != 0]
        colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[int(node)]["id"]) if id != 0]
        wedges, _ = ax.pie(
            sizes,
            center=pos[int(node)],
            colors=colors,
            radius=pie_radius,
            wedgeprops={
                "linewidth": wedgewidth,
                "edgecolor": wedgecolor,
                "alpha": pie_alpha,
                "antialiased": True
            }
        )
        for w in wedges:
            w.set_alpha(pie_alpha)
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
    _ = ax.set_xlim(np.min(array_pos[:, 0]) - pad, np.max(array_pos[:, 0]) + pad)
    _ = ax.set_ylim(np.min(array_pos[:, 1]) - pad, np.max(array_pos[:, 1]) + pad)
    _ = ax.axis("off")
    plt.gcf().tight_layout()


def linkcommunity_matrix_map(
    G: nx.DiGraph | nx.Graph,
    H: npt.NDArray,
    Z: npt.NDArray,
    K: int,
    R: int,
    palette="hls",
    labels: npt.ArrayLike | None = None,
    colors: npt.ArrayLike | None = None,
    remove_labels=False,
    linewidth=1.5,
    font_color=None,
    cbar_position=[1.01, 0.15, 0.02, 0.7],      # [left, bottom, width, height]
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
    palette : str, optional
        Seaborn color palette for link communities. Default is "hls".
    labels : array-like or None, optional
        Labels for nodes. If None, uses default integer labels. Default is None.
    colors : array-like or None, optional
        Colors for node labels. If None, defaults to black. Default is None.
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
    
    if labels is not None:
        assert len(labels) == G.number_of_nodes(), f"labels must have {G.number_of_nodes()} elements, got {len(labels)}"
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

    if labels is not None:
        labels_copy = labels.copy()[den_order]

    if colors is None:
        colors_copy = np.array(["#000000"] * G.number_of_nodes())
    else:
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

    cbar_ax = fig.add_axes(cbar_position)

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
            linecolor="gray",
            linewidths=0.5,
            cbar_ax=cbar_ax,
            cbar_kws={"label": "Link Community Membership"},
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


def link_statistics_graph(
    link_stats: pd.DataFrame,
    x="height",
    y="S",
    xlabel=None,
    ylabel=None,
    testxoffset=0.1,
    textyoffset=0.1,
    color="tab:blue",
    ax : Axes | None = None,
):
    own_axes = ax is None
    if own_axes:
        fig, ax = plt.subplots()

    sns.lineplot(data=link_stats, x=x, y=y, color=color, ax=ax)

    ax.set_xlabel(xlabel if xlabel else x)
    ax.set_ylabel(ylabel if ylabel else y)
    ax.set_title(f'{y} vs {x}')
    ax.minorticks_on()
    sns.despine(ax=ax)

    # Highlight max y value
    max_idx = link_stats[y].idxmax()
    max_x = link_stats.loc[max_idx, x]
    max_y = link_stats.loc[max_idx, y]
    ax.axvline(max_x, color='gray', linestyle='--', lw=1)
    ax.text(
        max_x + testxoffset, max_y - textyoffset,
        f'Max {y}\n{x}={max_x:.3f}',
        color='gray',
        ha='left', va='bottom',
        fontsize=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )

    if own_axes:
        ax.figure.tight_layout()
        plt.show()

def dendrogram(
    Z: npt.NDArray,
    R: int,
    labels: npt.ArrayLike | None = None,
    ylabel: str = "Height",
    palette: str = "hls",
    leaf_font_size: int = 20,
    remove_labels: bool = False,
    ax: Axes | None = None,
    **kwargs
):
    """
    Plot a hierarchical clustering dendrogram with colored branches for each cluster.

    Branches above the height corresponding to the required number of communities (R)
    are plotted in gray. Leaf labels are colored according to cluster membership.

    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix from hierarchical clustering.
    R : int
        Number of clusters to cut the dendrogram at.
    labels : array-like or None, optional
        Labels for the leaves (nodes). If None, uses default integer labels.
    ylabel : str, optional
        Label for the y-axis. Default is "Height".
    palette : str, optional
        Seaborn color palette for cluster colors. Default is "hls".
    leaf_font_size : int, optional
        Font size for leaf labels. Default is 20.
    remove_labels : bool, optional
        If True, hides leaf labels. Default is False.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, uses current axes.
    **kwargs
        Additional keyword arguments for scipy dendrogram.
    """
    from src.LinkNodeCommunity.utils import fast_cut_tree, linear_partition, collapsed_partition
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    from matplotlib.colors import to_hex

    sns.set_style("ticks")

    N = Z.shape[0] + 1

    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = np.arange(Z.shape[0] + 1).astype(str)
    else:
        assert len(labels) == N, f"labels must have {N} elements, got {len(labels)}"

    # Compute partition and assign cluster colors
    partition = fast_cut_tree(Z, n_clusters=R)
    new_partition = collapsed_partition(partition)
    new_partition = linear_partition(partition)
    unique_clusters_id = np.unique(new_partition)
    cmap = sns.color_palette(palette, len(unique_clusters_id))
    gray_col = "#808080"

    # Assign colors to leaves based on cluster membership
    D_leaf_colors = {}
    for i, _ in enumerate(labels):
        if new_partition[i] != -1:
            D_leaf_colors[i] = to_hex(cmap[new_partition[i]])
        else:
            D_leaf_colors[i] = gray_col

    # Assign colors to branches: gray if above threshold, else cluster color
    link_cols = {}
    for i, i12 in enumerate(Z[:, :2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x] for x in i12)
        link_cols[i + 1 + len(Z)] = c1 if c1 == c2 else gray_col

    dendro_kwargs = dict(
        Z=Z,
        color_threshold=Z[N - R, 2],
        link_color_func=lambda k: link_cols[k],
        leaf_rotation=90,
        ax=ax,
        **kwargs
    )
    if not remove_labels:
        dendro_kwargs["labels"] = labels
        dendro_kwargs["leaf_font_size"] = leaf_font_size
        scipy_dendrogram(**dendro_kwargs)
        ax.tick_params(axis="x", labelrotation=90, labelsize=leaf_font_size)
    else:
        dendro_kwargs["no_labels"] = True
        scipy_dendrogram(**dendro_kwargs)

    ax.set_ylabel(ylabel, fontsize=16, weight="bold")
    sns.despine(ax=ax, trim=True, offset=10)

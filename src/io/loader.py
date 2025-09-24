import numpy as np
import pandas as pd

general_colors = pd.DataFrame(
    {
      "REGION" : [
        "Occipital",
        "Temporal",
        "Parietal",
        "Frontal",
        "Prefrontal",
        "Cingulate"
      ],
      "COLOR" : [
        "#006141",
        "#ff7e00",
        "#800080",
        "#fec20c",
        "#ed1c24",
        "#2a52be"
      ]
    }
)

general_colors = dict(zip(general_colors["REGION"], general_colors["COLOR"]))


def load_macaque_fln_40d91(path=""):
    """Load the Macaque FLN 40D91 dataset.

    Returns
    -------
    edgelist : np.ndarray
        Array of shape (M, 3) with columns [source, target, weight].
    labels : np.ndarray
        Array of node labels.
    regions : np.ndarray
        Array of region names corresponding to each label.
    colors : np.ndarray
        Array of color codes corresponding to each region.
    """
    data = pd.read_csv(f"{path}/FLN40d91.csv", header=None)

    # Load labels and regions
    labels = pd.read_csv(f"{path}/labels40.csv", header=None).squeeze().to_numpy()
    regions = pd.read_csv(f"{path}/Table_areas_regions_09_2019.csv", header=0).to_numpy()
    regions = {regions[i, 0].lower(): regions[i, 1] for i in np.arange(regions.shape[0])}
    regions = np.array([regions[l] for l in labels])

    # colors
    colors = np.array([general_colors[r] for r in regions])

    # Convert the DataFrame to a numpy array if it's not already
    adj_matrix = data.values  # shape (91, 40)

    # Extract edge list: (source, target, weight) for all nonzero entries
    sources, targets = np.nonzero(adj_matrix)
    weights = adj_matrix[sources, targets]
    edgelist = np.column_stack((sources, targets, weights))
    edgelist = np.array(edgelist)

    return edgelist, labels, regions, colors
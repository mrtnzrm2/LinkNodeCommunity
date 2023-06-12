import pandas as pd
import numpy as np
from matplotlib.colors import to_hex
from os.path import join
# Personal libs ----
from various.network_tools import match

class colregion:
  def __init__(self, NET, labels_name="labels", **kwargs) -> None:
    # Define attributes ----
    self.nodes = NET.nodes
    self.auto = False
    # Define auto ----
    if NET.version == "ellipse" or NET.version == "scalefree":
      self.labels = np.arange(NET.nodes).astype(str)
      self.auto = True
    elif "labels" in kwargs.keys():
      self.auto = True
      if len(kwargs["labels"]) == NET.nodes:
        self.labels = kwargs["labels"].astype(str)
      else:
        raise ValueError("Number of labels different than number of nodes.")
    else:
      self.subject = NET.subject
      # Import labels ----
      labels_path = join(
        NET.labels_path, f"{labels_name}.csv"
      )
      self.labels = pd.read_csv(
        labels_path
      ).to_numpy().reshape(-1)
      self.labels = np.char.lower(self.labels.astype(str))
      # Get regions path ----
      self.regions_path = NET.regions_path

  def MAC_region_colors(self):
    maxc = 255
    colors = pd.DataFrame(
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
          to_hex((0 ,97/maxc, 65/maxc)),
          to_hex((1, 126/maxc, 0)),
          "#800080",
          "#ffd500",
          to_hex((237/maxc, 28/maxc, 36/maxc)),
          "#2a52be"
        ]
      }
    )
    return colors

  def get_regions(self):
    if self.auto:
      self.regions = pd.DataFrame(
        {
          "AREA" : self.labels,
          "REGION" : ["UNDEFINED"] * self.nodes,
          "COLOR" : [to_hex((0., 0., 0.))] * self.nodes
        }
      )
    else:
      self.regions = pd.read_csv(
        self.regions_path
      )
      self.regions.columns = [
        "AREA", "REGION"
      ]
      self.regions["AREA"] = [
        np.char.lower(x) for x in self.regions["AREA"] if isinstance(x, str)
      ]
      if self.subject == "MAC":
        # Format area labels from regions ----
        from various.label_format import MAC_areas_regions
        MAC_areas_regions(self.regions)
        # Set colors to region dataframe ----
        colors = self.MAC_region_colors()
        self.regions["COLOR"] = colors.loc[
          match(
            self.regions["REGION"],
            colors["REGION"]
          ),
          "COLOR"
        ].to_numpy()
      



    

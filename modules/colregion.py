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
    self.get_regions()

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
          "#fec20c",
          # "#ffd500",
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
      
class colECoG:
  def __init__(self, NET) -> None:
      self.subject = NET.subject
      self.version = NET.version
      self.labels = NET.struct_labels
      self.regions_path = NET.regions_path
      self.nodes = NET.nodes
      self.get_regions()
  
  def get_regions(self):
    ## COLOR ----
    # REGION=c('V1', 'V2', 'V4', 'TEO', 'TPt', 'DP',
    #                                     '7A', '7B', 'S1', '5', 'F1', 'F4',
    #                                     'F2', '8L', '8M'),
    #                            COLOR=c('#AF0000', '#EF0000', '#FF3000', '#FF7000',
    #                                    '#FFAF00', '#FFEF00', '#CFFF30', '#8FFF70',
    #                                    '#50FFAF', '#10FFEF', '#00CFFF', '#008FFF',
    #                                    '#0050FF', '#0010FF', '#0000CF'))

    color = {
      "V1" :'#AF0000',
      "V2" :'#EF0000',
      "V4" :'#FF3000', 
      "TEO" : '#FF7000',
      "TPt" : '#FFAF00',
      "DP" : '#FFEF00',
      "7A" : '#CFFF30',
      "7B" : '#8FFF70',
      "S1" : '#50FFAF',
      "5" : '#10FFEF',
      "F1" : '#00CFFF',
      "F4" : '#008FFF',
      "F2" : '#0050FF',
      "8L" : '#0010FF',
      "8M" : "#0000CF"
    }

    ##
    file = pd.read_table(f"{self.regions_path}", sep="\t", header=None, index_col=0)
    if self.version == "MK1":
      coli = 1
      colf = 2
    elif self.version == "MK2":
      coli = 3
      colf = 4
    else:
      raise ValueError("Version unknown")
    self.regions = pd.DataFrame(
        {
          "AREA" : self.labels,
          "REGION" : ["UNDEFINED"] * self.nodes,
          "COLOR" : [to_hex((0., 0., 0.))] * self.nodes
        }
    )
    regions = file.index.to_numpy()
    regions = [re.split(" ")[0] for re in regions]
    for re, u, v in zip(regions, file[coli], file[colf]):
      self.regions["REGION"].loc[np.isin(self.regions.AREA, np.arange(u, v + 1))] = re
    for key in color.keys():
      self.regions["COLOR"].loc[self.regions.REGION == key] = color[key]
    self.regions["AREA_REGION"] = [f"{a}_{r}" for a, r in zip(self.regions.AREA, self.regions.REGION)]
    


    

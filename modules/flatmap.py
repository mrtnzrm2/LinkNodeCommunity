import numpy as np
import json
import matplotlib.pyplot as plt
from os.path import join
from pathlib import Path
import seaborn as sns

class FLATMAP:
  def __init__(self, NET, regions, **kwargs) -> None:
    # Get NET parameters ----
    self.nodes = NET.nodes
    self.version = NET.version
    if "EC" in kwargs.keys():
      if kwargs["EC"]:
        self.struct_labels = NET.struct_labels[:NET.nodes]
      else:
        self.struct_labels = NET.struct_labels
    else:
      self.struct_labels = NET.struct_labels
    self.struct_labels = np.char.lower(self.struct_labels.astype(str))    # Pay attention: could vary!!!
    self.overlap = NET.overlap
    self.overlap = np.char.lower(self.overlap.astype(str))     # Pay attention: could vary!!!
    self.plot_path = join(NET.plot_path, "flatmap")
    # Set attributes ----
    self.regions = regions

  def set_para(self, K, R, labels):
    self.K = K
    self.R = R
    self.labels = labels
    Path(
      self.plot_path, "K_{}".format(K)
    ).mkdir(exist_ok=True, parents=True)

  def format_areas(self, areas):
      for i, a in enumerate(areas):
          if ("." in a and a != "pro.st."):
              areas[i] = a.replace(".", "/")
      return areas

  def get_area_index(self, area, labels):
      labels = np.asarray(labels)
      index = np.where(labels == area)[0]
      if (len(index) > 0):
          return index[0]
      return -1

  def nasty_work_220617(self, L):
    from pandas import concat, DataFrame
    from matplotlib.colors import to_hex
    # Retrive uf and lf labels ----
    self.struct_labels = L.labels[:self.nodes]
    v4lf = np.where(L.labels[:self.nodes] == "v4lf")[0]
    # v4uf = np.where(L.labels[:self.nodes] == "v4uf")[0]
    id_v4lf = self.labels[v4lf]
    # id_v4uf = self.labels[v4uf]
    # Takout v4uf and v4lf ----
    ## struct_labels ----
    self.struct_labels = [
      n for n in self.struct_labels if n != "v4uf" and n != "v4lf"
    ]
    ## labels ----
    self.labels = [
      n for i, n in enumerate(self.labels) if  i != v4lf
    ]
    ## regions ----
    self.regions = self.regions.loc[
      (self.regions["AREA"] != "v4uf") &
      (self.regions["AREA"] != "v4lf")
    ]
    # Add new areas ----
    ## structu_labels
    self.struct_labels = np.hstack(
      (
        self.struct_labels,
        np.array(
          [
            "v4pcuf",
            "v4fpuf",
            "v4pclf",
            "v4fplf"
          ]
        )
      )
    )
    ## labels ----
    self.labels = np.hstack(
      (
        self.labels,
        np.array([
          id_v4lf, id_v4lf,
          id_v4lf, id_v4lf
        ]).reshape(-1)
      )
    )
    ## regions ----
    self.regions = concat(
      [
        self.regions,
        DataFrame(
          {
            "AREA" : [
              "v4pcuf",
              "v4fpuf",
              "v4pclf",
              "v4fplf"
            ],
            "REGION" : ["Occipital"] * 4,
            "COLOR" : [
              to_hex((0 ,97/255, 65/255))
            ] * 4
          }
        )
      ]
    )

  def read_data_org(self):
    # Read flatmap data ----
    index, name, sequence, _, x, y, name2 = np.loadtxt(
        "utils/flatmap/flatmapdataframe2c.csv",
        skiprows=1,
        delimiter=',',
        unpack=True,
        dtype=str
    )
    name = [nme.lower().replace("\"", "") for nme in name]
    name2 = [nm2.lower().replace("\"", "") for nm2 in name2]
    # Read labels and positions ----
    f = open('utils/flatmap/F99-107-centres.json')
    data = json.load(f)
    f.close()
    # Prepare polygons ----
    sequence = sequence.astype(int)
    x = x.astype(float)
    y = y.astype(float)
    total_areas = len(set(name2))
    start = np.zeros(total_areas)
    end = np.zeros(total_areas)
    sindex = 0
    eindex = 0
    for index, s in enumerate(sequence):
      if(s == 1):
        start[sindex] = index
        sindex += 1
      if(s == 3):
        end[eindex] = index
        eindex += 1
    start = start.astype(int)
    end = end.astype(int)
    return start, end, x, y, name, total_areas, data

  def read_data_newseg_2022(self, **kwargs):
    # Read flatmap data ----
    if "flatmap_path" in kwargs.keys():
      _, name, name2, x, y = np.loadtxt(
        kwargs["flatmap_path"],
        skiprows=1,
        delimiter=',',
        unpack=True,
        dtype=str
      )
    else:
      _, name, name2, x, y = np.loadtxt(
        "utils/flatmap/FlatmapCoordinates_NewSeg_2022.csv",
        skiprows=1,
        delimiter=',',
        unpack=True,
        dtype=str
      )
    name = [nme.lower().replace("\"", "") for nme in name]
    name2 = [nm2.lower().replace("\"", "") for nm2 in name2]
    # Prepare polygons ----
    name = np.array(name)
    name2 = np.array(name2)
    x = x.astype(float)
    y = y.astype(float)
    total_areas = len(set(name2))
    start = np.zeros(total_areas)
    end = np.zeros(total_areas)
    sindex = 1
    eindex = 0
    ## Start ----
    pre = name2[0]
    uname = [pre]
    for idx in np.arange(1, len(name2)):
      area = name2[idx]
      if area != pre:
        start[sindex] = idx
        pre = area
        uname.append(pre)
        sindex += 1
    ## End ----
    for idx in np.arange(len(name2)-1):
      area = name2[idx]
      fut = name2[idx + 1]
      if area != fut:
        end[eindex] = idx
        eindex += 1
    end[eindex] = len(name2) - 1
    start = start.astype(int)
    end = end.astype(int)
    # label area coords ----
    from shapely.geometry import Polygon
    data = np.zeros((len(uname), 2))
    for i, nme in enumerate(uname):
      idx = np.where(name2 == nme)[0]
      X = np.vstack(
        [x[idx], y[idx]]
      ).T
      P = Polygon(X).representative_point().coords[:][0]
      data[i, 0] = P[0]
      data[i, 1] = P[1]
    return start, end, x, y, name, total_areas, data, uname

  def plot_flatmap(self, **kwargs):
    sns.set_theme()
    # Format network area names ----
    self.struct_labels = self.format_areas(
      self.struct_labels
    )
    # Format Area regions ----
    self.regions["AREA"] = self.format_areas(
      self.regions["AREA"].to_numpy()
    )
    # Read data ----
    start, end, x, y, name, total_areas, data, uname = self.read_data_newseg_2022(**kwargs)
    # Define colormap ----
    nc = len(np.unique(self.labels))
    cm = sns.color_palette("hls", nc)
    # Creating layout ----
    fsize = 10 
    _, axFlatmap = plt.subplots(
      1, 1, figsize=(fsize, fsize)
    )
    # Start plotting!!! ----
    for index in np.arange(total_areas):
      px = x[start[index]:(end[index] + 1)]
      py = y[start[index]:(end[index] + 1)]
      vertices = np.column_stack((px, py))
      aindex = self.get_area_index(
        name[start[index]], self.struct_labels
      )
      if aindex != -1 and self.labels[aindex] != -1 and self.struct_labels[aindex] not in self.overlap:
        color = cm[self.labels[aindex]]
        color = list(color)
        color = tuple(color)
        pattern=""
      elif self.struct_labels[aindex] in self.overlap and aindex != -1:
        color = tuple((153/255, 0, 18/255, 0.5))
        pattern="////"
      elif self.labels[aindex] == -1 and aindex != -1 and self.struct_labels[aindex] not in self.overlap:
        color = tuple((0, 158/255, 115/255, 0.5))
        pattern="////"
      else:
        pattern = "\\\\\\"
        color = tuple((.5, .5, .5, 0.3))
      axFlatmap.add_patch(
        plt.Polygon(
          vertices,
          closed=True,
          edgecolor=[0, 0, 0],
          linewidth=.5,
          facecolor=color,
          hatch=pattern
        )
      )
    for i, nme in enumerate(uname):
      axFlatmap.text(data[i, 0], data[i, 1], nme, fontsize=6)
    axFlatmap.set_aspect("equal")
    axFlatmap.autoscale()
    axFlatmap.set_axis_off()
    plt.savefig(
      join(
        self.plot_path,
        "K_{}".format(self.K),
        "{}.png".format(self.R)
      ),
      dpi=300
    )

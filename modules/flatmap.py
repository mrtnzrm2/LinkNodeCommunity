import numpy as np
import json
import matplotlib.pyplot as plt
from os.path import join
from pathlib import Path
import seaborn as sns
from various.network_tools import *

class FLATMAP:
  def __init__(self, nodes, version, labels, regions, plot_path, **kwargs) -> None:
    # Get NET parameters ----
    self.nodes = nodes
    self.version = version
    if "EC" in kwargs.keys():
      if kwargs["EC"]:
        self.struct_labels = labels[:nodes]
      else:
        self.struct_labels = labels
    else:
      self.struct_labels = labels
    self.struct_labels = np.char.lower(self.struct_labels.astype(str))    # Pay attention: could vary!!!
    self.plot_path = join(plot_path, "flatmap")
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
    ## struct_labels
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

  def plot_flatmap(self, direction="source", cmap_name="deep", **kwargs):
    sns.set_theme()
    plt.style.use("dark_background")
    sns.set_context("talk")
    import matplotlib.patheffects as path_effects
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
    cm = sns.color_palette(cmap_name, nc)
    # Creating layout ----
    fsize = 10 
    _, axFlatmap = plt.subplots(1, 1, figsize=(fsize, fsize))
    # Start plotting!!! ----
    for index in np.arange(total_areas):
      px = x[start[index]:(end[index] + 1)]
      py = y[start[index]:(end[index] + 1)]
      vertices = np.column_stack((px, py))
      aindex = self.get_area_index(name[start[index]], self.struct_labels)
      if aindex != -1 and self.labels[aindex] != -1:
        color = cm[self.labels[aindex]]
        # color = np.hstack([color, [0.4]])

        color = list(color)
        color = tuple(color)
        # color = tuple((1, 1, 1, 1))
        pattern=""
      elif aindex != -1:
        color = tuple((1, 1, 1, 1))
        pattern=""
        color = tuple((153/255, 0, 18/255, 0.5))
        pattern="////"
      elif self.labels[aindex] == -1 and aindex != -1:
        # color = tuple((0, 158/255, 115/255, 0.5))
        color = tuple((1, 1, 1, 1))
        # pattern="////"
        pattern=""
      else:
        pattern = "\\\\\\"
        color = tuple((.5, .5, .5, 0.3))
        # color = tuple((1, 1, 1, 1))
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
      txt = axFlatmap.text(data[i, 0], data[i, 1], nme, fontsize=6)
      txt.set_path_effects(
        [
          path_effects.Stroke(linewidth=1, foreground='black'),
          path_effects.Normal()
        ]
      )
    axFlatmap.set_aspect("equal")
    axFlatmap.autoscale()
    axFlatmap.set_axis_off()

    fname = join(self.plot_path,"K_{}".format(self.K), "{}_{}.png".format(self.R, direction))
    print(fname)
    plt.savefig(fname, dpi=300)
    plt.close()

  def plot_flatmap_index(self, pivot, values, max_value=None, index_name="Hellinger2", cmap_name="flare", **kwargs):
    sns.set_theme()
    # plt.style.use("dark_background")
    sns.set_context("talk")

    import matplotlib.patheffects as path_effects
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
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
    cm = sns.color_palette(cmap_name, as_cmap=True)
    # Creating layout ----
    # fsize = 5
    _, axFlatmap = plt.subplots(1, 1)
    # Start plotting!!! ----
    pindex = aindex = self.get_area_index(pivot, self.struct_labels)
    for index in np.arange(total_areas):
      px = x[start[index]:(end[index] + 1)]
      py = y[start[index]:(end[index] + 1)]
      vertices = np.column_stack((px, py))
      aindex = self.get_area_index(name[start[index]], self.struct_labels)

      pattern=""
      edgecolor =  [0, 0 , 0]
      linewidth = 0.5

      if aindex != -1 and aindex != pindex and aindex < values.shape[0]:
        if max_value:
          color = cm(values[aindex]/max_value)
        else:
          color = cm(values[aindex])
        # color = np.hstack([color, [0.4]])
        color = list(color)
        color = tuple(color)
        
      elif aindex != -1 and aindex == pindex and aindex < values.shape[0]:
        color = cm(0)
        edgecolor = "#ff0033"
        linewidth = 2

      else:
        pattern = "\\\\\\"
        color = tuple((.5, .5, .5, 0.3))
      axFlatmap.add_patch(
        plt.Polygon(
          vertices,
          closed=True,
          edgecolor=edgecolor,
          linewidth=linewidth,
          facecolor=color,
          hatch=pattern
        )
      )
    for i, nme in enumerate(uname):
      axFlatmap.text(data[i, 0], data[i, 1], nme, fontsize=5)

    axFlatmap.set_aspect("equal")
    axFlatmap.autoscale()
    axFlatmap.set_axis_off()

    # Add colorbar ---
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axFlatmap)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)

    if max_value:
      plt.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=max_value), cmap=cm), ax=axFlatmap, cax=ax_cb)
    else:
      plt.colorbar(ScalarMappable(cmap=cm), ax=axFlatmap, cax=ax_cb)

    plt.tight_layout()

    # Arrange path ----
    plot_path = join(self.plot_path, index_name)
    # Crate path ----
    
    Path(
      plot_path
    ).mkdir(exist_ok=True, parents=True)

    pivot = pivot.replace("/", "_")
    print(join(plot_path, "{}.png".format(pivot)))
    plt.savefig(
      join(plot_path, "{}.png".format(pivot)),
      dpi=300
    )
    plt.close()

  def plot_flatmap_91(self, NET, H, ax: plt.Axes, cmap="deep", color_order=None, index_name="Hellinger2"):
    import matplotlib.patheffects as path_effects
    from modules.discovery import discovery_channel
    file_path = "/Users/jmarti53/Documents/Projects/LINKPROJECT/Dlink/utils/flatmap_91/"
    # Get best K and R ----
    K, R, _ = get_best_kr_equivalence("_S", H)
    k = K[0]
    r = R[0]

    clustering = get_labels_from_Z(H.Z, r)
    clustering = skim_partition(clustering)
    labels = NET.struct_labels[:NET.nodes]

    print(">> reading data")
    clusterColors = sns.color_palette(cmap)

    print("> reading flatmap data")
    f = open(file_path + "macaque_flatmap_91B.json")
    polygonData = json.load(f)
    f.close()

    f = open(file_path + "label_coord.json")
    data = json.load(f)
    f.close()

    print(">>> preparing data for creating polygons")
    index = []
    name = []
    x = []
    y = []
    for pdindex, pd in enumerate(polygonData):            
        for e in polygonData[pd][0]:          
            index.append(pdindex)
            name.append(pd)      
            x.append(e[0])
            y.append(e[1])

    for index, n in enumerate(name):
        name[index] = n.replace("\"", "")

    totalAreas = len(set(name))
    start = np.zeros(totalAreas)
    end = np.zeros(totalAreas)
    sindex = 0
    eindex = 0
    for index, n in enumerate(name):

        if(index == 0):
            start[sindex] = index
            sindex += 1
        if(index != len(name) - 1):
            if(n != name[index + 1]):
                end[eindex] = index
                eindex += 1
                start[sindex] = index + 1
                sindex += 1
        else:
            end[eindex] = len(name) - 1

    start = start.astype(int)
    end = end.astype(int)

    print("> Covers")
    _, nocs, noc_sizes, partition  = discovery_channel["discovery_7"](
        H, k, clustering, direction="both", index="Hellinger2"
    )

    unique_clusters_id = np.unique(partition)
    keff = len(unique_clusters_id)
    # Generate all the colors in the color map -----
    if -1 in unique_clusters_id:
      save_colors = sns.color_palette(cmap, keff - 1)
      cmap_heatmap = [[]] * keff
      # cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
      cmap_heatmap[0] = [1., 1., 1.]
      cmap_heatmap[1:] = save_colors
    else:
      save_colors = sns.color_palette(cmap, keff)
      cmap_heatmap = [[]] * (keff+1)
      # cmap_heatmap[0] = [199 / 255.0, 0, 57 / 255.0]
      cmap_heatmap[0] = [1., 1., 1.]
      cmap_heatmap[1:] = save_colors
    cmap_heatmap = np.array(cmap_heatmap)
    if isinstance(color_order, np.ndarray):
      cmap_heatmap[1:] = cmap_heatmap[1:][color_order]

    def getAreaIndex(area, labels):
      for index, l in enumerate(labels):
          if(l == area):
              return index
          else:
              if((l == "v1" or l == "v2" or l == "v4") and l in area and len(area) > 3):
                  return index
      return -1

    nocs_indeces = [
        getAreaIndex(labels[n].lower(), labels) for n in np.where(clustering == -1)[0]
    ]

    nodes_memberships = {
      getAreaIndex(labels[n].lower(), labels) : {"id" : [0] * (keff+1), "size" : [0] * (keff+1)} for n in nocs_indeces
    }

    for _, key in enumerate(nocs.keys()):
        index_key = getAreaIndex(key, labels)
        if index_key not in nocs_indeces: continue
        for id in nocs[key]:
            if id == -1:
              nodes_memberships[index_key]["id"][0] = 1
              nodes_memberships[index_key]["size"][0] = 1
            else:
              nodes_memberships[index_key]["id"][id + 1] = 1
              nodes_memberships[index_key]["size"][id + 1] = noc_sizes[key][id]

    print("> plotting")

    for index in range(totalAreas):
        px = x[start[index]:(end[index] + 1)]
        py = y[start[index]:(end[index] + 1)]

        vertices = np.column_stack((px, py))

        aindex = getAreaIndex(name[start[index]].lower(), labels)
        if aindex != -1 and (clustering[aindex] != -1 or name[start[index]].lower() not in nocs.keys()):
            colorIndex = clustering[aindex]
            color = clusterColors[colorIndex]
        elif aindex != -1 and clustering[aindex] == -1:
            color = tuple((1, 1, 1, 0))
        else:
            color = tuple((.5, .5, .5, .25))
        
        pattern = ''

        if aindex != -1 and clustering[aindex] != -1:
          ax.add_patch(plt.Polygon(vertices, closed=True, edgecolor=[1, 1, 1], linewidth=.5, facecolor=color, hatch=pattern))
        elif aindex != -1 and clustering[aindex] == -1:
          ax.add_patch(plt.Polygon(vertices, closed=True, edgecolor=[0, 0, 0], linewidth=.5, facecolor=color, hatch=pattern))
        else:
          ax.add_patch(plt.Polygon(vertices, closed=True, edgecolor=[1, 1, 1], linewidth=.5, facecolor=color, hatch=pattern))


    for d in data:
        aindex = getAreaIndex(d.lower(), labels)
        if aindex != -1:
          t = ax.text(data[d][0][0], data[d][0][1], d, fontsize=6, weight="bold", color="yellow")
          t.set_path_effects(
          [
            path_effects.Stroke(linewidth=0.5, foreground='k'),
            path_effects.Normal()
          ])
        if d.lower() in nocs.keys():
            wedgecolor = "k"
            ax.pie(
              [s for s in nodes_memberships[aindex]["size"] if s != 0],
              center=np.array([data[d][0][0], data[d][0][1]]),
              colors = [cmap_heatmap[i] for i, id in enumerate(nodes_memberships[aindex]["id"]) if id != 0],
              radius=0.45,
              wedgeprops={"linewidth" : 0.35, "edgecolor": wedgecolor}
            )

    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_axis_off()

    # plt.show()

    # Arrange path ----
    plot_path = join(self.plot_path, index_name)
    # Crate path ----
    
    Path(
      plot_path
    ).mkdir(exist_ok=True, parents=True)

    print(join(plot_path, "{}.png".format(K[0])))
    plt.savefig(
      join(plot_path, "{}.png".format(K[0])),
      dpi=300
    )
    plt.close()

    print("> done!")


  def plot_regions(self, direction="source", cmap_name="deep", **kwargs):
    # sns.set_theme()
    plt.style.use("dark_background")

    # import matplotlib

    # # Say, "the default sans-serif font is COMIC SANS"
    # matplotlib.rcParams['font.sans-serif'] = "Arial"
    # # Then, "ALWAYS use sans-serif fonts"
    # matplotlib.rcParams['font.family'] = "sans-serif"
    font = "Arial"

    import matplotlib.patheffects as path_effects
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
    # Creating layout ----
    fsize = 10
    _, axFlatmap = plt.subplots(1, 1, figsize=(fsize, fsize))
    # Start plotting!!! ----
    for index in np.arange(total_areas):
      px = x[start[index]:(end[index] + 1)]
      py = y[start[index]:(end[index] + 1)]
      vertices = np.column_stack((px, py))
      aindex = self.get_area_index(name[start[index]], self.struct_labels)
      if aindex != -1:
        color = self.regions["COLOR"].loc[self.regions["AREA"] == self.labels[aindex]].to_numpy()[0]
        # color = np.hstack([color, [0.4]])
        # color = list(color)
        # color = tuple(color)
        pattern=""
      else:
        pattern = "\\\\\\"
        color = tuple((.5, .5, .5, 0.3))
      axFlatmap.add_patch(
        plt.Polygon(
          vertices,
          closed=True,
          edgecolor=[0.5, 0.5, 0.5],
          linewidth=.5,
          facecolor=color,
          hatch=pattern,
          alpha=0.5
        )
      )

    fontsize=15
    fontcolor = "black"
    for i, nme in enumerate(uname):
      if nme == "v1fpuf_2": continue
      elif nme == "v1fplf":
        txt = axFlatmap.text(data[i, 0] - 20, data[i, 1] - 10, nme, fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v1pcuf_2": continue
      elif nme == "ento_2": continue
      elif nme == "ento_3": continue
      elif nme == "46v_1": continue
      elif nme == "v1fpuf_1":
        txt = axFlatmap.text(data[i, 0] - 25, data[i, 1], "v1fpuf", fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v1pcuf_1":
        txt = axFlatmap.text(data[i, 0] - 20, data[i, 1], "v1pcuf", fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v2pclf":
        txt = axFlatmap.text(data[i, 0] - 25, data[i, 1] - 5, "v2pclf", fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v2fplf":
        txt = axFlatmap.text(data[i, 0] - 25, data[i, 1], "v2fplf", fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v2pcuf": 
        txt = axFlatmap.text(data[i, 0] - 25, data[i, 1], "v2pcuf", fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v2c":
        txt = axFlatmap.text(data[i, 0] - 15, data[i, 1] - 35, "v2c", fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v6":
        txt = axFlatmap.text(data[i, 0] - 10, data[i, 1] - 25, "v6", fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v1pclf":
        txt = axFlatmap.text(data[i, 0] - 30, data[i, 1] - 10, "v1pclf", fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v3fpuf":
        txt = axFlatmap.text(data[i, 0] - 30, data[i, 1], "v2fpuf", fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v2fpuf":
        txt = axFlatmap.text(data[i, 0] - 30, data[i, 1], "v2fpuf", fontsize=fontsize, font=font, color=fontcolor)
      elif nme == "v4uf":
        txt = axFlatmap.text(data[i, 0] - 15, data[i, 1]+10, "v4uf", fontsize=fontsize, font=font, color=fontcolor)
      else:
        txt = axFlatmap.text(data[i, 0], data[i, 1], nme, fontsize=fontsize, font=font, color=fontcolor)
      # txt.set_path_effects(
      #   [ 
      #     path_effects.Stroke(linewidth=1, foreground='black'),
      #     path_effects.Normal()
      #   ]
      # )
    axFlatmap.set_aspect("equal")
    axFlatmap.autoscale()
    axFlatmap.set_axis_off()
    plt.savefig(
      join(
        self.plot_path,
        "flat_map_regions.svg"
      ),
      dpi=300, transparent=True
    )
    plt.close()

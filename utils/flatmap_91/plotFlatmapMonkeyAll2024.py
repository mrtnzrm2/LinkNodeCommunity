# TODO: not the good version because only the last thing is visible, transform it to be a plotted on a grid!!!

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys
import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import copy

np.set_printoptions(linewidth=240)

def getAreaIndex(area, labels):
    for index, l in enumerate(labels):
        if(l == area):
            return index
        else:
            if((l == "V1" or l == "V2" or l == "V4") and l in area and len(area) > 3):
                return index
    return -1

print("> defining colors for clusters")
clusterColors = []
clusterColors.append(tuple((31./255, 119./255, 180./255))) #blue #1f77b4
clusterColors.append(tuple((255./255, 127./255, 14./255))) #orange #ff7f0e
clusterColors.append(tuple((44./255, 160./255, 44./255))) #green #2ca02c
clusterColors.append(tuple((214./255, 39./255, 40./255))) #red #d62728
clusterColors.append(tuple((148./255, 103./255, 189./255))) #violet #9467bd
clusterColors.append(tuple((140./255, 86./255, 75./255))) #brown #8c564b
clusterColors.append(tuple((227./255, 119./255, 194./255))) #purple #e377c2
clusterColors.append(tuple((127./255, 127./255, 127./255))) #gray #7f7f7f
clusterColors.append(tuple((188./255, 189./255, 34./255))) #keki #bcbd22
clusterColors.append(tuple((23./255, 190./255, 207./255))) #turquoise #17becf
clusterColors.append(tuple((48./255, 25./255, 52./255))) #dark purple #301934
clusterColors.append(tuple((114./255, 47./255, 55./255))) #wine #722F37
clusterColors.append(tuple((0./255, 139./255, 139./255))) #cyan #008b8b

print("> creating layout")
fsize = 10 
rows = 3
columns = 2

maxClusterNumber = 8

left, width = 0.04, 0.44
bottom, height = 0.03, 0.28
smallHeightWidth = 0.04
spacing = 0.04

# animals = ("mouse", "monkey")
animals = ("monkey", "mouse")
# animal_folder = ("19x47", "29x91")
animal_folder = ("29x91", "19x47")
# animal_filenames = ("Mouse_Database_linkpred", "Neurons91x29_Arithmean_DBV23.45_linkpred")
animal_filenames = ("Neurons91x29_Arithmean_DBV23.45_linkpred", "Mouse_Database_linkpred",)
size = (47, 91)
imputation_methods = ("GB", "RF")
axIndexes = (2, 5, 1, 4, 0, 3)
panelLabels = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")

for animalIndex, animal in enumerate(animals):
    generators = []

    if(animal == "mouse"):
        rows = 2
        maxClusterNumber = 6
        height = 0.44
        axIndexes = (1, 3, 0, 2)

    print("> processing {}".format(animal))

    for imindex, im in enumerate(imputation_methods):
        fig = plt.figure(figsize=(columns * fsize, 0.75 * rows * fsize))

        axFlatmap = {}      
        axIndex = 0
        for c in range(columns):
            for r in range(rows):
                flatmap = [left + c%columns * (width + spacing), bottom + r%rows * (height + spacing), width, height]
                axFlatmap[axIndex] = fig.add_axes(flatmap)
                axIndex += 1
        axIndex = 0
        
        for clusterNumber in range(2, maxClusterNumber): # rethink if 8 cluster state needed, only if lobe distribution can be plotted with different color scheme, also update fig size if this is included!!!
            print("> processing {} cluster state".format(clusterNumber))
            foldername = "results/{}/{}".format(animal_folder[animalIndex], im)
            filename = "{}".format(animal_filenames[animalIndex])

            print(">> reading data")
            # dt = np.dtype('U8, d')
            clustering = np.loadtxt("{}/{}/{}_clustering.txt".format(foldername, clusterNumber, filename))
            clustering = clustering.astype(int)

            clusteringList = list(clustering)
            clusterCount = len(list(filter(lambda x: (x < 0), clusteringList)))
            if(clusterCount == 0):
                continue

            labels = np.loadtxt("{}/{}_areas.txt".format(foldername, filename), dtype=str)
            # print(labels)

            print(">> building consistent coloring")
            for index, c in enumerate(clustering):
                if(c < 0 and labels[index] not in generators):
                    generators.append(labels[index])

            print("> reading flatmap data")
            if(animal == "mouse"):
                index, name, sequence, view, x, y, name2 = np.loadtxt("utils/flatmap/mouse_sbri_flatmap_contours.csv", skiprows=1, delimiter=',', unpack=True, dtype=str)
                f = open('utils/flatmap/mouse_sbri_flatmap_contours.json')
                data = json.load(f)
                f.close()
            else:
                f = open('utils/flatmap/macaque_flatmap_91B.json')
                polygonData = json.load(f)
                f.close()

                f = open('utils/flatmap/label_coord.json')
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

            print("> plotting")
            # counter = imindex * rows + (clusterNumber - 2)
            # print(counter)
            # if(counter < 4):
            #     axIndex = 3 - counter
            # else:
            #     axIndex = 10 - counter
            # print(axIndex)
            axIndex = axIndexes[clusterNumber - 2]

            for index in range(totalAreas):
                px = x[start[index]:(end[index] + 1)]
                py = y[start[index]:(end[index] + 1)]

                vertices = np.column_stack((px, py))

                aindex = getAreaIndex(name[start[index]], labels)
                if(aindex != -1):
                    colorIndex = generators.index(labels[list(clustering).index(-1. * abs(clustering[aindex]))])
                    color = clusterColors[colorIndex]
                else:
                    color = tuple((.5, .5, .5, .25))
                
                if(clustering[aindex] < 0):
                    pattern = '...'
                else:
                    pattern = ''

                axFlatmap[axIndex].add_patch(plt.Polygon(vertices, closed=True, edgecolor=[1, 1, 1], linewidth=.5, facecolor=color, hatch=pattern))

            for d in data:
                if(animal == "mouse"):
                    axFlatmap[axIndex].text(data[d][0], data[d][1], d, fontsize=12)
                else:
                    axFlatmap[axIndex].text(data[d][0][0], data[d][0][1], d, fontsize=12)

            axFlatmap[axIndex].set_aspect("equal")
            axFlatmap[axIndex].autoscale()
            axFlatmap[axIndex].set_axis_off()

            #TODO: create proper legend
            # norm = matplotlib.colors.Normalize(vmin=lower, vmax=upper)
            # cb = matplotlib.colorbar.ColorbarBase(axLegend, cmap=cm, norm=norm, orientation='vertical')
            # labels = [item.get_text() for item in axLegend.get_yticklabels()]
            # labels[0] = '100%\nventral'
            # labels[-1] = '100%\ndorsal'
            # axLegend.set_yticklabels(labels)
            # cb.ax.tick_params(labelsize=12) 
            # axLegend.set_axis_off()

            axFlatmap[axIndex].set_title("{}".format(panelLabels[clusterNumber - 2]), loc="left", fontsize=26, fontweight="bold")

        print(">> exporting plot")
        plt.savefig("{}/{}_flatmap_all_in_one.pdf".format(foldername, filename))
        plt.close()

print("> done!")

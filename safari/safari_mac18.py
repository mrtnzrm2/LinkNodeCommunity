# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Boolean aliases ----
T = True
F = False
# Standard libs ----
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
# Personal libs ---- 
from networks.MAC.mac11 import MAC18
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = F
lookup = F
prob = F
cut = F
structure = "SLN"
mode = "ZERO"
distance = "MAP3D"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = 0
opt_score = ["_S"]
save_data = T
version = "18d91"
__nodes__ = 18
__inj__ = 18


import networkx as nx
import matplotlib.pyplot as plt

# Sample dictionary with keys as numbers and values as lists of strings
data_dict = {
    1: ['A'],
    2: ['D'],
    3: ['F']
}

# Create a directed graph
G = nx.Graph()

# Add nodes and edges to the graph based on the dictionary
for key, values in data_dict.items():
    G.add_node(key)
    for value in values:
        G.add_node(value)
        G.add_edge(key, value)

print(G.nodes)
nx.set_node_attributes(G, {1: "1", 2: "2", 3: "3", "A" : "1", "D": "2", "F" : "3"}, 'subset')

# Create a layout for the nodes
pos = nx.multipartite_layout(G, align='vertical')

# Draw the network plot
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, font_weight='bold')

plt.title('Series of Parallel Disconnected Networks')
plt.show()



# # Start main ----
# if __name__ == "__main__":
#   # Load structure ----
#   NET = MAC18(
#     linkage, mode,
#     nlog10 = nlog10,
#     structure = structure,
#     lookup = lookup,
#     version = version,
#     nature = nature,
#     model = imputation_method,
#     distance = distance,
#     inj = __inj__,
#     topology = topology,
#     index = index,
#     mapping = mapping,
#     cut = cut,
#     b = bias
#   )

#   NET.get_sln()
  
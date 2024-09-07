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
import networkx as nx
# Personal libs ---- 
from networks.MAC.mac57 import MAC57
from modules.hierarentropy import Hierarchical_Entropy
from modules.flatmap import FLATMAP
from modules.colregion import colregion
from various.network_tools import *


from various.hit import EHMI, NHMI, HMI, check

A = [[0, 2], [1]]
B = [[0], [1], [2]]

print(NHMI(A, B))
print(EHMI(A, B))
print(HMI(A, B))

print(NHMI(A, A))
print(EHMI(A, A))
print(HMI(A, A))
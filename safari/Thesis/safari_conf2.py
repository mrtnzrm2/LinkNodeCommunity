import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from networks.swapnet import SWAPNET

N = 1000
models = []
P = [0.01]

ALPHA = [0.4]
gamma = 0.05


models.append(nx.erdos_renyi_graph(N, P[0]))

degree_sequence = [k for n, k in models[-1].degree()]
models.append(nx.configuration_model(degree_sequence, create_using=nx.Graph))

models.append(nx.scale_free_graph(N, alpha=ALPHA[0], beta=1-gamma-ALPHA[0]))

degree_in_sequence = [k for n, k in models[-1].in_degree()]
degree_out_sequence = [k for n, k in models[-1].out_degree()]
degree_sequence = [k for n, k in models[-1].degree()]

kp  = 1.1
degree_prime = np.power(degree_sequence, 1/kp )
max_degree_prime = np.max(degree_prime)

RAND = SWAPNET(N, N,"single", "ZERO", 0)
RAND.A = nx.adjacency_matrix(models[-1]).todense().astype(float)
RAND.A = np.array(RAND.A)
# np.fill_diagonal(RAND.A, 0)

RAND.random_one_k(run=True, swaps=10000, on_save_csv=False) 

models.append(nx.DiGraph(RAND.A))

# models.append(
#   nx.directed_configuration_model(degree_in_sequence, degree_out_sequence, create_using=nx.DiGraph)
# )


fig, axes = plt.subplots(2, 2)
 
# edge_widths = [0.1, 0.1]
# alphas = [0.4, 0.4]
 
edge_widths = 0.1
alphas = 0.4
  
# axes[i].set_title(r"$\rho = $"+f"{P[i]}")
# axes[i].set_title(r"$\rho_{c} = $"+f"{P[i]}")

# ER

pos = nx.spring_layout(models[0])
nx.draw_networkx_nodes(models[0], pos=pos, node_size=10, node_color="black", ax=axes[0,0])
nx.draw_networkx_edges(models[0], pos=pos, width=edge_widths, ax=axes[0,0], alpha=alphas)
axes[0,0].axis('off')

pos = nx.spring_layout(models[1])
nx.draw_networkx_nodes(models[1], pos=pos, node_size=10, node_color="black", ax=axes[0,1])
nx.draw_networkx_edges(models[1], pos=pos, width=edge_widths, ax=axes[0,1], alpha=alphas)
axes[0,1].axis('off')

# SF

cmap = sns.color_palette("viridis", as_cmap=True)

pos = nx.spring_layout(models[2])
[nx.draw_networkx_nodes(models[2], pos=pos, nodelist=[n], node_size=kp, node_color=cmap(kp / max_degree_prime), ax=axes[1,0]) for n, kp in enumerate(degree_prime)]
nx.draw_networkx_edges(models[2], pos=pos, width=edge_widths, ax=axes[1,0], alpha=alphas)
axes[1,0].axis('off')

pos = nx.spring_layout(models[3])
[nx.draw_networkx_nodes(models[3], pos=pos, nodelist=[n], node_size=kp, node_color=cmap(kp / max_degree_prime), ax=axes[1,1]) for n, kp in enumerate(degree_prime)]
nx.draw_networkx_edges(models[3], pos=pos, width=edge_widths, ax=axes[1,1], alpha=alphas)
axes[1,1].axis('off')

fig.set_figheight(15)
fig.set_figwidth(15)
fig.tight_layout()

# plt.savefig("../plots/RAN/Thesis/conf2.svg", trasparent=True)
plt.savefig("../plots/RAN/Thesis/conf2.png", dpi=300)
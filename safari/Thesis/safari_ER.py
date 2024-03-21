import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

N = 1000
models = []
P = [0.0001, 1/N, 0.01]


for p in P:
  models.append(nx.erdos_renyi_graph(N, p))


fig, axes = plt.subplots(1, 3)
 
edge_widths = [1, 0.1, 0.1]
alphas = [1, 0.5, 0.2]

for i in np.arange(3):
  
  if i != 1:
    axes[i].set_title(r"$\rho = $"+f"{P[i]}")
  else:
    axes[i].set_title(r"$\rho_{c} = $"+f"{P[i]}")
  pos = nx.spring_layout(models[i])
  nx.draw_networkx_nodes(models[i], pos=pos, node_size=10, node_color="black", ax=axes[i])
  nx.draw_networkx_edges(models[i], pos=pos, width=edge_widths[i], ax=axes[i], alpha=alphas[i])
  axes[i].axis('off')

fig.set_figheight(5)
fig.set_figwidth(15)
fig.tight_layout()

plt.savefig("../plots/RAN/Thesis/ER.png", dpi=300)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

N = 1000
models = []
ALPHA = [0.4]
# BETA = [0.2, 0.4, 0.6]
gamma = 0.05


for al in ALPHA:
  models.append(nx.scale_free_graph(N, alpha=al, beta=1-gamma-al))

# for be in BETA:
#   models.append(nx.scale_free_graph(N, alpha=1-gamma-be, beta=be))


fig, axes = plt.subplots(1, 1)
 
edge_widths = [0.1]
alphas = [0.4]

# for i in np.arange(1):
  
# axes.set_title(r"$\alpha: $"+f"{ALPHA[0]}")
# axes[i].set_title(r"$\beta: $"+f"{BETA[i]}")

pos = nx.spring_layout(models[0])
degree_sequence = np.array([d for n, d in models[0].degree()])
dp = 1.1
degree_sequence_pr = np.power(degree_sequence, 1/dp)
max_degree_pr = np.max(degree_sequence_pr)

cmap = sns.color_palette("viridis", as_cmap=True)


[nx.draw_networkx_nodes(models[0], pos=pos, nodelist=[n], node_size=kp, node_color=cmap(kp / max_degree_pr), ax=axes) for n, kp in enumerate(degree_sequence_pr)]
nx.draw_networkx_edges(models[0], pos=pos, width=edge_widths[0], ax=axes, alpha=alphas[0])
axes.axis('off')

fig.set_figheight(15)
fig.set_figwidth(15)
fig.tight_layout()

plt.savefig("../plots/RAN/Thesis/SF.png", dpi=300)
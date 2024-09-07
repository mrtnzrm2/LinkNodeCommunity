import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from matplotlib import gridspec, colors
import matplotlib as mpl

# Load plots ----
from directed_neighborhoods import directed_neighborhoods_plot
from link_neighbors import link_neighbors_plot
from disjointed_links import disjointed_links_plot
from link_similarity import link_similarity_plot
from distinguishing_two_nodes import distinguishing_two_nodes_plot
from sim_both_ways import sim_both_ways_plot

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.size"] = 12

fig = plt.figure(1, figsize=(7.08661, 5))
gs = gridspec.GridSpec(4, 3)
gs.update(wspace=0.1, hspace=0)

ax = fig.add_subplot(gs[0:2, 0:1])
xmin1, xmax1, ymin1, ymax1 = directed_neighborhoods_plot()
s1 = 0.3
plt.xlim(-s1 + xmin1, s1 + xmax1)
plt.ylim(-s1 + ymin1, s1 + ymax1)
ax.text(
  0.0, 1, "a", transform=ax.transAxes,
  fontsize=20, va='top', fontfamily='sans-serif', weight="bold", ha="right"
)

ax = fig.add_subplot(gs[0:2, 1:2])
ax.text(
  0.0, 1, "b", transform=ax.transAxes,
  fontsize=20, va='top', fontfamily='sans-serif', weight="bold", ha="right"
)
xmin2, xmax2, ymin2, ymax2 = link_neighbors_plot()
s2 = 0.2
plt.xlim(-s2 + xmin2, s2 + xmax2)
plt.ylim(-s2 + ymin2, s2 + ymax2)

ax = fig.add_subplot(gs[0:2, 2:3])
ax.text(
  0.0, 1, "c", transform=ax.transAxes,
  fontsize=20, va='top', fontfamily='sans-serif', weight="bold", ha="right"
)
xmin3, xmax3, ymin3, ymax3 = disjointed_links_plot()
s3 = 0.1
plt.xlim(-s3 + xmin3, s3 + xmax3)
plt.ylim(-s3 + ymin3, s3 + ymax3)

s4, s5 = 0.1, 0.1
link_similarity_plot(fig, gs, s4, s5)


ax = fig.add_subplot(gs[2:4, 1:2])
ax.text(
  0.0, 1, "f", transform=ax.transAxes,
  fontsize=20, va='top', fontfamily='sans-serif', weight="bold", ha="right"
)
xmin5, xmax5, ymin5, ymax5 = distinguishing_two_nodes_plot()
s5 = 0.7
plt.xlim(-s5 + xmin5, s5 + xmax5)
plt.ylim(-s5 + ymin5, s5 + ymax5)

ax = fig.add_subplot(gs[2:4, 2:3])
ax.text(
  0.0, 1, "g", transform=ax.transAxes,
  fontsize=20, va='top', fontfamily='sans-serif', weight="bold", ha="right"
)

xmin6, xmax6, ymin6, ymax6 = sim_both_ways_plot()
s6 = 0.1
plt.xlim(-s6 + xmin6, s6 + xmax6)
plt.ylim(-s6 + ymin6, s6 + ymax6)
 

sns.despine(top=True, bottom=True, left=True, right=True)
# plt.gca().set_aspect('equal')
# plt.gcf().tight_layout()

plt.savefig(
  "../Publication/Nature Neuroscience/Figures/1/Figure1_5.svg"
)
# plt.savefig(
#   "../Publication/Nature Neuroscience/Figures/1/Figure1_4.pdf"
# )
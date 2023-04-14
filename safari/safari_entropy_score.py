import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt


networks = np.array(["Zachary", "Erdos-Renyi", "Scale-free", "HRG", "Data", "EDR", "HSF"])
hierarchy = np.array(["link", "node"])
data = pd.DataFrame(
  {
    "Network" : np.repeat(networks, 2),
    "SH" : [0.2062, 0.1543, 0.2047, 0.0694, 0.2659, 0.3419, 0.3171, 0.5575, 0.4383, 0.2666, 0.4626, 0.2880, 0.5053, 0.4080],
    "hierarchy" : np.tile(hierarchy, 7)
  }
).sort_values("SH")

_, ax = plt.subplots(1, 1, figsize=(8, 7.5))
sns.barplot(
    data=data,
    x="Network",
    y="SH",
    hue="hierarchy",
    ax=ax
)
ax.set_ylabel(r"$s_{H}$")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
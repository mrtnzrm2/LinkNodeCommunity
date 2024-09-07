import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cmp = sns.color_palette("deep")
len_cmp = len(cmp)

print(cmp.as_hex())


x = cmp.as_hex()
y = [1] * len_cmp

data = {"color_name":x, "y":y}

sns.barplot(
  data=data,
  x="color_name",
  y="y",
  hue="color_name",
  palette="deep",
  legend=False
)

plt.xticks(rotation=90)
# plt.show()
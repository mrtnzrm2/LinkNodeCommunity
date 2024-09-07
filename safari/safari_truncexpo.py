import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncexpon
fig, ax = plt.subplots(1, 1)

x = np.linspace(3, 55.8, 100)

rv = truncexpon(55.8, scale=1/0.188, loc=3)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

ax.plot(x, truncexpon.pdf(x, 55.8, loc=3, scale=1/0.188),
  'r-', lw=5, alpha=0.6, label='truncexpon pdf')

r = truncexpon.rvs(55.8, loc=3, scale=1/0.188, size=1000)

print(np.min(r), np.max(r))

ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim([x[0], x[-1]])
ax.legend(loc='best', frameon=False)
plt.show()
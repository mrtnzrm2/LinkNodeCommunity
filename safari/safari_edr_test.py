# is it possible to have 2 EDR datasets with lambdas l1, l2,
# such that merging the 2 dataset and fitting lambdas leads to l3 
# that is outside the range (l1, l2) ?


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N = 5000

D1 = np.random.normal(25, 10, size=N)
where = np.logical_or(D1 < 0, D1 > 60)
while np.any(where):
    M = np.sum(where)
    D1[where] = np.random.normal(25, 10, size=M)
    where = np.logical_or(D1 < 0, D1 > 60)

D2 = np.abs(np.random.normal(25, 40, size=N))
where = np.logical_or(D2 < 0, D2 > 60)
while np.any(where):
    M = np.sum(where)
    D2[where] = np.random.normal(25, 40, size=M)
    where = np.logical_or(D2 < 0, D2 > 60)

#%%
A = 6 * np.exp(-0.18 * D1) * np.abs(np.random.normal(0, 1, size=N))
B = 6 * np.exp(-0.23 * D2) * np.abs(np.random.normal(0, 2, size=N))

def func(x, *args):
    a, b = args
    return a * np.exp(b * x)

plt.figure(1, figsize=(6,6), dpi=100)
plt.clf()

plt.semilogy(D1, A, 'ro', markersize=1)
plt.semilogy(D2, B, 'ko', markersize=1)

xx = np.linspace(1, 55, 100)

opt1, conv1 = curve_fit(func, D1, A, p0=(1, -0.2))
yy1 = func(xx, *opt1)
plt.plot(xx, yy1, 'r-', label="lam=%4.3f" % opt1[1])


opt2, conv2 = curve_fit(func, D2, B, p0=(1, -0.2))
yy2 = func(xx, *opt2)
plt.plot(xx, yy2, 'k-', label="lam=%4.3f" % opt2[1])


opt3, conv3 = curve_fit(func, np.hstack((D1,D2)).reshape(-1), np.hstack((A,B)).reshape(-1), p0=(1, -0.2))
yy3 = func(xx, *opt3)
plt.plot(xx, yy3, 'b-', label="lam=%4.3f" % opt3[1])


plt.legend()

plt.tight_layout()

plt.show()
#%%  distance distrib over inj/imp only

# from linkpred.data.loader40d91 import load_monkey_distances
# D, _ = load_monkey_distances()

# plt.figure(2, figsize=(6,4), dpi=100)
# plt.clf()

# D1 = D[:,:40]
# D2 = D[:,40:]

# plt.hist([D1[~np.isnan(D1)], D2[~np.isnan(D2)]], bins=np.linspace(0, 60, 13), density=True, label=['known', 'imputed'])
# plt.legend()

# plt.xlabel("distance")
# plt.ylabel("probability")
# plt.title("macaque distance distribution")

# plt.tight_layout()


# plt.savefig("distance_distribution.png")
import numpy as np
import numpy.typing as npt
from typing import Any
import ctools as ct

def H2(u : npt.NDArray[np.float64], v : npt.NDArray[np.float64]) -> np.float64:
    mu = np.sum(u)
    mv = np.sum(v)
    p = 0.
    for i in np.arange(u.shape[0]):
        p += np.power(np.sqrt(u[i] / mu) - np.sqrt(v[i] / mv), 2.)
    return p / 2.

def sample_complexity(h2 : np.float64, delta=0.05) -> np.float64:
  return np.log(1/delta) / h2

def mod_log(u : np.float64, v : np.float64) -> np.float64:
    if u == 0 and v == 0: return 0.
    elif u == 0 and v > 0: return -50.
    elif u > 0 and v == 0: return 50.
    elif u > 0 and v > 0: return np.log(u / v)
    else: raise ValueError("Problems with the input probabilities")

def likelihood_test(h0 : npt.NDArray[np.float64], ha : npt.NDArray[np.float64], samples : npt.NDArray[np.int64]) -> np.int8:
    p0 = h0 / np.sum(h0)
    pa = ha / np.sum(ha)
 
    test = -2 * np.sum(np.array([s * (mod_log(p0[i], pa[i])) for i, s in enumerate(samples)])) / (1 + (((np.sum(1 / p0[p0 > 0])) - 1) / (6 * np.sum(samples)) * (p0.shape[0] - 1)))

    if test > 0:
        return 0
    elif test < 0:
        return 1
    else:
        return np.argmax(np.random.multinomial(1, pvals=[0.95, 0.05]))

def gen_samples(u : npt.NDArray[np.float64], num_samples=100) -> npt.NDArray[np.int64]:
    # normalize u ----
    pu = u / np.sum(u)
    return np.random.multinomial(num_samples, pvals=pu, size=1)

if __name__ == "__main__":
    
    a = np.array([0.001, 100, 100, 0.001])
    b = np.array([50, 50, 0.001, 100])


    h2 = H2(a, b)
    print(h2)
    sc = sample_complexity(h2)
    if sc - np.floor(sc) > 0.01: sc = np.ceil(sc).astype(int)
    else: sc = np.floor(sc).astype(int)

    print(sc)

    N = 1000
    error = 0
    for i in np.arange(N):
      test_samples = gen_samples(a, num_samples=sc)
      t = likelihood_test(a, b, test_samples)
      if t == 1: error += 1

    print(error / N)

    print(np.log(1/0.05)/0.35)





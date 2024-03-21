# Standard libs ----
import numpy as np
from sklearn.linear_model import LinearRegression
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from itertools import chain, repeat, count, islice
from collections import Counter
# Personal libs ----
import signsimquest as squest

class Sim:
  def __init__(
    self, nodes : int, A, R, D, mode, topology="MIX",
    index="Hellinger2", lookup=0, undirected=False,
    architecture="product-positive"
  ):
    # Parameters ----
    self.nodes = nodes
    self.mode = mode
    self.A = A
    self.R = R
    self.D = D
    self.undirected = undirected
    self.architecture = architecture
    if not self.undirected:
      self.nonzero = (A != 0)
    else:
      self.nonzero = (np.triu(A) != 0)
    self.lup = lookup
    # Number of connections in the EC component ----
    if not self.undirected:
      self.leaves = np.sum(self.A[:nodes, :nodes] != 0).astype(int)
    else:
      self.leaves =  int(np.sum(self.A[:nodes, :nodes] != 0) / 2)
    self.topologies = {
      "MIX" : 0, "SOURCE" : 1, "TARGET" : 2
    }
    self.indices = {"Hellinger2": 0}
    self.topology = topology
    self.index = index

  def get_aik(self):
    aik =  self.R.copy()
    for i in np.arange(self.nodes):
      if self.mode == "ZERO":
        aik[i, i, 0] = 0
        aik[i, i, 1] = 0
      else:
        raise ValueError("Bad mode")
    return aik
  
  def get_aik_d(self):
    aikd = self.D.copy()[:, :self.nodes]
    return aikd

  def get_aki(self):
    R = self.R.copy()
    aki = np.zeros((R.shape[1], R.shape[0], R.shape[2]))
    aki[:, :, 0] = R[:, :, 0].T
    aki[:, :, 1] = R[:, :, 1].T
    for i in np.arange(self.nodes):
      if self.mode == "ZERO":
        aki[i, i, 0] = 0
        aki[i, i, 1] = 0
      else:
        raise ValueError("Bad mode")
    return aki
  
  def get_aki_d(self):
    akid = self.D.copy()[:, :self.nodes].T
    return akid

  def get_id_matrix(self):
    self.id_mat = self.A[:self.nodes, :]
    if self.undirected:
      self.id_mat = np.triu(self.id_mat)
    self.id_mat[self.id_mat != 0] = np.arange(1, self.leaves + 1)
    self.id_mat = self.id_mat.astype(int)
  
  def similarity_by_feature_cpp(self):
    
    if self.architecture == "product-positive":
       arch = 0
    elif self.architecture == "only-positive":
       arch = 1
    elif self.architecture == "only-negative":
       arch = 2
    elif self.architecture == "product-negative":
       arch = 3
    elif self.architecture == "all":
       arch = 4
    else:
       raise ValueError("Architecture not defined.")
    
    Quest = squest.signsimquest(
      self.nonzero, self.A, self.D, self.get_aki(), self.get_aik(),
      self.nodes, self.leaves, self.topologies[self.topology], self.indices[self.index],
      arch
      # 5
    )

    self.linksim_matrix = np.array(Quest.linksim_matrix)
    # print(self.linksim_matrix)
    self.source_sim_matrix = np.array(Quest.source_matrix)
    # print(self.source_sim_matrix)
    self.target_sim_matrix = np.array(Quest.target_matrix)

def repeat_chain(values, counts):
    return chain.from_iterable(map(repeat, values, counts))

def unique_combinations_from_value_counts(values, counts, r):
    n = len(counts)
    indices = list(islice(repeat_chain(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), repeat_chain(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), repeat_chain(count(j), counts[j:])):
            indices[i] = j

def unique_combinations(iterable, r):
    values, counts = zip(*Counter(iterable).items())
    return unique_combinations_from_value_counts(values, counts, r)
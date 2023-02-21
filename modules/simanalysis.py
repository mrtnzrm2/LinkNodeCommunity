# Standard libs ----
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from itertools import chain, repeat, count, islice
from collections import Counter
# Personal libs ----
import simquest as squest

class Sim:
  def __init__(
    self, nodes : int, A, R, D, mode, topology="MIX", index="jacp", lookup=0
  ):
    # Parameters ----
    self.nodes = nodes
    self.mode = mode
    self.A = A
    self.R = R
    self.D = D
    self.nonzero = (A != 0)
    self.lup = lookup
    # Get number of rows ----
    self.rows = self.A.shape[0]
    # Number of connections in the EC component ----
    self.leaves = np.sum(self.A[:nodes, :nodes] != 0).astype(int)
    self.topologies = {
      "MIX" : 0, "SOURCE" : 1, "TARGET" : 2
    }
    self.indices = {
      "jacp" : 0, "tanimoto" : 1, "cos" : 2, "jacw" : 3, "bsim" : 4
    }
    self.topology = topology
    self.index = index

  def get_aik(self):
    aik =  self.R.copy()
    aki = self.R.copy().T
    for i in np.arange(self.nodes):
      if self.mode == "ALPHA":
        aik[i, i] = np.nanmean(
          aik[i, :][
            aik[i, :] != self.lup
          ]
        )
      elif self.mode == "BETA":
        aik[i, i] = np.nanmean(
          aki[i, :][
            aki[i, :] != self.lup
          ]
        )
    return aik

  def get_aki(self):
    aki = self.R.copy().T
    aik = self.R.copy()
    for i in np.arange(self.nodes):
      if self.mode == "ALPHA":
        aki[i, i] = np.nanmean(
          aki[i, :][
            aki[i, :] != self.lup
          ]
        )
      elif self.mode == "BETA":
        aki[i, i] = np.nanmean(
          aik[i, :][
            aik[i, :] != self.lup
          ]
        )
    return aki

  def get_id_matrix(self):
    self.id_mat = self.A.copy()[:self.nodes, :]
    self.id_mat[self.id_mat != 0] = np.arange(1, self.leaves + 1)
    self.id_mat = self.id_mat.astype(int)

  def target_similarity(self, f, *args):
    aki = self.get_aki()
    target_mat = np.zeros((self.nodes, self.nodes)) * np.nan
    for i in np.arange(1, self.nodes):
      for j in np.arange(i):
        target_mat[i, j] = f(
          aki[i, :], aki[j, :], self.R.shape[0], self.lup, *args
        )
        target_mat[j, i] = target_mat[i, j]
    np.fill_diagonal(target_mat, 0)
    return target_mat

  def source_similarity(self, f, *args):
    aik = self.get_aik()
    source_mat = np.zeros((self.nodes, self.nodes)) * np.nan
    for i in np.arange(1, self.nodes):
      for j in np.arange(i):
        source_mat[i, j] = f(
          aik[i, :], aik[j, :], self.R.shape[1], self.lup, *args
        )
        source_mat[j, i] = source_mat[i, j]
    np.fill_diagonal(source_mat, 0)
    return source_mat

  def dtw_source_similarity(self):
    dist = self.D[:, :self.nodes]
    source_mat = np.zeros(
      (self.nodes, self.nodes)
    ) * np.nan
    for i in np.arange(self.nodes):
      for j in np.arange(i+1, self.nodes):
        seci = (self.R[i, :] != self.lup) & (~np.isnan(self.R[i, :]))
        secj = (self.R[j, :] != self.lup) & (~np.isnan(self.R[j, :]))
        i_array = np.zeros((np.sum(seci), 2))
        j_array = np.zeros((np.sum(secj), 2))
        i_array[:, 0] = dist[i, seci]
        i_array[:, 1] = self.R[i, seci]
        j_array[:, 0] = dist[j, secj]
        j_array[:, 1] = self.R[j, secj]
        i_array = i_array[np.argsort(i_array[:, 0]) ,:]
        j_array = j_array[np.argsort(j_array[:, 0]) ,:]
        dtw, _ = fastdtw(i_array, j_array, dist=euclidean, radius=10)
        source_mat[i, j] = dtw
        source_mat[j, i] = dtw
    source_mat = source_mat / self.nodes
    source_mat = 1 - source_mat
    np.fill_diagonal(source_mat, 0)
    return source_mat

  def dtw_target_similarity(self):
    dist = self.D[:, :self.nodes]
    target_mat = np.zeros(
      (self.nodes, self.nodes)
    ) * np.nan
    for i in np.arange(self.nodes):
      for j in np.arange(i+1, self.nodes):
        seci = (self.R[:, i] != self.lup) & (~np.isnan(self.R[:, i]))
        secj = (self.R[:, j] != self.lup) & (~np.isnan(self.R[:, j]))
        i_array = np.zeros((np.sum(seci), 2))
        j_array = np.zeros((np.sum(secj), 2))
        i_array[:, 0] = dist[seci, i]
        i_array[:, 1] = self.R[seci, i]
        j_array[:, 0] = dist[secj, j]
        j_array[:, 1] = self.R[secj, j]
        i_array = i_array[np.argsort(i_array[:, 0]) ,:]
        j_array = j_array[np.argsort(j_array[:, 0]) ,:]
        dtw, _ = fastdtw(i_array, j_array, dist=euclidean, radius=10)
        target_mat[i, j] = dtw
        target_mat[j, i] = dtw
    target_mat = target_mat / self.R.shape[0]
    target_mat = 1 - target_mat
    np.fill_diagonal(target_mat, 0)
    return target_mat

  def similarity_matrix(self, source_mat, target_mat):
    # Load important things ----
    self.get_id_matrix()
    # Create leave matrix ----
    self.linksim_matrix = np.zeros((self.leaves, self.leaves))
    for i in np.arange(self.nodes):
      for j in np.arange(self.nodes):
        if self.id_mat[i, j] == 0: continue
        x = self.id_mat[i, j]
        for k in np.arange(j, self.nodes):
          y = self.id_mat[i, k]
          if k == j or y == 0: continue
          self.linksim_matrix[x -1, y - 1] = target_mat[j, k]
        for k in np.arange(i, self.nodes):
          y = self.id_mat[k,j]
          if k == i or y == 0 : continue
          self.linksim_matrix[x - 1, y - 1] = source_mat[i, k]
    self.linksim_matrix = self.linksim_matrix + self.linksim_matrix.T

  def similarity_by_feature(self):
    if "TARGET" in self.topology:
      if "DTW" in self.topology: mat = self.dtw_target_similarity()
      else: mat = self.target_similarity(self.index)
      self.similarity_matrix(mat, mat)
      self.source_sim_matrix = mat
      self.target_sim_matrix = mat
    elif "SOURCE" in self.topology:
      if "DTW" in self.topology: mat1 = self.dtw_source_similarity()
      else: mat1 = self.source_similarity(self.index)
      self.similarity_matrix(mat1, mat1)
      self.source_sim_matrix = mat1
      self.target_sim_matrix = mat1
    elif "MIX" in self.topology:
      if "DTW" in self.topology:
        mat1 = self.dtw_source_similarity()
        mat2 = self.dtw_target_similarity()
      else:
        mat1 = self.source_similarity(self.index)
        mat2 = self.target_similarity(self.index)
      self.similarity_matrix(mat1, mat2)
      self.sisource_sim_matrixm1 = mat1
      self.target_sim_matrix = mat2
    else:
      raise RuntimeError("Not compatible direction.")
  
  def similarity_by_feature_cpp(self):
    Quest = squest(
      self.A, self.get_aki(), self.get_aik(), self.nodes,
      self.leaves, self.topologies[self.topology],
      self.indices[self.index]
    )
    self.linksim_matrix = np.array(Quest.get_linksim_matrix())
    self.source_sim_matrix = np.array(Quest.get_source_matrix())
    self.target_sim_matrix = np.array(Quest.get_target_matrix())

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
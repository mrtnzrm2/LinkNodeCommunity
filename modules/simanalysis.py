import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
# For unique combinations
from itertools import chain, repeat, count, islice
from collections import Counter
# Personal libs ----
from various.data_transformations import maps
from various.similarity_indices import *

class Sim:
  def __init__(
    self, n, A, D, mode, nlog10=True,
    prob=True, lookup=False,
    mapping="R1", index="jacp", topology="MIX",
    b=1e-5, **kwargs
  ):
    # Parameters ----
    self.nodes = n
    self.mode = mode
    self.A = A
    self.nonzero = (A != 0)
    self.D = D
    self.nlog10 = nlog10
    self.mapping = mapping
    self.topology = topology
    self.index = sims[index]
    # Treat sln ----
    if "sln" in kwargs.keys():
      self.sln = kwargs["sln"]
      min_sln = np.min(self.sln[self.sln > 0])
      self.sln[self.sln == 0] = min_sln / 10
    # Map data ----
    if mapping != "R1" and mapping != "R2" and nlog10:
      raise ValueError("\n  Log transformation of data, but mapping not found.\n")
    self.R, self.lup, self.shift = maps[mapping](A, nlog10, lookup, prob, b=b)
    # Get number of rows ----
    self.rows = self.A.shape[0]
    # Number of connections in the EC component ----
    self.leaves = np.sum(self.A[:n, :n] != 0)

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

  # def paper_source_b(self):
  #   B = self.R.copy()
  #   B[B != 0] = 1
  #   nodes = B.shape[1]
  #   np.fill_diagonal(B, 0)
  #   dist = np.zeros((self.nodes, self.nodes)) * np.nan
  #   for i in np.arange(self.nodes):
  #     for j in np.arange(i+1, self.nodes):
  #       sim = np.sum(B[i, :] == B[j, :]) / nodes
  #       pi = np.sum(B[i, :]) / nodes
  #       pj = np.sum(B[j, :]) / nodes
  #       sim -=  1 - pj - pi + 2*pi*pj
  #       dist[i, j] = sim
  #       dist[j, i] = sim
  #   np.fill_diagonal(dist, 0)
  #   return dist

  # def paper_target_b(self):
  #   B = self.R.copy()
  #   B[B != 0] = 1
  #   np.fill_diagonal(B, 0)
  #   nodes = B.shape[0]
  #   dist = np.zeros((B.shape[1], B.shape[1])) * np.nan
  #   for i in np.arange(B.shape[1]):
  #     for j in np.arange(i+1, B.shape[1]):
  #       sim = np.sum(B[:, i] == B[:, j]) / nodes
  #       pi = np.sum(B[:, i]) / nodes
  #       pj = np.sum(B[:, j]) / nodes
  #       sim -=  1 - pj - pi + 2*pi*pj
  #       dist[i, j] = sim
  #       dist[j, i] = sim
  #   np.fill_diagonal(dist, 0)
  #   return dist

  def similarity_matrix(self, source_mat, target_mat):
    # Load important things ----
    self.get_id_matrix()
    # Create leave matrix ----
    self.sim_mat = np.zeros((self.leaves, self.leaves))
    for i in np.arange(self.nodes):
      for j in np.arange(self.nodes):
        if self.id_mat[i, j] == 0: continue
        x = self.id_mat[i, j]
        for k in np.arange(j, self.nodes):
          y = self.id_mat[i, k]
          if k == j or y == 0: continue
          self.sim_mat[x -1, y - 1] = target_mat[j, k]
        for k in np.arange(i, self.nodes):
          y = self.id_mat[k,j]
          if k == i or y == 0 : continue
          self.sim_mat[x - 1, y - 1] = source_mat[i, k]
    self.sim_mat = self.sim_mat + self.sim_mat.T

  def similarity_matrix_motif_5(self, source_mat, target_mat):
    # Load important things ----
    self.get_id_matrix()
    # Create leave matrix ----
    self.sim_mat = np.zeros((self.leaves, self.leaves))
    for i in np.arange(self.nodes):
      for j in np.arange(self.nodes):
        if self.id_mat[i, j] == 0: continue
        x = self.id_mat[i, j]
        for k in np.arange(j, self.nodes):
          y = self.id_mat[i, k]
          if k == j or y == 0: continue
          dir_jk = (self.id_mat[j, k] != 0)
          dir_kj = (self.id_mat[k, j] != 0)
          if (dir_jk and ~dir_kj) or (~dir_jk and dir_kj):
            self.sim_mat[x -1, y - 1] = target_mat[j, k]
        for k in np.arange(i, self.nodes):
          y = self.id_mat[k,j]
          if k == i or y == 0 : continue
          dir_ik = (self.id_mat[i, k] != 0)
          dir_ki = (self.id_mat[k, i] != 0)
          if (dir_ik and ~dir_ki) or (~dir_ik and dir_ki):
            self.sim_mat[x - 1, y - 1] = source_mat[i, k]
    self.sim_mat = self.sim_mat + self.sim_mat.T

  def similarity_matrix_motif_6(self, source_mat, target_mat):
    # Load important things ----
    self.get_id_matrix()
    # Create leave matrix ----
    self.sim_mat = np.zeros((self.leaves, self.leaves))
    for i in np.arange(self.nodes):
      for j in np.arange(self.nodes):
        if self.id_mat[i, j] == 0: continue
        x = self.id_mat[i, j]
        for k in np.arange(j, self.nodes):
          y = self.id_mat[i, k]
          if k == j or y == 0: continue
          dir_jk = (self.id_mat[j, k] != 0)
          dir_kj = (self.id_mat[k, j] != 0)
          if dir_jk and dir_kj:
            self.sim_mat[x -1, y - 1] = target_mat[j, k]
        for k in np.arange(i, self.nodes):
          y = self.id_mat[k,j]
          if k == i or y == 0 : continue
          dir_ik = (self.id_mat[i, k] != 0)
          dir_ki = (self.id_mat[k, i] != 0)
          if dir_ik and dir_ki:
            self.sim_mat[x - 1, y - 1] = source_mat[i, k]
    self.sim_mat = self.sim_mat + self.sim_mat.T

  def similarity_matrix_motif_9(self, source_mat, target_mat):
    # Load important things ----
    self.get_id_matrix()
    # Create leave matrix ----
    self.sim_mat = np.zeros((self.leaves, self.leaves))
    for i in np.arange(self.nodes):
      for j in np.arange(self.nodes):
        if self.id_mat[i, j] == 0: continue
        x = self.id_mat[i, j]
        for k in np.arange(i, self.nodes):
          y = self.id_mat[k, i]
          if k == j or y == 0: continue
          dir = (self.id_mat[j, k] != 0)
          if dir:
            self.sim_mat[x -1, y - 1] = target_mat[j, k]
        for k in np.arange(j, self.nodes):
          y = self.id_mat[j, k]
          if k == i or y == 0 : continue
          dir = (self.id_mat[k, i] != 0)
          if dir:
            self.sim_mat[x - 1, y - 1] = source_mat[i, k]
    self.sim_mat = self.sim_mat + self.sim_mat.T

  def sln_out_similarity(self, i, j, k, aik_i, aik_j, f=jacp):
    if self.sln[i, k] >= 0.5 and self.sln[j, k] >= 0.5:
      hood = (self.sln[i, :] >= 0.5) & (self.sln[j, :] >= 0.5)
      n = np.sum(hood)
      return f(aik_i[hood], aik_j[hood], n, self.lup)
    elif self.sln[i, k] < 0.5 and self.sln[j, k] >= 0.5:
      hood = (self.sln[i, :] < 0.5) & (self.sln[j, :] >= 0.5)
      n = np.sum(hood)
      return f(aik_i[hood], aik_j[hood], n, self.lup)
    elif self.sln[i, k] >= 0.5 and self.sln[j, k] < 0.5:
      hood = (self.sln[i, :] >= 0.5) & (self.sln[j, :] < 0.5)
      n = np.sum(hood)
      return f(aik_i[hood], aik_j[hood], n, self.lup)
    elif self.sln[i, k] < 0.5 and self.sln[j, k] < 0.5:
      hood = (self.sln[i, :] < 0.5) & (self.sln[j, :] < 0.5)
      n = np.sum(hood)
      return f(aik_i[hood], aik_j[hood], n, self.lup)
    else: 
      raise Exception("Compared links have strange sln values")

  def sln_in_similarity(self, i, j, k, aki_i, aki_j, f=jacp):
    if self.sln[k, i] >= 0.5 and self.sln[k, j] >= 0.5:
      hood = (self.sln[:, i] >= 0.5) & (self.sln[:, j] >= 0.5)
      n = np.sum(hood)
      return f(aki_i[hood], aki_j[hood], n, self.lup)
    elif self.sln[k, i] < 0.5 and self.sln[k, j] >= 0.5:
      hood = (self.sln[:, i] < 0.5) & (self.sln[:, j] >= 0.5)
      n = np.sum(hood)
      return f(aki_i[hood], aki_j[hood], n, self.lup)
    elif self.sln[k, i] >= 0.5 and self.sln[k, j] < 0.5:
      hood = (self.sln[:, i] >= 0.5) & (self.sln[:, j] < 0.5)
      n = np.sum(hood)
      return f(aki_i[hood], aki_j[hood], n, self.lup)
    elif self.sln[k, i] < 0.5 and self.sln[k, j] < 0.5:
      hood = (self.sln[:, i] < 0.5) & (self.sln[:, j] < 0.5)
      n = np.sum(hood)
      return f(aki_i[hood], aki_j[hood], n, self.lup)
    else:
      raise Exception("Compared links have strange sln values")
  
  def similarity_matrix_SLN(self):
    # Load important things ----
    aik = self.get_aik()
    aki = self.get_aki()
    self.get_id_matrix()
    # Create leave matrix ----
    self.sim_mat = np.zeros((self.leaves, self.leaves))
    for i in np.arange(self.nodes):
      for j in np.arange(self.nodes):
        if self.id_mat[i, j] == 0: continue
        x = self.id_mat[i, j]
        for k in np.arange(j, self.nodes):
          y = self.id_mat[i, k]
          if k == j or y == 0: continue
          self.sim_mat[x -1, y - 1] = self.sln_in_similarity(j, k, i, aki[j, :], aki[k, :])
        for k in np.arange(i, self.nodes):
          y = self.id_mat[k, j]
          if k == i or y == 0 : continue
          self.sim_mat[x - 1, y - 1] = self.sln_out_similarity(i, k, j, aik[i, :], aik[k, :])
    self.sim_mat = self.sim_mat + self.sim_mat.T
    # if np.sum((self.sim_mat > 1) | (self.sim_mat < 0)) > 0:
    #   raise ValueError("Similarity can not be greater than one")

  def parallel_similarity(self, i, k, j, aik, akj, f=jacp):
    if self.sln[i, k] >= 0.5 and self.sln[k, j] >= 0.5:
      hood = (self.sln[i, :] >= 0.5) & (self.sln[:, j][:self.nodes] >= 0.5)
      n = np.sum(hood)
      return f(aik[hood], akj[:self.nodes][hood], n, self.lup)
    if self.sln[i, k] >= 0.5 and self.sln[k, j] < 0.5:
      hood = (self.sln[i, :] >= 0.5) & (self.sln[:, j][:self.nodes] < 0.5)
      n = np.sum(hood)
      return f(aik[hood], akj[:self.nodes][hood], n, self.lup)
    if self.sln[i, k] < 0.5 and self.sln[k, j] >= 0.5:
      hood = (self.sln[i, :] < 0.5) & (self.sln[:, j][:self.nodes] >= 0.5)
      n = np.sum(hood)
      return f(aik[hood], akj[:self.nodes][hood], n, self.lup)
    if self.sln[i, k] < 0.5 and self.sln[k, j] < 0.5:
      hood = (self.sln[i, :] < 0.5) & (self.sln[:, j][:self.nodes] < 0.5)
      n = np.sum(hood)
      return f(aik[hood], akj[:self.nodes][hood], n, self.lup)
    else: 
      raise Exception("Compared links have strange sln values")

  def similarity_matrix_SLN_2(self):
    # Load important things ----
    aik = self.get_aik()
    aki = self.get_aki()
    self.get_id_matrix()
    # Create leave matrix ----
    self.sim_mat = np.zeros((self.leaves, self.leaves))
    for i in np.arange(self.nodes):
      for j in np.arange(self.nodes):
        if self.id_mat[i, j] == 0: continue
        x = self.id_mat[i, j]
        for k in np.arange(j, self.nodes):
          y = self.id_mat[i, k]
          if k == j or y == 0: continue
          self.sim_mat[x -1, y - 1] = self.sln_in_similarity(j, k, i, aki[j, :], aki[k, :])
          self.sim_mat[y - 1, x - 1] = self.sim_mat[x -1, y - 1]
        for k in np.arange(i, self.nodes):
          y = self.id_mat[k, j]
          if k == i or y == 0 : continue
          self.sim_mat[x - 1, y - 1] = self.sln_out_similarity(i, k, j, aik[i, :], aik[k, :])
          self.sim_mat[y - 1, x - 1] = self.sim_mat[x -1, y - 1]
    for k in np.arange(self.nodes):
      availabe_nodes = np.arange(self.nodes)
      availabe_nodes = availabe_nodes[availabe_nodes != k]
      choices = list(unique_combinations(availabe_nodes, 2))
      for i, j in choices:
        ## right ->
        u = self.id_mat[i, k]
        v = self.id_mat[k, j]
        if u != 0 and v != 0:
          self.sim_mat[u - 1, v - 1] = self.parallel_similarity(i, k, j, aik[i, :], aki[j, :])
          self.sim_mat[v - 1, u - 1] = self.sim_mat[u - 1, v - 1]
        ## left <-
        u = self.id_mat[j, k]
        v = self.id_mat[k, i]
        if u != 0 and v != 0:
          self.sim_mat[u - 1, v - 1] = self.parallel_similarity(j, k, i, aik[j, :], aki[i, :])
          self.sim_mat[v - 1, u - 1] = self.sim_mat[u - 1, v - 1]

  def similarity_by_feature(self):
    if "TARGET" in self.topology:
      if "DTW" in self.topology: mat = self.dtw_target_similarity()
      else: mat = self.target_similarity(self.index, self.shift)
      self.similarity_matrix(mat, mat)
      self.sim1 = mat
      self.sim2 = mat
    elif "SOURCE" in self.topology:
      if "DTW" in self.topology: mat1 = self.dtw_source_similarity()
      else: mat1 = self.source_similarity(self.index, self.shift)
      self.similarity_matrix(mat1, mat1)
      self.sim1 = mat1
      self.sim2 = mat1
    elif "MIX" in self.topology:
      if "DTW" in self.topology:
        mat1 = self.dtw_source_similarity()
        mat2 = self.dtw_target_similarity()
      else:
        mat1 = self.source_similarity(self.index, self.shift)
        mat2 = self.target_similarity(self.index, self.shift)
      self.similarity_matrix(mat1, mat2)
      self.sim1 = mat1
      self.sim2 = mat2
    elif self.topology == "SLN":
      self.sim1 = np.array([0])
      self.sim2 = np.array([0])
      self.similarity_matrix_SLN()
    elif self.topology == "SLN2":
      """w/Parallel similarities"""
      self.sim1 = np.array([0])
      self.sim2 = np.array([0])
      self.similarity_matrix_SLN_2()
    else:
      raise RuntimeError("Not compatible direction.")

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
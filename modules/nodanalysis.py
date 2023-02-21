import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
# Personal libs ----
from modules.simanalysis import Sim
from various.network_tools import adj2df
from various.similarity_indices import *

class NODA(Sim):
  def __init__(self, nodes : int, NET, R, lookup=0, **kwargs):
    super().__init__(
      nodes, NET.A, R, NET.D, NET.mode,
      topology=NET.topology, index=NET.index,
      lookup=lookup, **kwargs
    )
    self.dA = adj2df(self.A)
    self.dA = self.dA.loc[self.dA.weight != 0]
    AA = self.A.copy()
    np.fill_diagonal(AA, np.nan)
    self.W1 = AA.copy()
    self.W2 = AA.copy()
    self.W1[self.W1 != 0] = np.log(self.W1[self.W1 != 0])
    self.W2[self.W2 != 0] = np.log10(self.W2[self.W2 != 0]) 
    self.W2[self.W2 != 0] = self.W2[self.W2 != 0] + np.ceil(np.abs(np.nanmin(self.W2[self.W2 != 0])))
    if lookup:
      m1 = np.nanmin(self.W1[self.W1 != 0])
      m2 = np.nanmin(self.W2[self.W2 != 0])
      self.m1 = m1 - 1
      self.W1[self.W1 == 0] = self.m1
      self.W2[self.W2 == 0] = m2 / 2
    self.D = NET.D.copy()
    self.inj = int(NET.inj)
    self.struct_labels = NET.struct_labels
    # Plotting ----
    self.plot_path = NET.plot_path

  def to_open_unit(self, A):
    min_value = np.nanmin(A)
    max_value = np.nanmax(A)
    m = 0.99 / (max_value - min_value)
    A = (A - min_value) * m + 1e-3
    return A

  def to_logit(self, A):
    B = A.copy()
    B = 1 / (1 + np.exp(-B))
    return B

  def to_arctan(self, A):
    B = A.copy()
    B = 0.5 + np.arctan(B) / np.pi
    return B

  def to_distance(self, A):
    from sklearn.linear_model import LinearRegression
    m, n = A.shape
    AA = A.copy()
    D = self.D[:m, :n]
    e = (AA != 0) & (~np.isnan(AA))
    x = D[e].copy().reshape(-1, 1)
    y = AA[e].copy().ravel()
    model = LinearRegression().fit(x, y)
    n_e = AA == 0
    AA[n_e] = model.predict(D[n_e].reshape(-1, 1))
    return AA

  def jacp_source(self):
    W = self.W1.copy()
    W[W != 0] = -W[W != 0]
    if self.lup: max_W = self.m1
    else: max_W = 0
    distance = np.zeros((self.nodes, self.nodes)) * np.nan
    for i in np.arange(self.nodes):
      for j in np.arange(i + 1, self.nodes):
        w_i = W[i, :]
        w_j = W[j, :]
        if i < W.shape[1]: w_i[i] = np.nanmean(w_i[w_i != max_W])
        if j < W.shape[1]: w_j[j] = np.nanmean(w_j[w_j != max_W])
        distance[i, j] = jacp(w_i, w_j, W.shape[1], self.lup)
        distance[j, i] = distance[i, j]
    if np.sum(distance <= 0) > 0:
      distance = self.to_distance(distance)
    if np.sum(distance >= 1) > 0:
      distance[distance == 1] = np.power(np.nanmax(distance[distance < 1]), 1/2)
    distance = 1 - distance
    np.fill_diagonal(distance, 0)
    return distance

  def jacp_target(self):
    W = self.W1.copy()
    W[W != 0] = -W[W != 0]
    if self.lup: max_W = self.m1
    else: max_W = 0
    distance = np.zeros((W.shape[1], W.shape[1])) * np.nan
    for i in np.arange(W.shape[1]):
      for j in np.arange(i + 1, W.shape[1]):
        w_i = W[:, i].ravel()
        w_j = W[:, j].ravel()
        w_i[i] = np.nanmean(w_i[w_i != max_W])
        w_j[j] = np.nanmean(w_j[w_j != max_W])
        distance[i, j] = jacp(w_i, w_j, W.shape[0], self.lup)
        distance[j, i] = distance[i, j]
    if np.sum(distance <= 0) > 0:
      distance = self.to_distance(distance)
    if np.sum(distance >= 1) > 0:
      distance[distance == 1] = np.sqrt(np.nanmax(distance[distance < 1]))
    distance = 1 - distance
    np.fill_diagonal(distance, 0)
    return distance

  def jacw_source(self):
    W = self.W2.copy()
    distance = np.zeros((self.nodes, self.nodes)) * np.nan
    for i in np.arange(self.nodes):
      for j in np.arange(i + 1, self.nodes):
        w_i = W[i, :]
        w_j = W[j, :]
        if i < W.shape[1]: w_i[i] = np.nanmean(w_i[w_i != 0])
        if j < W.shape[1]: w_j[j] = np.nanmean(w_j[w_j != 0])
        distance[i, j] = jacw(w_i, w_j, W.shape[1], self.lup)
        distance[j, i] = distance[i, j]
    # if np.sum(distance <= 0) > 0:
    #   distance = self.to_distance(distance)
    # if np.sum(distance >= 1) > 0:
    #   distance[distance == 1] = np.power(np.nanmax(distance[distance < 1]), 1/2)
    distance = 1 - distance
    np.fill_diagonal(distance, 0)
    return distance

  def jacw_target(self):
    W = self.W2.copy()
    distance = np.zeros((W.shape[1], W.shape[1])) * np.nan
    for i in np.arange(W.shape[1]):
      for j in np.arange(i + 1, W.shape[1]):
        w_i = W[:, i].ravel()
        w_j = W[:, j].ravel()
        w_i[i] = np.nanmean(w_i[w_i != 0])
        w_j[j] = np.nanmean(w_j[w_j != 0])
        distance[i, j] = jacw(w_i, w_j, W.shape[0], self.lup)
        distance[j, i] = distance[i, j]
    # if np.sum(distance <= 0) > 0:
    #   distance = self.to_distance(distance)
    # if np.sum(distance >= 1) > 0:
    #   distance[distance == 1] = np.sqrt(np.nanmax(distance[distance < 1]))
    distance = 1 - distance
    np.fill_diagonal(distance, 0)
    return distance

  def dtw_source(self):
    W = self.W1.copy()
    W[W != 0] = -W[W != 0]
    D = self.D.copy()[:, :self.inj]
    dtw_distance = np.zeros((self.nodes, self.nodes))
    for i in np.arange(self.nodes):
      for j in np.arange(i + 1, self.nodes):
        seci = (W[i, :] != 0) & (~np.isnan(W[i, :]))
        secj = (W[j, :] != 0) & (~np.isnan(W[j, :]))
        i_array = np.zeros((np.sum(seci), 2))
        j_array = np.zeros((np.sum(secj), 2))
        i_array[:, 0] = D[i, seci]
        i_array[:, 1] = W[i, seci]
        j_array[:, 0] = D[j, secj]
        j_array[:, 1] = W[j, secj]
        i_array = i_array[np.argsort(i_array[:, 0]) ,:]
        j_array = j_array[np.argsort(j_array[:, 0]) ,:]
        dtw, _ = fastdtw(i_array, j_array, dist=euclidean, radius=10)
        dtw_distance[i, j] = dtw
        dtw_distance[j, i] = dtw
    # dtw_distance = (dtw_distance - np.nanmean(dtw_distance)) / np.nanstd(dtw_distance)
    # dtw_distance = self.to_arctan(dtw_distance)
    return dtw_distance

  def dtw_target(self):
    W = self.W1.copy()
    W[W != 0] = -W[W != 0]
    D = self.D.copy()[:, :self.inj]
    dtw_distance = np.zeros((W.shape[1], W.shape[1]))
    for i in np.arange(W.shape[1]):
      for j in np.arange(i + 1, W.shape[1]):
        seci = (W[:, i] != 0) & (~np.isnan(W[:, i]))
        secj = (W[:, j] != 0) & (~np.isnan(W[:, j]))
        i_array = np.zeros((np.sum(seci), 2))
        j_array = np.zeros((np.sum(secj), 2))
        i_array[:, 0] = D[seci, i]
        i_array[:, 1] = W[seci, i]
        j_array[:, 0] = D[secj, j]
        j_array[:, 1] = W[secj, j]
        i_array = i_array[np.argsort(i_array[:, 0]) ,:]
        j_array = j_array[np.argsort(j_array[:, 0]) ,:]
        dtw, _ = fastdtw(i_array, j_array, dist=euclidean, radius=10)
        dtw_distance[i, j] = dtw
        dtw_distance[j, i] = dtw
    # dtw_distance = (dtw_distance - np.nanmean(dtw_distance)) / np.nanstd(dtw_distance)
    # dtw_distance = self.to_arctan(dtw_distance)
    return dtw_distance

  def dtw_b_source(self):
    W = self.W1.copy()
    D = self.D.copy()[:, :self.inj]
    W[W != 0] = 1
    dtw_distance = np.zeros((W.shape[0], W.shape[0]))
    for i in np.arange(W.shape[0]):
      for j in np.arange(i + 1, W.shape[0]):
        i_array = np.zeros((W.shape[1], 2))
        j_array = np.zeros((W.shape[1], 2))
        i_array[:, 0] = D[i, :]
        i_array[:, 1] = W[i, :]
        j_array[:, 0] = D[j, :]
        j_array[:, 1] = W[j, :]
        dtw, _ = fastdtw(i_array, j_array, dist=euclidean, radius=10)
        dtw_distance[i, j] = dtw
        dtw_distance[j, i] = dtw
    return dtw_distance

  def dtw_both(self):
    W = self.W1.copy()
    W[W != 0] = -W[W != 0]
    D = self.D.copy()[:, :self.inj]
    dtw_vector = np.zeros(W.shape[1])
    for i in np.arange(W.shape[1]):
      secs = (W[i, :] != 0) & (~np.isnan(W[i, :]))
      sect = (W[:W.shape[1], i] != 0) & (~np.isnan(W[:W.shape[1], i]))
      src = np.zeros((np.sum(secs), 2))
      tgt = np.zeros((np.sum(sect), 2))
      src[:, 0] = D[i, secs]
      src[:, 1] = W[i, secs]
      tgt[:, 0] = D[:W.shape[1], i][sect]
      tgt[:, 1] = W[:W.shape[1], i][sect]
      src = src[np.argsort(src[:, 0]) ,:]
      tgt = tgt[np.argsort(tgt[:, 0]) ,:]
      dtw, _ = fastdtw(src, tgt, dist=euclidean, radius=10)
      dtw_vector[i] = dtw
    return dtw_vector

  def dtw_b_both(self):
    W = self.W1.copy()
    W[W != 0] = 1
    D = self.D.copy()[:, :self.inj]
    dtw_vector = np.zeros(W.shape[1])
    for i in np.arange(W.shape[1]):
      src = np.zeros((W.shape[1], 2))
      tgt = np.zeros((W.shape[1], 2))
      src[:, 0] = D[i, :]
      src[:, 1] = W[i, :]
      tgt[:, 0] = D[:W.shape[1], i]
      tgt[:, 1] = W[:W.shape[1], i]
      src = src[np.argsort(src[:, 0]) ,:]
      tgt = tgt[np.argsort(tgt[:, 0]) ,:]
      dtw, _ = fastdtw(src, tgt, dist=euclidean, radius=10)
      dtw_vector[i] = dtw
    return dtw_vector

  def paper_source_b(self):
    B = self.W1.copy()
    B[B != 0] = 1
    nodes = B.shape[1]
    np.fill_diagonal(B, 0)
    dist = np.zeros((self.nodes, self.nodes)) * np.nan
    for i in np.arange(self.nodes):
      for j in np.arange(i+1, self.nodes):
        sim = np.sum(B[i, :] == B[j, :]) / nodes
        pi = np.sum(B[i, :]) / nodes
        pj = np.sum(B[j, :]) / nodes
        sim -=  1 - pj - pi + 2*pi*pj
        dist[i, j] = sim
        dist[j, i] = sim
    if np.sum(dist == 0) > 0 and np.sum(dist < 0) == 0:
      dist = self.to_distance(dist)
    elif np.sum(dist == 0) > 0 and np.sum(dist < 0) > 0:
      dist = self.to_distance(dist)
    #   dist = self.to_arctan(dist)
    # elif np.sum(dist == 0) == 0 and np.sum(dist < 0) > 0:
    #   dist = self.to_arctan(dist)
    # dist = dist - np.nanmin(dist) + np.power(np.nanmin(dist[dist != np.nanmin(dist)]), 2)
    dist = 1 - dist
    np.fill_diagonal(dist, 0)
    return dist

  def paper_target_b(self):
    B = self.W1.copy()
    B[B != 0] = 1
    np.fill_diagonal(B, 0)
    nodes = B.shape[0]
    dist = np.zeros((B.shape[1], B.shape[1])) * np.nan
    for i in np.arange(B.shape[1]):
      for j in np.arange(i+1, B.shape[1]):
        sim = np.sum(B[:, i] == B[:, j]) / nodes
        pi = np.sum(B[:, i]) / nodes
        pj = np.sum(B[:, j]) / nodes
        sim -=  1 - pj - pi + 2*pi*pj
        dist[i, j] = sim
        dist[j, i] = sim
    if np.sum(dist == 0) > 0 and np.sum(dist < 0) == 0:
      dist = self.to_distance(dist)
    elif np.sum(dist == 0) > 0 and np.sum(dist < 0) > 0:
      dist = self.to_distance(dist)
    #   dist = self.to_arctan(dist)
    # elif np.sum(dist == 0) == 0 and np.sum(dist < 0) > 0:
    #   dist = self.to_arctan(dist)
    # dist = dist - np.nanmin(dist) + np.power(np.nanmin(dist[dist != np.nanmin(dist)]), 2)
    dist = 1 - dist
    np.fill_diagonal(dist, 0)
    return dist

  def cossim_source(self):
    W = self.W1.copy()
    W[W != 0] = -W[W != 0]
    if self.lup: max_W = self.m1
    else: max_W = 0
    dist = np.zeros((self.nodes, self.nodes)) * np.nan
    for i in np.arange(self.nodes):
      for j in np.arange(i+1, self.nodes):
        w_i = W[i, :]
        w_j = W[j, :]
        if i < W.shape[1]: w_i[i] = np.nanmean(w_i[w_i != max_W])
        if j < W.shape[1]: w_j[j] = np.nanmean(w_j[w_j != max_W])
        dist[i, j] = np.dot(w_i, w_j) / (
          np.sqrt(
            np.dot(w_i, w_i)
          ) * np.sqrt(
            np.dot(w_j, w_j)
          )
        )
        dist[j, i] = dist[i, j]
    if np.sum(dist == 0) > 0 and np.sum(dist < 0) == 0:
      dist = self.to_distance(dist)
    elif np.sum(dist == 0) > 0 and np.sum(dist < 0) > 0:
      dist = self.to_distance(dist)
    #   dist = self.to_arctan(dist)
    # elif np.sum(dist == 0) == 0 and np.sum(dist < 0) > 0:
    #   dist = self.to_arctan(dist)
    dist = 1 - dist
    np.fill_diagonal(dist, 0)
    return dist

  def cossim_target(self):
    W = self.W1.copy()
    W[W != 0] = -W[W != 0]
    if self.lup: max_W = self.m1
    else: max_W = 0
    dist = np.zeros((W.shape[1], W.shape[1])) * np.nan
    for i in np.arange(W.shape[1]):
      for j in np.arange(i+1, W.shape[1]):
        w_i = W[:, i].ravel()
        w_j = W[:, j].ravel()
        w_i[i] = np.nanmean(w_i[w_i != max_W])
        w_j[j] = np.nanmean(w_j[w_j != max_W])
        dist[i, j] = np.dot(w_i, w_j) / (
          np.sqrt(np.dot(w_i, w_i)) * np.sqrt(np.dot(w_j, w_j))
        )
        dist[j, i] = dist[i, j]
    if np.sum(dist == 0) > 0 and np.sum(dist < 0) == 0:
      dist = self.to_distance(dist)
    elif np.sum(dist == 0) > 0 and np.sum(dist < 0) > 0:
      dist = self.to_distance(dist)
    #   dist = self.to_arctan(dist)
    # elif np.sum(dist == 0) == 0 and np.sum(dist < 0) > 0:
    #   dist = self.to_arctan(dist)
    dist = 1 - dist
    np.fill_diagonal(dist, 0)
    return dist
  
  def select_feature(self, feature="jacp_source"):
    if feature == "jacp_target":
      return self.jacp_target()
    elif feature == "dtw_source":
      return self.dtw_source()
    elif feature == "dtw_target":
      return self.dtw_target()
    elif feature == "dtw_b_source":
      return  self.dtw_b_source()
    elif feature == "jacp_source":
      return self.jacp_source()
    elif feature == "paper_source_b":
      return self.paper_source_b()
    elif feature == "paper_target_b":
      return self.paper_target_b()
    elif feature == "cossim_source":
      return self.cossim_source()
    elif feature == "cossim_target":
      return self.cossim_target()
    elif feature == "dtw_both":
      return self.dtw_both()
    elif feature == "jacw_target":
      return self.jacw_target()
    elif feature == "jacw_source":
      return self.jacw_source()
    else:
      raise RuntimeError("Not compatible feature.")
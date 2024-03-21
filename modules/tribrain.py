import numpy as np
import numpy.typing as npt
import pandas as pd
from collections import Counter

from various.network_tools import match

class triPred :
  def __init__(
      self, distance_matrix : npt.ArrayLike, count_matrix : npt.ArrayLike,
      nearest_neighbors :dict, node_labels : npt.ArrayLike, known_node_labels : npt.ArrayLike,
      dist_bins=20
    ) -> None:

    self.distance_matrix = distance_matrix
    self.count_matrix = count_matrix
    self.nearest_neighbors = nearest_neighbors
    self.node_labels = node_labels
    self.known_node_labels = known_node_labels
    self.dist_bins = dist_bins
    
    self.total_nb_nodes = node_labels.shape[0]
    self.nb_known_nodes = self.known_node_labels.shape[0]
    self.know_node_indices = match(self.known_node_labels, self.node_labels)

    self.distance_bin_index_range = np.arange(self.dist_bins)
    self.total_nodes_index_range = np.arange(self.total_nb_nodes)

  def nn2indices_nn(self) -> None:
    self.node_indices_nn = dict()
    for area, nn in self.nearest_neighbors.items():
      self.node_indices_nn[match([area], self.node_labels)[0]] = match(nn, self.node_labels)
  
  def make_boundaries(self) -> None:
    Dmin = np.min(self.distance_matrix[self.distance_matrix > 0])
    Dmax = np.max(self.distance_matrix)

    self.distance_bin_boundaries = np.linspace(Dmin, Dmax, self.dist_bins + 1)
    self.distance_bin_centers = self.distance_bin_boundaries.copy()
    self.distance_bin_boundaries[-1] += 1e-6

    delta = self.distance_bin_centers[1] - self.distance_bin_centers[0]
    self.distance_bin_centers = self.distance_bin_centers[:-1] + delta / 2

  def setup_target_distance_distributions(self) -> None:
    self.target_distance_distributions = np.zeros((self.total_nb_nodes, self.dist_bins))
  
  def calculate_known_target_distance_distributions(self) -> None:
    for known_node_index in self.know_node_indices:
      known_node_total_count = np.sum(self.count_matrix[:, known_node_index])
      for j_dist_entrance in self.total_nodes_index_range:
        dj = self.distance_matrix[j_dist_entrance, known_node_index]
        cj = self.count_matrix[j_dist_entrance, known_node_index]
        for k_bin_boundary in self.distance_bin_index_range:
          if dj >= self.distance_bin_boundaries[k_bin_boundary] and dj < self.distance_bin_boundaries[k_bin_boundary + 1]:
            self.target_distance_distributions[known_node_index, k_bin_boundary] += cj
            break
      self.target_distance_distributions[known_node_index] /= known_node_total_count

      self.thr_prob = np.min(self.target_distance_distributions[known_node_index][self.target_distance_distributions[known_node_index] > 0])

  def estimate_target_probability_distribution(self, target_node : int, known_nodes_nn : npt.ArrayLike, samples=100000) -> npt.DTypeLike:
    target_inferred_distance_distribution= np.zeros(self.dist_bins)
    
    ### WEIGHTED BY DISTANCE OR UNIFORM SAMPLING
    nearest_neighbors_distaces_to_target = np.exp(-self.distance_matrix[target_node, known_nodes_nn])
    weights = nearest_neighbors_distaces_to_target / np.sum(nearest_neighbors_distaces_to_target)

    for i, known_node in enumerate(known_nodes_nn):
      target_inferred_distance_distribution += weights[i] * np.log(self.target_distance_distributions[known_node, :] + 1e-7)

    target_inferred_distance_distribution = np.exp(target_inferred_distance_distribution)

    target_inferred_distance_distribution /= np.sum(target_inferred_distance_distribution)

    target_inferred_distance_distribution[target_inferred_distance_distribution < self.thr_prob] = 0

    target_inferred_distance_distribution /= np.sum(target_inferred_distance_distribution)

    # nearest_neighbor_draw = np.random.choice(known_nodes_nn, samples, p=weights)
    # count_draw = Counter(nearest_neighbor_draw)

    # for known_node, nb_draws in count_draw.items():
    #     distance_bin_draw = np.random.choice(self.distance_bin_index_range, nb_draws, p=self.target_distance_distributions[known_node, :])
    #     count_distance_bin_draw = Counter(distance_bin_draw)
    #     for distance_bin_index, nb_distance_draws in count_distance_bin_draw.items():
    #         target_inferred_distance_distribution[distance_bin_index] += nb_distance_draws

    # target_inferred_distance_distribution /= np.sum(target_inferred_distance_distribution)

    return target_inferred_distance_distribution

  def predict_unknown_node_target_distance_distributions(self) -> None:
      predicted_node_list = list(self.know_node_indices)
      unknown_nodes_list = [i for i in self.total_nodes_index_range if i not in self.know_node_indices]

      while len(predicted_node_list) < self.total_nb_nodes:
        save_predictions = dict()
        for unknown_node in unknown_nodes_list:
          unknown_node_nn = self.node_indices_nn[unknown_node]
          known_nn = np.array([i for i in unknown_node_nn if i in predicted_node_list])
          if len(unknown_node_nn) > 2 and len(known_nn) >= 2:
            inferred_unknow_distribution = self.estimate_target_probability_distribution(unknown_node, known_nn)
            save_predictions[unknown_node] = inferred_unknow_distribution
          elif len(known_nn) == 1 and len(unknown_node_nn) == 1:
            inferred_unknow_distribution = self.estimate_target_probability_distribution(unknown_node, known_nn)
            save_predictions[unknown_node] = inferred_unknow_distribution
          elif len(known_nn) == 2 and len(known_nn) == 2: 
            inferred_unknow_distribution = self.estimate_target_probability_distribution(unknown_node, known_nn)
            save_predictions[unknown_node] = inferred_unknow_distribution
        for inferred_node, inferred_distribution in save_predictions.items():
          self.target_distance_distributions[inferred_node] = inferred_distribution
          predicted_node_list.append(inferred_node)
        
        unknown_nodes_list = [i for i in self.total_nodes_index_range if i not in predicted_node_list]
      
      # threshold_prob_distance_distribution = np.min(self.target_distance_distributions[:self.nb_known_nodes][self.target_distance_distributions[:self.nb_known_nodes] > 0])
      # self.target_distance_distributions[self.nb_known_nodes:][self.target_distance_distributions[self.nb_known_nodes:] < threshold_prob_distance_distribution] = 0

  def nodes_pairs_within_distance_bin_in_targets(self) -> None:
    self.target_distance_pairs = dict()
    for j_target_index in self.total_nodes_index_range:
      self.target_distance_pairs[j_target_index] = {}
      for j_distance_index in self.distance_bin_index_range:
        self.target_distance_pairs[j_target_index][j_distance_index] = []
    
    for j in self.total_nodes_index_range:
      for i in self.total_nodes_index_range:
        if i == j: continue
        dij = self.distance_matrix[i, j]
        for j_distance_index in self.distance_bin_index_range:
          if self.distance_bin_boundaries[j_distance_index] <= dij and self.distance_bin_boundaries[j_distance_index + 1] > dij:
            self.target_distance_pairs[j][j_distance_index].append(i)
            break
      
  def network_density(self, A : npt.NDArray) -> int:
    return np.sum(A[:self.nb_known_nodes, :self.nb_known_nodes] > 0) / (self.nb_known_nodes * (self.nb_known_nodes - 1))
  
  def recompute_random_distance(self, target_node : int, samples=60000) -> list:
    return [np.random.choice(self.distance_bin_index_range, samples, p=self.target_distance_distributions[target_node]), 0]
  
  def recompute_random_target(self, samples=60000) -> list:
    return [np.random.choice(self.total_nodes_index_range, samples, replace=True), 0]

  def generate_count_matrix(self, target_density = 0.59, initial_samples=60000, verbose=0, nrows=None) -> npt.NDArray:
    print(">>> Generate count random matrix.")
    self.nodes_pairs_within_distance_bin_in_targets()

    if not nrows:
      generate_count_matrix = np.zeros((self.total_nb_nodes, self.total_nb_nodes))
    elif isinstance(nrows, int):
      generate_count_matrix = np.zeros((nrows, self.total_nb_nodes))
    else:
      raise ValueError("nrows must be an integer.")

    precomputed_random_distance_indices = dict()
    for i in self.total_nodes_index_range:
      random_distances = np.random.choice(self.distance_bin_index_range, initial_samples, p=self.target_distance_distributions[i])
      precomputed_random_distance_indices[i] = [random_distances, 0]

    precomputed_random_targets = [np.random.choice(self.total_nodes_index_range, initial_samples, replace=True), 0]

    current_percentage = 0
    rho = 0 # current network density
    while rho < target_density:
      precomputed_array_position = precomputed_random_targets[1]
      if precomputed_array_position < initial_samples:
        random_target_index = precomputed_random_targets[0][precomputed_array_position]
        preocomputed_distance_array_position = precomputed_random_distance_indices[random_target_index][1]

        if preocomputed_distance_array_position < initial_samples:
          random_distance_index = precomputed_random_distance_indices[random_target_index][0][preocomputed_distance_array_position]
          available_sources = self.target_distance_pairs[random_target_index][random_distance_index]
          nb_sources = len(available_sources)

          if nb_sources > 0:
            random_source_index = np.random.randint(nb_sources)
            generate_count_matrix[available_sources[random_source_index], random_target_index] += 1

          precomputed_random_distance_indices[random_target_index][1] += 1
        else:
          precomputed_random_distance_indices[random_target_index] = self.recompute_random_distance(random_target_index)

        precomputed_random_targets[1] += 1
      else:
        precomputed_random_targets = self.recompute_random_target()

      rho = self.network_density(generate_count_matrix)
      rho_percentage = int((rho / target_density) * 100)
      if verbose > 0:
        if np.mod(rho_percentage, 10) == 0 and current_percentage != rho_percentage:
          print(f">>> {rho_percentage}%")
          current_percentage = rho_percentage
    
    print(">>> Done.")
    # print(np.sum(generate_count_matrix[:, :self.nb_known_nodes]))
    
    return generate_count_matrix
  
  def setup_engine(self) -> None:
    self.nn2indices_nn()
    self.make_boundaries()
    self.setup_target_distance_distributions()

    print(">>> Computing the connection probabilities for known targets.")
    self.calculate_known_target_distance_distributions()
    print(">>> Predicting all target distributions by triangulation.")
    self.predict_unknown_node_target_distance_distributions()
    
    print(">>> Setup complete.")
from networks.edrnet import EDR
import os
import numpy as np

class ELLIPTICNET(EDR):
  def __init__(self, nodes, version, model, mode) -> None:
    super().__init__(nodes, model)
    # Directory details ----
    self.version = "ellipse"
    # Elliptic parameters ----
    self.w = 85 # [mm] width
    self.h = self.get_b() # [mm] height
    self.common_path = os.path.join(
      self.folder, self.version, self.model
    )
    # Create path ----
    self.plot_path = os.path.join(
      "../plots", self.common_path,
      "N_{}".format(self.nodes), str(version),
      mode
    )
    self.csv_path = os.path.join(
      "../CSV", self.common_path,
      "N_{}".format(self.nodes), str(version)
    )
    self.pickle_path = os.path.join(
      "../pickle", self.common_path,
      "N_{}".format(self.nodes), str(version),
      mode
    )

  def create_plot_path(self):
    self.create_directory(self.plot_path)
  
  def create_pickle_path(self):
    self.create_directory(self.pickle_path)

  def create_csv_path(self):
    self.create_directory(self.csv_path)

  def throw_nodes_randomly(self):
    # r ----
    r = np.random.uniform(size=self.nodes)
    r = np.sqrt(r)
    # theta ----
    theta = np.random.uniform(
      high=2*np.pi, size=self.nodes
    )
    # positions ----
    A = np.zeros((self.nodes, 2))
    A[:, 0] = r * np.cos(theta) * self.w / 2
    A[:, 1] = r * np.sin(theta) * self.h / 2
    return A

  def get_b(self):
    return self.Area / (self.w * np.pi)

  def distance_matrix(self, A, save=True):
    path = os.path.join(
      self.csv_path, "distance.csv"
    )
    if save:
      D = np.zeros((self.nodes, self.nodes))
      for i in np.arange(1, self.nodes):
        for j in np.arange(i):
          D[i, j] = np.linalg.norm(A[i, :] - A[j, :])
      D = D + D.T
      np.savetxt(
        path, D, delimiter=","
      )
    else:
      D = np.genfromtxt(path, delimiter=",")
    return D
  
  def random_net(self, D, save=True):
    from rand_network import sample_elliptic
    path = os.path.join(
      self.csv_path, "Count.csv"
    )
    if save:
      A = sample_elliptic(
        D, self.nodes, self.rho, self.lb
      )
      A = np.array(A)
      print("A density: {:.5f}".format(
          self.den(A)
        )
      )
      np.savetxt(
        path, A, delimiter=","
      )
    else:
      A = np.genfromtxt(path, delimiter=",")
      print(
        "A density: {:.5f}".format(self.den(A))
      )
    return A

  def random_const_net(self, D, save=True):
    path = os.path.join(
      self.csv_path, "Count.csv"
    )
    if save:
      from rand_network import const_sample_elliptic
      A = const_sample_elliptic(
        D, self.nodes, self.counter,
        self.rho, self.lb
      )
      A = np.array(A)
      print("A density: {:.5f}".format(
          self.den(A)
        )
      )
      print("A counter: {}".format(
          self.count(A)
        )
      )
      np.savetxt(
        path, A, delimiter=","
      )
    else:
      A = np.genfromtxt(path, delimiter=",")
      print(
        "A density: {:.5f}".format(self.den(A))
      )
      print("A counter: {}".format(
          self.count(A)
        )
      )
    return A
# Insert path ---
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR)
# Boolean aliases ----
T = True
F = False
#Import libraries ----
from manim import *
import networkx as nx
import random
# Personal libraries ----
ORANGE
from networks.structure import MAC
from various.network_tools import *

# Declare global variables ----
linkage = "single"
nlog10 = T
lookup = F
prob = F
cut = F
structure = "FLN"
mode = "ZERO"
distance = "tracto16"
nature = "original"
imputation_method = ""
topology = "MIX"
mapping = "trivial"
index  = "Hellinger2"
bias = float(0)
alpha = 0.
opt_score = ["_S"]
save_data = T
version = "47d106"
__nodes__ = 47
__inj__ = 47

NET = MAC[f"MAC{__inj__}"](
  linkage, mode,
  nlog10 = nlog10,
  structure = structure,
  lookup = lookup,
  version = version,
  nature = nature,
  model = imputation_method,
  distance = distance,
  inj = __inj__,
  topology = topology,
  index = index,
  mapping = mapping,
  cut = cut,
  b = bias,
  alpha = alpha,
  discovery="discovery_7"
)
    
class GraphFLNet(Scene):
  def graph_configure(self, graph, **kwargs):    
        # This is to configure the Graph object to wiggle the verts
        for submob in graph.vertices.values():
            submob.jiggling_direction = rotate_vector(
                RIGHT, np.random.random() * TAU *1.5,
            )
            submob.jiggling_phase = np.random.random() * TAU *1.5
        for key, value in kwargs.items():
            setattr(graph, key, value)

  def data2nx(self):
      data = NET.A
      cols = data.shape[1]
      data[data > 0] = -np.log(data[data > 0])
      G = nx.DiGraph(data[:cols, :cols])
      return G
  
  def random_points_squarelike(self, pt : int):
      xy = (np.random.rand(1000, 2) * 2) - 1
      xy[:, 0] *= 6.5
      xy[:, 1] *= 3.5
      pos_random = np.zeros((pt, 3))
      e = 0
      while e != pt:
          for i in np.arange(1000):
              if np.abs(xy[i, 0]) > 3.5 or np.abs(xy[i, 1]) > 2.5:
                  pos_random[e, :2] = xy[i]
                  e += 1
                  if e == pt:
                      break
          
          if e != pt:
              xy = (np.random.rand(1000, 2) * 2) - 1
              xy[:, 0] *= 6.5
              xy[:, 1] *= 3.5
      return pos_random    

  def set_graph_stroke(self,graph,**kwargs):
      for e in graph.edges.values():
          e.set_stroke(**kwargs)
          
  
  def construct(self):
      
    def wiggle_graph_updater(graph, dt):
        # Updater to wiggle vertices of a given graph
        for key in graph.edges:
            edge=graph.edges[key]
            edge.start = graph.vertices[key[0]].get_center()
            edge.end = graph.vertices[key[1]].get_center()
        
        for submob in graph.vertices.values():
            submob.jiggling_phase += dt * graph.jiggles_per_second * TAU
            submob.shift(
                graph.jiggle_amplitude *
                submob.jiggling_direction *
                np.sin(submob.jiggling_phase) * dt
            )
            
    random.seed(246)
    np.random.seed(4812)

    Gx = self.data2nx()
    G = DiGraph.from_networkx(Gx, layout="kamada_kawai", layout_scale=3)
    for e in Gx.edges:
        G.remove_edges(tuple(e))
    
    self.play(Create(G))
    self.graph_configure(G, jiggle_amplitude=0.1, jiggles_per_second=0.1)
    G.add_updater(wiggle_graph_updater)
    self.wait(5)
    rectangle_positions = self.random_points_squarelike(NET.nodes)
    self.play(
        *[G[v].animate.move_to(rectangle_positions[ind]) for ind, v in enumerate(G.vertices)]
    )
    self.wait()
    title1 = Text("Link and node communities of the", font="Monaco", font_size=32)
    title2 = Text("macaque interareal cortical network", font="Monaco", font_size=32)
    title2.next_to(title1, DOWN)
    self.play(Write(title1), Write(title2))
    self.wait()
    Author = Text("Jorge Martinez Armas", font="Monaco", font_size=14)
    Author.next_to(title2, 2 * DOWN)
    self.play(Write(Author))
    self.wait(3)
    self.play(Unwrite(title1), Unwrite(title2), Unwrite(Author))
    self.wait()
    self.play(Uncreate(G))
    # Random dots around the canvas

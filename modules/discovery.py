# Python libs ----
import numpy as np
import numpy.typing as npt
from typing import Any
from collections import Counter
# Dlink libs ----
from various.discovery_channel import *
from modules.hierarmerge import Hierarchy
from networks.toy import TOY

class Discovery:
    def __init__(self, H : Hierarchy, K : int, Cr : npt.NDArray[Any], score="_S", dir="source", undirected=False, **kwargs) -> None:
        self.Cr = Cr
        self.A = H.A
        self.dA = H.dA
        self.rows = H.rows
        self.nodes = H.nodes
        self.score = score
        self.undirected = undirected
        self.labels = H.colregion.labels[:H.nodes]
    
    def Aragog(self):
        nocs = {}
        nocs_size = {}
        cr = self.Cr.copy()
        ## Single nodes ----
        single_nodes = np.where(self.Cr == -1)[0]
        ## Nodes with single community membership ----
        NSC = [(np.where(self.Cr == i)[0], i) for i in np.unique(self.Cr) if i != -1]
        sA = self.supress_NSC(NSC)

        for sn in single_nodes:
            target_nodes = self.dA.target.loc[self.dA.source == sn].to_numpy()
            source_nodes = self.dA.source.loc[self.dA.target == sn].to_numpy()

            nodes = np.hstack([source_nodes, target_nodes, [sn]])
            nodes = np.sort(np.unique(nodes)).astype(int)

            Asn = np.zeros((nodes.shape[0] + (self.rows - self.nodes), nodes.shape[0]))

            Asn[:nodes.shape[0], :] = sA[nodes, :][:, nodes]
            Asn[nodes.shape[0]:, :] = sA[self.nodes:, :][:, nodes]

            NET = TOY(Asn, "single", False, False, False, "trivial", "Hellinger2", "MIX")
            H = Hierarchy(NET, Asn, Asn, np.zeros(Asn.shape), Asn.shape[1], "single", "ZERO", undirected=self.undirected)

            ## Compute features ----
            H.BH_features_cpp_no_mu()
            ## Compute link entropy ----
            H.link_entropy_cpp("short", cut=False)
            ## Compute lq arbre de merde ----
            H.la_abre_a_merde_cpp(H.BH[0])

            K, R = get_best_kr_equivalence(self.score, H)
            k , r = K[0], R[0]

            rlabels = get_labels_from_Z(H.Z, r)
            rlabels = skim_partition(rlabels)

            C = dict(Counter(rlabels))
            C = np.array(list(C.keys()))[list(C.values()) == np.max(list(C.values()))][0]
            
            nodes_after_H = nodes[rlabels == C]
            covers = self.match_after_H(nodes_after_H, NSC)
            print(covers)
            nocs[self.labels[sn]] = covers
            nocs_size[self.labels[sn]] = {c : 1 for c in covers}

        not_nocs = []
        
        for key in nocs.keys():
          if len(nocs[key]) == 1:
            not_nocs.append(key)
          i = match([key], self.labels)
          if len(nocs[key]) == 1 and self.Cr[i] == -1:
            cr[i] = nocs[key][0]

        for key in not_nocs:
          del nocs[key]
          del nocs_size[key]

        return  np.array(list(nocs.keys())), nocs, nocs_size, cr

    
    def supress_NSC(self, NSC : list):
        A = self.A.copy()
        for nsc, _ in NSC:
            rows, cols = np.meshgrid(nsc, nsc)
            A[rows, cols]= 0.
        return A
    
    def match_after_H(self, after : npt.NDArray[Any], NSC : list):
        m = []
        for nsc, c in NSC:
            if len(set(after).intersection(set(nsc))) > 0:
                m.append(c)
        return m



discovery_channel = {
    "discovery_3" : discovery_3,
    "discovery_4" : discovery_4,
    "discovery_5" : discovery_5,
    "discovery_6" : discovery_6,
    "discovery_7" : discovery_7,
    "discovery_8" : discovery_8,
    "discovery_9" : discovery_9,
    "discovery_10" : Discovery
}


        

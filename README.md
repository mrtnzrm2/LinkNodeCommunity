# Dlink
Dlink is a link community algorithm created to find communities in directed networks with very heterogeneous weights.

 ## Introduction
 Welcome everyone! How good of you to spend some of your time taking a look to this software. I consider this software to be an extension of the original link community algorithm (Ahn et al. 2010).
 
 The problem of community dectedtion is very deep since it is connected to the problem of data encoding. Each day, millons of bytes of information are saved by the goverments and companies all around the world. However, all the data is useless unless you can process it and reveal the structures, symmetries, global features, and laws inside it. Finding communties is similar to separating the data into homogeneous pieces. In the context of information theory, paritioning the data into clusters can allow you decode the information faster since the data arranged in the right way, like your cloths when you decide to fold them nicely, is easier to intepret. Networks in the real world naturally form communities since it is the way to minimize the entropy, steps to reach a certain information or execute a particular action, a property crucial if the system is under the forces of natural selection.

 There are several community detection algorithms for different network classes. However, finding communities in **dense**, **directed**, **weighted**, and **heterogeneous** network, as the macaque fraction of labeled neurons (FLN) network (Markov et al. 2011 and 2012), is a still an open question since the it is not clear what a community represent in such complex systems. For that reason, we started to work on an algorithm to overcome the challenges of indentifying communities in this type of networks.

 Our journey led us to the link community algorithm which has many important features such as assigning nodes to more than one cluster. Nevertheless, to make it work in our network, we had to add several features to the algorithm that we list below. We have baptized the algorithm **Dlink** to distinguish it from the original.

## Key features
Some key features that make our algorithm convenient are the following:

1. **New link neighborhood concepts**: We introduce **new** link neighborhood definitions for directed graphs. Our interpretation diverges from the literature, especially from the concept of line digraphs. We believe that the new criteria has many graph theoretical implications still undiscover.

2. **Node community dendrogram**: Link communities, when introduced, were exciting since nodes naturally can be classified into multiple communities. This property is natural in many social, economical, and biological networks. However, node communities, or partitions, are easier to interpret, and, in the original link community, nodes had belonged to so many communities that it was hard to interpret. We made a **new** algorithm that projects the link merging process to a node merging process which allows one to obtain a node community dendrogram which greatly simplies the search of meaningful clusters.

3. **Computing similarities using different topologies**: You can choose to find node communities for how similar are their source or/and target connections. In directed networks, nodes can have different functions from the perspective of acting or receiving the action of other nodes. A good parition from the perspective of how the nodes act on other nodes could be very differeny from how their receive the actions since those groups can potentially be different. Our algorithm can produce partitions taking into account the direction of interest which produce partitions that are easier to interpret.

4. **New quality functions ($\mu$-score)**: As it is well know, the concept of a community can have multiple interpretation; however, it is well accepted that the communities tend to be formed by set of nodes with more connections between them than with the rest of the network. But, what happens when the network is so dense that modularity, i.e., density of a cluster compared to a random network, stops being a a good quality function to detect the best partition? To solve that problem, we introduce a new quality function that we called the $\mu$-score which spots a parition with a good balance of link communities' sizes. We believe that, although the function probably needs improvements, it is a sound start towards new alternatives for quality functions in dense networks. Preliminary tests show that in the partitions selected using the $\mu$-score, the algorithm has better sensitivity ans specificity identifying the nodes with overlapping memberships (NOCs) using the LFR benchmark.

## Why is it different from the rest of community detection algorithms?


## Pybind11 C++ libraries

The code is implemented mostly in Python (3.9.13) with some C++ libraries used to speed up the link community processing to find the right parition.

The steps to pybind (mandatory) the C++ code are the following:

1. Install **cmake** version [3.24.0-rc4](https://cmake.org/files/). To use cmake, do not forget to add it to your path.

```
export PATH="/Applications/CMake.app/Contents/bin:/usr/local/bin:$PATH"
```

2. Install pybind11.

```
pip3 install pybind11
```

3. Download the hclust-cpp repository created by [Daniel Müllner](http://danifold.net/) and [Christoph Dalitz](https://lionel.kr.hs-niederrhein.de/~dalitz/data/hclust/).

```
https://github.com/cdalitz/hclust-cpp.git
```
4. Paste the repository in the cpp/process_hclust/src and cpp/la_arbre_a_merde/src.

5. Install the C++ libraries in python by running:

```
pip3 install cpp/simquest
pip3 install cpp/process_hclust
pip3 install cpp/la_arbre_a_merde
pip3 install cpp/rand_network
pip3 install cpp/WDN
```

To create directed, weighted and possible nodes with overlapping communities, we modified the LFR benchmark from [Andrea Lancichinetti](https://sites.google.com/site/andrealancichinetti/) and [Santo Fortunato](https://www.santofortunato.net/resources) (package 4) to pybind it. The modified code can be seen in cpp/WDN.

## Examples
We have created several examples in the form of Jupyter Notebooks to get a better understanding of how to use the algorithm.

- ER_exmaple: Running the algorithm in an Erdos-Renyi random graph with high density. The lack of structure in the node dendrogram shows that the algorithm does not find structure in this null model.

## Drawbacks
There still plenty of work to do. Some of the points to improve are:

- Low computational speed. Currently, the processing of link communities to identify the most interesting partitions is slow and scales as $O(M^{2})$ where $M$ is the number of links in the network. The link-to-node dendrogam projection also scales in the same way.

- The algorithm identifies well the NOCs, however, it does not identify well community membership that they belong. However, by looking to the node dendrogram and link community matrix, one can get an idea about which communty they belong to.

- For undirected networks, we need to check if the link-to-node hierarchy projection algorithm works.

## References
- Ahn, YY., Bagrow, J. & Lehmann, S. Link communities reveal multiscale complexity in networks. Nature 466, 761–764 (2010). https://doi.org/10.1038/nature09182
- Lancichinetti, A., & Fortunato, S. (2009). Benchmarks for testing community detection algorithms on directed and weighted graphs with overlapping communities. Phys. Rev. E, 80, 016118.
- Markov, N.T., Misery, P., Falchier, A., Lamy, C., Vezoli, J., Quilodran, R., Gariel,
M.A., Giroud, P., Ercsey-Ravasz, M., Pilaz, L.J., et al. (2011). Weight consistency
specifies regularities of macaque cortical networks. Cereb. Cortex 21,
1254–1272.
- Markov, N.T., Ercsey-Ravasz, M.M., Ribeiro Gomes, A.R., Lamy, C., Magrou,
L., Vezoli, J., Misery, P., Falchier, A., Quilodran, R., Gariel, M.A., et al. (2012). A
weighted and directed interareal connectivity matrix for macaque cerebral
cortex. Cereb. Cortex. Published online September 25, 2012. http://dx.doi.
org/10.1093/cercor/bhs1270.
- Müllner, D. (2013). fastcluster: Fast Hierarchical, Agglomerative Clustering Routines for R and Python. Journal of Statistical Software, 53(9), 1–18. https://doi.org/10.18637/jss.v053.i09

# LinkCommunities
 Link community algorithm applied to weighted and directed networks.

 ## Introduction
 Welcome everyone! How good of you to spend some of your time taking a look to this software. I consider this software to be an extension of the original link community algorithm (Ahn et al. 2010).

 The project I am working tries to find communities in a **dense**, **directed**, **weighted**, and **heterogeneous** network, the FLN network (Markov et al. 2014a). This type of network represents a challenge to any state-of-the-art community detection algorithm. For that reason, Pr. Zoltan Toroczkai, and I, Jorge Martinez Armas, started to work on an algorithm to overcome the challenges of indentifying communities in this network.

 Our journey led us to the link community algorithm which has many important features such as assigning nodes to more than one cluster. However, to make it work in our network, we had to add several features to the algorithm that we list below.

## Key features
Some key features that make our algorithm convenient are the following:

1. **Designed for directed networks**: We introduce **new** link neighborhood definitions for directed graphs. Our interpretation diverges from the literature, especially from the concept of line digraphs. We believe that the new criteria has many graph theoretical implications still undiscover.

2. **Node community dendrogram**: Link communities, when introduced, were exciting since nodes naturally can be classified into multiple communities. This property is natural in many social, economical, and biological networks. However, node communities, or partitions, are easier to interpret, and, in the original link community, nodes had belonged to so many communities that it was hard to interpret. We made a **new** algorithm that projects the link merging process to a node merging process which allows one to obtain a node community dendrogram which greatly simplies the search of meaningful clusters.

3. **Flexible topology**: You can choose to find node communities for how similar are their source or/and target connections. In directed networks, nodes can have different functions from the perspective of acting or receiving the action of other nodes. A good parition from the perspective of how the nodes act on other nodes could be very differeny from how their receive the actions since those groups can potentially be different. Our algorithm can produce partitions taking into account the direction of interest which produce partitions that are easier to interpret.

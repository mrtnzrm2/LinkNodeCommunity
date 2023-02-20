# LinkCommunities
 Link community algorithm applied to weighted and directed networks.

 ## Introduction
 Welcome everyone! How good of you to spend some of your time taking a look to this software. I consider this software to be an extension of the original link community algorithm (Ahn et al. 2010).
 
 The problem of community dectedtion is very deep since it is connected to the problem of data encoding. Each day, millons of bytes of information are saved by the goverments and companies all around the world. However, all that data is useless unless you can process it and reveal the structures, symmetries, global features, and laws inside. Finding communties is similar to separating the data into homogeneous parts which implies low entropy in the system. In the context of information theory, paritioning the data into clusters with similar features can allow you decode the information faster, a property prominent if the system is under the forces of natural selection.

 There are several community detection algorithms for different network classes. However, the project I am working tries to find communities in a **dense**, **directed**, **weighted**, and **heterogeneous** network, the FLN network (Markov et al. 2014a). This type of network represents a challenge to any state-of-the-art algorithm. For that reason, Pr. Zoltan Toroczkai, and I, Jorge Martinez Armas, started to work on an algorithm to overcome the challenges of indentifying communities in this network.

 Our journey led us to the link community algorithm which has many important features such as assigning nodes to more than one cluster. Nevertheless, to make it work in our network, we had to add several features to the algorithm that we list below.

## Key features
Some key features that make our algorithm convenient are the following:

1. **Designed for directed networks**: We introduce **new** link neighborhood definitions for directed graphs. Our interpretation diverges from the literature, especially from the concept of line digraphs. We believe that the new criteria has many graph theoretical implications still undiscover.

2. **Node community dendrogram**: Link communities, when introduced, were exciting since nodes naturally can be classified into multiple communities. This property is natural in many social, economical, and biological networks. However, node communities, or partitions, are easier to interpret, and, in the original link community, nodes had belonged to so many communities that it was hard to interpret. We made a **new** algorithm that projects the link merging process to a node merging process which allows one to obtain a node community dendrogram which greatly simplies the search of meaningful clusters.

3. **Flexible topology**: You can choose to find node communities for how similar are their source or/and target connections. In directed networks, nodes can have different functions from the perspective of acting or receiving the action of other nodes. A good parition from the perspective of how the nodes act on other nodes could be very differeny from how their receive the actions since those groups can potentially be different. Our algorithm can produce partitions taking into account the direction of interest which produce partitions that are easier to interpret.

4. **New quality functions**: As it is well know, the concept of a community can have multiple interpretation; however, it is well accepted that the communities tend to be formed by set of nodes with more connections between them than with the rest of the network. But, what happens when the network is so dense that modularity, i.e., density of a cluster compared to a random network, stops being a a good quality function to detect the best partition? To solve that problem, we introduce a **quality function** that we called the $\mu$-score (not very creative but makes its job).

In the context of hierarchical clustering, one needs to select the tree partition. The $\mu$-score finds a parition with a good balance of link communities' sizes. We believe that, although the function probably needs improvements, it is a sound start towards new alternatives to find partitions in dense networks.

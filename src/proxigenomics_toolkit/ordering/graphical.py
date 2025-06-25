import logging
from typing import Generator, Iterator, List, Optional, cast

import community
import lap
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from ..types import EdgeData

logger = logging.getLogger(__name__)


def dijkstra_all_shortest_numpy(g: nx.Graph) -> np.ndarray:
    """
    Computes all-pairs shortest paths for a given graph using Dijkstra's algorithm
    and returns the distance matrix as a NumPy array.

    This function leverages NetworkX's `all_pairs_dijkstra_path_length` to compute
    the shortest path lengths between all nodes in the input graph. The results are
    stored in a distance matrix where each cell (u, v) contains the shortest distance
    from node `u` to node `v`. Nodes are indexed by their numeric order in the graph.

    :param g: The input graph on which to compute all-pairs shortest paths. The graph
              should be a NetworkX graph object.
    :type g: nx.Graph
    :return: A distance matrix as a NumPy 2D array, where cell (u, v) represents
             the shortest distance from node `u` to node `v` in the graph. If no
             path exists, the value will be 0.
    :rtype: np.ndarray
    """
    dist_mat = np.zeros((g.order(), g.order()), dtype=np.float64)
    for u, d in nx.all_pairs_dijkstra_path_length(g):
        for v, w in d.items():
            dist_mat[u, v] = w
    return dist_mat


def hc_order(g: nx.Graph,
             metric: str='cityblock',
             method: str='ward',
             use_olo: bool=True) -> np.ndarray:
    """
    Basic hierarchical clustering to determine an ordering of contigs, using
    optimal leaf ordering (poor time complexity) to adjust tips.
    :param g: the graph to order, where larger edge weight implies higher connectivity.
    :param metric: any metric supported by scipy.spatial.distance.pdist, e.g. cityblock, euclidean, cosine, etc..
    :param method: ward or complete.
    :param use_olo: use optimal leaf ordering.
    :return: an ordering.
    """
    # assuming larger weights implies higher connectivity, we invert them
    # and subsequently compute all shortest paths.
    dist_mat = dijkstra_all_shortest_numpy(inverse_edge_weights(g.copy()))
    # push this through pdist to eliminate complaints by the linkage method
    dist_mat = pdist(dist_mat, metric=metric)
    z = linkage(dist_mat, method=method, metric=metric, optimal_ordering=use_olo)
    return np.array(dendrogram(z, no_plot=True)['leaves'])


def adhoc_order(g: nx.Graph, alpha: float=1.0) -> np.ndarray:
    """
    Attempt to determine an ordering based only upon cross-terms
    between contigs using graphical techniques.

    1. Begin with a contig graph where edges are weighted by contact frequency.
    2. The graph is then partitioned into subgraphs using Louvain modularity.
    3. Using inverse edge weights, the shortest path of the minimum spanning
    tree of each subgraph is used to define an order.
    4. The subgraph orderings are then concatenated together to define a full
    ordering of the sample.
    5. TODO add optimal leaf ordering is possible.
    6. Unconnected contigs are included by order of appearance.

    :param g: graph to order.
    :param alpha: additive constant used in inverse weighting.
    :return: an ordering.
    """
    sg_list = decompose_graph(g)

    # calculate inter-subgraph weights for use in ordering
    w = inter_weight_matrix(g, sg_list, norm=True)

    # perform LAP, but first convert weight matrix to a "cost" matrix
    ord_x, ord_y = lap.lapjv(1.0 / (w + alpha), return_cost=False)

    # reorder the subgraphs, just using row order
    sg_list = [sg_list[i] for i in ord_x]

    # now find the order through each subgraph
    isolates = []
    new_order = []
    for gi in sg_list:
        # if there is more than one node
        if gi.order() > 1:
            inverse_edge_weights(gi)
            mst = nx.minimum_spanning_tree(gi)
            inverse_edge_weights(gi)
            new_order.extend(edgeiter_to_nodelist(dfs_weighted(mst)))
        else:
            isolates.extend(gi.nodes())

    return np.array(new_order + isolates)


def decompose_graph(g: nx.Graph, reso: float=1.0) -> list[nx.Graph]:
    """
    Using the Louvain algorithm for community detection, as
    implemented in the community module, determine the partitioning
    which maximises the modularity. For each individual partition
    create the sub-graph of g

    :param g: the graph to decompose
    :param reso: louvain clustering threshold, smaller -> more partitions
    :return: the set of sub-graphs which form the best partitioning of g
    """

    decomposed = []
    part = community.best_partition(g, resolution=reso)
    part_labels = np.unique(list(part.values()))

    # for each partition, create the sub-graph
    for pi in part_labels:
        # start with a complete copy of the graph
        gi = g.copy()
        # build the list of nodes not in this partition and remove them
        to_remove = [n for n in g.nodes() if part[n] != pi]
        gi.remove_nodes_from(to_remove)
        decomposed.append(gi)

    return decomposed


def inter_weight_matrix(g: nx.Graph, sg: List[nx.Graph], norm: bool=True) -> np.ndarray:
    """
    Calculate the weight of interconnecting edges between subgraphs identified from
    Louvain decomposition.

    :param g: the original graph
    :param sg: the list of subgraphs
    :param norm: normalize the counts by the number of shared edges
    :return: two matrices, 'w'  the weights of shared edges, 'n' the counts of shared edges
    """

    nsub = len(sg)
    w = np.zeros((nsub, nsub))
    n = None if not norm else np.zeros_like(w, dtype=np.int64)

    # for each subgraph i
    for i in range(nsub):

        # for every node in subgraph i
        for u in sg[i].nodes():

            # for every other subgraph j
            for j in range(i+1, nsub):

                # for every node in subgraph j
                for v in sg[j].nodes():

                    # sum weight of edges connecting subgraphs i and j
                    if g.has_edge(u, v):
                        w[i, j] += g[u][v]['rawweight']
                        if norm:
                            n[i, j] += 1

    if norm:
        # only touch non-zero elements
        ix = np.where(n > 0)
        w[ix] /= n[ix]

    return w


def dfs_weighted(g: nx.Graph, source: Optional[int]=None) -> Generator:
    """
    Depth-first search, guided by edge weights
    :param g: the graph to traverse
    :param source: the starting node used during recursion
    :return: yields nodes
    """
    # either produce edges for all components or only those in list
    if source is None:
        nodes = g
    else:
        nodes = [source]

    visited = set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        # for node 'start' visit neighbours by edge weight
        stack = [(start,
                  iter(sorted(g[start],
                              key=lambda x: cast(EdgeData, g[start][x])['weight'], reverse=True)))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    yield parent, child
                    visited.add(child)
                    stack.append((child,
                                  iter(sorted(g[child],
                                              key=lambda x: cast(EdgeData, g[child][x])['weight'], reverse=True))))
            except StopIteration:
                stack.pop()


def edgeiter_to_nodelist(edge_iter: Iterator) -> list:
    """
    Create a list of nodes from an edge iterator
    :param edge_iter: edge iterator
    :return: list of node ids
    """
    nlist = []
    for ei in edge_iter:
        for ni in ei:
            if ni not in nlist:
                nlist.append(ni)
    return nlist


def inverse_edge_weights(g: nx.Graph, alpha: float=1.0) -> nx.Graph:
    """
    Invert the weights on a graph's edges
    :param g: the graph
    :param alpha: additive constant in denominator to avoid DBZ
    """
    for u, v in g.edges():
        edge_data = cast(EdgeData, g[u][v])
        edge_data['weight'] = 1.0 / (edge_data['weight'] + alpha)
    return g
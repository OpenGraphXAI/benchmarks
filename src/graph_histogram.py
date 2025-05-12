from collections import Counter
from hashlib import blake2b
from typing import Any

from networkx import Graph, DiGraph, set_node_attributes, ego_graph


__all__ = ['weisfeiler_lehman_graph_histogram', 'get_wl_label_nodes', 'get_wl_label_subgraphs']


def weisfeiler_lehman_graph_histogram(G: Graph | DiGraph, node_label: str | None = None,
                                      edge_label: str | None = None, iterations: int = 3,
                                      use_ego: bool = True, digest_size: int = 16,
                                      set_wl_labels: bool = True) -> Counter:
    """
    Weisfeiler Lehman (WL) graph histogram.

    The function iteratively aggregates and hashes neighborhoods of each node.
    After each node's neighbors are hashed to obtain updated node labels,
    a histogram of resulting labels is returned as a `Counter` object.

    Histograms are identical for isomorphic graphs and strong guarantees that
    non-isomorphic graphs will get different histograms. See [1]_ for details.

    If no node or edge attributes are provided, the degree of each node
    is used as its initial label.
    Otherwise, node and/or edge labels are used to compute the histogram.

    .. [1] Shervashidze, Schweitzer, Van Leeuwen, Mehlhorn, Borgwardt (2011).
     Weisfeiler Lehman Graph Kernels. Journal of Machine Learning Research, vol. 12, pp. 2539-2561.
    http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf

    :param G: The graph to be represented.
        Can have node and/or edge attributes. Can also have no attributes.
    :param node_label: The key in node attribute dictionary to be used for hashing.
        If None, and no edge_label is given, use the degrees of the nodes as labels.
    :param edge_label: The key in edge attribute dictionary to be used for hashing.
        If None, edge labels are ignored.
    :param iterations: Number of neighbor aggregations to perform.
        Should be larger for larger graphs.
    :param use_ego: Whether to consider ego node labels in relabeling.
    :param digest_size: Size (in bits) of blake2b hash digest to use for hashing node labels.
    :param set_wl_labels: If true adds the WL labels for each iteration to input graph.
    :return: Counter object corresponding to histogram of the input graph.
    """
    wl_labels = _init_node_labels(G, node_label, edge_label)
    histogram = Counter(f'00:{label}' for label in wl_labels.values())
    if set_wl_labels:
        set_node_attributes(G, values=wl_labels, name='WL:00')
    for iteration in range(iterations):
        wl_labels = _weisfeiler_lehman_step(G, wl_labels, edge_label, use_ego, digest_size)
        histogram += Counter(f'{iteration + 1:02d}:{label}' for label in wl_labels.values())
        if set_wl_labels:
            set_node_attributes(G, values=wl_labels, name=f'WL:{iteration + 1:02d}')
    return histogram


def _init_node_labels(G: Graph | DiGraph, node_label: str | None = None, edge_label: str | None = None) -> dict:
    """
    Initialize node labels for WL iterations.

    :param G: The graph to be represented.
        Can have node and/or edge attributes. Can also have no attributes.
    :param node_label: The key in node attribute dictionary to be used for hashing.
        If None, and no edge_label is given, use the degrees of the nodes as labels.
    :param edge_label: The key in edge attribute dictionary to be used for hashing.
        If None, edge labels are ignored.
    :return: Dictionary of nodes and corresponding initial labels.
    """
    if node_label:
        return {node: str(data[node_label]) for node, data in G.nodes(data=True)}
    elif edge_label:
        return {node: '' for node in G}
    else:
        return {node: '0' for node in G}


def _neighborhood_aggregate(G: Graph | DiGraph, node: Any, previous_node_labels: dict,
                            edge_label: str | None = None) -> Counter:
    """
    Aggregate for given node the labels of each node's neighbors.

    :param G: The graph to be represented.
        Can have node and/or edge attributes. Can also have no attributes.
    :param previous_node_labels: The dictionary of previous WL iteration node labels.
    :param edge_label: The key in edge attribute dictionary to be used for hashing.
        If None, edge labels are ignored.
    :return: Counter object of neighbors' labels.
    """
    neighbor_labels = []
    for neighbor in G.neighbors(node):
        if edge_label:
            neighbor_labels.append(str((previous_node_labels[neighbor], str(G[node][neighbor][edge_label]))))
        else:
            neighbor_labels.append(previous_node_labels[neighbor])
        prefix = "" if edge_label is None else str(G[node][neighbor][edge_label])
        neighbor_labels.append(prefix + previous_node_labels[neighbor])
    return Counter(neighbor_labels)


def _hash_label(ego_label: str, neighbor_labels: Counter, digest_size: int) -> str:
    """
    Hash ego and neighbourhood label into shorter digest hash.

    :param label: Node label to be hashed in a shorter string.
    :param digest_size: Size (in bits) of blake2b hash digest to use for hashing node labels.
    :return: Shorter hash label.
    """
    label = ego_label + str(sorted(neighbor_labels.items()))
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()


def _weisfeiler_lehman_step(G: Graph | DiGraph, previous_node_labels: dict, edge_label: str | None = None,
                            use_ego: bool = True, digest_size: int = 16) -> dict:
    """
    Perform an iteration of WL.

    :param G: The graph to be represented.
        Can have node and/or edge attributes. Can also have no attributes.
    :param previous_node_labels: The dictionary of previous WL iteration node labels.
    :param edge_label: The key in edge attribute dictionary to be used for hashing.
        If None, edge labels are ignored.
    :param use_ego: Whether to consider ego node labels in relabeling.
    :param digest_size: Size (in bits) of blake2b hash digest to use for hashing node labels.
    :return: The dictionary of new WL node labels.
    """
    new_node_labels = {}
    for node in G.nodes():
        neighborhood = _neighborhood_aggregate(G, node, previous_node_labels, edge_label)
        if use_ego:
            new_node_labels[node] = _hash_label(previous_node_labels[node], neighborhood, digest_size)
        else:
            new_node_labels[node] = _hash_label('', neighborhood, digest_size)
    return new_node_labels


def get_wl_label_nodes(G: Graph | DiGraph, wl_label: str) -> list[tuple[Any, int]]:
    """
    Return graph nodes with a WL label.

    :param G: The graph with WL labels assigned to nodes.
    :param wl_label: The label to look for.
    :return: List of pairs `(node, WL iteration)`
    """
    iteration, label = wl_label.split(':')
    return [(node, int(iteration)) for node, attributes in G.nodes(data=True) if attributes[f'WL:{iteration}'] == label]


def get_wl_label_subgraphs(G: Graph | DiGraph, wl_label: str) -> list[Graph | DiGraph]:
    """
    Return sub-graphs corresponding to a WL label.

    :param G: The graph with WL labels assigned to nodes.
    :param wl_label: The label to look for.
    :return: List of sub-graphs which correspond to `wl_label`.
    """
    return [ego_graph(G, node, radius) for node, radius in get_wl_label_nodes(G, wl_label)]

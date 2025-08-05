import networkx as nx
import numpy as np
import pytest

from proxigenomics_toolkit.ordering import (
    adhoc_order,
    decompose_graph,
    dfs_weighted,
    dijkstra_all_shortest_numpy,
    edgeiter_to_nodelist,
    hc_order,
    inter_weight_matrix,
    inverse_edge_weights,
)


@pytest.fixture
def simple_graph():
    """A simple graph for basic tests."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2)])
    return g

@pytest.fixture
def weighted_graph():
    """A simple weighted graph for testing weight-based functions."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=4.0)
    g.add_edge(0, 2, weight=9.0)
    return g

@pytest.fixture
def community_graph():
    """Graph with two distinct communities connected by a weak link."""
    g = nx.Graph()
    # Community 1: nodes 0, 1, 2
    g.add_edge(0, 1, weight=10.0, rawweight=10.0)
    g.add_edge(0, 2, weight=10.0, rawweight=10.0)
    g.add_edge(1, 2, weight=10.0, rawweight=10.0)
    # Community 2: nodes 3, 4, 5
    g.add_edge(3, 4, weight=10.0, rawweight=10.0)
    g.add_edge(3, 5, weight=10.0, rawweight=10.0)
    g.add_edge(4, 5, weight=10.0, rawweight=10.0)
    # Weak link between communities
    g.add_edge(2, 3, weight=1.0, rawweight=1.0)
    # Isolated node
    g.add_node(6)
    return g

def test_inverse_edge_weights(weighted_graph):
    """Test in-place inversion of edge weights."""
    g = weighted_graph.copy()
    original_weight = g[1][2]['weight']

    # Test with default alpha
    inverse_edge_weights(g)
    assert g[1][2]['weight'] == pytest.approx(1.0 / (original_weight + 1.0))

    # Test with custom alpha
    g = weighted_graph.copy()
    inverse_edge_weights(g, alpha=0.5)
    assert g[1][2]['weight'] == pytest.approx(1.0 / (original_weight + 0.5))

def test_edgeiter_to_nodelist():
    """Test conversion of an edge iterator to a unique node list."""
    edge_iter = iter([(0, 1), (1, 3), (1, 2), (3, 0)])
    node_list = edgeiter_to_nodelist(edge_iter)
    assert node_list == [0, 1, 3, 2]

def test_dfs_weighted(weighted_graph):
    """Test depth-first search guided by highest edge weights."""
    g = weighted_graph

    # DFS from source 0 should yield edges in descending order of weight
    edges = list(dfs_weighted(g, source=0))
    # Expected path: 0 -> 2 (weight 9), then 2 -> 1 (weight 4)
    assert edges == [(0, 2), (2, 1)]

    # Test full DFS without a source
    edges_full = list(dfs_weighted(g))
    assert edges_full == [(0, 2), (2, 1)]

def test_decompose_graph(community_graph):
    """Test graph decomposition into communities."""
    # Test with default resolution
    subgraphs = decompose_graph(community_graph)

    # The graph has 2 main communities and 1 isolate, so 3 subgraphs
    assert len(subgraphs) == 3

    nodes_in_subgraphs = sorted([tuple(sorted(sg.nodes())) for sg in subgraphs])
    expected_nodes = [(0, 1, 2), (3, 4, 5), (6,)]
    assert nodes_in_subgraphs == expected_nodes

    # Test with higher resolution, which may create more partitions
    subgraphs_reso = decompose_graph(community_graph, reso=0.5)
    assert len(subgraphs_reso) >= len(subgraphs)

def test_inter_weight_matrix(community_graph):
    """Test calculation of weights between subgraphs."""
    subgraphs = [
        community_graph.subgraph([0, 1, 2]).copy(),
        community_graph.subgraph([3, 4, 5]).copy(),
        community_graph.subgraph([6]).copy()
    ]

    # Test without normalization
    weights = inter_weight_matrix(community_graph, subgraphs, norm=False)
    # Expected weights: edge (2,3) has rawweight=1.0
    expected_weights = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
    np.testing.assert_array_almost_equal(weights, expected_weights)

    # Test with normalization (should be same here as there's 1 edge)
    weights_norm = inter_weight_matrix(community_graph, subgraphs, norm=True)
    np.testing.assert_array_almost_equal(weights_norm, expected_weights)

@pytest.mark.parametrize("method, metric",
                         [("ward", "euclidean"),
                          ("complete", "euclidean"),
                          ("complete", "cityblock")])
def test_hc_order(community_graph, method, metric):
    """Test hierarchical clustering order with different parameters."""
    g = community_graph.copy()
    # Remove isolated node as it can mess up clustering assertions
    g.remove_node(6)
    inverse_edge_weights(g)
    d_mat = nx.floyd_warshall_numpy(g)

    order = hc_order(g, method=method, metric=metric, use_olo=True)

    assert isinstance(order, np.ndarray)
    assert len(order) == g.number_of_nodes()
    assert set(order) == set(g.nodes())

    # Check if nodes from the same community are clustered together
    community1 = {0, 1, 2}
    community2 = {3, 4, 5}

    # Find the indices of the community members in the ordering
    c1_indices = {i for i, node in enumerate(order) if node in community1}
    c2_indices = {i for i, node in enumerate(order) if node in community2}

    # The indices for a community should be contiguous
    is_c1_contiguous = max(c1_indices) - min(c1_indices) == len(community1) - 1
    is_c2_contiguous = max(c2_indices) - min(c2_indices) == len(community2) - 1
    assert is_c1_contiguous
    assert is_c2_contiguous

def test_adhoc_order(community_graph):
    """Test the adhoc ordering algorithm."""
    order = adhoc_order(community_graph)

    assert isinstance(order, np.ndarray)
    assert len(order) == community_graph.number_of_nodes()
    assert set(order) == set(community_graph.nodes())

    # Check if nodes from the same community are grouped, similar to hc_order
    community1 = {0, 1, 2}
    community2 = {3, 4, 5}
    isolate = {6}

    # The order of sub-lists (communities and isolates) can vary.
    # We expect three groups in the final list.
    order_list = list(order)

    # Find where the communities ended up in the list
    try:
        c1_start_index = min(order_list.index(n) for n in community1)
        c2_start_index = min(order_list.index(n) for n in community2)
    except ValueError:
        pytest.fail("A node from a community was not found in the final order.")

    c1_ordered_subset = set(order_list[c1_start_index : c1_start_index + len(community1)])
    c2_ordered_subset = set(order_list[c2_start_index : c2_start_index + len(community2)])

    assert c1_ordered_subset == community1
    assert c2_ordered_subset == community2
    assert 6 in order_list # Check isolate is present


@pytest.fixture
def disconnected_graph():
    """A graph with a disconnected component to test path existence."""
    g = nx.Graph()
    g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0)])
    g.add_node(3)  # Isolated node
    return g


def test_dijkstra_all_shortest_numpy_on_weighted_graph(weighted_graph):
    """
    Tests that the function correctly calculates shortest paths on a weighted graph,
    preferring a multi-hop shorter path over a single-hop longer one.
    """
    dist_matrix = dijkstra_all_shortest_numpy(weighted_graph)

    # Expected distances:
    # path(0,2) should be via node 1 (cost 1+4=5), not the direct edge (cost 9)
    expected = np.array([
        [0., 1., 5.],
        [1., 0., 4.],
        [5., 4., 0.]
    ])

    np.testing.assert_array_almost_equal(dist_matrix, expected)


def test_dijkstra_all_shortest_numpy_with_disconnected_component(disconnected_graph):
    """
    Tests that the function handles disconnected components correctly,
    where paths do not exist and distances should be 0.
    """
    dist_matrix = dijkstra_all_shortest_numpy(disconnected_graph)

    # Expected distances:
    # No path to/from node 3, so distances should remain 0 as per the implementation.
    expected = np.array([
        [0., 1., 2., 0.],
        [1., 0., 1., 0.],
        [2., 1., 0., 0.],
        [0., 0., 0., 0.]
    ])

    np.testing.assert_array_equal(dist_matrix, expected)


def test_dijkstra_all_shortest_numpy_on_empty_graph():
    """
    Tests that the function handles an empty graph gracefully,
    returning a 0x0 matrix.
    """
    g = nx.Graph()
    dist_matrix = dijkstra_all_shortest_numpy(g)
    assert dist_matrix.shape == (0, 0)


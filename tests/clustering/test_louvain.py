import logging

import networkx as nx
import pytest

from proxigenomics_toolkit.clustering.louvain import cluster, decompose_graph, print_info, write_mcl, write_output


def test_decompose_graph_single_partition():
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3)])
    subgraphs = decompose_graph(g)
    assert len(subgraphs) == 1, "Graph with one community should result in one subgraph."


def test_decompose_graph_multiple_partitions():
    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])
    subgraphs = decompose_graph(g)
    assert len(subgraphs) == 2, "Graph with two communities should result in two subgraphs."
    assert all(isinstance(subgraph, nx.Graph) for subgraph in
               subgraphs), "All items in result should be NetworkX Graph objects."


def test_decompose_graph_empty_graph():
    g = nx.Graph()
    subgraphs = decompose_graph(g)
    assert len(subgraphs) == 0, "Empty graph should result in no subgraphs."


def test_decompose_graph_disconnected_nodes():
    g = nx.Graph()
    g.add_nodes_from([1, 2, 3])
    subgraphs = decompose_graph(g)
    assert len(subgraphs) == 3, "Graph with disconnected nodes should result in one subgraph per node."


def test_decompose_graph_no_edges_partition():
    g = nx.Graph()
    g.add_nodes_from([1, 2, 3, 4])
    subgraphs = decompose_graph(g)
    assert len(subgraphs) == len(g.nodes), "Graph with no edges should result in one subgraph per node."


@pytest.fixture
def graph_with_isolates():
    """Provides a networkx graph with two main communities and two isolated nodes."""
    g = nx.Graph()
    # Community 1: nodes 0, 1, 2
    g.add_edges_from([(0, 1, {'weight': 1.0}), (1, 2, {'weight': 1.0}), (0, 2, {'weight': 1.0})])
    # Community 2: nodes 3, 4, 5
    g.add_edges_from([(3, 4, {'weight': 1.0}), (4, 5, {'weight': 1.0}), (3, 5, {'weight': 1.0})])
    # Bridge
    g.add_edge(2, 3, weight=0.1)
    # Isolated nodes
    g.add_node(6)
    g.add_node(7)
    return g

def test_louvain_cluster_default(graph_with_isolates):
    """Tests the default behavior of louvain clustering."""
    g = graph_with_isolates
    communities = cluster(g, no_iso=False, ragbag=False)

    # Should find 4 communities: 2 main ones and 2 for each isolate
    assert len(communities) == 4

    node_to_community = {node: cid for cid, members in communities.items() for node in members}
    assert node_to_community[0] == node_to_community[1] == node_to_community[2]
    assert node_to_community[3] == node_to_community[4] == node_to_community[5]
    assert node_to_community[0] != node_to_community[3]
    assert 6 in node_to_community
    assert 7 in node_to_community

def test_louvain_cluster_no_iso(graph_with_isolates):
    """Tests the no_iso option, which should remove isolated nodes."""
    g = graph_with_isolates.copy()
    communities = cluster(g, no_iso=True, ragbag=False)

    # Should find 2 communities, isolates are removed
    assert len(communities) == 2

    all_clustered_nodes = {node for members in communities.values() for node in members}
    assert 6 not in all_clustered_nodes
    assert 7 not in all_clustered_nodes

def test_louvain_cluster_ragbag(graph_with_isolates):
    """Tests the ragbag option, which should group isolates into one cluster."""
    g = graph_with_isolates.copy()
    communities = cluster(g, no_iso=False, ragbag=True)

    # Should find 3 communities: 2 main ones and 1 ragbag
    assert len(communities) == 3

    ragbag_cluster_id = None
    for cid, members in communities.items():
        if 6 in members or 7 in members:
            ragbag_cluster_id = cid
            break

    assert ragbag_cluster_id is not None
    assert communities[ragbag_cluster_id].keys() == {6, 7}


def test_print_info_logs_correct_message(caplog):
    # Create a small graph for testing
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 1)])

    # Use caplog to capture log output
    with caplog.at_level(logging.INFO):
        print_info(graph)

    # Check that the log message is correct
    assert "Graph composed of 3 nodes and 3 edges" in caplog.text, "Log message did not match expected output."


@pytest.fixture
def sample_communities():
    """Provides a standard dictionary of communities for testing I/O."""
    return {
        1: {101: 1.0, 103: 1.0, 102: 0.5},
        0: {202: 1.0, 201: 1.0},
        10: {301: 1.0}
    }


def test_write_mcl(sample_communities, tmp_path):
    """Tests that write_mcl produces a correctly formatted MCL file."""
    p = tmp_path / "output.mcl"
    write_mcl(sample_communities, str(p))

    with open(p, 'r') as f:
        content = f.read().strip()

    expected_content = (
        "201 202\n"
        "101 102 103\n"
        "301"
    )
    assert content == expected_content


def test_write_output_mcl_format(sample_communities, tmp_path):
    """Tests write_output function for the 'mcl' format."""
    p = tmp_path / "output.mcl"
    write_output(sample_communities, str(p), ofmt='mcl')

    with open(p, 'r') as f:
        content = f.read().strip()

    expected_content = (
        "201 202\n"
        "101 102 103\n"
        "301"
    )
    assert content == expected_content


def test_write_output_graphml_format(sample_communities, tmp_path):
    """Tests write_output function for the 'graphml' format."""
    p = tmp_path / "output.graphml"
    write_output(sample_communities, str(p), ofmt='graphml')

    # Read the graph back and verify its structure
    cg = nx.read_graphml(str(p))

    # Check nodes: cluster IDs and sequence IDs should be nodes
    expected_nodes = (list(sample_communities.keys()) +
                      [item for sublist in sample_communities.values() for item in sublist])
    assert set(int(n) for n in cg.nodes()) == set(expected_nodes)

    # Check edges: should go from cluster ID to sequence ID
    assert set(cg.edges()) == {(str(k), str(vi)) for k, v in sample_communities.items() for vi in v}


def test_write_output_unsupported_format(sample_communities, tmp_path):
    """Tests that write_output raises an error for unsupported formats."""
    p = tmp_path / "output.txt"
    with pytest.raises(RuntimeError, match="Unsupported format type: invalid_format"):
        write_output(sample_communities, str(p), ofmt='invalid_format')

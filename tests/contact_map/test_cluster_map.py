import numpy as np
import pytest
from Bio.SeqRecord import SeqRecord

from proxigenomics_toolkit.contact_map.cluster_map import (
    add_cluster_names,
    flye_extractor,
    megahit_extractor,
    read_infomap_tree,
    spades_extractor,
)
from proxigenomics_toolkit.exceptions import InvalidCoverageFormatError


def test_spades_extractor_valid_input():
    seq_record = SeqRecord(id="test_id", name="NODE_1_length_1000_cov_30.5", description="", seq="ACGT")
    coverage = spades_extractor(seq_record)
    assert coverage == 30.5, "Coverage did not match the expected value."


def test_spades_extractor_invalid_input():
    seq_record = SeqRecord(id="test_id", name="invalid_name_without_coverage", description="", seq="ACGT")
    with pytest.raises(InvalidCoverageFormatError) as excinfo:
        spades_extractor(seq_record)
    assert "Failed to extract coverage" in str(excinfo.value), "Exception message did not match expected content."


def test_spades_extractor_edge_case_NAN():
    seq_record = SeqRecord(id="test_id", name="NODE_2_length_500_cov_XYZ", description="", seq="ACGT")
    with pytest.raises(InvalidCoverageFormatError,
                       match='Failed to extract coverage for NODE_2_length_500_cov_XYZ. '
                             '"NODE_2_length_500_cov_XYZ" did not match spades_extractor pattern'):
        spades_extractor(seq_record)

def test_megahit_extractor_valid():
    seq_record = SeqRecord(id="test1", description="MEGAHIT_contig_1 multi=25.4", seq="ACGT")
    coverage = megahit_extractor(seq_record)
    assert coverage == 25.4, "Coverage extracted is not as expected."


def test_megahit_extractor_invalid():
    seq_record = SeqRecord(id="test2", description="Invalid description", seq="ACGT")
    with pytest.raises(InvalidCoverageFormatError):
        megahit_extractor(seq_record)


def test_flye_extractor_valid_description():
    seq_record = SeqRecord(id="edge_1", description="dp:5", seq="ACGT")
    assert flye_extractor(seq_record) == 5


def test_flye_extractor_invalid_description():
    seq_record = SeqRecord(id="edge_99", description="something something", seq="ACTG")
    with pytest.raises(InvalidCoverageFormatError):
        flye_extractor(seq_record)


def test_add_cluster_names_basic():
    clustering = {
        0: {"seq_names": [], "seq_ids": [], "extent": 100, "status": "active"},
        1: {"seq_names": [], "seq_ids": [], "extent": 200, "status": "inactive"},
    }
    expected_names = ["CL1", "CL2"]

    add_cluster_names(clustering, prefix="CL")

    assert clustering[0]["name"] == expected_names[0], "Cluster 0 name incorrect."
    assert clustering[1]["name"] == expected_names[1], "Cluster 1 name incorrect."


def test_add_cluster_names_diff_prefix():
    clustering = {
        0: {"seq_names": [], "seq_ids": [], "extent": 150, "status": "active"},
        1: {"seq_names": [], "seq_ids": [], "extent": 250, "status": "active"},
    }
    expected_names = ["CLUSTER1", "CLUSTER2"]

    add_cluster_names(clustering, prefix="CLUSTER")

    assert clustering[0]["name"] == expected_names[0], "Cluster 0 name with prefix incorrect."
    assert clustering[1]["name"] == expected_names[1], "Cluster 1 name with prefix incorrect."


def test_add_cluster_names_large_cluster_ids():
    clustering = {
        10: {"seq_names": [], "seq_ids": [], "extent": 300, "status": "active"},
        21: {"seq_names": [], "seq_ids": [], "extent": 400, "status": "inactive"},
    }
    expected_names = ["CL11", "CL22"]

    add_cluster_names(clustering, prefix="CL")

    assert clustering[10]["name"] == expected_names[0], "Cluster 10 name with large IDs incorrect."
    assert clustering[21]["name"] == expected_names[1], "Cluster 21 name with large IDs incorrect."


def test_add_cluster_names_empty_clustering():
    clustering = {}

    with pytest.raises(ValueError, match="Cannot assign cluster names to empty clustering solution"):
        add_cluster_names(clustering, prefix="CL")


def test_add_cluster_names_single_cluster():
    clustering = {
        0: {"seq_names": [], "seq_ids": [], "extent": 500, "status": "active"},
    }
    expected_name = "CL1"

    add_cluster_names(clustering, prefix="CL")

    assert clustering[0]["name"] == expected_name, "Single cluster name incorrect."

# The tests will be added to this file, assuming it might contain other tests.


def test_read_infomap_tree_standard_file():
    """
    Tests reading a standard, well-formatted Infomap tree file.
    Ensures clusters are correctly identified and sorted by size.
    """
    tree_file = "test_data/infomap.tree"
    result = read_infomap_tree(tree_file)

    # Cluster '2' has 3 members, cluster '1' has 2. So '2' should be key 0.
    expected = {
        0: np.array(["edge_1", "edge_2"]),
        1: np.array(["edge_3", "edge_4"]),
        2: np.array(["edge_5", "edge_6"]),
        3: np.array(["edge_7", "edge_8"]),
        4: np.array(["edge_9", "edge_10"]),
        5: np.array(["edge_11"]),
        6: np.array(["edge_12"]),
        7: np.array(["edge_13"]),
        8: np.array(["edge_14"]),
        9: np.array(["edge_15"]),
    }

    assert len(result) == 10
    assert set(range(10)) == set(result)

    # Use numpy's testing utilities for robust array comparison
    for i in range(10):
        np.testing.assert_array_equal(result[i], expected[i])


def test_read_infomap_tree_with_comments_and_empty_lines(tmp_path):
    """
    Tests that the function correctly ignores comments and empty lines.
    """
    infomap_content = """# This is a header comment
#
# path flow node_id name
1:1 0.1 "node_A" 1

2:1 0.2 "node_B" 2
1:2 0.3 "node_C" 3

# This is a footer comment
"""
    p = tmp_path / "test_with_comments.tree"
    p.write_text(infomap_content)

    result = read_infomap_tree(str(p))

    # Cluster '1' has 2 members, cluster '2' has 1.
    expected = {
        0: np.array(["node_A", "node_C"]),
        1: np.array(["node_B"]),
    }

    assert len(result) == 2
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])


def test_read_infomap_tree_empty_file(tmp_path):
    """
    Tests that reading an empty file returns an empty dictionary.
    """
    p = tmp_path / "empty.tree"
    p.write_text("")  # Create an empty file

    with pytest.raises(ValueError, match="The supplied tree file contained no results"):
        read_infomap_tree(str(p))

def test_read_infomap_tree_only_comments(tmp_path):
    """
    Tests that a file containing only comments results in an empty dictionary.
    """
    infomap_content = """# This file only has comments.
# No data lines here.
# path flow node_id name
"""
    p = tmp_path / "only_comments.tree"
    p.write_text(infomap_content)

    with pytest.raises(ValueError, match="The supplied tree file contained no results"):
        read_infomap_tree(str(p))

def test_read_infomap_tree_multi_level_hierarchy(tmp_path):
    """
    Tests that cluster IDs are correctly parsed from a multi-level hierarchy.
    The cluster ID should be everything before the final colon.
    """
    infomap_content = """# path flow name node_id 
1:1:1 0.1 "a" 1
1:2:1 0.2 "c" 2
1:1:2 0.3 "b" 3
2:1:1 0.4 "d" 4
"""
    p = tmp_path / "multi_level.tree"
    p.write_text(infomap_content)

    result = read_infomap_tree(str(p))

    # Clusters are '1:1', '1:2', and '2:1'.
    # Sizes: '1:1' (2 members), '1:2' (1 member), '2:1' (1 member)
    # The sort is stable, so '1:2' should come before '2:1' if sizes are equal.
    expected_cluster_0 = np.array(["a", "b"])  # from '1:1'
    expected_cluster_1 = np.array(["c"])      # from '1:2'
    expected_cluster_2 = np.array(["d"])      # from '2:1'

    assert len(result) == 3
    np.testing.assert_array_equal(result[0], expected_cluster_0)

    # The order of size-1 clusters depends on iteration order, which is not guaranteed.
    # So we check for presence instead of exact key mapping for clusters of the same size.
    assert (np.array_equal(result[1], expected_cluster_1) and np.array_equal(result[2], expected_cluster_2)) or \
           (np.array_equal(result[1], expected_cluster_2) and np.array_equal(result[2], expected_cluster_1))
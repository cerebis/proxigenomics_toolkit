import os
import re

import numpy as np
import pytest

from proxigenomics_toolkit.ordering import (
    read_lkh,
    reciprocal_counts,
    scale_mat,
    similarity_to_distance,
    write_lkh,
)


@pytest.fixture
def similarity_matrix():
    """Provides a simple 4x4 similarity matrix with zeros."""
    return np.array([
        [0.0, 1.0, 0.5, 0.0],
        [1.0, 0.0, 0.8, 0.2],
        [0.5, 0.8, 0.0, 0.9],
        [0.0, 0.2, 0.9, 0.0]
    ], dtype=np.float64)

@pytest.fixture
def distance_matrix():
    """Provides a simple 4x4 integer distance matrix."""
    return np.array([
        [0, 10, 20, 99],
        [10, 0, 15, 25],
        [20, 15, 0, 5],
        [99, 25, 5, 0]
    ], dtype=int)

def test_read_lkh(tmp_path):
    """Tests reading a valid LKH tour file."""
    tour_content = (
        "NAME : test_tour\n"
        "TYPE : TOUR\n"
        "DIMENSION : 4\n"
        "TOUR_SECTION\n"
        "1\n"
        "3\n"
        "4\n"
        "2\n"
        "-1\n"
        "EOF\n"
    )
    tour_file = tmp_path / "test.tour"
    tour_file.write_text(tour_content)

    tour_data = read_lkh(str(tour_file))

    assert tour_data['NAME'] == 'test_tour'
    assert tour_data['TYPE'] == 'TOUR'
    assert tour_data['DIMENSION'] == 4
    # Path should be 0-based and have the end-marker removed:
    # [1, 3, 4, 2] -> [0, 2, 3, 1]
    np.testing.assert_array_equal(tour_data['path'], np.array([0, 2, 3, 1]))

def test_write_lkh_files(tmp_path, distance_matrix):
    """
    Tests the creation and content of LKH .par and .dat files with a full matrix.
    """
    root_name = tmp_path / "test_run"
    dim = len(distance_matrix)
    fixed_edges = [(1, 2), (3, 4)]

    write_lkh(
        root_path_name=str(root_name),
        m=distance_matrix,
        dim=dim,
        max_trials=100,
        runs=10,
        seed=12345,
        fixed_edges=fixed_edges,
        pop_size=20,
        mat_fmt='full'
    )

    par_file = root_name.with_suffix(".par")
    dat_file = root_name.with_suffix(".dat")

    # Verify .par file content
    assert par_file.exists()
    par_content = par_file.read_text()
    assert "SPECIAL" in par_content
    assert "POPULATION_SIZE = 20" in par_content
    assert f"PROBLEM_FILE = {dat_file}" in par_content
    assert "MAX_TRIALS = 100" in par_content
    assert "RUNS = 10" in par_content
    assert "SEED = 12345" in par_content

    # Verify .dat file content
    assert dat_file.exists()
    dat_content = dat_file.read_text()
    assert f"NAME: {os.path.basename(root_name)}" in dat_content
    assert f"DIMENSION: {dim}" in dat_content
    assert "EDGE_WEIGHT_FORMAT: FULL_MATRIX" in dat_content

    # Check fixed edges section
    assert "FIXED_EDGES_SECTION" in dat_content
    assert "1 2" in dat_content
    assert "3 4" in dat_content

    # Check matrix data by extracting and loading it
    matrix_section = re.search(r"EDGE_WEIGHT_SECTION\n(.*?)\nFIXED_EDGES_SECTION", dat_content, re.DOTALL)
    assert matrix_section is not None
    loaded_matrix = np.loadtxt(matrix_section.group(1).splitlines(), dtype=int)
    np.testing.assert_array_equal(loaded_matrix, distance_matrix)

def test_write_lkh_upper_row(tmp_path, distance_matrix):
    """Tests the 'upper' matrix format for the .dat file."""
    root_name = tmp_path / "test_run_upper"

    write_lkh(
        root_path_name=str(root_name),
        m=distance_matrix,
        dim=len(distance_matrix),
        mat_fmt='upper'
    )

    dat_file = root_name.with_suffix(".dat")
    assert dat_file.exists()
    dat_content = dat_file.read_text()

    assert "EDGE_WEIGHT_FORMAT: UPPER_ROW" in dat_content

    # Check for the correctly formatted upper-triangular matrix data
    expected_data = "10 20 99\n15 25\n5"
    assert expected_data in dat_content

def test_similarity_to_distance(similarity_matrix):
    """Tests the similarity_to_distance conversion logic."""
    dist_mat = similarity_to_distance(
        similarity_matrix.copy(),
        method='inverse',
        alpha=2
    )

    assert dist_mat.shape == similarity_matrix.shape
    # Diagonal should be 0, but we limit the value sizes (min and max).
    # The function works on a copy, so we check the output
    assert np.all(np.diag(dist_mat) <= 2.68435456e+08)

    # Locations of zeros in the input should be the max values in the output
    zero_indices = np.where(similarity_matrix == 0)
    nonzero_indices = np.where(similarity_matrix != 0)

    max_dist_val = np.max(dist_mat)
    assert np.all(dist_mat[zero_indices] == max_dist_val)
    assert np.max(dist_mat[nonzero_indices]) < max_dist_val

def test_reciprocal_counts(similarity_matrix):
    """Tests the reciprocal_counts distance conversion."""
    dist_mat = reciprocal_counts(similarity_matrix.copy(), alpha=0.1)

    assert np.all(np.diag(dist_mat) == 0)
    assert np.all(dist_mat >= 0)

    # The smallest non-zero distance should be scaled to be approximately 1.01
    min_dist = np.min(dist_mat[np.nonzero(dist_mat)])
    assert np.isclose(min_dist, 1.01)

def test_scale_mat():
    """Tests the in-place matrix scaling function."""
    mat = np.array([[1, 2], [3, 4]], dtype=float)
    scaled_mat = scale_mat(mat.copy(), _min=10, _max=20)

    assert np.min(scaled_mat) == 10
    assert np.max(scaled_mat) == 20
    # Check a middle value: (2-1)/(4-1) * (20-10) + 10 = 1/3*10 + 10 = 13.33...
    assert np.isclose(scaled_mat[0, 1], 13.3333333)
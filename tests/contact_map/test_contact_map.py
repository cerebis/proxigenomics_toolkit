import numpy as np
import pytest
from scipy.stats import binom, poisson

from proxigenomics_toolkit.contact_map.contact_map import (
    arithmetic_mean,
    bin_indices,
    count_bin_sites,
    fast_length_norm,
    fast_norm_bysite,
    fast_norm_gothic,
    fast_norm_tipbased_bylength,
    fast_norm_tipbased_bysite,
    find_containing_bin,
    geometric_mean,
    harmonic_mean,
    mean_selector,
)
from proxigenomics_toolkit.exceptions import ApplicationException

# A tolerance for floating-point comparisons
TOL = 1e-9

# --- Tests for individual mean functions ---

@pytest.mark.parametrize("x, y, expected", [
    (2, 8, 4.0),          # Standard case
    (1, 1, 1.0),          # Identity
    (10, 0, 0.0),         # With zero
    (2.5, 10.0, 5.0),     # With floats
    (5, 5, 5.0),          # Identical inputs
])
def test_geometric_mean(x, y, expected):
    """Tests the geometric_mean function with various inputs."""
    assert geometric_mean(x, y) == pytest.approx(expected, TOL)

@pytest.mark.parametrize("x, y, expected", [
    (2, 2, 2.0),          # Identity
    (1, 4, 1.6),          # Standard case
    (10, 0, 0.0),         # With zero
    (3.0, 6.0, 4.0),      # With floats
    (10, 10, 10.0),       # Identical inputs
])
def test_harmonic_mean(x, y, expected):
    """Tests the harmonic_mean function with various inputs."""
    assert harmonic_mean(x, y) == pytest.approx(expected, TOL)

def test_harmonic_mean_handles_zero_sum():
    """Tests the harmonic_mean function for division by zero when inputs sum to zero."""
    # The sum x + y is in the denominator, so if the sum is 0, it should raise an error.
    with pytest.raises(ZeroDivisionError):
        harmonic_mean(1, -1)

@pytest.mark.parametrize("x, y, expected", [
    (2, 8, 5.0),          # Standard case
    (10, 0, 5.0),         # With zero
    (-5, 5, 0.0),         # Positive and negative
    (2.5, 7.5, 5.0),      # With floats
    (10, 10, 10.0),       # Identical inputs
])
def test_arithmetic_mean(x, y, expected):
    """Tests the arithmetic_mean function with various inputs."""
    assert arithmetic_mean(x, y) == pytest.approx(expected, TOL)


# --- Tests for the mean_selector function ---

def test_mean_selector_returns_correct_functions():
    """Tests that mean_selector returns the correct function for each valid name."""
    assert mean_selector('geometric') is geometric_mean
    assert mean_selector('harmonic') is harmonic_mean
    assert mean_selector('arithmetic') is arithmetic_mean

def test_mean_selector_raises_error_for_invalid_name():
    """Tests that mean_selector raises a RuntimeError for an unsupported mean type."""
    with pytest.raises(RuntimeError, match=r'unsupported mean type \[invalid_mean\]'):
        mean_selector('invalid_mean')

@pytest.mark.parametrize("name, x, y, expected", [
    ('geometric', 2, 8, 4.0),
    ('harmonic', 3, 6, 4.0),
    ('arithmetic', 4, 6, 5.0),
])
def test_mean_selector_returned_function_works(name, x, y, expected):
    """
    Tests that the function returned by mean_selector produces the correct output.
    """
    mean_func = mean_selector(name)
    assert mean_func(x, y) == pytest.approx(expected, TOL)


@pytest.fixture
def group_sites_data():
    """Provides a standard, sorted numpy array for testing find_containing_bin."""
    return np.array([
        [  0, 0],
        [100, 0],
        [200, 1],
        [300, 2],
        [400, 3]
    ], dtype=np.int64)


@pytest.mark.parametrize("query_pos, expected_bin", [
    (50, 0),  # Before the first site
    (100, 0),  # Exactly at the first site
    (101, 1),  # Exactly at the last site
    (150, 1),  # Between two sites
    (399, 3),  # Just before a site
    (400, 3),  # Exactly at the last site
    (500, 3),  # After the last site
])
def test_find_containing_bin(group_sites_data, query_pos, expected_bin):
    """Tests the find_containing_bin function with various query positions."""
    result = find_containing_bin(group_sites_data, query_pos)
    assert result == expected_bin


def test_fast_norm_tipbased_bylength_inplace_modification():
    """Tests that fast_norm_tipbased_bylength correctly modifies the data array in-place."""
    coords = np.array([[0, 1, 0], [1, 0, 2]], dtype=np.int64)
    data = np.array([10.0, 5.0, 2.0], dtype=np.float64)
    tip_lengths = np.array([50, 80, 100], dtype=np.int64)
    tip_size = 200

    # Expected values calculation:
    # data[0] = 10 * 200**2 / (50 * 80) = 10 * 40000 / 4000 = 100
    # data[1] = 5 * 200**2 / (80 * 50) = 5 * 40000 / 4000 = 50
    # data[2] = 2 * 200**2 / (50 * 100) = 2 * 40000 / 5000 = 16
    expected_data = np.array([100.0, 50.0, 16.0])

    fast_norm_tipbased_bylength(coords, data, tip_lengths, tip_size)
    np.testing.assert_allclose(data, expected_data, rtol=TOL)


def test_fast_norm_tipbased_bysite_inplace_modification():
    """Tests that fast_norm_tipbased_bysite correctly modifies the data array in-place."""
    coords = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    data = np.array([1.0, 4.0], dtype=np.float64)
    sites = np.array([[2, 5], [4, 10]], dtype=np.int64)

    # Expected values calculation:
    # data[0] = 1.0 * 1.0 / (sites[0,0] * sites[1,1]) = 1.0 / (2 * 10) = 0.05
    # data[1] = 4.0 * 1.0 / (sites[1,1] * sites[0,0]) = 4.0 / (10 * 2) = 0.2
    expected_data = np.array([0.05, 0.2])

    fast_norm_tipbased_bysite(coords, data, sites)
    np.testing.assert_allclose(data, expected_data, rtol=TOL)


def test_fast_norm_tipbased_bysite_handles_zero():
    """Tests that fast_norm_tipbased_bysite results in 'inf' with zero in sites."""
    coords = np.array([[0], [1], [0], [1]], dtype=np.int64)
    data = np.array([10.0], dtype=np.float64)
    sites = np.array([[2, 5], [4, 0]], dtype=np.int64)  # sites[1,1] is zero

    with pytest.raises(ZeroDivisionError):
        fast_norm_tipbased_bysite(coords, data, sites)


@pytest.fixture
def gothic_test_data():
    """Provides a consistent set of data for testing fast_norm_gothic."""
    return {
        "rows": np.array([0, 1, 0]),
        "cols": np.array([1, 0, 2]),
        "data": np.array([5, 5, 10], dtype=np.float64),
        "rel_cov": np.array([0.01, 0.02, 0.03]),
        "total_obs": 100000,
        "frac_random": 0.1
    }


def test_fast_norm_gothic_binomial_mode(gothic_test_data):
    """Tests fast_norm_gothic in 'binomial' mode."""
    # Unpack and copy data to avoid modifying the fixture
    d = gothic_test_data
    data_copy = d['data'].copy()

    # Calculate expected values
    pij = 2 * d['rel_cov'][d['rows']] * d['rel_cov'][d['cols']] * d['frac_random']
    expected = binom.sf(data_copy, d['total_obs'], pij)

    # Run the function
    fast_norm_gothic(d['rows'], d['cols'], data_copy, d['rel_cov'], d['total_obs'], d['frac_random'], mode='binomial')

    # Assert that the data was modified correctly
    np.testing.assert_allclose(data_copy, expected, rtol=TOL)


def test_fast_norm_gothic_poisson_mode(gothic_test_data):
    """Tests fast_norm_gothic in 'poisson' mode."""
    d = gothic_test_data
    data_copy = d['data'].copy()

    # Calculate expected values
    pij = 2 * d['rel_cov'][d['rows']] * d['rel_cov'][d['cols']] * d['frac_random']
    lambda_val = d['total_obs'] * pij
    expected = poisson.sf(data_copy, lambda_val)

    # Run the function
    fast_norm_gothic(d['rows'], d['cols'], data_copy, d['rel_cov'], d['total_obs'], d['frac_random'], mode='poisson')

    # Assert that the data was modified correctly
    np.testing.assert_allclose(data_copy, expected, rtol=TOL)


def test_fast_norm_gothic_invalid_mode_raises_exception(gothic_test_data):
    """Tests that an unsupported mode raises an ApplicationException."""
    d = gothic_test_data
    data_copy = d['data'].copy()

    with pytest.raises(ApplicationException, match=r'unsupported mode \[invalid_mode\]'):
        fast_norm_gothic(d['rows'], d['cols'], data_copy, d['rel_cov'], d['total_obs'], d['frac_random'],
                         mode='invalid_mode')


@pytest.mark.parametrize("coords, bins, expected", [
    # Standard case
    (np.array([5, 15, 25, 35]), np.array([[0, 10], [10, 20], [20, 30]]), np.array([1, 1, 1])),
    # Coords on bin edges (lower bound inclusive)
    (np.array([10, 20, 30]), np.array([[10, 20], [20, 30], [30, 40]]), np.array([1, 1, 1])),
    # Empty bin
    (np.array([5, 25]), np.array([[0, 10], [10, 20], [20, 30]]), np.array([1, 0, 1])),
])
def test_count_bin_sites(coords, bins, expected):
    """Tests the count_bin_sites function with various inputs."""
    result = count_bin_sites(coords, bins)
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize("coords, bins, expected", [
    # No coords
    (np.array([], dtype='i8'), np.array([[0, 10], [10, 20]]), np.array([0, 0])),
    # No bins
    (np.array([5, 15], dtype='i8'), np.array([[],[]], dtype='i8'), np.array([], dtype='i8'))
])
def test_count_bin_sites_invalid(coords, bins, expected):
    """Tests the count_bin_sites function with various inputs."""
    with pytest.raises(ValueError, match=r'neither coords nor bins parameters can be empty'):
        count_bin_sites(coords, bins)

# --- Tests for fast_norm_bysite ---

def test_fast_norm_bysite_inplace_modification():
    """Tests that fast_norm_bysite correctly modifies the data array in-place."""
    rows = np.array([0, 1, 2])
    cols = np.array([1, 2, 0])
    data = np.array([10.0, 20.0, 30.0])
    sites = np.array([2.0, 5.0, 4.0])

    data_copy = data.copy()

    # Expected values calculation:
    # data[0] = 10.0 / (sites[0] * sites[1]) = 10.0 / (2.0 * 5.0) = 1.0
    # data[1] = 20.0 / (sites[1] * sites[2]) = 20.0 / (5.0 * 4.0) = 1.0
    # data[2] = 30.0 / (sites[2] * sites[0]) = 30.0 / (4.0 * 2.0) = 3.75
    expected = np.array([1.0, 1.0, 3.75])

    fast_norm_bysite(rows, cols, data_copy, sites)
    np.testing.assert_allclose(data_copy, expected, rtol=TOL)

# --- Tests for fast_length_norm ---

def test_fast_length_norm_inplace_modification():
    """Tests that fast_length_norm correctly modifies the data array in-place."""
    row = np.array([0, 1], dtype=np.int64)
    col = np.array([1, 2], dtype=np.int64)
    data = np.array([100.0, 150.0], dtype=np.float64)
    nnz = len(data)
    len_lookup = np.array([1000, 2000, 1000], dtype=np.float64)

    data_copy = data.copy()

    # Expected values calculation (using arithmetic_mean):
    # w_01 = 1e-3 * 0.5 * (1000 + 2000) = 1.5
    # data[0] = 100.0 / 1.5 = 66.666...
    # w_12 = 1e-3 * 0.5 * (2000 + 1000) = 1.5
    # data[1] = 150.0 / 1.5 = 100.0
    expected = np.array([66.66666667, 100.0])

    fast_length_norm(row, col, data_copy, nnz, len_lookup, arithmetic_mean)
    np.testing.assert_allclose(data_copy, expected, rtol=TOL)

# --- Tests for bin_indices ---

@pytest.fixture
def cumulative_bins_data():
    """Provides a cumulative bin array for testing bin_indices."""
    # Represents original bins at [0, 10), [10, 30), [30, 60), [60, 100)
    return np.array([10, 30, 60, 100])

@pytest.mark.parametrize("i, j, expected_bi, expected_bj", [
    (5, 15, 0, 1),      # Standard case
    (0, 9, 0, 0),       # First bin
    (9, 10, 0, 1),      # Edges (searchsorted side='right')
    (10, 29, 1, 1),     # Values in the same bin
    (60, 99, 3, 3),     # Last bin
    (100, 101, 4, 4)    # Out of bounds (past the last bin)
])
def test_bin_indices(cumulative_bins_data, i, j, expected_bi, expected_bj):
    """Tests the bin_indices function for converting extent-map to sequence-map indices."""
    bi, bj = bin_indices(i, j, cumulative_bins_data)
    assert bi == expected_bi
    assert bj == expected_bj

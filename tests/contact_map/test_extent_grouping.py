from collections import namedtuple

import numpy as np
import pytest

from proxigenomics_toolkit.contact_map.contact_map import ExtentGrouping
from proxigenomics_toolkit.exceptions import ZeroLengthException

# A simple structure to mock sequence information objects for testing
MockSeqInfo = namedtuple("MockSeqInfo", ["id", "length"])

@pytest.fixture
def seq_info_provider():
    """Provides a factory for creating sequence info lists for tests."""
    def _provider(configs):
        # The ExtentGrouping constructor expects a list of objects with 'id' and 'length' attributes.
        return [MockSeqInfo(id=f"seq{i}", length=length) for i, length in enumerate(configs)]
    return _provider

class TestExtentGrouping:
    """Unit tests for the ExtentGrouping class."""

    def test_init_perfect_division(self, seq_info_provider):
        """Tests initialization with a sequence length perfectly divisible by bin_size."""
        seq_info = seq_info_provider([1000])
        grouping = ExtentGrouping(seq_info, 100)

        assert grouping.total_bins == 10
        assert grouping.bin_size == 100
        np.testing.assert_array_equal(grouping.bins, [10])
        assert len(grouping.map) == 1
        assert grouping.map[0].shape == (10, 2)
        np.testing.assert_array_equal(grouping.borders[0], [0, 10])

    def test_init_small_remainder_contraction(self, seq_info_provider):
        """Tests bin contraction when the length remainder is small (< 0.5 bins)."""
        seq_info = seq_info_provider([1040]) # 1040 / 100 = 10.4, should result in 10 bins
        grouping = ExtentGrouping(seq_info, 100)

        assert grouping.total_bins == 10
        np.testing.assert_array_equal(grouping.bins, [10])

    def test_init_large_remainder_expansion(self, seq_info_provider):
        """Tests bin expansion when the length remainder is large (>= 0.5 bins)."""
        seq_info = seq_info_provider([1050]) # 1050 / 100 = 10.5, should result in 11 bins
        grouping = ExtentGrouping(seq_info, 100)

        assert grouping.total_bins == 11
        np.testing.assert_array_equal(grouping.bins, [11])

    def test_init_length_smaller_than_bin_size(self, seq_info_provider):
        """Tests initialization with a sequence shorter than the bin size, expecting one bin."""
        seq_info = seq_info_provider([50])
        grouping = ExtentGrouping(seq_info, 100)

        assert grouping.total_bins == 1
        np.testing.assert_array_equal(grouping.bins, [1])
        assert grouping.get_bin_lengths()[0] == 50

    def test_init_multiple_sequences(self, seq_info_provider):
        """Tests initialization with multiple sequences, checking cumulative totals."""
        seq_info = seq_info_provider([1000, 560]) # seq1: 10 bins, seq2: 6 bins (from 5.6)
        grouping = ExtentGrouping(seq_info, 100)

        assert grouping.total_bins == 16
        np.testing.assert_array_equal(grouping.bins, [10, 6])
        np.testing.assert_array_equal(grouping.borders[0], [0, 10])
        np.testing.assert_array_equal(grouping.borders[1], [10, 16])
        assert np.sum(grouping.get_bin_lengths()) == 1560

    def test_init_zero_length_sequence_raises_exception(self, seq_info_provider):
        """Tests that a ZeroLengthException is raised for a zero-length sequence."""
        seq_info = seq_info_provider([100, 0])
        with pytest.raises(ZeroLengthException, match="Sequence \\[seq1\\] has zero length"):
            ExtentGrouping(seq_info, 100)

    def test_calc_borders(self, seq_info_provider):
        """Tests the calculation of genomic coordinate borders for bins."""
        # 250 / 100 = 2.5, which should expand to 3 bins.
        # np.linspace(0, 250, 4) -> [0, 83, 166, 250]
        seq_info = seq_info_provider([250])
        grouping = ExtentGrouping(seq_info, 100)

        expected_borders = np.array([[0, 83], [83, 166], [166, 250]])
        actual_borders = grouping.calc_borders(0)

        np.testing.assert_array_equal(actual_borders, expected_borders)

    def test_get_bin_lengths(self, seq_info_provider):
        """Tests the calculation of individual bin lengths across all sequences."""
        # seq1: 200 / 100 = 2 bins
        # seq2: 150 / 100 = 1.5 -> expands to 2 bins
        seq_info = seq_info_provider([200, 150])
        grouping = ExtentGrouping(seq_info, 100)

        assert grouping.total_bins == 4

        # Expected lengths: seq1 -> [100, 100], seq2 -> [75, 75]
        expected_lengths = np.array([100, 100, 75, 75])
        actual_lengths = grouping.get_bin_lengths()

        np.testing.assert_array_equal(actual_lengths, expected_lengths)
        assert np.sum(actual_lengths) == 350

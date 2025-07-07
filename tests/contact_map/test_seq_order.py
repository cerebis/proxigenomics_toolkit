from collections import namedtuple

import numpy as np
import pytest

from proxigenomics_toolkit.contact_map.contact_map import SeqOrder
from proxigenomics_toolkit.types import STRUCT_NPTYPE, INDEX_NPTYPE

# Create a simple mock object for sequence information
MockSeqInfo = namedtuple("MockSeqInfo", ["name", "length"])

@pytest.fixture
def seq_info_list():
    """Provides a standard list of mock sequence information for testing."""
    return [
        MockSeqInfo("seqA", 1000),
        MockSeqInfo("seqB", 2000),
        MockSeqInfo("seqC", 1500),
        MockSeqInfo("seqD", 500),
    ]

@pytest.fixture
def seq_order_instance(seq_info_list):
    """Provides a fresh SeqOrder instance for each test."""
    return SeqOrder(seq_info_list)

class TestSeqOrder:
    """Unit tests for the SeqOrder class."""

    def test_initialization(self, seq_order_instance):
        """Tests that SeqOrder initializes correctly with a list of sequences."""
        assert len(seq_order_instance.order) == 4
        assert seq_order_instance.order.shape == (4,)
        assert seq_order_instance.order.dtype == STRUCT_NPTYPE

        # Check initial state of the structured array
        np.testing.assert_array_equal(seq_order_instance.order['pos'], [0, 1, 2, 3])
        np.testing.assert_array_equal(seq_order_instance.order['ori'], [1, 1, 1, 1])
        np.testing.assert_array_equal(seq_order_instance.order['mask'], [True, True, True, True])
        np.testing.assert_array_equal(seq_order_instance.order['length'], [1000, 2000, 1500, 500])

    def test_counting_functions(self, seq_order_instance):
        """Tests count_accepted and count_excluded."""
        assert seq_order_instance.count_accepted() == 4
        assert seq_order_instance.count_excluded() == 0

        # Mask two sequences and re-test
        seq_order_instance.order['mask'][[1, 3]] = False
        assert seq_order_instance.count_accepted() == 2
        assert seq_order_instance.count_excluded() == 2

    def test_masking_operations(self, seq_order_instance):
        """Tests mask_vector, mask, and new_mask methods."""
        # Test mask_vector
        np.testing.assert_array_equal(seq_order_instance.mask_vector(), [True, True, True, True])

        # Test mask
        new_mask_arr = np.array([True, False, True, False])
        seq_order_instance.mask(1)
        seq_order_instance.mask(3)
        np.testing.assert_array_equal(seq_order_instance.order['mask'], new_mask_arr)
        assert seq_order_instance.count_accepted() == 2

        # Test new_mask
        true_mask = seq_order_instance.new_mask(True)
        np.testing.assert_array_equal(true_mask, [True, True, True, True])
        false_mask = seq_order_instance.new_mask(False)
        np.testing.assert_array_equal(false_mask, [False, False, False, False])

    def test_accepted_and_excluded_indices(self, seq_order_instance):
        """Tests the accepted and excluded methods."""
        seq_order_instance.order['mask'][[0, 2]] = False
        
        np.testing.assert_array_equal(seq_order_instance.accepted(), [1, 3])
        np.testing.assert_array_equal(seq_order_instance.excluded(), [0, 2])

    def test_flip(self, seq_order_instance):
        """Tests the flip method for changing sequence orientation."""
        indices_to_flip = [0, 3]
        for _id in indices_to_flip:
            seq_order_instance.flip(_id)

        expected_orientation = np.array([-1, 1, 1, -1])
        np.testing.assert_array_equal(seq_order_instance.order['ori'], expected_orientation)

        # Flipping again should revert to the original orientation
        for _id in indices_to_flip:
            seq_order_instance.flip(_id)
        np.testing.assert_array_equal(seq_order_instance.order['ori'], [1, 1, 1, 1])

    def test_lengths(self, seq_order_instance, seq_info_list):
        """Tests the lengths method."""
        all_lengths = np.array([1000, 2000, 1500, 500])
        np.testing.assert_array_equal(seq_order_instance.lengths(), all_lengths)

        mask_indices = [0, 2]
        subset_indices = [1, 3]
        # mask off 2
        for _id in mask_indices:
            seq_order_instance.mask(_id)

        np.testing.assert_array_equal(seq_order_instance.lengths(True), all_lengths[subset_indices])

        # unmask
        for _id in mask_indices:
            seq_order_instance.unmask(_id)
        np.testing.assert_array_equal(seq_order_instance.lengths(True), all_lengths)
    
    def test_shuffle(self, seq_order_instance):
        """Tests the shuffle method."""
        np.random.seed(42) # for reproducibility
        
        original_positions = seq_order_instance.order['pos'].copy()
        seq_order_instance.shuffle()
        shuffled_positions = seq_order_instance.order['pos']

        # The order should have changed
        assert not np.array_equal(original_positions, shuffled_positions)
        # The set of positions should be the same
        np.testing.assert_array_equal(np.sort(original_positions), np.sort(shuffled_positions))
        # Ensure that the internal positions are also updated
        assert not np.array_equal(original_positions, seq_order_instance.all_positions())

    def test_ordering_queries(self, seq_order_instance):
        """Tests the 'before' and 'intervening' methods with a custom order."""
        # Manually set a new order: seq3, seq0, seq2, seq1
        # Original indices:         3,    0,    2,    1
        # Corresponding positions:  0,    1,    2,    3
        seq_order_instance.set_order_only([1, 3, 2, 0])

        # Test before()
        # Sequences before original index 1 (which is at pos 3) are 3, 0, 2.
        assert seq_order_instance.before(1, 3), 'Sequences index 1 should be before 3'
        assert not seq_order_instance.before(2, 3), 'Sequences index 2 should not be before 3'
        assert seq_order_instance.before(1, 0), 'Sequences index 1 should not be before 0'

        # Test intervening()
        # Sequence(s) between original index 3 (pos 0) and 2 (pos 2) is index 0 (pos 1).
        intervening_sequences = seq_order_instance.intervening(1, 3)
        np.testing.assert_array_equal(intervening_sequences, [0])
        intervening_sequences = seq_order_instance.intervening(1, 2)
        np.testing.assert_array_equal(intervening_sequences, [500])

    def test_gapless_remapping(self, seq_order_instance):
        """Tests gapless_positions and remap_gapless with a masked order."""
        # Mask sequences at original indices 1 and 3
        seq_order_instance.set_mask_only([True, False, True, False])

        # Test gapless_positions
        # The accepted sequences are at original indices 0 and 2. Their positions are also 0 and 2.
        # After masking, _update_positions is called, and their new dense positions will be 0 and 1.
        # gapless_positions should return the original indices sorted by their new position.
        expected_gapless_pos = np.array([0, 1])
        np.testing.assert_array_equal(seq_order_instance.gapless_positions(), expected_gapless_pos)

        # Test remap_gapless
        # A dense index array [1, 0] means the second accepted item (original index 2) should come
        # before the first accepted item (original index 0).
        gapless_indices = np.array([1, 0])
        remapped = seq_order_instance.remap_gapless(gapless_indices)
        np.testing.assert_array_equal(remapped, [2, 0])

        # Test remap_gapless with INDEX_NPTYPE
        gapless_indices_oriented = np.array([(1, -1), (0, 1)], dtype=INDEX_NPTYPE)
        remapped_oriented = seq_order_instance.remap_gapless(gapless_indices_oriented)
        expected_oriented = np.array([(2, -1), (0, 1)], dtype=INDEX_NPTYPE)
        np.testing.assert_array_equal(remapped_oriented, expected_oriented)

    def test_gapless_remapping_edge_cases(self, seq_order_instance):
        """Tests gapless remapping when no or all sequences are masked."""
        # Case 1: No sequences masked
        assert seq_order_instance.count_accepted() == 4
        np.testing.assert_array_equal(seq_order_instance.gapless_positions(), [0, 1, 2, 3])
        seq_order_instance.mask(1)
        np.testing.assert_array_equal(seq_order_instance.remap_gapless([1, 2, 0]), [2, 3, 0])

        # Case 2: All sequences masked
        seq_order_instance.set_mask_only([False, False, False, False])
        assert seq_order_instance.count_accepted() == 0
        assert seq_order_instance.gapless_positions().shape == (0,)
        assert seq_order_instance.remap_gapless([]).shape == (0,)
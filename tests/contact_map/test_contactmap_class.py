from collections import OrderedDict
from typing import List
from unittest.mock import MagicMock

import Bio.SeqIO
import numpy as np
import pysam
import pytest
import scipy.sparse as sp
import sparse

import proxigenomics_toolkit.types
from proxigenomics_toolkit.contact_map.contact_map import ContactMap, SeqOrder
from proxigenomics_toolkit.exceptions import NoneAcceptedException
from proxigenomics_toolkit.seq_utils import SiteCounter


@pytest.fixture
def create_mock_fasta(tmp_path):

    def _factory(seed: int,
                 num_seqs: int,
                 seq_len: int,
                 enzyme_site_seqs: List[str],
                 num_sites: int,
                 prob_n: float = 1e-2) -> str:
        """
        Generates a mock multi-FASTA file with random DNA sequences.

        Can optionally embed a specified number of non-overlapping recognition sites
        into each sequence.

        Args:
            seed (int): The seed for the random number generator.
            num_seqs (int): The number of sequences to generate.
            seq_len (int): The length of each DNA sequence.
            enzyme_site_seqs (List[str]): The DNA sequence of the recognition sites to
                                   embed (e.g., 'GATC').
            num_sites (int): The number of times of each enzyme to embed in each
                             sequence.
            prob_n (float, optional): The probability of a degenerate site (N)
        """
        # Define the DNA alphabet and their corresponding weights.
        # 'N' has a much lower probability (2%) compared to A, C, G, T (24.5% each).
        bases = ["A", "C", "G", "T", "N"]
        base_weight = (1 - prob_n) / 4
        weights = [base_weight] * 4 + [prob_n]

        random_state = np.random.RandomState(seed)

        file_path = tmp_path / "test_data.fasta"

        try:
            with open(file_path, "w") as f:
                for i in range(num_seqs):
                    # Create a unique header for each sequence
                    header = f">sequence_{i + 1}_mock_dna\n"
                    f.write(header)

                    # Generate the initial random sequence
                    sequence = "".join(random_state.choice(bases, size=seq_len, p=weights))

                    # If a recognition site is provided, embed it
                    for site in enzyme_site_seqs:
                        site_len = len(site)
                        if num_sites * site_len > seq_len:
                            raise ValueError("Total length of recognition sites exceeds sequence length.")

                        # Convert to list for mutable operations
                        sequence_list = list(sequence)

                        # Get all possible start indices for the site
                        available_indices = list(range(seq_len - site_len + 1))

                        for _ in range(num_sites):
                            if not available_indices:
                                # This can happen if seq_length is small and num_sites is large
                                raise ValueError(
                                    f"Could not place {num_sites} non-overlapping sites of length {site_len} "
                                    f"in a sequence of length {seq_len}."
                                )

                            # Choose a random start position from the available spots
                            start_pos = random_state.choice(available_indices)

                            # Overwrite the sequence with the recognition site
                            for j in range(site_len):
                                sequence_list[start_pos + j] = site[j]

                            # Remove all indices from the available list that would now cause an overlap.
                            # An overlap occurs if a new site starts anywhere from
                            # (start_pos - site_len + 1) to (start_pos + site_len - 1).
                            invalid_start = start_pos - site_len + 1
                            invalid_end = start_pos + site_len - 1
                            available_indices = [
                                idx for idx in available_indices if not (invalid_start <= idx <= invalid_end)
                            ]

                        sequence = "".join(sequence_list)

                    # Write the final sequence
                    f.write(sequence + "\n")

            return str(file_path)

        except IOError as e:
            print(f"Error writing to file {file_path}: {e}")
            raise e

    return _factory


# Define a fixture to provide a temporary directory and file paths
@pytest.fixture
def contact_map_assets(tmp_path):
    """Provides file paths for mock FASTA and BAM files."""
    fasta_path = tmp_path / "test.fasta"
    bam_path = tmp_path / "test.bam"
    return {"fasta": str(fasta_path), "bam": str(bam_path)}


def create_mock_aligned_segment(mocker, query_name, ref_id, ref_pos, mapq, align_len,
                                is_unmapped=False,
                                is_duplicate=False,
                                is_secondary=False,
                                is_supplementary=False,
                                is_reverse=False):
    """
    Helper function to create a comprehensive mock pysam.AlignedSegment that
    satisfies the filtering logic within ContactMap._bin_map.
    """
    seg = mocker.MagicMock(spec=pysam.AlignedSegment)

    seg.query_name = query_name
    seg.reference_id = ref_id
    seg.reference_start = ref_pos
    if is_reverse:
        seg.reference_end = ref_pos - align_len
    else:
        seg.reference_end = ref_pos + align_len
    seg.mapping_quality = mapq

    # Boolean flags checked by _bin_map
    seg.is_unmapped = is_unmapped
    seg.is_secondary = is_secondary
    seg.is_supplementary = is_supplementary
    seg.is_duplicate = is_duplicate
    seg.is_reverse = is_reverse

    # Alignment properties checked by _bin_map
    seg.query_alignment_length = align_len
    seg.query_length = align_len  # Assuming no soft clipping for simplicity
    seg.cigarstring = f"{align_len}M"
    seg.cigartuples = [(0, align_len)]

    # Mock the get_tag method, crucial for the edit distance ('NM') filter
    seg.get_tag = mocker.MagicMock(return_value=0)
    return seg


@pytest.fixture
def binned_contact_map(mocker, tmp_path, create_mock_fasta):
    """
    A factory fixture that produces fully initialized ContactMap instances by
    running the _bin_map method on a stream of mock AlignedSegment data.
    This provides a realistic, internally consistent object for testing.
    """
    def _factory(seed,
                 num_seqs,
                 ref_len,
                 num_pairs,
                 align_len,
                 bin_size,
                 min_separation,
                 no_duplicates,
                 min_signal,
                 enzymes,
                 tip_size=None,
                 simulated_fasta=False):

        random_state = np.random.RandomState(seed=seed)


        # 2. Mock pysam.AlignmentFile
        mock_bamfile = mocker.MagicMock(spec=pysam.AlignmentFile)
        mock_bamfile.__enter__.return_value = mock_bamfile
        mock_bamfile.header.to_dict.return_value = {'HD': {'SO': 'queryname', "VN": "1.6"}}
        mocker.patch("pysam.AlignmentFile", return_value=mock_bamfile)

        # make actual fake sequences on the filesystem
        if simulated_fasta:
            # get the recogniition site(s) from the supplied enzyme(s)
            enzyme_site_seqs = SiteCounter(*enzymes, tip_size=tip_size).recognition_sites

            fasta_path = create_mock_fasta(seed=seed,
                                           num_seqs=num_seqs,
                                           seq_len=ref_len,
                                           enzyme_site_seqs=enzyme_site_seqs,
                                           num_sites=10)
            seqs = OrderedDict({s.id: len(s.seq) for s in Bio.SeqIO.parse(fasta_path, format='fasta')})
            mock_bamfile.references = list(seqs.keys())
            mock_bamfile.lengths = list(seqs.values())
        # mocked fasta info
        else:
            fasta_path = "dummy.fna"
            # 1. Mock FASTA info
            mock_fasta_info = {
                f'ref_{n+1}': {'length': ref_len,
                               'sites': random_state.poisson(ref_len / 256),
                               'gc': random_state.beta(2, 2),
                               'coords': np.linspace(1, ref_len, 21, endpoint=True, dtype='i8')}
                for n in range(num_seqs)
            }
            mocker.patch.object(ContactMap, 'initialise_fasta_info', return_value=mock_fasta_info)
            mock_bamfile.references = sorted(mock_fasta_info.keys())
            mock_bamfile.lengths = [v['length'] for k, v in sorted(mock_fasta_info.items(), key=lambda x: x[0])]

        # 3. Generate mock AlignedSegments
        mock_alignments = []
        for n in range(num_pairs):
            name = f"read_pair_{n+1}"
            ref_id = random_state.randint(0, num_seqs, size=2)
            ref_pos = random_state.randint(0, ref_len - align_len, size=2)

            is_rev = random_state.randint(2) == 0
            if is_rev:
                ref_pos[1] = ref_pos[1] + align_len

            # Create a pair of reads, occasionally making one reverse orientation
            mock_alignments.append(create_mock_aligned_segment(mocker, name,
                                                               ref_id[0], ref_pos[0],
                                                               60, align_len,
                                                               is_reverse=False))
            mock_alignments.append(create_mock_aligned_segment(mocker, name,
                                                               ref_id[1], ref_pos[1],
                                                               60, align_len,
                                                               is_reverse=is_rev))

        mock_bamfile.fetch.return_value = iter(mock_alignments)

        # 4. Instantiate the REAL ContactMap, which will call _bin_map
        contact_map = ContactMap(
            bam_file="dummy.bam",
            seq_file=fasta_path,
            enzymes=enzymes,
            min_separation=min_separation,
            bin_size=bin_size,
            no_duplicates=no_duplicates,
            min_sig=min_signal,
            tip_size=tip_size,
        )
        return contact_map

    return _factory


class TestBinnedContactMap:
    """Tests using the binned_contact_map fixture."""

    @pytest.fixture
    def sample_contact_map(self, binned_contact_map):
        # Create ContactMap instance
        return binned_contact_map(
            seed=12345,
            num_seqs=5,
            ref_len=10000,
            num_pairs=500,
            align_len=100,
            bin_size=500,
            min_separation=50,
            no_duplicates=True,
            min_signal=0,
            enzymes=["DpnII"],
        )


    @pytest.mark.parametrize("num_seqs", [1, 5])
    @pytest.mark.parametrize(",num_pairs", [0, 1, 50, 500])
    def test_map_creation_and_consistency(self, binned_contact_map, num_seqs, num_pairs):
        """
        Verify that maps are created and are internally consistent.
        """
        cm = binned_contact_map(seed=12345,
                                num_seqs=num_seqs,
                                ref_len=10000,
                                num_pairs=num_pairs,
                                align_len=150,
                                bin_size=1000,
                                min_separation=100,
                                no_duplicates=True,
                                min_signal=0,
                                enzymes=["DpnII"])

        assert cm is not None
        assert cm.min_separation == 100
        assert cm.enzymes == ["DpnII"]
        assert len(cm.seq_info) == num_seqs
        assert isinstance(cm.order, SeqOrder)
        assert cm.seq_map is not None
        assert cm.seq_map.shape == (num_seqs, num_seqs)

        # check the matrices for consistency
        assert sp.isspmatrix(cm.seq_map)
        if num_pairs > 0:
            assert cm.seq_map.nnz > 0
        else:
            assert cm.seq_map.nnz == 0
        assert cm.seq_map.shape == (num_seqs, num_seqs)

        assert sp.isspmatrix(cm.extent_map)
        if num_pairs > 0:
            assert cm.extent_map.nnz > 0
        else:
            assert cm.extent_map.nnz == 0

        # check they have the same sum
        assert np.sum(sp.triu(cm.seq_map)) == np.sum(sp.triu(cm.extent_map))

        # transform the extent map into a seq map
        derived_seq_map = cm.extent_to_seq(cm.extent_map, make_symmetric=True)

        # check that the two matrices have elements are equal (these both should be integer counts)
        assert np.sum(cm.seq_map - derived_seq_map) == 0

    @pytest.mark.parametrize("bin_size", [None, 500, 1000, 10_000])
    def test_prepare_seq_map_from_extent(self, binned_contact_map, bin_size):
        """
        Test the preparation of the processed map. This is generally represented as a non-integer
        matrix, which has had some form of bias correction applied. That said, the number of non-zero
        entries should be the same as the simple counting matrix.

        When bin_size == None, the extent map is not created and therefore you cannot prepare the
        processed map from it. (Tested)
        """
        cm = binned_contact_map(seed=12345,
                                num_seqs=5,
                                ref_len=10000,
                                num_pairs=500,
                                align_len=150,
                                bin_size=bin_size,
                                min_separation=100,
                                no_duplicates=True,
                                min_signal=0,
                                enzymes=["DpnII"])
        if bin_size is None:
            with pytest.raises(AssertionError, match='this instance of ContactMap does not contain an extent-based map.'):
                cm.prepare_seq_map(from_extent=True, norm_method='sites')
        else:
            cm.prepare_seq_map(from_extent=True, norm_method="sites")
            assert cm.processed_map.nnz == cm.seq_map.nnz
            assert cm.processed_map is not None
            assert cm.processed_map.shape == (5, 5)

        try:
            cm.prepare_seq_map(from_extent=False, norm_method='sites')
        except NoneAcceptedException:
            pytest.skip("Skipping test: No sequences accepted after masking.")

        assert cm.processed_map.nnz == cm.seq_map.nnz
        assert cm.processed_map is not None
        assert cm.processed_map.shape == (5, 5)

    def test_initialization_raises_error_for_unsorted_bam(self, mocker, contact_map_assets):
        """
        Tests that ContactMap initialization raises an IOError if the BAM file
        is not sorted by read name.
        """
        mocker.patch.object(ContactMap, 'initialise_fasta_info', return_value={})

        mock_bam_file = mocker.MagicMock()

        # Mock a header that is NOT sorted by query name
        mock_header = {"HD": {"SO": "coordinate"}}
        mock_bam_file.header.to_dict.return_value = mock_header

        mocker.patch("pysam.AlignmentFile", return_value=mock_bam_file)

        with pytest.raises(IOError, match="BAM file must be sorted by read name"):
            ContactMap(
                bam_file=contact_map_assets["bam"],
                seq_file=contact_map_assets["fasta"],
                enzymes=["DpnII"],
                min_separation=10,
                no_duplicates=True,
                min_len=1,
            )

    def test_initialization_raises_error_noheader_bam(self, mocker, contact_map_assets):
        """
        Tests that ContactMap initialization raises an IOError if the BAM file
        is not sorted by read name.
        """
        mocker.patch.object(ContactMap, 'initialise_fasta_info', return_value={})

        mock_bam_file = mocker.MagicMock()

        # Mock a header that is NOT sorted by query name
        mock_header = {}
        mock_bam_file.header.to_dict.return_value = mock_header

        mocker.patch("pysam.AlignmentFile", return_value=mock_bam_file)

        with pytest.raises(IOError, match="BAM file must be sorted by read name"):
            ContactMap(
                bam_file=contact_map_assets["bam"],
                seq_file=contact_map_assets["fasta"],
                enzymes=["DpnII"],
                min_separation=10,
                no_duplicates=True,
                min_len=1,
            )

    def test_make_reverse_index_success(self, sample_contact_map):
        """
        Tests that make_reverse_index correctly creates a lookup dictionary from a unique field.
        """
        # Test with 'name' field
        result = sample_contact_map.make_reverse_index('name')

        # Should return a dict mapping sequence names to their indices
        expected_names = ['ref_1', 'ref_2', 'ref_3', 'ref_4', 'ref_5']
        assert len(result) == 5
        for i, name in enumerate(expected_names):
            assert result[name] == i

    def test_make_reverse_index_non_unique_field_raises_error(self, sample_contact_map):
        """
        Tests that make_reverse_index raises a RuntimeError if the chosen field has duplicates.
        """
        # Manually create seq_info with duplicate values for testing
        from proxigenomics_toolkit.types import SeqInfo

        # Store original seq_info
        original_seq_info = sample_contact_map.seq_info

        # Create seq_info with duplicate lengths
        sample_contact_map.seq_info = [
            SeqInfo(offset=0, refid=0, name='seq_a', length=1000, sites=5, gc=0.5),
            SeqInfo(offset=1, refid=1, name='seq_b', length=1000, sites=5, gc=0.5),  # Same length
            SeqInfo(offset=2, refid=2, name='seq_c', length=2000, sites=10, gc=0.4)
        ]

        with pytest.raises(RuntimeError, match='field contains non-unique entries'):
            sample_contact_map.make_reverse_index('length')

        # Restore original seq_info
        sample_contact_map.seq_info = original_seq_info

    def test_map_weight_returns_sum(self, sample_contact_map):
        """
        Tests that map_weight returns the sum of the sparse matrix.
        """
        weight = sample_contact_map.map_weight()
        assert isinstance(weight, (int, float, np.integer))
        assert weight >= 340  # Weight should be non-negative

    def test_is_empty_reflects_map_state(self, sample_contact_map):
        """
        Tests that is_empty correctly reflects whether the map has zero weight.
        """
        # Test current state
        is_empty = sample_contact_map.is_empty()
        map_weight = sample_contact_map.map_weight()

        # is_empty should be True if and only if map_weight is 0
        assert is_empty == (map_weight == 0)

    def test_is_tipbased_checks_tip_size(self, sample_contact_map):
        """
        Tests that is_tipbased correctly identifies if the map is tip-based.
        """
        # Default ContactMap should not be tip-based
        assert not sample_contact_map.is_tipbased()
        assert sample_contact_map.tip_size is None

        # Manually set tip_size to test positive case
        sample_contact_map.tip_size = 1000
        assert sample_contact_map.is_tipbased()

    def test_has_extent_map_checks_extent_map(self, sample_contact_map):
        """
        Tests that has_extent_map correctly identifies if the map has an extent map.
        """
        # ContactMap when bin_size defined should have an extent map
        assert sample_contact_map.has_extent_map()
        assert sample_contact_map.extent_map is not None

        # Manually set extent_map to the negative
        sample_contact_map.extent_map = None
        assert not sample_contact_map.has_extent_map()

    def test_compress_extent_input_validation(self, mocker, sample_contact_map):
        """
        Tests that _compress_extent validates input types correctly.
        """
        import scipy.sparse as sp

        # Test 1: Non-sparse matrix should raise AssertionError
        dense_matrix = np.array([[1, 2], [3, 4]])
        with pytest.raises(AssertionError, match='Extent matrix is not a scipy sparse matrix type'):
            sample_contact_map._compress_extent(dense_matrix)

        # Test 2: Valid scipy sparse matrices should work
        valid_matrices = [
            sp.csr_matrix([[1, 0], [0, 1]]),
            sp.csc_matrix([[1, 0], [0, 1]]),
            sp.coo_matrix([[1, 0], [0, 1]]),
            sp.lil_matrix([[1, 0], [0, 1]])
        ]

        # for matrix in valid_matrices:
        #     # Should not raise an exception
        #     try:
        #         mocker.patch.object(ContactMap, 'grouping', side_effect=ValueError('valid execution stops here'))
        #         with pytest.raises(ValueError, match='valid execution stops here'):
        #             result = sample_contact_map._compress_extent(matrix)
        #             assert sp.isspmatrix_coo(result), f"Result should be COO matrix for input type {type(matrix)}"
        #     except Exception as e:
        #         pytest.fail(f"Valid sparse matrix type {type(matrix)} raised exception: {e}")

        for matrix in valid_matrices:
            # We want to mock 'self.order' to be a PropertyMock
            # that raises an AttributeError when its 'order' attribute is accessed.
            # Patch sample_contact_map.order directly.
            # The PropertyMock needs to be applied to the *class* of the object
            # if you're trying to mock a property defined with @property.
            # If 'order' is just a regular instance attribute, patching the instance attribute directly is fine.

            # Let's assume 'order' is an instance attribute that might itself be an object with an 'order' attribute.
            # We need to mock sample_contact_map.order to be a MagicMock, and then mock that MagicMock's 'order' attribute.

            mock_order_attribute = mocker.PropertyMock(side_effect=ValueError('Execution stopped at self.order.order'))
            # Patch the 'order' attribute on the instance itself
            mocker.patch.object(sample_contact_map, 'order', new_callable=MagicMock)
            # Now, patch the 'order' attribute *of the mock that replaced sample_contact_map.order*
            # This targets the nested access: sample_contact_map.order.order
            type(sample_contact_map.order).order = mock_order_attribute

            with pytest.raises(ValueError, match='Execution stopped at self.order.order'):
                sample_contact_map._compress_extent(matrix)
                # The assert for sp.isspmatrix_coo(result) will not be reached
                # because the ValueError is raised earlier.

    def test_compress_extent_converts_to_coo(self, sample_contact_map):
        """
        Tests that _compress_extent converts input to COO format.
        """
        import scipy.sparse as sp

        # Create a CSR matrix (not COO)
        csr_matrix = sp.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])

        result = sample_contact_map._compress_extent(csr_matrix)

        # Result should always be COO format
        assert sp.isspmatrix_coo(result), "Result should be in COO format"

    def test_compress_extent_simple_case(self, mocker, contact_map_assets):
        """
        Tests _compress_extent with a simple, controlled case to validate correctness.
        """
        import scipy.sparse as sp

        # Create a minimal ContactMap with controlled setup
        mock_fasta_info = {
            'seq1': {'length': 100, 'sites': 2, 'gc': 0.5, 'coords': np.array([0, 50, 100])},
            'seq2': {'length': 100, 'sites': 2, 'gc': 0.5, 'coords': np.array([0, 50, 100])},
            'seq3': {'length': 100, 'sites': 2, 'gc': 0.5, 'coords': np.array([0, 50, 100])}
        }
        mocker.patch.object(ContactMap, 'initialise_fasta_info', return_value=mock_fasta_info)

        mock_bamfile = mocker.MagicMock(spec=pysam.AlignmentFile)
        mock_bamfile.__enter__.return_value = mock_bamfile
        mock_bamfile.header.to_dict.return_value = {'HD': {'SO': 'queryname', "VN": "1.6"}}
        mock_bamfile.references = sorted(mock_fasta_info.keys())
        mock_bamfile.lengths = [100, 100, 100]
        mock_bamfile.fetch.return_value = iter([])
        mocker.patch("pysam.AlignmentFile", return_value=mock_bamfile)

        contact_map = ContactMap(
            bam_file=contact_map_assets["bam"],
            seq_file=contact_map_assets["fasta"],
            enzymes=["DpnII"],
            min_separation=50,
            min_sig=0,
            bin_size=50  # This should create 2 bins per sequence
        )

        # Create a simple test extent matrix (6x6 for 3 sequences with 2 bins each)
        # Matrix structure: seq1_bin1, seq1_bin2, seq2_bin1, seq2_bin2, seq3_bin1, seq3_bin2
        test_data = [1, 2, 3, 4, 5, 6]
        test_row = [0, 1, 2, 3, 4, 5]
        test_col = [0, 1, 2, 3, 4, 5]
        test_matrix = sp.coo_matrix((test_data, (test_row, test_col)), shape=(6, 6))

        # Mock the order and grouping to control masking behavior
        # Let's say we want to keep sequences 1 and 3, but mask sequence 2
        mock_order = np.array([
            (0, 1, True, 100),   # seq1: masked=True (keep)
            (1, 1, False, 100),  # seq2: masked=False (remove)
            (2, 1, True, 100)    # seq3: masked=True (keep)
        ], dtype=proxigenomics_toolkit.types.STRUCT_NPTYPE)

        contact_map.order.order = mock_order

        # Mock grouping to specify bins per sequence
        mock_grouping = mocker.MagicMock()
        mock_grouping.bins = [2, 2, 2]  # 2 bins per sequence
        mock_grouping.total_bins = 6
        contact_map.grouping = mock_grouping

        # Run the compression
        result = contact_map._compress_extent(test_matrix)

        # Validate result properties
        assert sp.isspmatrix_coo(result), "Result should be COO matrix"
        assert result.shape[0] <= test_matrix.shape[0], "Result should have equal or fewer rows"
        assert result.shape[1] <= test_matrix.shape[1], "Result should have equal or fewer columns"

        # With seq2 masked out, we should have 4 bins remaining (2 from seq1, 2 from seq3)
        expected_size = 4
        assert result.shape == (expected_size, expected_size), f"Expected shape ({expected_size}, {expected_size}), got {result.shape}"

    def test_compress_extent_preserves_data_structure(self, sample_contact_map):
        """
        Tests that _compress_extent preserves the essential structure of sparse matrices.
        """
        import scipy.sparse as sp

        # Create a test matrix with known structure
        data = [1, 2, 3, 4]
        row = [0, 0, 1, 1]
        col = [0, 1, 0, 1]
        test_matrix = sp.coo_matrix((data, (row, col)), shape=(2, 2))

        result = sample_contact_map._compress_extent(test_matrix)

        # Basic structural checks
        assert hasattr(result, 'data'), "Result should have data attribute"
        assert hasattr(result, 'row'), "Result should have row attribute"
        assert hasattr(result, 'col'), "Result should have col attribute"
        assert hasattr(result, 'shape'), "Result should have shape attribute"

        # Data should be numeric
        assert np.issubdtype(result.data.dtype, np.number), "Result data should be numeric"

        # Indices should be integers
        assert np.issubdtype(result.row.dtype, np.integer), "Row indices should be integers"
        assert np.issubdtype(result.col.dtype, np.integer), "Column indices should be integers"

    def test_reorder_extent_input_validation(self, mocker, sample_contact_map):
        """
        Tests that _reorder_extent accepts different sparse matrix types.
        """
        import scipy.sparse as sp

        dense_matrix = np.array([[1, 2], [3, 4]])
        with pytest.raises(AssertionError, match='Extent matrix is not a scipy sparse matrix type'):
            sample_contact_map._compress_extent(dense_matrix)

        # Test various scipy sparse matrix types
        valid_matrices = [
            sp.csr_matrix([[1, 2], [3, 4]]),
            sp.csc_matrix([[1, 2], [3, 4]]),
            sp.coo_matrix([[1, 2], [3, 4]]),
            sp.lil_matrix([[1, 2], [3, 4]])
        ]

        # Prepare the mock for self.order.gapless_positions()
        # This will apply to all iterations in the loop.
        mock_order_instance = MagicMock()
        mock_order_instance.gapless_positions.side_effect = ValueError("Execution stopped at gapless_positions")
        mocker.patch.object(sample_contact_map, 'order', new=mock_order_instance)

        for matrix in valid_matrices:
            # We expect a ValueError because of our mock,
            # meaning the input type was accepted, but internal logic was interrupted.
            with pytest.raises(ValueError, match="Execution stopped at gapless_positions"):
                sample_contact_map._reorder_extent(matrix)

            # No `assert sp.isspmatrix_coo(result)` here because no result is expected
            # when an exception is raised.

    def test_reorder_extent_returns_csr_format(self, sample_contact_map):
        """
        Tests that _reorder_extent always returns CSR format as documented.
        """
        import scipy.sparse as sp

        # Test with different input formats
        _m = sample_contact_map.extent_map
        test_matrices = [
            _m.tocoo(),
            _m.tolil(),
            _m.tocsr(),
        ]

        for matrix in test_matrices:
            result = sample_contact_map._reorder_extent(matrix)
            assert sp.isspmatrix_csr(result), f"Result should be CSR format for input type {type(matrix)}"

    def test_reorder_extent_preserves_shape(self, sample_contact_map):
        """
        Tests that _reorder_extent preserves the shape of the input matrix.
        """
        # Test with different shapes
        test_cases = [
            sample_contact_map.extent_map
        ]

        for matrix in test_cases:
            result = sample_contact_map._reorder_extent(matrix)
            assert result.shape == matrix.shape, \
                f"Shape should be preserved: expected {matrix.shape}, got {result.shape}"

    def test_reorder_extent_simple_case(self, mocker, contact_map_assets):
        """
        Tests _reorder_extent with a controlled simple case to validate correctness.
        """
        import scipy.sparse as sp

        # Create a minimal ContactMap with 2 sequences for controlled testing
        mock_fasta_info = {
            'seq1': {'length': 100, 'sites': 2, 'gc': 0.5, 'coords': np.array([0, 50, 100])},
            'seq2': {'length': 100, 'sites': 2, 'gc': 0.5, 'coords': np.array([0, 50, 100])}
        }
        mocker.patch.object(ContactMap, 'initialise_fasta_info', return_value=mock_fasta_info)

        mock_bamfile = mocker.MagicMock(spec=pysam.AlignmentFile)
        mock_bamfile.__enter__.return_value = mock_bamfile
        mock_bamfile.header.to_dict.return_value = {'HD': {'SO': 'queryname', "VN": "1.6"}}
        mock_bamfile.references = ['seq1', 'seq2']
        mock_bamfile.lengths = [100, 100]
        mock_bamfile.fetch.return_value = iter([])
        mocker.patch("pysam.AlignmentFile", return_value=mock_bamfile)

        contact_map = ContactMap(
            bam_file=contact_map_assets["bam"],
            seq_file=contact_map_assets["fasta"],
            enzymes=["DpnII"],
            min_separation=50,
            min_sig=0,
            bin_size=50  # 2 bins per sequence
        )

        # Create a test matrix (4x4 for 2 sequences with 2 bins each)
        test_matrix = sp.csr_matrix([
            [1, 2, 3, 4],  # seq1_bin1 interactions
            [5, 6, 7, 8],  # seq1_bin2 interactions
            [9, 10, 11, 12],  # seq2_bin1 interactions
            [13, 14, 15, 16]   # seq2_bin2 interactions
        ])

        # Mock the order and grouping for predictable behavior
        # Let's test identity permutation first (no reordering)
        mock_order = np.array([
            (0, 1, True, 100),   # seq1: position 0, orientation +1
            (1, 1, True, 100)    # seq2: position 1, orientation +1
        ], dtype=proxigenomics_toolkit.types.STRUCT_NPTYPE)

        contact_map.order.order = mock_order

        # Mock the required methods
        contact_map.order.gapless_positions = mocker.MagicMock(return_value=np.array([0, 1]))
        contact_map.order.mask_vector = mocker.MagicMock(return_value=np.array([True, True]))

        mock_grouping = mocker.MagicMock()
        mock_grouping.bins = np.array([2, 2])  # 2 bins per sequence
        contact_map.grouping = mock_grouping

        # Test identity reordering (should return same matrix)
        result = contact_map._reorder_extent(test_matrix)

        # Basic validation
        assert sp.isspmatrix_csr(result), "Result should be CSR matrix"
        assert result.shape == test_matrix.shape, "Shape should be preserved"

        # For identity permutation, result should be close to original
        # (allowing for floating point precision issues)
        np.testing.assert_array_almost_equal(
            result.toarray(),
            test_matrix.toarray(),
            err_msg="Identity permutation should preserve matrix values"
        )

    def test_reorder_extent_reverse_orientation(self, mocker, contact_map_assets):
        """
        Tests _reorder_extent with reverse orientation to validate orientation handling.
        """
        import scipy.sparse as sp

        # Create a minimal ContactMap
        mock_fasta_info = {
            'seq1': {'length': 100, 'sites': 2, 'gc': 0.5, 'coords': np.array([0, 50, 100])}
        }
        mocker.patch.object(ContactMap, 'initialise_fasta_info', return_value=mock_fasta_info)

        mock_bamfile = mocker.MagicMock(spec=pysam.AlignmentFile)
        mock_bamfile.__enter__.return_value = mock_bamfile
        mock_bamfile.header.to_dict.return_value = {'HD': {'SO': 'queryname', "VN": "1.6"}}
        mock_bamfile.references = ['seq1']
        mock_bamfile.lengths = [100]
        mock_bamfile.fetch.return_value = iter([])
        mocker.patch("pysam.AlignmentFile", return_value=mock_bamfile)

        contact_map = ContactMap(
            bam_file=contact_map_assets["bam"],
            seq_file=contact_map_assets["fasta"],
            enzymes=["DpnII"],
            min_separation=50,
            min_sig=0,
            bin_size=50
        )

        # Create a simple 2x2 test matrix
        test_matrix = sp.csr_matrix([
            [1, 2],
            [3, 4]
        ])

        # Mock for reverse orientation
        mock_order = np.array([
            (0, -1, True, 100)   # seq1: position 0, orientation -1 (reverse)
        ], dtype=proxigenomics_toolkit.types.STRUCT_NPTYPE)

        contact_map.order.order = mock_order
        contact_map.order.gapless_positions = mocker.MagicMock(return_value=np.array([0]))
        contact_map.order.mask_vector = mocker.MagicMock(return_value=np.array([True]))

        mock_grouping = mocker.MagicMock()
        mock_grouping.bins = np.array([2])  # 2 bins for the sequence
        contact_map.grouping = mock_grouping

        result = contact_map._reorder_extent(test_matrix)

        # Validate result properties
        assert sp.isspmatrix_csr(result), "Result should be CSR matrix"
        assert result.shape == test_matrix.shape, "Shape should be preserved"

        # With reverse orientation, the matrix should be different from original
        # (specific values depend on the permutation logic)
        result_dense = result.toarray()
        original_dense = test_matrix.toarray()

        # Should have same total sum (conservation of data)
        assert np.sum(result_dense) == np.sum(original_dense), "Total sum should be conserved"

    def test_reorder_extent_matrix_properties(self, sample_contact_map):
        """
        Tests that _reorder_extent preserves important matrix properties.
        """
        import scipy.sparse as sp

        test_matrix = sample_contact_map.extent_map
        result = sample_contact_map._reorder_extent(test_matrix)

        # Basic properties
        assert sp.isspmatrix_csr(result), "Result should be CSR matrix"
        assert result.shape == test_matrix.shape, "Shape should be preserved"

        # Data should be numeric
        assert np.issubdtype(result.data.dtype, np.number), "Result data should be numeric"

        # Total sum should be preserved (permutation preserves sum)
        original_sum = test_matrix.sum()
        result_sum = result.sum()
        np.testing.assert_almost_equal(
            result_sum, original_sum,
            err_msg="Permutation should preserve total sum"
        )

    # @pytest.mark.parametrize("method", ["sites", "length", "gothic"])
    @pytest.mark.parametrize(('tip_based','method'),
                             [(False, 'length'),
                              (False, 'sites'),
                              (False, 'gothic'),
                              (True, 'length'),
                              (True, 'sites'),
                              pytest.param(True, 'gothic',
                                           marks=pytest.mark.xfail(reason="Gothic method not implemented yet"))
                              ])
    def test_norm_seq_returns_coo_matrix_when_tip_based_is_false(self, binned_contact_map, tip_based, method):
        """
        Tests that _norm_seq returns a coo_matrix when tip_based is False.
        """
        contact_map = binned_contact_map(seed=12345,
                                num_seqs=5,
                                ref_len=10000,
                                num_pairs=500,
                                align_len=150,
                                bin_size=None,
                                min_separation=100,
                                no_duplicates=True,
                                min_signal=0,
                                enzymes=["DpnII"],
                                tip_size=1000 if tip_based else None,
                                simulated_fasta=True)

        normalized_map = contact_map._norm_seq(contact_map.seq_map, tip_based=tip_based, method=method)
        if tip_based:
            assert isinstance(normalized_map, sparse.COO), "Result should be of type sparse.COO"
        else:
            assert sp.isspmatrix_coo(normalized_map), "Result should be of type scipy.sparse.coo_matrix"
        assert normalized_map.shape[0] == contact_map.total_seq
        assert normalized_map.shape[1] == contact_map.total_seq

    def test_get_sites(self, sample_contact_map):
        sites = sample_contact_map._get_sites()
        assert isinstance(sites, np.ndarray)
        assert sites.dtype == np.float64
        assert sites.shape[0] == sample_contact_map.total_seq
        assert np.all(sites >= 1)

    def test_bisto_seq(self, sample_contact_map):
        bisto_map, bisto_scale = sample_contact_map._bisto_seq(sample_contact_map.seq_map)
        assert isinstance(bisto_map, sp.coo_matrix)
        assert isinstance(bisto_scale, np.ndarray)
        assert bisto_map.shape[0] == sample_contact_map.total_seq
        assert bisto_map.shape[1] == sample_contact_map.total_seq
        assert bisto_scale.shape[0] == sample_contact_map.total_seq

    def test_reorder_seq_returns_coo_matrix(self, sample_contact_map):
        """
        Tests that _reorder_seq returns a coo_matrix.
        """
        input_matrix = sample_contact_map.seq_map
        reordered_map = sample_contact_map._reorder_seq(input_matrix)
        assert isinstance(reordered_map, sp.coo_matrix)
        assert reordered_map.shape == input_matrix.shape

    def test_reorder_seq_correctly_reorders(self, binned_contact_map, mocker):
        """
        Tests that _reorder_seq correctly reorders the matrix.
        """

        contact_map = binned_contact_map(
            seed=42, num_seqs=4, ref_len=1000, num_pairs=100, align_len=100,
            bin_size=None, min_separation=0, no_duplicates=True, min_signal=0,
            enzymes=['DpnII']
        )
        # mock the the input and order matrices so as to guarantee
        # the predicted outcome
        coords = [[0, 1, 2, 3], [1, 0, 3, 2]]
        data = [10, 20, 30, 40]
        input_matrix = sp.coo_matrix((data, coords), shape=(4, 4))
        reorder_array = np.array([1, 0, 3, 2])
        mocker.patch.object(contact_map, 'order', mocker.MagicMock())
        contact_map.order.gapless_positions.return_value = reorder_array

        # call reordering
        reordered_map = contact_map._reorder_seq(_map=input_matrix)

        expected_matrix = np.array([
            [0, 20, 0, 0],
            [10, 0, 0, 0],
            [0, 0, 0, 40],
            [0, 0, 30, 0]
        ])

        np.testing.assert_array_equal(reordered_map.toarray(), expected_matrix)

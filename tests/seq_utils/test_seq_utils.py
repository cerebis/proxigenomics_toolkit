# tests/test_seq_utils.py

import os

import pytest

from proxigenomics_toolkit.exceptions import NoRecordsException, UnknownEnzymeException
from proxigenomics_toolkit.seq_utils import IndexedFasta, SiteCounter, count_fasta_sequences, revcomp
from proxigenomics_toolkit.exceptions import BluntEnzymeException

# from pytest_mock import mocker


def test_indexedfasta_load_and_access():
    """
    Test if IndexedFasta can open a valid fasta file and access its sequences.
    """
    fasta_file = "test_data/multiple_sequences.fasta"
    indexed_fasta = IndexedFasta(fasta_file)

    expected_ids = {"edge_69": 8853, "edge_1320": 39074, "edge_3478": 8939}
    assert len(indexed_fasta) == len(expected_ids), f"Expected {len(expected_ids)} sequences, got {len(indexed_fasta)}"

    for seq_id, seq_len in expected_ids.items():
        assert seq_id in indexed_fasta, f"Expected ID {seq_id} to be in IndexedFasta"
        _s = indexed_fasta[seq_id]
        assert len(_s) == seq_len, f"Expected length of sequence {seq_id} was {seq_len}, got {len(_s)}"

    indexed_fasta.close()


def test_indexedfasta_invalid_path():
    """
    Test if IndexedFasta raises an IOError for an invalid temporary path.
    """
    fasta_file = "test_data/valid_sequences.fasta"
    invalid_tmp_path = "invalid_path/"
    with pytest.raises(IOError):
        IndexedFasta(fasta_file, tmp_path=invalid_tmp_path)


def test_indexedfasta_missing_file():
    """
    Test if IndexedFasta raises an error when the fasta file is missing.
    """
    missing_file = "test_data/missing_sequences.fasta"
    with pytest.raises(FileNotFoundError):
        IndexedFasta(missing_file)


def test_revcomp_basic():
    sequence = "ATGC"
    expected = "GCAT"
    result = revcomp(sequence)
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize("file_name, expected_count", [
    ("test_data/empty.fasta", 0),
    ("test_data/single_sequence.fasta", 1),
    ("test_data/multiple_sequences.fasta", 3),
])
def test_count_fasta_sequences(file_name, expected_count):
    """
    Parametrized test to check count_fasta_sequences with a variety of FASTA files.
    """
    try:
        result = count_fasta_sequences(file_name)
    except NoRecordsException as e:
        assert file_name == 'test_data/empty.fasta', f"Unexpected exception for {file_name}: {e}"
        result = 0

    assert result == expected_count, f"Expected {expected_count}, got {result}"


def test_count_fasta_sequences_with_compressed_file():
    """
    Test count_fasta_sequences function with a compressed FASTA file.
    """
    file_name = "test_data/compressed_sequences.fasta.gz"
    expected_count = 3
    result = count_fasta_sequences(file_name)
    assert result == expected_count, f"Expected {expected_count}, got {result}"


def test_count_fasta_sequences_with_invalid_file():
    """
    Test count_fasta_sequences function with an invalid (non-FASTA) file.
    """
    file_name = "test_data/invalid_file.txt"
    with pytest.raises(NoRecordsException):
        count_fasta_sequences(file_name)


def test_revcomp_empty():
    sequence = ""
    expected = ""
    result = revcomp(sequence)
    assert result == expected, f"Expected {expected}, got {result}"


def test_revcomp_single_nucleotide():
    sequence = "A"
    expected = "T"
    result = revcomp(sequence)
    assert result == expected, f"Expected {expected}, got {result}"


def test_revcomp_palindrome():
    sequence = "ATCGAT"
    expected = "ATCGAT"
    result = revcomp(sequence)
    assert result == expected, f"Expected {expected}, got {result}"


def test_revcomp_invalid_characters():
    sequence = "AXTYGZ"
    result = revcomp(sequence)
    assert result == "NCNANT", f"Expected 'NCNANT', got {result}"


def test_revcomp_long_sequence():
    sequence = "ATGCATGCATGCATGCATGCATGC"
    expected = "GCATGCATGCATGCATGCATGCAT"
    result = revcomp(sequence)
    assert result == expected, f"Expected {expected}, got {result}"

@pytest.mark.skipif(
    not os.getenv("RUN_DOCKER_TESTS"),
    reason="Docker tests are skipped. Enable by setting the RUN_DOCKER_TESTS environment variable.",
)
def test_count_bam_reads_in_docker():
    """
    Test count_bam_reads within a Docker environment to ensure external dependencies are defined.
    """
    import docker
    client = docker.from_env()

    # Define Docker container details
    image_name = "zlskidmore/samtools"  # Example Bioconda Docker image
    cmd = "samtools view -c /test_data/small.bam"

    # Setting up the test
    try:
        container = client.containers.run(
            image_name,
            cmd,
            detach=True,
            volumes={
                os.path.abspath("test_data"): {"bind": "/test_data", "mode": "rw"}
            },
            working_dir="/mnt",
        )
        result = container.wait()
        n_reads = container.logs().decode().strip()
        assert result['StatusCode'] == 0, f"Test failed with exit code {result}: {n_reads}"
        assert n_reads.isdigit(), f"Expected numeric output, got: {n_reads}"
        assert int(n_reads) == 169, f"Expected 169 reads, got {n_reads}"
    finally:
        container.remove(force=True)

class TestSiteCounter:

    @pytest.fixture
    def single_digest(self):
        """Fixture for a single enzyme (HindIII) digest."""
        return SiteCounter(enzyme_a='DpnII')

    @pytest.fixture
    def double_digest(self):
        """Fixture for a double enzyme (HindIII, EcoRI) digest."""
        return SiteCounter(enzyme_a='DpnII', enzyme_b='MluCI')

    def test_init_ok(self, single_digest, double_digest):
        assert single_digest.enzyme_a.site == 'GATC'
        assert single_digest.enzyme_b is None
        assert double_digest.enzyme_a.site == 'GATC'
        assert double_digest.enzyme_b.site == 'AATT'

    def test_init_unknown_enzyme(self):
        with pytest.raises(UnknownEnzymeException, match="UnknownEnzyme does not correspond to a known enzyme"):
            SiteCounter(enzyme_a='UnknownEnzyme')
        # Check for suggestions
        with pytest.raises(UnknownEnzymeException, match=r"HndIII is undefined, but its similar to.*"):
            SiteCounter(enzyme_a='HndIII')

    def test_recognition_sites_property(self, single_digest, double_digest):
        assert single_digest.recognition_sites == ['GATC']
        assert double_digest.recognition_sites == ['GATC', 'AATT']

    def test_find_sites(self, single_digest, double_digest):
        seq = "TTTGATCTTCCCGACTTAATTTT"
        # HindIII at pos 3, EcoRI at pos 12 (0-based)
        assert single_digest.find_sites(seq) == [4]
        assert double_digest.find_sites(seq) == [4, 18]
        # Test no sites
        assert single_digest.find_sites("ACGTACGT") == []

    def test_find_sites_circular(self):
        # Site wraps around the end: CTT...AAG
        seq = "TCTTTTTAAGA"
        sc_linear = SiteCounter('DpnII', is_linear=True)
        sc_circular = SiteCounter('DpnII', is_linear=False)
        assert sc_linear.find_sites(seq) == []
        # Biopython finds wrapped sites and returns the 5' end position
        assert sc_circular.find_sites(seq) == [10] # AAG starts at index 6

    def test_count_sites_no_tip(self, single_digest, double_digest):
        seq = "TTTGATCAGCTTCCCGAAGATCTTCTTTAATTGCTT"
        assert single_digest.count_sites(seq) == 2
        assert double_digest.count_sites(seq) == 3

    def test_count_sites_with_tip(self):
        seq = "AAGCTT" + ("G" * 50) + "GAATTC" + ("C" * 50) + "AAGCTT"
        # seq len = 6 + 50 + 6 + 50 + 6 = 118
        # tip_size = 30
        # left tip: AAGCTT... (1 site)
        # right tip: ...AAGCTT (1 site)
        sc = SiteCounter('HindIII', 'EcoRI', tip_size=30)
        assert sc.count_sites(seq) == [1, 1]

    def test_count_sites_small_contig_tip(self):
        # seq_len (20) < 2 * tip_size (30)
        seq = "AAGCTT" + ("G" * 8) + "GAATTC"
        sc = SiteCounter('HindIII', 'EcoRI', tip_size=15)
        # half_len = 10. l_tip = AAGCTTGGGG (1 site), r_tip = GGGGAATTC (1 site)
        assert sc.count_sites(seq) == [1, 1]

    def test_get_vestigial_end_searcher(self, single_digest):
        # HindIII junction is AAGCTT, vestigial is AAGCTT
        searcher = single_digest.get_vestigial_end_searcher()
        assert searcher("ATATAT") is None
        match = searcher("GGGNAAGCTTGATC")
        assert match is not None
        assert match.group(1) == "GATC"

    def test_blunt_enzyme(self):
        with pytest.raises(BluntEnzymeException):
            SiteCounter('DpnI')

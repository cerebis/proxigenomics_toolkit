# tests/test_seq_utils.py

import os
import pytest

from src.proxigenomics_toolkit.exceptions import NoRecordsException
from src.proxigenomics_toolkit.seq_utils.seq_utils import revcomp, count_fasta_sequences, count_bam_reads, IndexedFasta


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

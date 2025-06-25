import numpy as np
import pytest
import scipy.sparse as scisp
import sparse

from proxigenomics_toolkit.linalg import (
    Sparse2DAccumulator,
    Sparse4DAccumulator,
    add_matrices,
    compress,
    compress_4d,
    downsample,
    is_hermitian,
    make_symmetric,
)


@pytest.fixture
def sparse_matrix_2d_coo():
    """Provides a standard 2D COO sparse matrix for testing."""
    coords = [[0, 1, 2, 3],
              [1, 0, 2, 3]]
    data = [10, 20, 30, 40]
    shape = (4, 4)
    return scisp.coo_matrix((data, coords), shape=shape)

@pytest.fixture
def sparse_matrix_4d_coo():
    """Provides a standard 4D COO sparse matrix for testing."""
    coords = [[0, 0, 1, 1],
              [0, 0, 1, 1],
              [0, 1, 0, 1],
              [0, 1, 0, 1]]
    data = [1, 2, 3, 4]
    shape = (2, 2, 2, 2)
    return sparse.COO(coords, data, shape, has_duplicates=False)

def test_add_matrices(sparse_matrix_2d_coo):
    """Tests the addition of two sparse matrices."""
    m1 = sparse_matrix_2d_coo
    m2 = scisp.coo_matrix(([5,5], [[0, 1], [0, 1]]), shape=(4, 4))
    result = add_matrices(m1, m2)
    result_dense = result.todense()

    assert result.shape == (4, 4)
    assert result_dense[0, 1] == 10
    assert result_dense[1, 0] == 20
    assert result_dense[0, 0] == 5
    assert result_dense[1, 1] == 5

def test_is_hermitian():
    """Tests the is_hermitian function with both a Hermitian and a non-Hermitian matrix."""
    # Hermitian matrix
    coords_herm = [[0, 1], [1, 0]]
    data_herm = [1, 1]
    m_herm = sparse.COO(coords_herm, data_herm, shape=(2, 2))
    assert is_hermitian(m_herm)

    # Non-Hermitian matrix
    coords_non_herm = [[0, 1], [1, 1]]
    data_non_herm = [1, 2]
    m_non_herm = sparse.COO(coords_non_herm, data_non_herm, shape=(2, 2))
    assert not is_hermitian(m_non_herm)

def test_make_symmetric(sparse_matrix_2d_coo):
    """Tests if a non-symmetric matrix is correctly made symmetric."""
    result = make_symmetric(sparse_matrix_2d_coo)
    result_dense = result.todense()
    assert result.shape == sparse_matrix_2d_coo.shape
    assert np.all(result_dense == result_dense.T)

def test_downsample():
    """Tests the downsampling of a sparse matrix."""
    coords = [[0, 0, 1, 1, 2, 2, 3, 3, 3, 0],
              [0, 1, 0, 1, 2, 3, 2, 3, 0, 3]]
    data = [1, 2, 3, 4, 5, 6, 7, 8, 1, 4]
    m = scisp.coo_matrix((data, coords), shape=(4, 4), dtype=np.float64)
    downsampled_m = downsample(m, 2)
    dense = downsampled_m.todense()
    assert downsampled_m.shape == (2, 2)
    assert dense[0, 0] == 2.5
    assert dense[1, 1] == 6.5
    assert dense[0, 1] == 1
    assert dense[1, 0] == 0.25

def test_compress():
    """Tests the compression of a sparse matrix using a boolean mask."""
    coords = [[0, 1, 2, 3], [1, 0, 3, 2]]
    data = [1, 2, 3, 4]
    m = scisp.coo_matrix((data, coords), shape=(4, 4), dtype=np.float64)
    mask = np.array([True, False, True, True])
    compressed_m = compress(m, mask)
    expected_coords = [[1, 2], [2, 1]]
    expected_data = [3, 4]
    assert compressed_m.shape == (3, 3)
    np.testing.assert_array_equal(compressed_m.coords, expected_coords)
    np.testing.assert_array_equal(compressed_m.data, expected_data)

def test_compress_4d(sparse_matrix_4d_coo):
    """Tests the compression of a 4D sparse matrix."""
    mask = np.array([True, False])
    result = compress_4d(sparse_matrix_4d_coo, mask)
    expected_coords = [[0, 0], [0, 0], [0, 1], [0, 1]]
    expected_data = [1, 2]

    assert result.shape == (1, 1, 2, 2)
    assert result.nnz == 2
    np.testing.assert_array_equal(result.coords, expected_coords)
    np.testing.assert_array_equal(result.data, expected_data)

def test_sparse2d_accumulator():
    """Tests the functionality of the Sparse2DAccumulator class."""
    acc = Sparse2DAccumulator(5)
    acc[0, 0] = 99
    acc[1, 2] = 10
    acc[3, 4] = 20
    acc[1, 2] += 5

    assert acc[0, 0] == 99
    assert acc[1, 2] == 15
    assert acc[3, 4] == 20

    coo = acc.get_coo()
    dense = coo.todense()
    assert isinstance(coo, scisp.coo_matrix)
    assert coo.shape == (5, 5)
    assert coo.nnz == 5
    assert dense[1, 2] == 15
    assert dense[3, 4] == 20

def test_sparse4d_accumulator():
    """Tests the functionality of the Sparse4DAccumulator class."""
    acc = Sparse4DAccumulator(2)
    acc[0, 0] += np.array([[7,1],[1,7]], dtype=np.uint32)
    acc[0, 1] += np.array([[1,5],[5,1]], dtype=np.uint32)

    coo = acc.get_coo()
    dense = coo.todense()

    assert isinstance(coo, sparse.COO)
    assert coo.shape == (2, 2, 2, 2)
    assert dense[0, 0, 1, 1] == 7
    assert dense[1, 1, 0, 0] == 0
    assert dense[0, 1, 0, 1] == 5

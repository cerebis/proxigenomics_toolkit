import logging
from math import ceil
from typing import Any, Dict, Optional, Tuple

import numba as nb
import numpy as np
import scipy.sparse as scisp
import sparse

from ..types import SparseMatrix

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.INFO)


def add_matrices(a: SparseMatrix, b: SparseMatrix) -> SparseMatrix:
    if isinstance(a, scisp.spmatrix) and isinstance(b, scisp.spmatrix):
        return a.tocsr() + b.tocsr()
    elif isinstance(a, sparse.COO) and isinstance(b, sparse.COO):
        return sparse.elemwise(np.add, a, b)
    else:
        raise ValueError('Adding two different matrix types is not supported')


def is_hermitian(m: np.ndarray | SparseMatrix, tol: float=1e-6) -> bool:
    """
    Test that a sparse matrix is hermitian (also suffices for symmetric)

    :param m: square matrix
    :param tol: tolernace above zero for m - m.T < tol
    :return: True matrix is Hermitian
    """
    if isinstance(m, np.ndarray):
        return bool(np.all(~(np.abs(m - m.T) >= tol)))
    m = m.tocsr()
    return np.all(~(np.abs(m - m.conjugate().T) >= tol).data)


def make_symmetric(_map: SparseMatrix, use_upper: bool=True) -> SparseMatrix:
    """
    Make a sparse matrix symmetric by taking either the upper or lower triangle as the source, and copying
    that to the opposite triangle. Double-summation of the diagonal is avoided.

    :param _map: the map to make symmetric
    :param use_upper: if true the upper triangle is copied to the lower, otherwise the lower is copied to the upper.
    :return: a symmetric matrix
    """
    if use_upper:
        return scisp.triu(_map) + scisp.triu(_map, k=1).T
    else:
        return scisp.tril(_map) + scisp.tril(_map, k=-1).T


def tensor_print(tensor: SparseMatrix) -> None:
    """
    Pretty print a dense (numpy) 4D matrix. Users should consider the size of the matrix before
    printing, as they can be large! More useful for smaller objects

    :param tensor: the tensor to print wit dim: (N,M,n,m)
    """

    try:
        pw = int(np.ceil(np.log10(tensor.max())))
    except OverflowError:
        pw = 1
    for _i in range(tensor.shape[0]):
        for _k in range(tensor.shape[2]):
            print('|', end='')
            for _j in range(tensor.shape[1]):
                print('[', end='')
                for _l in range(tensor.shape[3]):
                    print('{0:{1}d}'.format(tensor[_i, _j, _k, _l], pw), end='')

                print(']', end='')
            print('|')
        if _i < tensor.shape[1] - 1:
            print('+')
    print('')


def downsample(m: SparseMatrix, block_size: int, method: str='mean') -> SparseMatrix:
    """
    Perform a down-sampling of a 2D matrix (scipy.sparse or ndarray)
    by a factor of block_size in each dimension. Block size must be
    an integer larger than 1.

    When employing mean method, zero padding on edges is not compensated
    for.

    :param m: a matrix (scipy.sparse or ndarray)
    :param block_size: an integer reduction factor
    :param method: per block mean or maximum.
    :return: scipy.sparse.csr_matrix
    """
    assert isinstance(m, (np.ndarray, scisp.spmatrix)), 'supplied array must be of type np.ndarray or scipy.spmatrix'
    assert block_size > 1 and isinstance(block_size, int), 'block_size must be an integer larger than 1'

    def pad_size(N: int, n: int) -> int:
        return int(ceil(N / float(n)) * n) - N

    pad_row = pad_size(m.shape[0], block_size)
    pad_col = pad_size(m.shape[1], block_size)

    # only dok has resize method up until scipy 1.1.0
    if isinstance(m, np.ndarray):
        m = scisp.dok_matrix(m)
    else:
        m = m.todok()
    # eliminate linter false positive about "Unexpected attribute"
    #   when the shape elements individually passed to resize()
    newshape = (m.shape[0] + pad_row, m.shape[1] + pad_col)
    m.resize(*newshape)

    # conversion to csr here appears necessary to properly preserve matrix
    m = sparse.COO(m.tocsr())

    # TODO using mean does not handle zero-padded edge effect on mean values
    if method == 'mean':
        m = m.reshape((m.shape[0] // block_size, block_size,
                       m.shape[1] // block_size, block_size)).sum(axis=(1, 3)).tocsr()
        m *= 1.0 / block_size**2
    elif method == 'max':
        m = m.reshape((m.shape[0] // block_size, block_size,
                       m.shape[1] // block_size, block_size)).max(axis=(1, 3)).tocsr()

    return m


def kr_bistochastic(m: SparseMatrix,
                    tol: float=1e-6,
                    x0: Optional[float]=None,
                    delta: float=0.1,
                    Delta: float=3,
                    max_iter: int=1000) -> Tuple[SparseMatrix, np.ndarray]:
    """
    Normalise a matrix to be bistochastic using Knight-Ruiz algorithm. This method is expected
    to converge more quickly.

    :param m: the input matrix (fully symmetric)
    :param tol: precision tolerance
    :param x0: an initial guess
    :param delta: how close balancing vector can get
    :param Delta: how far balancing vector can get
    :param max_iter: maximum number of iterations before abandoning.
    :return: tuple containing the bistochastic matrix and the scale factors
    """
    assert scisp.isspmatrix(m), 'input matrix must be sparse matrix from scipy.spmatrix'
    assert m.shape[0] == m.shape[1], 'input matrix must be square'

    _orig = m.copy()

    # replace 0 diagonals with 1, on the working matrix. This avoids potentially
    # exploding scale-factors. KR should be regularized!
    m = m.tolil()
    is_zero = m.diagonal() == 0
    if np.any(is_zero):
        logger.warning('treating {} zeros on diagonal as ones'.format(np.sum(is_zero)))
        ix = np.where(is_zero)
        m[ix, ix] = 1

    if not scisp.isspmatrix_csr(m):
        m = m.tocsr()

    if not is_hermitian(m, tol):
        logger.warning('input matrix is expected to be fully symmetric')

    n = m.shape[0]
    e = np.ones(n)

    if not x0:
        x0 = e.copy()

    g = 0.9
    etamax = 0.1
    eta = etamax
    stop_tol = tol * 0.5

    x = x0.copy()
    rt = tol ** 2
    v = x * m.dot(x)

    rk = 1 - v
    rho_km1 = rk.T.dot(rk)  # transpose possibly implicit
    rho_km2 = 1
    rout = rho_km1
    rold = rout

    n_iter = 0
    i = 0
    y = np.empty_like(e)
    while rout > rt and n_iter < max_iter:

        i += 1
        k = 0
        y[:] = e

        inner_tol = max(rout * eta ** 2, rt)
        while rho_km1 > inner_tol:

            k += 1
            if k == 1:
                Z = rk / v
                p = Z
                rho_km1 = rk.T.dot(Z)
            else:
                beta = rho_km1 / rho_km2
                p = Z + beta * p

            w = x * m.dot(x * p) + v * p
            alpha = rho_km1 / p.T.dot(w)
            ap = alpha * p

            ynew = y + ap

            if np.amin(ynew) <= delta:
                if delta == 0:
                    break
                ind = np.where(ap < 0)[0]
                gamma = np.amin((delta - y[ind]) / ap[ind])
                y += gamma * ap
                break

            if np.amax(ynew) >= Delta:
                ind = np.where(ynew > Delta)[0]
                gamma = np.amin((Delta - y[ind]) / ap[ind])
                y += gamma * ap
                break

            y = ynew
            rk = rk - alpha * w
            rho_km2 = rho_km1

            Z = rk * v
            rho_km1 = np.dot(rk.T, Z)

            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                raise RuntimeError('scale vector has developed invalid values (NAN or Inf)!')

        x *= y
        v = x * m.dot(x)

        rk = 1 - v
        rho_km1 = np.dot(rk.T, rk)
        rout = rho_km1
        n_iter += k + 1

        rat = rout / rold
        rold = rout
        res_norm = np.sqrt(rout)
        eta_o = eta
        eta = g * rat

        if g * eta_o ** 2 > 0.1:
            eta = max(eta, g * eta_o ** 2)
        eta = max(min(eta, etamax), stop_tol / res_norm)

    if n_iter > max_iter:
        raise RuntimeError('matrix balancing failed to converge in {} iterations'.format(n_iter))

    del m

    logger.debug('It took {} iterations to achieve bistochasticity'.format(n_iter))

    if n_iter >= max_iter:
        logger.warning('Warning: maximum number of iterations ({}) reached without convergence'.format(max_iter))

    X = scisp.spdiags(x, 0, n, n, 'csr')
    return X.T.dot(_orig.dot(X)).tocoo(), x


class Sparse2DAccumulator(object):

    def __init__(self, size: int) -> None:
        self.shape = (size, size)
        self.mat = {}
        # fixed counting type
        self.dtype = np.uint32

    def __setitem__(self, index: Tuple[int, int], value: int | np.int32) -> None:
        assert len(index) == 2 and index[0] >= 0 and index[1] >= 0, 'invalid index: {}'.format(index)
        assert isinstance(value, (int, np.int64)), 'values must be integers'
        self.mat[index] = value

    def __getitem__(self, index: Tuple[int, int]) -> int | np.int32:
        if index in self.mat:
            return self.mat[index]
        return 0

    def get_coo(self, make_symm: bool=True) -> scisp.coo_matrix:
        """
        Create a COO format sparse representation of the accumulated values.

        :param make_symm: ensure matrix is symmetric on return
        :return: a scipy.coo_matrix sparse matrix
        """
        _coords = [[], []]
        _data = []
        _m = self.mat
        for i, j in _m.keys():
            _coords[0].append(i)
            _coords[1].append(j)
            _data.append(_m[i, j])

        _m = scisp.coo_matrix((_data, _coords), shape=self.shape, dtype=self.dtype)

        if make_symm:
            _m += scisp.tril(_m.T, k=-1)

        return _m.tocoo()


@nb.jit(nopython=True)
def fast_offdiag(_data: np.ndarray, _row: np.ndarray, _col: np.ndarray, _shape: np.ndarray) -> np.ndarray:
    """
    Determine the maximum off-diagonal elements using the
    internal attributes of a scipy coo matrix. The matrix
    is assumed to be square.

    :param _data: the corresponding adata of the coo matrix
    :param _row: the row indices of the coo matrix
    :param _col: the column indices of the coo matrix
    :param _shape: the dimension of the square matrix (NxN)
    :return: an array of size N of maximum off-diagonal values
    """
    mx = np.zeros(_shape, dtype=_data.dtype)
    for i in range(_row.shape[0]):
        if _row[i] == _col[i]:
            continue
        if mx[_row[i]] < _data[i]:
            mx[_row[i]] = _data[i]
    return mx


def max_offdiag(_m: SparseMatrix) -> np.ndarray:
    """
    Determine the maximum off-diagonal values of a given symmetric matrix. As this
    is assumed to be symmetric, we consider only the rows.

    :param _m: a scipy.sparse matrix
    :return: the off-diagonal maximum values
    """
    assert scisp.isspmatrix(_m), 'Input matrix is not a scipy.sparse object'
    if not scisp.isspmatrix_coo(_m):
        _m = _m.tocoo()
    return fast_offdiag(_m.data, _m.row, _m.col, _m.shape[0])


@nb.jit(nopython=True)
def fast_zero_weak(_val: float, _data: np.ndarray, _row: np.ndarray, _col: np.ndarray, _shape: np.ndarray) -> None:
    """
    Modify in-place, zeroing any element of the matrix which
    falls below the threshold minimum value.

    :param _val: the minimum acceptable value
    :param _data: the corresponding adata of the coo matrix
    :param _row: the row indices of the coo matrix
    :param _col: the column indices of the coo matrix
    :param _shape: the dimension of the square matrix (NxN)
    """
    for i in range(_row.shape[0]):
        # ignore diagonal
        if _row[i] == _col[i]:
            continue
        if _data[i] < _val:
            _data[i] = 0


def zero_weak_offdiag(_m: SparseMatrix, _val: float) -> SparseMatrix:
    """
    For any non-zero elements below the specified threshold, zero them out.

    :param _m: a scipy.sparse matrix
    :param _val: the minimum acceptable value
    :return: the off-diagonal maximum values
    """
    assert scisp.isspmatrix(_m), 'Input matrix is not a scipy.sparse object'
    if not scisp.isspmatrix_coo(_m):
        _m = _m.tocoo()
    fast_zero_weak(_val, _m.data, _m.row, _m.col, _m.shape[0])
    # refresh sparsity
    _m.eliminate_zeros()
    return _m


@nb.jit(nopython=True)
def fast_retained(_data: np.ndarray,
                  _row: np.ndarray,
                  _col: np.ndarray,
                  _nnz: int,
                  _mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a mask, determine the elements of a coo matrix attributes
    data, row and column and the resulting shifts.
    :param _data: the corresponding data of the coo matrix
    :param _row: the row indices of the coo matrix
    :param _col: the column indices of the coo matrix
    :param _nnz: the number of non-zero elements in the coo matrix
    :param _mask: the boolean mask to apply
    :return: tuple of retained data, row, col and resulting shift
    """
    keep_row = []
    keep_col = []
    keep_data = []
    for i in range(_nnz):
        if _mask[_row[i]] and _mask[_col[i]]:
            keep_row.append(_row[i])
            keep_col.append(_col[i])
            keep_data.append(_data[i])

    keep_row = np.array(keep_row)
    keep_col = np.array(keep_col)
    keep_data = np.array(keep_data)

    # adjustments for removed rows/column indices
    shift = np.cumsum(~_mask)

    # TODO move this in the above loop
    for i in range(len(keep_row)):
        keep_row[i] -= shift[keep_row[i]]
        keep_col[i] -= shift[keep_col[i]]

    return keep_data, keep_row, keep_col, shift


def compress(_m: SparseMatrix, _mask: np.ndarray) -> SparseMatrix:
    """
    Remove rows and columns using a 1d boolean mask.

    :param _m: matrix
    :param _mask: True (keep), False (drop)
    :return: a coo_matrix of only the accepted rows/columns
    """
    assert scisp.isspmatrix(_m), 'Input matrix is not a scipy sparse matrix type'

    if not scisp.isspmatrix_coo(_m):
        _m = _m.tocoo()

    _data, _row, _col, _shift = fast_retained(_m.data, _m.row, _m.col, _m.nnz, _mask)

    return scisp.coo_matrix((_data, (_row, _col)), shape=np.array(_m.shape) - _shift[-1])


class Sparse4DAccumulator(object):
    """
    Simple square sparse tensor of dimension (N, N, 2, 2)
    There is limited functionality and mainly intended to save memory while not performing operations.
    """
    def __init__(self, size: int) -> None:
        self.shape = (size, size, 2, 2)
        self.mat = {}
        # fixed counting type
        self.dtype = np.uint32

    def __setitem__(self, index: Tuple[int,int,int,int], value: int | np.int32) -> None:
        assert isinstance(index, tuple), 'index must be a list of indices'
        if len(index) == 4:
            assert 0 <= index[0] < self.shape[0] and \
                   0 <= index[1] < self.shape[1] and \
                   0 <= index[2] < 2 and \
                   0 <= index[3] < 2, 'invalid range {} for dimension {}'.format(index, self.shape)

            if index[:2] not in self.mat and np.any(value != 0):
                self.mat.setdefault(index[:2], self._make_elem())[index[2:]] = value

        if len(index) == 2:
            assert 0 <= index[0] < self.shape[0] and \
                   0 <= index[1] < self.shape[1], 'invalid range {} for dimension {}'.format(index, self.shape)

            if index not in self.mat:
                self.mat.setdefault(index, self._make_elem())[:] = value

    def __getitem__(self, index: Tuple[int,int,int,int]) -> np.ndarray:
        return self.mat.setdefault(index, self._make_elem())

    def _make_elem(self) -> np.ndarray:
        return np.zeros((2, 2), dtype=self.dtype)

    def get_coo(self, make_symm: bool=True) -> sparse.COO:
        """
        Create a COO format sparse representation of the accumulated values. NOTE: As scipy
        does not support multidimensional arrays, this object is from the "sparse" module.

        :param make_symm: ensure matrix is symmetric on return
        :return: a sparse.COO matrix
        """
        _coords = [[], [], [], []]
        _data = []
        _m = self.mat
        _inner_indices = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for _i, _j in _m.keys():
            for _k, _l in _inner_indices:
                v = _m[_i, _j][_k, _l]
                if v != 0:
                    _coords[0].append(_i)
                    _coords[1].append(_j)
                    _coords[2].append(_k)
                    _coords[3].append(_l)
                    _data.append(v)

        _m = sparse.COO(_coords, _data, self.shape, has_duplicates=False)

        if make_symm:
            _m = Sparse4DAccumulator.symm(_m)

        return _m

    @staticmethod
    def _flip(c_row: np.ndarray) -> np.ndarray:
        """
        Flip indices (coordinates) as pairs: (i,j), (k,l) -> (j,i), (l,k)

        :param c_row: coordinate row to flip
        :return the flipped indices
        """
        c_row = c_row.copy()
        c_row[0], c_row[1] = c_row[1], c_row[0]
        c_row[2], c_row[3] = c_row[3], c_row[2]
        return c_row

    @staticmethod
    def symm(_m: SparseMatrix) -> SparseMatrix:
        """
        Make a 4D COO matrix symmetric, all elements above and below the diagonal are included.
        Duplicate entries will be summed.

        :param _m: the NxNx2x2 matrix to make symmetric
        :return: a new symmetric version
        """
        # collect indices of diagonal elements along primary axes (0 and 1)
        ix = np.where(~np.apply_along_axis(lambda x: x[0] == x[1], 0, _m.coords))[0]
        # append every non-zero, non-diag coord and accompanying data to a new sparse object
        # and also perform the transpose (i,j), (k,l) -> (j,i), (l,k)
        _coords = np.hstack((_m.coords, np.apply_along_axis(Sparse4DAccumulator._flip, 0, _m.coords[:, ix])))
        _data = np.hstack((_m.data, _m.data[ix]))
        return sparse.COO(_coords, _data, shape=_m.shape, has_duplicates=True)


def max_offdiag_4d(_m: SparseMatrix) -> np.ndarray:
    """
    Determine the maximum off-diagonal summed signal, where "summed signal" refers to reducing the
    the tensor to a 2d matrix by summing over the last two axes (2x2 submatrices).

    :param _m: a 4d sparse.COO or DOK matrix with dimension NxNx2x2.
    :return: a vector of length N containing off-diagonal maximums.
    """
    return max_offdiag(_m.sum(axis=(2, 3)).tocsr())


def flatten_tensor_4d(_m: SparseMatrix) -> SparseMatrix:
    """
    Flatten a 4D tensor into 2D by doubling the first two dimensions. It is assumed that the matrix
    has already been made symmetric (if required).

    :param _m: a 4d sparse.COO matrix with dimension NxNx2x2
    :return: 2d sparse matrix of type scipy.sparse.coo_matrix
    """
    _coords = [[], []]
    _data = []
    for n in range(_m.nnz):
        _i, _j, _k, _l = _m.coords[:, n]
        ii = 2*_i
        jj = 2*_j
        _coords[0].append(ii+_k)
        _coords[1].append(jj+_l)
        _data.append(_m.data[n])

    _m = scisp.coo_matrix((_data, _coords), shape=(2*_m.shape[0], 2*_m.shape[1]))
    return _m


def compress_4d(_m: SparseMatrix, _mask: np.ndarray) -> SparseMatrix:
    """
    Remove rows and columns of a sparse 4D matrix using a 1d boolean mask. Masking operates on
    only the first two primary axes (essentially a 2D matrix with 2x2 cells). If the input is not
    of sparse.COO type, it will be cast. An exception is raised if the matrix is not of
     sparse.DOK or sparse.COO type. The returned matrix is of type sparse.COO.

    :param _m: matrix
    :param _mask: True (keep), False (drop)
    :return: a sparse.COO of only the accepted rows/columns
    """
    assert isinstance(_m, (sparse.COO, sparse.DOK)), 'Input matrix must be of sparse.COO or sparse.DOK type'
    if not isinstance(_m, sparse.COO):
        _m = _m.to_coo()

    # collect those values not in the excluded rows/columns
    keep_coords = []
    keep_data = []
    accept_index = set(np.where(_mask)[0])
    for i in range(_m.nnz):
        if _m.coords[0, i] in accept_index and _m.coords[1, i] in accept_index:
            keep_coords.append(_m.coords[:, i])
            keep_data.append(_m.data[i])
    keep_coords = np.array(keep_coords, dtype=np.int64).T

    # remaining data needs adjustments to compensate for removed rows/column indices
    shift = np.cumsum(~_mask, dtype=np.int64)
    keep_coords[:2, :] -= shift[keep_coords[:2, :]]

    # create new smaller matrix
    new_shape = list(_m.shape)
    new_shape[:2] -= shift[-1]
    return sparse.COO(keep_coords, keep_data, shape=new_shape, has_duplicates=False)


def dotdot(_m: SparseMatrix, _a: np.ndarray) -> SparseMatrix:
    """
    Assuming A is a vector representing the trace of a diagonal matrix, dotdot
    performs the transformation dot(A.T,dot(M,A)) on  a sparse matrix.

    :param _m: the sparse matrix, modified in-place
    :param _a: the 1d trace of a diagonal matrix
    :return: the in-place modified matrix
    """
    for n in range(_m.nnz):
        i, j = _m.coords[:2, n]
        _m.data[n] *= _a[i] * _a[j]
    return _m


def kr_bistochastic_4d(m4d: SparseMatrix, **kwargs: Optional[Dict[str,Any]]) -> Tuple[SparseMatrix, np.ndarray]:
    """
    Knight-Ruiz applied to a NxNx2x2 tensor. The scale factors are determined by first converting
    this to a 2D matrix, summed on axis 2 and 3. The method is intended for determining scale-factors
    of the doublet matrix.

    :param m4d: a NxNxmxn matrix
    :param kwargs: options to kr_bistochastic()
    :return: a scaled matrix, scale-factors
    """
    # reduce to a 2D array, where we're summing the 2x2 submatrices
    m2d = m4d.astype(np.float64).sum(axis=(2, 3)).tocsr()
    _, scl = kr_bistochastic(m2d, **kwargs)
    return dotdot(m4d.astype(np.float64), scl), scl

import logging
import os
import re
import subprocess
from typing import IO, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse

logger = logging.getLogger(__name__)

MatrixType = Union[np.ndarray, sparse.spmatrix]

def reciprocal_counts(m: MatrixType, alpha: Optional[float]=0.1) -> MatrixType:
    """
    Express a measure of similarity as a distance by taking the reciprocal. The array
    is expected to be strictly positive reals. Additional fiddly steps are taken to
    better support LKH, and its restriction to integers when expressing explicit
    distances.
    1. The diagonal is zeroed
    2. Similarity is converted to [0,1] prior to taken recip.
    3. Effort is taken to ensure that the shortest distance (the largest similarity) is 1 and not 0.
    :param m: Input similarity matrix, which is also written to.
    :param alpha: Additive smoothing factor, which avoids zero.
    :return: Distance matrix
    """
    assert m.min() >= 0, 'the input matrix should be strictly positive'
    assert np.issubdtype(m.dtype, np.float64) or np.issubdtype(m.dtype, np.float32), 'matrix must be floats'

    # begin with removal of diagonal self-self interactions
    np.fill_diagonal(m, 0)
    m += alpha
    # scale elements [0,1]
    m = m / m.max()
    # take inverse
    m = 1.0 / m
    # Rescale all values.
    # As this will be integer truncated, we make sure the smallest value is > 1.
    # The reason for this is to protect the shortest distance elements
    # from going to zero when the matrix is converted to integers (requirement of LKH).
    m *= 1.01 / m.min()
    np.fill_diagonal(m, 0)
    return m


def scale_mat(m: MatrixType, _min: float, _max: float) -> MatrixType:
    """
    In-place rescaling of matrix elements to be within the range [_min, _max].
    :param m: The target matrix.
    :param _min: The smallest allowable output value.
    :param _max: The largest allowable output value.
    :return: The rescaled matrix (matrix is changed in-place).
    """
    m[:] = (m - m.min()) / (m.max() - m.min())
    m *= _max - _min
    m += _min
    return m


def similarity_to_distance(m: MatrixType,
                           method: str,
                           alpha: float=2,
                           beta: float=1,
                           headroom: float=1) -> MatrixType:
    """
    Convert a matrix representing similarity (larger values indicate increasing association)
    to a distance matrix (or dissimilarity matrix) where larger values indicate decreasing
    association (further away).

    As LKH requires distance as 32bit integers, the transformation must control the size of
    the largest value and make good use of the available range. Therefore, similarity zeros
    (which translate to be the largest distances) are constrained to be only a factor
    "alpha" worse than the most distant (originally non-zero) value, while also being
    smaller than a fixed limit imposed by LKH. Exceeding this limit causes LKH to simply
    truncate those values to the limit -- a potential undesirable/unpredictable outcome.

    There are three transformation functions from which to choose:

    "inverse": y = (1/x)^beta
     "neglog": y = -(log x/xmax)^beta
     "linear": y = (1 - x/xmax)^beta

    All three functions display different treatment of small and large values.

    :param m: The target similarity matrix.
    :param method: "Inverse", "neglog" or "linear".
    :param alpha: Factor to which similarity zeros are set in the distance beyond the largest distance.
    :param beta: An exponent to raise each element (default =1 i.e., no effect).
    :param headroom: Further factor of constraint to impose on the largest integer allowed.
    :return: Distance matrix.
    """

    assert alpha >= 1, 'alpha cannot be less than 1'
    maximum_int_value = 2147483647.0
    largest = maximum_int_value / 2.0 / len(m) / headroom
    logger.debug('Largest available integer: {:d}'.format(int(largest)))

    # copy input matrix and remove diagonal
    m = m.astype(np.float64)
    np.fill_diagonal(m, 0)

    # remember where zeros were
    zeros = (m == 0)
    logger.debug('Zero count: {}'.format(np.sum(zeros)))
    logger.debug('Initial non-zero range: {:.3e} {:.3e}'.format(m[np.where(~zeros)].min(), m.max()))

    nzix = np.where(~zeros)

    # transform similarity to distance, avoiding div-zero in some cases
    if method == 'inverse':
        m[nzix] = 1.0 / m[nzix]
    elif method == 'linear':
        c = 1.0 / m.max()
        m = 1.0 - c * m
    elif method == 'neglog':
        c = 1.0 / m.max()
        m[nzix] = - np.log(c * m[nzix])
    else:
        raise RuntimeError('unsupported method: {}'.format(method))

    # apply element-wise power if requested
    if beta != 1:
        m = np.power(m, beta)

    # assign zeros (no observations) as a 'worst-case'
    max_m = m.max()
    logger.debug('Transformed range: {:.3e} {:.3e}'.format(m[np.where(~zeros)].min(), max_m))

    m[np.where(zeros)] = alpha * max_m
    logger.debug('Zeros assigned worst case of: {:.3e}'.format(alpha * max_m))

    # rescale to use available integer range
    m = scale_mat(m, 1, largest)
    logger.debug('Rescaled range: {:.3e} {:.3e}'.format(m.min(), m.max()))

    return m


def lkh_order(m: MatrixType,
              base_name: str,
              precision: int=1,
              lkh_exe: Optional[str]=None,
              runs: Optional[int]=None,
              seed: Optional[int]=None,
              dist_func: Callable[[MatrixType], MatrixType]=reciprocal_counts,
              fixed_edges: Optional[List]=None,
              special: bool=True,
              pop_size: Optional[int]=None,
              stdout: Optional[IO]=None) -> list | npt.NDArray:
    """
    Employ LKH TSP solver to find the best order through a distance matrix. By default, it is assumed that
    LKH is on the path. A CalledProcessError is raised if execution fails. The input to LKH is an explicit
    definition of the full connected distance matrix, and sparse matrices will be converted to dense
    representations. For large problems, this can be memory demanding.

    :param m: The distance matrix.
    :param base_name: Base name of LKH control files.
    :param precision: LKH internal precision factor, larger values limit maximum representable number.
    :param lkh_exe: Path to binary, otherwise assumed on the path.
    :param runs: The number of runs to perform LKH.
    :param seed: Random seed (milli-time if unspecified).
    :param dist_func: A custom distance function with which to convert matrix m.
    :param fixed_edges: List of edge tuples (u,v) that _must_ occur in the tour.
    :param special: Use LKH "special" meta-setting.
    :param pop_size: Population size of tours used in the special genetic algorithm component (default: runs/4).
    :param stdout: Redirection for stdout of lkh.
    :return: 0-based order as a numpy array.
    """
    if sparse.isspmatrix(m):
        m = np.asarray(m.todense())
    m = dist_func(m.astype(np.float64))

    try:
        write_lkh(base_name, m, len(m), max_trials=2*len(m), runs=runs, seed=seed, fixed_edges=fixed_edges,
                  pop_size=pop_size, special=special, precision=precision)
        if not lkh_exe:
            lkh_exe = 'LKH'
        subprocess.check_call([lkh_exe, '{}.par'.format(base_name)], stdout=stdout, stderr=subprocess.STDOUT)
        tour = read_lkh('{}.tour'.format(base_name))
    except subprocess.CalledProcessError as e:
        logger.error('Failed to start LKH subprocess using path \'{}\''.format(lkh_exe))
        raise e

    return tour['path']


def write_lkh(root_path_name: str,
              m: npt.NDArray,
              dim: int,
              max_trials: Optional[int]=None,
              runs: Optional[int]=None,
              seed: Optional[int]=None,
              mat_fmt: str='upper',
              fixed_edges: Optional[List[Tuple]]=None,
              pop_size: Optional[int]=None,
              special: bool=True,
              lkh_verbose: bool=False,
              precision: int=1) -> None:
    """
    Create the control (.par) and data file (.dat) for the LKH executable. The implementation
    has many additional control parameters which could be included. Refer to LKH-3 documentation.
    :param root_path_name: root file name and path for output lkh files
    :param m: the data (2D distance matrix or edges)
    :param dim: the number of nodes (cities)
    :param max_trials: maximum number of trials (default = dim)
    :param runs: number of runs (default = 10)
    :param seed: random seed for algorithm (default milliseconds)
    :param mat_fmt: matrix format
    :param fixed_edges: list of edge tuples (u,v) that _must_ occur in the tour
    :param pop_size: population size of tours used in special genetic algorithm component (default: runs/4)
    :param special: use LKH "special" meta-setting
    :param lkh_verbose: make LKH verbose during runs
    :param precision: scale integer values
    """

    def write_full_matrix(_out_h: IO, _m: npt.NDArray) -> None:
        _out_h.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        _out_h.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX\n')
        _out_h.write('EDGE_WEIGHT_SECTION\n')
        np.savetxt(_out_h, _m, fmt='%d')

    def write_upper_row(_out_h: IO, _m: npt.NDArray) -> None:
        _out_h.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        _out_h.write('EDGE_WEIGHT_FORMAT: UPPER_ROW\n')
        out_h.write('EDGE_WEIGHT_SECTION\n')
        for i in range(len(m)-1):
            out_h.write(' '.join([str(int(vi)) for vi in _m[i, i+1:]]))
            out_h.write('\n')

    assert isinstance(m, np.ndarray), 'the matrix must be a numpy array'
    if not seed:
        import time
        seed = round(time.time() * 1000)
    else:
        assert isinstance(seed, int), 'random seed must be an integer'

    base_name = os.path.basename(root_path_name)
    control_file = '{}.par'.format(root_path_name)
    data_file = '{}.dat'.format(root_path_name)
    with open(control_file, 'wt') as out_h:
        if special:
            # SPECIAL is a meta-setting for the following
            # out_h.write('GAIN23 = NO\n')
            # out_h.write('KICKS = 1\n')
            # out_h.write('KICK_TYPE = 4\n')
            # out_h.write('MAX_SWAPS = 0\n')
            # out_h.write('MOVE_TYPE = 5 SPECIAL\n')
            out_h.write('SPECIAL\n')
        if pop_size:
            out_h.write('POPULATION_SIZE = {}\n'.format(pop_size))
        out_h.write('PROBLEM_FILE = {}\n'.format(data_file))
        if max_trials:
            out_h.write('MAX_TRIALS = {}\n'.format(max_trials))
        if runs:
            out_h.write('RUNS = {}\n'.format(runs))
        out_h.write('SEED = {}\n'.format(seed))
        out_h.write('OUTPUT_TOUR_FILE = {}.tour\n'.format(root_path_name))
        out_h.write('PRECISION = {}\n'.format(precision))
        out_h.write('TRACE_LEVEL = {}'.format(int(lkh_verbose)))

    with open(data_file, 'wt') as out_h:
        out_h.write('NAME: {}\n'.format(base_name))
        out_h.write('TYPE: TSP\n')
        out_h.write('DIMENSION: {}\n'.format(dim))
        out_h.write('SALESMEN: 1\n')

        # currently only supporting one explicit data type
        if mat_fmt == 'full':
            write_full_matrix(out_h, m)
        elif mat_fmt == 'upper':
            write_upper_row(out_h, m)

        if fixed_edges:
            out_h.write('FIXED_EDGES_SECTION:\n')
            for u, v in fixed_edges:
                out_h.write('{} {}\n'.format(u, v))
            out_h.write('-1\n')


def read_lkh(filename: str) -> Dict[str, Union[int, str, npt.NDArray[np.int64]]]:
    """
    Read the resulting solution (output tour) file from LKH.
    :param filename: The solution file name.
    :return: Dict of the tour information.
    """
    tour = dict()
    tour['path'] = []
    with open(filename, 'rt') as in_h:
        in_tour = False
        for line in in_h:
            line = line.strip()
            if not line or line == 'EOF':
                break
            if line.startswith('TOUR_SECTION'):
                in_tour = True
            elif not in_tour:
                m = re.search(r'(\w+)[\s:=]+(\S+)', line)
                if not m:
                    continue
                if m.group(1) == 'DIMENSION':
                    tour[m.group(1)] = int(m.group(2))
                else:
                    tour[m.group(1)] = m.group(2)
            else:
                tour.setdefault('path', []).append(int(line))

        # convert ids to 0-based and remove end marker
        tour['path'] = np.array(tour['path'][:-1]) - 1

    return tour

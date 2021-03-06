import logging
import numpy as np
import scipy.sparse as sparse
import os
import re
import subprocess

logger = logging.getLogger(__name__)


def reciprocal_counts(m, alpha=0.1):
    """
    Express a measure of similarity as a distance by taking the reciprocal. The array
    is expected to be strictly positive reals. Additional fiddly steps are taken to
    better support LKH, and its restriction to integers when expressing explicit
    distances.
    1. the diagonal is zeroed
    2. similarity is converted to [0,1] prior to taken recip.
    3. effort is taken to ensure the shortest distance (largest similarity) is 1 and not 0.
    :param m: input similarity matrix, which is also written to.
    :param alpha: additive smoothing factor, which avoids zero.
    :return: distance matrix
    """
    assert m.min() >= 0, 'the input matrix should be stricly positive'
    assert np.issubdtype(m.dtype, np.float64) or np.issubdtype(m.dtype, np.float32), 'matrix must be floats'

    # begin with removal of diagonal self-self interactions
    np.fill_diagonal(m, 0)
    m += alpha
    # scale elements [0,1]
    m = m / m.max()
    # take inverse
    m = 1.0 / m
    # rescale all values
    # As this will be integer truncated, we make sure the smallest value is > 1
    # the reason for this is to protect the shortest distance elements
    # from going to zero when the matrix is converted to integers (requirement of LKH)
    m *= 1.01 / m.min()
    np.fill_diagonal(m, 0)
    return m


def scale_mat(M, _min, _max):
    """
    In-place rescaling of matrix elements to be within the range [_min, _max]
    :param M: the target matrix
    :param _min: the smallest allowable output value
    :param _max: the largest allowable output value
    :return: the rescaled matrix (matrix is changed in-place)
    """
    M[:] = (M - M.min()) / (M.max() - M.min())
    M *= _max - _min
    M += _min
    return M


def similarity_to_distance(M, method, alpha=2, beta=1, headroom=1):
    """
    Convert a matrix representing similarity (larger values indicate increasing association)
    to a distance matrix (or dissimilarity matrix) where larger values indicate decreasing
    association (further away).

    As LKH requires distance as 32bit integers, the transformation must control the size of
    the largest value and make good use of the available range. Therefore similarity zeros
    (which translate to be the largest distances) are constrained to be only a factor
    "alpha" worse than the most distant (originally non-zero) value, while also being
    smaller than a fixed limit imposed by LKH. Exceeding this limit causes LKH to simply
    truncate those values to the limit -- a potential undesirable/unpredictable outcome.

    There are three transformation functions from which to choose:

    "inverse": y =  (1/x)^beta
     "neglog": y = -(log x/xmax)^beta
     "linear": y =  (1 - x/xmax)^beta

    All three functions display different treatment of small and large values.

    :param M: the target similarity matrix
    :param method: "inverse", "neglog" or "linear"
    :param alpha: factor to which similarity zeros are set in the distance beyond the largest distance.
    :param beta: an exponent to raise each element (default =1 ie. no effect)
    :param headroom: further factor of constraint to impose on largest integer allowed
    :return: distance matrix
    """

    assert alpha >= 1, 'alpha cannot be less than 1'
    INT_MAX = 2147483647.0
    largest = INT_MAX / 2.0 / len(M) / headroom
    logger.debug('Largest available integer: {:d}'.format(int(largest)))

    # copy input matrix and remove diagonal
    M = M.astype(np.float)
    np.fill_diagonal(M, 0)

    # remember where zeros were
    zeros = (M == 0)
    logger.debug('Zero count: {}'.format(zeros.sum()))
    logger.debug('Initial non-zero range: {:.3e} {:.3e}'.format(M[np.where(~zeros)].min(), M.max()))

    nzix = np.where(~zeros)

    # transform similarity to distance, avoiding div-zero in some cases
    if method == 'inverse':
        M[nzix] = 1.0/M[nzix]
    elif method == 'linear':
        c = 1.0/M.max()
        M = 1.0 - c * M
    elif method == 'neglog':
        c = 1.0/M.max()
        M[nzix] = - np.log(c * M[nzix])
    else:
        raise RuntimeError('unsupported method: {}'.format(method))

    # apply element-wise power if requested
    if beta != 1:
        M = np.power(M, beta)

    # assign zeros (no observations) as a 'worst-case'
    maxM = M.max()
    logger.debug('Transformed range: {:.3e} {:.3e}'.format(M[np.where(~zeros)].min(), maxM))

    M[np.where(zeros)] = alpha * maxM
    logger.debug('Zeros assigned worst case of: {:.3e}'.format(alpha * maxM))

    # rescale to use available integer range
    M = scale_mat(M, 1, largest)
    logger.debug('Rescaled range: {:.3e} {:.3e}'.format(M.min(), M.max()))

    return M


def lkh_order(m, base_name, precision=1, lkh_exe=None, runs=None, seed=None, dist_func=reciprocal_counts,
              fixed_edges=None, special=True, pop_size=None, stdout=None):
    """
    Employ LKH TSP solver to find the best order through a distance matrix. By default, it is assumed that
    LKH is on the path. A CalledProcessError is raised if execution fails. The input to LKH is an explicit
    definition of the full connected distance matrix, therefore sparse matrices will be converted to dense
    representations. For large problems, this can be memory demanding.

    :param m: the distance matrix
    :param base_name: base name of LKH control files
    :param precision: LKH internal precision factor, larger values limit maximum representable number
    :param lkh_exe: Path to binary, otherwise assumed on the path
    :param runs: the number of runs to perform LKH
    :param seed: random seed (milli-time if unspecified)
    :param dist_func: a custom distance function with which to convert matrix m.
    :param fixed_edges: list of edge tuples (u,v) that _must_ occur in the tour
    :param special: use LKH "special" meta-setting
    :param pop_size: population size of tours used in special genetic algorithm component (default: runs/4)
    :param stdout: redirection for stdout of lkh
    :return: 0-based order as a numpy array
    """
    if sparse.isspmatrix(m):
        m = m.toarray()
    m = dist_func(m.astype(np.float))

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


def write_lkh(base_name, m, dim, max_trials=None, runs=None, seed=None, mat_fmt='upper',
              fixed_edges=None, pop_size=None, special=True, lkh_verbose=False, precision=1):
    """
    Create the control (.par) and data file (.dat) for the LKH executable. The implementation
    has many additional control parameters which could be included. Refer to LKH-3 documentation.
    :param base_name: base lkh file name
    :param m: the data (2D distance matrix or edges)
    :param dim: the number of nodes (cities)
    :param max_trials: maximum number of trials (default = dim)
    :param runs: number of runs (default = 10)
    :param seed: random seed for algorithm (default milliseconds)
    :param mat_fmt: matrix format
    :param fixed_edges: list of edge tuples (u,v) that _must_ occur in the tour
    :param special: use LKH "special" meta-setting
    :param lkh_verbose: make LKH verbose during runs
    :param precision: scale integer values
    :param pop_size: population size of tours used in special genetic algorithm component (default: runs/4)
    """

    def write_full_matrix(out_h, m):
        out_h.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        out_h.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX\n')
        out_h.write('EDGE_WEIGHT_SECTION\n')
        np.savetxt(out_h, m, fmt='%d')

    def write_upper_row(out_h, m):
        out_h.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        out_h.write('EDGE_WEIGHT_FORMAT: UPPER_ROW\n')
        out_h.write('EDGE_WEIGHT_SECTION\n')
        for i in xrange(len(m)-1):
            out_h.write(' '.join([str(int(vi)) for vi in m[i, i+1:]]))
            out_h.write('\n')

    assert isinstance(m, np.ndarray), 'the matrix must be a numpy array'
    if not seed:
        import time
        seed = int(round(time.time() * 1000))
    else:
        assert isinstance(seed, int), 'random seed must be an integer'

    nopath = os.path.basename(base_name)
    control_file = '{}.par'.format(base_name)
    data_file = '{}.dat'.format(base_name)
    with open(control_file, 'w') as out_h:
        if special:
            # SPECIAL is a meta setting for the following
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
        out_h.write('OUTPUT_TOUR_FILE = {}.tour\n'.format(base_name))
        out_h.write('PRECISION = {}.tour\n'.format(precision))
        out_h.write('TRACE_LEVEL = {}'.format(int(lkh_verbose)))

    with open(data_file, 'w') as out_h:
        out_h.write('NAME: {}\n'.format(nopath))
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


def read_lkh(fname):
    """
    Read the resulting solution (output tour) file from LKH.
    :param fname: the solution file name
    :return: dict of the tour information
    """
    tour = {'path': []}
    with open(fname, 'r') as in_h:
        in_tour = False
        for line in in_h:
            line = line.strip()
            if not line or line == 'EOF':
                break
            if line.startswith('TOUR_SECTION'):
                in_tour = True
            elif not in_tour:
                m = re.search('(\w+)[\s:=]+(\S+)', line)
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

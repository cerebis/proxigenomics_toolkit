from ..io_utils import io_utils
from ..linalg import sparse_utils
from ..misc_utils import package_path
from .. import ordering
from ..seq_utils.seq_utils import *
from collections import OrderedDict, namedtuple, defaultdict
from functools import partial
from scipy.stats import binom, poisson
import numba as nb
import Bio.SeqIO as SeqIO
import Bio.SeqUtils as SeqUtils
import logging
import numpy as np
import pysam
import scipy.sparse as sp
import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn

# package logger
logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.INFO)

SeqInfo = namedtuple('SeqInfo', ['offset', 'refid', 'name', 'length', 'sites', 'gc'])


"""
Basic Mean functions
"""


@nb.jit(nopython=True)
def geometric_mean(x, y):
    return (x*y)**0.5


@nb.jit(nopython=True)
def harmonic_mean(x, y):
    return 2*x*y/(x+y)


@nb.jit(nopython=True)
def arithmetic_mean(x, y):
    return 0.5*(x+y)


def mean_selector(name):
    try:
        mean_switcher = {
            'geometric': geometric_mean,
            'harmonic': harmonic_mean,
            'arithmetic': arithmetic_mean
        }
        return mean_switcher[name]
    except KeyError:
        raise RuntimeError('unsupported mean type [{}]'.format(name))


@nb.jit('int64(int64[:, :], int64)', nopython=True)
def find_nearest_jit(group_sites, x):
    """
    Find the nearest site from a given position on a contig.

    :param group_sites:
    :param x: query position
    :return: tuple of site and group number
    """
    ix = np.searchsorted(group_sites[:, 0], x)
    if ix == len(group_sites):
        # raise RuntimeError('find_nearest: {} didnt fit in {}'.format(x, group_sites))
        return group_sites[-1, 1]
    return group_sites[ix, 1]


@nb.jit(nopython=True)
def fast_norm_tipbased_bylength(coords, data, tip_lengths, tip_size):
    """
    In-place normalisation of the sparse 4D matrix used in tip-based maps.

    As tip-based normalisation is slow for large matrices, the inner-loop has been
    moved to a Numba method.

    :param coords: the COO matrix coordinate member variable (4xN array)
    :param data:  the COO matrix data member variable (1xN array)
    :param tip_lengths: per-element min(sequence_length, tip_size)
    :param tip_size: tip size used in map
    """
    for ii in range(coords.shape[1]):
        i, j = coords[:2, ii]
        data[ii] *= tip_size**2 / (tip_lengths[i] * tip_lengths[j])


@nb.jit(nopython=True)
def fast_norm_tipbased_bysite(coords, data, sites):
    """
    In-place normalisation of the sparse 4D matrix used in tip-based maps.

    As tip-based normalisation is slow for large matrices, the inner-loop has been
    moved to a Numba method.

    :param coords: the COO matrix coordinate member variable (4xN array)
    :param data:  the COO matrix data member variable (1xN array)
    :param sites: per-element min(sequence_length, tip_size)
    """
    for n in range(coords.shape[1]):
        i, j, k, l = coords[:, n]
        data[n] *= 1.0/(sites[i, k] * sites[j, l])


# first 30 factorial values as approx as floats
LOOKUP_TABLE = np.fromiter((np.math.factorial(i) for i in range(31)), dtype=np.float64)


@nb.jit(nopython=True)
def fast_factorial(n):
    if n > 30:
        return np.math.gamma(n+1)
    return LOOKUP_TABLE[n]


@nb.jit(nopython=True)
def poisson_cdf(x, n, p):
    L = n * p
    sum = 0
    for i in range(0, x+1):
        sum += L**i / fast_factorial(i)
    return sum * np.exp(-L)


@nb.jit(nopython=True)
def max_interactions(_ir, _breaks, _indices, _values):

    cols = []
    sums = []
    for _ic in range(len(_breaks)-1):

        if _ic == _ir:
            continue

        _a, _b = _breaks[_ic], _breaks[_ic+1]

        # func
        # _val = _values[(_indices >= _a) & (_indices < _b)].sum()
        _val = _values[(_indices >= _a) & (_indices < _b)]

        if len(_val) != 0:
            cols.append(_ic)
            sums.append(_val.max())

    return cols, sums


def reduce_seqmap_to_accepted(_map, contact_map, _min_sig=None, _min_len=None):
    return sparse_utils.compress(_map.tocoo(), contact_map.get_primary_acceptance_mask())


def gothic_extent_to_seqmap(contact_map, gothic_method):
    """
    Reduce the extent map to a sequence map, where the the most significant locus<->locus interaction
    between two contigs is taken as the indicator of overall significant for the contig<->contig interaction.

    The extent map splits contigs into small windows. This is how GotHiC was originally conceived, as there
    is no accounting for difference in interaction length between sequences. Therefore, here we use the
    binned extent map, calculate the interaction significances between all windows (loci) and then
    select a representative interactions between entire contigs via a summary statistic.

    :param contact_map: the contact map to use
    :param gothic_method: GotHiC normalisation method (binomial, poisson, effect)
    :return: seq_map where each element represents the maximally significant locus<->locus interaction between contigs
    """

    assert gothic_method in {'gothic-binomial', 'gothic-poisson'}, \
        'Normalisation method must be either gothic-binomial or gothic-poisson'

    # for now, we want to act on the complete contact map not a fitlered version
    _mask = contact_map.set_primary_acceptance_mask(min_sig=0, min_len=0, update=True)
    contact_map.order.set_mask_only(_mask)
    # get the full extent map and the bins
    _extent_map = contact_map.get_extent_map(norm=True, norm_method=gothic_method)
    # TODO to support filtered maps, we need to handle filtered bins
    _extent_bins = contact_map.grouping.bins

    if not sp.isspmatrix_csr(_extent_map):
        _extent_map = _extent_map.tocsr()

    _num_bins = len(_extent_bins)

    # cumulative extent bins, including 0
    _cbins = np.concatenate(([0], np.cumsum(_extent_bins)))

    _rows = []
    _cols = []
    _data = []

    # this method is kind of slow, so we've got a progress par
    for ix in tqdm.tqdm(range(_num_bins)):

        # diagonal (self bin)
        a0, a1 = _cbins[ix], _cbins[ix+1]

        _bin_slice = _extent_map[a0:a1, :]

        # just the upper triangle along the diagonal
        c = [ix]

        # check if tocsc() prior to slicing here is faster
        # TODO decide how best to handle multiple interaction terms.
        d = [sp.triu(_bin_slice.T[a0:a1]).mean()]

        _bin_slice = _bin_slice.tocoo()
        c_others, d_others = max_interactions(ix, _cbins, _bin_slice.col, _bin_slice.data)

        c += c_others
        d += d_others

        _rows.extend([ix] * len(c))
        _cols.extend(c)
        _data.extend(d)

    # construct a sparse output map, where each element is now a contig<->contig contact
    _out_map = sp.coo_matrix((_data, (_rows, _cols)), shape=(_num_bins, _num_bins), dtype=np.float64)
    _out_map.eliminate_zeros()
    return _out_map


def fast_norm_gothic(rows, cols, data, rel_cov, total_links, frac_random, mode='binomial'):
    """
    Inplace application of GOTHiC link significance. Here, modifications have been
    made to approximate the Binomial using the Poisson for large N and small p.
    This allows the easy implementation of a fast Poisson CDF calculator.
    :param rows: COO matrix rows
    :param cols: COO matrix cols
    :param data: COO matrix data values to modify
    :param rel_cov: relative coverages
    :param total_links: total number of links (pairs) in the map
    :param frac_random: the fraction of read-pairs in the Hi-C library arising from spurious ligation
    :param mode: 'binomial': -log binnom signif, 'poisson': -log poisson signif, 'effect': effect-size
    """
    # we will cap the smallest values to tiny, preventing log errors
    tiny = np.finfo(data.dtype).tiny
    pij = 2 * rel_cov[rows] * rel_cov[cols] * frac_random
    if mode == 'binomial':
        pr = binom.sf(data - 1, total_links, pij)
        data[:] = - np.log(np.where(pr < tiny, tiny, pr))
    elif mode == 'poisson':
        # assume for large N and small Pr that Poisson and Binomial are very similar
        pr = poisson.sf(data - 1, total_links * pij)
        data[:] = - np.log(np.where(pr < tiny, tiny, pr))
    elif mode == 'effect':
        data /= pij * total_links
        data += 1
        data[:] = np.log2(data)
    else:
        raise ApplicationException('unsupported mode [{}]'.format(mode))


@nb.jit(nopython=True)
def fast_norm_fullseq_bysite(rows, cols, data, sites):
    """
    In-place normalisation of the scipy.coo_matrix for full sequences

    :param rows: the COO matrix coordinate member variable (4xN array)
    :param cols: the COO matrix coordinate member variable (4xN array)
    :param data:  the COO matrix data member variable (1xN array)
    :param sites: per-element min(sequence_length, tip_size)
    """
    for n in range(data.shape[0]):
        i = rows[n]
        j = cols[n]
        data[n] *= 1.0/(sites[i] * sites[j])


@nb.jit(nopython=True, parallel=True)
def fast_length_norm(row, col, data, nnz, len_lookup, mean_func):
    """
    Normalise an extent map by contig length. This method is intended
    to be used internally on the members of a scipy.sparse COO matrix.

    :param data: coo data member
    :param row:  coo row member
    :param col:  coo col member
    :param nnz:  coo nnz attribute
    :param len_lookup: contig length lookup for any index of extent map
    :param mean_func: mean function to apply to length_i and length_j
    :return: the normalised data array only
    """

    for n in nb.prange(nnz):
        w_ij = 1e-3 * mean_func(len_lookup[row[n]], len_lookup[col[n]])
        data[n] /= w_ij


class ExtentGrouping(object):

    def __init__(self, seq_info, bin_size):
        self.bins = []
        self.bin_size = bin_size
        self.map = []
        self.borders = []
        self.centers = []
        self.total_bins = 0

        for n, seq in tqdm.tqdm(enumerate(seq_info), total=len(seq_info), desc='Making bins'):

            if seq.length == 0:
                raise ZeroLengthException(seq.id)

            # integer bin estimation
            num_bins = seq.length / bin_size
            if num_bins == 0:
                num_bins += 1
            # handle non-integer discrepancy by contracting/expanding all bins equally
            # the threshold between contract/expand being half a bin size
            if seq.length % bin_size != 0 and seq.length/float(bin_size) - num_bins >= 0.5:
                num_bins += 1

            edges = np.linspace(0, seq.length, num_bins+1, endpoint=True, dtype=np.int)

            self.bins.append(num_bins)

            # Per reference coordinate pairs (bin_edge, map_index)
            first_bin = self.total_bins
            last_bin = first_bin + num_bins
            self.map.append(np.vstack((edges[1:], np.arange(first_bin, last_bin))).T)
            self.borders.append(np.array([first_bin, last_bin], dtype=np.int))

            self.total_bins += num_bins

            c_nk = edges[:-1] + 0.5*(edges[1] - edges[0]) - 0.5*seq.length
            self.centers.append(c_nk.reshape((1, len(c_nk))))
            # logger.debug('{}: {} bins'.format(n, num_bins))

        self.bins = np.array(self.bins)

    def get_bin_lengths(self):
        """
        Compute the width of each bin. The bin widths vary slightly, as the grouping algorithm
        attempts to disperse extra extent across all bins.
        :return: bin lengths
        """
        bin_len = np.zeros(self.total_bins, dtype=np.int64)
        n = 0
        for i in range(len(self.map)):
            bin_len[n] = self.map[i][0, 0]
            n += 1
            for j in range(1, len(self.map[i]) ):
                bin_len[n] = self.map[i][j, 0] - self.map[i][j-1, 0]
                n += 1
        return bin_len


class SeqOrder(object):

    FORWARD = 1
    REVERSE = -1

    ACCEPTED = True
    EXCLUDED = False

    STRUCT_TYPE = np.dtype([('pos', np.int32), ('ori', np.int8), ('mask', np.bool), ('length', np.int32)])
    INDEX_TYPE = np.dtype([('index', np.int32), ('ori', np.int8)])

    def __init__(self, seq_info):
        """
        Initial order is determined by the order of supplied sequence information dictionary. Sequences
        are given surrogate ids using consecutive integers. Member functions expect surrogate ids
        not original names.

        The class also retains orientation and masking state. Orientation defines whether a sequence
        should be in its original direction (as read in) (1) or reverse complemented (-1).

        Masking state defines whether a input sequence shall been excluded from further consideration.
        (accepted=1, excluded=0)

        :param seq_info: sequence information dictionary
        """
        self._positions = None
        _ord = np.arange(len(seq_info), dtype=np.int32)
        self.order = np.array(
            [(_ord[i], SeqOrder.FORWARD, SeqOrder.ACCEPTED, seq_info[i].length) for i in range(len(_ord))],
            dtype=SeqOrder.STRUCT_TYPE)

        self._update_positions()

    @staticmethod
    def asindex(_ord):
        """
        Convert a simple list or ndarray of indices, to INDEX_TYPE array with default forward orientation.

        :param _ord: list/ndarray of indices
        :return: INDEX_TYPE array
        """
        assert isinstance(_ord, (list, np.ndarray)), 'input must be a list or ndarray'
        return np.array(zip(_ord, np.ones_like(_ord, dtype=np.bool)), dtype=SeqOrder.INDEX_TYPE)

    def _update_positions(self):
        """
        An optimisation, whenever the positional state changes, this method must be called to
        maintain the current state in a separate array. This avoids unnecessary recalculation
        overhead.
        """
        # Masked sequences last, then by current position.
        sorted_indices = np.lexsort([self.order['pos'], ~self.order['mask']])
        for n, i in enumerate(sorted_indices):
            self.order[i]['pos'] = n
        self._positions = np.argsort(self.order['pos'])

    def remap_gapless(self, gapless_indices):
        """
        Recover the original, potentially sparse (gapped) indices from a dense (gapless) set
        of indices. Gaps originate from sequences being masked in the order. External tools
        often expect and return dense indices. When submitting changes to the current order
        state, it is important to first apply this method and reintroduce any gaps.

        Both a list/array of indices or a INDEX_TYPE array can be passed.

        :param gapless_indices: dense list of indices or an ndarray of type INDEX_TYPE
        :return: remappped indices with gaps (of a similar type to input)
        """
        # not as yet verified but this method is being replaced by the 50x faster numpy
        # alternative below. The slowless shows for large problems and repeated calls.
        # we ~could~ go further and maintain the shift array but this will require
        # consistent and respectful (fragile) use of mutator methods and not direct access on mask
        # or an observer.

        # the accumulated shifts due to masked sequences (the gaps).
        # we remove the masked sequences to make this array gapless
        shift = np.cumsum(~self.order['mask'])[self.order['mask']]

        # now reintroduce the gaps to the gapless representation supplied

        remapped = []
        # handle our local type
        if isinstance(gapless_indices, np.ndarray) and gapless_indices.dtype == SeqOrder.INDEX_TYPE:
            for oi in gapless_indices:
                remapped.append((oi['index'] + shift[oi['index']], oi['ori']))
            remapped = np.array(remapped, dtype=SeqOrder.INDEX_TYPE)
        # handle plain collection
        else:
            for oi in gapless_indices:
                remapped.append(oi + shift[oi])
            remapped = np.array(remapped)

        return remapped

    def accepted_positions(self, copy=True):
        """
        The current positional order of only those sequences which have not been excluded by the mask.
        :param copy: return a copy.

        Note: see usage of all_positions() for warning about when positional data must be refreshed.

        :return: all accepted positons in order of index
        """
        return self.all_positions(copy=copy)[:self.count_accepted()]

    def all_positions(self, copy=True):
        """
        The current positional order of all sequences. Internal logic relegates masked sequences to always come
        last and ascending surrogate id order.

        Note: positions are updated when ContactMap.__init__(), .mask(), .set_mask_only(), .set_order_and_orientation()
        and .shuffle() are called. Users should take care when copying and then using outdated positional data.

        :param copy: return a copy of the positions
        :return: all positions in order of index, masked or not.
        """
        if copy:
            _p = self._positions.copy()
        else:
            _p = self._positions
        return _p

    @staticmethod
    def double_order(_ord):
        """
        For doublet maps, the stored order must be re-expanded to reference the larger (2x) map.

        :param _ord:
        :return: expanded order
        """
        return np.array([[2*oi, 2*oi+1] for oi in _ord]).ravel()

    def gapless_positions(self):
        """
        A dense index range representing the current positional order without masked sequences. Therefore
        the returned array does not contain surrogate ids, but rather the relative positions of unmasked
        sequences, when all masked sequences have been discarded.

        :return: a dense index range of positional order, once all masked sequences have been discarded.
        """
        # accumulated shift from gaps
        gap_shift = np.cumsum(~self.order['mask'])
        # just unmasked sequences
        _p = np.argsort(self.order['pos'])
        _p = _p[:self.count_accepted()]
        # removing gaps leads to a dense range of indices
        _p -= gap_shift[_p]
        return _p

    def set_mask_only(self, _mask):
        """
        Set the mask state of all sequences, where indices in the mask map to
        sequence surrogate ids.

        :param _mask: mask array or list, boolean or 0/1 valued
        """
        _mask = np.asarray(_mask, dtype=np.bool)
        assert len(_mask) == len(self.order), 'supplied mask must be the same length as existing order'
        assert np.all((_mask == SeqOrder.ACCEPTED) | (_mask == SeqOrder.EXCLUDED)), \
            'new mask must be {} or {}'.format(SeqOrder.ACCEPTED, SeqOrder.EXCLUDED)

        # assign mask
        self.order['mask'] = _mask
        self._update_positions()

    def set_order_only(self, _ord, implicit_excl=False):
        """
        Convenience method to set the order using a list or 1D ndarray. Orientations will
        be assumed as all forward (+1).

        :param _ord: a list or ndarray of surrogate ids
        :param implicit_excl: implicitly extend the order to include unmentioned excluded sequences.
        """
        assert isinstance(_ord, (list, np.ndarray)), 'Wrong type supplied, order must be a list or ndarray'
        if isinstance(_ord, np.ndarray):
            _ord = np.ravel(_ord)
            assert np.ndim(_ord) == 1, 'orders as numpy arrays must be 1-dimensional'
        # augment the order to include default orientations
        _ord = SeqOrder.asindex(_ord)
        self.set_order_and_orientation(_ord, implicit_excl=implicit_excl)

    def set_order_and_orientation(self, _ord, implicit_excl=False):
        """
        Set only the order, while ignoring orientation. An ordering is defined
        as a 1D array of the structured type INDEX_TYPE, where elements are the
        position and orientation of each indices.

        NOTE: This definition can be the opposite of what is returned by some
        ordering methods, and np.argsort(_v) should inverse the relation.

        NOTE: If the order includes only active sequences, setting implicit_excl=True
        the method will implicitly assume unmentioned ids are those currently
        masked. An exception is raised if a masked sequence is included in the order.

        :param _ord: 1d ordering
        :param implicit_excl: implicitly extend the order to include unmentioned excluded sequences.
        """
        assert _ord.dtype == SeqOrder.INDEX_TYPE, 'Wrong type supplied, _ord should be of INDEX_TYPE'

        if len(_ord) < len(self.order):
            # some sanity checks
            assert implicit_excl, 'Use implicit_excl=True for automatic handling ' \
                                  'of orders only mentioning accepted sequences'
            assert len(_ord) == self.count_accepted(), 'new order must mention all ' \
                                                       'currently accepted sequences'
            # those surrogate ids mentioned in the order
            mentioned = set(_ord['index'])
            assert len(mentioned & set(self.excluded())) == 0, 'new order and excluded must not ' \
                                                               'overlap when using implicit assignment'
            assert len(mentioned ^ set(self.accepted())) == 0, 'incomplete new order supplied,' \
                                                               'missing accepted ids'
            # assign the new orders
            self.order['pos'][_ord['index']] = np.arange(len(_ord), dtype=np.int32)
            self.order['ori'][_ord['index']] = _ord['ori']
            # mask remaining, unmentioned indices
            _mask = np.zeros_like(self.mask_vector(), dtype=np.bool)
            _mask[_ord['index']] = True
            self.set_mask_only(_mask)
        else:
            # just a simple complete order update
            assert len(_ord) == len(self.order), 'new order was a different length'
            assert len(set(_ord['index']) ^ set(self.accepted())) == 0, 'incomplete new order supplied,' \
                                                                        'missing accepted ids'
            self.order['pos'][_ord['index']] = np.arange(len(_ord), dtype=np.int32)
            self.order['ori'][_ord['index']] = _ord['ori']

        self._update_positions()

    def accepted_order(self):
        """
        :return: an INDEX_TYPE array of the order and orientation of the currently accepted sequences.
        """
        idx = np.where(self.order['mask'])
        ori = np.ones(self.count_accepted(), dtype=np.int)
        return np.array(zip(idx, ori), dtype=SeqOrder.INDEX_TYPE)

    def mask_vector(self):
        """
        :return: the current mask vector
        """
        return self.order['mask']

    def mask(self, _id):
        """
        Mask an individual sequence by its surrogate id

        :param _id: the surrogate id of sequence
        """
        self.order[_id]['mask'] = False
        self._update_positions()

    def count_accepted(self):
        """
        :return: the current number of accepted (unmasked) sequences
        """
        return self.order['mask'].sum()

    def count_excluded(self):
        """
        :return: the current number of excluded (masked) sequences
        """
        return len(self.order) - self.count_accepted()

    def accepted(self):
        """
        :return: the list surrogate ids for currently accepted sequences
        """
        return np.where(self.order['mask'])[0]

    def excluded(self):
        """
        :return: the list surrogate ids for currently excluded sequences
        """
        return np.where(~self.order['mask'])[0]

    def flip(self, _id):
        """
        Flip the orientation of the sequence

        :param _id: the surrogate id of sequence
        """
        self.order[_id]['ori'] *= -1

    def lengths(self, exclude_masked=False):
        # type: (bool) -> np.ndarray
        """
        Sequence lengths

        :param exclude_masked: True include only umasked sequencces
        :return: the lengths of sequences
        """
        if exclude_masked:
            return self.order['length'][self.order['mask']]
        return self.order['length']

    def shuffle(self):
        """
        Randomize order
        """
        np.random.shuffle(self.order['pos'])
        self._update_positions()

    def before(self, a, b):
        """
        Test if a comes before another sequence in order.

        :param a: surrogate id of sequence a
        :param b: surrogate id of sequence b
        :return: True if a comes before b
        """
        assert a != b, 'Surrogate ids must be different'
        return self.order['pos'][a] < self.order['pos'][b]

    def intervening(self, a, b):
        """
        For the current order, calculate the length of intervening
        sequences between sequence a and sequence b.

        :param a: surrogate id of sequence a
        :param b: surrogate id of sequence b
        :return: total length of sequences between a and b.
        """
        assert a != b, 'Surrogate ids must be different'

        pa = self.order['pos'][a]
        pb = self.order['pos'][b]
        if pa > pb:
            pa, pb = pb, pa
        inter_ix = self._positions[pa+1:pb]
        return np.sum(self.order['length'][inter_ix])


class ContactMap(object):

    def append_map(self, other):
        if not isinstance(other, ContactMap):
            raise ValueError('ContactMap value is required')

        if self.extent_map is not None or other.extent_map is not None:
            logger.error('Appending contact maps with extent mapping not implemented')

        # issue debug warnings if the following attributes are different
        for _attr in ['min_mapq', 'min_insert', 'min_len', 'min_sig', 'min_extent',
                      'min_size', 'max_fold', 'max_edist', 'min_alen']:
            a = self.__dict__[_attr]
            b = other.__dict__[_attr]
            if a != b:
                logger.debug('Differing values for attribute {}: {} and {}'.format(_attr, a, b))

        # raise an error if the following attributes are different
        for _attr in ['bin_size', 'tip_size']:
            a = self.__dict__[_attr]
            b = other.__dict__[_attr]
            if a != b:
                logger.error('Cannot combine contact maps with differing values for attribute {}: {} and {}'
                             .format(_attr, a, b))

        # compare the sequence sets on name, length and number of sites.
        # we assume for potentially non-unique sequence naming practices, checking
        # the number of sites is likely a decent proxy for comparing actual sequences
        # Note: we also assume the sequence orders are the same, which greatly simplifies
        # combining arrays.
        a_info = [(si.name, si.length, si.sites, si.gc) for si in self.seq_info]
        b_info = [(si.name, si.length, si.sites, si.gc) for si in other.seq_info]
        if a_info != b_info:
            logger.error('Cannot combine contact maps with differing sets of DNA sequences.')

        if self.seq_map.shape != other.seq_map.shape:
            logger.error('Cannot combine contact maps with differing dimensions {} vs {}'
                         .format(self.seq_map.shape, other.seq_map.shape))

        logger.debug('Combining sequence maps with size {}'.format(self.seq_map.shape))
        logger.debug('Initial total map weights: {:,} and {:,}'.format(self.map_weight(), other.map_weight()))
        self.seq_map = sparse_utils.add_matrices(self.seq_map, other.seq_map)
        logger.debug('Final total map weight: {:,}'.format(self.map_weight()))

        logger.debug('Reinitializing primary acceptance mask')
        self.set_primary_acceptance_mask(update=True)

    def __init__(self, bam_file, enzymes, seq_file, min_separation, min_mapq=0, min_len=0, min_sig=1, min_extent=0,
                 min_size=0, max_edist=2, min_alen=25, max_fold=None, random_seed=None, bin_size=None, tip_size=None,
                 no_duplicates=True, precount=False, threads=4):

        self.no_duplicates = no_duplicates
        self.bam_file = bam_file
        self.bin_size = bin_size
        self.min_mapq = min_mapq
        self.max_edit_distance = max_edist
        self.min_align_length = min_alen
        self.min_separation = min_separation
        self.min_len = min_len
        self.min_sig = min_sig
        self.min_extent = min_extent
        self.min_size = min_size
        self.max_fold = max_fold
        self.random_state = np.random.RandomState(random_seed)
        self.seq_info = []
        self.seq_map = None
        self.seq_file = seq_file
        self.grouping = None
        self.extent_map = None
        self.order = None
        self.tip_size = tip_size
        self.precount = precount
        self.total_reads = None
        self.processed_map = None
        self.primary_acceptance_mask = None
        self.bisto_scale = None
        self.enzymes = enzymes
        self.site_counter = None

        # prepare the site counter for the given experimental conditions
        assert 0 < len(enzymes) <= 2, 'no more than two enzymes can be specified'
        self.site_counter = SiteCounter(*enzymes, tip_size=tip_size, is_linear=True)

        # build a dictionary of reference features
        fasta_info = {}
        with io_utils.open_input(seq_file) as multi_fasta:
            # get an estimate of sequences for progress
            fasta_count = count_fasta_sequences(seq_file)
            for n_seq, seqrec in tqdm.tqdm(enumerate(SeqIO.parse(multi_fasta, 'fasta')),
                                           total=fasta_count, desc='Analyzing reference sequences'):
                if len(seqrec) < min_len:
                    continue
                fasta_info[seqrec.id] = {'sites': self.site_counter.count_sites(seqrec.seq),
                                         'length': len(seqrec),
                                         'gc': SeqUtils.GC(seqrec.seq)}
            logger.info('From FASTA, {} of {} sequences were accepted'.format(n_seq, len(fasta_info)))

        # now inspect the BAM header
        with pysam.AlignmentFile(bam_file, 'rb', threads=threads) as bam:

            # test that BAM file is the correct sort order
            if 'SO' not in bam.header['HD'] or bam.header['HD']['SO'] != 'queryname':
                raise IOError('BAM file must be sorted by read name')

            # keep a record of all reference lengths
            self.refid_to_reflen = np.array([li for li in bam.lengths], dtype=np.int64)

            # determine the set of active sequences
            # where the first filtration step is by length
            ref_count = {'seq_missing': 0, 'too_short': 0}
            offset = 0
            logger.info('Reading sequences...')
            for n, (rname, rlen) in enumerate(zip(bam.references, bam.lengths)):

                # minimum length threshold
                if rlen < min_len:
                    ref_count['too_short'] += 1
                    continue

                try:
                    fa = fasta_info[rname]
                except KeyError:
                    logger.info('From BAM, reference {} was not present in supplied fasta'.format(rname))
                    ref_count['seq_missing'] += 1
                    continue

                assert fa['length'] == rlen, \
                    'BAM and FASTA lengths do not agree for reference {}: {} != {}'.format(rname, fa['length'], rlen)

                self.seq_info.append(SeqInfo(offset, n, rname, rlen, fa['sites'], fa['gc']))

                offset += rlen

            # total extent covered
            self.total_len = offset
            self.total_seq = len(self.seq_info)

            # all sequences begin as active in mask
            # when filtered, mask[i] = False
            self.current_mask = np.ones(self.total_seq, dtype=np.bool)

            if self.total_seq == 0:
                logger.info('No sequences in BAM found in FASTA')
                raise ParsingError('No sequences in BAM found in FASTA')

            logger.info('Accepted {} sequences covering {} bp'.format(self.total_seq, self.total_len))
            logger.info('References excluded: {}'.format(ref_count))

            if self.bin_size:
                logger.info('Determining bins...')
                self.grouping = ExtentGrouping(self.seq_info, self.bin_size)

            if self.precount:
                logger.info('Counting reads in bam file for ETA projection...')
                self.total_reads = count_bam_reads(bam_file)
                logger.info('BAM file contains {0} alignments'.format(self.total_reads))
            else:
                logger.info('Skipping pre-count of BAM file, no ETA will be offered')

            # initialise the order
            self.order = SeqOrder(self.seq_info)

            # accumulate
            self._bin_map(bam)

            # create an initial acceptance mask
            self.set_primary_acceptance_mask()

    def _bin_map(self, bam):
        """
        Accumulate read-pair observations from the supplied BAM file.
        Maps are initialized here. Logical control is achieved through initialisation of the
        ContactMap instance, rather than supplying this function arguments.

        :param bam: this instance's open bam file.
        """

        def _strict_acceptance(r):
            """
            Carefully parse the read mapping record for suitability. This tests mapping quality,
            cigar existence, alignment length, edit distance, first read position, and the condition
            that ended the alignment. Reads which terminate before their 3p end is reached, must either
            exceed the reference extent or terminate at the expected ezymatic cutsite.
            :param r: the read to test
            :return: True - the read mapping is accepted, False - it is rejected
            """

            if r.mapping_quality < _min_mapq:
                counts['mapq'] += 1
                return False

            if r.cigarstring is None:
                counts['cigar'] += 1
                return False

            if r.query_length < _min_alen:
                counts['alen'] += 1
                return False

            # restrict the maximum allowed edit distance
            # This assumes BWA MEM style records, where NM = edit distance
            try:
                ed = r.get_tag('NM')
                if ed > _max_edist:
                    counts['edist'] += 1
                    return False
            except KeyError:
                counts['edist'] += 1
                return False

            # insist that read alignments begin at position 0.
            cig = r.cigartuples[-1] if r.is_reverse else r.cigartuples[0]
            if cig[0] != 0:  # 0 -> Match
                counts['5p_match'] += 1
                return False

            # accept full-length alignments that exceed minimum
            if r.query_alignment_length == r.query_length:
                return True

            if r.is_reverse:
                # accept alignments where the 3' end goes beyond the end of the reference
                if r.reference_start == 0:
                    return True
                # extract the read's aligned sequence
                seq = revcomp(r.seq)
                aln_seq = seq[r.query_length - r.query_alignment_end: r.query_length - r.query_alignment_start]
            else:
                # accept alignments where the 3' end goes beyond the end of the reference
                if r.reference_end >= _refid_to_reflen[r.reference_id]:
                    return True
                # extract the read's aligned sequence
                seq = r.seq
                aln_seq = seq[r.query_alignment_start: r.query_alignment_end]

            # accept alignments which terminate at cute-site remnant
            _match = _endswith_vestigial(aln_seq)
            if _match is None:
                counts['cs_end'] += 1
                return False

            return True

        def _next_informative(_bam_iter, _pbar):
            while True:
                r = _bam_iter.next()
                _pbar.update()
                if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate:
                    continue
                break
            return r

        def _on_tip_withlocs(p1, p2, l1, l2, _tip_size):
            tailhead_mat = np.zeros((2, 2), dtype=np.uint32)
            i = None
            j = None

            # contig1 tips won't overlap
            if l1 > 2 * _tip_size:
                if p1 < _tip_size:
                    i = 0
                elif p1 > l1 - _tip_size:
                    i = 1

            # contig1 tips will overlap
            else:
                # assign to whichever end is closest
                if p1 < l1 - p1:
                    i = 0
                elif l1 - p1 < p1:
                    i = 1

            # only bother with second tip assignment if the first was ok
            if i is not None:

                # contig2 tips won't overlap
                if l2 > 2 * _tip_size:
                    if p2 < _tip_size:
                        j = 0
                    elif p2 > l2 - _tip_size:
                        j = 1

                # contig2 tips will overlap
                else:
                    # assign to whichever end is closest
                    if p2 < l2 - p2:
                        j = 0
                    elif l2 - p2 < p2:
                        j = 1

            tailhead_mat[i, j] = 1
            return i is not None and j is not None, tailhead_mat

        def _always_true(*args):
            return True, 1

        # lookup table for reference lengths
        _refid_to_reflen = self.refid_to_reflen
        # set read acceptance method
        _accept_read = _strict_acceptance
        # prepare method which checks that alignments strings end in a vestigial cut-site.
        _endswith_vestigial = self.site_counter.get_vestigial_end_searcher()

        # set tip acceptance method
        _on_tip = _always_true if not self.is_tipbased() else _on_tip_withlocs

        # initialise a sparse matrix for accumulating the map
        if not self.is_tipbased():
            # just a basic NxN sparse array for normal whole-sequence binning
            _seq_map = sparse_utils.Sparse2DAccumulator(self.total_seq)
        else:
            # each tip is tracked separately resulting in the single count becoming a 2x2 interaction matrix.
            # therefore the tensor has dimension NxNx2x2
            _seq_map = sparse_utils.Sparse4DAccumulator(self.total_seq)

        # if binning also requested, initialise another sparse matrix
        if self.bin_size:
            logger.info('Initialising contact map of {0}x{0} fragment bins, '
                        'representing {1} bp over {2} sequences'.format(self.grouping.total_bins,
                                                                        self.total_len, self.total_seq))
            _extent_map = sparse_utils.Sparse2DAccumulator(self.grouping.total_bins)
            _grouping_map = self.grouping.map
        else:
            _grouping_map = None
            _extent_map = None

        with tqdm.tqdm(total=self.total_reads) as progress_bar:

            # locals for read filtering
            _min_sep = self.min_separation
            _min_mapq = self.min_mapq
            _min_alen = self.min_align_length
            _max_edist = self.max_edit_distance

            _idx = self.make_reverse_index('refid')

            # locals for tip checking
            _len = bam.lengths
            _tip_size = self.tip_size

            counts = OrderedDict({
                'accepted': 0,
                'mapq': 0,
                'edist': 0,
                'alen': 0,
                'cigar': 0,
                'cs_end': 0,
                '5p_match': 0,
                'not_tip': 0,
                'short_insert': 0,
                'ref_excluded': 0,
                'median_excluded': 0,
                'end_buffered': 0,
                'poor_match': 0})

            pair_store = None
            if self.no_duplicates:
                pair_store = defaultdict(int)

            bam.reset()
            bam_iter = bam.fetch(until_eof=True)
            while True:

                try:
                    r1 = _next_informative(bam_iter, progress_bar)
                    while True:
                        # read records until we get a pair
                        r2 = _next_informative(bam_iter, progress_bar)
                        if r1.query_name == r2.query_name:
                            break
                        r1 = r2
                except StopIteration:
                    break

                if r1.reference_id not in _idx or r2.reference_id not in _idx:
                    counts['ref_excluded'] += 1
                    continue

                if not _accept_read(r1) or not _accept_read(r2):
                    counts['poor_match'] += 1
                    continue

                if r1.is_read2:
                    r1, r2 = r2, r1

                # use 5-prime base depending on orientation
                r1pos = r1.reference_start if not r1.is_reverse else r1.reference_end
                r2pos = r2.reference_start if not r2.is_reverse else r2.reference_end

                if pair_store is not None:
                    # calculate a map identifier (map_id) from the pair
                    # assuming identical map_id implies technical duplication, we count it only once
                    map_id = hash((r1.reference_id, r1pos, r1.is_reverse, r1.cigarstring,
                                   r2.reference_id, r2pos, r2.is_reverse, r2.cigarstring))
                    pair_store[map_id] += 1
                    if pair_store[map_id] > 1:
                        continue

                # filter inserts deemed "short" which tend to be heavily WGS signal
                if _min_sep:
                    if r1.reference_id == r2.reference_id:
                        _pair_sep = r2pos - r1pos if r1pos <= r2pos else r1pos - r2pos
                    else:
                        # take the minimum distance each read lies from the ends of its reference
                        r1_shortest_edge = min(_refid_to_reflen[r1.reference_id] - r1.reference_end, r1.reference_start)
                        r2_shortest_edge = min(_refid_to_reflen[r2.reference_id] - r2.reference_end, r2.reference_start)
                        # attribute the least possible separation
                        _pair_sep = r1_shortest_edge + r2_shortest_edge
                    if _pair_sep < _min_sep:
                        counts['short_insert'] += 1
                        continue

                # get reference lengths
                l1 = _len[r1.reference_id]
                l2 = _len[r2.reference_id]

                # get internal indices
                ix1 = _idx[r1.reference_id]
                ix2 = _idx[r2.reference_id]

                # maintain just a half-matrix
                if ix2 < ix1:
                    ix1, ix2 = ix2, ix1
                    r1pos, r2pos = r2pos, r1pos
                    l1, l2 = l2, l1

                if _extent_map:
                    b1 = find_nearest_jit(_grouping_map[ix1], r1pos)
                    b2 = find_nearest_jit(_grouping_map[ix2], r2pos)

                    # maintain half-matrix
                    if b1 > b2:
                        b1, b2 = b2, b1

                    # tally all mapped reads for binned map, not just those considered in tips
                    _extent_map[b1, b2] += 1

                # for seq-map, we may reject reads outside of a defined tip region
                tip_info = _on_tip(r1pos, r2pos, l1, l2, _tip_size)
                if not tip_info[0]:
                    counts['not_tip'] += 1
                    continue

                counts['accepted'] += 1

                _seq_map[ix1, ix2] += tip_info[1]

        # default to always making matrices symmetric
        if self.bin_size:
            self.extent_map = _extent_map.get_coo()
            del _extent_map

        self.seq_map = _seq_map.get_coo()
        del _seq_map

        # calculate the proportion of duplicate pair mappings and
        # a truncated histogram covering observed duplicates between 0 to 10+ times
        if self.no_duplicates:
            map_count = np.bincount(pair_store.values(), minlength=11)
            dupe_rate = map_count[2:].sum() / map_count.sum(dtype=np.float64)
            map_count[10] = map_count[10:].sum()
            del pair_store
            logger.debug('Duplication histogram: {}'.format([(n, ci) for n, ci in enumerate(map_count[:11])]))
            logger.info('Duplication rate: {:.2f}%'.format(dupe_rate * 100))
        else:
            logger.warning('Duplicate removal was disabled by user')

        logger.info('Pair accounting: {}'.format(counts))
        logger.info('Total extent map weight {}'.format(self.map_weight()))

    @staticmethod
    def get_fields():
        """
        :return: the list of fields used in seq_info dict.
        """
        return SeqInfo._fields

    def make_reverse_index(self, field_name):
        """
        Make a reverse look-up (dict) from the chosen field in seq_info to the internal index value
        of the given sequence. Non-unique fields will raise an exception.

        :param field_name: the seq_info field to use as the reverse.
        :return: internal array index of the sequence
        """
        rev_idx = {}
        for n, seq in enumerate(self.seq_info):
            fv = getattr(seq, field_name)
            if fv in rev_idx:
                raise RuntimeError('field contains non-unique entries, a 1-1 mapping cannot be made')
            rev_idx[fv] = n
        return rev_idx

    def map_weight(self):
        """
        :return: the total map weight (sum ij)
        """
        return self.seq_map.sum()

    def is_empty(self):
        """
        :return: True if the map has zero weight
        """
        return self.map_weight() == 0

    def is_tipbased(self):
        """
        :return: True if the seq_map is a tip-based 4D tensor
        """
        return self.tip_size is not None

    def find_order(self, _map, seed, inverse_method='inverse', runs=5, work_dir='.'):
        """
        Using LKH TSP solver, find the best ordering of the sequence map in terms of proximity ligation counts.
        Here, it is assumed that sequence proximity can be inferred from number of observed trans read-pairs, where
        an inverse relationship exists.

        :param _map: the seq map to analyze
        :param seed: a random seed
        :param inverse_method: the chosen inverse method for converting count (similarity) to distance
        :param runs: number of individual runs of lkh to perform
        :param work_dir: working directory
        :return: the surrogate ids in optimal order
        """

        # a minimum of three sequences is required to run LKH
        if _map.shape[0] < 3:
            raise TooFewException(_map.shape[0], 'find_order')

        # we'll supply a partially initialized distance function
        dist_func = partial(ordering.similarity_to_distance, method=inverse_method, alpha=1.2, beta=1)

        with open(os.path.join(work_dir, 'lkh.log'), 'w+') as stdout:

            control_base_name = os.path.join(work_dir, 'lkh_run')

            if self.is_tipbased():

                lkh_o = ordering.lkh_order(_map, control_base_name, lkh_exe=package_path('external', 'LKH'), precision=1,
                                           seed=seed, runs=runs, pop_size=50, dist_func=dist_func, special=False,
                                           stdout=stdout, fixed_edges=[(i, i+1) for i in range(1, _map.shape[0], 2)])

                # To solve this with TSP, doublet tips use a graph transformation, where each node comes a pair. Pairs
                # possess fixed inter-connecting edges which must be included in any solution tour.
                # Eg. node 0 -> (0,1) or node 1 -> (2,3). The fixed paths are undirected, depending on which direction
                # is traversed, defines the orientation of the sequence.

                # 1.  pair adjacent nodes by reshape the 1D array into a two-column array of half the length
                lkh_o = lkh_o.reshape(lkh_o.shape[0]/2, 2)
                # 2. convert to surrogate ids and infer orientation from paths taken through doublets.
                #   0->1 forward (+1): 1->0 reverse (-1).
                lkh_o = np.fromiter(((oi[0]/2, oi[1]-oi[0]) for oi in lkh_o), dtype=SeqOrder.INDEX_TYPE)

            else:

                lkh_o = ordering.lkh_order(_map, control_base_name, lkh_exe=package_path('external', 'LKH'), precision=1,
                                           seed=seed, runs=runs, pop_size=50, dist_func=dist_func, special=False,
                                           stdout=stdout)

                # for singlet tours, no orientation can be inferred.
                lkh_o = np.fromiter(((oi, 1) for oi in lkh_o), dtype=SeqOrder.INDEX_TYPE)

        # lkh ordering references the supplied matrix indices, not the surrogate ids.
        # we must map this consecutive set to the contact map indices.
        lkh_o = self.order.remap_gapless(lkh_o)

        return lkh_o

    def get_primary_acceptance_mask(self):
        assert self.primary_acceptance_mask is not None, 'Primary acceptance mask has not be initialized'
        return self.primary_acceptance_mask.copy()

    def set_primary_acceptance_mask(self, min_len=None, min_sig=None, max_fold=None, update=False):
        """
        Determine and set the filter mask using the specified constraints across the entire
        contact map. The mask is True when a sequence is considered acceptable wrt to the
        constraints. The mask is also returned by the function for convenience.

        :param min_len: override instance value for minimum sequence length
        :param min_sig: override instance value for minimum off-diagonal signal (counts)
        :param max_fold: maximum locally-measured fold-coverage to permit
        :param update: replace the current primary mask if it exists
        :return: an acceptance mask over the entire contact map
        """
        assert max_fold is None, 'Filtering on max_fold is currently disabled'

        # If parameter based critiera were unset, use instance member values set at instantiation time
        if min_len is None:
            min_len = self.min_len
        if min_sig is None:
            min_sig = self.min_sig

        assert min_len is not None, 'Filtering criteria min_len is None'
        assert min_sig is not None, 'Filtering criteria min_sig is None'

        logger.debug('Setting primary acceptance mask with '
                     'filtering criterion min_len: {} min_sig: {}'.format(min_len, min_sig))

        # simply return the current mask if it has already been determined
        # and an update is not requested
        if not update and self.primary_acceptance_mask is not None:
            logger.debug('Using existing mask')
            return self.get_primary_acceptance_mask()

        acceptance_mask = np.ones(self.total_seq, dtype=np.bool)

        # mask for sequences shorter than limit
        _mask = self.order.lengths() >= min_len
        logger.debug('Minimum length threshold removing: {}'.format(self.total_seq - _mask.sum()))
        acceptance_mask &= _mask

        # mask for sequences weaker than limit
        if self.is_tipbased():
            signal = sparse_utils.max_offdiag_4d(self.seq_map)
        else:
            signal = sparse_utils.max_offdiag(self.seq_map)
        _mask = signal >= min_sig
        logger.debug('Minimum signal threshold removing: {}'.format(self.total_seq - _mask.sum()))
        acceptance_mask &= _mask

        # retain the union of all masks.
        self.primary_acceptance_mask = acceptance_mask

        logger.debug('Accepted sequences: {}'.format(self.primary_acceptance_mask.sum()))

        return self.get_primary_acceptance_mask()

    def prepare_seq_map(self, norm=True, bisto=False, mean_type='geometric', norm_method='sites'):
        """
        Prepare the sequence map (seq_map) by application of various filters and normalisations.

        :param norm: normalisation by sequence lengths
        :param bisto: make the output matrix bistochastic
        :param mean_type: when performing normalisation, use "geometric, harmonic or arithmetic" mean.
        :param norm_method: normalisation method to apply to contact map
        """

        logger.info('Preparing sequence map with full dimensions: {}'.format(self.seq_map.shape))

        _mask = self.get_primary_acceptance_mask()

        self.order.set_mask_only(_mask)

        if self.order.count_accepted() < 1:
            raise NoneAcceptedException()

        _map = self.seq_map.astype(np.float64)

        # apply length normalisation if requested
        if norm:
            _map = self._norm_seq(_map, self.is_tipbased(), mean_type=mean_type, method=norm_method)
            logger.debug('Map normalized')

        # make map bistochastic if requested
        if bisto:
            # TODO balancing may be better done after compression
            _map, scl = self._bisto_seq(_map)
            # retain the scale factors
            self.bisto_scale = scl
            logger.debug('Map balanced')

        # cache the results for optional quick access
        self.processed_map = _map

    def get_subspace(self, permute=False, external_mask=None, marginalise=False, flatten=True,
                     dtype=np.float64):
        """
        Using an already normalized full seq_map, return a subspace as indicated by an external
        mask or if none is supplied, the full map without filtered elements.

        The supplied external mask must refer to all sequences in the map.

        :param permute: reorder the map with the current ordering state
        :param external_mask: an external mask to combine with the existing primary mask
        :param marginalise: Assuming 4D NxNx2x2 tensor, sum 2x2 elements to become a 2D NxN
        :param flatten: convert a NxNx2x2 tensor to a 2Nx2N matrix
        :param dtype: return map with specific element type
        :return: subspace map
        """
        assert (not marginalise and not flatten) or np.logical_xor(marginalise, flatten), \
            'marginalise and flatten are mutually exclusive'

        # starting with the normalized map
        _map = self.processed_map.astype(dtype)

        # from a union of the sequence filter and external mask
        if external_mask is not None:
            _mask = self.get_primary_acceptance_mask()
            logger.info('Beginning with sequences after primary filtering: {}'.format(_mask.sum()))
            _mask &= external_mask
            logger.info('Active sequences after applying external mask: {}'.format(_mask.sum()))
            self.order.set_mask_only(_mask)

        # remove masked sequences from the map
        if self.order.count_accepted() < self.total_seq:
            if self.is_tipbased():
                _map = sparse_utils.compress_4d(_map, self.order.mask_vector())
            else:
                _map = sparse_utils.compress(_map.tocoo(), self.order.mask_vector())
            logger.info('After removing filtered sequences map dimensions: {}'.format(_map.shape))

        # convert tip-based tensor to other forms
        if self.is_tipbased():
            if marginalise:
                logger.debug('Marginalising NxNx2x2 tensor to NxN matrix')
                # sum counts of the 2x2 confusion matrices into 1 value
                _map = _map.sum(axis=(2, 3)).to_scipy_sparse()
            elif flatten:
                logger.debug('Flattening NxNx2x2 tensor to 2Nx2N matrix')
                # convert the 4D map into a 2Nx2N 2D map.
                _map = sparse_utils.flatten_tensor_4d(_map)

        if permute:
            _map = self._reorder_seq(_map, flatten=flatten)
            logger.debug('Map reordered')

        return _map

    def get_extent_map(self, norm=True, bisto=False, permute=False, mean_type='geometric', norm_method='length'):
        """
        Return the extent map after applying specified processing steps. Masked sequences are always removed.

        :param norm: sequence length normalisation
        :param bisto: make map bistochastic
        :param permute: permute the map using current order
        :param mean_type: length normalisation mean (geometric, harmonic, arithmetic)
        :param norm_method: methods used to normalise the matrix (length, gothic-effect,
        gothic-binomial, gothic-poisson)
        :return: processed extent map
        """

        logger.info('Preparing extent map with full dimension: {}'.format(self.extent_map.shape))

        _map = self.extent_map.astype(np.float64)

        # apply length normalisation if requested
        if norm:
            _map = self._norm_extent(_map, method=norm_method, mean_type=mean_type)
            logger.debug('Map normalized')

        # if there are sequences to mask, remove them from the map
        if self.order.count_accepted() < self.total_seq:
            _map = self._compress_extent(_map)
            logger.info('After removing filtered sequences map dimensions: {}'.format(_map.shape))

        # make map bistochastic if requested
        if bisto:
            _map, scl = sparse_utils.kr_biostochastic(_map)
            logger.debug('Map balanced')

        # reorder using current order state
        if permute:
            _map = self._reorder_extent(_map)
            logger.debug('Map reordered')

        return _map

    def extent_to_seq(self):
        """
        Convert the extent map into a single-pixel per sequence "seq_map". This method
        is useful when only a tip based seq_map has been produced, and an analysis would be
        better done on a full accounting of mapping interactions across each sequences full
        extent.

        :return: a seq_map representing all counts across each sequence
        """
        _map = self.extent_map.tocsr()
        _bins = self.grouping.bins
        _cbins = np.cumsum(_bins)
        _out = sparse_utils.Sparse2DAccumulator(self.total_seq)

        a0 = 0
        for i in range(len(_bins)):
            a1 = _cbins[i]
            # sacrifice memory for significant speed up slicing below
            row_i = _map[a0:a1, :].A
            b0 = 0
            for j in range(i, len(_bins)):
                b1 = _cbins[j]
                mij = row_i[:, b0:b1].sum()
                if mij == 0:
                    continue
                _out[i, j] = int(mij)
                b0 = b1
            a0 = a1
        return _out.get_coo()

    def _reorder_seq(self, _map, flatten=False):
        """
        Reorder a simple sequence map using the supplied map.

        :param _map: the map to reorder
        :param flatten: tip-based tensor converted to 2Nx2N matrix, otherwise the assumption is marginalisation
        :return: ordered map
        """
        assert sp.isspmatrix(_map), 'reordering expects a sparse matrix type'

        _order = self.order.gapless_positions()
        if self.is_tipbased() and flatten:
            _order = SeqOrder.double_order(_order)

        assert _map.shape[0] == _order.shape[0], 'supplied map and unmasked order are different sizes'
        p = sp.lil_matrix(_map.shape)
        for i in range(len(_order)):
            p[i, _order[i]] = 1.
        p = p.tocsr()
        return p.dot(_map.tocsr()).dot(p.T)

    def _bisto_seq(self, _map):
        """
        Make a contact map bistochastic. This is another form of normslisation. Automatically
        handles 2D and 4D maps.

        :param _map: a map to balance (make bistochastic)
        :return: the balanced map
        """
        logger.debug('Balancing contact map')

        if self.is_tipbased():
            _map, scl = sparse_utils.kr_biostochastic_4d(_map)
        else:
            _map, scl = sparse_utils.kr_biostochastic(_map)
        return _map, scl

    def _get_sites(self):
        _sites = np.array([si.sites for si in self.seq_info], dtype=np.float)
        # all sequences are assumed to have a minimum of 1 site -- even if not observed
        # TODO test whether it would be more accurate to assume that all sequences are under counted by 1.
        _sites[np.where(_sites == 0)] = 1
        return _sites

    def _norm_seq(self, _map, tip_based, method='sites', mean_type='geometric', gothic_noself=True):
        """
        Normalise a simple sequence map in place by the geometric mean of interacting contig pairs lengths.
        The map is assumed to be in starting order.

        :param _map: the target map to apply normalisation
        :param tip_based: treat the supplied map as a tip-based tensor
        :param method: the normalisation method to use [sites, length, gothic-effect, gothic-binomial, gothic-poisson]
        :param mean_type: for length normalisation, choice of mean (harmonic, geometric, arithmetic)
        :param gothic_noself: exclude self-self interactions when calculating relative coverage.
        :return: normalized map
        """
        if method == 'sites':

            logger.debug('Doing site based normalisation')
            _sites = self._get_sites()
            _map = _map.astype(np.float)
            if tip_based:
                fast_norm_tipbased_bysite(_map.coords, _map.data, _sites)
            else:
                if not sp.isspmatrix_coo(_map):
                    _map = _map.tocoo()
                fast_norm_fullseq_bysite(_map.row, _map.col, _map.data, _sites)

        elif method == 'length':

            logger.debug('Doing length based normalisation')
            if tip_based:
                _tip_lengths = np.minimum(self.tip_size, self.order.lengths()).astype(np.float)
                fast_norm_tipbased_bylength(_map.coords, _map.data, _tip_lengths, self.tip_size)
            else:
                # TODO convert this to numba or remove
                logger.warning('length normalisation is not optimised and therefore very slow')
                _mean_func = mean_selector(mean_type)
                _len = self.order.lengths().astype(np.float)
                _map = _map.tolil().astype(np.float)
                for i in range(_map.shape[0]):
                    _map[i, :] /= np.fromiter((1e-3 * _mean_func(_len[i],  _len[j])
                                               for j in range(_map.shape[0])), dtype=np.float)
                _map = _map.tocsr()

        elif method.startswith('gothic'):

            if tip_based:
                raise ApplicationException('GOTHiC tip based normalisation not supported')

            if method == 'gothic-effect':
                goth_mode = 'effect'
                logger.debug('Doing GOTHiC based effect size normalisation')
            elif method == 'gothic-binomial':
                goth_mode = 'binomial'
                logger.debug('Doing GOTHiC based Binomial significance normalisation')
            elif method == 'gothic-poisson':
                goth_mode = 'poisson'
                logger.debug('Doing GOTHiC based Poisson significance normalisation')
            else:
                raise ApplicationException('Unknown method {} in GOTHiC normalisation'.format(method))

            if gothic_noself:
                _map = _map.tolil()
                _map.setdiag(0)

            _map = _map.tocsr()

            # take triangular sum as total number of links (pairs) in map
            total_links = sp.triu(_map).sum()

            # calculate relative length-normalized contig coverage
            seq_len = self.order.order['length'].astype(np.float64)
            rel_cov = _map.sum(axis=1).astype(np.float64)
            rel_cov = np.asarray(rel_cov).squeeze()
            # gothic normalises this as reads_j / 2N
            # we introduce relative to the number of 5kb chunks
            rel_cov /= 2 * total_links * (seq_len / 5000.)
            _map = _map.tocoo().astype(np.float64)
            fast_norm_gothic(_map.row, _map.col, _map.data, rel_cov, total_links, 1., goth_mode)

        else:
            raise ApplicationException('unknown method {}'.format(method))

        return _map

    def _norm_extent(self, _map, method='length', mean_type='geometric', gothic_noself=True):
        """
        Normalise a extent map in place by the geometric mean of interacting contig pairs lengths.

       :param method: the normalisation method to use [sites, length, gothic-sig, gothic-es]
       :param mean_type: for length normalisation, choice of mean (harmonic, geometric, arithmetic)
       :param gothic_noself: exclude self-self interactions when calculating relative coverage.
       :return: a normalized extent map in lil_matrix format
        """
        assert sp.isspmatrix(_map), 'Extent matrix is not a scipy matrix type'

        if not sp.isspmatrix_coo(_map):
            _map = _map.tocoo()

        # normalised data array
        if method == 'length':

            logger.debug('Doing length based normalisation')

            # prepare the lookup array mapping contig length to any index of extent map
            _bins = self.grouping.bins
            _len = self.order.lengths()
            _len_lookup = []
            for i in range(len(_bins)):
                _len_lookup.extend([_len[i]] * _bins[i])
            _len_lookup = np.array(_len_lookup, dtype=np.float64)

            fast_length_norm(_map.row, _map.col, _map.data, _map.nnz, _len_lookup, mean_selector(mean_type))

        elif method.startswith('gothic'):

            if method == 'gothic-effect':
                goth_mode = 'effect'
                logger.debug('Doing GOTHiC based effect size normalisation')
            elif method == 'gothic-binomial':
                goth_mode = 'binomial'
                logger.debug('Doing GOTHiC based Binomial significance normalisation')
            elif method == 'gothic-poisson':
                goth_mode = 'poisson'
                logger.debug('Doing GOTHiC based Poisson significance normalisation')
            else:
                raise ApplicationException('Unknown method {} in GOTHiC normalisation'.format(method))

            # as we might change sparsity, using lil avoids a warning from scipy
            _map = _map.tolil()
            _seqmap = self.seq_map.tolil()

            if gothic_noself:
                _map.setdiag(0)
                _seqmap.setdiag(0)

            # get total weight from seq_map as it is much faster
            total_links = sp.triu(_seqmap.tocsr()).sum()
            del _seqmap

            # marginals represent total hits per bin
            _map = _map.tocsr()
            rel_cov = _map.sum(axis=1).astype(np.float64)
            rel_cov = np.asarray(rel_cov).squeeze()
            # gothic normalises this as reads_j / 2N
            rel_cov /= 2 * total_links

            _map = _map.tocoo().astype(np.float64)
            fast_norm_gothic(_map.row, _map.col, _map.data, rel_cov, total_links, 1., goth_mode)

        else:
            raise ApplicationException('unknown method {}'.format(method))

        return _map

    def _reorder_extent(self, _map):
        """
        Reorder the extent map using current order.

        :return: sparse CSR format permutation of the given map
        """
        _order = self.order.gapless_positions()
        _bins = self.grouping.bins[self.order.mask_vector()]
        _ori = self.order.order['ori'][np.argsort(self.order.order['pos'])]

        # create a permutation matrix
        p = sp.lil_matrix(_map.shape)
        _shuf_bins = _bins[_order]
        for i, oi in enumerate(_order):
            j_off = _bins[:oi].sum()
            i_off = _shuf_bins[:i].sum()
            if _ori[i] > 0:
                for k in range(_bins[oi]):
                    p[i_off+k, j_off+k] = 1
            else:
                # rot90 those with reverse orientation
                _nb = _bins[oi]
                for k in range(_nb):
                    p[i_off+_nb-(k+1), j_off+k] = 1

        # permute the extent_map
        p = p.tocsr()
        return p.dot(_map.tocsr()).dot(p.T)

    def _compress_extent(self, _map):
        """
        Compress the extent map for each sequence that is presently masked. This will eliminate
        all bins which pertain to a given masked sequence.

        :return: a scipy.sparse.coo_matrix pertaining to only the unmasked sequences.
        """
        assert sp.isspmatrix(_map), 'Extent matrix is not a scipy sparse matrix type'
        if not sp.isspmatrix_coo(_map):
            _map = _map.tocoo()

        _order = self.order.order
        _bins = self.grouping.bins

        # build a list of every accepted element.
        # TODO this could be done as below, without the memory requirements of realising all elements
        s = 0
        accept_bins = []
        # accept_index = set(np.where(_mask)[0])
        for i in range(len(_order)):
            # if i in accept_index:
            if _order[i]['mask']:
                accept_bins.extend([j+s for j in range(_bins[i])])
            s += _bins[i]

        _mask = np.zeros(self.grouping.total_bins, dtype=np.bool)
        _mask[np.array(accept_bins)] = True

        _data, _row, _col, _shift = sparse_utils.fast_retained(_map.data, _map.row, _map.col, _map.nnz, _mask)

        return sp.coo_matrix((_data, (_row, _col)), shape=np.array(_map.shape) - _shift[-1])

    def plot_seqnames(self, fname, simple=True, permute=False, **kwargs):
        """
        Plot the contact map, annotating the map with sequence names. WARNING: This can often be too dense
        to be legible when there are many (1000s) of sequences.

        :param fname: output file name
        :param simple: True plot seq map, False plot the extent map
        :param permute: permute the map with the present order
        :param kwargs: additional options passed to plot()
        """
        if permute:
            seq_id_iter = self.order.accepted_positions()
        else:
            seq_id_iter = range(self.order.count_accepted())

        tick_labs = []
        for i in seq_id_iter:
            if self.order.order[i]['ori'] < 0:
                tick_labs.append('- {}'.format(self.seq_info[i].name))
            else:
                tick_labs.append('+ {}'.format(self.seq_info[i].name))

        if simple:
            step = 2 if self.is_tipbased() else 1
            tick_locs = range(2, step*self.order.count_accepted()+step, step)
        else:
            if permute:
                _cbins = np.cumsum(self.grouping.bins[self.order.accepted_positions()])
            else:
                _cbins = np.cumsum(self.grouping.bins[self.order.accepted()])
            tick_locs = _cbins - 0.5

        self.plot(fname, permute=permute, simple=simple, tick_locs=tick_locs, tick_labs=tick_labs, **kwargs)

    def plot(self, fname, simple=False, tick_locs=None, tick_labs=None, norm=True, permute=False, pattern_only=False,
             dpi=180, width=25, height=22, zero_diag=True, alpha=0.01, robust=False, max_image_size=None,
             flatten=False, norm_method=None):
        """
        Plot the contact map. This can either be as a sparse pattern (requiring much less memory but without visual
        cues about intensity), simple sequence or full binned map and normalized or permuted.

        :param fname: output file name
        :param tick_locs: major tick locations (minors take the midpoints)
        :param tick_labs: minor tick labels
        :param simple: if true, sequence only map plotted
        :param norm: normalize intensities by geometric mean of lengths
        :param permute: reorder map to current order
        :param pattern_only: plot only a sparse pattern (much lower memory requirements)
        :param dpi: adjust DPI of output
        :param width: plot width in inches
        :param height: plot height in inches
        :param zero_diag: set bright self-interactions to zero
        :param alpha: log intensities are log (x + alpha)
        :param robust: use seaborn robust dynamic range feature
        :param max_image_size: maximum allowable image size before rescale occurs
        :param flatten: for tip-based, flatten matrix rather than marginalise
        :param norm_method: normalisation method to apply to contact map
        """

        plt.style.use('ggplot')

        fig = plt.figure()
        fig.set_figwidth(width)
        fig.set_figheight(height)
        ax = fig.add_subplot(111)

        if simple or self.bin_size is None:
            if norm_method is None:
                norm_method = 'sites'
            # prepare the map if not already done. This overwrites
            # any current ordering mask beyond the primary acceptance mask
            if self.processed_map is None:
                self.prepare_seq_map(norm=norm, bisto=True, norm_method=norm_method)
            _map = self.get_subspace(permute=permute, marginalise=False if flatten else True, flatten=flatten)
            # amplify values for plotting
            _map *= 10
        else:
            if norm_method is None:
                norm_method = 'length'
            _map = self.get_extent_map(norm=norm, bisto=True, permute=permute, norm_method=norm_method)

        if pattern_only:
            # sparse matrix plot, does not support pixel intensity
            if zero_diag:
                if sp.isspmatrix(_map) and not sp.isspmatrix_lil(_map):
                    _map = _map.tolil()
                _map.setdiag(0)
            ax.spy(_map.tocsr(), markersize=5 if simple else 1)

        else:
            # a dense array plot

            # if too large, reduced it while sparse.
            if max_image_size is not None:
                full_size = _map.shape
                if np.max(full_size) > max_image_size:
                    reduce_factor = int(np.ceil(np.max(full_size) / float(max_image_size)))
                    logger.info('Full {} image reduction factor: {}'.format(full_size, reduce_factor))
                    # downsample the map
                    _map = sparse_utils.downsample(_map, reduce_factor)
                    # ticks adjusted to match
                    tick_locs = np.floor(tick_locs.astype(np.float) / reduce_factor)
                    logger.info('Map has been reduced from {} to {}'.format(full_size, _map.shape))

            _map = _map.toarray()

            if zero_diag:
                logger.debug('Removing diagonal')
                np.fill_diagonal(_map, 0)

            _map = np.log(_map + alpha)

            logger.debug('Making raster image')
            seaborn.heatmap(_map, robust=robust, square=True, linewidths=0, ax=ax, cbar=False)

        if tick_locs is not None:

            plt.tick_params(axis='both', which='both',
                            right=False, left=False, bottom=False, top=False,
                            labelright=False, labelleft=False, labelbottom=False, labeltop=False)

            if tick_labs is not None:
                min_labels = ticker.FixedFormatter(tick_labs)
                ax.tick_params(axis='y', which='minor', left=True, labelleft=True, labelsize=10)

                min_ticks = ticker.FixedLocator(tick_locs[:-1] + 0.5 * np.diff(tick_locs))

                ax.yaxis.set_minor_formatter(min_labels)
                ax.yaxis.set_minor_locator(min_ticks)

            # seaborn will not display the grid, so we make our own.
            ax.hlines(tick_locs, *ax.get_xlim(), color='grey', linewidth=0.2, linestyle='-.')
            ax.vlines(tick_locs, *ax.get_ylim(), color='grey', linewidth=0.2, linestyle='-.')

        logger.debug('Saving plot')
        fig.tight_layout()
        plt.savefig(fname, dpi=dpi)
        plt.close(fig)

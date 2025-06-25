import logging
import os
from collections import OrderedDict, defaultdict, namedtuple
from functools import partial
from typing import Any, Callable, Dict, Hashable, Iterator, List, Optional, Self, Tuple

import Bio.SeqIO as SeqIO
import Bio.SeqUtils as SeqUtils
import matplotlib
import numba as nb
import numpy as np
import numpy.typing as npt
import pysam
import scipy.sparse as sp
import sparse
import tqdm
from numpy import signedinteger
from scipy.stats import binom, poisson
from statsmodels.stats.multitest import multipletests

from .. import ordering
from ..exceptions import ApplicationException, NoneAcceptedException, ParsingError, TooFewException, ZeroLengthException
from ..io_utils import io_utils
from ..linalg import sparse_utils
from ..misc_utils import package_path
from ..seq_utils import SiteCounter, count_bam_reads, count_fasta_sequences, revcomp
from ..types import SparseMatrix

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn

# logging setup
logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)

SeqInfo = namedtuple('SeqInfo', ['offset', 'refid', 'name', 'length', 'sites', 'gc'])


"""
Basic Mean functions
"""


@nb.jit(nopython=True)
def geometric_mean(x: float, y: float) -> float:
    """
    Calculate the geometric mean of two numbers.

    The geometric mean is the square root of the product of two numbers and is
    used to calculate a central tendency in situations involving ratios.

    :param x: The first number for which the geometric mean is calculated.
    :param y: The second number for which the geometric mean is calculated.
    :return: The geometric mean of the two input numbers.
    :rtype: float
    """
    return (x * y)**0.5


@nb.jit(nopython=True)
def harmonic_mean(x: float, y: float) -> float:
    """
    Calculate the harmonic mean of two numbers.

    The harmonic mean is a measure of the average of two numbers, computed
    as twice the product of the numbers divided by their sum. It is used
    in various contexts to determine rates or averages in a way that gives
    equal weight to the contributions of the values.

    :param x: First number for the harmonic mean calculation.
    :param y: Second number for the harmonic mean calculation.
    :return: The harmonic mean of the two input numbers.
    """
    return 2 * x * y / (x + y)


@nb.jit(nopython=True)
def arithmetic_mean(x: float, y: float) -> float:
    """
    Calculate the arithmetic mean of two numbers.

    This function computes the arithmetic mean of two given floating-point numbers
    using the provided formula. It uses the `numba` JIT decorator for performance
    optimization in numeric computations.

    :param x: The first floating-point number.
    :param y: The second floating-point number.
    :return: The arithmetic mean of the two input numbers.
    """
    return 0.5 * (x + y)


def mean_selector(name: str) -> Callable:
    """
    Selects and returns a mean calculation function by name.

    The available mean types are 'geometric', 'harmonic', and 'arithmetic'.
    If the given mean type is not recognized, an exception is raised.

    :param name: Name of the mean type. Possible values are 'geometric',
        'harmonic', and 'arithmetic'.
    :return: A callable function corresponding to the selected mean type.
    :raises RuntimeError: If the specified mean type is not supported.
    """
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
def find_containing_bin(group_sites: npt.NDArray[int], x: int) -> int:
    """
    Find the nearest site from a given position on a contig.

    :param group_sites:
    :param x: Query position.
    :return: Tuple of site and group number.
    """
    ix = np.searchsorted(group_sites[:, 0], x)
    if ix == len(group_sites):
        return group_sites[-1, 1].item()
    return group_sites[ix, 1].item()


@nb.jit(nopython=True)
def fast_norm_tipbased_bylength(coords: np.ndarray,
                                data: np.ndarray,
                                tip_lengths: np.ndarray,
                                tip_size: int) -> None:
    """
    In-place normalization of the sparse 4D matrix used in tip-based maps.

    As tip-based normalization is slow for large matrices, the inner-loop has been
    moved to a Numba method.

    :param coords: The COO matrix coordinate member variable (4xN array).
    :param data:  The COO matrix data member variable (1xN array).
    :param tip_lengths: Per-element min(sequence_length, tip_size).
    :param tip_size: Tip size used in the map.
    """
    for ii in range(coords.shape[1]):
        i, j = coords[:2, ii]
        data[ii] *= tip_size**2 / (tip_lengths[i] * tip_lengths[j])


@nb.jit(nopython=True)
def fast_norm_tipbased_bysite(coords: np.ndarray,
                              data: np.ndarray,
                              sites: np.ndarray) -> None:
    """
    In-place normalization of the sparse 4D matrix used in tip-based maps.

    As tip-based normalization is slow for large matrices, the inner-loop has been
    moved to a Numba method.

    :param coords: The COO matrix coordinate member variable (4xN array).
    :param data:  The COO matrix data member variable (1xN array).
    :param sites: Per-element min(sequence_length, tip_size).
    """
    for n in range(coords.shape[1]):
        _i, _j, _k, _l = coords[:, n]
        data[n] *= 1.0 / (sites[_i, _k] * sites[_j, _l])


def fast_norm_gothic(rows: np.ndarray,
                     cols: np.ndarray,
                     data: np.ndarray,
                     rel_cov: np.ndarray,
                     total_obs: int,
                     frac_random: float,
                     mode: str='binomial') -> None:
    """
    Inplace application of GOTHiC link significance. Here, modifications have been
    made to approximate the Binomial using the Poisson for large N and small p.
    This allows the easy implementation of a fast Poisson CDF calculator.

    :param rows: COO matrix rows.
    :param cols: COO matrix cols.
    :param data: COO matrix data values to modify.
    :param rel_cov: Relative coverage data.
    :param total_obs: Total number of links (pairs) in the map.
    :param frac_random: The fraction of read-pairs in the Hi-C library arising from spurious ligation.
    :param mode: 'Binomial': -log binnom signif, 'poisson': -log poisson signif, 'effect': effect-size.
    """
    # we will cap the smallest values to tiny, preventing log errors
    tiny = np.finfo(data.dtype).tiny

    # estimate the probability of observing a link between two loci is proportional
    # to the relative coverage of the loci and the fraction of random ligation events
    pij = 2 * rel_cov[rows] * rel_cov[cols] * frac_random

    if mode == 'binomial':
        pr = binom.sf(data, total_obs, pij)
        # avoid float precision errors for extremely small values
        data[:] = np.where(pr < tiny, tiny, pr)

    # for large N and small probability, the Poisson and Binomial are very similar
    elif mode == 'poisson':
        pr = poisson.sf(data, total_obs * pij)
        # avoid float precision errors for extremely small values
        data[:] = np.where(pr < tiny, tiny, pr)

    else:
        raise ApplicationException('unsupported mode [{}]'.format(mode))


@nb.jit(nopython=True)
def count_bin_sites(coords: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    For a set of genomic coordinates, representing cut-site locations, and a set of borders
    in the same coordinate space, count the number of sites in each bin.

    :param coords: Genomic coords of cut-sites in a sequence.
    :param bins: Borders of the bins for a sequence.
    :return: 1d-array of counts for each bin.
    """
    return np.array([((coords >= bi[0]) & (coords < bi[1])).sum() for bi in bins], dtype='int')


@nb.jit(nopython=True)
def fast_norm_bysite(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, sites: np.ndarray) -> None:
    """
    In-place normalization of the scipy.coo_matrix for full sequences.

    :param rows: The COO matrix coordinate member variable (4xN array).
    :param cols: The COO matrix coordinate member variable (4xN array).
    :param data:  The COO matrix data member variable (1xN array).
    :param sites: Per-element min(sequence_length, tip_size).
    """
    for n in range(data.shape[0]):
        i = rows[n]
        j = cols[n]
        data[n] *= 1.0 / (sites[i] * sites[j])


@nb.jit(nopython=True, parallel=True)
def fast_length_norm(row: np.ndarray,
                     col: np.ndarray,
                     data: np.ndarray,
                     nnz: float,
                     len_lookup: np.ndarray,
                     mean_func: Callable) -> None:
    """
    Normalize an extent map by contig length. This method is intended
    to be used internally on the members of a scipy.sparse COO matrix.

    :param data: Coo data member.
    :param row:  Coo row member.
    :param col:  Coo col member.
    :param nnz:  Coo nnz attribute.
    :param len_lookup: Contig length lookup for any index of extent map.
    :param mean_func: Mean function to apply to length_i and length_j.
    :return: The normalized data array only.
    """

    for n in nb.prange(nnz):
        w_ij = 1e-3 * mean_func(len_lookup[row[n]], len_lookup[col[n]])
        data[n] /= w_ij

@nb.jit(nopython=True)
def bin_indices(i: int, j: int, cumu_bins: np.ndarray) -> Tuple[signedinteger, signedinteger]:
    """
    Convert an index pair from the extent-map to an index pair on the sequence map.

    :param i: A row index from the extent map.
    :param j: A column index from the extent map.
    :param cumu_bins: The extent map's cumulative bin borders.
    :return: (bi, bj) the corresponding row and column indices on the sequence map.
    """
    bi = np.searchsorted(cumu_bins, i, side='right')
    bj = np.searchsorted(cumu_bins, j, side='right')
    return bi, bj


class ExtentGrouping(object):
    """
    Class to group sequence data into bins of approximately equal size.

    This class is designed to process a list of sequence information and divide
    each sequence into a specified number of bins, ensuring that the bins are
    approximately equal in size. It manages the binning process by storing the
    bin edges, mapping between bins and their corresponding sequence indices,
    borders for each sequence, and bin centers.

    The class handles specific cases where a sequence length is zero by raising
    an exception, and ensures that sequences of non-integer lengths are divided
    into bins by contracting or expanding the bin sizes slightly.

    :ivar bins: Array representing the number of bins for each sequence.
    :ivar bin_size: The size of each bin in base pairs.
    :ivar map: List storing mappings between bin indices and coordinate pairs.
    :ivar borders: List storing the start and end indices for bins of each sequence.
    :ivar centers: Array of bin center offsets for each sequence with respect to the
        sequence's midpoint.
    :ivar total_bins: Total number of bins across all sequences.
    """

    def __init__(self, seq_info: list, bin_size: int) -> None:
        self.bin_size = bin_size
        self.map : List[npt.NDArray[int]] = []
        self.borders : List[npt.NDArray[int]] = []
        self.centers : List[npt.NDArray[float]] = []
        self.total_bins : int = 0

        _bins : List[int] = []
        for n, seq in tqdm.tqdm(enumerate(seq_info), total=len(seq_info), desc='Making bins'):

            if seq.length == 0:
                raise ZeroLengthException(seq.id)

            # integer bin estimation
            num_bins = seq.length // bin_size
            if num_bins == 0:
                num_bins += 1
            # handle non-integer discrepancy by contracting/expanding all bins equally
            # the threshold between contract/expand being half a bin size
            if seq.length % bin_size != 0 and seq.length / float(bin_size) - num_bins >= 0.5:
                num_bins += 1

            edges = np.linspace(0, seq.length, num_bins+1, endpoint=True, dtype=np.int64)

            _bins.append(num_bins)

            # Per reference coordinate pairs (bin_edge, map_index)
            first_bin = self.total_bins
            last_bin = first_bin + num_bins
            self.map.append(np.vstack((edges[1:], np.arange(first_bin, last_bin)), dtype=np.int64).T)
            self.borders.append(np.array([first_bin, last_bin], dtype=np.int64))

            self.total_bins += num_bins

            c_nk = (edges[:-1] + 0.5*(edges[1] - edges[0]) - 0.5*seq.length).astype(np.float64)
            self.centers.append(c_nk.reshape((1, len(c_nk))))

        self.bins = np.array(_bins, dtype=np.int64)

    def calc_borders(self, _seq_id: int) -> npt.NDArray[int]:
        """
        Calculate the bin borders in genomic coordinates for a given sequence.

        :param _seq_id: The sequence index to consider.
        :return: 2d list of [[bin0_begin, bin0_end], [bin1_begin, bin1_end], ...].
        """
        coord_borders = np.hstack([[0], self.map[_seq_id][:, 0]])
        return np.array([[coord_borders[i], coord_borders[i+1]] for i in range(len(coord_borders)-1)], dtype='int')

    def get_bin_lengths(self) -> npt.NDArray[int]:
        """
        Compute the width of each bin. The bin widths vary slightly, as the grouping algorithm
        attempts to disperse the extra extent across all bins.

        :return: Bin lengths.
        """
        bin_len = np.zeros(self.total_bins, dtype=np.int64)
        n = 0
        for i in range(len(self.map)):
            bin_len[n] = self.map[i][0, 0]
            n += 1
            for j in range(1, len(self.map[i])):
                bin_len[n] = self.map[i][j, 0] - self.map[i][j-1, 0]
                n += 1
        return bin_len


class SeqOrder(object):
    """
    Provides functionalities for sequence ordering, orientation, and masking, designed to manage
    surrogate IDs for sequences, enable operations on their positions, and map between dense and sparse
    indices when sequences are masked or unmasked. The class is optimized to maintain and update the
    state efficiently, ensuring sequences maintain their order and mask state.

    The sequence orientation supports forward (1) and reverse (-1). Masking indicates whether a sequence
    is included for operations (True) or excluded (False). The class uses structured NumPy arrays for
    efficient representation and manipulation of sequence metadata.

    :ivar FORWARD: Indicates forward orientation of the sequence.
    :type FORWARD: int

    :ivar REVERSE: Indicates reverse orientation of the sequence.
    :type REVERSE: int

    :ivar ACCEPTED: Signifies that a sequence is included in operations.
    :type ACCEPTED: bool

    :ivar EXCLUDED: Signifies that a sequence is excluded from operations.
    :type EXCLUDED: bool

    :ivar STRUCT_TYPE: Data type for storing sequence positional and state information.
    :type STRUCT_TYPE: np.dtype

    :ivar INDEX_TYPE: Data type for representing indices with orientation.
    :type INDEX_TYPE: np.dtype

    :ivar _positions: Cached sorted positional representation of sequences. Updated when masking or positional
        states change.
    :type _positions: None | npt.NDArray[int]

    :ivar order: Structured NumPy array containing sequence information, including surrogate ID, orientation,
        mask state, and sequence length.
    :type order: npt.NDArray
    """
    FORWARD = 1
    REVERSE = -1

    ACCEPTED = True
    EXCLUDED = False

    STRUCT_TYPE = np.dtype([('pos', np.int32), ('ori', np.int8), ('mask', bool), ('length', np.int32)])
    INDEX_TYPE = np.dtype([('index', np.int32), ('ori', np.int8)])

    def __init__(self, seq_info: list) -> None:
        """
        The initial order is determined by the order of supplied sequence information dictionary. Sequences
        are given surrogate ids using consecutive integers. Member functions expect surrogate ids
        not original names.

        The class also retains orientation and masking state. Orientation defines whether a sequence
        should be in its original direction (as read in) (1) or reverse complemented (-1).

        Masking state defines whether an input sequence shall be excluded from further consideration.
        (accepted=1, excluded=0)

        :param seq_info: Sequence information dictionary.
        """
        _ord = np.arange(len(seq_info), dtype=np.int32)
        self.order: npt.NDArray[SeqOrder.STRUCT_TYPE] = np.array(
            [(_ord[i], SeqOrder.FORWARD, SeqOrder.ACCEPTED, seq_info[i].length) for i in range(len(_ord))],
            dtype=SeqOrder.STRUCT_TYPE)

        self._update_positions()

    @staticmethod
    def asindex(_ord: npt.NDArray | list) -> npt.NDArray:
        """
        Convert a simple list or ndarray of indices, to an INDEX_TYPE array with default forward orientation.

        :param _ord: list/ndarray of indices.
        :return: INDEX_TYPE array.
        """
        assert isinstance(_ord, (list, np.ndarray)), 'input must be a list or ndarray'
        return np.fromiter(zip(_ord, np.ones_like(_ord, dtype=bool)), dtype=SeqOrder.INDEX_TYPE)

    def _update_positions(self) -> None:
        """
        An optimization, whenever the positional state changes, this method must be called to
        maintain the current state in a separate array. This avoids unnecessary recalculation
        overhead.
        """
        # Masked sequences last, then by current position.
        sorted_indices = np.lexsort([self.order['pos'], ~self.order['mask']])
        for n, i in enumerate(sorted_indices):
            self.order[i]['pos'] = n
        self._positions = np.argsort(self.order['pos'])

    def remap_gapless(self, gapless_indices: npt.NDArray | list) -> npt.NDArray:
        """
        Recover the original, potentially sparse (gapped) indices from a dense (gapless) set
        of indices. Gaps originate from sequences being masked in the order. External tools
        often expect and return dense indices. When submitting changes to the current order
        state, it is important to first apply this method and reintroduce any gaps.

        Both a list/array of indices or an INDEX_TYPE array can be passed.

        :param gapless_indices: Dense list of indices or a ndarray of type INDEX_TYPE.
        :return: Remapped indices with gaps (of a similar type to input).
        """
        # Not as yet verified, but this method is being replaced by the 50x faster numpy
        # alternative below. The slowless shows for large problems and repeated calls.
        # We ~could~ go further and maintain the shift array, but this will require
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
        # handle a plain collection
        else:
            for oi in gapless_indices:
                remapped.append(oi + shift[oi])
            remapped = np.array(remapped)

        return remapped

    def accepted_positions(self, copy: bool=True) -> npt.NDArray:
        """
        The current positional order of only those sequences which have not been excluded by the mask.
        :param copy: Return a copy.

        Note: see usage of all_positions() for warning about when positional data must be refreshed.

        :return: All accepted positons, in order of index.
        """
        return self.all_positions(copy=copy)[:self.count_accepted()]

    def all_positions(self, copy: bool=True) -> npt.NDArray:
        """
        The current positional order of all sequences. Internal logic relegates masked sequences to always come
        last and ascending surrogate id order.

        Note: positions are updated when ContactMap.__init__(), .mask(), .set_mask_only(), .set_order_and_orientation()
        and .shuffle() are called. Users should take care when copying and then using outdated positional data.

        :param copy: Return a copy of the positions.
        :return: All positions in order of index, masked or not.
        """
        if copy:
            _p = self._positions.copy()
        else:
            _p = self._positions
        return _p

    @staticmethod
    def double_order(_ord: npt.NDArray) -> npt.NDArray:
        """
        For doublet maps, the stored order must be re-expanded to reference the larger (2x) map.

        :param _ord:
        :return: Expanded order.
        """
        return np.array([[2*oi, 2*oi+1] for oi in _ord]).ravel()

    def gapless_positions(self) -> npt.NDArray:
        """
        A dense index range representing the current positional order without masked sequences. Therefore,
        the returned array does not contain surrogate ids, but rather the relative positions of unmasked
        sequences, when all masked sequences have been discarded.

        :return: A dense index range of positional order, once all masked sequences have been discarded.
        """
        # accumulated shift from gaps
        gap_shift = np.cumsum(~self.order['mask'])
        # just unmasked sequences
        _p = np.argsort(self.order['pos'])
        _p = _p[:self.count_accepted()]
        # removing gaps leads to a dense range of indices
        _p -= gap_shift[_p]
        return _p

    def set_mask_only(self, _mask: npt.NDArray) -> None:
        """
        Set the mask state of all sequences, where indices in the mask map to
        sequence surrogate ids.

        :param _mask: Mask array or list, boolean or 0/1 valued.
        """
        _mask = np.asarray(_mask, dtype=bool)
        assert len(_mask) == len(self.order), 'supplied mask must be the same length as existing order'
        assert np.all((_mask == SeqOrder.ACCEPTED) | (_mask == SeqOrder.EXCLUDED)), \
            'new mask must be {} or {}'.format(SeqOrder.ACCEPTED, SeqOrder.EXCLUDED)

        # assign mask
        self.order['mask'] = _mask
        self._update_positions()

    def set_order_only(self, _ord: npt.NDArray, implicit_excl: bool=False) -> None:
        """
        Convenience method to set the order using a list or 1D ndarray. Orientations will
        be assumed as all forward (+1).

        :param _ord: A list or ndarray of surrogate ids.
        :param implicit_excl: Implicitly extend the order to include unmentioned excluded sequences.
        """
        assert isinstance(_ord, (list, np.ndarray)), 'Wrong type supplied, order must be a list or ndarray'
        if isinstance(_ord, np.ndarray):
            _ord = np.ravel(_ord)
            assert np.ndim(_ord) == 1, 'orders as numpy arrays must be 1-dimensional'
        # augment the order to include default orientations
        _ord = SeqOrder.asindex(_ord)
        self.set_order_and_orientation(_ord, implicit_excl=implicit_excl)

    def set_order_and_orientation(self, _ord: npt.NDArray, implicit_excl: bool=False) -> None:
        """
        Set only the order, while ignoring orientation. An ordering is defined
        as a 1D array of the structured type INDEX_TYPE, where elements are the
        position and orientation of each indexed sequence.

        NOTE: This definition can be the opposite of what is returned by some
        ordering methods, and np.argsort(_v) should inverse the relation.

        NOTE: If the order includes only active sequences, setting implicit_excl=True
        the method will implicitly assume unmentioned ids are those currently
        masked. An exception is raised if a masked sequence is included in the order.

        :param _ord: 1d ordering.
        :param implicit_excl: Implicitly extend the order to include unmentioned excluded sequences.
        """
        assert _ord.dtype == SeqOrder.INDEX_TYPE, 'Wrong type supplied, _ord should be of INDEX_TYPE'

        if len(_ord) < len(self.order):
            # some sanity checks
            assert implicit_excl, 'Use implicit_excl=True for automatic handling ' \
                                  'of orders only mentioning accepted sequences'
            assert len(_ord) == len(set(_ord['index'])), 'new order must not contain duplicate indices'
            assert set(_ord['index']) == set(self.accepted()), 'new order must mention all ' \
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
            _mask = np.zeros_like(self.mask_vector(), dtype=bool)
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

    def accepted_order(self) -> npt.NDArray:
        """
        :return: an INDEX_TYPE array of the order and orientation of the currently accepted sequences.
        """
        idx = np.where(self.order['mask'])
        ori = np.ones(self.count_accepted(), dtype=np.int64)
        return np.fromiter(zip(idx, ori), dtype=SeqOrder.INDEX_TYPE)

    def mask_vector(self) -> npt.NDArray:
        """
        :return: the current mask vector
        """
        return self.order['mask']

    def mask(self, _id: int) -> None:
        """
        Mask an individual sequence by its surrogate id.

        :param _id: The surrogate id of a sequence.
        """
        self.order[_id]['mask'] = False
        self._update_positions()

    def new_mask(self, default: bool=True) -> npt.NDArray:
        """
        Create a new mask, where all sequences begin M.
        :return:
        """
        _mask = np.empty_like(self.mask_vector(), dtype=bool)
        _mask[:] = default
        return _mask

    def count_accepted(self) -> int:
        """
        :return: the current number of accepted (unmasked) sequences.
        """
        return self.order['mask'].sum()

    def count_excluded(self) -> int:
        """
        :return: the current number of excluded (masked) sequences.
        """
        return len(self.order) - self.count_accepted()

    def accepted(self) -> npt.NDArray:
        """
        :return: the list surrogate ids for currently accepted sequences.
        """
        return np.where(self.order['mask'])[0]

    def excluded(self) -> npt.NDArray:
        """
        :return: the list surrogate ids for currently excluded sequences.
        """
        return np.where(~self.order['mask'])[0]

    def flip(self, _id: int) -> None:
        """
        Flip the orientation of the sequence.

        :param _id: The surrogate id of a sequence.
        """
        self.order[_id]['ori'] *= -1

    def lengths(self, exclude_masked: bool=False) -> npt.NDArray:
        """
        Retrieve the lengths of all sequences.

        :param exclude_masked: When True include only the unmasked sequences.
        :return: The lengths of sequences.
        """
        if exclude_masked:
            return self.order['length'][self.order['mask']]
        return self.order['length']

    def shuffle(self) -> None:
        """
        Randomize order.
        """
        np.random.shuffle(self.order['pos'])
        self._update_positions()

    def before(self, a: int, b: int) -> bool:
        """
        Test if A comes before another sequence B in the current order.

        :param a: Surrogate id of sequence A.
        :param b: Surrogate id of sequence B.
        :return: True if A comes before B.
        """
        assert a != b, 'Surrogate ids must be different'
        return (self.order['pos'][a] < self.order['pos'][b]).item()

    def intervening(self, a: int, b: int) -> int:
        """
        For the current order, calculate the length of intervening
        sequences between sequence a and sequence b.

        :param a: Surrogate id of sequence A.
        :param b: Surrogate id of sequence B.
        :return: total length of sequences between A and B.
        """
        assert a != b, 'Surrogate ids must be different'

        pa = self.order['pos'][a]
        pb = self.order['pos'][b]
        if pa > pb:
            pa, pb = pb, pa
        inter_ix = self._positions[pa+1:pb]
        return np.sum(self.order['length'][inter_ix])


class ContactMap(object):

    def append_map(self, other: Self) -> None:
        if not isinstance(other, ContactMap):
            raise ValueError('ContactMap value is required')

        assert not self.has_extent_map() and not other.has_extent_map(), \
            'Appending contact maps with extent mapping not implemented'

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

    def __init__(self,
                 bam_file: str,
                 enzymes: list,
                 seq_file: str,
                 min_separation: int,
                 min_mapq: int=0,
                 min_len: int=0,
                 min_sig: int=1,
                 min_extent: int=0,
                 min_size: int=0,
                 max_edist: int=2,
                 min_alen: int=25,
                 max_fold: Optional[float]=None,
                 random_seed: Optional[int]=None,
                 bin_size: Optional[int]=None,
                 tip_size: Optional[int]=None,
                 no_duplicates: bool=True,
                 precount: bool=False,
                 threads: int=4) -> None:

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
        self.seq_sites = []
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

        # build a dictionary of features/details for each reference sequence
        fasta_info = self.initialise_fasta_info()

        # now inspect the BAM header
        with pysam.AlignmentFile(bam_file, 'rb', threads=threads) as bam:

            # test that BAM file is the correct sort order
            header = bam.header.to_dict() # pedantically obtain the dictionary form to make typing happy.
            if 'SO' not in header['HD'] or header['HD']['SO'] != 'queryname':
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
                    fa_info = fasta_info[rname]
                except KeyError:
                    logger.info('From BAM, reference {} was not present in supplied fasta'.format(rname))
                    ref_count['seq_missing'] += 1
                    continue

                assert fa_info['length'] == rlen, \
                    'BAM and FASTA lengths do not agree for reference {}: {} != {}'.format(
                        rname, fa_info['length'], rlen)

                self.seq_info.append(SeqInfo(offset, n, rname, rlen, fa_info['sites'], fa_info['gc']))
                self.seq_sites.append(fa_info['coords'])

                offset += rlen

            # total extent covered
            self.total_len = offset
            self.total_seq = len(self.seq_info)

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
                self.total_reads = count_bam_reads(bam_file, threads)
                logger.info('BAM file contains {0} alignments'.format(self.total_reads))
            else:
                logger.info('Skipping pre-count of BAM file, no ETA will be offered')

            # initialize the order
            self.order = SeqOrder(self.seq_info)

            # accumulate
            self._bin_map(bam)

            # create an initial acceptance mask
            self.set_primary_acceptance_mask()

    def initialise_fasta_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan the reference fasta file and extract information about each sequence.

        :return: Dict of dicts, keyed by sequence name.
        """
        fasta_info = {}
        with io_utils.open_input(self.seq_file, 'rt') as multi_fasta:
            # get an estimate of sequences for progress
            fasta_count = count_fasta_sequences(self.seq_file)
            for n_seq, seqrec in tqdm.tqdm(enumerate(SeqIO.parse(multi_fasta, 'fasta')),
                                           total=fasta_count, desc='Analyzing reference sequences'):
                cs_coords = np.array(self.site_counter.find_sites(seqrec.seq))
                fasta_info[seqrec.id] = {'sites': len(cs_coords),
                                         'length': len(seqrec),
                                         # TODO, biopython has made an interface change for GC
                                         'gc': SeqUtils.GC(seqrec.seq),
                                         'coords': cs_coords}

            logger.info(f'From FASTA, {n_seq} of {len(fasta_info)} sequences were accepted')

        return fasta_info

    def refresh_seqsites(self, fasta_path: Optional[str]=None) -> None:
        """
        Refresh the list of cut-site coordinates for each sequence.
        This can be used with older contact maps prior to the introduction of the
        class member "seq_sites".

        :param fasta_path: Path to the reference fasta file.
        """
        # assume that the path to the reference fasta is still correct
        if fasta_path is not None:
            self.seq_file = fasta_path
        fasta_info = self.initialise_fasta_info()
        self.seq_sites = [fasta_info[si.name]['coords'] for si in self.seq_info]

    def _bin_map(self, bam: pysam.AlignmentFile) -> None:
        """
        Accumulate read-pair observations from the supplied BAM file.
        Maps are initialized here. Logical control is achieved through initialization of the
        ContactMap instance, rather than supplying arguments to this function.

        :param bam: This instance's open bam file.
        """

        def _strict_acceptance(r: pysam.AlignedSegment) -> bool:
            """
            Carefully parse the read mapping record for suitability. This tests mapping quality,
            cigar existence, alignment length, edit distance, first read position, and the condition
            that ended the alignment. Reads which terminate before their 3p end is reached, must either
            exceed the reference extent or terminate at the expected enzymatic cut-site.
            :param r: The read to test.
            :return: True - the read mapping is accepted, False - it is rejected.
            """

            if r.mapping_quality < _min_mapq:
                counts['mapq'] += 1
                return False

            if r.cigarstring is None:
                counts['cigar'] += 1
                return False

            if r.query_alignment_length < _min_alen:
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

            # accept full-length alignments that exceed the minimum
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

            # accept alignments which terminate at cut-site remnant
            _match = _endswith_vestigial(aln_seq)
            if _match is None:
                counts['cs_end'] += 1
                return False

            return True

        def _next_informative(_bam_iter: Iterator, _pbar: tqdm.tqdm) -> pysam.AlignedSegment:
            while True:
                r = next(_bam_iter)
                _pbar.update()
                if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate:
                    continue
                break
            return r

        def _on_tip_withlocs(p1: int, p2: int, l1: int, l2: int, _tip_size: int) -> Tuple[bool, npt.NDArray]:
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

        def _always_true(*args: Any) -> Tuple[bool, int]:
            return True, 1

        # lookup table for reference lengths
        _refid_to_reflen = self.refid_to_reflen
        # set read acceptance method
        _accept_read = _strict_acceptance
        # prepare method which checks that alignments strings end in a vestigial cut-site.
        _endswith_vestigial = self.site_counter.get_vestigial_end_searcher()

        # set tip acceptance method
        _on_tip = _always_true if not self.is_tipbased() else _on_tip_withlocs

        # initialize a sparse matrix for accumulating the map
        if not self.is_tipbased():
            # just a basic NxN sparse array for normal whole-sequence binning
            _seq_map = sparse_utils.Sparse2DAccumulator(self.total_seq)
        else:
            # each tip is tracked separately, resulting in the single count becoming a 2x2 interaction matrix.
            # therefore, the tensor has dimension NxNx2x2
            _seq_map = sparse_utils.Sparse4DAccumulator(self.total_seq)

        # if binning also requested, initialize another sparse matrix
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
                    # assuming identical map_id implies technical duplication we count it only once
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
                    b1 = find_containing_bin(_grouping_map[ix1], r1pos)
                    b2 = find_containing_bin(_grouping_map[ix2], r2pos)

                    # maintain half-matrix
                    if b1 > b2:
                        b1, b2 = b2, b1

                    # tally all mapped reads for the binned map, not just those considered in tips
                    _extent_map[b1, b2] += 1

                # for seq-map, we may reject reads outside a defined tip region
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
        # a truncated histogram covering observed duplicates between 0 and 10+ times
        if self.no_duplicates:
            map_count = np.bincount(list(pair_store.values()), minlength=11)
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
    def get_fields() -> Tuple[str, ...]:
        """
        :return: the list of fields used in seq_info dict.
        """
        return SeqInfo._fields

    def make_reverse_index(self, field_name: str) -> Dict[Hashable, int]:
        """
        Make a reverse look-up (dict) from the chosen field in seq_info to the internal index value
        of the given sequence. Non-unique fields will raise an exception.

        :param field_name: The seq_info field to use as the reverse.
        :return: Internal array index of the sequence.
        """
        rev_idx = {}
        for n, seq in enumerate(self.seq_info):
            fv = getattr(seq, field_name)
            if fv in rev_idx:
                raise RuntimeError('field contains non-unique entries, a 1-1 mapping cannot be made')
            rev_idx[fv] = n
        return rev_idx

    def map_weight(self) -> int:
        """
        :return: the total map weight (sum ij)
        """
        return self.seq_map.sum()

    def is_empty(self) -> bool:
        """
        :return: True if the map has zero weight
        """
        return self.map_weight() == 0

    def is_tipbased(self) -> bool:
        """
        :return: True if the seq_map is a tip-based 4D tensor
        """
        return self.tip_size is not None

    def has_extent_map(self) -> bool:
        """
        :return: True if the map has an extent-based map
        """
        return self.extent_map is not None

    def find_order(self,
                   _map: SparseMatrix,
                   seed: int,
                   inverse_method: str='inverse',
                   runs: int=5,
                   work_dir: str='.') -> npt.NDArray:
        """
        Using LKH TSP solver, find the best ordering of the sequence map in terms of proximity ligation counts.
        Here, it is assumed that sequence proximity can be inferred from the number of observed trans read-pairs,
        where an inverse relationship exists.

        :param _map: The seq map to analyze.
        :param seed: A random seed.
        :param inverse_method: The chosen inverse method for converting count (similarity) to distance.
        :param runs: Number of individual runs of lkh to perform.
        :param work_dir: Working directory.
        :return: The surrogate ids in optimal order.
        """

        # a minimum of three sequences is required to run LKH
        if _map.shape[0] < 3:
            raise TooFewException(_map.shape[0], 'find_order')

        # we'll supply a partially initialized distance function
        dist_func = partial(ordering.similarity_to_distance, method=inverse_method, alpha=1.2, beta=1)

        with open(os.path.join(work_dir, 'lkh.log'), 'w+') as stdout:

            control_base_name = os.path.join(work_dir, 'lkh_run')

            if self.is_tipbased():

                lkh_o = ordering.lkh_order(_map, control_base_name, lkh_exe=package_path('external', 'LKH'),
                                           precision=1, seed=seed, runs=runs, pop_size=50, dist_func=dist_func,
                                           special=False, stdout=stdout,
                                           fixed_edges=[(i, i+1) for i in range(1, _map.shape[0], 2)])

                # To solve this with TSP, doublet tips use a graph transformation, where each node comes a pair. Pairs
                # possess fixed interconnecting edges which must be included in any solution tour.
                # E.g., node 0 -> (0,1) or node 1 -> (2,3). The fixed paths are undirected, depending on which direction
                # is traversed, defines the orientation of the sequence.

                # 1.  pair adjacent nodes by reshape the 1D array into a two-column array of half the length
                lkh_o = lkh_o.reshape(lkh_o.shape[0] // 2, 2)
                # 2. convert to surrogate ids and infer orientation from paths taken through doublets.
                #   0->1 forward (+1): 1->0 reverse (-1).
                lkh_o = np.fromiter(((oi[0] // 2, oi[1]-oi[0]) for oi in lkh_o), dtype=SeqOrder.INDEX_TYPE)

            else:

                lkh_o = ordering.lkh_order(_map, control_base_name, lkh_exe=package_path('external', 'LKH'),
                                           precision=1, seed=seed, runs=runs, pop_size=50, dist_func=dist_func,
                                           special=False, stdout=stdout)

                # for singlet tours, no orientation can be inferred.
                lkh_o = np.fromiter(((oi, 1) for oi in lkh_o), dtype=SeqOrder.INDEX_TYPE)

        # lkh ordering references the supplied matrix indices, not the surrogate ids.
        # we must map this consecutive set to the contact map indices.
        lkh_o = self.order.remap_gapless(lkh_o)

        return lkh_o

    def get_primary_acceptance_mask(self) -> npt.NDArray:
        assert self.primary_acceptance_mask is not None, 'Primary acceptance mask has not be initialized'
        return self.primary_acceptance_mask.copy()

    def set_primary_acceptance_mask(self,
                                    min_len: Optional[int]=None,
                                    min_sig: Optional[int]=None,
                                    max_fold: Optional[int]=None,
                                    update: bool=False) -> npt.NDArray:
        """
        Determine and set the filter mask using the specified constraints across the entire
        contact map. The mask is True when a sequence is considered acceptable wrt to the
        constraints. The mask is also returned by the function for convenience.

        :param min_len: Override instance value for minimum sequence length.
        :param min_sig: Override instance value for minimum off-diagonal signal (counts).
        :param max_fold: Maximum locally measured fold-coverage to permit.
        :param update: Replace the current primary mask if it exists.
        :return: An acceptance mask over the entire contact map.
        """
        assert max_fold is None, 'Filtering on max_fold is currently disabled'

        # If any parameter-based criterion is not set, then use instance
        # member values set at instantiation time.
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

        acceptance_mask = np.ones(self.total_seq, dtype=bool)

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

    def prepare_seq_map(self,
                        norm: bool=True,
                        bisto: bool=False,
                        mean_type: str='geometric',
                        norm_method: str='sites',
                        from_extent: bool=False,
                        fdr_alpha: float=0.05) -> None:
        """
        Prepare the sequence map (seq_map) by application of various filters and normalizations.

        :param norm: Normalization by sequence lengths.
        :param bisto: Make the output matrix bistochastic.
        :param mean_type: When performing normalization, use "geometric, harmonic or arithmetic" mean.
        :param norm_method: Normalization method to apply to contact map.
        :param from_extent: When normalizing, condense the normalized extent map rather than act on the sequence map.
        :param fdr_alpha: FDR alpha used in gothic normalization.
        """

        _mask = self.get_primary_acceptance_mask()

        self.order.set_mask_only(_mask)
        if self.order.count_accepted() < 1:
            raise NoneAcceptedException()

        _map = self.seq_map.astype(np.float64)
        logger.info('Full sequence map dimensions: {}, {} nnz'.format(_map.shape, _map.nnz))

        if norm:
            if from_extent:
                logger.debug('Using extent map as a basis for normalisation')
                assert not self.is_tipbased(), 'Condensing tip based maps from extent is not implemented'
                _map = self.get_extent_map(norm=True, bisto=False, norm_method=norm_method,
                                           mean_type=mean_type, apply_mask=False, fdr_alpha=fdr_alpha)
                _map = self.extent_to_seq(_map, make_symmetric=True, summary_func=np.sum)
                logger.info(f'Normalised from_extent {_map.shape} with {_map.nnz} nnz')
            else:
                logger.debug('Using sequence map as a basis for normalisation')
                # tmp trial of removing weak off-diagonal elements
                # _map = sparse_utils.zero_weak_offdiag(_map, 2)
                # apply length normalization if requested
                _map = self._norm_seq(_map, self.is_tipbased(), method=norm_method, mean_type=mean_type)
            logger.debug('Map normalized')

        # make map bistochastic if requested
        if bisto:
            # TODO balancing may be better done after compression
            _map, scl = self._bisto_seq(_map)
            # retain the scale factors
            self.bisto_scale = scl
            qlo, qmed, qhi = np.quantile(scl, q=[0.025, 0.5, 0.975])
            logger.debug(f'Map balanced, scale factor range median:{qmed:.3f}, 95%:[{qlo:.3f},{qhi:.3f}]')

        # cache the results for optional quick access
        self.processed_map = _map

    def get_subspace(self,
                     permute: bool=False,
                     external_mask: Optional[npt.NDArray|list]=None,
                     marginalise: bool=False,
                     flatten: bool=True,
                     dtype: npt.DTypeLike=np.float64) -> SparseMatrix:
        """
        Using an already normalized full seq_map, return a subspace as indicated by an external
        mask or if none is supplied, the full map without filtered elements.

        The supplied external mask must refer to all sequences in the map.

        :param permute: Reorder the map with the current ordering state.
        :param external_mask: An external mask to combine with the existing primary mask.
        :param marginalise: Assuming 4D NxNx2x2 tensor, sum 2x2 elements to become a 2D NxN.
        :param flatten: Convert a NxNx2x2 tensor to a 2Nx2N matrix.
        :param dtype: Return map with the specific element type.
        :return: Subspace map.
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

    def get_extent_map(self,
                       norm: bool=True,
                       bisto: bool=False,
                       permute: bool=False,
                       mean_type: str='geometric',
                       norm_method: str='sites',
                       add_blocks: bool=False,
                       apply_mask: bool=True,
                       fdr_alpha: float=0.05) -> SparseMatrix:
        """
        Return the extent map after applying specified processing steps. Masked sequences are always removed.

        :param norm: Sequence length normalization.
        :param bisto: Make map bistochastic.
        :param permute: Permute the map using current order.
        :param mean_type: Choice of normalization mean (geometric, harmonic, arithmetic).
        :param norm_method: Methods used to normalize the matrix (length, sites, binomial).
        :param add_blocks: Add eulerian blocks to the map for contiguity during clustering.
        :param apply_mask: Apply the current primary acceptance mask.
        :param fdr_alpha: FDR alpha used in gothic normalization.
        :return: Processed extent map.
        """

        def calculate_blockweights(_map: sparse.COO, _borders: List[npt.NDArray]) -> npt.NDArray:
            """
            Rather than a single weight for all blocks, use weights relative
            to the interaction intensity of the block (intra and inter).

            :param _map: Extent map.
            :param _borders: Borders of blocks.
            :return: Weights for each block.
            """
            # use this for weights when a block has no interactions.
            _global = np.median(_map.data)
            # prepare triangular matrices for slicing
            _t = sp.triu(_map, k=0)
            # row slicing
            _trow = _t.tocsr()
            # col slicing
            _tcol = _t.tocsc()
            weights = []
            for a, b in _borders:
                _all_interactions = np.hstack([_tcol[:, a:b].data, _trow[a:b, :].data], dtype=np.float64)
                if len(_all_interactions) == 0:
                    weights.append(_global)
                else:
                    weights.append(np.median(_all_interactions))
            return np.array(weights)

        assert self.has_extent_map(), 'this instance of ContactMap does not contain an extent-based map.'

        logger.info('Preparing extent map with full dimension: {}'.format(self.extent_map.shape))

        _map = self.extent_map.astype(np.float64)

        # normalise map if requested
        if norm:
            _map = self._norm_extent(_map, method=norm_method, mean_type=mean_type, fdr_alpha=fdr_alpha)
            logger.debug('Map normalized')

        # make map bistochastic if requested
        if bisto:
            _map, scl = self._bisto_seq(_map)
            qlo, qmed, qhi = np.quantile(scl, q=[0.025, 0.5, 0.975])
            logger.debug(f'Map balanced, scale factor range median:{qmed:.3f}, 95%:[{qlo:.3f},{qhi:.3f}]')

        if add_blocks:
            # _edge_weight = 0.01 * np.median(_map.data)
            _edge_weight = calculate_blockweights(_map, self.grouping.borders)
            _blocks = sp.block_diag([_edge_weight[i] * np.ones((nn, nn)) for i, nn in enumerate(self.grouping.bins)])
            logger.debug(f'Eulerian block matrix dimensions: {_blocks.shape} and {_blocks.nnz:,} non-zero entries')
            _map += _blocks
            logger.debug(f'Augmented full map dimensions now: {_map.shape} and {_map.nnz:,} non-zero entries')

        # if there are sequences to mask, remove them from the map
        if apply_mask and self.order.count_accepted() < self.total_seq:
            _map = self._compress_extent(_map)
            logger.info('After removing filtered sequences, map dimensions: {}'.format(_map.shape))

        # reorder using current order state
        if permute:
            _map = self._reorder_extent(_map)
            logger.debug('Map reordered')

        return _map

    def extent_to_seq(self,
                      _ext_map: SparseMatrix,
                      make_symmetric: bool=False,
                      summary_func: Callable=np.sum) -> sparse.COO:
        """
        Convert the extent map to a simple sequence map. This is done by summing coincident interactions.

        :param _ext_map: The extent map to convert.
        :param make_symmetric: Make the output map a full symmetric matrix.
        :param summary_func: Function to produce summary values for sequences that span
        multiple bins within the extent map.
        :return: Seqmap.
        """
        logger.info('Condensing extent map to sequence map')
        _map_dim = self.grouping.bins.shape[0]
        _cbins = np.cumsum(self.grouping.bins)
        # this is a symmetric matrix, hence we use only the upper half
        _ext_map = sp.triu(_ext_map.tocoo())

        # iterate over the nonzero elements, summing within each bin
        _seq_map = defaultdict(list)
        for i, j, v in zip(_ext_map.row, _ext_map.col, _ext_map.data):
            _seq_map[bin_indices(i, j, _cbins)].append(v)

        # convert the dictionary representation to a sparse matrix
        _seq_map = sp.coo_matrix(([summary_func(v) for v in _seq_map.values()],
                                  ([row[0] for row in _seq_map], [col[1] for col in _seq_map])),
                                 shape=(_map_dim, _map_dim),
                                 dtype=np.float64)

        if make_symmetric:
            _seq_map = sparse_utils.make_symmetric(_seq_map)

        # if log_space:
        #     logger.debug('Extent-to-seq: Converting to log-space')
        #     _seq_map.data[:] = - np.log(_seq_map.data)

        # Update the acceptance mask, as it is possible that operations on the extent matrix
        # have resulted in sequences with zero interactions. These sequences will fail to be
        # represented in an edge-list format graph
        _mask = self.get_primary_acceptance_mask()
        reject_mask = _seq_map.sum(axis=0).A.squeeze() > 0
        logger.debug('Extent-to-seq: there were {} non-interacting sequences'.format((~reject_mask).sum()))
        # _mask &= reject_mask
        # self.order.set_mask_only(_mask)
        # self.primary_acceptance_mask = _mask

        return _seq_map.tocoo()

    def _reorder_seq(self, _map: SparseMatrix, flatten: bool=False) -> SparseMatrix:
        """
        Reorder a simple sequence map using the supplied map.

        :param _map: The map to reorder.
        :param flatten: Tip-based tensor converted to 2Nx2N matrix, otherwise the assumption is marginalization.
        :return: The ordered map.
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

    def _bisto_seq(self, _map: SparseMatrix) -> Tuple[SparseMatrix, npt.NDArray]:
        """
        Make a contact map bistochastic. This is another form of normalization. Automatically
        handles 2D and 4D maps.

        :param _map: A map to balance (make bistochastic).
        :return: The balanced map.
        """
        logger.debug('Balancing contact map')

        if self.is_tipbased():
            _map, scl = sparse_utils.kr_bistochastic_4d(_map)
        else:
            _map, scl = sparse_utils.kr_bistochastic(_map, delta=1e-3, Delta=1e2, tol=1e-8, max_iter=10000)
        return _map, scl

    def _get_sites(self) -> npt.NDArray:
        _sites = np.array([si.sites for si in self.seq_info], dtype=np.float64)
        # all sequences are assumed to have a minimum of 1 site -- even if not observed
        # TODO test whether it would be more accurate to assume that all sequences are under counted by 1.
        _sites[np.where(_sites == 0)] = 1
        return _sites

    def _norm_seq(self, _map: SparseMatrix,
                  tip_based: bool,
                  method: str='sites',
                  mean_type: str='geometric',
                  gothic_noself: bool=True) -> SparseMatrix:
        """
        Normalize a simple sequence map in place by the geometric mean of interacting contig pairs lengths.
        The map is assumed to be in starting order.

        :param _map: The target map to apply normalization.
        :param tip_based: Treat the supplied map as a tip-based tensor.
        :param method: The normalization method to use [sites, length, gothic].
        :param mean_type: For length normalization, choice of mean (harmonic, geometric, arithmetic).
        :param gothic_noself: Exclude self-self interactions when calculating relative coverage.
        :return: The normalized map.
        """
        if method == 'sites':

            logger.debug('Doing site based normalisation')
            _sites = self._get_sites()
            _map = _map.astype(np.float64)
            if tip_based:
                fast_norm_tipbased_bysite(_map.coords, _map.data, _sites)
            else:
                if not sp.isspmatrix_coo(_map):
                    _map = _map.tocoo()
                fast_norm_bysite(_map.row, _map.col, _map.data, _sites)

        elif method == 'length':

            logger.debug('Doing length based normalisation')
            if tip_based:
                _tip_lengths = np.minimum(self.tip_size, self.order.lengths()).astype(np.float64)
                fast_norm_tipbased_bylength(_map.coords, _map.data, _tip_lengths, self.tip_size)
            else:
                # TODO convert this to numba or remove
                logger.warning('length normalisation is not optimised and therefore very slow')
                _mean_func = mean_selector(mean_type)
                _len = self.order.lengths().astype(np.float64)
                _map = _map.tolil().astype(np.float64)
                for i in range(_map.shape[0]):
                    _map[i, :] /= np.fromiter((1e-3 * _mean_func(_len[i],  _len[j])
                                               for j in range(_map.shape[0])), dtype=np.float64)
                _map = _map.tocsr()

        elif method == 'gothic':
            if tip_based:
                raise ApplicationException('GOTHiC normalisation not supported with tip-based maps')
            goth_mode = 'binomial'
            logger.debug(f'Doing GOTHiC based {goth_mode} significance normalisation')

            if gothic_noself:
                _map = _map.tolil()
                _map.setdiag(0)

            _map = _map.tocsr()
            # take the upper triangle sum as total number of links (pairs) in the map
            total_links = sp.triu(_map).sum()
            # calculate relative length-normalized contig coverage
            seq_len = self.order.order['length'].astype(np.float64)
            rel_cov = _map.sum(axis=1).astype(np.float64)
            rel_cov = np.asarray(rel_cov).squeeze()
            # gothic normalizes this as reads_j / 2N
            # we introduce relative to the number of 5kb chunks
            rel_cov /= 2 * total_links * (seq_len / 5000.)
            _map = _map.tocoo().astype(np.float64)
            fast_norm_gothic(_map.row, _map.col, _map.data, rel_cov, total_links, 1., goth_mode)

        else:
            raise ApplicationException('unknown method {}'.format(method))

        return _map

    def _norm_extent(self,
                     _map: SparseMatrix,
                     method: str='length',
                     mean_type: str='geometric',
                     fdr_alpha: float=0.01,
                     reject_insig: bool=True) -> SparseMatrix:
        """
        Normalize an extent map in place by the geometric mean of interacting contig pairs lengths.

       :param method: The normalization method to use [sites, length, gothic].
       :param mean_type: For length normalization, choice of mean (harmonic, geometric, arithmetic).
       :param fdr_alpha: The FDR-BH alpha value to use for GOTHiC significance normalization.
       :param reject_insig: Reject insignificant interactions as determined after FDR correction -- for GOTHiC only.
       :return: A normalized extent map in lil_matrix format.
        """
        assert sp.isspmatrix(_map), 'Extent matrix is not a scipy matrix type'

        if not sp.isspmatrix_coo(_map):
            _map = _map.tocoo()

        # normalised data array
        if method == 'length':

            logger.debug('Extent_map: doing length based normalisation')

            # prepare the lookup array mapping contig length to any index of extent map
            _bins = self.grouping.bins
            _len = self.order.lengths()
            _len_lookup = []
            for i in range(len(_bins)):
                _len_lookup.extend([_len[i]] * _bins[i])
            _len_lookup = np.array(_len_lookup, dtype=np.float64)

            fast_length_norm(_map.row, _map.col, _map.data, _map.nnz, _len_lookup, mean_selector(mean_type))

        elif method == 'sites':

            logger.debug('Extent_map: doing site based normalisation')

            if not hasattr(self, 'seq_sites') or self.seq_sites is None:
                logger.warning('The contact map did not contain cut-site coords, attempting to refresh.')
                self.refresh_seqsites()

            # for all sequences, count the number of sites falling into each bin
            _nz = 0
            _cs_lookup = []
            for _seq_id, _sites in enumerate(self.seq_sites):
                _count = count_bin_sites(np.array(_sites), self.grouping.calc_borders(_seq_id))
                _nz += np.sum(_count == 0)
                _cs_lookup.append(_count)

            logger.debug(f'Number of bins with 0 observed cut-sites: {_nz:,}')

            # flatten the nested list of counts, this covers the full extent map dimensions.
            _cs_lookup = np.fromiter((_bin_count for _seq in _cs_lookup for _bin_count in _seq), dtype=int)
            # Adjust all bins by 1 to avoid div-zero: "smoothing" approach
            _cs_lookup += 1
            logger.debug(f'There were {len(_cs_lookup):,} bins containing {_cs_lookup.sum():,} predicted sites')

            fast_norm_bysite(_map.row, _map.col, _map.data, _cs_lookup)

        elif method == 'gothic':

            goth_mode = 'binomial'
            logger.debug(f'Extent_map: doing GOTHiC based {goth_mode} significance normalisation')

            # total number of off-diagonal observations
            total_obs = sp.triu(_map, k=1).sum()
            # marginals represent total hits per bin
            _map = _map.tocsr()
            # per-bin relative number of observations -- without diagonal elements
            rel_cov = _map.sum(axis=0) - _map.diagonal()
            rel_cov = rel_cov.astype(np.float64) / (2 * total_obs)
            # make the result a simple array
            rel_cov = rel_cov.A.squeeze()
            # only consider cross-bin interactions
            _map = sp.triu(_map.tocoo().astype(np.float64), k=1)
            fast_norm_gothic(_map.row, _map.col, _map.data, rel_cov, total_obs, 1., goth_mode)
            logger.debug(f'Extent_map: revising p-values using Benjamini-Hochberg FDR correction. alpha={fdr_alpha}')
            reject_h0,  pv_corr, _, _,  = multipletests(_map.data,
                                                        method='fdr_bh',
                                                        alpha=fdr_alpha,
                                                        is_sorted=False,
                                                        returnsorted=False)
            # revise p-values post-FDR
            _map.data[:] = pv_corr
            if reject_insig:
                _n_insig = (~reject_h0).sum()
                _p_insig = _n_insig / len(reject_h0) * 100
                logger.info(f'Extent_map: {_n_insig:,} insignificant interactions were removed ({_p_insig:.2f}%)')
                _map.data[~reject_h0] = 0
                _map.eliminate_zeros()

            # For Infomap clustering, values must be inverted
            # TODO explore other transformations
            _map.data[:] =  1 -_map.data

            _map = sparse_utils.make_symmetric(_map)
            # clear-up any lingering zeroed elements
            _map = _map.tocoo()

        else:
            raise ApplicationException('unknown method {}'.format(method))

        return _map

    def _reorder_extent(self, _map: SparseMatrix) -> SparseMatrix:
        """
        Reorder the extent map using current order.

        :return: Sparse CSR format permutation of the given map.
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

    def _compress_extent(self, _map: SparseMatrix) -> sp.coo_matrix:
        """
        Compress the extent map for each sequence that is presently masked. This will eliminate
        all bins that pertain to a given masked sequence.

        :return: A scipy.sparse.coo_matrix pertaining to only the unmasked sequences.
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

        _mask = np.zeros(self.grouping.total_bins, dtype=bool)
        _mask[np.array(accept_bins)] = True

        _data, _row, _col, _shift = sparse_utils.fast_retained(_map.data, _map.row, _map.col, _map.nnz, _mask)

        return sp.coo_matrix((_data, (_row, _col)), shape=np.array(_map.shape) - _shift[-1])

    def plot_seqnames(self,
                      fname: str,
                      simple: bool=True,
                      permute: bool=False,
                      **kwargs: Dict[str, Any]) -> None:
        """
        Plot the contact map, annotating the map with sequence names. WARNING: This can often be too dense
        to be legible when there are hundreds to thousands of sequences.

        :param fname: Output file name.
        :param simple: True plot seq map, False plot the extent map.
        :param permute: Permute the map with the present order.
        :param kwargs: Additional options passed to plot().
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
            tick_locs = np.arange(2, step*self.order.count_accepted()+step, step)
        else:
            if permute:
                _cbins = np.cumsum(self.grouping.bins[self.order.accepted_positions()])
            else:
                _cbins = np.cumsum(self.grouping.bins[self.order.accepted()])
            tick_locs = _cbins - 0.5

        self.plot(fname, permute=permute, simple=simple, tick_locs=tick_locs, tick_labs=tick_labs, **kwargs)

    def plot(self,
             fname: str,
             simple: bool=False,
             tick_locs: Optional[npt.NDArray[np.int_]]=None,
             tick_labs: Optional[List[str]]=None,
             norm: bool=True,
             permute: bool=False,
             pattern_only: bool=False,
             dpi: int=180,
             width: int=25,
             height: int=22,
             zero_diag: bool=True,
             alpha: float=0.001,
             max_image_size: Optional[int]=None,
             flatten: bool=False,
             norm_method: Optional[str]=None,
             bisto: bool=True) -> None:
        """
        Plot the contact map. This can either be as a sparse pattern (requiring much less memory but without visual
        cues about intensity), simple sequence or full binned map and normalized or permuted.

        :param fname: Output file name.
        :param tick_locs: Major tick locations (minors take the midpoints).
        :param tick_labs: Minor tick labels.
        :param simple: If true, sequence only map plotted.
        :param norm: Normalize intensities by the geometric mean of lengths.
        :param permute: Reorder map to current order.
        :param pattern_only: Plot only a sparse pattern (much lower memory requirements).
        :param dpi: Adjust DPI of output.
        :param width: Plot width in inches.
        :param height: Plot height in inches.
        :param zero_diag: Set bright self-interactions to zero.
        :param alpha: Log intensities are log (x + alpha).
        :param max_image_size: Maximum allowable image size before rescaling occurs.
        :param flatten: For tip-based, flatten matrix rather than marginalize.
        :param norm_method: Normalization method to apply to contact map.
        :param bisto: Make map bistochastic.
        """

        plt.style.use('ggplot')

        fig = plt.figure()
        fig.set_figwidth(width)
        fig.set_figheight(height)
        ax = fig.add_subplot(111)

        if simple or self.bin_size is None:
            if norm_method is None:
                norm_method = 'sites'
            # Prepare the map if not already done. This overwrites
            # any current ordering mask beyond the primary acceptance mask
            if self.processed_map is None:
                self.prepare_seq_map(norm=norm, bisto=bisto, norm_method=norm_method)
            _map = self.get_subspace(permute=permute, marginalise=False if flatten else True, flatten=flatten)
            # amplify values for plotting
            _map *= 10
        else:
            if norm_method is None:
                norm_method = 'length'
            _map = self.get_extent_map(norm=norm, bisto=bisto, permute=permute, norm_method=norm_method)

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
                    tick_locs = np.floor(tick_locs.astype(np.float64) / reduce_factor)
                    logger.info('Map has been reduced from {} to {}'.format(full_size, _map.shape))

            _map = _map.toarray()

            if zero_diag:
                logger.debug('Removing diagonal')
                np.fill_diagonal(_map, 0)

            _map = np.log(_map + alpha)

            logger.debug('Making raster image')
            plt.imshow(_map, cmap=seaborn.color_palette("rocket", as_cmap=True), interpolation=None)

        if tick_locs is not None:

            plt.tick_params(axis='both', which='both',
                            right=False, left=False, bottom=False, top=False,
                            labelright=False, labelleft=False, labelbottom=False, labeltop=False)

            if tick_labs is not None:
                min_labels = ticker.FixedFormatter(tick_labs)
                ax.tick_params(axis='y', which='minor', left=True, labelleft=True, labelsize=10)

                min_ticks = ticker.FixedLocator(tick_locs[:-1] + 0.5 * np.diff(tick_locs))

                ax.yaxis.set_minor_locator(min_ticks)
                ax.yaxis.set_minor_formatter(min_labels)

            ax.yaxis.set_major_locator(ticker.FixedLocator(tick_locs))
            ax.xaxis.set_major_locator(ticker.FixedLocator(tick_locs))
            ax.grid(True, which='major', axis='both', color='grey', linewidth=0.2, linestyle='-.')

        logger.debug('Saving plot')
        fig.tight_layout()
        plt.savefig(fname, dpi=dpi)
        plt.close(fig)

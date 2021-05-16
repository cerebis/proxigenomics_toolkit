from ..exceptions import *
from ..misc_utils import exe_exists
from Bio.Restriction import Restriction
from difflib import SequenceMatcher
from collections import Mapping, namedtuple
from scipy.stats import mstats
import Bio.SeqIO as SeqIO
import heapq
import itertools
import numpy as np
import networkx as nx
import os
import re
import subprocess
import tempfile
import uuid
import yaml
import logging
import multiprocessing

logger = logging.getLogger(__name__)

# translation table used for complementation
COMPLEMENT_TABLE = str.maketrans('acgtumrwsykvhdbnACGTUMRWSYKVHDBN',
                                 'TGCAAnnnnnnnnnnnTGCAANNNNNNNNNNN')


def revcomp(seq):
    """
    Reverse complement a string representation of a sequence. This uses string.translate.
    :param seq: input sequence as a string
    :return: revcomp sequence as a string
    """
    return seq.translate(COMPLEMENT_TABLE)[::-1]


def count_bam_reads(file_name, max_cpu=None):
    """
    Use samtools to quickly count the number of non-header lines in a bam file. This is assumed to equal
    the number of mapped reads.
    :param file_name: a bam file to scan (neither sorted nor an index is required)
    :param max_cpu: maximum number of cpus to use for accessing bam file
    :return: estimated number of mapped reads
    """
    assert exe_exists('samtools'), 'required tool samtools was not found on path'
    assert exe_exists('wc'), 'required tool wc was not found on path'

    if not os.path.exists(file_name):
        raise IOError('{} does not exist'.format(file_name))
    if not os.path.isfile(file_name):
        raise IOError('{} is not a file'.format(file_name))

    opts = ['samtools', 'view', '-c']
    if max_cpu is None:
        max_cpu = multiprocessing.cpu_count()
    opts.append('-@{}'.format(max_cpu))

    proc = subprocess.Popen(opts + [file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    value_txt = proc.stdout.readline().strip()
    try:
        return int(value_txt)
    except ValueError:
        raise RuntimeError('Encountered a problem determining alignment count. samtools returned [{}]'
                           .format(value_txt))


def count_fasta_sequences(file_name):
    """
    Estimate the number of fasta sequences in a file by counting headers. Decompression is automatically attempted
    for files ending in .gz. Counting and decompression is by why of subprocess calls to grep and gzip. Uncompressed
    files are also handled. This is about 8 times faster than parsing a file with BioPython and 6 times faster
    than reading all lines in Python.

    :param file_name: the fasta file to inspect
    :return: the estimated number of records
    """
    if file_name.endswith('.gz'):
        proc_uncomp = subprocess.Popen(['gzip', '-cd', file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc_read = subprocess.Popen(['grep', r'^>'], stdin=proc_uncomp.stdout, stdout=subprocess.PIPE)
    else:
        proc_read = subprocess.Popen(['grep', r'^>', file_name], stdout=subprocess.PIPE)

    n = 0
    for _ in proc_read.stdout:
        n += 1
    return n


class IndexedFasta(Mapping):
    """
    Provides indexed access to a sequence file in Fasta format which can be compressed using bgzip.

    The SeqIO.index_db method is used for access, and the temporary index is stored in the system
    tmp directory by default. At closing, the index will be closed and the temporary file removed.

    Wrapped with contextlib.closing(), instances of this object are with() compatible.
    """

    def __init__(self, fasta_file, tmp_path=None):
        """
        :param fasta_file: the input fasta file to access
        :param tmp_path: temporary directory for creating index
        """
        if tmp_path is None:
            tmp_path = tempfile.gettempdir()
        elif not os.path.exists(tmp_path):
            raise IOError('specified temporary path [{}] does not exist'.format(tmp_path))
        self._tmp_file = os.path.join(tmp_path, '{}.seqdb'.format(str(uuid.uuid4())))
        self._fasta_file = fasta_file
        self._index = SeqIO.index_db(self._tmp_file, self._fasta_file, 'fasta')

    def __getitem__(self, _id):
        """
        Access a sequence object by its Fasta identifier.

        :param _id: fasta sequence identifier
        :return: SeqRecord object representing the sequence
        """
        return self._index[_id]

    def __iter__(self):
        """
        :return: iterator over sequence identifiers (keys)
        """
        return iter(self._index)

    def __len__(self):
        """
        :return: the number of sequences in the file
        """
        return len(self._index)

    def close(self):
        """
        Close the index and remove the associated temporary file
        """
        if self._index:
            self._index.close()
        if os.path.exists(self._tmp_file):
            os.unlink(self._tmp_file)


# Information pertaining to digestion / private class for readability
digest_info = namedtuple('digest_info', ('enzyme', 'end5', 'end3', 'overhang'))
# Information pertaining to a Hi-C ligation junction
ligation_info = namedtuple('ligation_info',
                           ('enz5p', 'enz3p', 'junction', 'vestigial', 'junc_len',
                            'vest_len', 'pattern', 'cross'))


class SiteCounter(object):

    def __init__(self, enzyme_a, enzyme_b=None, tip_size=None, is_linear=True):
        """
        Simple class to count the total number of enzymatic cut sites for the given
        list if enzymes.

        :param enzyme_a: the first enzyme involved in digestion (case sensitive NEB name)
        :param enzyme_b: the optional second enzyme involved in digestion (case sensitive NEB name)
        :param tip_size: when using tip based counting, the size in bp
        :param is_linear: Treat sequence as linear.
        """
        self.enzyme_a = SiteCounter._get_enzyme_instance(enzyme_a)
        self.enzyme_b = None if enzyme_b is None else SiteCounter._get_enzyme_instance(enzyme_b)
        self.is_linear = is_linear
        self.tip_size = tip_size
        self.junctions = self.junction_duplication()
        # Digested ends, which are religated will not necessarily possess the entire
        # cut-site, but will possess a remnant/vestigial sequence
        self.any_vestigial = re.compile('({})'.format('|'.join(
            sorted(set([v.vestigial for v in self.junctions.values()]),
                   key=lambda x: (-len(x), x))).replace('N', '[ACGT]')))
        self.end_vestigial = re.compile('{}$'.format(self.any_vestigial.pattern))

    def get_vestigial_end_searcher(self):
        return self.end_vestigial.search

    @staticmethod
    def _get_enzyme_instance(enz_name):
        """
        Fetch an instance of a given restriction enzyme by its name.

        :param enz_name: the case-sensitive name of the enzyme
        :return: RestrictionType the enzyme instance
        """
        try:
            # this has to match exactly
            return getattr(Restriction, enz_name)
        except AttributeError:
            # since we're being strict about enzyme names, be helpful with errors
            similar = []
            for a in dir(Restriction):
                score = SequenceMatcher(None, enz_name.lower(), a.lower()).ratio()
                if score >= 0.8:
                    similar.append(a)
            raise UnknownEnzymeException(enz_name, similar)

    def _count(self, seq):
        return sum(len(en.search(seq, self.is_linear)) for en in [self.enzyme_a, self.enzyme_b] if en is not None)

    def count_sites(self, seq):
        """
        Count the number of sites found in the given sequence, where sites from
        all specified enzymes are combined

        :param seq: Bio.Seq object
        :return: the total number of sites
        """
        if self.tip_size:
            seq_len = len(seq)
            if seq_len < 2*self.tip_size:
                # small contigs simply divide their extent in half
                l_tip = seq[:seq_len/2]
                r_tip = seq[-seq_len/2:]
            else:
                l_tip = seq[:self.tip_size]
                r_tip = seq[-self.tip_size:]
            # left and right tip counts
            sites = [self._count(l_tip), self._count(r_tip)]
        else:
            # one value for whole sequence
            sites = self._count(seq)

        return sites

    @staticmethod
    def enzyme_ends(enzyme):
        """
        Determine the 5` and 3` ends of a restriction endonuclease. Here, "ends"
        are the parts of the recognition site not involved in overhang.
        :param enzyme: Biopython ins
        :return: a pair of strings containing the 5' and 3' ends
        """
        end5, end3 = '', ''
        ovhg_size = abs(enzyme.ovhg)
        if ovhg_size > 0 and ovhg_size != enzyme.size:
            a = abs(enzyme.fst5)
            if a > enzyme.size // 2:
                a = enzyme.size - a
            end5, end3 = enzyme.site[:a], enzyme.site[-a:]
        return end5.upper(), end3.upper()

    @staticmethod
    def vestigial_site(enz, junc):
        """
        Determine the part of the 5-prime end cut-site that will remain when a ligation junction is
        created. Depending on the enzymes involved, this may be the entire cut-site or a smaller portion
        and begins from the left.
        """
        i = 0
        while i < enz.size and (enz.site[i] == junc[i] or enz.site[i] == 'N'):
            i += 1
        return str(junc[:i]).upper()

    def junction_duplication(self):
        """
        For the enzyme cocktail, generate the set of possible ligation junctions.
        :return: a dictionary of ligation_info objects
        """
        end5, end3 = SiteCounter.enzyme_ends(self.enzyme_a)
        enz_list = [digest_info(self.enzyme_a, end5, end3, self.enzyme_a.ovhgseq.upper())]

        if self.enzyme_b is not None:
            end5, end3 = SiteCounter.enzyme_ends(self.enzyme_b)
            enz_list.append(digest_info(self.enzyme_b, end5, end3, self.enzyme_b.ovhgseq.upper()))

        _junctions = {}
        for a, b in itertools.product(enz_list, repeat=2):
            # the actual sequence
            _junc_seq = '{}{}{}{}'.format(a.end5, a.overhang, b.overhang, b.end3)
            if _junc_seq not in _junctions:
                _vest_seq = SiteCounter.vestigial_site(a.enzyme, _junc_seq)
                _junctions[_junc_seq] = ligation_info(str(a.enzyme), str(b.enzyme),
                                                      _junc_seq, _vest_seq,
                                                      len(_junc_seq), len(_vest_seq),
                                                      re.compile(_junc_seq.replace('N', '[ACGT]')),
                                                      str(a.enzyme) == str(b.enzyme))
        return _junctions


class SequenceAnalyzer(object):

    COV_TYPE = np.dtype([('index', np.int16), ('status', np.bool), ('node', np.float),
                         ('local', np.float), ('fold', np.float)])

    @staticmethod
    def read_report(file_name):
        return yaml.safe_load(open(file_name, 'r'))

    def __init__(self, seq_map, seq_report, seq_info, tip_size):
        self.seq_map = seq_map
        self.seq_report = seq_report
        self.seq_info = seq_info
        self.tip_size = tip_size

    def _contact_graph(self):
        g = nx.Graph()
        n_seq = len(self.seq_info)

        for i in range(n_seq):
            si = self.seq_info[i]
            d = self.seq_report['seq_info'][si.name]
            if self.tip_size:
                g.add_node(i, _id=si.name,
                           _cov=float(d['coverage']),
                           # this is a 2 element list for tip mapping
                           _sites=d['sites'],
                           _len=int(d['length']))
            else:
                g.add_node(i, _id=si.name,
                           _cov=float(d['coverage']),
                           _sites=int(d['sites']),
                           _len=int(d['length']))

        _m = self.seq_map
        if self.tip_size:
            _m = _m.sum(axis=(2, 3))

        for i in range(n_seq):
            for j in range(i, n_seq):
                if _m[i, j] > 0:
                    g.add_edge(i, j, weight=float(_m[i, j]))

        return g

    @staticmethod
    def _nlargest(g, u, n, k=0, local_set=None):
        """
        Build a list of nodes of length n within a radius k hops of node u in graph g, which have the
        largest weight. For Hi-C data, after normalisation, high weight can be used as a means of inferring
        proximity.

        :param g: the graph which to analyse
        :param u: the target node
        :param n: the length of the 'nearby nodes' list
        :param k: the maximum number of hops away from u
        :param local_set: used in recursion
        :return: a set of nodes.
        """
        if not local_set:
            local_set = set()

        neighbors = [v[0] for v in heapq.nlargest(n+1, g[u].items(), key=lambda x: x[1]['weight'])]
        local_set.update(neighbors)
        if k > 0:
            for v in neighbors:
                if v == u:
                    continue
                SequenceAnalyzer._nlargest(g, v, n, k-1, local_set)

        return list(local_set)

    def report_degenerates(self, fold_max, min_len=0):
        """
        Making the assumption that degenerate sequences (those sequences which are repeats) have high coverage
        relative to their local region, report those nodes in the graph whose coverage exceeds a threshold.

        :param fold_max: the maximum relative coverage allowed (between a node and its local region)
        :param min_len: the shorest allowable sequence to consider
        :return: a report of all sequences degenerate status
        """

        g = self._contact_graph()

        degens = []
        for u in g.nodes():
            if g.nodes[u]['_len'] < min_len or g.degree[u] == 0:
                continue

            signif_local_nodes = SequenceAnalyzer._nlargest(g, u, 4, 1)
            local_mean_cov = mstats.gmean(np.array([g.nodes[v]['_cov'] for v in signif_local_nodes]))
            fold_vs_local = g.nodes[u]['_cov'] / float(local_mean_cov)

            is_degen = True if fold_vs_local > fold_max else False

            degens.append((u, is_degen, g.nodes[u]['_cov'], local_mean_cov, fold_vs_local))

        degens = np.array(degens, dtype=SequenceAnalyzer.COV_TYPE)

        if len(degens) == 0:
            logger.debug('No degenerate sequences found')
        else:
            logger.debug('Degenerate sequence report')
            for di in degens[degens['status']]:
                logger.debug(di)

        return degens

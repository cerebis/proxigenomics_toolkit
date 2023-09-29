from ..exceptions import InvalidSequenceException
from ..io_utils import load_object, open_input
import logging
from collections import Counter, namedtuple, OrderedDict, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas
import rpy2.robjects as robjects
import scipy.sparse as sp
import scipy.stats as st
import seaborn as sb
import tqdm
from astropy.stats import sigma_clip
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.mixture import BayesianGaussianMixture
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# TODO suppressing FutureWarnings from seaborn until a release (>12.2) addresses (added 2023-09-18)
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')


def sequence_details(contact_map, coverage_info, mappability_info, clustering):
    """
    Prepare a table of sequence information to annotate the seq2cluster graph.

    :param contact_map: a bin3C contact map
    :param coverage_info: coverage data for every sequence
    :param mappability_info: mappability index for every sequence
    :param clustering: a bin3C clustering solution for contact_map
    :return: pandas dataframe of per sequence annotation details
    """
    # prepare a dataframe of sequence details
    # 1. start with seq_info from contact map
    seq_info = np.array(contact_map.seq_info, dtype=np.dtype(
        [('offset', np.int64), ('refid', np.int64), ('name', np.object_),
         ('length', np.int64), ('sites', np.int64), ('gc', np.float64)]))
    # 2. drop unneeded the refid and offset columns
    seq_info = pandas.DataFrame(seq_info).drop(['refid', 'offset'], axis=1)
    seq_info.index.name = 'seq_id'

    # 3. join the table of coverage, using sequence name as the key
    seq_info = seq_info.reset_index(drop=False)
    before_len = len(seq_info)
    seq_info = seq_info.set_index('name', drop=False).join(coverage_info, how='inner')
    assert len(seq_info) == before_len, ('Coverage data did not contain information '
                                         'for all clustered sequences')
    # just include uniq_frac column
    seq_info = seq_info.join(mappability_info[['uniq_frac']], how='inner')
    assert len(seq_info) == before_len, ('Mappability data did not contain information '
                                         'for all clustered sequences')

    # 4. prepare a lookup table from seq_id to sequence length and number of sites
    # - this is used for removing the contribution of an excluded
    #   sequence from its cluster.
    _ix2info = seq_info.set_index('seq_id').loc[:,
               ['length', 'sites', 'gc', 'coverage', 'uniq_frac']].to_dict(orient='records')

    # create a cluster id column, with isolated sequences assigned -1
    _nm2cl = [(contact_map.seq_info[_si].name, _cl) for _cl, _d in clustering.items() for _si in _d['seq_ids']]
    _nm2cl = pandas.DataFrame.from_records(_nm2cl, columns=['name', 'cluster']).set_index('name')
    seq_info = seq_info.join(_nm2cl, how='left')

    # for sequences with no cluster membership, assign a cluster id of -1.
    seq_info['cluster'] = pandas.to_numeric(seq_info.cluster.fillna(-1), downcast='integer')
    logger.info('{:,} of {:,} sequences were assigned to no cluster'.format(
        (seq_info['cluster'] == -1).sum(), len(seq_info)))

    _id2cl = seq_info.set_index('seq_id')['cluster'].to_dict()

    return seq_info, _ix2info, _id2cl


def create_seq2cluster_graph(contact_map, clustering, coverage_info, mappability_info, min_seq_length, tidy=True):
    """
    Create bipartite graph

    :param contact_map:
    :param clustering:
    :param coverage_info:
    :param mappability_info: mappability index for every sequence
    :param min_seq_length:
    :param tidy: remove collections of member attributes
    :return:
    """

    def sum_min1(x):
        """
        Sum values of x, where any element less than 1 becomes 1.
        :param x: array to sum
        :return: sum of x
        """
        return np.sum(np.maximum(1, x))

    seq_info, _ix2info, _ix2cl = sequence_details(contact_map, coverage_info, mappability_info, clustering)

    # calculation using weighted means (by length of sequence)
    seq_grp = seq_info.assign(wcv=lambda x: x.length * x.coverage,
                              wgc=lambda x: x.length * x.gc,
                              wuf=lambda x: x.length * x.uniq_frac)

    cl_agg = (seq_grp.groupby('cluster')
              .aggregate({'wcv': 'sum',
                          'wgc': 'sum',
                          'wuf': 'sum',
                          'sites': sum_min1,
                          'length': ['sum', 'count']})
              # rename columns for ease of reference
              .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1))
              # final values per cluster
              .assign(length=lambda x: x.length_sum,
                      size=lambda x: x.length_count,
                      sites=lambda x: x.sites_sum_min1,
                      coverage=lambda x: x.wcv_sum / x.length_sum,
                      gc=lambda x: x.wgc_sum / x.length_sum,
                      uniq_frac=lambda x: x.wuf_sum / x.length_sum)
              # remove unneeded columns
              .drop(['wcv_sum', 'wgc_sum', 'wuf_sum', 'sites_sum_min1', 'length_sum', 'length_count'], axis=1))

    # simple unweighted means
    # seq_grp = seq_info.groupby('cluster').agg({'length': ['sum', 'count'],
    #                                            'sites': sum_min1,
    #                                            'coverage': 'mean',
    #                                            'gc': 'mean',
    #                                            'uniq_frac': 'mean'}, axis=0)
    # cl_agg = pandas.DataFrame()
    # cl_agg['length'] = pandas.to_numeric(seq_grp['length', 'sum'], downcast='integer')
    # cl_agg['size'] = pandas.to_numeric(seq_grp['length', 'count'], downcast='integer')
    # cl_agg['sites'] = pandas.to_numeric(seq_grp['sites', 'sum_min1'], downcast='integer')
    # cl_agg['coverage'] = seq_grp['coverage', 'mean']
    # cl_agg['gc'] = seq_grp['gc', 'mean']
    # cl_agg['uniq_frac'] = seq_grp['uniq_frac', 'mean']
    del seq_grp

    # get the upper diagonal of the contact_map, excluding x=y (i.e. without self-self interactions)
    _map = sp.triu(contact_map.seq_map, k=1).tocsr()

    # Steps in building the graph
    # 1. add cluster nodes
    # 2. add sequence nodes
    #   a. for all sequences u:
    #   b.   for all sequences ui which interact with (are neighbors of) the source node u
    #   c.     find the owning cluster for neighbour sequence ui, which becomes the destination node v
    #   d.     for the edge (u, v) add interaction count (edge weight: w(u, ui)) between sequences u and ui.

    g = nx.Graph()

    # first create all the "cluster" nodes
    # where their ids follow the syntax "('c', int)"
    for _cl, _cl_dat in clustering.items():
        nattr = cl_agg.loc[_cl].to_dict(dict)
        assert len(_cl_dat['seq_ids']) == int(nattr['size']), 'problem with pandas calculated cluster size'
        nattr['bipartite'] = 0
        g.add_node(('c', _cl), **nattr)

    def validate_sequence(ix, track_removed=False):
        """
        Check that a sequence belongs to a cluster and is sufficiently long. In addition, the
        method keeps track of accepted/rejected sequences and their attributes for each cluster.
        This is later used to calculate updated cluster attributes. As a consequence, this methods
        modifies nodes within the graph.
        :ix: sequence index
        :track_removed: when true, attributes of removed sequences are stored with each cluster node
        :return: cluster_id
        :raise: InvalidSequenceException for rejected sequences
        """
        cl = _ix2cl[ix]
        if cl == -1:
            no_clust.add(ix)
            raise InvalidSequenceException

        cluster_id = ('c', cl)
        assert g.has_node(cluster_id), f'cluster node {cluster_id} is missing from graph'

        if ix in removed_seqs:
            raise InvalidSequenceException

        info = _ix2info[ix]
        if info['length'] < min_seq_length:
            if track_removed:
                cl_node = g.nodes[cluster_id]
                cl_node.setdefault('removed', {})[ix] = info
            removed_seqs.add(ix)
            raise InvalidSequenceException

        cl_node = g.nodes[cluster_id]
        cl_node.setdefault('members', {})[ix] = info

        return cluster_id

    # create sequence nodes and link to clusters through Hi-C interactions
    # where their ids follow the syntax "('n', int)"
    removed_seqs = set()
    no_clust = set()
    # we will iterate over sequences using the table of per-sequence annotation details
    seq_info_array = seq_info.loc[:,
                     ['seq_id', 'name', 'length', 'sites', 'coverage', 'gc', 'cluster', 'uniq_frac']].values
    for _si, _name, _len, _sites, _cov, _gc, _clust, _uf in tqdm.tqdm(seq_info_array):

        try:
            validate_sequence(_si)

            # sequence node definition with attributes
            node_from = ('n', _name)
            nattr = {'name': _name,
                     'length': _len,
                     'sites': _sites,  # assign 1 site to those sequences with 0
                     'coverage': _cov,  # flye can report 0 depth on assembly graph segments
                     'membership': _clust,
                     'gc': _gc,
                     'uniq_frac': _uf,
                     'bipartite': 1}
            g.add_node(node_from, **nattr)

        except InvalidSequenceException:
            continue

        # iterate over all interactions between sequence _si and any other sequence in the map _sj
        _contacts = _map[_si].tocoo()
        for _sj, _contacts_ij in zip(_contacts.col, _contacts.data):

            try:
                cluster_to = validate_sequence(_sj)

                # add the edge if new
                if not g.has_edge(node_from, cluster_to):
                    g.add_edge(node_from, cluster_to, contacts=0)

                # accumulate these new contacts between sequence and members of cluster
                # - this comes into effect when a sequence interacts with many sequences in
                #   the same cluster.
                g[node_from][cluster_to]['contacts'] += _contacts_ij
            except InvalidSequenceException:
                continue

    # update cluster nodes to reflect loss of rejected sequences
    cluster_nodes = [u for u in g.nodes() if u[0] == 'c']
    for u in cluster_nodes:
        if 'members' in g.nodes[u]:
            df_attribs = pandas.DataFrame(g.nodes[u]['members']).T
            g.nodes[u].update({'size': len(df_attribs),
                               'length': df_attribs.length.sum(),
                               'sites': df_attribs.sites.sum(),
                               'gc': (df_attribs.gc * df_attribs.length / df_attribs.length.sum()).sum(),
                               'coverage': (df_attribs.coverage * df_attribs.length / df_attribs.length.sum()).sum(),
                               'uniq_frac': (df_attribs.uniq_frac * df_attribs.length / df_attribs.length.sum()).sum()})
            if tidy:
                del g.nodes[u]['members']
        if tidy and 'removed' in g.nodes[u]:
            del g.nodes[u]['removed']

    logger.info('{:,} sequences were too short to be destinations (< {:,} bp)'.format(
        len(removed_seqs), min_seq_length))
    logger.info('{:,} inter-sequence associations did not involve a cluster'.format(
        len(no_clust)))

    return g


def simple_spurious_estimation(seq2cl_graph):
    """
    Rough estimate of spurious fraction

    :param seq2cl_graph:
    :return:
    """

    inter_list = []
    intra_list = []
    ulen = []
    udeg = []

    # list of cluster nodes
    cluster_nodes = []
    for u in seq2cl_graph.nodes():
        if u[0] != 'c' or seq2cl_graph.nodes[u]['length'] < 200000:
            continue
        cluster_nodes.append(u)

    logger.info('Inspecting the {} largest clusters'.format(len(cluster_nodes)))

    # interate over all cluster nodes
    for u in cluster_nodes:

        inter = 0
        intra = 0
        n = 0
        m = 0
        # interate over the all neighbours of u
        for v in seq2cl_graph.neighbors(u):
            if seq2cl_graph.nodes[v]['length'] < 2000:
                continue
            # count within-cluster edges
            if seq2cl_graph.nodes[v]['membership'] == u[1]:
                intra += seq2cl_graph[u][v]['contacts']
                n += 1
            # count between-cluster edges
            else:
                inter += seq2cl_graph[u][v]['contacts']
                m += 1

        if n < 10 or m < 10:
            continue
        udeg.append(seq2cl_graph.degree(u))
        ulen.append(seq2cl_graph.nodes[u]['length'])
        inter_list.append(inter)
        intra_list.append(intra)

    in_out = pandas.DataFrame({'in': intra_list, 'out': inter_list, 'ulen': ulen, 'udeg': udeg})

    in_out['R'] = in_out['out'] / (in_out['in'] + in_out['out'])
    p = sb.displot(in_out, x='R', kde=True, rug=True)
    p.savefig('fraction_inter.png')
    logger.info('95% CI: [{:.3f}:{:.3f}]'.format(
        *st.t.interval(0.95, len(in_out) - 1, loc=np.mean(in_out.R.mean()), scale=st.sem(in_out.R))))
    logger.info('Mean: {:.3f} Median: {:.3f}'.format(
        in_out.R.mean(), in_out.R.median()))
    logger.info('Weighted mean against cluster extent: {:.3f}'.format(
        sum(in_out['R'] * in_out['ulen'] / in_out['ulen'].sum())))


def calculate_rejection_thresholds(seq2cl_graph, prob_outlier=0.98):
    """
    Calculate the rejection threshold for observed interacting sequences, whose characteristics would
    suggest that they are likely to be significant (true) interactions. Here, we consider the ratio of
    contacts to number of sites (cps) and length to number of sites (lps). High values of either
    quantity indicates sequences which statistically we would expect to interact strongly. These sequences
    are often long, abundant and with high density of sites.

    :param seq2cl_graph: the bipartite sequence to cluster graph
    :param prob_outlier: the quantile to use in establishing rejection thresholds
    """

    Info = namedtuple('info', ['node_id', 'length', 'sites', 'contacts'])

    df_inter = []
    # iterate over all cluster nodes
    cluster_nodes = [u for u in seq2cl_graph if u[0] == 'c']
    for u in cluster_nodes:
        # for all sequences interacting with a cluster
        #   if the sequence placed in the cluster -> it is an intra-genome_bin interaction
        #   if the sequence was placed in another cluster -> it is an inter-genome_bin interaction
        for v in seq2cl_graph.neighbors(u):
            v_dat = seq2cl_graph.nodes[v]
            if v_dat['membership'] == u[1]:
                continue
            df_inter.append(Info(v, v_dat['length'], v_dat['sites'], seq2cl_graph[u][v]['contacts']))

    # calculate some statistics
    df_inter = pandas.DataFrame(df_inter)
    # upgrade # sites from 0 to 1
    df_inter['sites'].replace(0, 1, inplace=True)
    logger.info('There were {:,} inter-genome_bin observations'.format(len(df_inter)))

    # contacts per # sites
    df_inter['cps'] = df_inter['contacts'] / df_inter['sites']
    # seq length per # sites
    df_inter['lps'] = df_inter['length'] / df_inter['sites']

    # simple quantile filtering
    cps_max = df_inter.cps.quantile(q=prob_outlier)
    lps_max = df_inter.lps.quantile(q=prob_outlier)
    logger.info('Max CPS {} LPS {}'.format(cps_max, lps_max))

    # indices of observations with less than these value are rejected
    ix_cps = df_inter.cps > cps_max
    ix_lps = df_inter.lps > lps_max
    logger.info('CPS rejects: {:,}'.format(ix_cps.sum()))
    logger.info('LPS rejects: {:,}'.format(ix_lps.sum()))

    # we will reject the union of these criteria
    ix = ix_cps | ix_lps
    logger.info('Quantile filtering will exclude {:,} outliers'.format(ix.sum()))

    # plot a sample of the total observations
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(10)
    _smpl = df_inter[~ix].sample(2000)
    p = sb.jointplot(x='lps', y='cps', data=_smpl, kind='kde')
    p.savefig('inter_lps-cps_jointplot.png')

    return cps_max, lps_max


def fill_zeros(df, columns, value):
    for cn in columns:
        zix = df[cn] == 0
        logger.info('For {} replacing {} zeros with {}'.format(cn, zix.sum(), value))
        df.loc[zix, cn] = value


def rmatrix2pandas(_rmat):
    """
    Convert an R matrix object into a pandas dataframe
    :param _rmat: R matrix object from rpy2
    :return: pandas dataframe
    """
    _it = _rmat.items()
    _out = defaultdict(dict)
    for j in range(_rmat.dim[1]):
        for i in range(_rmat.dim[0]):
            _out[_rmat.rownames[i]][_rmat.colnames[j]] = next(_it)[1]
    _out = pandas.DataFrame(_out).T
    return _out


def rvector2dict(_rvec):
    """
    Convert an R vector object into a dictionary
    :param _rvec: R vector object from rpy2
    :return: dictionary
    """
    return dict(_rvec.items())


def robust_read_csv(csv_name, sep=','):
    """
    Read CSV file whether or not it has a header. If a non-numeric row-0 is
    found, drop it from the table.

    :param csv_name: csv file name
    :param sep: separator used
    :return: pandas.DataFrame
    """
    _col_def = OrderedDict({'name': str, 'coverage': np.float64})
    try:
        df = pandas.read_csv(csv_name, header=None, sep=sep, names=list(_col_def), dtype=_col_def)
    except ValueError:
        # first row was not numeric, re-read assuming the CSV file had a header
        df = pandas.read_csv(csv_name, sep=sep, skiprows=1, names=list(_col_def), dtype=_col_def)
    return df


def mappability_report(filename, kmer_size):
    """
    Prepare a report on per-sequence mappability from a genmap text output.
    The estimate of the unique fraction ignores the last k-1 positions since GenMap always reports zero.

    :param filename: genmap text file
    :param kmer_size: k-mer size used in genmap 'map' analysis
    :return: pandas.DataFrame
    """

    def read_genmap(filename):
        """
        Generator for reading records from the fasta-format genmap text output.
        The first line of each record is the standard FASTA header, while the next contains space-delimited
        floats represent relative uniqueness of the k-mer that begins at that position.

        :param filename: genmap text file
        :return: tuple(seq_id, numpy array of mappability values)
        """
        with open_input(filename, 'rt') as input_h:
            try:
                while True:
                    header = next(input_h).strip()
                    assert header.startswith('>'), f'Invalid genmap text file: {filename}'
                    seq = header[1:].split(' ')[0]
                    data = np.fromiter((float(vi) for vi in next(input_h).split(' ')), dtype=float)
                    yield seq, data
            except StopIteration:
                pass

    map_data = {}
    for seq, data in read_genmap(filename):
        assert seq not in map_data, f'Duplicate sequence id {seq} in {filename}'
        assert len(data) > kmer_size, f'Record for {seq} was shorter than specified k-mer size'
        data = data[:-(kmer_size - 1)]
        uniq_frac = 1 - ((data < 1).sum() / len(data))
        map_data[seq] = [len(data), np.mean(data), np.median(data), uniq_frac]

    logger.debug(f'Read mappability results for {len(map_data):,} sequences')

    map_data = pandas.DataFrame.from_dict(map_data, orient='index', columns=['length','mean','median','uniq_frac'])
    map_data.index.name = 'name'
    return map_data


class SignificantLinks(object):

    def __init__(self, contact_map_file, clustering_file, coverage_file,
                 mappability_file, mappability_k,
                 output_basename, seed):

        self.contact_map_file = contact_map_file
        self.clustering_file = clustering_file
        self.coverage_file = coverage_file
        self.mappability_file = mappability_file
        self.mappability_k = mappability_k
        self.output_basename = output_basename
        self.seed = seed
        # find the R script within this current package folder
        self.find_significant_function_r = SignificantLinks._source_r_function('model_fit.R', 'find_significant')
        self.seq2cl_graph = None
        self.all_contacts = None
        self.spurious = None
        self.spurious_model = None
        self.fit_summary = None
        self.fitted = None

    @staticmethod
    def _source_r_function(r_script, func_name):
        """
        Find the R script within this current package folder
        :param r_script: the R scripo to source
        :param func_name: the name of the function to return
        :return: callable R object function
        """
        with localconverter(robjects.default_converter):
            robjects.r.source(os.path.join(os.path.dirname(os.path.abspath(__file__)), r_script))
            return robjects.globalenv[func_name]

    def create_seq2cluster_graph(self, sep=',', min_seq_length=5000):
        """
        Create the bipartite graph between sequences and genome_bins (clusters).

        :param sep: variable separator for csv file
        :param min_seq_length: minimum length for a sequence to be considered
        :return:
        """
        # load bin3C objects
        contact_map = load_object(self.contact_map_file)
        clustering = load_object(self.clustering_file)

        logger.info('Extracting coverage info')
        coverage_info = robust_read_csv(self.coverage_file, sep=sep).set_index('name')

        mappability_info = mappability_report(self.mappability_file, self.mappability_k)

        logger.info('Creating bipartite graph between clusters and sequences')
        seq2cl_graph = create_seq2cluster_graph(contact_map, clustering, coverage_info,
                                                mappability_info, min_seq_length)

        logger.info('Initial graph info: nodes={:,}, edges={:,}'.format(
            seq2cl_graph.order(), seq2cl_graph.size()))

        isolated_nodes = list(nx.isolates(seq2cl_graph))
        iso_count = Counter(u[0] for u in isolated_nodes)
        logger.info('Of {:,} isolated nodes, {:,} are clusters and {:,} are sequences'.format(
            sum(iso_count.values()), iso_count['c'], iso_count['n']))
        # remove the isolates that provide nothing to this analysis
        seq2cl_graph.remove_nodes_from(isolated_nodes)
        logger.info('After isolate removal: nodes={:,}, edges={:,}'.format(
            seq2cl_graph.order(), seq2cl_graph.size()))
        con_count = Counter(u[0] for u in seq2cl_graph.nodes())

        logger.info('Bipartite graph is composed of {:,} clusters and {:,} sequences'.format(
            con_count['c'], con_count['n']))

        self.seq2cl_graph = seq2cl_graph

    def graph_to_table(self):
        """
        Create the inter-genome_bin observation table for model fitting.
        :return: complete unfiltered data frame
        """

        # initialise an array long enough to hold everything, even though we may not fill it up
        _dtype = np.dtype([('seq', np.object_),
                           ('cluster', 'i4'),
                           ('size_v', 'i4'),
                           ('contacts', 'i4'),
                           ('length_u', 'i4'),
                           ('length_v', 'i4'),
                           ('cov_u', 'i4'),
                           ('cov_v', 'i4'),
                           ('sites_u', 'i4'),
                           ('sites_v', 'i4'),
                           ('gc_u', 'f4'),
                           ('gc_v', 'f4'),
                           ('uf_u', 'f4'),
                           ('uf_v', 'f4'),
                           ('intra', 'bool')])

        # cps_max, lps_max = calculate_rejection_thresholds(seq2cl_graph)
        seq2cl_graph = self.seq2cl_graph
        node_to_cluster = []

        # iterate over the sequence nodes
        seq_nodes = [u for u in seq2cl_graph.nodes() if u[0] == 'n']
        for u in tqdm.tqdm(seq_nodes):
            # every neighbour of a sequence will be a cluster
            for v in seq2cl_graph.neighbors(u):
                u_dat = seq2cl_graph.nodes[u]
                v_dat = seq2cl_graph.nodes[v]
                assert seq2cl_graph.has_edge(u, v), 'missing neighbor edge -- should not happen'
                # skip associations for excluded clusters
                node_to_cluster.append((u[1],
                                        v[1],
                                        v_dat['size'],
                                        seq2cl_graph[u][v]['contacts'],
                                        u_dat['length'],
                                        v_dat['length'],
                                        u_dat['coverage'],
                                        v_dat['coverage'],
                                        u_dat['sites'],
                                        v_dat['sites'],
                                        u_dat['gc'],
                                        v_dat['gc'],
                                        u_dat['uniq_frac'],
                                        v_dat['uniq_frac'],
                                        u_dat['membership'] == v[1]))

        # make this a dataframe for easy manipulation
        node_to_cluster = np.array(node_to_cluster, dtype=_dtype)
        node_to_cluster = pandas.DataFrame(node_to_cluster)
        logger.info(f'Sequence to cluster table contains {len(node_to_cluster):,} observations')

        # write table of all observations
        node_to_cluster.to_csv('{}_all.csv'.format(self.output_basename))
        self.all_contacts = node_to_cluster

    def outlier_removal(self, initial_sigma=3, pr_cutoff=0.0004, n_samples=10000, plot=True):
        """
        Outlier removal aimed at decreasing the FPR within the spurious data table. FPR in this
        case are interactions that are actually non-spurious. These are cases where a sequence
        has true proximity interactions with a genome bin, but was not a member. Sources of this
        include: genome_bin splitting, mobile elements, and conserved regions too confounding to be
        clustered.

        :param initial_sigma: sigma-clipping threshold used in stage 1
        :param pr_cutoff: probability threshold used in stage 2
        :param n_samples: number of samples to use in stage 2 model fitting.
        :return: filtered table
        """
        # plots limited to sensible number of points
        _max_points = 2000

        # outlier filtering will use 4 statistics.
        df = self.spurious.assign(cpcc=lambda x: x.contacts / (x.cov_u.astype('f4') * x.cov_v.astype('f4')),
                                  cpss=lambda x: x.contacts / (x.sites_u.astype('f4') * x.sites_v.astype('f4')),
                                  cps=lambda x: x.contacts / x.sites_u.astype('f4'),
                                  lps=lambda x: x.length_u / x.sites_u.astype('f4'))

        _clip_cols = ['cpcc', 'cpss', 'cps', 'lps']
        if len(df) < n_samples:
            logger.warning('Outlier filtering will use the entire table as sample size '
                           'exceeds table size. {:,} > {:,}'.format(n_samples, len(df)))
            n_samples = len(df)
            clip_data = df[_clip_cols]
            use_all = True
        else:
            clip_data = df[_clip_cols].sample(n_samples, random_state=self.seed)
            use_all = False

        # plot will use n_points
        n_points = n_samples if n_samples < _max_points else _max_points

        if plot:
            # plot initial scatter and kde
            df_plot = np.log(clip_data)
            g = sb.PairGrid(df_plot, diag_sharey=False)
            g.map_upper(sb.scatterplot, s=15)
            g.map_lower(sb.kdeplot)
            g.map_diag(sb.kdeplot, lw=2)
            g.savefig('{}_outlier_stage0.png'.format(self.output_basename))

        # stage 1: remove outliers using sigma clipping to lessen the impact of
        # significant outliers on stage 2.
        clip_data = np.log(np.ma.compress_rows(sigma_clip(clip_data, axis=0, sigma=initial_sigma)))

        if plot:
            # plot outcome of sigma clipping
            df_plot = pandas.DataFrame(clip_data, columns=_clip_cols)
            if not use_all:
                df_plot = df_plot.sample(n_points, random_state=self.seed)
            g = sb.PairGrid(df_plot, diag_sharey=False)
            g.map_upper(sb.scatterplot, s=15)
            g.map_lower(sb.kdeplot)
            g.map_diag(sb.kdeplot, lw=2)
            g.savefig('{}_outlier_stage1.png'.format(self.output_basename))

        # stage 2: fit a single-component gaussian model to the pre-filtered space
        clip_model = BayesianGaussianMixture(covariance_type='diag', tol=1e-5, random_state=self.seed, max_iter=1000)
        clip_model = clip_model.fit(clip_data)
        logger.info('Outlier model means: {:.3g} {:.3g} {:.3g} {:.3g}'.format(*np.exp(clip_model.means_[0])))
        logger.info('Outlier model variances: {:.3g} {:.3g} {:.3g} {:.3g}'.format(*np.exp(clip_model.covariances_[0])))

        # assign probabilities and use them for rejection.
        df['clip_logpr'] = clip_model.score_samples(np.log(df[_clip_cols]))
        _outlier_cutoff = np.log(pr_cutoff)
        df = df.query('clip_logpr > @_outlier_cutoff')

        if plot:
            # plot final outcome
            df_plot = df[_clip_cols]
            if not use_all:
                df_plot = df_plot.sample(n_points, random_state=self.seed)
            # outcome of stage 2
            g = sb.PairGrid(np.log(df_plot), diag_sharey=False)
            g.map_upper(sb.scatterplot, s=15)
            g.map_lower(sb.kdeplot)
            g.map_diag(sb.kdeplot, lw=2)
            g.savefig('{}_outlier_stage2.png'.format(self.output_basename))

        self.spurious = df
        logger.info('After outlier filtering, {:,} observations passed'.format(len(self.spurious)))

    def create_spurious_table(self, excluded_clusters=None, excluded_sequences=None,
                              min_bin_size=5, min_bin_length=100000,
                              min_single_length=1000000, big_threshold=0.8, small_value=1):
        """
        Create a table of inter-genome_bin interactions which are likely to be spurious. Excluded
        objects (clusters/sequences) are those which are likely to introduce false positives.

        :param excluded_clusters: problematic clusters to exclude
        :param excluded_sequences: problematic sequences to exclude
        :param min_bin_size:
        :param min_bin_length:
        :param min_single_length:
        :param big_threshold:
        :param small_value:
        :return:
        """

        # begin with a copy, since we will make some inplace changes
        spurious = self.all_contacts.copy()

        if excluded_clusters is not None and len(excluded_clusters) > 0:
            assert isinstance(excluded_clusters, list), 'excluded_clusters must be a list'
            n_before = len(spurious)
            spurious = spurious.query('cluster not in @excluded_clusters').reset_index(drop=True)
            logger.info(f'Excluded clusters removed {n_before - len(spurious):,} observations')

        if excluded_sequences is not None and len(excluded_sequences) > 0:
            assert isinstance(excluded_sequences, list), 'excluded_sequences must be a list'
            n_before = len(spurious)
            spurious = spurious.query('seq not in @excluded_sequences').reset_index(drop=True)
            logger.info(f'Excluded sequences removed {n_before - len(spurious):,} observations')

        # Replace zeros with a small value when they occur in certain columns
        # TODO check why Flye reports zero coverage occasionally. Should we be dropping them instead?
        fill_zeros(spurious, ['cov_v', 'cov_u', 'sites_v', 'sites_u'], small_value)

        # initial mask accepts all interactions
        ix_accepted = pandas.Series(np.ones(len(spurious), dtype='bool'))

        if big_threshold is not None:
            ix_big_member = 1 / big_threshold * spurious['length_u'] > spurious['length_v']
            logger.info('{:,} observations involved a sequence over {:.1f}% of total bin extent'.format(
                ix_big_member.sum(), big_threshold * 100))
            ix_accepted &= ~ix_big_member

        ix_short_bin = spurious['length_v'] < min_bin_length
        logger.info('{:,} observations involved a bin of very short extent (<{:,} bp)'.format(
            ix_short_bin.sum(), min_bin_length))
        ix_accepted &= ~ix_short_bin

        ix_small_bin = spurious['size_v'] < min_bin_size
        logger.info('{:,} observations involved a bin with very few members (<{} members)'.format(
            ix_small_bin.sum(), min_bin_size))
        ix_accepted &= ~ix_small_bin

        # exclude single member bins unless their length is a significant portion of a microbial genome
        ix_small_single_bins = (spurious.size_v == 1) & (spurious.length_v < min_single_length)
        logger.info('{:,} observations involved singleton bins of small extent (<{:,} bp)'.format(
            ix_small_single_bins.sum(), min_single_length))
        ix_accepted &= ~ix_small_single_bins

        # This last threshold can result in the exclusion of single-sequence bins.
        if min_bin_size > 1:
            logger.warning('If long-read data, excluding bins with less than {} sequences '
                           'might eliminate nearly complete genomes.'.format(min_bin_size))

        # exclude interactions which are not marked as inter-genome_bin
        logger.info('{:,} observations were marked as intra-genome_bin'.format(spurious['intra'].sum()))
        ix_accepted &= ~spurious['intra']

        spurious = spurious[ix_accepted]
        spurious.to_csv('{}_inter.csv'.format(self.output_basename))
        logger.info('After basic rejections, {:,} observations passed'.format(len(spurious)))

        self.spurious = spurious

    def estimate_significance_model(self, n_samples=10000,
                                    distrib_func='nbinom2',
                                    fixed_model='contacts1m ~ sites_z * cov_z',
                                    disp_model='~ sites_z / cov_z',
                                    zi_model='~ cov_z'):
        """
        For a table of contig-to-genome_bin interactions.

        Estimate the parameters of a Zero-Inflated Negative Binomial model for the interactions
        involving contigs and genome_bins for which the contigs are suspected of _not_ belonging.
        The dominant form of these interactions _should_ be spurious. To maximise the proportion,
        outlier rejection is used to remove strong interactions -- identified as those where
        contacts/(site_u*site_v) or contacts/(cov_u*cov_v) is large.

        Exogenous variables are the product value pairs for the variables: length, sites, coverage,
        and gc. Here, each pair is made up of one contig (u) and one genome_bin (v). These products
        are log transformed and standardised. e.g. scale(log(length_u * length_v))

        Exogenous variables: length_z, sites_z, coverage_z, gc_z
        Endogenous variable is: contacts - 1

        After fitting the model to selected "non-local/spurious" observations, the model is then used to
        predict responses for all observed interactions. Comparison of model predictions to actual values
        is then used to assign p-values that the observed interaction is non-local/spurious.

        The modelling is current carried out in R using the glmmTMB package, but could potentially
        be done using statsmodels.

        Distribution family choices include: nbinom1 or nbinom2 (negative binomial p=1|2),
        genpois (Generalised), compois (Conway-Maxwell). In experimenting, we have found the most
        applicable distributions are nbinom2 and compois. Although consistenly producing superior
        AIC, BIC and AICc, the Conway-Maxwell distribution is expensive to calculate. Therefore
        users should be prepared to wait significantly longer for modelling to complete or
        reduce the number of points supplied. For the simple model of 3 conditional parameters,
        10k points appears to be more than sufficient.

        :param n_samples: the number of samples to use in model estimation
        :param distrib_func: distribution family to use in model (eg. nbinom2, genpois, compois)
        :param fixed_model: custom fixed-effects model for R
        :param disp_model: custom dispersion model for R
        :param zi_model: custom zero-inflation model for R
        """
        def _drop_stale_columns(*dataframes):
            """
            On multiple calls to fit routine, in-place remove the return columns
            :param dataframes: a list of dataframes to alter
            """
            for df in dataframes:
                df.drop(columns=[
                    'contacts1m', 'length_z', 'sites_z', 'cov_z',
                    'gc_z', 'uf_z', 'response', 'pvalue'], errors='ignore', inplace=True)

        logger.info('Total observation pool: {:,}'.format(len(self.spurious)))
        assert len(self.spurious.query('intra == True')) == 0, 'There are intra-genome_bin entries in the table'

        fitting = self.spurious.query('sites_u > 0 and sites_v > 0 and cov_u > 0 and cov_v > 0')
        logger.info('After zero removal: {:,}'.format(len(fitting)))

        if n_samples > len(fitting):
            logger.warning('Significance model will use the entire table as sample size '
                           'exceeds table size. {:,} > {:,}'.format(n_samples, len(fitting)))
            n_samples = len(fitting)
        fitting = fitting.sample(n_samples, random_state=self.seed)

        with localconverter(robjects.default_converter + pandas2ri.converter) as cv:
            logger.info('Converting pandas table to R')
            _drop_stale_columns(fitting, self.all_contacts)
            fitting = cv.py2rpy(fitting)
            df_all = cv.py2rpy(self.all_contacts)

        with localconverter(robjects.default_converter):
            logger.info('Calling R method')
            ret_r = self.find_significant_function_r(fitting, df_all,
                                                     fixed_model=fixed_model,
                                                     disp_model=disp_model,
                                                     zi_model=zi_model,
                                                     output_path=self.output_basename,
                                                     distrib_func=distrib_func,
                                                     seed=self.seed)

            # extract various return information from the R objects
            # ANOVA result
            anova = rmatrix2pandas(ret_r.rx2['anova'])
            anova['Df'] = pandas.to_numeric(anova['Df'], downcast='unsigned')

            # Build summary information dictionary
            summary = {'AIC': ret_r.rx2['AIC'][0],
                       'BIC': ret_r.rx2['BIC'][0],
                       'logLik': ret_r.rx2['logLik'][0],
                       'coeffs': rvector2dict(ret_r.rx2['fixef']),
                       'sigma': ret_r.rx2['sigma'][0],
                       'anova': anova}

            self.fit_summary = summary

        with localconverter(robjects.default_converter + pandas2ri.converter):
            # Convert the returned tables from R back into pandas
            # these tables now have extra columns pertaining to:
            # significance: 'pvalue, response'
            # scaled exogenous: 'length_z, sites_z, cov_z, gc_z'
            # endogenous: contacts1m (contacts - 1)
            self.fitted = robjects.conversion.rpy2py(ret_r.rx2('fitted'))
            self.all_contacts = robjects.conversion.rpy2py(ret_r.rx2('all_contacts'))
            logger.info("Significance testing was computed for {:,} observations".format(len(self.all_contacts)))

    def fdr_correction(self, pr_signif=0.05):
        """
        Apply Benjamini-Hochberge false-discovery rate correction. The adjusted p-values will
        appear as a new column `adj_pvalue` in the all_contacts table.

        Currently two-step BH is used.

        :param pr_signif: target significance level.
        """
        logger.info('Performing FDR correction')

        self.all_contacts['adj_pvalue'] = multipletests(
            self.all_contacts.pvalue, alpha=pr_signif, method='fdr_tsbh')[1]

        n_signif = len(self.all_contacts.query('adj_pvalue < 0.05'))
        logger.info('Using adjusted p-values there were {:,} significant interactions ({:.1f}%)'.format(
            n_signif, n_signif / len(self.all_contacts)))

    def prepare_data(self, sep=',', excluded_clusters=None, excluded_sequences=None,
                     min_seq_length=5000, min_bin_size=5, min_bin_length=100000,
                     min_single_length=1000000, big_threshold=0.8,
                     initial_sigma=3, pr_cutoff=0.0004, n_samples=10000, small_value=1,
                     plot_outliers=False):
        """
        From the Hi-C dataset, prepare the input data for significance testing

        :param sep: variable separator for coverage csv file
        :param excluded_clusters: list of clusters to exclude from analysis
        :param excluded_sequences: list of sequences to exclude from analysis
        :param min_seq_length: minimum sequence length to be considered
        :param min_bin_size: minimum bin size (number of sequences) to be considered
        :param min_bin_length: minimum bin length (total sum of seq lengths) to be considered
        :param min_single_length: minimum length of a single-sequence bin to be considered
        :param big_threshold: maximum fraction a sequence to represent for a bin to be considered
        :param initial_sigma: sigma clipping sigma used in outlier removal stage-1
        :param pr_cutoff: probability minimum for outlier stage-2
        :param n_samples: number of samples to use in fitting model
        :param small_value: small value for replacing observed zeros
        :param plot_outliers: generate diagnostic plots from outlier removal
        """
        self.create_seq2cluster_graph(sep=sep, min_seq_length=min_seq_length)
        self.graph_to_table()
        self.create_spurious_table(excluded_clusters=excluded_clusters, excluded_sequences=excluded_sequences,
                                   min_bin_size=min_bin_size, min_bin_length=min_bin_length,
                                   min_single_length=min_single_length, big_threshold=big_threshold,
                                   small_value=small_value)
        self.outlier_removal(initial_sigma, pr_cutoff, n_samples, plot=plot_outliers)

    def fit_model(self, n_samples=10000, pr_signif=0.05, fixed_model=None, disp_model=None, zif_model=None):
        """
        Using the prepared data, fit the Zinb model and adjust
        the resulting p-values for FDR.

        :param n_samples: the number of samples to use in fitting
        :param pr_signif: the threshold probability at which to control FDR
        :param fixed_model: custom fixed-effects model for R
        :param disp_model: custom dispersion model for R
        :param zif_model: custom zero-inflation model for R
        """
        self.estimate_significance_model(n_samples,
                                         fixed_model=fixed_model,
                                         disp_model=disp_model,
                                         zi_model=zif_model)
        self.fdr_correction(pr_signif)

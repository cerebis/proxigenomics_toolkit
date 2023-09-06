from ..misc_utils import package_path, make_dir
from ..seq_utils import IndexedFasta
from ..linalg import kr_biostochastic
from ..exceptions import *
from .contact_map import SeqOrder
from collections import defaultdict
from gfa_io import GFA
from copy import deepcopy
import Bio.SeqIO as SeqIO
import Bio.SeqUtils as SeqUtils
import contextlib
import logging
import networkx as nx
import numpy as np
import os
import pandas
import pysam
import re
import scipy.sparse as sp
import subprocess
import tqdm
import warnings

logger = logging.getLogger(__name__)

SPADES_PATTERN = re.compile(r'NODE_\d+_length_\d+_cov_(\d+\.\d*)')
MEGAHIT_PATTERN = re.compile(r'.*?multi=(\d+\.\d*).*?')
FLYE_PATTERN = re.compile(r'.*dp:(\d+).*')


def coverage_data_extractor(cov_data):
    """
    For supplied coverage data, return values from the DataFrame object
    :param cov_data: pandas DataFrame indexed on seq_id
    :return: extractor instance
    """
    def _extractor(seq_record):
        """
        :param seq_record: Bio.SeqRecord instance
        :return: float coverage
        """
        try:
            return cov_data.loc[seq_record.id].values[0]
        except KeyError:
            raise NotFoundException('In supplied coverage data, sequence_id', seq_record.id)
    return _extractor


def spades_extractor(seq_record):
    """
    For SPAdes assemblies, we can extract the coverage statistic contained in the contig name.
    :param seq_record: Bio.SeqRecord instance
    :return: float coverage
    """
    m = SPADES_PATTERN.match(seq_record.name)
    if m is None:
        raise InvalidCoverageFormatError(seq_record.name, 'spades_extractor', seq_record.name)
    return float(m.group(1))


def megahit_extractor(seq_record):
    """
    For Megahit assemblies, we can extract the coverage statistic contained in the contig description.
    :param seq_record: Bio.SeqRecord instance
    :return: float coverage
    """
    m = MEGAHIT_PATTERN.match(seq_record.description)
    if m is None:
        raise InvalidCoverageFormatError(seq_record.name, 'megahit_extractor', seq_record.description)
    return float(m.group(1))


def flye_extractor(seq_record):
    """
    For Flye assemblies, we can extract the coverage statistic contained in the contig description.
    Note: older versions of Flye did not include this information in the contig description.
    :param seq_record: Bio.SeqRecord instance
    :return: float coverage
    """
    m = FLYE_PATTERN.match(seq_record.description)
    if m is None:
        raise InvalidCoverageFormatError(seq_record.name, 'flye_extractor', seq_record.description)
    return float(m.group(1))


def add_cluster_names(clustering, prefix='CL'):
    """
    Add sequential names beginning from 1 to a clustering in-place.

    A pedantic determination of how many digits are required for the
    largest cluster number is performed, so names will sort conveniently
    in alphanumeric order and align to the eye in output information.

    :param clustering: clustering solution returned from ContactMap.cluster_map
    :param prefix: static prefix of cluster names.
    """
    try:
        num_width = max(1, int(np.ceil(np.log10(max(clustering)+1))))
    except OverflowError:
        num_width = 1

    for cl_id in clustering:
        # names will 1-based
        clustering[cl_id]['name'] = '{0}{1:0{2}d}'.format(prefix, cl_id+1, num_width)


def bistochastic_graph(g):
    _adj_mat = sp.csr_matrix(nx.adjacency_matrix(g))
    _adj_mat, _ = kr_biostochastic(_adj_mat)
    _adj_mat = _adj_mat.tocoo()
    name_lookup = list(g.nodes())
    g_out = nx.Graph()
    for i, j, d in zip(_adj_mat.row, _adj_mat.col, _adj_mat.data):
        g_out.add_edge(name_lookup[i], name_lookup[j], weight=d)
    return g_out


def cluster_map(contact_map, seed, work_dir='.', n_iter=None,
                 exclude_names=None, norm_method='sites', append_singletons=True,
                 from_extent=False, fdr_alpha=0.05, use_entropy=False, gfa_file=None,
                 markov_scale=None):
    """
    Cluster a contact map into groups, as an approximate proxy for "species" bins.

    :param contact_map: an instance of ContactMap to cluster
    :param seed: a random seed
    :param work_dir: working directory to which files are written during clustering
    :param n_iter: for method supporting iterations, specify a non-default number
    :param exclude_names: a collection of sequence identifiers to exclude when clustering
    :param norm_method: noramlisation method to apply to contact map
    :param append_singletons: append additional clusters for isolated sequences not subjected
    to clustering.
    :param from_extent: use the extent of the contact map to determine the number of clusters
    :param fdr_alpha: FDR alpha used in gothic normalisation
    :param use_entropy: enable entropy correction within infomap clustering
    :param gfa_file: path to a GFA file, if supplied, multilayered clustering will be performed
    :param markov_scale: adjust scale of markov time in Infomap clustering (default: 1.0)
    :return: a dictionary detailing the full clustering of the contact map
    """

    def _read_tree(pathname):
        """
        Read a tree clustering file as output by Infomap.

        :param pathname: the path to the tree file
        :return: dict of cluster_id to array of seq_ids
        """
        with open(pathname, 'r') as in_h:
            cl_map = defaultdict(list)
            for line in in_h:
                line = line.strip()
                if not line:
                    break
                if line.startswith('#'):
                    continue
                fields = line.split()
                # the cluster is identified by the first N-1 terms in the label (: delimited)
                _cl_id = fields[0][:fields[0].rindex(':')]
                # this takes the name
                cl_map[_cl_id].append(fields[2].strip('"'))

        # order clusters by descending size, then create a basic map on ascending integers
        cl_map = sorted([(len(cl_map[_id]), np.array(cl_map[_id])) for _id in cl_map],
                        key=lambda x: x[0],
                        reverse=True)
        return {n: _seqs for n, (_size, _seqs) in enumerate(cl_map)}

    assert os.path.exists(work_dir), 'supplied output path [{}] does not exist'.format(work_dir)

    # build a graph of the complete map
    base_name = 'cm_graph'
    g = to_graph(contact_map, node_id_type='external', norm=True, bisto=True, scale=True,
                 exclude_names=exclude_names, norm_method=norm_method,
                 from_extent=from_extent, fdr_alpha=fdr_alpha)

    logger.info('Clustering contact graph using method: infomap')

    try:
        with open(os.path.join(work_dir, 'infomap.log'), 'w+') as stdout:
            edge_file = os.path.join(work_dir, f'{base_name}.pajek')
            if gfa_file is not None:
                logger.info('GFA file supplied, preparing for multilayered clustering')
                g_gfa = read_gfa(gfa_file, paths_to_edges=True, read_progress=True)
                g_gfa = bistochastic_graph(g_gfa)
                layers = [g, g_gfa]
                write_multilayer_pajek(edge_file, layers, add_inter=False)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    nx.write_pajek(g, edge_file)

            # Infomap v2.7.1 interface
            options = ['--flow-model', 'undirected', '--verbose',
                       '--core-loop-codelength-threshold', '1e-20',
                       '--tune-iteration-relative-threshold', '1e-10',
                       '-M', '50',
                       '--seed', str(seed)]

            if use_entropy:
                options.append('--entropy-corrected')

            if markov_scale is not None:
                options.extend(['--markov-time', str(markov_scale)])

            if n_iter is None:
                n_iter = 10
            options.extend(['-N', str(n_iter)])

            exe_path = package_path('external', 'Infomap')
            subprocess.check_call([exe_path] + options + [edge_file, work_dir],
                                  stdout=stdout, stderr=subprocess.STDOUT)

            cl_to_ids = _read_tree(os.path.join(work_dir, f'{base_name}.tree'))

    except OSError as e:
        logger.error(f'An error occurred starting the child process. Helper path was: {exe_path}')
        raise e

    logger.info(f'Clustering resulted in {len(cl_to_ids)} clusters')

    # the contact map does not contain information about sequences that were rejected
    # for short length during initial map construction.
    name2id = contact_map.make_reverse_index('name')

    clustering = {}
    for cl_id, _seqs in cl_to_ids.items():

        known_ids = []
        unref = []
        for sn in _seqs:
            if sn in name2id:
                known_ids.append(name2id[sn])
            else:
                unref.append(sn)

        # TODO clustering should be made class, as this dict approach is
        #   too ad-hoc and error prone.
        clustering[cl_id] = {
            'names': np.array(_seqs),
            'seq_ids': np.array(known_ids),
            'extent': contact_map.order.lengths()[known_ids].sum(),
            'status': 'primary',
            'unreferenced': unref,
            'deduplicated': []
        }

    # append singletons
    if append_singletons:
        lost_seq = find_lost_singletons(contact_map, clustering)
        for n, _seq in enumerate(lost_seq):
            # use simple temporary names prior to reassignment
            cl_id = f'tmp_{n}'
            assert cl_id not in clustering, f'Tried to overwrite existing cluster id: {cl_id}'
            clustering[cl_id] = {
                'names': np.array([_seq]),
                'seq_ids': np.array([_seq]),
                'extent': contact_map.seq_info[_seq].length,
                'status': 'rescued singleton',
                'unreferenced': [],
                'deduplicated': []
            }

        # sanity check
        # all_seqs = [seq_id for cl in clustering.values() for seq_id in cl['seq_ids']]
        # assert len(all_seqs) == len(set(all_seqs)), 'Duplicate sequences in clustering'
        logger.info('Appended {} isolated sequences as singleton clusters'.format(len(lost_seq)))
        logger.info('Total cluster count now {}'.format(len(clustering)))

    if gfa_file is not None:
        clustering = harden_clustering(clustering, contact_map)

    # assign cluster ids as ascending 0-based integers, clusters ordered by descending extent
    sorted_keys = sorted(clustering, key=lambda k: clustering[k]['extent'], reverse=True)
    clustering = {n: clustering[k] for n, k in enumerate(sorted_keys)}

    add_cluster_names(clustering)

    return clustering


def cluster_report(contact_map, clustering, source_fasta=None, assembler='generic', coverage_file=None):
    """
    For each cluster, analyze the member sequences and build a report.
    Update the clustering dictionary with this result by adding a "report" for each.

    :param contact_map: an instance of ContactMap to cluster
    :param clustering: clustering solution dictionary
    :param source_fasta: source assembly fasta (other than defined at instantiation)
    :param assembler: name of assembly software used in creating contigs
    :param coverage_file: csv file containing coverage information: seq_id,cov
    """

    logger.info('Analyzing the contents of each cluster')

    cov_data = None
    depth_extractor = None
    if coverage_file is not None:
        cov_data = pandas.read_csv(coverage_file, sep=',', index_col=0, skip_blank_lines=True,
                                   header=None, comment='#')
        logger.info('Coverage file contained {} records'.format(len(cov_data)))
        depth_extractor = coverage_data_extractor(cov_data)
        logger.info('Supplied coverage file {} will override any information stored with input sequences'
                    .format(coverage_file))
    else:
        if assembler == 'spades':
            depth_extractor = spades_extractor
        elif assembler == 'megahit':
            depth_extractor = megahit_extractor
        elif assembler == 'flye':
            depth_extractor = flye_extractor
        if depth_extractor is not None:
            logger.info('Contig FASTA headers are presumed follow {}\'s format'.format(assembler))

    if depth_extractor is None:
        logger.info('No assembler or coverage file was supplied, '
                    'therefore coverage information will not be available in report')

    # seq_info = contact_map.seq_info

    if source_fasta is None:
        source_fasta = contact_map.seq_file

    # set up indexed access to the input fasta
    logger.info('Building random access index for input FASTA sequences')
    with contextlib.closing(IndexedFasta(source_fasta)) as seq_db:
        # iterate over the cluster set, in the existing order
        for cl_id, cl_info in tqdm.tqdm(clustering.items(), total=len(clustering),
                                        desc='inspecting clusters'):
            _len = []
            _cov = []
            _gc = []
            _missing_gc = 0
            for _seq_id in cl_info['seq_ids']:

                _sinfo = contact_map.seq_info[_seq_id]

                # get the sequence's external name and length
                _name = _sinfo.name

                # fetch the SeqRecord object from the input fasta
                _seq = seq_db[_name]
                # sanity check for incorrect supplied reference fasta
                assert len(_seq) == _sinfo.length, \
                    f'source fasta {source_fasta} does not match information stored in contact_map for {_name}'
                _len.append(_sinfo.length)

                try:
                    _gc_i = _sinfo.gc
                except IndexError:
                    _missing_gc += 1
                    # handle older contact maps without GC data in seq_info
                    _gc_i = SeqUtils.GC(_seq.seq)
                _gc.append(_gc_i)

                if depth_extractor is not None:
                    _cov.append(depth_extractor(_seq))

            if _missing_gc > 0:
                logger.warning('GC values missing from {} records were recovered from FASTA sequences'.format(_missing_gc))

            if depth_extractor is None:
                assert len(_len) == len(_gc), \
                    'There were missing records while collecting per-contig statistics (GC, length). ' \
                    'Did you declare the correct assembler?'
            else:
                assert len(_len) == len(_cov) == len(_gc), \
                    'There were missing records while collecting per-contig statistics (GC, length and coverage). ' \
                    'Did you declare the correct assembler?'

            if len(_cov) > 0:
                report = np.fromiter(zip(_len, _gc, _cov),
                                     dtype=[('length', np.int64),
                                            ('gc', np.float64),
                                            ('cov', np.float64)])
            else:
                report = np.fromiter(zip(_len, _gc),
                                     dtype=[('length', np.int64),
                                            ('gc', np.float64)])

            clustering[cl_id]['report'] = report


def revise_clusters(target_clusters, contact_map, clustering: dict, algorithm=None, from_extent=False,
                    norm=True, bisto=True, scale=True, norm_method='sites', only_new=False):
    """
    Subject a list of target clusters to a new round of community detection. Each cluster is
    extracted as a subgraph and then clustered. Expected methods return type is a list of
    partitions, each containing the collection of member sequences. networkx.community
    algorithms fit this return type. leidenalg is another alternative.

    :param target_clusters: 0-based cluster ids
    :param contact_map: the relevant contact map
    :param clustering: the relevant clustering solution
    :param algorithm: specific algorithm returnning a list of community partitions
    :param from_extent: use the extent map to define the subgraph
    :param norm_method: normalisation method to apply to contact map
    :return:
    """
    if algorithm is None:
        algorithm = nx.community.label_propagation_communities

    # prepare the contact map's complete graph
    g_complete = to_graph(contact_map, node_id_type='external', norm=norm, bisto=bisto, scale=scale,
                          norm_method=norm_method, from_extent=from_extent)

    # begin with the current cluster solution
    revised = clustering.copy()

    # prepare a lookup table of existing sequence details, these will be transfered after revision
    seq2report = {}
    for cl_id in revised:
        assert 'report' in revised[cl_id], f'The clustering solution did not contain a report for {cl_id}'
        for si, ri in zip(revised[cl_id]['seq_ids'], revised[cl_id]['report']):
            seq2report[si] = ri

    # lookup table for switching back to seq_ids
    name2id = contact_map.make_reverse_index('name')

    n_new = 0
    for cl_id in target_clusters:
        # drop the cluster from the complete solution
        cluster = revised.pop(cl_id)
        logger.debug(f'Revising cluster {cl_id}: {len(cluster["seq_ids"])} members, {cluster["extent"]} extent')
        # extract the subgraph
        seq_names = [contact_map.seq_info[_id].name for _id in cluster['seq_ids']]
        g = nx.subgraph(g_complete, seq_names)
        logger.debug(f'Subgraph composition: order {g.order()} size {g.size()}')
        # partition the subgraph into communities
        partitions = algorithm(g)
        logger.debug(f'Cluster {cl_id} was split into {len(partitions)}')
        for members in partitions:
            n_new += 1
            _seqs = np.sort([name2id[nm] for nm in members])
            # negative ints as temporary keys
            revised[-n_new] = {
                'seq_ids': _seqs,
                'extent': contact_map.order.lengths()[_seqs].sum(),
                'status': 'revised',
                'report': np.squeeze(np.vstack([seq2report[_si] for _si in _seqs]))
            }

    if only_new:
        cl_list = list(revised.keys())
        for cl_id in cl_list:
            # TODO commented logic protects against older clustering objects without
            #      a status field. As we typically fail at maintaining backwards
            #      compatibility, this is probably not necessary.
            # if 'status' not in revised[cl_id] or revised[cl_id]['status'] != 'revised':
            if revised[cl_id]['status'] != 'revised':
                del revised[cl_id]

    # clusters will now be ordered/keyed by descending extent
    # TODO this might be confusing if work has already been done on an existing solution
    sorted_keys = sorted(revised, key=lambda k: revised[k]['extent'], reverse=True)
    revised = {n: revised[k] for n, k in enumerate(sorted_keys)}

    add_cluster_names(revised)

    return revised


def to_graph(contact_map, norm=True, bisto=False, scale=False, node_id_type='internal',
             clustering=None, cl_list=None, exclude_names=None, norm_method='sites',
             filter_weak_edges=False, from_extent=False, fdr_alpha=0.05):
    """
    Convert the seq_map to an undirected Networkx Graph.

    The contact map is used as an adjacency matrix, where sequences/contigs are the nodes and Hi-C interactions
    become weighted edges. Edge weigths are dictated by normalisation choices (bisto, norm, scale).

    Explaining node id type:

    - internal: the contiguous set of indices of the subspace map (after filtering) are used as node ids.
    - external: the sequence ids of the input fasta files are used a node ids.

    Depending on the intended use, how best to set node ids varies.

    Internally, when clustering the map, node ids which reference the subspace being clustered are the most convenient
    as they are a contiguous series and some clustering algorithms expect a contiguous series of integer identifiers.
    You must remember to convert results back into full map indices when finished.

    However, when interrogating interactions between nodes in other ways, ids which reference the entire contact map
    can be handier. This is because algorithms may require access to other features of the sequences, such as
    cluster membership, length, gc, etc. Data structures which store these additional features are indexed in the same
    way as the full contact map.

    Lastly there may be "external" uses of the graph representation of the contact map, where the original sequence ids
    are the most usedful.

    :param contact_map: an instance of ContactMap to cluster
    :param norm: normalize weights by length
    :param bisto: normalise using bistochasticity
    :param scale: scale weights (max_w = 1)
    :param node_id_type: select the node id type (filtered, complete, external)
    :param clustering: bin3C clustering solution, required for subsets of the total map
    :param cl_list: list of clusters to include in the graph (0-based internal ids)
    :param exclude_names: a collection of sequences (by external name) to exclude when clustering
    :param norm_method: normalisation method to apply to contact map
    :param filter_weak_edges: remove edges with weight in the bottem 5% of the distribution
    :param from_extent: normalised extent map acts as the basis for sequence map.
    :param fdr_alpha: FDR alpha used in gothic normalisation
    :return: graph of contigs
    """

    def make_gapless_lookup(iterable):
        """ create a lookup of gapless index to some corresponding set of values """
        return dict(zip(contact_map.order.gapless_positions(), iterable))

    def _ix_to_seqid(ix_sub):
        """ return external sequence name/id """
        return contact_map.seq_info[_to_seqid[ix_sub]].name

    def _ix_to_self(ix_sub):
        """ return the same value """
        return ix_sub

    type_map = {'external': _ix_to_seqid,
                'internal': _ix_to_self}
    try:
        _nn = type_map[node_id_type]
    except KeyError:
        raise ApplicationException('unknown node_id_type {}'.format(node_id_type))

    def _attr_basic(ix):
        return {'length': contact_map.seq_info[_to_seqid[ix]].length}

    def _attr_with_cov(ix):
        return {'cluster': _to_clid[ix],
                'length': int(_to_report[ix]['length']),
                'gc': float(_to_report[ix]['gc']),
                'cov': float(_to_report[ix]['cov'])}

    def _attr_without_cov(ix):
        return {'cluster': _to_clid[ix],
                'length': int(_to_report[ix]['length']),
                'gc': float(_to_report[ix]['gc'])}

    report_map = {True: _attr_with_cov,
                  False: _attr_without_cov}

    def create_graph(_map):
        """
        Build the graph, either rapidly or more slowly with additional attributes.
        ** Note:  ** this function makes use of local scope rather than pass parameters.

        :param _map: the contact map to convert to a graph
        :return: a Networkx Graph based on the supplied map
        """
        g = nx.Graph(name='contact_graph')
        # being a symmetric matrix, only upper/lower triangle is needed
        g.add_nodes_from((_nn(i), _node_attr(i)) for i in range(_map.shape[0]))
        # add edges (symmetric matrix and undirected graph)
        _triu = sp.triu(_map)
        for i, j, w in tqdm.tqdm(zip(_triu.row, _triu.col, _triu.data), desc='adding edges', total=_triu.nnz):
            g.add_edge(_nn(i), _nn(j), weight=float(w * scl))

        return g
    
    if cl_list is not None and not clustering:
        raise ApplicationException('When cl_list is specified, a clustering solution is required')

    contact_map.prepare_seq_map(norm=norm, bisto=bisto, norm_method=norm_method,
                                from_extent=from_extent, fdr_alpha=fdr_alpha)

    _node_attr = _attr_basic
    if cl_list is not None:
        # TODO this is brittle as it checks only the first cluster for presence of 'cov' key.
        if 'report' in clustering[cl_list[0]]:
            _node_attr = report_map['cov' in clustering[0]['report'].dtype.names]
        else:
            logger.warning('As some clusters were missing report information, only basic attributes will be added')

        cl_list = sorted(cl_list)
        cl_list = enable_clusters(contact_map, clustering, cl_list=cl_list, ordered_only=False)

    _map = contact_map.get_subspace(permute=False, marginalise=True, flatten=False)
    logger.info('Graph will have {} nodes'.format(contact_map.order.count_accepted()))

    if not sp.isspmatrix_coo(_map):
        _map = _map.tocoo()

    # if requested prepare a scale factor for maximum edge weight = 1
    scl = 1.0 / _map.max() if scale else 1

    logger.debug('Building graph from edges')
    if cl_list is not None:
        logger.info('Graph will only contain sequences from clusters: {}'.format(cl_list))
        # sub-space to full map index lookup.
        _to_seqid = make_gapless_lookup((_si for _cl in cl_list for _si in clustering[_cl]['seq_ids']))
        if _node_attr != _attr_basic:
            _to_report = make_gapless_lookup((_ri for _cl in cl_list for _ri in clustering[_cl]['report']))
        _to_clid = make_gapless_lookup(
            (clustering[_cl]['name'] for _cl in cl_list for _si in clustering[_cl]['seq_ids']))

        n_expected = len(_to_seqid)
        logger.info('Graph will have {} nodes'.format(n_expected))
        g = create_graph(_map)

    else:
        logger.info('Graph will contain all sequences that passed minimum length and signal criteria')
        # create a lookup from the dense indices of the filtered subspace to
        #   the original (potentially sparse) full map indices.
        _to_seqid = contact_map.order.remap_gapless(np.arange(_map.shape[0]))
        n_expected = contact_map.order.count_accepted()
        g = create_graph(_map)

    assert g.order() == n_expected, \
        f'order(graph) {g.order()} did not equal number of expected sequences {n_expected}'

    # disconnect any node mentioned in exclusion list
    if exclude_names:
        assert node_id_type == 'internal', 'Exclusion only supports graphs using internal node ids'

        logger.info('Before exclusion: {:,} nodes, {:,} edges'.format(g.order(), g.size()))

        # do this one at a time so we can report about missing ids
        name_to_matindex = {contact_map.seq_info[_id].name: _ix for _ix, _id in enumerate(_to_seqid)}
        n_failed = 0
        n_missing = 0
        n_removed_edges = 0
        for _name in exclude_names:
            try:
                u = name_to_matindex[_name]
            except KeyError as ex:
                logger.debug('while excluding nodes from clustering, no sequence named {} found'.format(_name))
                n_missing += 1
                continue
            # pedantically check that this mess of cross-referencing is valid.
            #assert contact_map.seq_info[u].name == _name, 'Conflict between name records'
            if not g.has_node(u):
                logger.debug('disconnecting {} failed as no corresponding node in graph'.format(_name))
                n_failed += 1
                continue
            to_remove = [(u, v) for v in g.neighbors(u)]
            n_removed_edges += len(to_remove)
            g.remove_edges_from(to_remove)
        if n_missing > 0:
            logger.warning('{} sequences mentioned for clustering exclusion were not found'.format(n_missing))
        if n_failed > 0:
            logger.warning('{} nodes were not found during exclusion'.format(n_failed))
        if n_failed == len(exclude_names):
            logger.warning('None of excluded sequences were found in graph. '
                           'Check that you did not mix datasets or garble excluded id list"')
        if n_removed_edges > 0:
            logger.info('Excluded sequences resulted in the deletion of {:,} edges'.format(n_removed_edges))

    if filter_weak_edges:
        min_weight = np.quantile(np.fromiter((d['weight'] for u, v, d in g.edges(data=True)), dtype='float'), q=0.25)
        logger.info('Removing weak edges with weight < {:.3e}'.format(min_weight))
        logger.info('Before weak removal: {:,} nodes, {:,} edges'.format(g.order(), g.size()))
        g.remove_edges_from([(u, v) for u, v, d in g.edges(data=True) if d['weight'] < min_weight])

    logger.info('Final graph: {:,} nodes, {:,} edges'.format(g.order(), g.size()))

    return g


def enable_clusters(contact_map, clustering, cl_list=None, ordered_only=True, min_extent=None):
    """
    Given a clustering and list of cluster ids (or none), enable (unmask) the related sequences in
    the contact map. If a requested cluster has not been ordered, it will be dropped.

    :param contact_map: an instance of ContactMap to cluster
    :param clustering: a clustering solution to the contact map
    :param cl_list: a list of cluster ids to enable or None (all ordered clusters)
    :param ordered_only: include only clusters which have been ordered
    :param min_extent: include only clusters whose total extent is greater
    :return: the filtered list of cluster ids in ascending numerical order
    """

    # start with all clusters if unspecified
    if cl_list is None:
        cl_list = list(clustering.keys())

    # at present, only primary clusters are supported in downstream methods
    cl_list = np.sort(np.array([(k, clustering[k]['status'] == 'primary') for k in cl_list],
                               dtype=np.dtype([('clid', int), ('is_primary', bool)])))
    logger.info(f'Enable_clusters: {(~cl_list["is_primary"]).sum()} non-primary clusters have been excluded')
    cl_list = cl_list[cl_list['is_primary']]['clid'].tolist()
    if len(cl_list) == 0:
        raise NoRemainingClustersException('There were no primary clusters')

    # use instance criterion if not explicitly set
    if min_extent is None:
        min_extent = contact_map.min_extent

    if min_extent:
        cl_list = [k for k in cl_list if clustering[k]['extent'] >= min_extent]
        n_seq = sum(len(clustering[k]['seq_ids']) for k in cl_list)
        logger.info('Enable_clusters: clusters passing minimum extent criterion: {}, containing {} sequences'.format(
            len(cl_list), n_seq))
        if len(cl_list) == 0:
            raise NoRemainingClustersException(
                'No clusters passed min_extent criterion of >= {}'.format(min_extent))

    if ordered_only:
        # drop any clusters that have not been ordered
        cl_list = [k for k in cl_list if 'order' in clustering[k]]
        n_seq = sum(len(clustering[k]['seq_ids']) for k in cl_list)
        logger.info('Enable_clusters: clusters passing ordered-only criterion: {}, containing {} sequences'.format(
            len(cl_list), n_seq))
        if len(cl_list) == 0:
            raise NoRemainingClustersException('No clusters passed ordered-only criterion')

    # impose a consistent order
    cl_list = sorted(cl_list)

    # only pass sequences which were accepted.
    # primarily this handles the situation where clustering solutions reference
    # sequences with weak/no Hi-C observations (low signal).
    _accepted = contact_map.get_primary_acceptance_mask()
    if ordered_only:
        # ordered by cluster and using a pre-determined order and orientation
        cmb_ord = np.fromiter((seq_id for k in cl_list
                               for seq_id in clustering[k]['order'] if _accepted[seq_id]), dtype=int)
    else:
        # ordered by cluster but internally arbitrary order and orientation
        cmb_ord = SeqOrder.asindex(np.fromiter((seq_id for k in cl_list
                                   for seq_id in clustering[k]['seq_ids'] if _accepted[seq_id]), dtype=int))

    if len(cmb_ord) == 0:
        raise NoRemainingClustersException('The set of enabled clusters contained no sequences')

    logger.info('Total number of interacting sequences involved in enabled clusters: {}'.format(len(cmb_ord)))

    # prepare a mask, blacklisting all sequences
    _mask = contact_map.order.new_mask(False)
    # enable those sequences contained in the clusters
    _mask[cmb_ord['index']] = True
    # apply the acceptance mask
    _mask &= contact_map.get_primary_acceptance_mask()
    logger.info('After joining with active sequence mask map: {}'.format(_mask.sum()))
    contact_map.order.set_mask_only(_mask)
    contact_map.order.set_order_and_orientation(cmb_ord, implicit_excl=True)

    return cl_list


def plot_clusters(contact_map, fname, clustering, cl_list=None, simple=True, permute=False, max_image_size=None,
                  ordered_only=False, min_extent=None, use_taxo=False, flatten=False, norm_method=None,
                  show_sequences=False, **kwargs):
    """
    Plot the contact map, annotating the map with cluster names and boundaries.

    For large contact maps, block reduction can be employed to reduce the size for plotting purposes. Using
    block_reduction=2 will reduce the map dimensions by a factor of 2. Must be integer.

    :param contact_map: an instance of ContactMap to cluster
    :param fname: output file name
    :param clustering: the cluster solution
    :param cl_list: the list of cluster ids to include in plot. If none, include all ordered clusters
    :param simple: True plot seq map, False plot the extent map
    :param permute: permute the map with the present order
    :param max_image_size:  maximum allowable image size before rescale occurs
    :param ordered_only: include only clusters which have been ordered
    :param min_extent: include only clusters whose total extent is greater
    :param use_taxo: use taxonomic information within clustering, assuming it exists
    :param flatten: for tip-based, flatten matrix rather than marginalise
    :param norm_method: normalisation method to apply to contact map
    :param show_sequences: true = grid lines and labels mark individual sequences rather than whole clusters
    :param kwargs: additional options passed to plot()
    """

    if cl_list is None:
        logger.info('Plotting heatmap of complete solution')
    else:
        logger.info('Plotting heatmap for {} specified clusters'.format(len(cl_list)))

    if simple or contact_map.bin_size is None:
        if norm_method is None:
            norm_method = 'sites'
        # prepare the map early as we wish to override the mask
        # which happens to be initialized in this method call
        if contact_map.processed_map is None:
            contact_map.prepare_seq_map(norm=True, bisto=True, norm_method=norm_method)
    else:
        if norm_method is None:
            # default normalisation for extent maps
            # TODO change this to sites, if it proves better
            norm_method = 'sites'

    # now build the list of relevant clusters and setup the associated mask
    cl_list = enable_clusters(contact_map, clustering, cl_list=cl_list, ordered_only=ordered_only,
                              min_extent=min_extent)
    if simple or contact_map.bin_size is None:
        assert not show_sequences, 'show_sequences=True is not supported for simple contact maps'
        # tick spacing simple the number of sequences in the cluster
        tick_locs = np.cumsum([0] + [len(clustering[k]['seq_ids']) for k in cl_list])
        if contact_map.is_tipbased() and flatten:
            tick_locs *= 2
    else:
        # tick spacing depends on cumulative bins for sequences in cluster
        # cumulative bin count, excluding masked sequences
        csbins = [0]
        if show_sequences:
            for k in cl_list:
                # get the order records for the sequences in cluster k
                _oi = contact_map.order.order[clustering[k]['seq_ids']]
                # count the cumulative bins at each cluster for those sequences which are not masked
                csbins.extend(contact_map.grouping.bins[clustering[k]['seq_ids'][_oi['mask']]])
            tick_locs = np.array(np.array(csbins).cumsum(), dtype=np.int32)
        else:
            for k in cl_list:
                # get the order records for the sequences in cluster k
                _oi = contact_map.order.order[clustering[k]['seq_ids']]
                # count the cumulative bins at each cluster for those sequences which are not masked
                csbins.append(contact_map.grouping.bins[clustering[k]['seq_ids'][_oi['mask']]].sum() + csbins[-1])
            tick_locs = np.array(csbins, dtype=np.int32)

    if show_sequences:
        _labels = [contact_map.seq_info[si].name for cl_id in cl_list for si in clustering[cl_id]['seq_ids']]
    else:
        if use_taxo:
            _labels = [clustering[cl_id]['taxon'] for cl_id in cl_list]
        else:
            _labels = [clustering[cl_id]['name'] for cl_id in cl_list]

    contact_map.plot(fname, permute=permute, simple=simple, tick_locs=tick_locs, tick_labs=_labels,
                     max_image_size=max_image_size, flatten=flatten, norm_method=norm_method, **kwargs)


def write_report(fname, clustering):
    """
    Create a tabular report of each cluster from a clustering report. Write the table to CSV.

    :param fname: the CSV output file name
    :param clustering: the input clustering, which contains a report
    """

    def _expect(w, x):
        """
        Weighted expectation of x with weights w. Weights do not need to be
        normalised

        :param w: weights
        :param x: variable
        :return: expectation value of x
        """
        wsum = float(w.sum())
        return np.sum(w * x) / wsum

    def _n50(x):
        """
        Calculate N50 for the given list of sequence lengths.
        :param x: a list of sequence lengths
        :return: the N50 value
        """
        x = np.sort(x)[::-1]
        return x[x.cumsum() > x.sum() / 2][0]

    df = []
    has_cov = False
    for k, v in clustering.items():
        try:
            sr = v['report']

            _cl_info = [k,
                        v['name'],
                        len(v['seq_ids']),
                        v['extent'],
                        v['status'],
                        _n50(sr['length']),
                        _expect(sr['length'], sr['gc']),
                        sr['gc'].mean(),
                        np.median(sr['gc']),
                        sr['gc'].std()]

            # if coverage information exists, add statistics to the table
            if 'cov' in sr.dtype.names:
                has_cov = True
                _cl_info.extend([_expect(sr['length'], sr['cov']),
                                sr['cov'].mean(),
                                np.median(sr['cov']),
                                sr['cov'].std()])

            df.append(_cl_info)

        except KeyError as e:
            raise ReportFormatException(str(e), k)

    _cols = ['id', 'name', 'size', 'extent', 'status', 'n50', 'gc_expect', 'gc_mean', 'gc_median', 'gc_std']
    if has_cov:
        _cols.extend(['cov_expect', 'cov_mean', 'cov_median', 'cov_std'])

    df = pandas.DataFrame(df, columns=_cols)
    df.set_index('id', inplace=True)
    df.to_csv(fname, sep=',')


def find_lost_singletons(contact_map, clustering):
    """
    Return the seq_ids of any sequence which was excluded from clustering. These sequences
    will have been excluded for being too short or with too few Hi-C observations.

    :param contact_map: the contact map in question
    :param clustering: a clustering solution for this contact map.
    :return: the sequence ids for each excluded sequence.
    """
    _lost = contact_map.order.new_mask(True)
    for v in clustering.values():
        if len(v['seq_ids']) > 0:
            _lost[v['seq_ids']] = False
    return np.where(_lost)[0]


def write_mcl(contact_map, fname, clustering):
    """
    Write out the clustering solution in the format used by MCL. Each line represents a cluster
    with all members on the line show as a space-delimited list.

    :param contact_map: an instance of ContactMap to cluster
    :param fname: output file name
    :param clustering: our clustering solution
    """
    with open(fname, 'w') as outh:
        seq_info = contact_map.seq_info
        cl_soln = {}
        for k, v in clustering.items():
            cl_soln[k] = [seq_info[ix].name for ix in np.sort(v['seq_ids'])]

        clid_ascending = sorted(cl_soln.keys())
        for k in clid_ascending:
            outh.write(' '.join(cl_soln[k]))
            outh.write('\n')


def write_fasta(contact_map, output_dir, clustering, cl_list=None, source_fasta=None, clobber=False, only_large=False):
    """
    Write out multi-fasta for all determined clusters in clustering.

    For each cluster, sequence order and orientation is as follows.
    1. for unordered clusters, sequences will be in descending nucleotide length and
       in original input orientation.
    2. for ordered clusters, sequences will appear in the prescribed order and
       orientation.

    :param contact_map: an instance of ContactMap to cluster
    :param output_dir: parent output path
    :param clustering: the clustering result, possibly also ordered
    :param cl_list: the list of cluster ids to include in plot. If none, include all ordered clusters
    :param source_fasta: specify a source fasta file, otherwise assume the same path as was used in parsing
    :param clobber: True overwrite files in the output path. Does not remove directories
    :param only_large: Limit output to only clusters whose extent exceedds min_extent setting
    """

    make_dir(output_dir, exist_ok=True)

    logger.info('Writing output to the path: {}'.format(output_dir))

    seq_info = contact_map.seq_info

    parent_dir = os.path.join(output_dir, 'fasta')
    make_dir(parent_dir, clobber)

    if source_fasta is None:
        source_fasta = contact_map.seq_file

    # analyze all if no subset list was provided
    if cl_list is None:
        cl_list = clustering.keys()
        logger.info('Writing FASTA for all suitable clusters')
    else:
        logger.info('Writing FASTA the following specified clusters: {}'.format(np.asarray(cl_list)+1))

    # set up indexed access to the input fasta
    with contextlib.closing(IndexedFasta(source_fasta)) as seq_db:

        # iterate over the cluster set, in the existing order
        for cl_id in cl_list:

            cl_info = clustering[cl_id]

            if only_large and cl_info['extent'] < contact_map.min_extent:
                continue

            # Each cluster produces a multi-fasta. Sequences are not joined
            cl_path = os.path.join(parent_dir, '{}.fna'.format(cl_info['name']))

            if not clobber and os.path.exists(cl_path):
                raise IOError('Output path exists [{}] and overwriting not enabled'.format(cl_path))

            # determine the number of digits required for cluster sequence names
            try:
                num_width = max(1, int(np.ceil(np.log10(len(cl_info['seq_ids'])+1))))
            except OverflowError:
                num_width = 1

            with open(cl_path, 'w') as output_h:

                logger.debug('Writing full unordered FASTA for cluster {} to {}'.format(cl_id, cl_path))

                # iterate simply over sequence ids, while imposing ascending numerical order
                for n, _seq_id in enumerate(np.sort(cl_info['seq_ids']), 1):
                    # fetch the SeqRecord object from the input fasta
                    _seq = seq_db[seq_info[_seq_id].name]
                    # add a description
                    _cl_desc = 'cluster:{}'.format(cl_info['name'])
                    if _seq.description is None or not _seq.description.strip():
                        _seq.description = _cl_desc
                    else:
                        _seq.description = '{} {}'.format(_seq.description, _cl_desc)
                    SeqIO.write(_seq, output_h, 'fasta')

            # write a separate ordered fasta as this is often a subset of all sequences
            if 'order' in cl_info:

                # Each cluster produces a multi-fasta. Sequences are not joined
                cl_path = os.path.join(parent_dir, '{}.ordered.fna'.format(cl_info['name']))

                if not clobber and os.path.exists(cl_path):
                    raise IOError('Output path exists [{}] and overwriting not enabled'.format(cl_path))

                with open(cl_path, 'w') as output_h:

                    logger.debug('Writing ordered FASTA for cluster {} to {}'.format(cl_id, cl_path))

                    # iterate over cluster members, in the determined order
                    for n, _oi in enumerate(cl_info['order'], 1):
                        # get the sequence's external name
                        _name = seq_info[_oi['index']].name
                        # fetch the SeqRecord object from the input fasta
                        _seq = seq_db[_name]
                        # reverse complement as needed
                        if _oi['ori'] == SeqOrder.REVERSE:
                            _seq = _seq.reverse_complement()
                            _ori_symb = '-'
                        elif _oi['ori'] == SeqOrder.FORWARD:
                            _ori_symb = '+'
                        else:
                            raise UnknownOrientationStateException(_oi['ori'])

                        # add a description
                        _cl_desc = 'cluster:{} ori:{}'.format(cl_info['name'], _ori_symb)
                        if _seq.description is None or not _seq.description.strip():
                            _seq.description = _cl_desc
                        else:
                            _seq.description = '{} {}'.format(_seq.description, _cl_desc)
                        SeqIO.write(_seq, output_h, 'fasta')


def extract_bam(contact_map, clustering, output_dir, cluster_ids, threads=4, clobber=False, bam_file=None,
                version=None, cmdline=None):
    """
    Extract a BAM file from the full source BAM file used in creating the contact map.
    Only read-pairs whose ends are both contained with the cluster are retained.

    :param contact_map: the contact_map from which to extract a cluster
    :param clustering: the clustering solution for this contact map
    :param output_dir: the output directory to write the extracted bam
    :param cluster_ids: the 0-based cluster identifier
    :param threads: the number of threads to use when parsing the bam file
    :param clobber: overwrite output if True
    :param bam_file: alternative location for BAM file
    :param version: version stamp string for BAM file
    :param cmdline: commandline options used for BAM file
    :return: tuple (output file name, number of pairs)
    """

    def _next_informative(_bam_iter, _pbar):
        while True:
            r = next(_bam_iter)
            _pbar.update()
            if not r.is_unmapped and not r.is_secondary and not r.is_supplementary:
                return r

    cluster_ids = np.asarray(cluster_ids)

    if len(cluster_ids) == 1:
        output_file = os.path.join(output_dir, 'cluster_{}.bam'.format(cluster_ids[0]+1))
    elif len(cluster_ids) <= 5:
        output_file = os.path.join(output_dir, 'cluster_{}.bam'.format('_'.join(str(_id) for _id in cluster_ids+1)))
    else:
        output_file = os.path.join(output_dir, 'cluster_many.bam')

    if clobber and os.path.exists(output_file):
        logger.error('{} already exists'.format(output_file))

    if bam_file is None:
        logger.debug('Attempting to open source BAM: {}'.format(contact_map.bam_file))
        bam_in = pysam.AlignmentFile(contact_map.bam_file, 'rb', threads=threads)
    else:
        logger.debug('Alternative source source BAM: {}'.format(bam_file))
        bam_in = pysam.AlignmentFile(bam_file, 'rb', threads=threads)

    logger.debug('Extracting BAM for clusters: {}'.format(cluster_ids + 1))

    # prepare header for extracted bam
    header = bam_in.header.to_dict()
    keepers = []
    ref_lookup = {}
    n = 0
    for _id in cluster_ids:
        for seq_id in clustering[_id]['seq_ids']:
            keepers.append(header['SQ'][contact_map.seq_info[seq_id].refid])
            ref_lookup[contact_map.seq_info[seq_id].refid] = n
            n += 1
    header['SQ'] = keepers
    header['PG'].append({'CL': 'bin3C extract' if cmdline is None else cmdline,
                         'ID': 'bin3C',
                         'PN': 'bin3C',
                         'VN': 'unknown' if version is None else version})

    n_refs = len(keepers)
    logger.info('The extracted BAM is based on {} references'.format(n_refs))

    # iterate over the complete bam, store those pairs which are contained
    # within the chosen cluster
    n_pairs = 0
    bam_iter = bam_in.fetch(until_eof=True)

    with pysam.AlignmentFile(output_file, 'wb', header=header, threads=threads) as bam_out:
        with tqdm.tqdm(total=contact_map.total_reads) as progress_bar:

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

                if r1.reference_id in ref_lookup and r2.reference_id in ref_lookup:
                    r1.reference_id = ref_lookup[r1.reference_id]
                    r1.next_reference_id = ref_lookup[r1.next_reference_id]
                    r2.reference_id = ref_lookup[r2.reference_id]
                    r2.next_reference_id = ref_lookup[r2.next_reference_id]
                    bam_out.write(r1)
                    bam_out.write(r2)
                    n_pairs += 1

        return output_file, n_refs, n_pairs


def write_multilayer_pajek(output_filename, layers, add_inter=True, inter_scale=0.1):
    """
    Write a Pajek format graph multi-layer graph.

    :param output_filename:
    :param layers: an iterable collection of graphs, each of which will be a layer
    :param add_inter: add inter-layer links between coincident nodes
    :param inter_scale: inter-layer edge weight
    """
    assert len(layers) > 1, 'more than one layer is required'

    with open(output_filename, 'w') as output_h:

        # include all nodes, even if not present in every layer
        node_registry = set()
        intra_edge_count = 0
        for n, lg in enumerate(layers, start=1):
            ln = set(lg.nodes())
            intra_edge_count += lg.size()
            logger.debug(f'Number of nodes in layer {n}: {len(ln)}')
            node_registry |= ln

        node_registry = {_id: n for n, _id in enumerate(sorted(node_registry), start=1)}
        assert len(node_registry) > 0, 'no nodes found in any layer'
        logger.debug(f'Number of nodes across all layers: {len(node_registry)} '
                     f'and {intra_edge_count} within-layer edges')

        output_h.write(f'*Vertices {len(node_registry)}\n')
        for _id, n in node_registry.items():
            output_h.write(f'{n} "{_id}"\n')

        output_h.write('*Intra\n')
        for n, lg in enumerate(layers, start=1):
            for u, v, d in lg.edges(data=True):
                output_h.write(f'{n} {node_registry[u]} {node_registry[v]} {d["weight"]}\n')

        # add explict interlayer edges.
        if add_inter:
            shared_nodes = set.intersection(*[set(lg.nodes()) for lg in layers])
            if len(shared_nodes) == 0:
                logger.debug('No shared nodes found, skipping inter-layer edges')
            else:
                output_h.write('*Inter\n')
                logger.debug(f'Creating {len(shared_nodes)} inter-layer edges')
                for _id in shared_nodes:
                    output_h.write(f'1 {node_registry[_id]} 2 {inter_scale}\n')


def read_gfa(gfa_filename, paths_to_edges=False, read_progress=False):
    """
    Convert a GFA file into a Networkx graph. At present, it is assumed that the GFA file contains
    optional depth attributes generated by the Flye assembler.

    :param gfa_filename: path to the input GFA file
    :param paths_to_edges: only include edges that are mentioned as paths, rather than all link records
    :return: undirected Networkx graph
    """
    gfa = GFA(gfa_filename, skip_sequence_data=True, progress=read_progress)

    g_out = nx.Graph()
    # all segments in the GFA will become nodes
    for _seg in gfa.segments.values():
        # we wish to annotate nodes with k-mer depth reported by Flye
        assert 'dp' in _seg.optionals, f'Segment {_seg.name} was missing depth field. Was this gfa created by Flye?'
        g_out.add_node(_seg.name, length=_seg.length, depth=_seg.optionals['dp'].get())

    # collect a registry of all inter-segment links
    link_registry = nx.Graph()
    for _link in gfa.links.values():
        u, v = _link.src, _link.dest
        # check that the RC field exists, this acts as edge weight
        assert 'RC' in _link.optionals, f'Link ({u},{v}) is missing the RC field. Was this gfa created by Flye?'
        cov = _link.optionals['RC'].get()
        if link_registry.has_edge(u, v):
            link_registry[u][v]['udir'].append(_link.src_orient)
            link_registry[u][v]['vdir'].append(_link.dest_orient)
            link_registry[u][v]['rc'].append(cov)
        else:
            cov = _link.optionals['RC'].get()
            link_registry.add_edge(u, v, udir=[_link.src_orient], vdir=[_link.dest_orient], rc=[cov])

    # Assemblers will likely treat links as directed and report more than one record per segment pair.
    # We therefore take the mean depth as edge weights.
    for u, v, d in link_registry.edges(data=True):
        link_registry[u][v]['weight'] = np.mean(d['rc'])

    # add edges, either selectively just for paths or all links
    if paths_to_edges:
        for _path in gfa.paths.values():
            for i in range(len(_path.segment_names)-1):
                u, udir = _path.segment_names[i]
                v, vdir = _path.segment_names[i+1]
                g_out.add_edge(u, v, contig=_path.name, weight=link_registry[u][v]['weight'])
    else:
        g_out.add_edges_from(link_registry.edges(data=True))

    return g_out


def harden_clustering(clustering, contact_map):
    """
    Reduce a potentially soft-clustering solution to a hard-clustering solution by removing
    repeated assignments for any given sequence. Only the assignment to the largest cluster
    is retained -- measured by extent.

    TODO ironically, the present codebase does not handle the notion of a soft-clustering solution
       and will have to be implemented if these solutions prove superior.

    :param clustering: the clustering solution
    :param contact_map: the contact map
    :return the deduplicated clustering
    """
    # potential one-to-many lookup seq->cluster
    seq2cl = defaultdict(set)
    for cl_id, cl_info in clustering.items():
        for seq_id in cl_info['seq_ids']:
            seq2cl[seq_id].add((cl_id, cl_info['extent']))

    hard_clustering = deepcopy(clustering)
    seq_info = np.array(contact_map.seq_info,
                        dtype=np.dtype([('offset', int),
                                        ('refid', int),
                                        ('name', 'U30'),
                                        ('length', int),
                                        ('sites', int),
                                        ('gc', float)]))

    n_duped = 0
    dupe_extent = 0
    for seq_id, memberships in seq2cl.items():
        if len(memberships) <= 1:
            continue
        n_duped += 1
        # keep only the assignment to the largest cluster
        # TODO this a simplistic approach for now.
        cl_targets = sorted(memberships, key=lambda x: -x[1])[1:]
        for cl_id, extent in cl_targets:
            _ix = np.where(hard_clustering[cl_id]['seq_ids'] == seq_id)[0]
            assert len(_ix) != 0, f'failed to find sequence {seq_id} in expected cluster {cl_id}'
            hard_clustering[cl_id]['seq_ids'] = np.delete(hard_clustering[cl_id]['seq_ids'], _ix)
            hard_clustering[cl_id]['extent'] -= seq_info[seq_id]['length']
            hard_clustering[cl_id]['deduplicated'].append(seq_id)
        dupe_extent += seq_info[seq_id]['length']

    logger.info(f'Clustering involves {len(seq2cl):,} sequences, '
                f'of which {n_duped:,} were assigned to more than one cluster')
    logger.info(f'Average length of a degenerate sequence: {dupe_extent // n_duped:,}')

    # it's possible that some clusters are now empty
    n_depleted = remove_empty_clusters(hard_clustering)
    if n_depleted > 0:
        logger.info(f'Removed {n_depleted} clusters that had no remaining members')

    return hard_clustering


def remove_empty_clusters(clustering):
    """
    In-place removal of any clusters which are empty, with the implication that ids within the
    clustering will no longer be consecutive integers.
    :param clustering:
    """
    n_depleted = 0
    for k in list(clustering):
        if len(clustering[k]['seq_ids']) == 0:
            n_depleted += 1
            del clustering[k]

    return n_depleted

from .embedding import MetagenomeEmbeddings

import logging
import numpy as np
import os
import pandas as pd
import warnings

from sklearn.metrics.pairwise import linear_kernel



logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def scaler(arr, mu=None, sig=None):
    if mu is None:
        mu = np.mean(arr)
        sig = np.std(arr)
        return (arr - mu) / sig, mu, sig
    else:
        return (arr - mu) / sig


def transform(df):
    return df.assign(intra_z = lambda x: x.intra.astype(np.uint8),
                     cov_z   = lambda x: scaler(np.log(x.cov_u * x.cov_v))[0],
                     freq_z  = lambda x: scaler(np.log(x.contacts / (x.sites_u*x.sites_v * x.uf_u*x.uf_v)))[0],
                     )


def anti_join(target, exclude_from):
    """
    Remove rows from target which are present in exclude_from.
    :param target: the target dataframe
    :param exclude_from: the exclusion table
    :return: modified target dataframe
    """
    excl_idx = exclude_from.set_index(['seq','cluster']).index
    target = target.set_index(['seq','cluster'])
    mask = target.index.isin(excl_idx)
    return target[~ mask].reset_index()


def high_quality_clusters(binning_qc_file, min_completeness, max_contamination, qc_method):
    """
    Select clusters which meet the minimum quality thresholds for completeness and contamination.
    :param binning_qc_file:
    :param min_completeness:
    :param max_contamination:
    :param qc_method: e.g. CheckMv1, CheckMv2, CoCoPye
    :return: set of cluster IDs
    """
    # Binning QC summary to eliminate contaminated bins
    df = pd.read_csv(binning_qc_file, header=[0,1], index_col=0)
    df.sort_index(axis=1, inplace=True)
    assert np.all(df.index.str.startswith('CL')), \
        'Binning QC table did not appear to be indexed by cluster names'
    hq_set = set(df[(qc_method,)].query(
        'Contamination <= @max_contamination and Completeness >= @min_completeness').index)
    logger.info(f'Referring to {qc_method}, there were {len(hq_set)} acceptable clusters '
                f'with Completeness>={min_completeness:.0f} and Contamination<={max_contamination:.0f}.')
    return hq_set


def exclude_clusters(df_target, accepted_clusters):
    """
    Exclude any contacts not involving the supplised accepted clusters
    :param df_target: target dataframe to filter
    :param accepted_clusters:
    :return:
    """
    result = df_target.query('cluster_name in @accepted_clusters')
    logger.info(f'Removing contacts involving acceptable quality clusters: in={len(df_target)}, out={len(result)}')
    return result


def identify_suspected_intra(df, accepted_clusters, min_similarity, min_contacts, min_cluster_length,
                             max_seq_length, min_degree=2, sort_by='similarity'):

    # number of relevant contacts per sequence
    degree = df.query('contacts>@min_contacts '
                      'and similarity>@min_similarity '
                      'and cluster_name in @accepted_clusters') \
        .groupby('seq') \
        .size()

    # reduce the table to only those with sufficiently high degree
    df = df.set_index('seq').loc[degree[degree >= min_degree].index].reset_index()

    suspected_intra = df.query('similarity > @min_similarity'
                               ' and contacts > @min_contacts'
                               ' and length_u < length_v'
                               ' and length_u < @max_seq_length'
                               ' and length_v > @min_cluster_length'
                               ' and cluster_name in @accepted_clusters') \
        .sort_values(sort_by, ascending=False) \
        .drop_duplicates('seq', keep='first') \
        .set_index(['seq','cluster'])

    logger.info(f'Suspected intra-cellular contacts: {len(suspected_intra)}')
    return suspected_intra.copy()


def seq2cluster_similarity(df, embeddings):
    ix = df[['seq','cluster']].values
    u = embeddings.seq_embeds.loc[ix[:, 0], range(768)].values
    v = embeddings.cluster_embeds.loc[ix[:, 1], range(768)].values
    assert u.shape == v.shape, 'U and V not of the same dimension'
    return np.fromiter((linear_kernel(u[[i]], v[[i]])[0][0] for i in range(u.shape[0])), dtype='f8')


def normalised_out_degree(x):
    """
    NOTE: Intended to be performed on a dataframe of associations grouped by sequence name.

    For each association made by an individual sequence, calculate the proportion of contacts
    made between that sequence and the associated cluster. The contact count is normalised
    by cluster length, coverage and uniqueness factor.

    :param x:
    :return:
    """
    # TODO this commented out formulation does not test as well, yet logically I'd expect
    #   it to be the more correct.
    #
    # # Compensate for the fact that contact counts are only between the sequence
    # # and other members of the cluster -- no self contacts. For very large
    # # members, we must remove their site count before normalising.
    # vdelu_sites = (x.sites_v - x.sites_u)
    # # ensure that there are no zeros in the denominator vector (singletons)
    # vdelu_sites[vdelu_sites <= 0] = 1
    # edge_weight = x.contacts / np.sqrt(vdelu_sites * x.cov_v * x.uf_v)

    edge_weight = x.contacts / np.sqrt(x.sites_v * x.cov_v * x.uf_v)
    return edge_weight / edge_weight.sum()


def replace_zeros(x, reduction_factor):
    """
    Replace zeros in a pandas series with a small value relative to the minimum non-zero value.
    :param x: series
    :param reduction_factor: the factor by which to multiply the minimum non-zero value
    :return: updated series
    """
    min_val = (x[x > 0]).min()
    return x.replace(0, reduction_factor * min_val)


class DataLabeller(object):

    _SMALL_UF = 1e-3
    _SMALL_COV = 1
    _GROUP_A = 1
    _GROUP_B = 2
    _MIN_NUM_OBS = 0
    _MIN_EXTENT = 100_000
    _BIG_EXTENT = 500
    _HQ_COMPL = 90
    _HQ_CONTAM = 10
    _PURE_COMPL = 50
    _PURE_CONTAM = 10

    _SUSP_MIN_CLUSTER_EXTENT = 1_000_000
    _SUSP_MIN_SEQ_LENGTH = 500_000
    _SUSP_MIN_SIM = 0.7
    _SUSP_MIN_NUM_OBS = 10
    _SUSP_MIN_DEGREE = 1

    OUTPUT_TABLES = {
        'training': 'training.csv',
        'undecided': 'undecided.csv',
        'combined': 'combined.csv'
    }

    @staticmethod
    def get_output_path(parent_dir, table_name):
        return os.path.join(parent_dir, DataLabeller.OUTPUT_TABLES[table_name])

    def __init__(self,
                 output_dir,
                 embeddings_file,
                 clustering_file,
                 faidx_file,
                 spurious_file,
                 all_contacts_file,
                 binning_qc_file,
                 qc_method='CheckMv1',
                 use_suspected=False,
                 plot_projection=False,
                 max_clusters=10):
        self.output_dir = output_dir
        self.embeddings_file = embeddings_file
        self.clustering_file = clustering_file
        self.faidx_file = faidx_file
        self.spurious_file = spurious_file
        self.all_contacts_file = all_contacts_file
        self.binning_qc_file = binning_qc_file
        self.use_suspected = use_suspected
        self.qc_method = qc_method
        self.embeddings = MetagenomeEmbeddings(self.embeddings_file,
                                               self.clustering_file,
                                               self.faidx_file)
        if plot_projection:
            self.embeddings.plot_scatter_projection(
                os.path.join(output_dir, 'Embedding_UMAP_Manhattan_projection.svg'),
                max_clusters=max_clusters)

    def write_table(self, df, table_name, description, index):
        """
        Standardised writing of a table to a file
        :param df: the pandas table
        :param table_name: name of the table to write (obtains file name)
        :param description: a description of logging
        :param index: whether to include
        """
        file_path = DataLabeller.get_output_path(self.output_dir, table_name)
        logger.info(f'Writing {description} to {file_path}')
        df.to_csv(file_path, index=index)

    def prepare_labelled_training_data(self):
        assert self.embeddings is not None, 'The embeddings data must be analysed first'

        # Read initial prediction of spurious contacts
        df_spur = pd.read_csv(self.spurious_file, index_col=0)
        # Read table of all real contacts (not the symbolic closed-sequence to singleton-cluster)
        df_all = pd.read_csv(self.all_contacts_file, index_col=0)
        logger.info(f'Before exclusion counts spurious: {len(df_spur)}, all: {len(df_all)}')

        hq_clusters = high_quality_clusters(self.binning_qc_file,
                                            DataLabeller._HQ_COMPL,
                                            DataLabeller._HQ_CONTAM,
                                            self.qc_method)

        # Reduce false positive rate in spurious table by keeping only
        # contacts involving sufficiently complete and uncontaminated clusters.
        n_before = len(df_spur)
        df_spur = df_spur.query('cluster_name in @hq_clusters').copy()
        logger.info(f'After filtering for high quality clusters: in={n_before}, out={len(df_spur)}')
        df_all = anti_join(df_all, df_spur)
        logger.info(f'Applying exclusion to all-contacts: {len(df_all)}')

        # concatenate the two tables, assigning a group label for later separation.
        df_all['group'] = DataLabeller._GROUP_A
        df_spur['group'] = DataLabeller._GROUP_B

        # combine the tables, make sure to remove duplicates but retain those which came
        # from the spurious table
        df_cmb = pd.concat([df_all, df_spur]) \
            .sort_values('group', ascending=False) \
            .drop_duplicates(['seq','cluster'], keep='first')

        # make sure that any zeros are replaced with a small value determined
        # by the supplied vector
        df_cmb['cov_u'] = replace_zeros(df_cmb.cov_u, 0.5)
        df_cmb['cov_v'] = replace_zeros(df_cmb.cov_v, 0.5)
        df_cmb['uf_u'] = replace_zeros(df_cmb.uf_u, 0.5)
        df_cmb['uf_v'] = replace_zeros(df_cmb.uf_v, 0.5)

        # calculate similarity between sequence and cluster
        logger.info('Calculating similarities')
        df_cmb['similarity'] = seq2cluster_similarity(df_cmb, self.embeddings)

        # reset the index to a simple integer, after first insuring an intuitive ordering
        logger.info('Calculating linkage coefficient')
        # calculate the proportion the sequence represents relative to the clusters extent
        df_cmb = df_cmb.assign(prop_cl = lambda x: x.length_u / x.length_v)
        df_cmb = df_cmb.sort_values(['seq','cluster']).reset_index(drop=True)
        # calculate linkage coefficient and assign
        linkage = df_cmb.groupby('seq', group_keys=False) \
                        .apply(normalised_out_degree, include_groups=False)
        # Log-transform and standardise the linkage coefficient, as its distribution is far
        # from smooth, with significant mass close to zero (spurious contacts).
        df_cmb['linkage'] = scaler(np.log(linkage))[0]

        logger.info('Standardising all observations together')
        df_cmb = transform(df_cmb)

        # decompose the two tables
        df_all = df_cmb.query(f'group == {DataLabeller._GROUP_A}').copy()
        df_spur = df_cmb.query(f'group == {DataLabeller._GROUP_B}').set_index(['seq','cluster']).copy()

        # Break down the table of all contacts, where the aim is to identify
        #   those contacts which evidence strongly indicates the contact intra-cellular

        # STEP ONE: basic filter for intuitively sensible seq->cluster relationships.
        # 1. sequence smaller than cluster (this is akin to looking at half the contact map)
        # 2. cluster minimum extent
        # 3. at least N contacts
        df_all = df_all.query(f'contacts > {DataLabeller._MIN_NUM_OBS}'
                              ' and length_u < length_v'
                              f' and length_v > {DataLabeller._MIN_EXTENT}') \
                       .set_index(['seq','cluster'])
        logger.info(f'Contacts after basic filtering: {len(df_all)} ')

        # STEP TWO: keep those that are intra-cluster contacts (intra=True) where the cluster is of reasonable size
        df_signif = df_all.query(f'intra and length_v > {DataLabeller._BIG_EXTENT}').copy()
        logger.info(f'Reliable intra-cluster contacts: {len(df_signif)}')

        # STEP THREE: prepare a table of undecided contacts by removing those determined to be reliably intra or inter
        # intra removal
        df_undecided = df_all[~df_all.index.isin(df_signif.index)].copy()
        logger.info(f'Undecided contacts, after removing reliable: {len(df_undecided)}')
        # spurious removal
        df_undecided = df_undecided[~df_undecided.index.isin(df_spur.index)]
        logger.info(f'Undecided contacts, after removing spurious: {len(df_undecided)}')

        # Add initial training labels
        df_spur['intra_z'] = 0
        df_signif['intra_z'] = 1

        # STEP 4: try to find additional "suspected intra-cluster" contacts from even larger
        # clusters, that may be split. These must still be low contamination.
        if self.use_suspected:
            pure_clusters = high_quality_clusters(self.binning_qc_file,
                                                  DataLabeller._PURE_COMPL, DataLabeller._PURE_CONTAM,
                                                  self.qc_method)
            df_suspected = identify_suspected_intra(df_undecided.reset_index(), pure_clusters,
                                                    DataLabeller._SUSP_MIN_SIM,
                                                    DataLabeller._SUSP_MIN_NUM_OBS,
                                                    DataLabeller._SUSP_MIN_CLUSTER_EXTENT,
                                                    DataLabeller._SUSP_MIN_SEQ_LENGTH,
                                                    min_degree=DataLabeller._SUSP_MIN_DEGREE,
                                                    sort_by='freq_z')
            df_undecided = df_undecided[~df_undecided.index.isin(df_suspected.index)]
            logger.info(f'Undecided contacts, after removing suspected intra: {len(df_undecided)}')
            df_suspected['intra_z'] = 1
            df_signif = pd.concat([df_signif, df_suspected])
            logger.info(f'Combining significant and suspects, there are now {len(df_signif)} intra-cluster contacts')

        df_train = pd.concat([df_spur, df_signif])
        logger.info(f'Checking for duplicate records yielded: {len(df_train) - len(df_train.reset_index().drop_duplicates(["seq","cluster"]))}')
        logger.info(f'After basic concatenation, training set contains {len(df_train)} contacts')
        logger.info(f'Undecided set: {len(df_undecided)}')

        # label contacts acceptable for training
        self.write_table(df_train, 'training', 'labelled training data', index=False)
        self.write_table(df_undecided, 'undecided', 'undecided contacts', index=False)
        df_train['train'] = True
        df_undecided['train'] = False

        # combine and reset the basic integer index to get unique values
        df_cmb = pd.concat([df_train.reset_index(), df_undecided.reset_index()]).reset_index(drop=True)
        self.write_table(df_cmb, 'combined', 'final combined table', index=False)

        return df_cmb

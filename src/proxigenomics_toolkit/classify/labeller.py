import logging
import os
import warnings
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

from .embedding import MetagenomeEmbeddings
from .significance import robust_read_csv

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def scaler(arr: np.ndarray,
           mu: Optional[float]=None,
           sig: Optional[float]=None) -> np.ndarray | Tuple[np.ndarray, float, float]:
    """
    Scales an array using Z-score normalization.

    This function standardizes an array by subtracting the mean and dividing by the
    standard deviation.

    - If `mu` and `sig` are both omitted, they are computed from `arr`. The function
      returns a tuple containing the scaled array, the computed mean, and the
      computed standard deviation.
    - If `mu` and `sig` are both provided, they are used to scale `arr`, and only
      the scaled array is returned.

    :param arr: The NumPy array to be scaled.
    :param mu: Optional pre-computed mean.
    :param sig: Optional pre-computed standard deviation.
    :return: A tuple `(scaled_array, mean, std_dev)` or the `scaled_array` itself,
             depending on whether `mu` and `sig` are provided.
    :raises ValueError: If only one of `mu` or `sig` is provided.
    """
    if mu is None:
        mu = np.mean(arr)
        sig = np.std(arr)
        return (arr - mu) / sig, mu, sig
    elif sig is not None:
        return (arr - mu) / sig
    else:
        raise ValueError('Both mu and sig must be provided if one is provided.')


def logistical_transform(x: np.ndarray) -> np.ndarray:
    """
    Apply a logistic transformation to the input data.

    The logistical transformation is defined as log(x / (1 - x)).
    It is typically applied to probabilities or data constrained
    in the interval [0, 1].

    :param x: Input array-like object representing data constrained
        in the interval [0, 1].
    :type x: np.ndarray
    :return: Transformed data after applying the logistic
        function, with the same shape as the input.
    :rtype: np.ndarra
    """
    return np.log(x / (1 - x))


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the given DataFrame by assigning new columns calculated with specific transformations.
    The function performs the following modifications:

    - Creates a new column `intra_z` by converting the `intra` column to type `uint8`.
    - Computes a new column `cov_z` by scaling the logarithm of the product of columns `cov_u` and `cov_v`.
    - Calculates a new column `freq_z` by scaling the logarithm of a fraction formed by the `contacts` column
      divided by the product of several other columns: `sites_u`, `sites_v`, `uf_u`, `uf_v`.

    :param df: Input DataFrame containing the required columns (`intra`, `cov_u`, `cov_v`, `contacts`, `sites_u`,
                 `sites_v`, `uf_u`, `uf_v`) to apply transformation operations.
    :type df: pd.DataFrame
    :return: Transformed DataFrame with newly assigned columns `intra_z`, `cov_z`, and `freq_z`.
    :rtype: pd.DataFrame
    """
    return df.assign(intra_z=lambda x: x.intra.astype(np.uint8),
                     cov_z=lambda x: scaler(np.log(x.cov_u * x.cov_v))[0],
                     sites_z=lambda x: scaler(np.log(x.sites_u * x.sites_v))[0],
                     freq_z=lambda x: scaler(np.log(x.contacts / (x.sites_u * x.sites_v * x.uf_u * x.uf_v)))[0],
                     uf_z=lambda x: scaler(logistical_transform(x.uf_u * x.uf_v))[0],
                     linkage_z=lambda x: scaler(np.log(x.linkage))[0],
                     log_covu=lambda x: scaler(np.log(x.cov_u))[0],
                     log_covv=lambda x: scaler(np.log(x.cov_v))[0],
                     )


def anti_join(target: pd.DataFrame, exclude_from: pd.DataFrame) -> pd.DataFrame:
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


class ClusterFilter(object):

    _METHOD_QUALITY_LIMITS : ClassVar[Dict[str, Dict[str, List[int]]]] = {
        'CheckMv1': {    'high': [90, 5],
                         'partial': [50, 5],
                         'moderate': [50, 10]},

        'CheckMv2': {    'high': [90, 5],
                         'partial': [50, 5],
                         'moderate': [50, 10]},

        'CoCoPye': {    'high': [90, 10],
                        'partial': [50, 10],
                        'moderate': [50, 20]}
    }

    def __init__(self, qc_filename: str, file_format: str='collated_qc') -> None:
        """
        Initializes the quality control report object by loading and validating data from
        a specified CSV file. The CSV file must adhere to a specific format and structure.

        :param qc_filename: Path to the quality control input file. Must point to a
            valid CSV formatted file containing the quality control report.
        :type qc_filename: str
        :param file_format: The expected file format of the quality control report.
            This method currently supports only the 'collated_qc' format. Defaults to 'collated_qc'.
        :type file_format: str
        :raises AssertionError: If the specified file format is not supported.
        :raises AssertionError: If the quality report does not have cluster names as index.
        """
        self.qc_filename = qc_filename
        assert file_format == 'collated_qc', 'Unsupported file format specified'
        self.quality_report = pd.read_csv(qc_filename, header=[0,1], index_col=0).sort_index(axis=1)
        assert np.all(self.quality_report.index.str.startswith('CL')), \
            'Binning QC table did not appear to be indexed by cluster names'

    def quality_filter(self, method: str, quality_type: str) -> Set[str]:
        """
        Filters clusters based on their quality metrics as defined by the specified
        method and quality type. This method evaluates the `Completeness` and
        `Contamination` thresholds provided in the method- and quality-type-specific
        configuration and returns the set of clusters that meet these criteria.

        :param method: The quality control method to use for filtering. It must
            match one of the keys in the defined `_METHOD_QUALITY_LIMITS`.
        :type method: str
        :param quality_type: The quality type associated with the QC method,
            specifying further thresholds for filtering. It must match the
            appropriate values in `_METHOD_QUALITY_LIMITS`.
        :type quality_type: str
        :return: A set of cluster indices that satisfy the quality requirements
            for the given method and quality type.
        :rtype: set
        """
        assert method in self._METHOD_QUALITY_LIMITS, f'QC method {method} not recognised'
        assert quality_type in self._METHOD_QUALITY_LIMITS[method], f'Quality type {quality_type} not recognised'
        min_compl, max_contam = ClusterFilter._METHOD_QUALITY_LIMITS[method][quality_type]
        cl_set = (self.quality_report.loc[:, (method)][['Completeness','Contamination']]
                  .query("Completeness >= @min_compl and Contamination <= @max_contam").index)
        return set(cl_set)


def exclude_clusters(df_target: pd.DataFrame,
                     accepted_clusters: set) -> pd.DataFrame:
    """
    Exclude any contacts not involving the supplised accepted clusters
    :param df_target: target dataframe to filter
    :param accepted_clusters:
    :return:
    """
    result = df_target.query('cluster_name in @accepted_clusters')
    logger.info(f'Removing contacts involving acceptable quality clusters: in={len(df_target)}, out={len(result)}')
    return result

# eliminate false positive "unused local symbol" due to parameters being embedded in pandas queries.
# noinspection PyUnusedLocal
def identify_suspected_intra(df: pd.DataFrame,
                             accepted_clusters: set,
                             min_similarity: float,
                             min_contacts: int,
                             min_cluster_length: int,
                             max_seq_length: int,
                             min_degree: int=2,
                             sort_by: str='similarity') -> pd.DataFrame:
    """
    Identifies sequences that are likely fragments of larger clusters.

    This function operates in two stages. First, it identifies sequences that
    have a minimum number of high-quality associations (`min_degree`), where
    "high-quality" is defined by `min_contacts` and `min_similarity`.

    Second, from this reduced set, it selects contacts that represent a small
    sequence (`length_u`) associating with a large cluster (`length_v`). For each
    sequence that meets these criteria, the function selects the single best
    association based on the `sort_by` parameter.

    This is useful for finding contigs that may have been incorrectly binned
    separately from their parent genome.

    :param df: DataFrame with contact data. Must include columns: 'seq',
               'cluster', 'cluster_name', 'contacts', 'similarity',
               'length_u', and 'length_v'.
    :param accepted_clusters: A set of cluster names to include in the analysis.
    :param min_similarity: The minimum similarity score for a contact to be
                           considered significant.
    :param min_contacts: The minimum number of contacts for an association to
                         be considered significant.
    :param min_cluster_length: The minimum length required for the target
                               cluster (`length_v`).
    :param max_seq_length: The maximum length allowed for the source
                           sequence (`length_u`).
    :param min_degree: The minimum number of significant associations a sequence
                       must have to be considered a candidate. Defaults to 2.
    :param sort_by: The column used to rank and select the best candidate for
                    each sequence. Defaults to 'similarity'.
    :return: A DataFrame containing the top suspected intra-cellular contact
             for each sequence, indexed by ['seq', 'cluster'].
    """

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


def seq2cluster_similarity(df: pd.DataFrame,
                           embeddings: MetagenomeEmbeddings,
                           dimension: int = 768) -> np.ndarray:
    """
    Calculates the similarity between sequence and cluster embeddings.

    For each sequence-cluster pair provided in the input DataFrame, this function
    retrieves their respective embeddings and computes the similarity using a
    linear kernel. The resulting array of scores corresponds to the order of
    pairs in the input DataFrame.

    :param df: A DataFrame with 'seq' and 'cluster' columns, specifying the
               pairs for which to calculate similarity.
    :param embeddings: An object containing `seq_embeds` and `cluster_embeds`
                       DataFrames with the pre-computed embeddings.
    :param dimension: The dimensionality of the embedding vectors to use for
                      the calculation. Defaults to 768.
    :return: A NumPy array of similarity scores, one for each input pair.
    :raises AssertionError: If the dimensions of the sequence and cluster
                            embedding matrices do not match.
    """

    ix = df[['seq','cluster']].values
    u = embeddings.seq_embeds.loc[ix[:, 0], range(dimension)].values
    v = embeddings.cluster_embeds.loc[ix[:, 1], range(dimension)].values
    assert u.shape == v.shape, f'U and V not of the same dimension: {dimension}'
    return np.fromiter((linear_kernel(u[[i]], v[[i]])[0][0] for i in range(u.shape[0])), dtype='f8')


def normalised_out_degree(x: pd.DataFrame) -> np.ndarray:
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


def replace_zeros(x: pd.Series, reduction_factor: float) -> pd.Series:
    """
    Replace zeros in a pandas series with a small value relative to the minimum non-zero value.
    :param x: series
    :param reduction_factor: the factor by which to multiply the minimum non-zero value
    :return: updated series
    """
    nz_ix = x > 0
    if nz_ix.sum() == 0:
        logger.warning('When attempting to replace zeros, there were no non-zero values in series')
        return x
    min_val = (x[nz_ix]).min()
    return x.replace(0, reduction_factor * min_val)


class DataLabeller(object):
    """
    Represents a data labelling tool for preparing and managing clustered datasets for training
    purposes. The class is responsible for processing various input datasets, filtering sequences
    and clusters based on quality metrics, and combining them into a labelled training dataset.

    The `DataLabeller` also facilitates utility functions for file management and projection
    visualizations.

    :ivar output_dir: Directory path for storing output files.
    :type output_dir: str
    :ivar embeddings_file: File path containing embedding data.
    :type embeddings_file: str
    :ivar clustering_file: File path containing clustering results.
    :type clustering_file: str
    :ivar faidx_file: File path of the fasta index associated with sequences.
    :type faidx_file: str
    :ivar spurious_file: File path containing spurious contact predictions.
    :type spurious_file: str
    :ivar all_contacts_file: File path containing all generic contact data.
    :type all_contacts_file: str
    :ivar excluded_file: File path for the list of excluded sequences.
    :type excluded_file: str
    :ivar binning_qc_file: File path containing quality control metrics for binning.
    :type binning_qc_file: str
    :ivar qc_method: Method used for quality control of clusters (default `'CheckMv1'`).
    :type qc_method: str
    :ivar use_suspected: Flag indicating whether suspected intra-cluster sequences should
        be considered (default `False`).
    :type use_suspected: bool
    """

    _SMALL_UF = 1e-3
    _SMALL_COV = 1
    _GROUP_A = 1
    _GROUP_B = 2
    _MIN_NUM_OBS = 2
    _MIN_EXTENT = 100_000
    _BIG_EXTENT = 500_000

    _SUSP_MIN_CLUSTER_EXTENT = 1_000_000
    _SUSP_MIN_SEQ_LENGTH = 500_000
    _SUSP_MIN_SIM = 0.7
    _SUSP_MIN_NUM_OBS = 10
    _SUSP_MIN_DEGREE = 1

    OUTPUT_TABLES: ClassVar[Dict[str, str]] = {
        'training': 'training.csv',
        'undecided': 'undecided.csv',
        'combined': 'combined.csv',
        'spurious_acceptable_clusters': 'spurious_acceptable_clusters.csv',
        'intra_acceptable_clusters': 'intra_acceptable_clusters.csv',
        'moderate_quality_clusters': 'moderate_quality_clusters.csv',
    }

    @staticmethod
    def get_output_path(parent_dir: str, table_name: str) -> str:
        """
        Constructs and returns the output path for a specified table by combining the
        parent directory with the table's designated name.

        :param parent_dir: The base directory in which the output file should be located
        :param table_name: The identifier for the table whose path should be retrieved
        :return: A string representing the full file path for the specified table
        """
        return os.path.join(parent_dir, DataLabeller.OUTPUT_TABLES[table_name])

    def __init__(self,
                 output_dir: str,
                 embeddings_file: str,
                 clustering_file: str,
                 faidx_file: str,
                 spurious_file: str,
                 all_contacts_file: str,
                 excluded_file: str,
                 binning_qc_file: str,
                 qc_method: str='CheckMv1',
                 use_suspected: bool=False,
                 plot_projection: bool=False,
                 plot_max_clusters: int=10) -> None:
        self.output_dir = output_dir
        self.embeddings_file = embeddings_file
        self.clustering_file = clustering_file
        self.faidx_file = faidx_file
        self.spurious_file = spurious_file
        self.all_contacts_file = all_contacts_file
        self.excluded_file = excluded_file
        self.binning_qc_file = binning_qc_file
        self.use_suspected = use_suspected
        self.qc_method = qc_method

        self.cluster_filter = ClusterFilter(self.binning_qc_file)

        self.embeddings = MetagenomeEmbeddings(self.embeddings_file,
                                               self.clustering_file,
                                               self.faidx_file)

        self.excluded_seqs = set(robust_read_csv(self.excluded_file,
                                                 {'seq': str}).seq.values)
        logger.info(f'There were {len(self.excluded_seqs)} sequences excluded from '
                    'clustering and therefore from training.')

        if plot_projection:
            self.embeddings.plot_scatter_projection(
                os.path.join(output_dir, 'Embedding_UMAP_Manhattan_projection.svg'),
                max_clusters=plot_max_clusters)

    def write_table(self,
                    df: pd.DataFrame,
                    table_name: str,
                    description: str,
                    index: bool) -> None:
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

    def prepare_labelled_training_data(self) -> pd.DataFrame:
        """
        Prepares data for labelled training by combining spurious and real contacts datasets into a
        structured training DataFrame, applying multiple filtering and processing steps.

        This function processes two primary datasets: a spurious contacts dataset and a generic real
        contacts dataset. It filters the spurious dataset based on high-quality cluster relationships
        and excluded sequences, reduces redundancy between datasets, calculates similarity measures,
        and assigns group labels indicating their origin. The processed datasets are standardized and
        decomposed into separate pools: general, significant intra-cluster, suspected intra-cluster,
        and undecided records. The final training dataset is prepared by combining labeled spurious
        and intra-cluster data.

        :raises AssertionError: If embeddings have not been analyzed before invoking.
        :return: A DataFrame ready for training, structured with labels indicating spurious and
            intra-cluster relationships.
        :rtype: pandas.DataFrame
        """
        assert self.embeddings is not None, 'The embeddings data must be analysed first'

        # Read initial prediction of spurious contacts
        df_spur = pd.read_csv(self.spurious_file, index_col=0)
        # Read table of all real contacts (not the symbolic closed-sequence to singleton-cluster)
        df_all = pd.read_csv(self.all_contacts_file, index_col=0)

        # Remove unwanted columns
        df_spur = df_spur.drop(columns=list(set(df_spur.columns) & {'cpcc', 'cpss', 'pr_norm'}))
        df_all = df_all.drop(columns=list(set(df_all.columns) & {'cpcc', 'cpss', 'pr_norm'}))

        logger.info(f'Spurious pool: count before exclusion: {len(df_spur)}, all: {len(df_all)}')

        hq_clusters = self.cluster_filter.quality_filter(self.qc_method, 'high')
        logger.info(f'There ae {len(hq_clusters)} high-quality clusters that can be '
                    f'used for inferring spurious contacts')
        # keep a record of those clusters deemed high-quality
        self.write_table(pd.DataFrame({'cluster': list(hq_clusters)}), 'spurious_acceptable_clusters',
                         'clusters deemed high quality', index=False)

        # Reduce false positive rate in spurious table by keeping only
        # contacts involving sufficiently complete and uncontaminated clusters.
        n_before = len(df_spur)
        df_spur = df_spur.query('cluster_name in @hq_clusters')
        logger.info(f'Spurious pool: after filtering for high quality clusters: in={n_before}, out={len(df_spur)}')
        # and any contacts involving sequences intentionally held-out from clustering
        n_before = len(df_spur)
        df_spur = df_spur.query('seq not in @self.excluded_seqs').copy()
        logger.info(f'Spurious pool: after excluding sequences held-out '
                    f'from clustering: in={n_before}, out={len(df_spur)}')

        df_all = anti_join(df_all, df_spur)
        logger.info(f'General pool: after subtracting spurious set: {len(df_all)}')

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
        df_cmb['linkage'] = linkage

        logger.info('Standardising all observations together')
        df_cmb = transform(df_cmb)

        # decompose the two tables
        df_all = df_cmb.query(f'group == {DataLabeller._GROUP_A}').copy()
        df_spur = df_cmb.query(f'group == {DataLabeller._GROUP_B}').set_index(['seq','cluster']).copy()

        # Break down the table of all contacts, where the aim is to identify
        #   those contacts which evidence strongly indicates the contact intra-cellular

        # STEP ONE: basic filter for intuitively sensible seq->cluster relationships.
        # 1. sequence not larger than cluster (this is akin to looking at half the seq->cl contact map).
        # 2. impose a minimum extent on clusters.
        # 3. impose a minimum number of seq->cl contacts (observations).
        df_all = df_all.query(f'contacts > {DataLabeller._MIN_NUM_OBS}'
                              ' and length_u <= length_v'
                              f' and length_v > {DataLabeller._MIN_EXTENT}') \
                       .set_index(['seq','cluster'])
        logger.info(f'General pool: after basic filtering: {len(df_all)} ')

        # STEP TWO: keep those that are intra-cluster contacts (intra=True) where the cluster is of reasonable size
        df_signif = df_all.query(f'intra and length_v > {DataLabeller._BIG_EXTENT}').copy()
        logger.info(f'Intra pool: initial contact count: {len(df_signif)}')

        mq_clusters = self.cluster_filter.quality_filter(self.qc_method, 'partial')
        logger.info(f"There are {len(mq_clusters)} partial (or better) clusters that can be "
                    f"used for inferring intra-cluster contacts")
        self.write_table(pd.DataFrame({'cluster': list(mq_clusters)}), 'intra_acceptable_clusters',
                         'clusters deemed medium quality', index=False)
        n_before = len(df_signif)
        df_signif = df_signif.query('cluster_name in @mq_clusters').copy()
        logger.info(f'Intra pool: after excluding contaminated clusters: in={n_before}, out={len(df_signif)}')

        # STEP THREE: prepare a table of undecided contacts by removing those determined to be reliably intra or inter
        # intra removal
        df_undecided = df_all[~df_all.index.isin(df_signif.index)].copy()
        logger.info(f'Undecided pool: after subtracting those assigned intra: {len(df_undecided)}')
        # spurious removal
        df_undecided = df_undecided[~df_undecided.index.isin(df_spur.index)]
        logger.info(f'Undecided pool: after subtracting those assigned spurious: {len(df_undecided)}')

        # Add initial training labels
        df_spur['intra_z'] = 0
        df_signif['intra_z'] = 1

        # STEP 4: try to find additional "suspected intra-cluster" contacts from even larger
        # clusters, that may be split. These must still be low contamination.
        if self.use_suspected:
            moderate_clusters = self.cluster_filter.quality_filter(self.qc_method, 'moderate')
            logger.info(f"There are {len(moderate_clusters)} moderate-quality (or better) clusters that can be "
                        f"used as to extend the number intra-cluster contacts")
            # keep a record of those clusters deemed as "moderate"
            self.write_table(pd.DataFrame({'cluster': list(moderate_clusters)}), 'moderate_quality_clusters',
                             'clusters selected as moderate quality', index=False)

            df_suspected = identify_suspected_intra(df_undecided.reset_index(), moderate_clusters,
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
        n_dupes = len(df_train) - len(df_train.reset_index().drop_duplicates(["seq", "cluster"]))
        logger.info(f'Checking for duplicate records yielded: {n_dupes}')
        logger.info(f'After basic concatenation, training set contains {len(df_train)} contacts')
        logger.info(f'Undecided set: {len(df_undecided)}')

        # label contacts acceptable for training
        self.write_table(df_train, 'training', 'labelled training data', index=True)
        self.write_table(df_undecided, 'undecided', 'undecided contacts', index=True)
        df_train['train'] = True
        df_undecided['train'] = False

        # combine and reset the basic integer index to get unique values
        df_cmb = pd.concat([df_train.reset_index(), df_undecided.reset_index()]).reset_index(drop=True)
        self.write_table(df_cmb, 'combined', 'final combined table', index=False)

        return df_cmb

import logging
import numpy as np
import pandas as pd
import os
import warnings

from sklearn.metrics.pairwise import linear_kernel
from proxigenomics_toolkit.classify import significance


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
    return df.assign(intra_z   = lambda x: x.intra.astype(np.uint8),
                     # freq_z   = lambda x: scaler(np.log(x.contacts / (x.cov_u*x.cov_v*x.uf_u*x.uf_v*x.sites_u*x.sites_v)))[0],
                     freq_z   = lambda x: scaler(np.log(x.contacts / (x.cov_u*x.cov_v*x.sites_u*x.sites_v)))[0],
                     # effcov_z  = lambda x: scaler(np.log(x.cov_u * x.uf_u * x.cov_v * x.uf_v))[0],
                     density_z = lambda x: scaler(np.log(x.sites_u/x.length_u * x.sites_v/x.length_v))[0],
                     length_z  = lambda x: scaler(np.log(x.length_u * x.length_v))[0],
                     sites_z   = lambda x: scaler(np.log(x.sites_u * x.sites_v))[0],
                     cov_z     = lambda x: scaler(np.log(x.cov_u * x.cov_v))[0],
                     uf_z      = lambda x: scaler(np.arcsin(x.uf_u * x.uf_v))[0],
                     gc_z      = lambda x: scaler(np.arcsin(x.gc_u - x.gc_v))[0],
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
    :param qc_method: eg. CheckMv1, CheckMv2, CoCoPye
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


def identify_suspected_intra(df, hq_clusters, min_similarity, min_contacts, min_cluster_length,
                             max_seq_length, min_degree=2, sort_by='similarity'):

    # number of relevant contacts per sequence
    degree = df.query('contacts>@min_contacts '
                      'and similarity>@min_similarity '
                      'and cluster_name in @hq_clusters') \
        .groupby('seq') \
        .size()

    # reduce the table to only those with sufficiently high degree
    df = df.set_index('seq').loc[degree[degree >= min_degree].index].reset_index()

    suspected_intra = df.query('similarity > @min_similarity'
                               ' and contacts > @min_contacts'
                               ' and length_u < length_v'
                               ' and length_u < @max_seq_length'
                               ' and length_v > @min_cluster_length'
                               ' and cluster_name in @hq_clusters') \
        .sort_values(sort_by, ascending=False) \
        .drop_duplicates('seq', keep='first') \
        .set_index(['seq','cluster'])

    logger.info(f'Suspected intra-cellular contacts: {len(suspected_intra)}')
    return suspected_intra.copy()


def similarity(df, embeddings):
    ix = df[['seq','cluster']].values
    u = embeddings.seq_embeds.loc[ix[:, 0], range(768)].values
    v = embeddings.cluster_embeds.loc[ix[:, 1], range(768)].values
    assert u.shape == v.shape, 'U and V not of the same dimension'
    return np.fromiter((linear_kernel(u[[i]], v[[i]])[0][0] for i in range(u.shape[0])), dtype='f8')


def out_degree_pow(x, p):
    edge_weight = x.contacts / np.power((x.sites_v * x.cov_v * x.uf_v), p)
    return edge_weight / edge_weight.sum()


def prepare_labelled_training_data(output_dir, binning_qc_file, embeddings):
    _SMALL_UF = 1e-3
    _SMALL_COV = 1

    # Read initial prediction of spurious contacts
    df_spur = pd.read_csv(f'{output_dir}/significance_spurious_no_outliers.csv', index_col=0)
    # Read table of all real contacts (not the symbolic closed-sequence to singleton-cluster)
    df_all = pd.read_csv(f'{output_dir}/significance_all_real.csv', index_col=0)
    logger.info(f'Before exclusion counts spurious: {len(df_spur)}, all: {len(df_all)}')

    hq_clusters = high_quality_clusters(binning_qc_file, 90, 10, 'CheckMv1')

    # Reduce false positive rate in spurious table by keeping only
    # contacts involving sufficiently complete and uncontaminated clusters.
    n_before = len(df_spur)
    df_spur = df_spur.query('cluster_name in @hq_clusters').copy()
    logger.info(f'After filtering for high quality clusters: in={n_before}, out={len(df_spur)}')
    df_all = anti_join(df_all, df_spur)
    logger.info(f'Applying exclusion to all-contacts: {len(df_all)}')

    # concatenate the two tables, assigning a group label for later separation.
    df_all['group'] = 1
    df_spur['group'] = 2
    # combine the tables, make sure to remove duplicates but retain those which came
    # from the spurious table
    df_cmb = pd.concat([df_all, df_spur]) \
        .sort_values('group', ascending=False) \
        .drop_duplicates(['seq','cluster'], keep='first')

    # make sure that any occasional zero is instead a small value
    df_cmb.loc[df_cmb.cov_u == 0, 'cov_u'] = _SMALL_COV
    df_cmb.loc[df_cmb.cov_v == 0, 'cov_v'] = _SMALL_COV
    df_cmb.loc[df_cmb.uf_v == 0, 'uf_v'] = _SMALL_UF

    # calculate similarity between sequence and cluster
    logger.info('Calculating similarities')
    df_cmb['similarity'] = similarity(df_cmb, embeddings)

    # reset the index to a simple integer, after first insuring an intuitive ordering
    logger.info('Calculating linkage coefficient')
    df_cmb = df_cmb.sort_values(['seq','cluster']).reset_index(drop=True)
    # calculate linkage coeff and assign
    linkage = df_cmb.groupby('seq').apply(out_degree_pow, p=0.5, include_groups=False)
    df_cmb['linkage'] = linkage.droplevel(0)

    logger.info('Standardising all observations together')
    df_cmb = transform(df_cmb)

    # decompose the two tables
    df_all = df_cmb.query('group==1').copy()
    df_spur = df_cmb.query('group==2').set_index(['seq','cluster']).copy()

    # Break down the table of all contacts, where the aim is to identify
    #   those contacts which evidence strongly indicates the contact intra-cellular

    # STEP ONE: basic filter for intuitively sensible seq->cluster relationships.
    # 1. sequence smaller than cluster (this is akin to looking at half the contact map)
    # 2. cluster minimum extent
    # 3. at least N contacts
    min_contacts = 0
    min_extent = 100_000
    df_all = df_all.query('contacts > @min_contacts and length_u < length_v and length_v > @min_extent') \
                   .set_index(['seq','cluster'])
    logger.info(f'Contacts after basic filtering: {len(df_all)} ')

    # STEP TWO: keep those that are intra-cluster contacts (intra=True) where the cluster is of reasonable size
    bigger_extent = 500_000
    df_signif = df_all.query('intra and length_v > @bigger_extent').copy()
    logger.info(f'Reliable intra-cluster contacts: {len(df_signif)}')

    # STEP THREE: prepare a table of undecided contacts by removing those determined to be reliably intra or inter
    # intra removal
    df_undecided = df_all[~df_all.index.isin(df_signif.index)].copy()
    logger.info(f'Undecided contacts, after removing reliable: {len(df_undecided)}')
    # spurious removal
    df_undecided = df_undecided[~df_undecided.index.isin(df_spur.index)]
    logger.info(f'Undecided contacts, after removing spurious: {len(df_undecided)}')

    # STEP 4: try to find additional "suspected intra-cluster" contacts from even larger
    # clusters, that may be split. These must still be low contamination.
    pure_clusters = high_quality_clusters(binning_qc_file, 50, 10,
                                          'CheckMv1')
    df_suspected = identify_suspected_intra(df_undecided.reset_index(), hq_clusters,
                                            0.7, 10, 1_000_000,
                                            500_000, min_degree=1, sort_by='freq_z')
    df_undecided = df_undecided[~df_undecided.index.isin(df_suspected.index)]
    logger.info(f'Undecided contacts, after removing suspected intra: {len(df_undecided)}')

    # Add labels, where suspected contacts are granted intra=True status
    df_spur['intra_z'] = 0
    df_suspected['intra_z'] = 1
    df_signif['intra_z'] = 1

    df_signif = pd.concat([df_signif, df_suspected])
    logger.info(f'Combining significant and suspects, there are now {len(df_signif)} intra-cluster contacts')

    df_train = pd.concat([df_spur, df_signif])
    logger.info(f'Checking for duplicate records yielded: {len(df_train) - len(df_train.reset_index().drop_duplicates(["seq","cluster"]))}')
    logger.info(f'After basic concatenation, training set contains {len(df_train)} contacts')
    logger.info(f'Undecided set: {len(df_undecided)}')

    # label contacts acceptable for training
    df_train.reset_index().to_csv(f'{output_dir}/training.csv', index=False)
    df_undecided.reset_index().to_csv(f'{output_dir}/undecided.csv', index=False)
    df_train['train'] = True
    df_undecided['train'] = False

    # combine and reset the basic integer index to get unique values
    df_cmb = pd.concat([df_train.reset_index(), df_undecided.reset_index()]).reset_index(drop=True)
    df_cmb.to_csv(f'{output_dir}/combined.csv', index=False)

    return df_cmb


# if __name__ == '__main__':
#     import argparse
#     import keras
#
#     parser = argparse.ArgumentParser(description='Predict significant Hi-C contacts between sequences and bins')
#     parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose output')
#     parser.add_argument('-s', '--seed', default=None, type=int)
#     parser.add_argument('--max-visible-clusters', default=10, type=int,
#                         help='Maximum number of clusters to display in UMAP plot '
#                              '(ordered by descending extent) [10]')
#     parser.add_argument('--kmer-size', default=50, type=int,
#                         help='K-mer size used in calculating GenMap mappability')
#     parser.add_argument('--n-epochs', default=100, type=int,
#                         help='Number of epochs for training [100]')
#     parser.add_argument('--batch-size', default=50, type=int,
#                         help='Batch size for training [50]')
#     # parser.add_argument('--n-fold', default=5, type=int,
#     #                     help='Number of folds for cross-validation [5]')
#     parser.add_argument('EMBEDDINGS', help='DNABERT-S sequence embeddings')
#     parser.add_argument('CLUSTERING', help='Bin3C clustering file')
#     parser.add_argument('FAIDX', help='Samtools FASTA index file')
#     parser.add_argument('CONTACT_MAP', help='Bin3C contact map')
#     parser.add_argument('COVERAGE', help='Sequence coverage file')
#     parser.add_argument('MAPPABILITY', help='Sequence mappability file')
#     parser.add_argument('MGE_REPORT', help='MGE 2-way report file')
#     parser.add_argument('OUTPUT_DIR', help='Output directory')
#     args = parser.parse_args()
#
#     keras.utils.set_random_seed(args.seed)
#
#     if not os.path.exists(args.OUTPUT_DIR):
#         os.mkdir(args.OUTPUT_DIR)
#
#     #
#     # Set up logging
#     #
#     logging.captureWarnings(True)
#     logger = logging.getLogger('main')
#     logging.getLogger('sklearn').setLevel(logging.ERROR)
#     logging.getLogger('requests').setLevel(logging.ERROR)
#     logging.getLogger('matplotlib').setLevel(logging.ERROR)
#
#     # root log listens to everything
#     root = logging.getLogger('')
#     root.setLevel(logging.INFO)
#
#     # log message format
#     formatter = logging.Formatter(fmt='%(levelname)-8s | %(asctime)s | %(name)7s | %(message)s')
#
#     # Runtime console listens to INFO by default
#     ch = logging.StreamHandler()
#     if args.verbose:
#         ch.setLevel(logging.DEBUG)
#     else:
#         ch.setLevel(logging.INFO)
#     ch.setFormatter(formatter)
#     root.addHandler(ch)
#
#     # # File log listens to all levels from root
#     # if args.log is not None:
#     #     log_path = args.log
#     # else:
#     #     log_path = os.path.join(args.OUTDIR, 'classifier.log')
#     # fh = logging.FileHandler(log_path, mode='a')
#     # fh.setLevel(logging.DEBUG)
#     # fh.setFormatter(formatter)
#     # root.addHandler(fh)
#
#     sig_links = significance.SignificantLinks(args.CONTACT_MAP,
#                                               args.CLUSTERING,
#                                               args.COVERAGE,
#                                               args.MAPPABILITY, args.kmer_size,
#                                               args.OUTPUT_DIR,
#                                               args.seed)
#
#     sig_links.prepare_data(excluded_clusters=[], excluded_sequences=[],
#                            min_seq_length=1_000, min_bin_length=100_000, min_bin_size=1, outlier_rejection=True,
#                            big_threshold=1, small_value=1, initial_sigma=3, min_prob=0.01, plot_outliers=True)
#
#     embeddings = MetagenomeEmbeddings(args.EMBEDDINGS,
#                                       args.CLUSTERING,
#                                       args.FAIDX)
#     embeddings.plot_scatter_projection(
#         os.path.join(args.OUTPUT_DIR, 'Embedding_UMAP_Manhattan_projection.svg'),
#         max_clusters=args.max_visible_clusters)
#
#     df_labeled = prepare_labelled_training_data(args.OUTPUT_DIR,
#                                                  args.MGE_REPORT,
#                                                  embeddings)
#
#     classifier = ContactClassifier(df_labeled,
#                                    args.seed,
#                                    args.n_epochs,
#                                    args.batch_size,
#                                    args.OUTPUT_DIR)
#     classifier.train_full_model()
#     classifier.classify(df_labeled)

import gzip
import pickle
import logging

import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import os
import umap
import warnings

from imblearn.under_sampling import RandomUnderSampler
from matplotlib.backends.backend_pdf import PdfPages
from plotnine import *
from scikeras.wrappers import KerasClassifier
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall, Metric
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import L2

from ..contact_map import significance
from ..io_utils import load_object

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def center_of_mass(embeds):
    """
    Given the sequences involved in a cluster, calculate the center of mass
    embedding vector (weighted by sequence length).
    :param embeds:
    :return: normalised CoM of cluster
    """
    if len(embeds) == 1:
        return (embeds.values[:,:768]).flatten()
    v = embeds.iloc[:, :768].values
    l = embeds.loc[:, ['length']].values
    return ((l * v).sum(axis=0) / l.sum()).flatten()


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



def kfold_model_training(seed, n_folds, n_epochs, batch_size, X, y, out_dir):

    tf.keras.backend.clear_session()

    hidden_layer_sizes = [72]*4

    # options = {'activation': 'relu',
    #            'dropout_rate': 0.3,
    #            'l2_bias': 0.026,
    #            'l2_kernel': 0.067,
    #            }

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    validation = {'precision': [],
                  'recall': [],
                  'accuracy': [],
                  'loss': [],
                  'crossentropy': [],
                  'fbeta': [],
                  }

    histories = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    logger.info(f'Number samples: train {X_train.shape[0]}, test {X_test.shape[0]}, all {X.shape[0]}')

    for n_fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train), start=1):

        logger.info(f"Computing fold: {n_fold}")

        Xf, yf = X_train[train_index], y_train[train_index]
        Xv, yv = X_train[val_index], y_train[val_index]

        best_model_file = f'{out_dir}/kfold_bestmodel_{n_fold}.keras'

        checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_file,
                                                        monitor='val_precision', verbose=0,
                                                        save_best_only=True, mode='max')

        # earlystop = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1)

        estimator = KerasClassifier(model=create_baseline,
                                    epochs=n_epochs,
                                    batch_size=batch_size,
                                    random_state=seed,
                                    verbose=0,
                                    callbacks=[checkpoint], #earlystop],
                                    hidden_layer_sizes=hidden_layer_sizes, )

        model = estimator.fit(Xf, yf, validation_data=(Xv, yv))

        histories.append(pd.DataFrame(model.history_))

        m = model.model_
        m.load_weights(best_model_file)
        results = m.evaluate(X_test, y_test, batch_size=250)
        results = dict(zip(m.metrics_names, results))
        for k, v in results.items():
            validation[k].append(v)

        tf.keras.backend.clear_session()

    for k in validation:
        logger.info(f'metric: {k:<15} mean: {np.mean(validation[k]):8.5f}, sd: {np.std(validation[k]):8.5f}')

    for n, df in enumerate(histories, start=1):
        df['fold'] = n


    df_plot = pd.concat(histories)
    _tra = df_plot.loc[:, ~df_plot.columns.str.startswith('val_') | (df_plot.columns == 'fold')].copy()
    _tra['set_type' ] = 'training'
    _val = df_plot.loc[:, df_plot.columns.str.startswith('val_') | (df_plot.columns == 'fold')].copy()
    _val['set_type' ] = 'validation'
    _val.columns = _tra.columns
    df_plot = pd.concat([_val, _tra]).reset_index()
    df_plot.melt(id_vars=['index', 'set_type', 'fold'])

    p = (ggplot(df_plot.query('index>=1').melt(id_vars=['index', 'set_type', 'fold']))
         + geom_point(aes(x='index', y='value', group='set_type', color='set_type'), size=0.5)
         + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6], legend_position="right"))

    p.save(filename=f'{out_dir}/kfold_model.png', dpi=300, verbose=False)
    p.save(filename=f'{out_dir}/kfold_model.svg', verbose=False)


class MetagenomeEmbeddings(object):

    def __init__(self, embeddings_file, clustering_file, faidx_file):
        self.embeddings_file = embeddings_file
        self.clustering_file = clustering_file
        self.faidx_file = faidx_file
        self.chunk_names = None
        self.chunk_embeds = None
        self.seq_names = None
        self.seq_embeds = None
        self.cluster_embeds = None

        self.load_embeddings()
        self.calculate_cluster_embeddings()

    def load_embeddings(self):
        """
        Load the embeddings dictionary produced by the tool seq_embed. Calculate
        the average normalised embedding for each sequence.
        """
        with gzip.open(self.embeddings_file, 'rb') as in_h:
            embeds_dict = pickle.load(in_h)

        chunk_names = []
        seq_names = []
        chunk_embeds = []
        seq_embeds = []

        for k, v in embeds_dict.items():
            chunk_names.extend([k] * v.shape[0])
            chunk_embeds.extend([v])

            seq_names.append(k)
            if v.shape[0] == 1:
                seq_embeds.append(v[0].reshape(1, -1))
            else:
                seq_embeds.append(v.mean(axis=0).reshape(1, -1))

        chunk_embeds = pd.DataFrame(chunk_names, columns=['seq']).join(pd.DataFrame(
            normalize(np.concatenate(chunk_embeds, axis=0), norm='l2')))

        seq_embeds = pd.DataFrame({'seq': seq_names}).join(pd.DataFrame(
            normalize(np.concatenate(seq_embeds, axis=0), norm='l2'))).set_index('seq')

        logger.info(f'Number of chunk embeddings: {len(chunk_embeds)}')
        logger.info(f'Number of sequence embeddings: {len(seq_embeds)}')

        self.chunk_embeds = chunk_embeds
        self.seq_embeds = seq_embeds

    def calculate_cluster_embeddings(self):
        """
        Using the sequence embeddings, calculate a Center of Mass embedding vector for each cluster.
        While performing this calculation, assign the cluster ID to any sequence involved in
        a cluster.
        """
        # We need a source of all sequence lengths, as the clustering
        #   solution can be missing references to some sequences.
        # This is a byproduct of sequences being included in a clustering
        #   result that would not participants in Hi-C.
        fai = pd.read_csv(self.faidx_file, sep='\t', header=None) \
            .drop(columns=range(2,5)) \
            .rename(columns={0: 'seq', 1: 'length'}) \
            .set_index('seq')

        _df = self.seq_embeds.join(fai, how='inner', validate='one_to_one')
        self.seq_embeds['cluster'] = None

        clustering = load_object(self.clustering_file)

        # Calculate mean embedding per cluster.
        cluster_embeds = []
        for cl_id, cl_info in clustering.items():
            cluster_embeds.append(center_of_mass(_df.loc[cl_info['seq_names']]))
            self.seq_embeds.loc[cl_info['seq_names'], 'cluster'] = cl_id

        cluster_embeds = pd.DataFrame(clustering.keys(), columns=['cluster']) \
            .join(pd.DataFrame(
            normalize(np.array(cluster_embeds), norm='l2')), how='inner', validate='1:1')

        logger.info(f'Number of cluster CoM embeddings: {len(cluster_embeds)}')
        self.cluster_embeds = cluster_embeds

    def plot_scatter_projection(self, output_path, max_clusters=10, verbose=False,
                                metric='manhattan', n_components=2, n_epochs=500, min_dist=0.2):
        """
        Plot the UMAP projection (default 2 component) of the embeddings, with the cluster centers overlaid.

        :param output_path: file path to save images (extension dictates format)
        :param max_clusters: number of clusters to display
        :param verbose: verbosity of UMAP call
        :param metric: distance metric used in projection
        :param n_components: number of components in project (only first 2 are plotted)
        :param n_epochs: UMAP epochs
        :param min_dist: UMAP minimum distance
        """

        # prepare a model using the chunked embeddings
        model = umap.UMAP(verbose=verbose, metric=metric, n_epochs=n_epochs,
                          n_components=n_components, min_dist=min_dist)
        model.fit(self.chunk_embeds.loc[:, range(768)])

        # apply the transformation
        chunk_2d = model.transform(self.chunk_embeds.loc[:, range(768)])
        seq_2d = model.transform(self.seq_embeds.loc[:, range(768)])
        cl_2d = model.transform(self.cluster_embeds.loc[:, range(768)])

        # prepare a table linking chunk index to cluster assignment
        _df = self.seq_embeds[['cluster']].join(self.chunk_embeds.set_index('seq')).reset_index()[['cluster']]
        # keep the first N clusters -- bin3C orders clusters largest (extent) to smallest
        accepted_cl = _df.query('cluster < @max_clusters')
        accepted_ix = accepted_cl.index

        p = (ggplot()
             + geom_point(aes(x=chunk_2d[accepted_ix, 0], y=chunk_2d[accepted_ix, 1],
                              colour='factor(accepted_cl["cluster"]+1)'), size=0.5, alpha=0.67)
             + geom_point(aes(x=cl_2d[:max_clusters, 0],
                              y=cl_2d[:max_clusters, 1]), fill='red', color='white', size=4, alpha=1)
             + theme(figure_size=[12,9], aspect_ratio=1)
             + labs(x='dim1', y='dim2', colour='Cluster ID'))

        # p.save(filename=f'{output_dir}/Embedding_UMAP_Manhattan_projection.png', dpi=300, verbose=False)
        p.save(filename=output_path, verbose=False)


class StatefullBinaryFBeta(Metric):
    """
    Custom metric for fbeta maximisation
    """

    def __init__(self, name='fbeta', beta=1, threshold=0.5, epsilon=1e-7, **kwargs):
        # initializing an object of the super class
        super(StatefullBinaryFBeta, self).__init__(name=name, **kwargs)

        # initializing state variables
        self.tp = self.add_weight(name='tp', initializer='zeros') # initializing true positives
        self.actual_positive = self.add_weight(name='fp', initializer='zeros') # initializing actual positives
        self.predicted_positive = self.add_weight(name='fn', initializer='zeros') # initializing predicted positives

        # initializing other atrributes that wouldn't be changed for every object of this class
        self.beta_squared = beta**2
        self.threshold = threshold
        self.epsilon = epsilon

    def update_state(self, ytrue, ypred, sample_weight=None):
        # casting ytrue and ypred as float dtype
        ytrue = tf.cast(ytrue, tf.float32)
        ypred = tf.cast(ypred, tf.float32)

        # setting values of ypred greater than the set threshold to 1 while those lesser to 0
        ypred = tf.cast(tf.greater_equal(ypred, tf.constant(self.threshold)), tf.float32)

        self.tp.assign_add(tf.reduce_sum(ytrue*ypred)) # updating true positives atrribute
        self.predicted_positive.assign_add(tf.reduce_sum(ypred)) # updating predicted positive atrribute
        self.actual_positive.assign_add(tf.reduce_sum(ytrue)) # updating actual positive atrribute

    def result(self):
        self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
        self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall
        # calculating fbeta
        self.fb = (1+self.beta_squared)*self.precision*self.recall / (self.beta_squared*self.precision + self.recall + self.epsilon)
        return self.fb

    def reset_state(self):
        self.tp.assign(0) # resets true positives to zero
        self.predicted_positive.assign(0) # resets predicted positives to zero
        self.actual_positive.assign(0) # resets actual positives to zero


L2_KERNEL = 0.067
L2_BIAS = 0.026
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.0001

def create_baseline(hidden_layer_sizes, meta):
    model = Sequential()
    model.add(Input(shape=(meta['n_features_in_'],)))
    for n, n_nodes in enumerate(hidden_layer_sizes, 1):
        model.add(Dense(n_nodes, kernel_initializer='he_uniform',
                        activation='relu',
                        kernel_regularizer=L2(l2=L2_KERNEL),
                        bias_regularizer=L2(l2=L2_BIAS)))
        if n < len(hidden_layer_sizes):
            model.add(Dropout(DROPOUT_RATE))
            # model.add(Dropout(DROPOUT_RATE, seed=1234))
    model.add(Dense(meta['n_outputs_'], activation='sigmoid'))

    loss_func = BinaryCrossentropy()

    model.compile(loss=loss_func,
                  optimizer=AdamW(learning_rate=LEARNING_RATE),
                  metrics=['accuracy', Precision(), Recall(), StatefullBinaryFBeta(), 'crossentropy'])
    return model


class ContactClassifier(object):

    _FIT_VARS = ['similarity', 'freq_z', 'cov_z', 'linkage']
    _CLASS_VAR = 'intra_z'

    def __init__(self, df_contacts, seed, n_epochs, batch_size, output_dir):
        self.df_contacts = df_contacts
        self.df_train = df_contacts.query('train==True')
        self.seed = seed
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.monitor_metric = 'fbeta'
        self.model = None

    @staticmethod
    def _get_fit_variables(df):
        return df.loc[:, ContactClassifier._FIT_VARS].values

    @staticmethod
    def _get_class_variable(df):
        return df.loc[:, ContactClassifier._CLASS_VAR].values

    @staticmethod
    def _make_table(X, y):
        df = pd.DataFrame({ContactClassifier._CLASS_VAR: y})
        df[ContactClassifier._FIT_VARS] = X
        return df

    def plot_variable_scatter(self, df, base_name, n_points=5000):
        with PdfPages(os.path.join(self.output_dir, base_name)) as pdf:
            if len(df) > n_points:
                df = df.sample(n_points, random_state=self.seed)
            pdf.savefig(sb.jointplot(df, x='similarity', y='freq_z', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='similarity', y='cov_z', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='similarity', y='linkage', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='freq_z', y='cov_z', hue="intra_z").figure)

    def apply_imbalanced_data_augmentation(self):

        # Apply data augmentation to equalise the training classes sizes
        #  - random under-sampling
        self.plot_variable_scatter(self.df_train, 'raw_training_scatter.pdf')

        X = ContactClassifier._get_fit_variables(self.df_train)
        y = ContactClassifier._get_class_variable(self.df_train)
        logger.info(f'Original set size:  X={X.shape}, y={y.shape}, class sizes: {np.bincount(y)}')

        sampler = RandomUnderSampler(random_state=self.seed)
        logger.info('Applying random under-sampling to balance classes')
        X_aug, y_aug = sampler.fit_resample(X, y)
        logger.info('After application of random under-sampling: '
                    f'X={X_aug.shape}, y={y_aug.shape}, class sizes: {np.bincount(y_aug)}')

        df_aug = ContactClassifier._make_table(X_aug, y_aug)
        self.plot_variable_scatter(df_aug, 'augmented_training_scatter.pdf')

        return X, y, X_aug, y_aug

    def train_full_model(self):

        X, y, X_aug, y_aug = self.apply_imbalanced_data_augmentation()

        tf.keras.backend.clear_session()

        best_model = f'{self.output_dir}/full_best_{self.monitor_metric}.keras'

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            best_model,
            monitor=self.monitor_metric,
            verbose=False,
            save_best_only=True,
            mode='max')

        estimator = KerasClassifier(model=create_baseline,
                                    epochs=self.n_epochs,
                                    batch_size=self.batch_size,
                                    random_state=self.seed,
                                    verbose=False,
                                    callbacks=[checkpoint],
                                    hidden_layer_sizes=[72]*4)

        logging.info('Beginning model training')
        model = estimator.fit(X_aug, y_aug)

        logger.info('Loading best model weights')
        model.model_.load_weights(best_model)
        self.model = model

        logger.info(f'Full model score: {model.score(X, y)}')

        df_plot = pd.DataFrame(model.history_).reset_index().rename(columns={'index': 'epoch'})
        p = (ggplot(df_plot.query('epoch>=1').melt(id_vars='epoch'))
             + geom_line(aes(x='epoch', y='value'), color='red')
             + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6]))
        p.save(filename=f'{self.output_dir}/full_model.png', dpi=300, verbose=False)
        p.save(filename=f'{self.output_dir}/full_model.svg', verbose=False)

    def classify(self, df):
        # self.full_model_fit()
        assert self.model is not None, 'Model has not been trained.'
        X = ContactClassifier._get_fit_variables(df)
        pred_significance = self.model.predict_proba(X)
        df['prob_intra'] = pred_significance[:, 1]
        df.to_csv(f'{self.output_dir}/predictions.csv')
        return df


if __name__ == '__main__':
    import argparse
    import keras

    parser = argparse.ArgumentParser(description='Predict significant Hi-C contacts between sequences and bins')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose output')
    parser.add_argument('-s', '--seed', default=None, type=int)
    parser.add_argument('--max-visible-clusters', default=10, type=int,
                        help='Maximum number of clusters to display in UMAP plot '
                             '(ordered by descending extent) [10]')
    parser.add_argument('--kmer-size', default=50, type=int,
                        help='K-mer size used in calculating GenMap mappability')
    parser.add_argument('--n-epochs', default=100, type=int,
                        help='Number of epochs for training [100]')
    parser.add_argument('--batch-size', default=50, type=int,
                        help='Batch size for training [50]')
    # parser.add_argument('--n-fold', default=5, type=int,
    #                     help='Number of folds for cross-validation [5]')
    parser.add_argument('EMBEDDINGS', help='DNABERT-S sequence embeddings')
    parser.add_argument('CLUSTERING', help='Bin3C clustering file')
    parser.add_argument('FAIDX', help='Samtools FASTA index file')
    parser.add_argument('CONTACT_MAP', help='Bin3C contact map')
    parser.add_argument('COVERAGE', help='Sequence coverage file')
    parser.add_argument('MAPPABILITY', help='Sequence mappability file')
    parser.add_argument('MGE_REPORT', help='MGE 2-way report file')
    parser.add_argument('OUTPUT_DIR', help='Output directory')
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)

    if not os.path.exists(args.OUTPUT_DIR):
        os.mkdir(args.OUTPUT_DIR)

    #
    # Set up logging
    #
    logging.captureWarnings(True)
    logger = logging.getLogger('main')
    logging.getLogger('sklearn').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    # root log listens to everything
    root = logging.getLogger('')
    root.setLevel(logging.INFO)

    # log message format
    formatter = logging.Formatter(fmt='%(levelname)-8s | %(asctime)s | %(name)7s | %(message)s')

    # Runtime console listens to INFO by default
    ch = logging.StreamHandler()
    if args.verbose:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # # File log listens to all levels from root
    # if args.log is not None:
    #     log_path = args.log
    # else:
    #     log_path = os.path.join(args.OUTDIR, 'classifier.log')
    # fh = logging.FileHandler(log_path, mode='a')
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)
    # root.addHandler(fh)

    sig_links = significance.SignificantLinks(args.CONTACT_MAP,
                                              args.CLUSTERING,
                                              args.COVERAGE,
                                              args.MAPPABILITY, args.kmer_size,
                                              args.OUTPUT_DIR,
                                              args.seed)

    sig_links.prepare_data(excluded_clusters=[], excluded_sequences=[],
                           min_seq_length=1_000, min_bin_length=100_000, min_bin_size=1, outlier_rejection=True,
                           big_threshold=1, small_value=1, initial_sigma=3, min_prob=0.01, plot_outliers=True)

    embeddings = MetagenomeEmbeddings(args.EMBEDDINGS,
                                      args.CLUSTERING,
                                      args.FAIDX)
    embeddings.plot_scatter_projection(
        os.path.join(args.OUTPUT_DIR, 'Embedding_UMAP_Manhattan_projection.svg'),
        max_clusters=args.max_visible_clusters)

    df_labeled = prepare_labelled_training_data(args.OUTPUT_DIR,
                                                 args.MGE_REPORT,
                                                 embeddings)

    classifier = ContactClassifier(df_labeled,
                                   args.seed,
                                   args.n_epochs,
                                   args.batch_size,
                                   args.OUTPUT_DIR)
    classifier.train_full_model()
    classifier.classify(df_labeled)

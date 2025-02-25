import gzip
import pickle

import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import umap
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

from proxigenomics_toolkit.contact_map import significance
from proxigenomics_toolkit.io_utils import load_object


def load_embeddings(embed_file):

    with gzip.open(embed_file, 'rb') as in_h:
        emb = pickle.load(in_h)

    chunk_names = []
    chunk_embed = []
    seq_names = []
    seq_embed = []
    for k, v in emb.items():

        chunk_names.extend([k] * v.shape[0])
        chunk_embed.extend([v])

        seq_names.append(k)
        if v.shape[0] == 1:
            seq_embed.append(v[0].reshape(1, -1))
        else:
            seq_embed.append(v.mean(axis=0).reshape(1, -1))

    chunk_embed = pd.DataFrame(chunk_names, columns=['seq']) \
        .join(pd.DataFrame(
        normalize(np.concatenate(chunk_embed, axis=0), norm='l2')))

    seq_embed = pd.DataFrame({'seq': seq_names}) \
        .join(pd.DataFrame(
        normalize(np.concatenate(seq_embed, axis=0), norm='l2'))).set_index('seq')

    print(f'Number of chunk embeddings: {len(chunk_embed)}\n'
          f'Number of avg seq embeddings: {len(seq_embed)}')

    return seq_embed, chunk_embed

def center_of_mass(embeds):
    if len(embeds) == 1:
        return (embeds.values[:,:768]).flatten()
    v = embeds.iloc[:, :768].values
    l = embeds.loc[:, ['length']].values
    return ((l * v).sum(axis=0) / l.sum()).flatten()


def calculate_cluster_embeddings(seq_embed, clustering, fai_file):

    # We need a source of all sequence lengths, as the clustering
    #   solution can be missing references to some sequences.
    # This is a byproduct of sequences being included in a clustering
    #   result that would not participants in Hi-C.
    fai = pd.read_csv(fai_file, sep='\t', header=None) \
        .drop(columns=range(2,5)) \
        .rename(columns={0: 'seq', 1: 'length'}) \
        .set_index('seq')

    _df = seq_embed.join(fai, how='inner', validate='one_to_one')

    seq_embed['cluster'] = None

    # Calculate mean embedding per cluster.
    cl_embed = []
    for cl_id, cl_info in clustering.items():
        cl_embed.append(center_of_mass(_df.loc[cl_info['seq_names']]))
        seq_embed.loc[cl_info['seq_names'], 'cluster'] = cl_id
    del _df

    cl_embed = pd.DataFrame(clustering.keys(), columns=['cluster']) \
        .join(pd.DataFrame(
        normalize(np.array(cl_embed), norm='l2')), how='inner', validate='1:1')

    print(f'Cluster mean embeddings: {len(cl_embed)}')
    return cl_embed

def plot_embedding(output_dir, chunk_embed, seq_embed, cl_embed, max_clusters=10,
                   verbose=True, metric='manhattan', n_components=2, n_epochs=500,
                   min_dist=0.2):

    # prepare a model using the chunked embeddings
    model = umap.UMAP(verbose=verbose, metric=metric, n_epochs=n_epochs,
                      n_components=n_components, min_dist=min_dist)
    model.fit(chunk_embed.loc[:, range(768)])

    # apply the transformation
    chunk_2d = model.transform(chunk_embed.loc[:, range(768)])
    seq_2d = model.transform(seq_embed.loc[:, range(768)])
    cl_2d = model.transform(cl_embed.loc[:, range(768)])

    # prepare a table linking chunk index to cluster assignment
    _df = seq_embed[['cluster']].join(chunk_embed.set_index('seq')).reset_index()[['cluster']]
    # keep the first N clusters -- bin3C orders clusters largest (extent) to smallest
    accepted_cl = _df.query('cluster < @max_clusters')
    accpeted_ix = accepted_cl.index

    p = (ggplot()
         + geom_point(aes(x=chunk_2d[accpeted_ix, 0], y=chunk_2d[accpeted_ix, 1], colour='factor(accepted_cl["cluster"]+1)'), size=0.5, alpha=0.67)
         + geom_point(aes(x=cl_2d[:max_clusters, 0], y=cl_2d[:max_clusters, 1]), fill='red', color='white', size=4, alpha=1)
         + theme(figure_size=[12,9], aspect_ratio=1)
         + labs(x='dim1', y='dim2', colour='Cluster ID'))
    ggsave(p, filename=f'{output_dir}/Embedding_UMAP_Manhattan_projection.png', dpi=300)
    ggsave(p, filename=f'{output_dir}/Embedding_UMAP_Manhattan_projection.svg')


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
    excl_idx = exclude_from.set_index(['seq','cluster']).index
    target = target.set_index(['seq','cluster'])
    mask = target.index.isin(excl_idx)
    return target[~ mask].reset_index(), mask.sum()

def filter_clusters(df_checkm, min_comp, max_con):
    return set(df_checkm.query('Contamination <= @max_con and Completeness >= @min_comp').index)

def exclude_lowqual_clusters(df_target, df_checkm, min_comp, max_con):
    n_in = len(df_target)
    accepted_clusters = filter_clusters(df_checkm, min_comp, max_con)
    print(f'There were {len(accepted_clusters)} accepted clusters.')
    result = df_target.query('cluster in @accepted_clusters')
    n_out = len(result)
    print(f'Removing contacts involving acceptable quality clusters ' +
          f'Completeness >= {min_comp}% and Contamination <= {max_con}%: in={n_in}, out={n_out}')
    return result

def identify_suspected_intra(df, min_similarity, min_contacts, min_cluster_length, max_seq_length, min_degree=2, sort_by='similarity'):

    # number of relevant contacts per sequence
    degree = df.query('contacts>@min_contacts and similarity>@min_similarity') \
        .groupby('seq') \
        .size()

    # reduce the table to only those with sufficiently high degree
    df = df.set_index('seq').loc[degree[degree >= min_degree].index].reset_index()

    suspected_intra = df.query('similarity > @min_similarity'
                               ' and contacts > @min_contacts'
                               ' and length_u < length_v'
                               ' and length_u < @max_seq_length'
                               ' and length_v > @min_cluster_length') \
        .sort_values(sort_by, ascending=False) \
        .drop_duplicates('seq', keep='first') \
        .set_index(['seq','cluster'])

    print(f'Identified {len(suspected_intra)} suspected intra-cellular contacts')
    return suspected_intra.copy()

def similarity(df, seq_embeddings, cl_embeddings):
    ix = df[['seq','cluster']].values
    u = seq_embeddings.loc[ix[:, 0], range(768)].values
    v = cl_embeddings.loc[ix[:, 1], range(768)].values
    assert u.shape == v.shape, 'U and V not of the same dimension'
    return np.fromiter((linear_kernel(u[[i]], v[[i]])[0][0] for i in range(u.shape[0])), dtype='f8')

def out_degree_pow(x, p):
    edge_weight = x.contacts / np.power((x.sites_v * x.cov_v * x.uf_v), p)
    return edge_weight / edge_weight.sum()


def plot_scatter(output_path, df, n_points=5000, seed=1234):
    with PdfPages(output_path) as pdf:
        df = df.query('train').sample(n_points, random_state=seed)
        pdf.savefig(sb.jointplot(df, x='similarity', y='freq_z', hue="intra_z").figure)
        pdf.savefig(sb.jointplot(df, x='similarity', y='cov_z', hue="intra_z").figure)
        pdf.savefig(sb.jointplot(df, x='similarity', y='linkage', hue="intra_z").figure)
        pdf.savefig(sb.jointplot(df, x='freq_z', y='cov_z', hue="intra_z").figure)


def prepapre_labelled_training_data(output_dir, mge_report_file,
                                    sequence_embeddings, cluster_embeddings):
    SMALL_UF = 1e-3
    SMALL_COV = 1

    # no outliers
    df_spur = pd.read_csv(f'{args.OUTPUT_DIR}/significance_spurious_no_outliers.csv', index_col=0)
    # all real contacts (not the symbolic closed-sequence to singleton-cluster)
    df_all = pd.read_csv(f'{args.OUTPUT_DIR}/significance_all_real.csv', index_col=0)
    # mge report used to eliminate contaminated bins
    df_checkm = pd.read_csv(mge_report_file, sep='\t')

    print(f'Before exclusion counts spurious:{len(df_spur)}, main table:{len(df_all)}')

    # Exclude "not intra" contacts involving split and/or
    #   contaminated bins (as identified with binning QA tool)
    #   as these could be intra-genomic contacts.
    df_spur = exclude_lowqual_clusters(df_spur, df_checkm, 90, 10)
    df_all, n_masked = anti_join(df_all, df_spur)
    print(f'After excluding spurious, main table:{len(df_all)}, masking {n_masked}')

    # # normalise and standarise as one set, the separate
    print('Standardising all observations together')
    df_all['group'] = 1
    df_spur['group']= 2
    # combine the tables, make sure to remove duplicates but retain those which came
    # from the spurious table
    df_cmb = pd.concat([df_all, df_spur]) \
        .sort_values('group', ascending=False) \
        .drop_duplicates(['seq','cluster'], keep='first')

    # calculate similarity between sequence and cluster
    print('Calculating similarities')
    df_cmb['similarity'] = similarity(df_cmb, sequence_embeddings, cluster_embeddings)
    print('Finished similarity calcs')

    # replace occasional zeros with something small
    df_cmb.loc[df_cmb.cov_u == 0, 'cov_u'] = SMALL_COV
    df_cmb.loc[df_cmb.cov_v == 0, 'cov_v'] = SMALL_COV
    df_cmb.loc[df_cmb.uf_v == 0, 'uf_v'] = SMALL_UF

    # reset the index to a simple integer, after first insuring an intuitive ordering
    df_cmb = df_cmb.sort_values(['seq','cluster']).reset_index(drop=True)
    # calculate linkage coeff and assign
    linkage = df_cmb.groupby('seq').apply(out_degree_pow, p=0.5, include_groups=False)
    df_cmb['linkage'] = linkage.droplevel(0)

    # transform all observations together
    df_cmb = transform(df_cmb)

    # decompose the two tables
    df_all = df_cmb.query('group==1').copy()
    df_spur = df_cmb.query('group==2').set_index(['seq','cluster']).copy()

    # STEP ONE: basic filter to keep just those contacting sensible seq->cluster relationships.
    # 1. sequence smaller than cluster
    # 2. cluster minimum size
    # 3. at least N contacts
    df_all = df_all.query('contacts > 0 and length_u < length_v and length_v > 100_000').set_index(['seq','cluster'])
    print(f'Contacts after filtering: {len(df_all)} ')

    # STEP TWO: keep those that are intra-cluster contacts (intra=True) where the cluster is of reasonable size
    df_signif = df_all.query('intra and length_v > 500_000').copy()
    print(f'Reliable intra-cluster contacts: {len(df_signif)}')

    # STEP THREE: prepare a table of undecided contacts by removing those determined to be reliably intra or inter
    df_undecided = df_all[~df_all.index.isin(df_signif.index)].copy()
    print(f'Undecided contacts, after removing reliable: {len(df_undecided)}')
    df_undecided = df_undecided[~df_undecided.index.isin(df_spur.index)]
    print(f'Undecided contacts, after removing spurious: {len(df_undecided)}')

    # Infer additional suspected intra-cluster contacts a larger set of clusters
    # i.e. clusters which may have been split -- possibly this should be constrained to low-contamination clusters
    df_suspected = identify_suspected_intra(df_undecided.reset_index(), 0.7, 10, 1_000_000, 1_000_000, min_degree=1, sort_by='freq_z')
    df_undecided = df_undecided[~df_undecided.index.isin(df_suspected.index)]
    print(f'Undecided contacts, after removing suspected intra: {len(df_undecided)}')

    # Add labels, where suspected contacts are granted intra=True status
    df_spur['intra_z'] = 0
    df_suspected['intra_z'] = 1
    df_signif['intra_z'] = 1

    df_signif = pd.concat([df_signif, df_suspected])
    print(f'Combining significant and suspects, there are now {len(df_signif)} intra-cluster contacts')

    df_train = pd.concat([df_spur, df_signif])
    print(f'Checking for duplicate records yielded: {len(df_train) - len(df_train.reset_index().drop_duplicates(["seq","cluster"]))}')
    print(f'After basic concatenation, training set contains {len(df_train)} contacts')
    print(f'Undecided set: {len(df_undecided)}')

    # label contacts acceptable for training
    df_train.reset_index().to_csv(f'{output_dir}/training.csv', index=False)
    df_undecided.reset_index().to_csv(f'{output_dir}/undecided.csv', index=False)
    df_train['train'] = True
    df_undecided['train'] = False

    # combine and reset the basic integer index to get unique values
    df_cmb = pd.concat([df_train.reset_index(), df_undecided.reset_index()]).reset_index(drop=True)

    return df_cmb


class StatefullBinaryFBeta(Metric):

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

def create_baseline(hidden_layer_sizes, meta):

    model = Sequential()
    model.add(Input(shape=(meta['n_features_in_'],)))

    for n, n_nodes in enumerate(hidden_layer_sizes, 1):
        model.add(Dense(n_nodes, kernel_initializer='he_uniform',
                        activation='relu',
                        kernel_regularizer=L2(l2=0.067),
                        bias_regularizer=L2(l2=0.026)))
        if n < len(hidden_layer_sizes):
            model.add(Dropout(0.3, seed=1234))
    model.add(Dense(meta['n_outputs_'], activation='sigmoid'))

    loss_func = BinaryCrossentropy()

    model.compile(loss=loss_func,
                  optimizer=AdamW(learning_rate=0.0001),
                  metrics=['accuracy', Precision(), Recall(), StatefullBinaryFBeta(), 'crossentropy'])
    return model

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
    print(f'Number samples: train {X_train.shape[0]}, test {X_test.shape[0]}, all {X.shape[0]}')

    for n_fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train), start=1):

        print(f"Computing fold: {n_fold}")

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
        print(f'metric: {k:<15} mean: {np.mean(validation[k]):8.5f}, sd: {np.std(validation[k]):8.5f}')

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

    ggsave(p, filename=f'{out_dir}/kfold_model.png', dpi=300)
    ggsave(p, filename=f'{out_dir}/kfold_model.svg')


def full_model_fit(seed, n_epochs, batch_size, X, y, out_dir, monitor_metric = 'fbeta'):

    tf.keras.backend.clear_session()

    best_model = f'{out_dir}/full_best_{monitor_metric}.keras'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model,
        monitor=monitor_metric,
        verbose=1,
        save_best_only=True,
        mode='max')

    estimator = KerasClassifier(model=create_baseline,
                                epochs=n_epochs,
                                batch_size=batch_size,
                                random_state=seed,
                                verbose=0,
                                callbacks=[checkpoint],
                                hidden_layer_sizes=[72]*4)

    model = estimator.fit(X_aug, y_aug)

    print('Loading best model weights')
    model.model_.load_weights(best_model)

    print(f'Full model score: {model.score(X, y)}')

    df_plot = pd.DataFrame(model.history_).reset_index().rename(columns={'index': 'epoch'})
    p = (ggplot(df_plot.query('epoch>=1').melt(id_vars='epoch'))
         + geom_line(aes(x='epoch', y='value'), color='red')
         + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6]))

    ggsave(p, filename=f'{out_dir}/full_model.png', dpi=300)
    ggsave(p, filename=f'{out_dir}/full_model.svg')

    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict significant Hi-C contacts between sequences and bins')
    parser.add_argument('-s', '--seed', default=None, type=int)
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

    seq_embed, chunk_embed = load_embeddings(args.EMBEDDINGS)

    clustering = load_object(args.CLUSTERING)

    cl_embed = calculate_cluster_embeddings(seq_embed, clustering, args.FAIDX)

    plot_embedding(args.OUTPUT_DIR, chunk_embed, seq_embed, cl_embed)

    # Generate the initial set of contacts between all sequences and clusters.
    # - This writes tabulated results for all and spurious contacts, where
    #   spurious is a heuristic.
    # - This code over-engineered as it was the basis of a nbinom2 model through R,
    #   but was abandoned due to a lack of classification performance during testing.

    sig_links = significance.SignificantLinks(args.CONTACT_MAP,
                                              args.CLUSTERING,
                                              args.COVERAGE,
                                              args.MAPPABILITY, args.kmer_size,
                                              args.OUTPUT_DIR,
                                              args.seed)

    sig_links.prepare_data(excluded_clusters=[], excluded_sequences=[],
                           min_seq_length=1_000, min_bin_length=100_000, min_bin_size=1, outlier_rejection=True,
                           big_threshold=1, small_value=1, initial_sigma=3, min_prob=0.01, plot_outliers=True)

    # Prepare the labelled training data
    df_combined = prepapre_labelled_training_data(args.OUTPUT_DIR,
                                                  args.MGE_REPORT,
                                                  seq_embed,
                                                  cl_embed)

    plot_scatter(f'{args.OUTPUT_DIR}/raw_training_scatter.pdf', df_combined)

    # Apply data augmentation to equalise the training classes sizes
    #  - random under-sampling
    df = df_combined.query('train')
    X = df.loc[:, ['similarity','freq_z', 'cov_z', 'linkage']].values
    y = df.intra_z.values
    print(f'Original set size:  X={X.shape}, y={y.shape}, class sizes: {np.bincount(y)}')

    sampler = RandomUnderSampler(random_state=args.seed)
    X_aug, y_aug = sampler.fit_resample(X, y)
    df_aug = pd.DataFrame({'intra_z': y_aug})
    df_aug[['similarity','freq_z', 'cov_z', 'linkage']] = X_aug
    print(f'Augmented set size: X={X_aug.shape}, y={y_aug.shape}, class sizes: {np.bincount(y_aug)}')

    plot_scatter(f'{args.OUTPUT_DIR}/augmented_training_scatter.pdf', df_aug)

    classifier = full_model_fit(args.seed, args.n_epochs, args.batch_size, X_aug, y_aug, out_dir='.')
    pred_significance = classifier.predict_proba(df_combined[['similarity','freq_z','cov_z','linkage']])
    df_combined['Pr'] = pred_significance[:, 1]

    df_combined.to_csv(f'{args.OUTPUT_DIR}/predictions.csv')

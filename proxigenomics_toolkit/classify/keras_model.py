import seaborn as sb
import tensorflow as tf
import logging
import pandas as pd
import numpy as np
import os

from imblearn.under_sampling import RandomUnderSampler
from matplotlib.backends.backend_pdf import PdfPages
from plotnine import *
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall, Metric
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import L2
import keras

logger = logging.getLogger(__name__)


L2_KERNEL = 0.1
L2_BIAS = 0.02
DROPOUT_RATE = 0.3


class StatefulBinaryFBeta(Metric):
    """
    Custom metric for fbeta maximisation
    """

    def __init__(self, name='fbeta', beta=1.0, threshold=0.5, epsilon=1e-7, **kwargs):
        # initializing an object of the super class
        super(StatefulBinaryFBeta, self).__init__(name=name, **kwargs)

        # initializing state variables
        self.tp = self.add_weight(name='tp', initializer='zeros') # initializing true positives
        self.actual_positive = self.add_weight(name='fp', initializer='zeros') # initializing actual positives
        self.predicted_positive = self.add_weight(name='fn', initializer='zeros') # initializing predicted positives

        # initializing other attributes that wouldn't be changed for every object of this class
        self.beta_squared = beta**2
        self.threshold = threshold
        self.epsilon = epsilon
        self.precision = None
        self.recall = None
        self.fb = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        # casting y_true and y_pred as float dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # setting values of y_pred greater than the set threshold to 1 while those lesser to 0
        y_pred = tf.cast(tf.greater_equal(y_pred, tf.constant(self.threshold)), tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred)) # updating true positives attribute
        self.predicted_positive.assign_add(tf.reduce_sum(y_pred)) # updating predicted positive attribute
        self.actual_positive.assign_add(tf.reduce_sum(y_true)) # updating actual positive attribute

    def result(self):
        self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
        self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall
        # calculating fbeta
        self.fb = (1 + self.beta_squared) * self.precision*self.recall / \
                   (self.beta_squared*self.precision + self.recall + self.epsilon)
        return self.fb

    def reset_state(self):
        self.tp.assign(0) # resets true positives to zero
        self.predicted_positive.assign(0) # resets predicted positives to zero
        self.actual_positive.assign(0) # resets actual positives to zero


def create_baseline(hidden_layer_sizes, learning_rate, meta):
    model = Sequential()
    model.add(Input(shape=(meta['n_features_in_'],)))
    for n, n_nodes in enumerate(hidden_layer_sizes, 1):
        model.add(Dense(n_nodes, kernel_initializer='he_uniform',
                        activation='relu',
                        kernel_regularizer=L2(l2=L2_KERNEL),
                        bias_regularizer=L2(l2=L2_BIAS)))
        if n < len(hidden_layer_sizes):
            model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(meta['n_outputs_'], activation='sigmoid'))

    loss_func = BinaryCrossentropy()

    model.compile(loss=loss_func,
                  optimizer=AdamW(learning_rate=learning_rate),
                  metrics=['accuracy', Precision(), Recall(), StatefulBinaryFBeta(beta=1.0), 'crossentropy'])
    return model


class ContactClassifier(object):

    _METRIC_NAME = 'fbeta'
    _FIT_VARS = ['similarity', 'freq_z', 'cov_z', 'linkage']
    _CLASS_VAR = 'intra_z'

    _HIDDEN_SIZE = 32
    _HIDDEN_DEPTH = 4
    _PATIENCE = 20

    OUTPUT_TABLES = {
        'predictions': 'predictions.csv',
    }

    @staticmethod
    def get_output_path(parent_dir, table_name) -> str:
        return os.path.join(parent_dir, str(ContactClassifier.OUTPUT_TABLES[table_name]))

    def __init__(self, output_dir, complete_labelled_file, seed, n_epochs, batch_size,
                 learning_rate=0.001, enable_tb=False, enable_es=True, verbose=False):
        """
        An MLP classifier for Hi-C contacts, where classification decides if an accumulated contact
        between a single sequence as a genome_bin is intra- or inter- cellular.

        :param output_dir: parent directory to which results are written
        :param complete_labelled_file: labelled training data
        :param seed: a random seed
        :param n_epochs: number of epochs for training
        :param batch_size: batch size for training
        :param learning_rate: global learning rate of AdamW optimizer
        :param enable_tb: enable tensorboard logging
        :param enable_es: enable early stopping callback when training ceases to improve for 20 iterations
        :param verbose: verbosity of logging
        """

        self.output_dir = output_dir
        self.complete_labeled_file = complete_labelled_file
        self.seed = seed
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.enable_tb = enable_tb
        self.enable_es = enable_es
        # read data for training
        self.df_combined = pd.read_csv(complete_labelled_file)
        self.df_train = self.df_combined.query('train==True')
        self.verbose = verbose

    @staticmethod
    def _get_fit_variables(df):
        return df.loc[:, ContactClassifier._FIT_VARS].values

    @staticmethod
    def _get_class_variable(df):
        return df.loc[:, ContactClassifier._CLASS_VAR].values

    @staticmethod
    def _make_table(x, y):
        df = pd.DataFrame({ContactClassifier._CLASS_VAR: y})
        df[ContactClassifier._FIT_VARS] = x
        return df

    def write_table(self, df, table_name, description, index):
        """
        Standardised writing of a table to a file
        :param df: the pandas table
        :param table_name: name of the table to write (obtains file name)
        :param description: a description of logging
        :param index: whether to include
        """
        file_path = ContactClassifier.get_output_path(self.output_dir, table_name)
        logger.info(f'Writing {description} to {file_path}')
        df.to_csv(file_path, index=index)

    def plot_variable_scatter(self, df, base_name, n_points=5000):
        with PdfPages(os.path.join(self.output_dir, base_name)) as pdf:
            if len(df) > n_points:
                df = df.sample(n_points, random_state=self.seed)
            pdf.savefig(sb.jointplot(df, x='similarity', y='freq_z', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='similarity', y='cov_z', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='similarity', y='linkage', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='freq_z', y='cov_z', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='freq_z', y='linkage', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='cov_z', y='linkage', hue="intra_z").figure)

    def apply_imbalanced_data_augmentation(self):

        # Apply data augmentation to equalise the training classes sizes
        #  - random under-sampling
        self.plot_variable_scatter(self.df_train, 'raw_training_scatter.pdf')

        x = ContactClassifier._get_fit_variables(self.df_train)
        y = ContactClassifier._get_class_variable(self.df_train)
        logger.info(f'Original set size:  x={x.shape}, y={y.shape}, class sizes: {np.bincount(y)}')

        sampler = RandomUnderSampler(random_state=self.seed)
        logger.info('Applying random under-sampling to balance classes')
        x_aug, y_aug = sampler.fit_resample(x, y)
        logger.info('After application of random under-sampling: '
                    f'x={x_aug.shape}, y={y_aug.shape}, class sizes: {np.bincount(y_aug)}')

        df_aug = ContactClassifier._make_table(x_aug, y_aug)
        self.plot_variable_scatter(df_aug, 'augmented_training_scatter.pdf')

        return x, y, x_aug, y_aug

    def tensorboard_callback(self):

        return keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.output_dir, 'logs'),
            histogram_freq=1,
            embeddings_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch")

    @staticmethod
    def earlystopping_callback(metric, verbose=False):
        return tf.keras.callbacks.EarlyStopping(monitor=metric,
                                                patience=ContactClassifier._PATIENCE,
                                                mode='max',
                                                min_delta=1e-4,
                                                start_from_epoch=50,
                                                verbose=verbose)
    @staticmethod
    def checkpoint_callback(best_model_file, verbose=False):
        return tf.keras.callbacks.ModelCheckpoint(best_model_file,
                                                  monitor=ContactClassifier._METRIC_NAME,
                                                  verbose=verbose,
                                                  save_best_only=True,
                                                  mode='max')

    # def hp_tuning(self):
    #
    #     from tensorboard.plugins.hparams import api as hp
    #
    #     X, y, X_aug, y_aug = self.apply_imbalanced_data_augmentation()
    #     X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug,
    #                                                         test_size=0.2, random_state=self.seed, stratify=y_aug)
    #
    #     tf.keras.backend.clear_session()
    #
    #     HP_NUM_UNITS = hp.HParam('hidden_size', hp.Discrete([32, 64, 128]))
    #     # HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.3])) #    RealInterval(0.1, 0.5))
    #     HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.00001, 0.001))
    #     # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'adamw', 'sgd', 'rmsprop']))
    #     # HP_L2BIAS = hp.HParam('l2_bias', hp.RealInterval(0.01, 0.15))
    #     # HP_L2KERNEL = hp.HParam('l2_kernel', hp.RealInterval(0.01, 0.1))
    #
    #     with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    #         hp.hparams_config(
    #             hparams=[HP_NUM_UNITS, HP_LEARNING_RATE],
    #             metrics=[hp.Metric(ContactClassifier._METRIC_NAME, display_name='fBeta')],)
    #
    #     def create_hp_model(hparams):
    #         model = Sequential()
    #         model.add(Input(shape=(4,)))
    #         for n in range(1, 5):
    #             model.add(Dense(hparams[HP_NUM_UNITS], kernel_initializer='he_uniform',
    #                             activation='relu',
    #                             kernel_regularizer=L2(l2=L2_KERNEL),
    #                             bias_regularizer=L2(l2=L2_BIAS)))
    #             if n < 4:
    #                 model.add(Dropout(DROPOUT_RATE))
    #         model.add(Dense(1, activation='sigmoid'))
    #
    #         loss_func = BinaryCrossentropy()
    #         model.compile(loss=loss_func,
    #           optimizer=AdamW(learning_rate=hparams[HP_LEARNING_RATE]),
    #           metrics=['accuracy', Precision(), Recall(), StatefulBinaryFBeta(), 'crossentropy'])
    #
    #         model.fit(X_train, y_train, epochs=20)
    #         loss, accuracy, precision, recall, f_beta, cross_entropy = model.evaluate(X_test, y_test)
    #         return f_beta
    #
    #     def experiment(experiment_dir, hparams):
    #         with tf.summary.create_file_writer(experiment_dir).as_default():
    #             hp.hparams(hparams)
    #             accuracy = create_hp_model(hparams)
    #             tf.summary.scalar(ContactClassifier._METRIC_NAME, accuracy, step=1)
    #
    #     experiment_no = 0
    #     for num_units in HP_NUM_UNITS.domain.values:
    #         for lr in np.linspace(HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value, 5):
    #             hparams = {
    #                 HP_NUM_UNITS: num_units,
    #                 HP_LEARNING_RATE: lr, }
    #
    #             experiment_name = f'Experiment {experiment_no}'
    #             print(f'Starting Experiment: {experiment_name}')
    #             print({h.name: hparams[h] for h in hparams})
    #             experiment('logs/hparam_tuning/' + experiment_name, hparams)
    #             experiment_no += 1

    def train_full_model(self):

        x, y, x_aug, y_aug = self.apply_imbalanced_data_augmentation()

        tf.keras.backend.clear_session()

        best_model_file = os.path.join(self.output_dir, f'full_best_{ContactClassifier._METRIC_NAME}.keras')

        callbacks = [self.checkpoint_callback(best_model_file)]
        if self.enable_tb:
            callbacks.append(self.tensorboard_callback())
        if self.enable_es:
            callbacks.append(self.earlystopping_callback('fbeta', verbose=self.verbose))

        estimator = KerasClassifier(model=create_baseline,
                                    epochs=self.n_epochs,
                                    batch_size=self.batch_size,
                                    random_state=self.seed,
                                    verbose=self.verbose,
                                    callbacks=callbacks,
                                    hidden_layer_sizes=[ContactClassifier._HIDDEN_SIZE] * ContactClassifier._HIDDEN_DEPTH,
                                    learning_rate=self.learning_rate)

        logging.info('Beginning model training')
        model = estimator.fit(x_aug, y_aug)

        logger.info('Loading best model weights')
        model.model_.load_weights(best_model_file)
        self.model = model

        logger.info(f'Full model score: {model.score(x, y)}')

        df_plot = pd.DataFrame(model.history_).reset_index().rename(columns={'index': 'epoch'})
        p = (ggplot(df_plot.query('epoch>=1').melt(id_vars='epoch'))
             + geom_line(aes(x='epoch', y='value'), color='red')
             + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6]))
        p.save(filename=os.path.join(self.output_dir,'full_model.svg'), verbose=False)

    def classify(self, df=None):
        assert self.model is not None, 'Model has not been trained.'
        if df is None:
            df = self.df_combined.copy()
        x = ContactClassifier._get_fit_variables(df)
        pred_significance = self.model.predict_proba(x)
        df['prob_intra'] = pred_significance[:, 1]
        self.write_table(df, 'predictions', 'final predictions', index=False)
        return df

    def train_kfold_model(self, n_folds):

        tf.keras.backend.clear_session()

        x, y, x_aug, y_aug = self.apply_imbalanced_data_augmentation()

        hidden_layer_sizes = [ContactClassifier._HIDDEN_SIZE] * ContactClassifier._HIDDEN_DEPTH
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)

        testing = {'precision': [],
                      'recall': [],
                      'accuracy': [],
                      'loss': [],
                      'crossentropy': [],
                      'fbeta': [],
                      }

        histories = []

        x_train, x_test, y_train, y_test = train_test_split(x_aug, y_aug,
                                                            test_size=0.2, random_state=self.seed, stratify=y_aug)

        logger.info(f'Number samples: train {x_train.shape[0]}, test {x_test.shape[0]}, all {x.shape[0]}')

        for n_fold, (train_index, val_index) in enumerate(kfold.split(x_train, y_train), start=1):

            logger.info(f"Computing fold: {n_fold}")

            x_fit, y_fit = x_train[train_index], y_train[train_index]
            x_val, y_val = x_train[val_index], y_train[val_index]

            best_model_file = os.path.join(self.output_dir, f'kfold_bestmodel_{n_fold}.keras')

            callbacks = [self.checkpoint_callback(best_model_file)]
            if self.enable_tb:
                callbacks.append(self.tensorboard_callback())
            if self.enable_es:
                callbacks.append(self.earlystopping_callback('fbeta', verbose=self.verbose))

            estimator = KerasClassifier(model=create_baseline,
                                        epochs=self.n_epochs,
                                        batch_size=self.batch_size,
                                        random_state=self.seed,
                                        verbose=self.verbose,
                                        callbacks=callbacks,
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        learning_rate=self.learning_rate)

            model = estimator.fit(x_fit, y_fit, validation_data=(x_val, y_val))

            histories.append(pd.DataFrame(model.history_))

            m = model.model_
            m.load_weights(best_model_file)
            results = m.evaluate(x_test, y_test, batch_size=250)
            results = dict(zip(m.metrics_names, results))
            for k, v in results.items():
                testing[k].append(v)

            tf.keras.backend.clear_session()

        for k in testing:
            logger.info(f'Test data - {k:>12} mean: {np.mean(testing[k]):8.5f}, sd: {np.std(testing[k]):8.5f}')

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
             # + geom_line(aes(x='index', y='value', group='set_type', color='set_type'))
             + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6], legend_position="right"))
        p.save(filename=os.path.join(self.output_dir, 'kfold_model.svg'), verbose=False)

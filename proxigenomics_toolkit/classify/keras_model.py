import seaborn as sb
import tensorflow as tf
import logging
import pandas as pd
import numpy as np
import os

from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier
from matplotlib.backends.backend_pdf import PdfPages
from plotnine import *
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall, Metric
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import L2
import keras

logger = logging.getLogger(__name__)


L2_KERNEL = 0.15
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
        model.add(Dense(n_nodes,
                        kernel_initializer='he_uniform',
                        bias_initializer='zeros',
                        activation='relu',
                        kernel_regularizer=L2(l2=L2_KERNEL),
                        bias_regularizer=L2(l2=L2_BIAS)))
        if n < len(hidden_layer_sizes):
            model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(meta['n_outputs_'], activation='sigmoid'))

    loss_func = BinaryCrossentropy()

    model.compile(loss=loss_func,
                  optimizer=AdamW(learning_rate=learning_rate,
                                  amsgrad=True),
                  metrics=['accuracy',
                           Precision(),
                           Recall(),
                           StatefulBinaryFBeta(beta=1.0),
                           'crossentropy'])
    return model


class ContactClassifier(object):

    _METRIC_NAME = 'fbeta'
    _FIT_VARS = ['similarity', 'freq_z', 'cov_z', 'linkage']
    _CLASS_VAR = 'intra_z'
    _PATIENCE = 20

    OUTPUT_TABLES = {
        'predictions': 'predictions.csv',
    }

    @staticmethod
    def get_output_path(parent_dir, table_name) -> str:
        return os.path.join(parent_dir, str(ContactClassifier.OUTPUT_TABLES[table_name]))

    def __init__(self,
                 output_dir,
                 complete_labelled_file,
                 hq_cluster_file,
                 seed,
                 n_epochs,
                 batch_size,
                 num_nodes=24,
                 num_layers=5,
                 learning_rate=1e-4,
                 num_estimators=10,
                 test_size=None,
                 enable_bag=False,
                 enable_tb=False, enable_es=True, verbose=False):
        """
        An MLP classifier for Hi-C contacts, where classification decides if an accumulated contact
        between a single sequence as a genome_bin is intra- or inter- cellular.

        :param output_dir: parent directory to which results are written
        :param complete_labelled_file: labelled training data
        :param hq_cluster_file: file containing cluster ids pertaining to those deemed high-quality
        :param seed: a random seed
        :param n_epochs: number of epochs for training
        :param batch_size: batch size for training
        :param num_nodes: number of nodes in the hidden layers
        :param num_layers: number of hidden layers
        :param learning_rate: global learning rate of AdamW optimizer
        :param num_estimators: number of estimators to use when the balanced bagging classifier is enabled
        :param test_size: if not None, set aside a portion of the data for testing vals:[0-1]
        :param enable_bag: enable balanced bagging classifier, rather than balancing data
        :param enable_tb: enable tensorboard logging
        :param enable_es: enable early stopping callback when training ceases to improve for 20 iterations
        :param verbose: verbosity of logging
        """

        self.output_dir = output_dir
        self.complete_labeled_file = complete_labelled_file
        self.hq_cluster_file = hq_cluster_file
        self.seed = seed
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_estimators = num_estimators
        self.test_size = test_size
        self.enable_bag = enable_bag
        self.enable_tb = enable_tb
        self.enable_es = enable_es
        self.verbose = verbose

        # read data for training
        self.df_complete = pd.read_csv(complete_labelled_file)
        # separate out the complete training set
        self.df_full_training = ContactClassifier._separate_training(self.df_complete)
        self.hq_clusters = set(pd.read_csv(hq_cluster_file)['cluster'].values)

        self.model = None
        self.x_full = None
        self.y_full = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # set a global seed through Keras, since there are
        #   many objects within the package which consume a seed.
        keras.utils.set_random_seed(self.seed)
        # prepare the training and possibly test dataset(s)
        self.prepare_training_data()

    @staticmethod
    def _separate_training(df):
        """
        Simply return the table containing only the data marked for training.
        :param df: a pandas dataframe
        :return: dataframe containing just training data
        """
        return df.query('train==True')

    @staticmethod
    def _extract_x(df):
        """
        Extract only the columns used in modelling contacts.
        :param df:
        :return: numpy array
        """
        return df.loc[:, ContactClassifier._FIT_VARS].values

    @staticmethod
    def _extract_y(df):
        """
        Extract the class variable used in modelling contacts.
        :param df:
        :return: numpy array
        """
        return df.loc[:, ContactClassifier._CLASS_VAR].values

    @staticmethod
    def _make_table(x, y):
        """
        Convenience method for making a dataframe from fit and class variables.
        :param x: fit variables
        :param y: class variable
        :return: dataframe
        """
        df = pd.DataFrame({ContactClassifier._CLASS_VAR: y})
        df[ContactClassifier._FIT_VARS] = x
        return df

    def prepare_training_data(self):
        """
        Prepare the training data for the model.
        This can involve splitting training and test sets, as well
        as applying data augmentation to balance the classes.
        """
        _x = ContactClassifier._extract_x(self.df_full_training)
        _y = ContactClassifier._extract_y(self.df_full_training)
        self.x_full = _x
        self.y_full = _y

        if not self.enable_bag:
            _x, _y = self.balance_data(_x, _y)

        if self.test_size is not None:
            (self.x_train,
             self.x_test,
             self.y_train,
             self.y_test) = train_test_split(_x, _y,
                                             test_size=self.test_size,
                                             random_state=self.seed,
                                             stratify=_y)
            logger.info(f'Split data into training (size: {len(self.x_train):,}) '
                        f'and test (size: {len(self.x_test):,}) sets')
        else:
            self.x_train = _x
            self.y_train = _y

    def balance_data(self, x, y):
        """
        Apply data augmentation to equalise the training classes sizes using
        a random undersampling procedure.
        :param x: fit variables
        :param y: class variable
        :return: balanced fit and class arrays
        """
        self.plot_variable_scatter(x, y, 'raw_training_scatter.pdf')

        logger.info(f'Original set size:  x={x.shape}, y={y.shape}, class sizes: {np.bincount(y)}')

        sampler = RandomUnderSampler(random_state=self.seed)
        logger.info('Applying random under-sampling to balance classes')
        x_aug, y_aug = sampler.fit_resample(x, y)
        logger.info('After application of random under-sampling: '
                    f'x={x_aug.shape}, y={y_aug.shape}, class sizes: {np.bincount(y_aug)}')

        self.plot_variable_scatter(x_aug, y_aug, 'augmented_training_scatter.pdf')

        return x_aug, y_aug

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

    def plot_variable_scatter(self, x, y, base_name, n_points=5000):
        """
        Create scatterplots of the different fit variable combinations and save
        to PDF.
        :param x:
        :param y:
        :param base_name:
        :param n_points:
        :return:
        """
        df = ContactClassifier._make_table(x, y)
        with PdfPages(os.path.join(self.output_dir, base_name)) as pdf:
            if len(df) > n_points:
                df = df.sample(n_points, random_state=self.seed)
            pdf.savefig(sb.jointplot(df, x='similarity', y='freq_z', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='similarity', y='cov_z', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='similarity', y='linkage', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='freq_z', y='cov_z', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='freq_z', y='linkage', hue="intra_z").figure)
            pdf.savefig(sb.jointplot(df, x='cov_z', y='linkage', hue="intra_z").figure)

    def tensorboard_callback(self):
        return keras.callbacks.TensorBoard(log_dir=os.path.join(self.output_dir, 'logs'),
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
                                                restore_best_weights=True,
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
        """
        Train the model on the full dataset.
        Depending on options at instantiation-time, this model is either fit using data-augmentation
        or a balanced bagging classifier. This is necessary as commonly there are many more negative
        class (not an intra-cellular contact) examples and positive (is an intra-cellular contact) class
        examples.

        The model can employ callbacks to record "best model", tensorboard and early-stopping. If
        early-stopping occurs, the best model is automatically reloaded.

        The history of the optimisation process is also saved to file.
        """
        tf.keras.backend.clear_session()

        best_model_file = os.path.join(self.output_dir, f'full_best_{ContactClassifier._METRIC_NAME}.keras')

        callbacks = [self.checkpoint_callback(best_model_file, self.verbose)]
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
                                    hidden_layer_sizes=[self.num_nodes] * self.num_layers,
                                    learning_rate=self.learning_rate)
        if self.enable_bag:
            logging.info('Classifier training will use balanced bagging')
            # wrap the base classifier in a balanced bagging classifier
            estimator = BalancedBaggingClassifier(estimator,
                                                  n_estimators=self.num_estimators,
                                                  replacement=False,
                                                  random_state=self.seed,
                                                  verbose=self.verbose)


        logging.info('Beginning model training')
        model = estimator.fit(self.x_train, self.y_train)
        self.model = model

        if self.enable_bag:
            # plot history of all estimators used in bagging
            df_plots = []
            for n, en in enumerate(model.estimators_, start=1):
                _model = en._final_estimator
                logger.info(f'Full dataset model score: {_model.score(self.x_full, self.y_full)}')
                _df = pd.DataFrame(_model.history_) \
                    .reset_index() \
                    .rename(columns={'index': 'epoch',
                                     f'precision_{n-1}': 'precision',
                                     f'recall_{n-1}': 'recall'})
                _df['estimator'] = n
                df_plots.append(_df)
            df_plots = pd.concat(df_plots)

            p = (ggplot(df_plots.query('epoch>=1').melt(id_vars=['epoch','estimator']))
                 + geom_line(aes(x='epoch', y='value', group='estimator',color='factor(estimator)'))
                 + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6])
                 + scale_color_discrete(name = "Estimator#"))
            p.save(filename=os.path.join(self.output_dir,'full_model.svg'), verbose=False)

        else:
            # plot history of the single estimator
            logger.info(f'Full dataset model score: {model.score(self.x_full, self.y_full)}')

            df_plot = pd.DataFrame(model.history_).reset_index().rename(columns={'index': 'epoch'})
            p = (ggplot(df_plot.query('epoch>=1').melt(id_vars='epoch'))
                 + geom_line(aes(x='epoch', y='value'), color='red')
                 + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6]))
            p.save(filename=os.path.join(self.output_dir,'full_model.svg'), verbose=False)

    def predict_best_threshold(self, df, plot=False):
        """
        Find the highest value for f1-score and associated threshold probability
        :param df: training data
        :return: best f1_score and threshold
        """
        df_test = df.query('cluster_name in @self.hq_clusters')
        precision, recall, thres = precision_recall_curve(df_test['intra_z'], df_test['pr_intracellular'])
        # avoid zeros in the denominator
        denominator = recall+precision
        denominator[denominator == 0] = 0.01
        f1_scores = 2 * recall * precision / denominator
        ix_max = np.argmax(f1_scores)
        logger.info(f'Probability threshold of {thres[ix_max]:.5f} achieves the '
                    f'highest F1-score: {f1_scores[ix_max]:.5f}')

        if plot:
            df_plot = pd.DataFrame({'Precision': precision[1:],
                                    'Recall': recall[1:],
                                    'F1-score': f1_scores[1:],
                                    'Pr_threshold': thres})
            p = (ggplot(df_plot.melt(id_vars='Pr_threshold'), aes(x='Pr_threshold', y='value', color='variable'))
                    + geom_line()
                    + scale_x_continuous(breaks=np.arange(0, 1.01, 0.1))
                    + scale_y_continuous(breaks=np.arange(0, 1.01, 0.1))
                    + theme(figure_size=[10,8]))
            p.save(filename=os.path.join(self.output_dir, 'precision_recall_curve.svg'), verbose=False)

        return f1_scores[ix_max], thres[ix_max]

    def classify(self, df=None):
        """
        Apply the trained model to the data and write the predictions to a file.
        :param df: optional dataframe -- if not supplied, use the complete dataset supplied at instantiation.
        :return: updated dataframe with probabilities column
        """
        assert self.model is not None, 'Model has not been trained.'
        if df is None:
            df = self.df_complete.copy()
        x = ContactClassifier._extract_x(df)
        df['pr_intracellular'] = self.model.predict_proba(x)[:, 1]

        # Using the training data, find the threshold probability returning the highest f1-score.
        f1_best, thres_best = self.predict_best_threshold(df, plot=True)
        # Use this threshold as a decision boundary on whether a contact is intra-cellular.
        df = df.assign(is_intracellular = lambda x: x.pr_intracellular > thres_best)
        # rename the original column to reduce confusion
        df.rename(columns={'intra': 'intracluster'}, inplace=True)
        self.write_table(df, 'predictions', 'final predictions', index=False)
        return df


    # TODO kfold needs better awareness for supporting the new logic to set aside a
    #  test set instantiation time. Currently, this logic is contained with the method
    #  below, and is now redundant. However, there is interference, as we might also have
    #  the case that no test set was set aside. The folds need to be analyzed using balanced
    #  data, which includes test, training and validation data.
    # def train_kfold_model(self, n_folds):
    #
    #     assert self.test_size is not None, 'test_size must be set to use kfold cross-validation'
    #
    #     x_aug, y_aug = self.apply_imbalanced_data_augmentation()
    #
    #     tf.keras.backend.clear_session()
    #
    #     hidden_layer_sizes = [self.num_nodes] * self.num_layers
    #     kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
    #
    #     testing = {'precision': [],
    #                   'recall': [],
    #                   'accuracy': [],
    #                   'loss': [],
    #                   'crossentropy': [],
    #                   'fbeta': [],
    #                   }
    #
    #     histories = []
    #     # TODO this is now redundant of the class can split data
    #     #   so remove when refactoring
    #     # x_train, x_test, y_train, y_test = train_test_split(x_aug, y_aug,
    #     #                                                     test_size=0.2, random_state=self.seed, stratify=y_aug)
    #
    #     logger.info(f'Number samples: train {x_train.shape[0]}, test {x_test.shape[0]}')
    #
    #     for n_fold, (train_index, val_index) in enumerate(kfold.split(x_train, y_train), start=1):
    #
    #         logger.info(f"Computing fold: {n_fold}")
    #
    #         x_fit, y_fit = x_train[train_index], y_train[train_index]
    #         x_val, y_val = x_train[val_index], y_train[val_index]
    #
    #         best_model_file = os.path.join(self.output_dir, f'kfold_bestmodel_{n_fold}.keras')
    #
    #         callbacks = [self.checkpoint_callback(best_model_file, self.verbose)]
    #         if self.enable_tb:
    #             callbacks.append(self.tensorboard_callback())
    #         if self.enable_es:
    #             callbacks.append(self.earlystopping_callback('fbeta', verbose=self.verbose))
    #
    #         estimator = KerasClassifier(model=create_baseline,
    #                                     epochs=self.n_epochs,
    #                                     batch_size=self.batch_size,
    #                                     random_state=self.seed,
    #                                     verbose=self.verbose,
    #                                     callbacks=callbacks,
    #                                     hidden_layer_sizes=hidden_layer_sizes,
    #                                     learning_rate=self.learning_rate)
    #
    #         model = estimator.fit(x_fit, y_fit, validation_data=(x_val, y_val))
    #
    #         histories.append(pd.DataFrame(model.history_))
    #
    #         m = model.model_
    #         m.load_weights(best_model_file)
    #         results = m.evaluate(x_test, y_test, batch_size=250)
    #         results = dict(zip(m.metrics_names, results))
    #         for k, v in results.items():
    #             testing[k].append(v)
    #
    #         tf.keras.backend.clear_session()
    #
    #     for k in testing:
    #         logger.info(f'Test data - {k:>12} mean: {np.mean(testing[k]):8.5f}, sd: {np.std(testing[k]):8.5f}')
    #
    #     for n, df in enumerate(histories, start=1):
    #         df['fold'] = n
    #
    #     df_plot = pd.concat(histories)
    #     _tra = df_plot.loc[:, ~df_plot.columns.str.startswith('val_') | (df_plot.columns == 'fold')].copy()
    #     _tra['set_type' ] = 'training'
    #     _val = df_plot.loc[:, df_plot.columns.str.startswith('val_') | (df_plot.columns == 'fold')].copy()
    #     _val['set_type' ] = 'validation'
    #     _val.columns = _tra.columns
    #     df_plot = pd.concat([_val, _tra]).reset_index()
    #     df_plot.melt(id_vars=['index', 'set_type', 'fold'])
    #
    #     p = (ggplot(df_plot.query('index>=1').melt(id_vars=['index', 'set_type', 'fold']))
    #          + geom_point(aes(x='index', y='value', group='set_type', color='set_type'), size=0.5)
    #          # + geom_line(aes(x='index', y='value', group='set_type', color='set_type'))
    #          + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6], legend_position="right"))
    #     p.save(filename=os.path.join(self.output_dir, 'kfold_model.svg'), verbose=False)

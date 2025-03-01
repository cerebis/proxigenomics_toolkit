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


logger = logging.getLogger(__name__)


L2_KERNEL = 0.067
L2_BIAS = 0.026
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.0001


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

import itertools
import logging
import os
import re
from collections import OrderedDict
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Tuple

import keras
import numpy as np
import numpy.typing as npt
import pandas as pd
import scikeras.wrappers
import seaborn as sb
import tensorflow as tf
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.backends.backend_pdf import PdfPages
from plotnine import (
    aes,
    facet_wrap,
    geom_line,
    ggplot,
    scale_color_discrete,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)
from scikeras.wrappers import KerasClassifier
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Metric, Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L2

logger = logging.getLogger(__name__)

L2_KERNEL = 0.15
L2_BIAS = 0.02
DROPOUT_RATE = 0.3


class DataSet(NamedTuple):
    """
    Represents a dataset with features and labels.

    This class provides a structure to store features `x` and labels `y`
    as numpy arrays, useful for use in machine learning or data analysis tasks.

    :ivar x: Features of the dataset.
    :type x: np.ndarray
    :ivar y: Labels corresponding to the features in the dataset.
    :type y: np.ndarray
    """
    x: np.ndarray
    y: np.ndarray

@tf.keras.utils.register_keras_serializable()
class StatefulBinaryFBeta(Metric):
    """
    Custom metric for fbeta maximisation
    """

    def __init__(self,
                 name: str='fbeta',
                 beta: float=1.0,
                 threshold: float=0.5,
                 epsilon: float=1e-7,
                 dtype: np.dtype=np.float32) -> None:
        # initializing an object of the super class
        super(StatefulBinaryFBeta, self).__init__(name=name, dtype=dtype)

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

    def update_state(self,
                     y_true: npt.ArrayLike,
                     y_pred: npt.ArrayLike,
                     sample_weight: Optional[npt.ArrayLike]=None) -> None:
        # casting y_true and y_pred as float dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # setting values of y_pred greater than the set threshold to 1 while those lesser to 0
        y_pred = tf.cast(tf.greater_equal(y_pred, tf.constant(self.threshold)), tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred)) # updating true positives attribute
        self.predicted_positive.assign_add(tf.reduce_sum(y_pred)) # updating predicted positive attribute
        self.actual_positive.assign_add(tf.reduce_sum(y_true)) # updating actual positive attribute

    def result(self) -> float:
        self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
        self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall
        # calculating fbeta
        self.fb = (1 + self.beta_squared) * self.precision*self.recall / \
                   (self.beta_squared*self.precision + self.recall + self.epsilon)
        return self.fb

    def reset_state(self) -> None:
        self.tp.assign(0) # resets true positives to zero
        self.predicted_positive.assign(0) # resets predicted positives to zero
        self.actual_positive.assign(0) # resets actual positives to zero

    def get_config(self) -> Dict:
        import math
        config = {
            'beta': math.sqrt(self.beta_squared),
            'epsilon': self.epsilon,
            "threshold": self.threshold,
        }
        base_config = super().get_config()
        return {**base_config, **config}


def create_baseline(hidden_layer_sizes: List[int],
                    learning_rate: float | keras.optimizers.schedules.LearningRateSchedule,
                    meta: dict) -> Sequential:
    """
    Creates and returns a baseline neural network model with the specified hidden
    layer sizes, learning rate, and metadata. The returned model is compiled with
    appropriate loss function, optimizer, and metrics.

    The architecture includes fully connected dense layers, batch normalization,
    and ReLU activations. Dropout layers are included between the hidden layers to
    reduce overfitting. The output layer uses a sigmoid activation function to
    predict binary output.

    :param hidden_layer_sizes: List of integers specifying the number of nodes
        in each hidden layer.
    :param learning_rate: The learning rate for the optimizer, which can be
        provided as a float or a Keras learning rate schedule.
    :param meta: Dictionary containing metadata required for model creation, such
        as the number of features (`n_features_in_`) and output classes (`n_outputs_`).
    :return: A compiled Keras Sequential model instance with the specified
        architecture.
    """
    model = Sequential()
    model.add(Input(shape=(meta['n_features_in_'],)))
    for n, n_nodes in enumerate(hidden_layer_sizes, 1):
        model.add(Dense(n_nodes,
                        kernel_initializer='he_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=L2(l2=L2_KERNEL),
                        bias_regularizer=L2(l2=L2_BIAS)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        if n < len(hidden_layer_sizes):
            model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(meta['n_outputs_'], activation='sigmoid'))

    logger.debug("--- Model Architecture Summary ---")
    model.summary(print_fn=logger.debug, show_trainable=True)

    loss_func = BinaryCrossentropy()

    model.compile(loss=loss_func,
                  optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, amsgrad=True),
                  metrics=['accuracy',
                           Precision(),
                           Recall(),
                           StatefulBinaryFBeta(beta=1.0),
                           'binary_crossentropy'])
    return model


class AddInstanceLogCallback(keras.callbacks.Callback):
    """
    A simple callback that adds the identity of the model so as to distinguish
    checkpoints between estimators in a BalanedBaggingClassifier run.

    Note: This callback must come before the checkpoint callback for the information
    to be available.
    """
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]]=None) -> None:
        if logs is None:
            logs = {}
        # Add your custom value to the logs dictionary.
        logs['model_id'] = id(self.model)


def find_root(x_values: npt.ArrayLike,
              y_values: npt.ArrayLike) -> float|None:
    """
    Finds the root of a function defined by empirical x and y data points using cubic spline
    interpolation. The function checks for an exact root or a sign change between consecutive
    points to estimate the root.

    :param x_values: Array-like sequence of x-coordinates for the function.
    :param y_values: Array-like sequence of y-coordinates representing the function's values.
    :return: A float representing the root if found, or None if no root exists in the interval.
    """
    # Cubic spline interpolation of the empirical data
    root_func = CubicSpline(x_values, y_values)
    # Check if the first point is a root.
    f_prev = root_func(x_values[0])
    if f_prev == 0:
        return x_values[0]
    # Iterate through the rest of the points
    for i in range(1, len(x_values)):
        x_prev = x_values[i - 1]
        x_curr = x_values[i]
        f_curr = root_func(x_curr)
        # Case 1: an exact root
        if f_curr == 0:
            return x_curr
        # Case 2: a change in sign
        if f_curr * f_prev < 0:
            return brentq(root_func, x_prev, x_curr)
        f_prev = f_curr
    return None


class ContactClassifier(object):
    """
    A classifier for predicting contacts between contigs.

    This class encapsulates the entire workflow for a contact classification
    problem. It handles data loading, preprocessing, model training, prediction,
    and evaluation. The primary goal is to train a model that can accurately
    predict whether two contigs are in contact based on a set of features.

    The typical workflow involves:
    1. Initializing the classifier with various parameters, including file paths
       for input data and directories for output.
    2. Preparing the training data using the `prepare_training_data` method, which
       involves reading data, splitting it into training and testing sets, and
       extracting features (X) and labels (y).
    3. Training a model using either k-fold cross-validation
       (`train_kfold_model`) or on the full training dataset
       (`train_full_model`).
    4. Classifying new, unseen data using the `classify` method.
    5. Evaluating the performance of the model using various methods like
       `assess_predictions`, `plot_precision_recall_curve`, and
       `compute_decision_boundary`.

    The class makes use of TensorFlow/Keras for building and training the
    neural network models. It also includes functionality for logging,
    early stopping, and model checkpointing to manage the training process
    effectively.
    """
    PAGE_WIDTH_MM = 297
    PAGE_HEIGHT_MM = 210

    _METRIC_NAME = 'fbeta'
    _FIT_VARS: ClassVar[List[str]]= ['similarity', 'freq_z', 'cov_z', 'linkage']
    _CLASS_VAR = 'intra_z'

    OUTPUT_TABLES: ClassVar[Dict[str, str]] = {
        'predictions': 'predictions.csv',
        'faceted_predictions': 'faceted_predictions.csv',
    }

    def __init__(self,
                 output_dir: str,
                 complete_labelled_file: str,
                 spurious_cluster_file: str,
                 intra_cluster_file: str,
                 seed: int,
                 n_epochs: int,
                 batch_size: int,
                 num_nodes: int=24,
                 num_layers: int=5,
                 learning_rate: float=1e-4,
                 num_estimators: int=10,
                 test_size: float=0.2,
                 validation_size: float=0.2,
                 enable_bag: bool=False,
                 enable_tb: bool=False,
                 enable_es: bool=True,
                 patience: int=10,
                 verbose: bool=False) -> None:
        """
        An MLP classifier for Hi-C contacts, where classification decides if an accumulated contact
        between a single sequence as a genome_bin is intra- or inter- cellular.

        :param output_dir: Parent directory to which results are written.
        :param complete_labelled_file: Labeled training data.
        :param spurious_cluster_file: File containing cluster ids accepted for spurious contacts.
        :param intra_cluster_file: File containing cluster ids accepted for intra contacts.
        :param seed: A random seed.
        :param n_epochs: Number of epochs for training.
        :param batch_size: Batch size for training.
        :param num_nodes: Number of nodes in the hidden layers.
        :param num_layers: Number of hidden layers.
        :param learning_rate: Global learning rate of AdamW optimizer.
        :param num_estimators: Number of estimators to use when the balanced bagging classifier is enabled.
        :param test_size: The portion of the data set aside for testing.
        :param validation_size: The portion of the data set aside for validation and when bagging the proportion of the
        sample set aside (eg. bag_size=1-validation_size).
        :param enable_bag: Enable balanced bagging classifier, rather than balancing data.
        :param enable_tb: Enable tensorboard logging.
        :param enable_es: Enable early stopping callback when training ceases to improve for 20 iterations.
        :param patience: Number of epochs to wait for when training ceases to improve.
        :param verbose: Verbosity of logging.
        """

        self.output_dir = output_dir
        self.complete_labeled_file = complete_labelled_file
        self.spurious_cluster_file = spurious_cluster_file
        self.intra_cluster_file = intra_cluster_file
        self.seed = seed
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_estimators = num_estimators
        self.test_size = test_size
        self.validation_size = validation_size
        self.enable_bag = enable_bag
        self.enable_tb = enable_tb
        self.enable_es = enable_es
        self.patience = patience
        self.verbose = verbose

        # Just a convienence property primarily for tensorboard logs.
        # This is intended to be informative rather than unique.
        self.run_name = f'{self.num_layers}x{self.num_nodes}-LR:{self.learning_rate}-BS:{self.batch_size}'
        self.model_filename = "{model_id}-bestmodel.keras"
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")

        # read data for training
        self.df_complete = pd.read_csv(complete_labelled_file)
        # separate out the complete training set
        self.df_full_training = ContactClassifier._separate_training(self.df_complete)
        self.spurious_clusters = set(pd.read_csv(spurious_cluster_file)['cluster'].values)
        self.intra_clusters = set(pd.read_csv(intra_cluster_file)['cluster'].values)

        self.model = None
        # complete input dataset
        self.full = None
        # the balanced dataset using undersampling
        self.balanced = None
        # split datasets
        self.train = None
        self.test = None
        self.val = None

        # set a global seed through Keras, since there are
        #   many objects within the package that consume a seed.
        keras.utils.set_random_seed(self.seed)
        # prepare the training and possibly test dataset(s)
        self.prepare_training_data()

    @staticmethod
    def get_output_path(parent_dir: str, table_name: str) -> str:
        """
        Generates the output file path for a specified table based on its parent directory
        and table name by joining them and aligning them with the corresponding entry in
        the OUTPUT_TABLES dictionary.

        :param parent_dir: Directory path where output files are located
        :param table_name: Name of the table being processed
        :return: Full file path to the output file for the specified table
        :rtype: str
        """
        return os.path.join(parent_dir, str(ContactClassifier.OUTPUT_TABLES[table_name]))

    @staticmethod
    def _separate_training(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the table containing only the data marked for training.
        :param df: A pandas dataframe.
        :return: Dataframe containing just training data.
        """
        return df.query('train==True')

    @staticmethod
    def _extract_x(df: pd.DataFrame) -> np.ndarray:
        """
        Extract only the columns used in modeling contacts.
        :param df: The dataframe containing the data.
        :return: Numpy array.
        """
        return df.loc[:, ContactClassifier._FIT_VARS].values

    @staticmethod
    def _extract_y(df: pd.DataFrame) -> np.ndarray:
        """
        Extract the class variable used in modeling contacts.
        :param df: The dataframe containing the data.
        :return: Numpy array.
        """
        return df.loc[:, ContactClassifier._CLASS_VAR].values

    @staticmethod
    def _make_table(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Convenience method for making a dataframe from fit and class variables.
        :param x: Fit variables.
        :param y: Class variable.
        :return: Dataframe.
        """
        df = pd.DataFrame({ContactClassifier._CLASS_VAR: y})
        df[ContactClassifier._FIT_VARS] = x
        return df


    @staticmethod
    def _split_dataset(full: DataSet,
                       test_size: float,
                       validation_size: Optional[float]=None,
                       seed: Optional[int]=None) -> Tuple[DataSet, DataSet] | Tuple[DataSet, DataSet, DataSet]:
        """
        Splits the given dataset into training, testing, and optionally validation subsets
        based on the specified proportions. The method ensures the proportions are valid
        and that stratification is maintained, providing consistent data distribution
        across the splits.

        :param full: The full feature set and corresponding labels provided as a DataSet object.
        :param test_size: Proportion of the dataset to allocate for testing, as a float between 0 and 1.
        :param validation_size: (Optional) Proportion of the dataset to allocate for validation,
                                as a float between 0 and 1. If not provided, validation is not performed,
                                and the remainder is split between training and testing only.
        :param seed: (Optional) Random seed for reproducibility, used in the splitting process.

        :return: A tuple comprising two or three `DataSet` objects (training and testing,
                 and optionally validation if `validation_size` is provided).
        """
        rs = np.random.RandomState(seed)

        train_size = 1 - test_size
        if validation_size is not None:
            train_size -= validation_size

        # Step 1: Split into training and conjoined temporary set of validation + test.
        x_train, x_temp, y_train, y_temp = train_test_split(full.x, full.y,
                                                            train_size=train_size,
                                                            stratify=full.y,
                                                            random_state=rs)
        if validation_size is None:
            return DataSet(x_train, y_train), DataSet(x_temp, y_temp)

        # Step 2: Split the conjoined temporary set into validation and test.
        # Using supplied proportions, calculate the test_size relative to the temporary set.
        split_ratio = test_size / (validation_size + test_size)

        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,
                                                        test_size=split_ratio,
                                                        stratify=y_temp,
                                                        random_state=rs)

        return DataSet(x_train, y_train), DataSet(x_test, y_test), DataSet(x_val, y_val)

    def _validate_sizes(self) -> None:
        """
        Validates dataset size proportions for training, validation, and testing. Ensures that
        the test and validation sizes are within a reasonable range and leaves a substantial
        portion of the dataset for training. Provides warnings if values fall outside typical
        guidelines.

        :param self: Instance of the class containing validation and test size attributes.
        :return: None
        """
        assert self.validation_size + self.test_size < 1, ('The test and validation sizes cannot exceed 1 and should '
                                                           'be small enough to leave a substantial part of the dataset '
                                                           'for training')

        for _nm, _val in [('test', self.test_size), ('validation', self.validation_size)]:
            assert 0 < _val < 1, f'{_nm} size must be a value between 0 and 1'
            if _val < 0.1:
                logger.warning(f'The specified {_nm} size is '
                               'less than 10% of the total dataset size. '
                               'Consider increasing.')
            elif _val > 0.25:
                logger.warning(f'The specified {_nm} size is '
                               'greater than 25% of the total dataset size. '
                               'Consider reducing.')

        if self.validation_size + self.test_size > 0.5:
            logger.warning('The specified test and validation proportions are '
                           'leaving less than half the data for training.')

    def prepare_training_data(self) -> None:
        """Prepares the data structures for model training and evaluation.

        This method orchestrates the data preparation pipeline. It begins by
        extracting the feature set (X) and target labels (y) from the
        initial dataframe (`self.df_full_training`).

        The process then diverges based on the `self.enable_bag` attribute:
        - If bagging is enabled, the data is split into training and testing
          sets. Data balancing is assumed to be handled by the bagging
          process itself.
        - If bagging is disabled, the method first balances the dataset using
          the `balance_data` method. The balanced data is then split into
          training, testing, and validation sets.

        As a result of this method, the following instance attributes are
        populated:
        - `self.full`: A DataSet object containing all features and labels.
        - `self.train`: A DataSet for training the model.
        - `self.test`: A DataSet for final model evaluation.
        - `self.val`: A DataSet for validation during training (if bagging is
          disabled).
        - `self.balanced`: A balanced DataSet (if bagging is disabled).
        """
        self._validate_sizes()

        self.full = DataSet(ContactClassifier._extract_x(self.df_full_training),
                            ContactClassifier._extract_y(self.df_full_training))

        # make a balanced version of the input data set,
        # we'll use it eventually regardless of approach
        self.balanced = self.balance_data(self.full)

        if self.enable_bag:
            # This method handles balancing itself and does not use a validation set directly,
            # instead it is handled in the sampling process of the underlying estimators.
            self.train, self.test = ContactClassifier._split_dataset(self.full,
                                                                     test_size=self.test_size,
                                                                     seed=self.seed)
        else:
            # Split into training, test, and validation sets (equal sizes)
            self.train, self.test, self.val = ContactClassifier._split_dataset(self.balanced,
                                                                               test_size=self.test_size,
                                                                               validation_size=self.validation_size,
                                                                               seed=self.seed)

    def balance_data(self, dataset: DataSet, rs: Optional[np.random.RandomState]=None) -> DataSet:
        """
        Apply data augmentation to equalize the training classes sizes using
        a random undersampling procedure.
        :param dataset: A dataset to balance
        :param rs: random state, otherwise uses instance seed
        :return: Balanced DataSet
        """
        self.plot_variable_scatter(dataset.x,
                                   dataset.y,
                                   ContactClassifier._FIT_VARS,
                                   'raw_training_scatter.pdf')

        logger.info(f'Original set size:  x={dataset.x.shape}, y={dataset.y.shape}, '
                    f'class sizes: {np.bincount(dataset.y)}')

        sampler = RandomUnderSampler(random_state=rs if rs is not None else self.seed)
        logger.info('Applying random under-sampling to balance classes')
        x_aug, y_aug = sampler.fit_resample(dataset.x, dataset.y)
        logger.info('After application of random under-sampling: '
                    f'x={x_aug.shape}, y={y_aug.shape}, class sizes: {np.bincount(y_aug)}')

        self.plot_variable_scatter(x_aug,
                                   y_aug,
                                   ContactClassifier._FIT_VARS,
                                   'augmented_training_scatter.pdf')

        return DataSet(x_aug, y_aug)

    def write_table(self,
                    df: pd.DataFrame,
                    table_name: str,
                    description: str,
                    index: bool,
                    format_columns: bool=True) -> None:
        """
        Standardised writing of a table to a file.
        :param df: The dataframe to write.
        :param table_name: Name of the table to write, for which the actual file name will be obtained.
        :param description: A description of logging.
        :param index: Whether to include.
        """

        COLUMN_FORMATS = OrderedDict({
            "seq": "{}",
            "cluster": "{:d}",
            "cluster_name": "{}",
            "size_v": "{:d}",
            "contacts": "{:d}",
            "length_u": "{:d}",
            "length_v": "{:d}",
            "cov_u": "{:0f}",
            "cov_v": "{:.1f}",
            "sites_u": "{:d}",
            "sites_v": "{:d}",
            "gc_u": "{:0.3f}",
            "gc_v": "{:0.3f}",
            "uf_u": "{:0.3f}",
            "uf_v": "{:0.3f}",
            "intracluster": "{}",
            "prop_cl": "{:0.5f}",
            "intra_z": "{:d}",
            "similarity": "{:0.5f}",
            "linkage": "{:0.5f}",
            "cov_z": "{:0.5f}",
            "freq_z": "{:0.5f}",
            "train": "{}",
            "intracellular_score": "{:0.5f}",
            "is_intracellular": "{}",
        })

        if format_columns:
            # reorder the columns in the dataframe, dropping those
            # which are not mentioned in the formatting dictionary
            logger.debug("Applying column-specific formatting to report for legibility")
            # dictate the order of columns
            df = pd.DataFrame(df, columns=[_cl for _cl in COLUMN_FORMATS if _cl in df.columns])
            # apply formats
            for _cn, _spec in COLUMN_FORMATS.items():
                try:
                    df[_cn] = df[_cn].apply(lambda x: "" if pd.isna(x) else _spec.format(x))
                except:
                    logger.error(f'Could not format column "{_cn}" using format string "{_spec}"')
                    raise

        file_path = ContactClassifier.get_output_path(self.output_dir, table_name)
        logger.info(f'Writing {description} to {file_path}')
        df.to_csv(file_path, index=index)

    def plot_variable_scatter(self,
                              x: np.ndarray,
                              y: np.ndarray,
                              params: List[str],
                              base_name: str,
                              n_points: int=5000) -> None:
        """
        Create scatter plots of the different fit variable combinations and save
        to PDF.
        :param x:
        :param y:
        :param params:
        :param base_name:
        :param n_points:
        :return:
        """
        df = ContactClassifier._make_table(x, y)
        with PdfPages(os.path.join(self.output_dir, base_name)) as pdf:
            if len(df) > n_points:
                df = df.sample(n_points, random_state=self.seed)
            for _x, _y in itertools.combinations(params, 2):
                fig = sb.jointplot(df, x=_x, y=_y, hue="intra_z").figure
                fig.set_size_inches(ContactClassifier.PAGE_WIDTH_MM / 25.4, ContactClassifier.PAGE_HEIGHT_MM / 25.4)
                pdf.savefig(fig)

    def tensorboard_callback(self) -> tf.keras.callbacks.Callback:
        log_path = os.path.join(self.output_dir, 'tensorboard', self.run_name)
        return keras.callbacks.TensorBoard(log_dir=log_path,
                                           histogram_freq=1,
                                           embeddings_freq=1,
                                           write_graph=True,
                                           write_images=True,
                                           update_freq="epoch")

    def earlystopping_callback(self, metric: str, verbose: bool=False) -> tf.keras.callbacks.Callback:
        return tf.keras.callbacks.EarlyStopping(monitor=metric,
                                                patience=self.patience,
                                                mode='max',
                                                min_delta=1e-4,
                                                restore_best_weights=True,
                                                start_from_epoch=10,
                                                verbose=verbose)

    @staticmethod
    def checkpoint_callback(best_model_file: str, verbose: bool=False) -> tf.keras.callbacks.Callback:
        return tf.keras.callbacks.ModelCheckpoint(best_model_file,
                                                  monitor=ContactClassifier._METRIC_NAME,
                                                  verbose=verbose,
                                                  save_best_only=True,
                                                  mode='max')

    def train_kfold_model(self, n_splits: int = 5) -> List[tf.keras.callbacks.History]:
        """
        Trains and evaluates the model using k-fold cross-validation.

        This method performs stratified k-fold cross-validation on the full training dataset.
        For each fold, it trains a new model and evaluates it on the hold-out validation set.
        The training data within each fold is balanced before training. Callbacks for
        TensorBoard, early stopping, and model checkpointing are supported.

        :param n_splits: The number of folds to use for cross-validation.
        :return: A list of Keras History objects, one for each fold.
        """
        rs = np.random.RandomState(self.seed)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)

        model = None
        histories = []
        for fold_no, (train_index, val_index) in enumerate(skf.split(self.full.x, self.full.y), 1):
            logger.info(f'--- Starting training for fold {fold_no}/{n_splits} ---')

            # balance both training and validation sets
            train = self.balance_data(DataSet(self.full.x[train_index, :], self.full.y[train_index]), rs)
            val = self.balance_data(DataSet(self.full.x[val_index, :], self.full.y[val_index]), rs)

            run_name_fold = os.path.join(self.output_dir, 'tb', f'fold_{fold_no}_{self.run_name}')
            callbacks = [keras.callbacks.TensorBoard(log_dir=run_name_fold,
                                                     histogram_freq=1,
                                                     embeddings_freq=1,
                                                     write_graph=True,
                                                     write_images=True,
                                                     update_freq="epoch")]

            model = KerasClassifier(
                model=create_baseline,
                epochs=self.n_epochs,
                batch_size=self.batch_size,
                random_state=self.seed,
                verbose=self.verbose,
                hidden_layer_sizes=[self.num_nodes] * self.num_layers,
                learning_rate=self.learning_rate,
                callbacks=callbacks,
            )

            # adding ignore of the following erroneous warning about incorrect type to validation_data
            # noinspection PyTypeChecker
            history = model.fit(train.x, train.y, validation_data=val)
            histories.append(history)

            scores = model.model_.evaluate(val.x, val.y, verbose=self.verbose)
            logger.info(f'Score for fold {fold_no}: {model.model_.metrics_names[0]} of {scores[0]}; '
                        f'{model.model_.metrics_names[1]} of {scores[1] * 100}%')

        # The trained model for the last fold is stored in self.model
        self.model = model
        return histories

    @staticmethod
    def column_renamer(cn: str) -> str:
        """
        Simple function for renaming individual columns in history dataframe when trraining using balanced
        bagging. This is largely to consolidate results for precision and recall, which receive
        _[INT] suffixes for different blocks of jobs. THis is intended to be supplied to
        the function `pandas.DataFrame.rename()`

        :param cn: a column name
        :return: modified column name
        """
        if cn == "index":
            return "epoch"
        elif cn.startswith("precision") or cn.startswith("recall"):
            return re.sub("_[0-9]+$", "", cn)
        return cn

    def faceted_kfold_analysis(self, num_folds: int=5) -> None:

        # we'll split the full training set into k folds
        k_splitter = (StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=self.seed)
                      .split(self.full.x, self.full.y))

        df_folds = []
        for fold_n, (train_index, test_index) in enumerate(k_splitter, 1):
            logger.info(f'--- Starting training for fold {fold_n}/{num_folds} ---')

            # use the (n-1) folds as training data.
            self.train = DataSet(self.full.x[train_index, :], self.full.y[train_index])
            # the nth fold will be held out for unbiased classification.
            self.test = DataSet(self.full.x[test_index, :], self.full.y[test_index])

            lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.learning_rate,
                alpha=0.1,
                # decay smoothly until the last expected training step
                decay_steps=len(self.train.y) // self.batch_size * self.n_epochs,
            )

            model = KerasClassifier(
                model=create_baseline,
                epochs=self.n_epochs,
                batch_size=self.batch_size,
                random_state=self.seed,
                verbose=self.verbose,
                callbacks=[self.earlystopping_callback('fbeta', verbose=self.verbose)],
                hidden_layer_sizes=[self.num_nodes] * self.num_layers,
                learning_rate=lr_scheduler,
            )

            per_bag_frac = 1 - self.validation_size / (1 - self.test_size)

            # wrap the base classifier in a balanced bagging classifier
            model = BalancedBaggingClassifier(model,
                                              oob_score=True,
                                              max_samples=per_bag_frac,
                                              n_estimators=self.num_estimators,
                                              replacement=True,
                                              random_state=self.seed,
                                              n_jobs=5,
                                              verbose=self.verbose)

            self.model = model.fit(self.train.x, self.train.y)

            logger.info('Fold {fold_no}: best ensemble model score on an example balanced data set: '
                        f'{model.score(self.balanced.x, self.balanced.y):.4f}')
            model._set_oob_score(self.train.x, self.train.y)
            logger.info(f'Fold {fold_n}: best ensemble model OOB accuracy: {model.oob_score_:.4f}')
            logger.info(f'Fold {fold_n}: best ensemble model OOB f1-score: '
                        f'{f1_score(self.train.y, np.argmax(model.oob_decision_function_, axis=1)):.4f}')

            df_plots = []
            for n, en in enumerate(model.estimators_, start=1):
                # get the contained instance of KerasClassifier
                keras_clzr = en._final_estimator

                logger.info(f'Fold {fold_n}: estimator {n}: best model score: '
                            f'{keras_clzr.score(self.train.x, self.train.y):.4f}')

                _df = pd.DataFrame(keras_clzr.history_) \
                    .reset_index() \
                    .rename(columns=ContactClassifier.column_renamer)
                _df['estimator'] = n
                df_plots.append(_df)

            df_plots = pd.concat(df_plots)

            p = (ggplot(df_plots.query('epoch>=0').melt(id_vars=['epoch', 'estimator']))
                 + geom_line(aes(x='epoch', y='value', group='estimator',color='factor(estimator)'))
                 + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6])
                 + scale_color_discrete(name = "Estimator#"))
            p.save(filename=os.path.join(self.output_dir,f'fold-{fold_n}_full_model.svg'),
                   width = ContactClassifier.PAGE_WIDTH_MM,
                   height = ContactClassifier.PAGE_HEIGHT_MM,
                   units = "mm",
                   verbose=False)

            self.assess_predictions(f"training-{fold_n}", self.train.y, model.predict_proba(self.train.x)[:, 1])
            if self.test is not None:
                self.assess_predictions(f"test-{fold_n}", self.test.y, model.predict_proba(self.test.x)[:, 1])

            df_folds.append(self.classify(0.95,
                                          self.df_full_training.iloc[test_index].copy(),
                                          table_name=None))

        df_folds = pd.concat(df_folds, ignore_index=True)
        self.write_table(df_folds, 'faceted_predictions',
                         "final faceted predictions", index=False)

    def train_full_model(self, n_jobs: int = 1) -> None:
        """
        Train the model on the full dataset.
        Depending on options at instantiation-time, this model is either fit using data-augmentation
        or a balanced bagging classifier. This is necessary as commonly there are many more negative
        class (not an intra-cellular contact) examples and positive (is an intra-cellular contact) class
        examples.

        The model can employ callbacks to record "best model", tensorboard and early-stopping. If
        early-stopping occurs, the best model is automatically reloaded.

        The history of the optimization process is also saved to file.

        :param n_jobs: Number of parallel jobs to run when the model is being trained using
        balanced bagging only.
        """
        if n_jobs > 1 and not self.enable_bag:
            logging.warning('The number of jobs is ignored when bagging is not enabled.')

        tf.keras.backend.clear_session()

        # always add the additional logging callback and checkpointing
        callbacks = []
        if self.enable_tb:
            callbacks.append(self.tensorboard_callback())
        if self.enable_es:
            callbacks.append(self.earlystopping_callback('fbeta', verbose=self.verbose))

        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            alpha=0.1,
            # decay smoothly until the last expected training step
            decay_steps=len(self.train.y) // self.batch_size * self.n_epochs,
        )

        model = KerasClassifier(model=create_baseline,
                                epochs=self.n_epochs,
                                batch_size=self.batch_size,
                                random_state=self.seed,
                                verbose=self.verbose,
                                callbacks=callbacks,
                                hidden_layer_sizes=[self.num_nodes] * self.num_layers,
                                learning_rate=lr_scheduler)

        if self.enable_bag:

            logging.info('Classifier training will use balanced bagging')

            # The fraction of the training set to use in a bag, leaving a subset
            # out in each bag. Out-of-bag (OOB) scoring requires that across all
            # bags, every sample point has been left out at least once. Even with
            # replacement, a size less than 1 is recommended otherwise the classifier
            # will require _many_ 10s of estimators.
            # Failure to do so will result in errors when OOB functions are called.
            # This is adjusted by what has alrady been removed for the test set.
            per_bag_frac = 1 - self.validation_size / (1 - self.test_size)

            # wrap the base classifier in a balanced bagging classifier
            model = BalancedBaggingClassifier(model,
                                              oob_score=True,
                                              max_samples=per_bag_frac,
                                              n_estimators=self.num_estimators,
                                              replacement=True,
                                              random_state=self.seed,
                                              n_jobs=n_jobs,
                                              verbose=self.verbose)

            logging.info("Beginning multi-estimator bagging model training ")

            # Fit using the bagging classifier, which does not support supplying a validation data set.
            self.model = model.fit(self.train.x, self.train.y)

            # Load the best model weights and extract the history for plotting
            df_plots = []
            for n, en in enumerate(model.estimators_, start=1):
                # get the contained instance of KerasClassifier
                keras_clzr = en._final_estimator

                logger.info(f'Estimator {n}: best model score: {keras_clzr.score(self.train.x, self.train.y):.4f}')

                _df = pd.DataFrame(keras_clzr.history_) \
                    .reset_index() \
                    .rename(columns=ContactClassifier.column_renamer)
                _df['estimator'] = n
                df_plots.append(_df)

            logger.info('Best ensemble model score on an example balanced data set: '
                        f'{model.score(self.balanced.x, self.balanced.y):.4f}')
            model._set_oob_score(self.train.x, self.train.y)
            logger.info(f'Best ensemble model OOB accuracy: {model.oob_score_:.4f}')
            logger.info('Best ensemble model OOB f1-score: '
                        f'{f1_score(self.train.y, np.argmax(model.oob_decision_function_, axis=1)):.4f}')

            # combine the results of all the estimators
            df_plots = pd.concat(df_plots)

            p = (ggplot(df_plots.query('epoch>=0').melt(id_vars=['epoch', 'estimator']))
                 + geom_line(aes(x='epoch', y='value', group='estimator',color='factor(estimator)'))
                 + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6])
                 + scale_color_discrete(name = "Estimator#"))
            p.save(filename=os.path.join(self.output_dir,'full_model.svg'),
                   width = ContactClassifier.PAGE_WIDTH_MM,
                   height = ContactClassifier.PAGE_HEIGHT_MM,
                   units = "mm",
                   verbose=False)

        else:
            logging.info("Beginning conventional model training")

            # Just fit using the KerasClassifier instance alone, include validation data.
            # adding ignore of the following erroneous warning about incorrect type to validation_data
            # noinspection PyTypeChecker
            self.model = model.fit(self.train.x, self.train.y,
                                   validation_data=(self.val.x, self.val.y[:, np.newaxis]))

            logger.info(f'Best model score on balanced dataset: {model.score(self.balanced.x, self.balanced.y):.4f}')

            # plot history of the single estimator
            df_plot = ContactClassifier._transform_history(model)
            p = (ggplot(df_plot.query('epoch>=0').melt(id_vars=['epoch', 'data_set']))
                 + geom_line(aes(x='epoch', y='value', color='data_set'))
                 + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6]))
            p.save(filename=os.path.join(self.output_dir,'full_model.svg'),
                   width = ContactClassifier.PAGE_WIDTH_MM,
                   height = ContactClassifier.PAGE_HEIGHT_MM,
                   units = "mm",
                   verbose=False)

        # plot combined F1, P, R curves for training and test data if used.
        pr_train = model.predict_proba(self.train.x)[:, 1]
        self.assess_predictions('training', self.train.y, pr_train)

        if self.test is not None:
            self.assess_predictions('test', self.test.y, model.predict_proba(self.test.x)[:, 1])
        if self.val is not None:
            self.assess_predictions('validation', self.val.y, model.predict_proba(self.val.x)[:, 1])

    @staticmethod
    def _transform_history(model: scikeras.wrappers.KerasClassifier) -> pd.DataFrame:
        """
        Transforms and prepares the model's history data for plotting.

        :param model: The model object with a recorded history.
        :return: A pandas DataFrame containing combined training and validation data with
           set type labels.
        """
        df_plot = pd.DataFrame(model.history_).reset_index().rename(columns={"index": "epoch"})
        # without leading val_, assume calculated from the training set
        _tra = df_plot.loc[:, ~df_plot.columns.str.startswith("val_") | (df_plot.columns == "epoch")].copy()
        _tra["data_set"] = "training"
        # with leading val_, assume calculated from the validation set
        _val = df_plot.loc[:, df_plot.columns.str.startswith("val_") | (df_plot.columns == "epoch")].copy()
        _val["data_set"] = "validation"
        _val.columns = _val.columns.str.replace('val_', '')
        return pd.concat([_val, _tra])

    def plot_precision_recall_curve(self,
                                    file_name: str,
                                    precision: np.ndarray,
                                    recall: np.ndarray,
                                    f1_scores: np.ndarray,
                                    pr_threshold: np.ndarray) -> None:

        df_plot = pd.DataFrame({'Precision': precision,
                                'Recall': recall,
                                'F1-score': f1_scores,
                                'Pr_threshold': pr_threshold})
        p = (ggplot(df_plot.melt(id_vars='Pr_threshold'), aes(x='Pr_threshold', y='value', color='variable'))
             + geom_line()
             + scale_x_continuous(breaks=np.arange(0, 1.01, 0.1))
             + scale_y_continuous(breaks=np.arange(0, 1.01, 0.1))
             + theme(figure_size=[10,8]))

        p.save(filename=os.path.join(self.output_dir, file_name),
               width=ContactClassifier.PAGE_WIDTH_MM,
               height=ContactClassifier.PAGE_HEIGHT_MM,
               units="mm",
               verbose=False)

    @staticmethod
    def compute_f1_curve(y_true: np.ndarray,
                         y_prob: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Computes the F1 score along with precision, recall, and thresholds for a
        given prediction probability array and corresponding true labels.

        Using the precision-recall curve, this method calculates the associated F1
        scores, while avoiding division by zero by adjusting the denominator when
        precision and recall are simultaneously zero.

        :param y_true: Ground truth binary labels as a numpy array (0 or 1).
        :param y_prob: Predicted probabilities from a classifier as a numpy array.
        :return: A tuple containing four numpy arrays:
                 - F1 scores (excluding last element to match thresholds).
                 - Precision values (excluding last element to match thresholds).
                 - Recall values (excluding last element to match thresholds).
                 - Thresholds corresponding to precision-recall values.
        """
        precision, recall, thres = precision_recall_curve(y_true, y_prob)
        # avoid zeros in the denominator
        denominator = recall+precision
        denominator[denominator == 0] = 0.01
        f1_scores = 2 * recall * precision / denominator
        # for simplicity, just drop the last element of F1, P and R so as to match
        # the length of thres.
        return f1_scores[:-1], precision[:-1], recall[:-1], thres

    @staticmethod
    def find_simple_maximum(x: np.ndarray, y: np.ndarray) -> (float, float):
        """
        Find the maximum value of y and the corresponding x value, using simple means without interpolation.
        :param x: Independent variable.
        :param y: Dependent variable.
        :return: "X at maximum y", "y max".
        """
        assert x.ndim == 1 and y.ndim == 1, 'The variables x, and y must be a 1D arrays'
        ix_max = np.argmax(y)
        return x[ix_max], y[ix_max]

    def assess_predictions(self,
                           name: str,
                           y_true: np.ndarray,
                           y_prob: np.ndarray) -> (float, float):
        """
        Compute and report statistics and plot the models predictive performance.

        :param name: Name of the dataset.
        :param y_true: True class variable.
        :param y_prob: Predicted probabilities.
        :return: Best f1_score and threshold.
        """
        f1_scores, precision, recall, thres = ContactClassifier.compute_f1_curve(y_true, y_prob)

        max_thres, max_f1  = ContactClassifier.find_simple_maximum(thres, f1_scores)
        logger.info(f'{name}: probability threshold of {max_thres:.5g} achieves the '
                    f'highest F1-score: {max_f1:.5g}')

        self.plot_precision_recall_curve(
            f'precision_recall_curve_{name}.svg',
            precision, recall, f1_scores, thres)

        return max_f1, max_thres

    # @staticmethod
    def compute_decision_boundary(self,
                                  set_name: str,
                                  precision_threshold: float) ->  float:
        """
        Using predictions and true values, compute the decision boundary (in terms of assigned model probability)
        at which overall dataset precision exceeds the requested threshold.
        :param set_name: specified data set by name [test, validation, training]
        :param precision_threshold: Requested threshold precision.
        :return: Probability boundary to achieve requested precision.
        """
        if set_name == 'test':
            y_true, y_prob = self.test.y, self.model.predict_proba(self.test.x)[:, 1]
        elif set_name == 'validation':
            y_true, y_prob = self.val.y, self.model.predict_proba(self.val.x)[:, 1]
        elif set_name == 'training':
            y_true, y_prob = self.train.y, self.model.predict_proba(self.train.x)[:, 1]
        else:
            raise ValueError(f'Unknown set name: {set_name}')

        f1_scores, precision, recall, thres = ContactClassifier.compute_f1_curve(y_true, y_prob)
        assert precision.max() >= precision_threshold, \
            (f'For the {set_name} set: the maximum value reached for Precision was {precision.max():.5g}, '
             f'which is less than the requested decision boundary threshold: {precision_threshold:.5g}')
        if len(thres) <= 1:
            logger.error(f'For the {set_name} set: predicted class probabilities have a '
                         f'single value: {np.unique(y_prob):.5g}')
            raise ValueError('Single-valued probability array suggests model fitting failure')

        logger.info(f'Maximal values for set \"{set_name}\": '
                     f'(pr,Pre)=({thres[precision.argmax()]:.4g}, {precision.max():.5g}), '
                     f'(pr,Rec)=({thres[recall.argmax()]:.4g}, {recall.max():.5g}), '
                     f'(pr,F1)=({thres[f1_scores.argmax()]:.4g}, {f1_scores.max():.5g})')

        decision_boundary = find_root(thres, precision - precision_threshold)
        assert decision_boundary is not None, 'The specified precision threshold was not reachable'
        logger.info(f'For set \"{set_name}\": the requested precision of {precision_threshold:.4g} '
                    f'is achieved when the probability threshold is {decision_boundary:.4g} ')
        return decision_boundary

    def classify(self, precision_thres: float,
                 df: pd.DataFrame=None,
                 table_name: Optional[str]='predictions') -> pd.DataFrame:
        """
        Apply the trained model to the data and write the predictions to a file.

        :param precision_thres: Estimated precision at which to classify intra-cellular contacts.
        :param df: Optional dataframe -- if not supplied, use the complete dataset supplied at instantiation.
        :param table_name: Optional table name -- if not supplied, use the default table name. If None, do not
        write an output file.
        :return: Updated dataframe with the column of probabilities.
        """
        assert self.model is not None, 'Model has not been trained.'

        # use the instance data if not supplied
        if df is None:
            df = self.df_complete.copy()

        # extract features and predict classes
        x = ContactClassifier._extract_x(df)
        df['intracellular_score'] = self.model.predict_proba(x)[:, 1]

        # Given the user-requested precision threshold, compute the decision boundary
        # in terms of a probability threshold.
        if precision_thres is not None:
            assert self.test_size is not None and self.test_size > 0, \
                'Computing a decision boundary requires a test set was set aside in training'

            boundary = self.compute_decision_boundary('test', precision_thres)

            # Use this threshold as a decision boundary on whether a contact is intracellular.
            df = df.assign(is_intracellular = lambda x: x.intracellular_score > boundary)
            # rename the original column to reduce confusion
            df = df.rename(columns={'intra': 'intracluster'})
            # reorder the table so that all the contacts for a given sequence are
            # adjacent rows, but give precedence to the greatest number of contacts.
            df = df.sort_values(['seq', 'contacts'], ascending=[True, False])
            if table_name is not None:
                self.write_table(df, table_name, 'final predictions', index=False)

        return df

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

import itertools
import logging
import os
import re
from collections import OrderedDict, defaultdict
from typing import Any, ClassVar, Dict, Generator, List, NamedTuple, Optional, Tuple

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
from sklearn.metrics import accuracy_score, fbeta_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Metric, Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L2

from ..io_utils import load_object, read_from_stream, save_object, serialize_simple_object

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

class TVStratifiedKFold(StratifiedKFold):
    """
    A K-fold cross-validator that yields indices for training, validation,
    and testing sets.

    In each split, one fold is used for testing, the next fold for validation,
    and the remaining K-2 folds are used for training.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 3.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the fold shuffling.
    """
    def __init__(self, n_splits: int, random_state: np.random.RandomState|int) -> None:
        """
        Initializes the instance of this class, setting up the number of splits and
        random state required for its internal functioning. Inherits behavior from
        the parent class constructor. Also, initializes internal attributes that
        handle information related to folds.

        :param n_splits: Number of folds to create for the splitting process
        :type n_splits: int
        :param random_state: Controls the randomness of the split
        :type random_state: int or None
        """
        if n_splits < 3:
            raise ValueError("n_splits must be at least 3 for train/validation/test split.")
        super().__init__(n_splits, random_state=random_state, shuffle=True)
        self.fold_indices = None

    def split(self,
              x: npt.ArrayLike,
              y: npt.ArrayLike,
              groups: Optional[object]=None) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Splits data into training, testing, and validation sets for a given number of splits.

        The method ensures that for each split, a portion of the dataset is allocated
        to training, testing, and validation subsets. The process is cyclical, ensuring
        the dataset is evenly distributed across the splits for each subset. If `fold_indices`
        are not precomputed, they are computed only once and reused across iterations.

        :param x: Data to be split.
        :param y: Target labels corresponding to the data.
        :param groups: unused compatibility argument.
        :return: A generator that yields tuples of three numpy arrays:
                 - training indices
                 - testing indices
                 - validation indices
        """
        if self.fold_indices is None:
            self.fold_indices = [yi for xi, yi in super().split(x, y)]
        for i in range(self.n_splits):
            test = self.fold_indices[i]
            j = (i + 1) % self.n_splits
            validate = self.fold_indices[j]
            train = np.hstack([self.fold_indices[k] for k in set(range(self.n_splits)) - {i, j}])
            yield np.sort(train), test, validate


@tf.keras.utils.register_keras_serializable()
class StatefulBinaryFBeta(Metric):
    """Computes the F-beta score for binary classification tasks in a stateful manner.

    This metric calculates the F-beta score, which is the weighted harmonic mean of
    precision and recall. It is a more general version of the Fbeta-score. As a
    stateful metric, it accumulates the counts for true positives, actual positives,
    and predicted positives over multiple batches of data. This allows for the
    correct calculation of the score over a full epoch or dataset during model
    training and evaluation.

    The `beta` parameter determines the weight of recall in the combined score.
    - `beta < 1` lends more weight to precision.
    - `beta > 1` favors recall.
    - `beta = 1` corresponds to the traditional Fbeta-score, where precision and
      recall are equally weighted.

    Usage:
    ```python
    model = tf.keras.Model(...)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[StatefulBinaryFBeta(beta=2.0, name='f2_score')]
    )
    ```

    Args:
        name (str): The name of the metric instance. Defaults to 'fbeta'.
        beta (float): The beta parameter that determines the weighting between
            precision and recall. Defaults to 1.0.
        threshold (float): The classification threshold to apply to the predicted
            probabilities. Values at or above this threshold are considered
            positive predictions. Defaults to 0.5.
        epsilon (float): A small constant added to denominators to avoid
            division by zero. Defaults to 1e-7.
        dtype (np.dtype): The data type for the metric's state variables.
            Defaults to np.float32.

    Attributes:
        tp (tf.Variable): Stores the cumulative count of true positives.
        actual_positive (tf.Variable): Stores the cumulative count of actual
            positive samples (true positives + false negatives).
        predicted_positive (tf.Variable): Stores the cumulative count of
            predicted positive samples (true positives + false positives).
        beta_squared (float): The squared value of the beta parameter, cached
            for computational efficiency.
    """

    def __init__(self,
                 name: str='fbeta',
                 beta: float=1.0,
                 threshold: float=0.5,
                 epsilon: float=1e-7,
                 dtype: np.dtype=np.float32) -> None:
        """
        Initialize a StatefulBinaryFBeta instance. This class calculates the F-beta score
        considering binary classification tasks. F-beta is a weighted harmonic mean of precision
        and recall, with beta determining the weight of recall in the combined score.

        :param name: The name of the metric. Defaults to "fbeta".
        :type name: str
        :param beta: Weight of recall in the combined score. A beta value of 1.0 weighs precision
            and recall equally, values greater than 1 emphasize recall, while values less than 1
            stress precision. Defaults to 1.0.
        :type beta: float
        :param threshold: Classification probability threshold. Predictions above this value are
            classified as positive. Defaults to 0.5.
        :type threshold: float
        :param epsilon: Small constant added to prevent division by zero or undefined values in
            precision, recall, or F-beta calculations. Defaults to 1e-7.
        :type epsilon: float
        :param dtype: Data type of the state variables, typically a float type. Defaults to np.float32.
        :type dtype: np.dtype
        """
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
        """
        Update the internal state of the metric by accumulating the true positives,
        predicted positives, and actual positives based on the provided true labels,
        predictions, and optional sample weights.

        :param y_true: Array of true labels.
        :type y_true: numpy.typing.ArrayLike
        :param y_pred: Array of predicted values.
        :type y_pred: numpy.typing.ArrayLike
        :param sample_weight: Optional array of weights for scaling the metric computation.
        :type sample_weight: Optional[numpy.typing.ArrayLike]
        :return: None
        """
        # casting y_true and y_pred as float dtype
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # setting values of y_pred greater than the set threshold to 1 while those lesser to 0
        y_pred = tf.cast(tf.greater_equal(y_pred, tf.constant(self.threshold)), tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred)) # updating true positives attribute
        self.predicted_positive.assign_add(tf.reduce_sum(y_pred)) # updating predicted positive attribute
        self.actual_positive.assign_add(tf.reduce_sum(y_true)) # updating actual positive attribute

    def result(self) -> float:
        """
        Calculates the F-beta score based on true positive, predicted positive, and actual positive counts.

        The F-beta score is computed using the formula:
            F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall + epsilon)
        where precision and recall are calculated as:
            precision = true_positive / (predicted_positive + epsilon)
            recall = true_positive / (actual_positive + epsilon)

        :return: The computed F-beta score.
        :rtype: float
        """
        self.precision = self.tp/(self.predicted_positive+self.epsilon) # calculates precision
        self.recall = self.tp/(self.actual_positive+self.epsilon) # calculates recall
        # calculating fbeta
        self.fb = (1 + self.beta_squared) * self.precision*self.recall / \
                   (self.beta_squared*self.precision + self.recall + self.epsilon)
        return self.fb

    def reset_state(self) -> None:
        """
        Resets the internal state of the metrics to their initial values.
        :return: None
        """
        self.tp.assign(0) # resets true positives to zero
        self.predicted_positive.assign(0) # resets predicted positives to zero
        self.actual_positive.assign(0) # resets actual positives to zero

    def get_config(self) -> Dict:
        """
        Serializes the metric's configuration for model saving and loading.

        This method is essential for Keras's serialization functionality. It
        returns a JSON-serializable dictionary of parameters that allows the
        framework to reconstruct the metric object when a model is saved and
        later loaded.

        It extends the base configuration from the parent `Metric` class with
        the specific parameters of this `StatefulBinaryFBeta` instance.

        :return: A dictionary containing the metric's configuration parameters.
        """
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
                    f_beta: float,
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
    :param f_beta: Weight of recall in the combined score.
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
                           StatefulBinaryFBeta(beta=f_beta),
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

    The class makes use of TensorFlow/Keras for building and training the
    neural network models. It also includes functionality for logging,
    early stopping, and model checkpointing to manage the training process
    effectively.
    """
    _PAGE_WIDTH_MM = 297
    _PAGE_HEIGHT_MM = 210

    _METRIC_NAME = 'fbeta'
    _FIT_VARS: ClassVar[List[str]]= ['similarity', 'freq_z', 'cov_z', 'linkage']
    _PREDICT_DTYPE = np.dtype([('intracellular_score', 'f8'),
                               ('is_intracellular', bool),
                               ('boundary', 'f8')])

    _METRIC_DTYPE = np.dtype([('proba', 'f8'),
                              ('precision', 'f8'),
                              ('recall', 'f8'),
                              ('fbeta', 'f8')])
    _CLASS_VAR = 'intra_z'

    def __init__(self,
                 output_dir: str,
                 complete_labelled_file: str,
                 spurious_cluster_file: str,
                 intra_cluster_file: str,
                 threshold: float,
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
                 enable_replacement: bool=True,
                 enable_oob: bool=True,
                 n_jobs: int=1,
                 patience: int=10,
                 f_beta: float=1.0,
                 verbose: bool=False) -> None:
        """
        An MLP classifier for Hi-C contacts, where classification decides if an accumulated contact
        between a single sequence as a genome_bin is intra- or inter- cellular.

        :param output_dir: Parent directory to which results are written.
        :param complete_labelled_file: Labeled training data.
        :param spurious_cluster_file: File containing cluster ids accepted for spurious contacts.
        :param intra_cluster_file: File containing cluster ids accepted for intra contacts.
        :param threshold: The threshold precision at which to carry out classifications.
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
        :param enable_replacement: Enable replacement of outliers in training data.
        :param enable_oob: Enable out-of-bag (OOB) scoring when enable_bag=True.
        :param n_jobs: Number of parallel jobs to use during training when enable_bag=True.
        :param patience: Number of epochs to wait for when training ceases to improve.
        :param f_beta: The F-beta score (range [0,]) to use for evaluation. Values >1 emphasise Recall, while
        values <1 emphasise Precision.
        :param verbose: Verbosity of logging.
        """

        self.output_dir = output_dir
        self.complete_labeled_file = complete_labelled_file
        self.spurious_cluster_file = spurious_cluster_file
        self.intra_cluster_file = intra_cluster_file
        self.threshold = threshold
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
        self.enable_replacement = enable_replacement
        self.enable_oob = enable_oob
        self.n_jobs = n_jobs
        self.patience = patience
        self.f_beta = f_beta
        self.verbose = verbose

        # Just a convienence property primarily for tensorboard logs.
        # This is intended to be informative rather than unique.
        self.run_name = f'{self.num_layers}x{self.num_nodes}-LR:{self.learning_rate}-BS:{self.batch_size}'
        self.model_filename = "{model_id}-bestmodel.keras"
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")

        # read data for training
        self.df_complete = pd.read_csv(complete_labelled_file)
        # separate out the complete training set
        self.df_labelled, self.df_unlabelled = ContactClassifier._split_training_unlabelled(self.df_complete)
        self.spurious_clusters = set(pd.read_csv(spurious_cluster_file)['cluster'].values)
        self.intra_clusters = set(pd.read_csv(intra_cluster_file)['cluster'].values)

        self.model = None
        self.history = None
        # complete input dataset
        self.labelled = None
        # complete unlabelled dataset
        self.unlabelled = None
        # split datasets
        self.train = None
        self.test = None
        self.val = None

        # set a global seed through Keras, since there are
        #   many objects within the package that consume a seed.
        keras.utils.set_random_seed(self.seed)
        # prepare the training and possibly test dataset(s)
        self._prepare_primary_data()
        # initialise the dictionary of runtime metadata
        self._init_metadata()

    def _init_metadata(self) -> None:
        """
        Resets the metadata of the object to default values for this instance.
        :return: metadata dict containing initial runtime values.
        """
        self.metadata = {'bagging': self.enable_bag,
                    'f_beta': self.f_beta,
                    'n_jobs': self.n_jobs,
                    'enable_es': self.enable_es,
                    'patience': self.patience,
                    'threshold': self.threshold,
                    'test_size': self.test_size,
                    'validation_size': self.validation_size,
                    'num_layers': self.num_layers,
                    'num_nodes': self.num_nodes,
                    'batch_size': self.batch_size,
                    'seed': self.seed,
                    'n_epochs': self.n_epochs,
                    'learning_rate': self.learning_rate,
                    'enable_replacement': self.enable_replacement,
                    'enable_oob': self.enable_oob}

    def _write_metrics(self, run_name: str, tables: List[np.ndarray]) -> None:
        """
        Writes decision metric tables to a compressed file based on the run type.

        This method validates the number of tables provided based on the type of run
        ('crossvalidated' or 'unlabelled') and saves these tables in a compressed
        format at a specified output directory.

        :param run_name: The name of the run, which determines validation requirements.
            Valid options include 'crossvalidated' and 'unlabelled'.
        :param tables: A list of decision metric tables to be written, which are
            validated based on the type of run.
        :return: None
        """
        if run_name == 'crossvalidated':
            assert len(tables) == self.metadata['k_folds'], \
                'The number of decision metric tables does not match the number of folds.'
        if run_name == 'unlabelled':
            assert len(tables) == 1, \
                'There should be only one decision metric tables for unlabelled data.'
        save_object(os.path.join(self.output_dir, f'{run_name}_metrics.p.gz'), tables)

    def _write_metadata(self, run_name: str) -> None:
        """
        Writes metadata to a JSON file in the specified output directory.

        This method takes the metadata stored in the ``self.metadata``
        attribute and writes it to a JSON file named "metadata.json"
        in the directory specified by the ``self.output_dir`` attribute.

        :param run_name: The name of the run for which the metadata is being written.
        :raises FileNotFoundError: If the specified output directory does not exist.
        :raises IOError: If there is an issue writing the file.
        :return: None
        """
        output_path = os.path.join(self.output_dir, f"{run_name}_metadata.json")
        logger.info(f"Writing metadata for {run_name} to {output_path}")
        serialize_simple_object(output_path, self.metadata, fmt='json', float_precision=5)

    @staticmethod
    def _split_training_unlabelled(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return two tables, one containing only the data marked for training and the remainder
        considered as the unlabelled data. This method assumes the input table contains the boolean
        column "train".
        :param df: A pandas dataframe from DataLabeller.
        :return: a tuple of two Dataframes containing either training or unlabelled data.
        """
        return (df.query('train==True').reset_index(drop=True),
                df.query('train==False').reset_index(drop=True))

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
    def _column_renamer(cn: str) -> str:
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

    @staticmethod
    def _split_dataset(dataset: DataSet,
                       test_size: float,
                       validation_size: Optional[float]=None,
                       seed: Optional[int]=None) -> Tuple[DataSet, DataSet] | Tuple[DataSet, DataSet, DataSet]:
        """
        Splits the given dataset into training, testing, and optionally validation subsets
        based on the specified proportions. The method ensures the proportions are valid
        and that stratification is maintained, providing consistent data distribution
        across the splits.

        :param dataset: The dataset to split.
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
        x_train, x_temp, y_train, y_temp = train_test_split(dataset.x, dataset.y,
                                                            train_size=train_size,
                                                            stratify=dataset.y,
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

    def _balance_data(self, dataset: DataSet, rs: Optional[np.random.RandomState]=None) -> DataSet:
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

    def _prepare_primary_data(self) -> None:
        """Prepares the primary data structures for model training and evaluation.

        This method orchestrates the data preparation pipeline. It begins by
        extracting the feature set (X) and target labels (y) from the
        initial dataframe (`self.df_labelled`).

        As a result of this method, the following instance attributes are
        populated:
        - `self.labelled`: A DataSet object containing the confidently labelled observations.
        - `self.unlabelled`: A DataSet of essentially unlabelled observations.
        """
        self._validate_sizes()

        self.labelled = DataSet(ContactClassifier._extract_x(self.df_labelled),
                                ContactClassifier._extract_y(self.df_labelled))

        self.unlabelled = DataSet(ContactClassifier._extract_x(self.df_unlabelled),
                                  ContactClassifier._extract_y(self.df_unlabelled))

    def _get_datasets(self, dataset: DataSet) -> Tuple[DataSet, DataSet, DataSet]:
        """
        Splits the input dataset into training, testing, and validation sets. If `enable_bag` is set to False,
        also balance the training set. only training

        :param dataset: Input dataset to be processed and split.
        :type dataset: DataSet
        :return: A tuple containing train, test, and validation datasets, where val is None
          if enable_bag is True.
        :rtype: Tuple[DataSet, DataSet, DataSet]
        """
        # Split into training, test, and validation sets
        self.train, self.test, self.val = ContactClassifier._split_dataset(dataset,
                                                                           test_size=self.test_size,
                                                                           validation_size=self.validation_size,
                                                                           seed=self.seed)
        if not self.enable_bag:
            # balance the training set for non-bagging classifier
            self.train = self._balance_data(self.train)

        return self.train, self.test, self.val

    def _tensorboard_callback(self) -> tf.keras.callbacks.Callback:
        log_path = os.path.join(self.output_dir, 'tensorboard', self.run_name)
        return keras.callbacks.TensorBoard(log_dir=log_path,
                                           histogram_freq=1,
                                           embeddings_freq=1,
                                           write_graph=True,
                                           write_images=True,
                                           update_freq="epoch")

    def _earlystopping_callback(self, metric: str, verbose: bool=False) -> tf.keras.callbacks.Callback:
        return tf.keras.callbacks.EarlyStopping(monitor=metric,
                                                patience=self.patience,
                                                mode='max',
                                                min_delta=1e-4,
                                                restore_best_weights=True,
                                                start_from_epoch=10,
                                                verbose=verbose)

    @staticmethod
    def _checkpoint_callback(best_model_file: str, verbose: bool=False) -> tf.keras.callbacks.Callback:
        return tf.keras.callbacks.ModelCheckpoint(best_model_file,
                                                  monitor=ContactClassifier._METRIC_NAME,
                                                  verbose=verbose,
                                                  save_best_only=True,
                                                  mode='max')

    def _get_callbacks(self) -> List:
        """
        A private helper to gather all enabled Keras callbacks.
        This cleans up the training logic and centralizes callback configuration.
        """
        callbacks = []
        if self.enable_tb:
            callbacks.append(self._tensorboard_callback())
        if self.enable_es:
            callbacks.append(self._earlystopping_callback('fbeta', verbose=self.verbose))
        return callbacks

    def _build_model(self) -> KerasClassifier | BalancedBaggingClassifier:
        """
        Constructs the classification model based on the instance's configuration.

        This factory method builds a `KerasClassifier` using the `create_baseline`
        function and configures it with the instance's parameters (e.g., learning rate,
        layers, epochs).

        If `self.enable_bag` is True, this base classifier is wrapped in a
        `BalancedBaggingClassifier` to create an ensemble model for improved
        performance on imbalanced datasets.

        Returns:
            KerasClassifier | BalancedBaggingClassifier: An unfitted scikit-learn
            compatible classifier instance ready for training.
        """

        if self.n_jobs > 1 and not self.enable_bag:
            logging.warning("The number of jobs is ignored when bagging is not enabled.")

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
                                callbacks=self._get_callbacks(),
                                hidden_layer_sizes=[self.num_nodes] * self.num_layers,
                                learning_rate=lr_scheduler,
                                f_beta=self.f_beta)

        if self.enable_bag:
            logging.info('Classifier model will use balanced bagging')
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
                                              oob_score=self.enable_oob,
                                              max_samples=per_bag_frac,
                                              n_estimators=self.num_estimators,
                                              replacement=self.enable_replacement,
                                              random_state=self.seed,
                                              n_jobs=self.n_jobs,
                                              verbose=self.verbose)

        return model

    @staticmethod
    def write_table(df: pd.DataFrame,
                    output_path: str,
                    reorder: bool=True,
                    format_columns: bool=True) -> None:
        """
        Standardised writing of a table to a file.
        :param df: The dataframe to write.
        :param output_path: The path and filename to write the table.
        :param reorder: Whether to reorder the table rows.
        :param format_columns: Whether to format the column values.
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
            "boundary": "{:0.5f}",
        })

        # first, if "intra" exists as a column, rename it something
        # a little more explicit for clarity.
        df = df.rename(columns={'intra': 'intracluster'})

        if reorder:
            # do sorting before formatting, as formatting converts
            # numerica data to strings, resulting in unexpected order.
            logger.debug('Reordering table')
            df = df.sort_values(['seq','contacts'], ascending=[True, False])

        if format_columns:
            # reorder the columns in the dataframe, dropping those
            # which are not mentioned in the formatting dictionary
            logger.debug('Applying column-specific formatting to table')
            # dictate the order of columns
            df = pd.DataFrame(df, columns=[_cl for _cl in COLUMN_FORMATS if _cl in df.columns])
            # apply formats
            for _cn, _spec in COLUMN_FORMATS.items():
                try:
                    df[_cn] = df[_cn].apply(lambda x: "" if pd.isna(x) else _spec.format(x))
                except:
                    logger.error(f'Could not format column "{_cn}" using format string "{_spec}"')
                    raise

        logger.info(f'Writing table to {output_path}')
        df.to_csv(output_path, index=False)

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
                fig.set_size_inches(ContactClassifier._PAGE_WIDTH_MM / 25.4, ContactClassifier._PAGE_HEIGHT_MM / 25.4)
                pdf.savefig(fig)

    def report_and_plot(self,
                        tag: str,
                        train: DataSet,
                        test: DataSet,
                        val: Optional[DataSet]=None) -> None:
        """
        Generates a performance report and training history plots for the model.

        This method performs an analysis of model performance during and after training.
        It handles both ensemble models with bagging enabled and single classifiers.
        Detailed metrics such as model scores for training, validation, and testing are
        computed and logged. For ensemble models, metrics of individual base models
        are also captured. The method creates comprehensive plots representing the
        model's training history and saves them into the specified output directory.

        :param tag: A string representing the tag used to identify the report and plot.
        :param train: The DataSet object containing the training data used in model fitting.
        :param test: The DataSet object containing test data.
        :param val: Optional DataSet object containing validation data.
        :return: None
        """

        # use a capitalized tag for leading
        cap_tag = tag.capitalize()
        # otherwise, consistent lower case
        tag = tag.lower()

        if self.enable_bag:
            df_plots = []
            estim_md = defaultdict(list)
            for n, en in enumerate(self.model.estimators_, start=1):
                # get the contained instance of KerasClassifier
                keras_clzr = en._final_estimator
                best_score = keras_clzr.score(train.x, train.y)
                estim_md[f'{tag}_estimator_score'].append(
                    best_score)
                estim_md[f'{tag}_estimator_precision'].append(
                    precision_score(train.y, keras_clzr.predict(train.x)))
                estim_md[f'{tag}_estimator_recall'].append(
                    recall_score(train.y, keras_clzr.predict(train.x)))
                estim_md[f'{tag}_estimator_fbeta'].append(
                    fbeta_score(train.y, keras_clzr.predict(train.x), beta=self.f_beta))

                logger.info(f'{cap_tag} - Estimator {n}: best model score: {best_score:.4f}')
                _df = pd.DataFrame(keras_clzr.history_) \
                    .reset_index() \
                    .rename(columns=ContactClassifier._column_renamer)
                _df['estimator'] = n
                df_plots.append(_df)

            self.metadata.update(estim_md)

            self.model._set_oob_score(train.x, train.y)
            self.metadata[f'{tag}_oob_score'] = self.model.oob_score_
            logger.info(f'{cap_tag} - Best ensemble model OOB accuracy: {self.model.oob_score_:.4f}')
            oob_fbeta = fbeta_score(train.y, np.argmax(self.model.oob_decision_function_, axis=1), beta=self.f_beta)
            self.metadata[f'{tag}_oob_fbeta'] = oob_fbeta
            logger.info(f'{cap_tag} - Best ensemble model OOB Fbeta score: {oob_fbeta:.4f}')

            # combine the results of all the estimators
            df_plots = pd.concat(df_plots)
            plt = (ggplot(df_plots.query('epoch>=0').melt(id_vars=['epoch', 'estimator']))
                   + geom_line(aes(x='epoch', y='value', group='estimator',color='factor(estimator)'))
                   + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6])
                   + scale_color_discrete(name = "Estimator#"))
        else:
            # plot history of the single estimator
            df_plot = ContactClassifier._transform_history(self.model)
            plt = (ggplot(df_plot.query('epoch>=0').melt(id_vars=['epoch', 'data_set']))
                   + geom_line(aes(x='epoch', y='value', color='data_set'))
                   + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6]))

        plt.save(filename=os.path.join(self.output_dir, f'model_training_history_{tag}.svg'),
                 width = ContactClassifier._PAGE_WIDTH_MM,
                 height = ContactClassifier._PAGE_HEIGHT_MM,
                 units = "mm",
                 verbose=False)

        pr_train = self.model.predict_proba(train.x)[:, 1]
        self.assess_predictions(f'{tag}', 'training', train.y, pr_train)
        if test is not None:
            self.assess_predictions(f'{tag}', 'test', test.y, self.model.predict_proba(test.x)[:, 1])
        if self.val is not None:
            self.assess_predictions(f'{tag}', 'validation', val.y, self.model.predict_proba(val.x)[:, 1])

    def fit(self, train: DataSet, validation: Optional[DataSet]=None) -> None:
        """
        Trains the configured model on the prepared dataset.

        This is the primary training method. It orchestrates data preparation,
        model building, and training, handling both the standard and bagging cases.
        The final trained model is stored in `self.model`.
        :param train: The DataSet object containing the training data used in model fitting.
        :param validation: Optional DataSet object containing validation data.
        :return: None
        """
        logger.info("Initiating model training process.")

        tf.keras.backend.clear_session()

        model = self._build_model()

        if self.enable_bag:
            logging.info("Beginning multi-estimator bagging model training ")
            self.history = model.fit(train.x, train.y)
            model._set_oob_score(train.x, train.y)
        else:
            logging.info("Beginning conventional model training")
            # validation_data is supplied correctly, disabling warning
            # noinspection PyTypeChecker
            self.history = model.fit(train.x, train.y,
                                     validation_data=(validation.x, validation.y[:, np.newaxis]))

        self.model = model
        logger.info("Model training complete.")

    def predict(self,
                samples: npt.NDArray,
                decision_set: str='validation',
                metrics: Optional[Dict[str, npt.ArrayLike]]=None) -> npt.NDArray:
        """
        Classifies input samples using a pre-trained model and returns the classification results.
        This method computes the probability of each sample belonging to a specific class using the
        trained model. It then applies a decision boundary to classify the samples based on a specified
        threshold precision. The results include both the computed probabilities and the classification labels.

        Assuming there are three sets, the decision boundary should be determined on either "test" or "validation",
        and that set should not be the source of the samples array. i.e. decision=test, predict=validation or
        visa versa.

        :param samples: A 2D array representing the input samples to be classified. Each row corresponds
            to a sample, and columns correspond to feature values required for prediction.
        :param decision_set: The set that will be used to determine the decision boundary used ib classification. This
            should be a set NOT used in training nor used again for prediction.
        :param metrics: Optional classification metrics that have been/will be used for decision making.
        :return: A structured array containing the probability scores and classification labels for the
            input samples. The first field of the array contains the probability scores, and the second field
            contains the classification labels (boolean).
        :raises AssertionError: If the model is not trained prior to calling this function or if the shape
            of the samples array does not match the expected number of features.
        :raises ValueError: If the prediction contains NaN values.
        """
        assert self.model is not None, 'Model has not been trained. Please call .fit() first.'
        assert samples.shape[1] == len(ContactClassifier._FIT_VARS), \
            'The supplied samples array does not contain the correct number of features.'

        logger.info(f"Classifying {len(samples)} samples.")
        result = np.zeros(shape=len(samples),
                          dtype=ContactClassifier._PREDICT_DTYPE)

        scor_col = ContactClassifier._PREDICT_DTYPE.names[0]
        clzz_col = ContactClassifier._PREDICT_DTYPE.names[1]
        bndr_col = ContactClassifier._PREDICT_DTYPE.names[2]

        prob_intra = self.model.predict_proba(samples)[:, 1]
        if np.any(np.isnan(prob_intra)):
            msg = f'There were {np.isnan(prob_intra).sum()} NaNs in the prediction result'
            logger.error(msg)
            raise ValueError(msg)

        result[scor_col] = prob_intra
        # determine the decision boundary for classification.
        boundary = self.compute_decision_boundary(decision_set, self.threshold, metrics)
        logger.info(f"Applying decision boundary at p > {boundary:.4f}")
        # apply it to the newly scored samples, tweak and reorder the table.
        result[clzz_col] = result[scor_col] > boundary
        result[bndr_col] = boundary
        return result

    def get_cross_validated_labelled_predictions(self, k_folds: int=5) -> pd.DataFrame:
        """
        Performs k-fold cross-validation on the complete labelled set (that is used for model
        training) so as to generate unbiased predictions.

        To do so, it splits the labelled set into k non-overlapping sets (k folds). A single
        fold is then held-out while the remainder is used to train the model. The model is then
        used to predict classes on the held-out fold. The process is repeated for each fold.

        The methods supports both the standard KerasClassifier and BalancedBaggingClassifier
        models. Keep in mind that calculation of the BalancedBaggingClassifier is computationally
        much more demanding (but tractable).

        :param k_folds: The number of folds to use for cross-validation.
        :return: A DataFrame containing the original training data with an added
                 'intracellular_score_cv' column.
        """
        logger.info(f'Obtaining {k_folds}-fold cross-validated classification of the labelled data.')
        logger.info(f'The fractional set sizes will be: training={(k_folds-2)//k_folds:.2f}%, '
                    f'validation/test={1//k_folds:.2f}%')

        self.metadata['k_folds'] = k_folds

        # Training, Test, and Validation sets adapted from K-fold.
        k_splitter = (TVStratifiedKFold(n_splits=k_folds, random_state=self.seed)
                      .split(self.labelled.x, self.labelled.y))

        predictions = np.zeros(shape=len(self.labelled.x),
                               dtype=ContactClassifier._PREDICT_DTYPE)

        cv_metrics = []
        fold_info = defaultdict(list)
        for fold_n, (train_index, test_index, val_index) in enumerate(k_splitter, 1):
            logger.info(f'--- Processing Fold {fold_n}/{k_folds} ---')

            fold_info['train_size'].append(len(train_index))
            fold_info['test_size'].append(len(test_index))
            fold_info['val_size'].append(len(val_index))

            # (n-2) folds are used as training data.
            self.train = DataSet(self.labelled.x[train_index, :], self.labelled.y[train_index])
            # 1 fold each for test and validation
            self.test = DataSet(self.labelled.x[test_index, :], self.labelled.y[test_index])
            self.val = DataSet(self.labelled.x[val_index, :], self.labelled.y[val_index])

            fold_info['positive_frac'].append(self.train.y.sum() / len(self.train.y))

            if not self.enable_bag:
                # balance only the training data.
                self.train = self._balance_data(self.train)
                fold_info['balanced_size'].append(len(self.train.x))
                fold_info['balanced_postive_frac'].append(self.train.y.sum() / len(self.train.y))

            self.fit(self.train, validation=self.val)
            self.report_and_plot(f'fold_{fold_n}', self.train, self.test, self.val)

            # compute predictions and keep a record of the decision making results for each fold.
            # this additional data can be used to reclassify the dataset without retraining or
            # recomputing classification.
            fold_metrics = {}
            predictions[test_index] = self.predict(self.test.x, metrics=fold_metrics)
            assert 'validation' in fold_metrics, 'Decision metrics must be derived from validation data'
            cv_metrics.append(fold_metrics['validation'])

        self.metadata.update(fold_info)
        self._write_metrics('crossvalidated', cv_metrics)
        self._write_metadata('crossvalidated')
        # clear result records from metadata.
        self._init_metadata()

        logger.info('Cross-validation complete.')
        df_result = self.df_labelled.join(pd.DataFrame(predictions), validate='1:1')
        ContactClassifier.write_table(df_result,
                                      os.path.join(self.output_dir, 'crossvalidated_predictions.csv'),
                                      reorder=False, format_columns=False)
        return df_result

    def get_unlabelled_predictions(self) -> pd.DataFrame:
        """
        Retrieves predictions for unlabelled data by performing model fitting
        on training data and generating a report and plot for validation.

        :return: A DataFrame containing the unlabelled data joined with
            the predicted values.
        """
        logger.info('Obtaining classification of the unlabelled data.')

        self.metadata['bagging'] = self.enable_bag

        train, test, val = self._get_datasets(self.labelled)
        self.metadata['balanced_size'] = len(self.train.x)
        self.metadata['balanced_postive_frac'] = self.train.y.sum() / len(self.train.y)

        self.fit(train, validation=val)
        self.report_and_plot('unlabelled', train, test, val)

        # compute predictions and keep a record of the decision making results.
        # this additional data can be used to reclassify the dataset without retraining or
        # recomputing classification.
        metrics = {}
        predictions = self.predict(self.unlabelled.x, metrics=metrics)
        assert 'validation' in metrics, 'Decision metrics must be derived from validation data'

        self._write_metrics('unlabelled', [metrics['validation']])
        self._write_metadata('unlabelled')
        # clear result records from metadata.
        self._init_metadata()

        logger.info('Unlabelled classification complete.')
        df_result = self.df_unlabelled.join(pd.DataFrame(predictions), validate='1:1')
        self.write_table(df_result,
                         os.path.join(self.output_dir, 'unlabelled_predictions.csv'),
                         reorder=False, format_columns=False)
        return df_result

    def plot_precision_recall_curve(self,
                                    file_name: str,
                                    precision: np.ndarray,
                                    recall: np.ndarray,
                                    fbeta_scores: np.ndarray,
                                    pr_threshold: np.ndarray) -> None:
        """
        Plots the Precision-Recall curve with Fbeta scores and saves the plot to the specified file. This function
        creates a visualization of Precision, Recall, and Fbeta-Score metrics as they vary with the predicted
        probability. The resulting plot is saved as an SVG file in the defined output directory.

        :param file_name: The name of the output file to save the plot.
        :param precision: An array of precision values.
        :param recall: An array of recall values.
        :param fbeta_scores: An array of Fbeta-score values.
        :param pr_threshold: An array of threshold values for the Precision-Recall curve.
        :return: None
        """

        df_plot = pd.DataFrame({'Precision': precision,
                                'Recall': recall,
                                'Fbeta-score': fbeta_scores,
                                'Pr_threshold': pr_threshold})
        plt = (ggplot(df_plot.melt(id_vars='Pr_threshold'), aes(x='Pr_threshold', y='value', color='variable'))
               + geom_line()
               + scale_x_continuous(breaks=np.arange(0, 1.01, 0.1))
               + scale_y_continuous(breaks=np.arange(0, 1.01, 0.1))
               + theme(figure_size=[10,8]))

        plt.save(filename=os.path.join(self.output_dir, file_name),
                 width=ContactClassifier._PAGE_WIDTH_MM,
                 height=ContactClassifier._PAGE_HEIGHT_MM,
                 units="mm",
                 verbose=False)

    @staticmethod
    def compute_fbeta_curve(y_true: np.ndarray,
                            y_prob: np.ndarray,
                            beta: float=1.0,
                            epsilon: float=1e-7) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Computes the Fbeta score along with precision, recall, and thresholds for a
        given prediction probability array and corresponding true labels.

        Using the precision-recall curve, this method calculates the associated Fbeta
        scores, while avoiding division by zero by adjusting the denominator when
        precision and recall are simultaneously zero.

        :param y_true: Ground truth binary labels as a numpy array (0 or 1).
        :param y_prob: Predicted probabilities from a classifier as a numpy array.
        :param beta: The beta value for the Fbeta score. Defaults to 1, which corresponds to the Fbeta score.
        :param epsilon: A small value to avoid division by zero. Defaults to 1e-7.
        :return: A tuple containing four numpy arrays:
                 - Fbeta scores (excluding last element to match thresholds).
                 - Precision values (excluding last element to match thresholds).
                 - Recall values (excluding last element to match thresholds).
                 - Probabilty corresponding to precision-recall values.
        """
        precision, recall, proba = precision_recall_curve(y_true, y_prob)
        # avoid zeros in the denominator
        # this is obsolete now.
        # denominator = recall+precision
        # denominator[denominator == 0] = 0.01

        fbeta_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + epsilon)

        # for simplicity, just drop the last element of Fbeta, P and R so as
        # to match the length of proba.
        return fbeta_scores[:-1], precision[:-1], recall[:-1], proba

    @staticmethod
    def find_simple_maximum(x: np.ndarray, y: np.ndarray) -> (float, float):
        """
        Find the maximum value of y and the corresponding x value, using simple means
        without interpolation.

        :param x: Independent variable.
        :param y: Dependent variable.
        :return: "X at maximum y", "y max".
        """
        assert x.ndim == 1 and y.ndim == 1, 'The variables x, and y must be a 1D arrays'
        ix_max = np.argmax(y)
        return x[ix_max], y[ix_max]

    def assess_predictions(self,
                           tag: str,
                           set_name: str,
                           y_true: np.ndarray,
                           y_prob: np.ndarray,
                           threshold: float=0.5) -> (float, float):
        """
        Compute and report statistics and plot the models predictive performance.

        :param tag: Name of the dataset.
        :param set_name: Name of the dataset.
        :param y_true: True class variable.
        :param y_prob: Predicted probabilities.
        :param threshold: Threshold for classification.
        :return: Best fbeta_score and threshold.
        """
        y_pred = y_prob > threshold
        self.metadata.setdefault(f'{set_name}_score', []).append(accuracy_score(y_true, y_pred))
        self.metadata.setdefault(f'{set_name}_precision', []).append(precision_score(y_true, y_pred))
        self.metadata.setdefault(f'{set_name}_recall', []).append(recall_score(y_true, y_pred))
        self.metadata.setdefault(f'{set_name}_fbeta', []).append(fbeta_score(y_true, y_pred, beta=self.f_beta))

        fbeta_scores, precision, recall, proba = ContactClassifier.compute_fbeta_curve(y_true, y_prob, beta=self.f_beta)

        max_thres, max_fbeta  = ContactClassifier.find_simple_maximum(proba, fbeta_scores)
        logger.info(f'{tag}: probability threshold of {max_thres:.5g} achieves the '
                    f'highest Fbeta-score: {max_fbeta:.5g}')

        self.plot_precision_recall_curve(
            f'precision_recall_curve_{tag}.svg',
            precision, recall, fbeta_scores, proba)

        return max_fbeta, max_thres

    @staticmethod
    def _find_valid_boundary(precision_threshold: float,
                             proba: np.ndarray,
                             precision: np.ndarray) -> float:
        assert 0 <= precision_threshold <= 1, 'A precision threshold must be between 0 and 1'
        decision_boundary = find_root(proba, precision - precision_threshold)
        assert decision_boundary is not None, 'The specified precision threshold was not reachable'
        return decision_boundary

    @staticmethod
    def reclassify(new_precision: float,
                   classification_dir: str) -> pd.DataFrame:
        """
        Reclassifies the dataset based on a new precision threshold using previously
        calculated precision and probability from validation. Both the crossvalidated
        and unlabelled data are reclassified.

        In applying decision boundaries, the method respects the original k-fold
        crossvalidation of labelled data.

        :param new_precision: New precision threshold for classification adjustment.
        :param classification_dir: Path to the directory containing the necessary input files
            and where the output will be saved.
        :return: A reclassified dataset containing both cv and unlabelled data (not reordered).
        :rtype: pd.DataFrame
        """
        logger.info(f'Reclassifying dataset with new precision threshold: {new_precision:.5g}')

        # Reclassify the CV data.

        # Load the runtime metadata and classification metrics
        with open(os.path.join(classification_dir, 'crossvalidated_metadata.json'), 'rt') as input_h:
            cv_metadata: dict = read_from_stream(input_h, 'json')
        cv_metrics = load_object(os.path.join(classification_dir, 'crossvalidated_metrics.p.gz'))

        # Load the original prediction result for crossvalidated only, dropping the group
        # column which is not present in the unlabelled data.
        df_cv = (pd.read_csv(os.path.join(classification_dir, 'crossvalidated_predictions.csv'))
                 .drop(columns=['group']))

        # Recompute the splits, so we can assign new boundaries to each.
        k_splitter = (TVStratifiedKFold(n_splits=cv_metadata['k_folds'],
                                        random_state=cv_metadata['seed']).split(df_cv, df_cv['intra_z']))
        # Initialise all to False
        df_cv['is_intracellular_new'] = False
        df_cv["boundary_new"] = None
        for fold_n, (train_index, test_index, val_index) in enumerate(k_splitter):
            logger.info(f'--- Processing Fold {fold_n+1}/{cv_metadata["k_folds"]} ---')

            boundary = ContactClassifier._find_valid_boundary(new_precision,
                                                              cv_metrics[fold_n]['proba'],
                                                              cv_metrics[fold_n]['precision'])

            logger.info(f'For set "{fold_n+1}": the new requested precision of {new_precision:.4g} '
                        f'is achieved when the probability threshold is {boundary:.4g}')

            df_cv.loc[test_index, 'is_intracellular'] = (
                    df_cv.loc[test_index, 'intracellular_score'] > boundary)
            df_cv.loc[test_index, 'boundary'] = boundary

        # Reclassify the unlabelled data
        logger.info('--- Processing the unlabelled set ---')
        ul_metrics = load_object(os.path.join(classification_dir, 'unlabelled_metrics.p.gz'))
        df_ul = pd.read_csv(os.path.join(classification_dir, 'unlabelled_predictions.csv'))

        boundary = ContactClassifier._find_valid_boundary(new_precision,
                                                          ul_metrics[0]["proba"],
                                                          ul_metrics[0]["precision"])

        logger.info(f'For the unlabelled set: the new requested precision of {new_precision:.4g} '
                    f'is achieved when the probability threshold is {boundary:.4g}')
        df_ul['is_intracellular'] = df_ul['intracellular_score'] > boundary
        df_ul['boundary'] = boundary

        # return combined but not reordered
        return pd.concat([df_cv, df_ul])

    def compute_decision_boundary(self,
                                  set_name: str,
                                  precision_threshold: float,
                                  metrics: Optional[Dict[str, npt.ArrayLike]]=None) -> float:
        """
        Using predictions and true values, compute the decision boundary (in terms of assigned model probability)
        at which overall dataset precision exceeds the requested threshold.
        :param set_name: specified data set by name [test, validation, training]
        :param precision_threshold: Requested threshold precision.
        :param metrics: Optional parameter for returning resulting metrics for this dataset
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

        fbeta_scores, precision, recall, proba = ContactClassifier.compute_fbeta_curve(y_true, y_prob, beta=self.f_beta)
        if metrics is not None:
            # store classification metrics as a contiguous structured numpy array, which
            # is keyed by the set type used.
            metrics[set_name] = np.fromiter(itertools.zip_longest(proba, precision, recall, fbeta_scores),
                                            dtype=ContactClassifier._METRIC_DTYPE)

        assert precision.max() >= precision_threshold, \
            (f'For the {set_name} set: the maximum value reached for Precision was {precision.max():.5g}, '
             f'which is less than the requested decision boundary threshold: {precision_threshold:.5g}')
        if len(proba) <= 1:
            logger.error(f'For the {set_name} set: predicted class probabilities have a '
                         f'single value: {np.unique(y_prob):.5g}')
            raise ValueError('Single-valued probability array suggests model fitting failure')

        logger.info(f'Maximal values for set \"{set_name}\": '
                     f'(pr,Pre)=({proba[precision.argmax()]:.4g}, {precision.max():.5g}), '
                     f'(pr,Rec)=({proba[recall.argmax()]:.4g}, {recall.max():.5g}), '
                     f'(pr,Fbeta)=({proba[fbeta_scores.argmax()]:.4g}, {fbeta_scores.max():.5g})')

        # add various scores to metadata run log
        for nm, arr in [('precision', precision), ('recall', recall), ('fbeta_score', fbeta_scores)]:
            self.metadata[f"{set_name}_{nm}_argmax"] = proba[arr.argmax()]
            self.metadata[f'{set_name}_{nm}_max'] = arr.max()

        decision_boundary = ContactClassifier._find_valid_boundary(precision_threshold, proba, precision)
        logger.info(f'For set \"{set_name}\": the requested precision of {precision_threshold:.4g} '
                    f'is achieved when the probability threshold is {decision_boundary:.4g} ')
        self.metadata['precision_threshold'] = precision_threshold
        self.metadata[f'{set_name}_decision_boundary'] = decision_boundary

        return decision_boundary

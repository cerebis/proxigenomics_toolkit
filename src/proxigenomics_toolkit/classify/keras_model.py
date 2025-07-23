import logging
import os
import platform
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
from sklearn.metrics import precision_recall_curve
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


class StatefulBinaryFBeta(Metric):
    """
    Custom metric for fbeta maximisation
    """

    def __init__(self,
                 name: str='fbeta',
                 beta: float=1.0,
                 threshold: float=0.5,
                 epsilon: float=1e-7,
                 **kwargs: Dict[str, Any]) -> None:
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


def get_adam_impl(ignore_platform: bool) -> type[keras.optimizers.AdamW | keras.optimizers.legacy.Adam]:
    """
    Determines and returns the appropriate Keras optimizer class based on the system's
    platform and processor type. This function ensures compatibility with MacOS systems
    using Apple Silicon (ARM architecture). It selects the legacy Adam optimizer for
    such systems and returns the AdamW optimizer for other cases.

    :param ignore_platform: If True, then the platform is ignored and the current implementation of
    AdamW is returned regardless. Otherwise, the platform is considered, potentially returning a
    legacy implementation of Adam.
    :return: Keras optimizer class suitable for the current platform.
    :rtype: type[Adam | AdamW].
    """
    mac_silicon = platform.system() == "Darwin" and platform.processor() == "arm"
    if ignore_platform or not mac_silicon:
        return keras.optimizers.AdamW
    else:
        logger.warning('Using legacy Adam optimizer for MacOS ARM architecture (override with "ignore-platform").')
        return keras.optimizers.legacy.Adam


def create_baseline(hidden_layer_sizes: List[int],
                    learning_rate: float | keras.optimizers.schedules.LearningRateSchedule,
                    meta: dict,
                    ignore_platform: bool=False) -> Sequential:
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
    :param ignore_platform: If True, then ignore computational platform when choosing optimizer.
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

    loss_func = BinaryCrossentropy()

    adam = get_adam_impl(ignore_platform)

    model.compile(loss=loss_func,
                  optimizer=adam(learning_rate=learning_rate, amsgrad=True),
                  metrics=['accuracy',
                           Precision(),
                           Recall(),
                           StatefulBinaryFBeta(beta=1.0),
                           'crossentropy'])
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

    _METRIC_NAME = 'fbeta'
    _FIT_VARS: ClassVar[List[str]]= ['similarity', 'freq_z', 'cov_z', 'linkage']
    _CLASS_VAR = 'intra_z'

    OUTPUT_TABLES: ClassVar[Dict[str, str]] = {
        'predictions': 'predictions.csv',
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
                 test_size: float=0.15,
                 enable_bag: bool=False,
                 enable_tb: bool=False,
                 enable_es: bool=True,
                 patience: int=10,
                 ignore_platform: bool=False,
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
        :param test_size: The portion of the data set aside for testing and possibly again for validation. It is
        important to note that when not bagging, 2 x test_size will be sacrificed for test + validation.
        :param enable_bag: Enable balanced bagging classifier, rather than balancing data.
        :param enable_tb: Enable tensorboard logging.
        :param enable_es: Enable early stopping callback when training ceases to improve for 20 iterations.
        :param patience: Number of epochs to wait for when training ceases to improve.
        :param ignore_platform: If True, then ignore platform when choosing optimizer.
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
        self.enable_bag = enable_bag
        self.enable_tb = enable_tb
        self.enable_es = enable_es
        self.patience = patience
        self.ignore_platform = ignore_platform
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
        self.test = None
        self.train = None
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
                       train_size: float,
                       test_size: float,
                       validation_size: Optional[float]=None,
                       seed: Optional[int]=None) -> Tuple[DataSet, DataSet] | Tuple[DataSet, DataSet, DataSet]:
        """
        Splits the given dataset into training, testing, and optionally validation subsets
        based on the specified proportions. The method ensures the proportions are valid
        and that stratification is maintained, providing consistent data distribution
        across the splits.

        :param full: The full feature set and corresponding labels provided as a DataSet object.
        :param train_size: Proportion of the dataset to allocate for training, as a float between 0 and 1.
        :param test_size: Proportion of the dataset to allocate for testing, as a float between 0 and 1.
        :param validation_size: (Optional) Proportion of the dataset to allocate for validation,
                                as a float between 0 and 1. If not provided, validation is not performed,
                                and the remainder is split between training and testing only.
        :param seed: (Optional) Random seed for reproducibility, used in the splitting process.

        :return: A tuple comprising two or three `DataSet` objects (training and testing,
                 and optionally validation if `validation_size` is provided).
        """
        assert 0 < train_size < 1, 'Train size must be a value between 0 and 1'
        assert 0 < test_size < 1, 'Test size must be a value between 0 and 1'
        if validation_size is not None:
            assert 0 < validation_size < 1, 'Validation size must be a value between 0 and 1'
            assert train_size + test_size + validation_size == 1.0, \
                'Training, test, and validaiton proportions must sum to 1'
        else:
            assert train_size + test_size == 1.0, 'Training and test proportions must sum to 1'

        rs = np.random.RandomState(seed)

        # Step 1: Split into training and conjoined temporary set of validation + test.
        x_train, x_temp, y_train, y_temp = train_test_split(full.x, full.y,
                                                            train_size=train_size,
                                                            stratify=full.y,
                                                            random_state=rs)
        if validation_size is None:
            return DataSet(x_train, y_train), DataSet(x_temp, y_temp)

        # Step 2: Split the conjoined temporary set into validation and test.
        # Using supplied proportions, calculate the test_size relative to the temporary set.
        val_test_split_ratio = test_size / (test_size + validation_size)

        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,
                                                        test_size=val_test_split_ratio,
                                                        stratify=y_temp,
                                                        random_state=rs)

        return DataSet(x_train, y_train), DataSet(x_test, y_test), DataSet(x_val, y_val)

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
        assert self.test_size is not None and 0 < self.test_size < 1, 'Test size must be a value between 0 and 1'

        self.full = DataSet(ContactClassifier._extract_x(self.df_full_training),
                            ContactClassifier._extract_y(self.df_full_training))

        if self.enable_bag:
            # This method handles balancing itself and does not use a validation set.
            self.train, self.test = ContactClassifier._split_dataset(self.full,
                                                                     train_size=1 - self.test_size,
                                                                     test_size=self.test_size,
                                                                     seed=self.seed)

        else:
            assert self.test_size < 0.5, 'Test sizes larger than 0.5 will leave no data for training'
            # balance the input data set
            self.balanced = self.balance_data(self.full)

            # Split into training, test, and validation sets (equal sizes)
            self.train, self.test, self.val = ContactClassifier._split_dataset(self.balanced,
                                                                               train_size=1 - 2*self.test_size,
                                                                               test_size=self.test_size,
                                                                               validation_size=self.test_size,
                                                                               seed=self.seed)

    def balance_data(self, dataset: DataSet, rs: Optional[np.random.RandomState]=None) -> DataSet:
        """
        Apply data augmentation to equalize the training classes sizes using
        a random undersampling procedure.
        :param dataset: A dataset to balance
        :param rs: random state, otherwise uses instance seed
        :return: Balanced DataSet
        """
        self.plot_variable_scatter(dataset.x, dataset.y, 'raw_training_scatter.pdf')

        logger.info(f'Original set size:  x={dataset.x.shape}, y={dataset.y.shape}, '
                    f'class sizes: {np.bincount(dataset.y)}')

        sampler = RandomUnderSampler(random_state=rs if rs is not None else self.seed)
        logger.info('Applying random under-sampling to balance classes')
        x_aug, y_aug = sampler.fit_resample(dataset.x, dataset.y)
        logger.info('After application of random under-sampling: '
                    f'x={x_aug.shape}, y={y_aug.shape}, class sizes: {np.bincount(y_aug)}')

        self.plot_variable_scatter(x_aug, y_aug, 'augmented_training_scatter.pdf')

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

    def plot_variable_scatter(self, x: np.ndarray, y: np.ndarray, base_name: str, n_points: int=5000) -> None:
        """
        Create scatter plots of the different fit variable combinations and save
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
                ignore_platform=self.ignore_platform
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

    def train_full_model(self) -> None:
        """
        Train the model on the full dataset.
        Depending on options at instantiation-time, this model is either fit using data-augmentation
        or a balanced bagging classifier. This is necessary as commonly there are many more negative
        class (not an intra-cellular contact) examples and positive (is an intra-cellular contact) class
        examples.

        The model can employ callbacks to record "best model", tensorboard and early-stopping. If
        early-stopping occurs, the best model is automatically reloaded.

        The history of the optimization process is also saved to file.
        """
        tf.keras.backend.clear_session()

        best_model_file = os.path.join(self.checkpoint_dir, self.model_filename)

        # always add the additional logging callback and checkpointing
        callbacks = [AddInstanceLogCallback(),
                     self.checkpoint_callback(best_model_file, self.verbose)]

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
                                learning_rate=lr_scheduler,
                                ignore_platform=self.ignore_platform)

        if self.enable_bag:

            logging.info('Classifier training will use balanced bagging')

            # wrap the base classifier in a balanced bagging classifier
            model = BalancedBaggingClassifier(model,
                                              oob_score=True,
                                              n_estimators=self.num_estimators,
                                              replacement=True,
                                              random_state=self.seed,
                                              verbose=self.verbose)

            logging.info("Beginning multi-estimator bagging model training ")

            # Fit using the bagging classifier, which does not support supplying a validation data set.
            self.model = model.fit(self.train.x, self.train.y)

            logger.info(f'Final model score on full dataset: {model.score(self.full.x, self.full.y)}')

            # Load the best model weights and extract the history for plotting
            df_plots = []
            for n, en in enumerate(model.estimators_, start=1):
                # get the contained instance of KerasClassifier
                keras_clzr = en._final_estimator

                logger.debug(f'Estimator {n} (id:{id(keras_clzr.model_)}): Final model score on full dataset: '
                             f'{keras_clzr.score(self.full.x, self.full.y)}')

                # the model instance id is derived from the underlying Keras object
                instance_best = best_model_file.format(model_id = id(keras_clzr.model_))
                logger.debug(f"Loading best model from: {instance_best}")
                keras_clzr.model_.load_weights(instance_best)

                logger.info(f'Estimator {n} (id:{id(keras_clzr.model_)}): Best model score on full dataset: '
                            f'{keras_clzr.score(self.full.x, self.full.y)}')

                _df = pd.DataFrame(keras_clzr.history_) \
                    .reset_index() \
                    .rename(columns={'index': 'epoch',
                                     f'precision_{n-1}': 'precision',
                                     f'recall_{n-1}': 'recall'})
                _df['estimator'] = n
                df_plots.append(_df)

            logger.info(f'Best model score on full dataset: {model.score(self.full.x, self.full.y)}')

            # combine the results of all the estimators and remove the
            #   uninformative model_id
            df_plots = pd.concat(df_plots).drop(columns='model_id')

            p = (ggplot(df_plots.query('epoch>=0').melt(id_vars=['epoch', 'estimator']))
                 + geom_line(aes(x='epoch', y='value', group='estimator',color='factor(estimator)'))
                 + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6])
                 + scale_color_discrete(name = "Estimator#"))
            p.save(filename=os.path.join(self.output_dir,'full_model.svg'), verbose=False)

        else:
            logging.info("Beginning conventional model training")

            # Just fit using the KerasClassifier instance alone, include validation data.
            # adding ignore of the following erroneous warning about incorrect type to validation_data
            # noinspection PyTypeChecker
            self.model = model.fit(self.train.x, self.train.y, validation_data=self.val)

            logger.info(f'Final model score on balanced dataset: {self.model.score(self.balanced.x, self.balanced.y)}')
            logger.info(f'Final model score on full dataset: {model.score(self.full.x, self.full.y)}')

            # load the best model
            best_name = best_model_file.format(model_id = id(self.model.model_))
            logger.debug(f'Loading best model from: {best_name}')
            model.model_.load_weights(best_name)
            logger.info(f'Best model score on balanced dataset: {model.score(self.balanced.x, self.balanced.y)}')
            logger.info(f'Best model score on full dataset: {model.score(self.full.x, self.full.y)}')

            # plot history of the single estimator
            df_plot = ContactClassifier._transform_history(model)
            p = (ggplot(df_plot.query('epoch>=0').melt(id_vars=['epoch', 'data_set']))
                 + geom_line(aes(x='epoch', y='value', color='data_set'))
                 + facet_wrap('~ variable', scales='free') + theme(figure_size=[10,6]))
            p.save(filename=os.path.join(self.output_dir,'full_model.svg'), verbose=False)

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

        df_plot = pd.DataFrame({'Precision': precision[1:],
                                'Recall': recall[1:],
                                'F1-score': f1_scores[1:],
                                'Pr_threshold': pr_threshold})
        p = (ggplot(df_plot.melt(id_vars='Pr_threshold'), aes(x='Pr_threshold', y='value', color='variable'))
             + geom_line()
             + scale_x_continuous(breaks=np.arange(0, 1.01, 0.1))
             + scale_y_continuous(breaks=np.arange(0, 1.01, 0.1))
             + theme(figure_size=[10,8]))
        p.save(filename=os.path.join(self.output_dir, file_name), verbose=False)

    @staticmethod
    def compute_f1_curve(y_true: np.ndarray,
                         y_prob: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

        precision, recall, thres = precision_recall_curve(y_true, y_prob)
        # avoid zeros in the denominator
        denominator = recall+precision
        denominator[denominator == 0] = 0.01
        f1_scores = 2 * recall * precision / denominator
        return f1_scores, precision, recall, thres

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
        logger.info(f'{name}: probability threshold of {max_thres:.5f} achieves the '
                    f'highest F1-score: {max_f1:.5f}')

        self.plot_precision_recall_curve(
            f'precision_recall_curve_{name}.svg',
            precision, recall, f1_scores, thres)

        return max_f1, max_thres

    @staticmethod
    def compute_decision_boundary(name: str,
                                  y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  precision_threshold: float) ->  float:
        """
        Using predictions and true values, compute the decision boundary (in terms of assigned model probability)
        at which overall dataset precision exceeds the requested threshold.
        :param name: Data set name.
        :param y_true: True values.
        :param y_prob: Model probabilities for the same dataset.
        :param precision_threshold: Requested threshold precision.
        :return: Probability boundary to achieve requested precision.
        """
        f1_scores, precision, recall, thres = ContactClassifier.compute_f1_curve(y_true, y_prob)
        logger.debug(f'Maximum values obtained P: {precision.max()}, R:{recall.max()}, F1:{f1_scores.max()}')
        assert precision.max() >= precision_threshold, \
            (f'The maximum precision score {precision.max()} is less than the requested '
             f'decision boundary threshold {precision_threshold}')
        if len(thres) <= 1:
            logger.error(f'Predicted class probabilities have a single value: {np.unique(y_prob)}')
            raise ValueError('Single-valued probability array suggests model fitting failure')

        # wrapping CubicSpline in a lambda to overcome type warning
        # when supplying the instance to brentq.
        decision_boundary = brentq(CubicSpline(thres, precision[:-1] - precision_threshold),
                                   thres[0],
                                   thres[-1])
        logger.info(f'{name}: requested precision of {precision_threshold:} '
                    f'achieved for probability threshold of {decision_boundary:.5f} ')
        return decision_boundary

    def classify(self, precision_thres: float, df: pd.DataFrame=None) -> pd.DataFrame:
        """
        Apply the trained model to the data and write the predictions to a file.

        :param precision_thres: Estimated precision at which to classify intra-cellular contacts.
        :param df: Optional dataframe -- if not supplied, use the complete dataset supplied at instantiation.
        :return: Updated dataframe with the column of probabilities.
        """
        assert self.model is not None, 'Model has not been trained.'

        # use the instance data if not supplied
        if df is None:
            df = self.df_complete.copy()

        # extract features and predict classes
        x = ContactClassifier._extract_x(df)
        df['intracellular_score'] = self.model.predict_proba(x)[:, 1]

        # Compute the decision boundary for the requested precision threshold
        if precision_thres is not None:
            assert self.test_size is not None and self.test_size > 0, \
                'Computing a decision boundary requires a test set was set aside in training'

            pr_test = self.model.predict_proba(self.test.x)[:, 1]
            boundary = ContactClassifier.compute_decision_boundary('test',
                                                                   self.test.y,
                                                                   pr_test,
                                                                   precision_thres)

            # Use this threshold as a decision boundary on whether a contact is intra-cellular.
            df = df.assign(is_intracellular = lambda x: x.intracellular_score > boundary)
            # rename the original column to reduce confusion
            df = df.rename(columns={'intra': 'intracluster'})
            # reorder the table so that all the contacts for a given sequence are
            # adjacent rows, but give precedence to the greatest number of contacts.
            df = df.sort_values(['seq', 'contacts'], ascending=[True, False])
            self.write_table(df, 'predictions', 'final predictions', index=False)

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

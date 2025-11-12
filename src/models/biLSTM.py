# -*- coding: utf-8 -*-
"""
Bidirectional LSTM Classifier for Driver Type Identification

This module implements a Bidirectional LSTM (biLSTM) neural network for classifying
driver types based on trajectory data. The model leverages bidirectional processing
to capture both forward and backward temporal dependencies in driving behavior.
"""

from tensorflow import keras
import numpy as np
from numba import cuda
import time

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix


class Classifier_biLSTM:
    """
    Bidirectional LSTM classifier for distinguishing between human and autonomous driving.

    This classifier implements a two-layer bidirectional LSTM architecture with batch
    normalization and dropout regularization. The model processes sequential trajectory
    data to identify driving patterns characteristic of human drivers versus autonomous vehicles.

    Attributes
    ----------
    output_directory : str or Path
        Directory path where trained models and checkpoints will be saved.
    model_name : str
        Filename for saving the trained model (default: 'biLSTM.hdf5').
    METRICS : list
        List of Keras metrics used to evaluate model performance during training.
        Includes binary accuracy, categorical accuracy, and AUC.
    verbose : int
        Verbosity level for training output (0=silent, 1=progress bar, 2=one line per epoch).
    model : keras.Model
        The compiled biLSTM model instance.
    label_encoder : LabelEncoder
        Scikit-learn encoder for converting string labels to integers.
    onehot_encoder : OneHotEncoder
        Scikit-learn encoder for converting integer labels to one-hot vectors.
    callbacks : list
        List of Keras callbacks including learning rate reduction and model checkpointing.
    """

    def __init__(
        self,
        output_directory,
        input_shape,
        nb_classes,
        model_name="biLSTM.hdf5",
        verbose=2,
        build=True,
    ):
        """
        Initialize the Bidirectional LSTM classifier.

        Parameters
        ----------
        output_directory : str or Path
            Directory where model checkpoints and results will be saved.
        input_shape : tuple
            Shape of input sequences (window_size, n_features).
        nb_classes : int
            Number of output classes (typically 2 for binary classification).
        model_name : str, optional
            Filename for the saved model (default: 'biLSTM.hdf5').
        verbose : int, optional
            Verbosity level during training (default: 2).
        build : bool, optional
            Whether to build the model during initialization (default: True).
        """
        self.output_directory = output_directory
        self.model_name = model_name

        # Define evaluation metrics for model training
        self.METRICS = [
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.Accuracy(name="accuracy"),
            keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            keras.metrics.AUC(name="auc"),
        ]

        self.verbose = verbose

        # Build the model architecture if requested
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose == True:
                self.model.summary()

        return

    def build_model(self, input_shape, nb_classes):
        """
        Construct the bidirectional LSTM architecture.

        The architecture consists of:
        - First biLSTM layer (172 units) with batch normalization and dropout
        - Second biLSTM layer (161 units) with batch normalization and dropout
        - Dense output layer with softmax activation for multi-class classification

        Parameters
        ----------
        input_shape : tuple
            Shape of input sequences (window_size, n_features).
        nb_classes : int
            Number of output classes for classification.

        Returns
        -------
        keras.Model
            Compiled bidirectional LSTM model with callbacks configured.
        """
        # Define input layer
        input_layer = keras.layers.Input(input_shape)

        # First bidirectional LSTM layer with 172 units
        # Processes sequences in both forward and backward directions
        biLSTM1 = keras.layers.Bidirectional(
            keras.layers.LSTM(172, return_sequences=True), input_shape=input_shape
        )(input_layer)
        biLSTM1 = keras.layers.BatchNormalization()(biLSTM1)
        biLSTM1 = keras.layers.Dropout(0.4)(biLSTM1)

        # Second bidirectional LSTM layer with 161 units
        biLSTM2 = keras.layers.Bidirectional(
            keras.layers.LSTM(161), input_shape=input_shape
        )(biLSTM1)
        biLSTM2 = keras.layers.BatchNormalization()(biLSTM2)
        biLSTM2 = keras.layers.Dropout(0.4)(biLSTM2)

        # Output layer with softmax activation for probability distribution over classes
        output_layer = keras.layers.Dense(nb_classes, activation="softmax")(biLSTM2)

        # Create the functional model
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        # Compile model with categorical crossentropy loss and Adam optimizer
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            metrics=self.METRICS,
        )

        # Configure learning rate reduction on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=50, min_lr=0.0001
        )

        # Configure model checkpoint to save best model based on validation loss
        file_path = self.output_directory + self.model_name
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, monitor="val_loss", mode="min", save_best_only=True
        )

        # Store callbacks for training
        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit_encoders(self, y):
        """
        Fit label encoders for transforming class labels.

        This method fits both a LabelEncoder (for integer encoding) and a
        OneHotEncoder (for categorical encoding) on the training labels.

        Parameters
        ----------
        y : array-like
            Training labels to fit the encoders on.
        """
        # Fit label encoder to convert string/categorical labels to integers
        label_encoder = LabelEncoder()
        self.label_encoder = label_encoder.fit(y)

        # Transform labels to integers and reshape for one-hot encoding
        y_transformed = label_encoder.transform(y)
        y_transformed = y_transformed.reshape(len(y_transformed), 1)

        # Fit one-hot encoder for categorical representation
        onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder = onehot_encoder.fit(y_transformed)

    def prepare_labels(self, y):
        """
        Transform labels using fitted encoders.

        Applies both label encoding (integer mapping) and one-hot encoding
        to prepare labels for categorical crossentropy loss.

        Parameters
        ----------
        y : array-like
            Labels to transform.

        Returns
        -------
        numpy.ndarray
            One-hot encoded labels ready for training.
        """
        # Convert labels to integers using fitted LabelEncoder
        new_y = self.label_encoder.transform(y)
        new_y = new_y.reshape(len(new_y), 1)

        # Convert integer labels to one-hot vectors
        new_y = self.onehot_encoder.transform(new_y)

        return new_y

    def fit(self, x_train, y_train, x_val, y_val):
        """
        Train the bidirectional LSTM model.

        This method trains the model using the provided training data, with validation
        monitoring and class weighting to handle imbalanced datasets. The training
        process includes automatic learning rate reduction and model checkpointing.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training sequences with shape (n_samples, window_size, n_features).
        y_train : numpy.ndarray
            Training labels.
        x_val : numpy.ndarray
            Validation sequences for monitoring (not used for gradient updates).
        y_val : numpy.ndarray
            Validation labels for monitoring performance.

        Notes
        -----
        - Training uses 120 epochs with batch size determined dynamically
        - Class weights are computed to handle class imbalance
        - Best model is saved based on validation loss
        - Learning rate is reduced when loss plateaus
        """
        # Define training hyperparameters
        batch_size = 100
        nb_epochs = 120

        # Calculate mini-batch size (minimum of 1/10 of data or batch_size)
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        # Record training start time
        start_time = time.time()

        # Calculate class weights to handle imbalanced datasets
        neg, pos = np.bincount(np.reshape(np.array(y_train), np.array(y_train).size))
        total = neg + pos

        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        # Fit label encoders on training data
        self.fit_encoders(y_train)

        # Transform labels to one-hot encoding
        y_train = self.prepare_labels(y_train)
        y_val = self.prepare_labels(y_val)

        # Train the model with callbacks and class weighting
        self.model.fit(
            x_train,
            y_train,
            batch_size=mini_batch_size,
            epochs=nb_epochs,
            class_weight=class_weight,
            use_multiprocessing=True,
            verbose=self.verbose,
            validation_data=(x_val, y_val),
            callbacks=self.callbacks,
        )

        # Calculate and display training duration
        duration = time.time() - start_time
        print(f"The training time was: {duration}")

        # Clear Keras session to free memory
        keras.backend.clear_session()

    def evaluate(self, x_evaluation, y_evaluation):
        """
        Evaluate the trained model on test data.

        Loads the best saved model and generates predictions on the evaluation set,
        returning confusion matrix components for detailed performance analysis.

        Parameters
        ----------
        x_evaluation : numpy.ndarray
            Test sequences with shape (n_samples, window_size, n_features).
        y_evaluation : numpy.ndarray
            True labels for the test set.

        Returns
        -------
        tuple
            Confusion matrix components (true_negatives, false_positives,
            false_negatives, true_positives).
        """
        # Load the best model saved during training
        model_path = self.output_directory + self.model_name
        model = keras.models.load_model(model_path)

        # Generate predictions and convert probabilities to class labels
        y_pred = model.predict(x_evaluation)
        y_pred = np.argmax(y_pred, axis=1)

        # Compute confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_pred, y_evaluation).ravel()

        return tn, fp, fn, tp

    def transfer_learning(self, x_train, y_train, x_val, y_val, epochs: int = 10):
        """
        Fine-tune the pre-trained model on new data.

        This method implements transfer learning by loading a pre-trained model
        and continuing training on a new dataset. Useful for adapting the model
        to new driving scenarios or datasets.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training sequences for fine-tuning.
        y_train : numpy.ndarray
            Training labels for fine-tuning.
        x_val : numpy.ndarray
            Validation sequences for monitoring.
        y_val : numpy.ndarray
            Validation labels for monitoring.
        epochs : int, optional
            Number of epochs for fine-tuning (default: 10).

        Returns
        -------
        keras.Model
            Fine-tuned model ready for evaluation or further training.
        """
        # Load pre-trained model
        model_path = self.output_directory.joinpath(self.model_name)
        model = keras.models.load_model(model_path)

        # Recompile model with fresh optimizer
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            metrics=self.METRICS,
        )

        # Use small batch size for fine-tuning
        mini_batch_size = 10

        # Fit encoders on new training data
        self.fit_encoders(y_train)

        # Transform labels to one-hot encoding
        y_train = self.prepare_labels(y_train)
        y_val = self.prepare_labels(y_val)

        # Fine-tune the model
        model.fit(
            x_train,
            y_train,
            batch_size=mini_batch_size,
            epochs=epochs,
            use_multiprocessing=True,
            verbose=self.verbose,
            validation_data=(x_val, y_val),
        )

        return model

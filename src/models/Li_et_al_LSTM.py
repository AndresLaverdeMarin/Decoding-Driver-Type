# -*- coding: utf-8 -*-
"""
Li et al. LSTM Model Architecture

This module implements the LSTM architecture proposed by Li et al. for driver type identification.
The model uses a three-layer stacked LSTM architecture with dropout regularization.

Reference:
    Li et al. - Driver identification model for intelligent vehicle based on driving behavior
"""

from tensorflow import keras


def model(input_shape, metrics):
    """
    Build and compile the Li et al. LSTM model for driver type classification.

    This architecture consists of three stacked LSTM layers with 500 units each,
    followed by dropout layers for regularization. The model is designed for
    binary classification tasks (human driver vs. autonomous vehicle).

    Parameters
    ----------
    input_shape : tuple
        Shape of the input data with format (window_size, n_features).
        Represents the temporal sequence length and number of features per timestep.
    metrics : list
        List of Keras metrics to evaluate during training, validation, and testing.
        Common metrics include accuracy, precision, recall, and AUC.

    Returns
    -------
    keras.Sequential
        Compiled LSTM model ready for training with binary crossentropy loss
        and Adam optimizer (learning rate: 1e-3).

    Notes
    -----
    - Each LSTM layer has 500 hidden units
    - Dropout rate is set to 0.5 to prevent overfitting
    - The first two LSTM layers return sequences for stacking
    - Final layer uses sigmoid activation for binary classification
    """
    # Initialize sequential model
    model = keras.Sequential()()

    # First LSTM layer with sequence return for stacking
    model.add(keras.layers.LSTM(500, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))

    # Second LSTM layer with sequence return
    model.add(keras.layers.LSTM(500, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))

    # Third LSTM layer (final recurrent layer)
    model.add(keras.layers.LSTM(500))
    model.add(keras.layers.Dropout(0.5))

    # Output layer for binary classification
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Compile model with Adam optimizer and binary crossentropy loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model

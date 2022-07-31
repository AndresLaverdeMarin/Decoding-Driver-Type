# -*- coding: utf-8 -*-
from tensorflow import keras


def model(input_shape, metrics):
    """

    Parameters
    ----------
    input_shape: tuple
        Contains the shape to train the model with the format (length, windows_size, features).
    metrics: list
        List with the metrics to be used during the training, test and validation.

    Returns
    -------
    Sequential model:
        Li et al. LSTM model.
    """
    model = keras.Sequential()()
    model.add(keras.layers.LSTM(500, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(500, return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.LSTM(500))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model

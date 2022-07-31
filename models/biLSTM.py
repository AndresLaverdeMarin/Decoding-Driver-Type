# -*- coding: utf-8 -*-
from tensorflow import keras
import tensorflow as tf


def model(input_shape, metrics, output_bias=None):
    """

    Parameters
    ----------
    input_shape: tuple
        Contains the shape to train the model with the format (length, windows_size, features).
    metrics: list
        List with the metrics to be used during the training, test and validation.
    output_bias: np.ndarray
        Output bias to be used in the output layer.

    Returns
    -------
    Sequential model:
        Bidirectional LTSM model.
    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = keras.Sequential()
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(172, return_sequences=True), input_shape=input_shape))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(161)))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1, activation="sigmoid", bias_initializer=output_bias))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model
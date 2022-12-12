# -*- coding: utf-8 -*-
from tensorflow import keras
import numpy as np
from numba import cuda
import time

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix


class Classifier_biLSTM:
    """Multi Channel Deep Convolutional Neural Network
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
        self.output_directory = output_directory
        self.model_name = model_name
        self.METRICS = [
            # keras.metrics.TruePositives(name="tp"),
            # keras.metrics.FalsePositives(name="fp"),
            # keras.metrics.TrueNegatives(name="tn"),
            # keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.Accuracy(name="accuracy"),
            keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            keras.metrics.AUC(name="auc"),
        ]
        self.verbose = verbose
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose == True:
                self.model.summary()
            # self.model.save_weights(self.output_directory + model_name)
        return
    
    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)
        
        biLSTM1 = keras.layers.Bidirectional(
            keras.layers.LSTM(172, return_sequences=True), input_shape=input_shape
        )(input_layer)
        biLSTM1 = keras.layers.BatchNormalization()(biLSTM1)
        biLSTM1 = keras.layers.Dropout(0.4)(biLSTM1)
        
        biLSTM2 = keras.layers.Bidirectional(
            keras.layers.LSTM(161), input_shape=input_shape
        )(biLSTM1)
        biLSTM2 = keras.layers.BatchNormalization()(biLSTM2)
        biLSTM2 = keras.layers.Dropout(0.4)(biLSTM2)
        
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(biLSTM2)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=1e-3), 
        metrics=self.METRICS)

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
        min_lr=0.0001)

        file_path = self.output_directory + self.model_name

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
        mode='min', save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

    def fit_encoders(self, y):
        label_encoder = LabelEncoder()
        self.label_encoder = label_encoder.fit(y)
        
        y_transformed = label_encoder.transform(y)
        y_transformed = y_transformed.reshape(len(y_transformed), 1)
        
        onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder = onehot_encoder.fit(y_transformed)
        
    def prepare_labels(self, y):   
        ### integer mapping using LabelEncoder
        new_y = self.label_encoder.transform(y)
        new_y = new_y.reshape(len(new_y), 1)
        
        ### One hot encoding
        new_y = self.onehot_encoder.transform(new_y)
    
        return new_y
    
    
    def fit(self, x_train, y_train, x_val, y_val):
        # x_val and y_val are only used to monitor the test loss and NOT for training  
        batch_size = 100
        nb_epochs = 120

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        start_time = time.time()
        
        neg, pos = np.bincount(np.reshape(np.array(y_train), np.array(y_train).size))
        total = neg + pos
        # initial_bias = np.log([pos / neg])
        
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        self.fit_encoders(y_train)
        
        y_train = self.prepare_labels(y_train)
        y_val = self.prepare_labels(y_val)

        self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            class_weight=class_weight,
            use_multiprocessing=True,verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)

        duration = time.time() - start_time
        
        print(f"The training time was: {duration}")

        keras.backend.clear_session()
        
    def evaluate(self, x_evaluation, y_evaluation):
        model_path = self.output_directory + self.model_name
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_evaluation)
        y_pred = np.argmax(y_pred, axis=1)
        tn, fp, fn, tp = confusion_matrix(y_pred, y_evaluation).ravel()
        return tn, fp, fn, tp

    def transfer_learning(self, x_train, y_train, x_val, y_val, epochs: int = 10):
        model_path = self.output_directory.joinpath(self.model_name)
        model = keras.models.load_model(model_path)
        
        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=1e-3), 
        metrics=self.METRICS)
        
        # mini_batch_size = int(len(x_train)/3)
        mini_batch_size = 10
        
        self.fit_encoders(y_train)
        
        y_train = self.prepare_labels(y_train)
        y_val = self.prepare_labels(y_val)
        
        model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=epochs,
            use_multiprocessing=True,verbose=self.verbose, validation_data=(x_val,y_val))

        return model
        
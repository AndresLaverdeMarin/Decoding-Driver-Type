# -*- coding: utf-8 -*-
from pathlib import Path
from os.path import abspath, join, dirname

from tensorflow import keras

from data.Li_et_al_processed_data import read

# Get the project path
prj_dir = Path(abspath(join(dirname(__file__), "..")))

X, y = read(
    prj_dir.joinpath("data").joinpath("AstaZero_data_processed_Li_et_al.csv"),
    "./final_models/scaler_Li_et_al_LSTM_5s.pkl",
    5,
)


model = keras.models.load_model(
    prj_dir.joinpath("src").joinpath("final_models").joinpath("Li_et_al_LSTM_5s.h5")
)

# Evaluation
model.evaluate(X, y, verbose=1)

# Prediction
model.predict(X)

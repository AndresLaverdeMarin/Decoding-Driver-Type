# -*- coding: utf-8 -*-
from pathlib import Path
from os.path import abspath, join, dirname

from tensorflow import keras

from data.processed_data import read

# Get the project path
prj_dir = Path(abspath(join(dirname(__file__), "..")))

X, y = read(
    prj_dir.joinpath("data").joinpath("AstaZero_data_processed.csv"),
    "./final_models/scaler_biLSTM_5s.pkl",
    5,
)


model = keras.models.load_model(prj_dir.joinpath("src").joinpath("final_models").joinpath("biLSTM_5s.h5"))

model.evaluate(X, y, verbose=1)
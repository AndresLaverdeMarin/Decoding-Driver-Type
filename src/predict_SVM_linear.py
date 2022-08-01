# -*- coding: utf-8 -*-
from pathlib import Path
from os.path import abspath, join, dirname

import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.metrics import accuracy_score, confusion_matrix

from data.processed_data import _create_trip_windows

# Get the project path
prj_dir = Path(abspath(join(dirname(__file__), "..")))

# Read csv file
data = pd.read_csv(prj_dir.joinpath("data").joinpath("AstaZero_data_processed.csv"))

# Load scaler
scaler = pkl.load(open(f"final_models/scaler_SVM_linear.pkl", "rb"))

# Load classifier
clf = pkl.load(open(f"final_models/SVM_linear.pkl", "rb"))

# Prediction by window
carID = data.groupby("Driver")

dfs = [carID.get_group(car).groupby("ACC") for car in data["Driver"].unique()]

acc_dfs = []
for df in dfs:
    for acc in data["ACC"].unique():
        try:
            acc_dfs.append(df.get_group(acc))
        except:
            print(f"group {acc} doesn't present")
            continue

# Select features
list_df = [
    df[
        [
            "ACC",
            "Speed",
            "Acceleration",
            "Timegap",
            "StdSpeed",
            "StdAcceleration",
            "StdTimegap",
        ]
    ]
    for df in acc_dfs
]

for window in [5]:
    labels_acc = []
    labels_hv = []
    preditions_acc = []
    preditions_hv = []
    input_trips = _create_trip_windows(windows_size=window * 10, list_dfs=list_df)
    for trip in input_trips:
        X = scaler.transform(
            trip[
                [
                    "Speed",
                    "Acceleration",
                    "Timegap",
                    "StdSpeed",
                    "StdAcceleration",
                    "StdTimegap",
                ]
            ].values
        )
        predition = clf.predict(X)
        counts = np.bincount(predition)
        if trip["ACC"].sum() == window * 10:
            labels_acc.append(1)
            preditions_acc.append(np.argmax(counts))
        if trip["ACC"].sum() == 0:
            labels_hv.append(0)
            preditions_hv.append(np.argmax(counts))

    cm_acc = confusion_matrix(labels_acc, preditions_acc)
    cm_hv = confusion_matrix(labels_hv, preditions_hv)

    TP_ACC = np.diag(cm_acc)
    FP_HV = cm_hv.sum(axis=0) - np.diag(cm_hv)
    FN_HV = cm_hv.sum(axis=1) - np.diag(cm_hv)
    TP_HV = np.diag(cm_hv)
    TN_HV = cm_hv.sum() - (FP_HV + FN_HV + TP_HV)

    acc_acc = accuracy_score(preditions_acc, labels_acc)
    acc_hv = accuracy_score(preditions_hv, labels_hv)

    print(
        f"For window {window}. Correctly classified ACC: {TP_ACC}. Correctly classified HVs: {TN_HV}"
    )
    print(f"For window {window}. ACC accuracy: {acc_acc}. HVs accuracy: {acc_hv}")

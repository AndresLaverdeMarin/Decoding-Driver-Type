# -*- coding: utf-8 -*-
import pickle as pkl
import pandas as pd
import numpy as np


def _read_scaler(path):
    """

    Parameters
    ----------
    path: str
        Scaler path.

    Returns
    -------
    scaler
        Scaler.

    """
    return pkl.load(open(path, "rb"))


def _create_trip_windows(windows_size: int, list_dfs: list) -> list:
    """

    Parameters
    ----------
    windows_size: int
        Window in steps.
    list_dfs: list
        Dataframes to be split in windows.


    Returns
    -------

    """
    final_list = []
    for df in list_dfs:
        size = windows_size + 1
        list_of_dfs = [df.iloc[i : i + size - 1, :] for i in range(0, len(df), size)]
        for trip in list_of_dfs:
            if len(trip) == windows_size:
                final_list.extend([trip])

    return final_list


def read(path, scaler_path, window_size):
    """
    Parameters
    ----------
    path: str
        Path to read the file.
    scaler_path: str
        Scaler path.
    window_size: int
        Windows size in [s].

    Returns
    -------
    X
        Standardized train data.
    y
        Train labels.
    initial_bias

    """
    data = pd.read_csv(path)

    scaler = pkl.load(open(scaler_path, "rb"))

    carID = data.groupby("Driver")

    dfs = [carID.get_group(car).groupby("ACC") for car in data["Driver"].unique()]

    acc_dfs = []
    for df in dfs:
        for acc in data["ACC"].unique():
            try:
                acc_dfs.append(df.get_group(acc))
            except:
                print(f"group {acc} is not present")
                continue

    list_df = [
        df[["ACC", "SpeedF", "SpeedL", "Acceleration", "Spacing"]] for df in acc_dfs
    ]

    labels = []
    X = []
    # 10 Hz data
    input_trips = _create_trip_windows(windows_size=window_size * 10, list_dfs=list_df)
    for trip in input_trips:
        X.append(trip[["SpeedF", "SpeedL", "Acceleration", "Spacing"]].values)
        if trip["ACC"].sum() == window_size * 10:
            labels.append(1)
        if trip["ACC"].sum() == 0:
            labels.append(0)

    X_tot = np.array(X)
    y = np.array(labels).reshape((-1, 1))
    X = np.array([scaler.transform(X_tot[i]) for i in range(0, len(X_tot))])

    return X, y

from operator import index
from src.models.mcdcnn import Classifier_CNN
from src.models.fcn import Classifier_FCN
from src.models.resNet import Classifier_RESNET
from src.models.biLSTM import Classifier_biLSTM
from src.models.Li_et_al_LSTM import Classifier_LSTM
from src.models.dcnn import Classifier_dCNN

from src.data.processed_data import _create_trip_windows

import pandas as pd
import numpy as np
from pathlib import Path
from random import randint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import confusion_matrix

from scipy.signal import hilbert
import emd


COLUMNS = ["Speed", "Acceleration", "Timegap", "StdSpeed", "StdAcceleration", "StdTimegap"]
# COLUMNS = ["SpeedF", "Acceleration", "SpeedL", "Spacing"]


def _read_data(dataset: str) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    dataset : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    base_dir = Path(__file__).parent
    match dataset:
        case "waymo":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("waymo_processed.csv"))
            data["followerId"] = data["followerId"].astype(str)
            data["leadId"] = data["leadId"].astype(str)
            data["seg"] = data["seg"].astype(str)
            data["Driver"] = data["followerId"].str.cat(data["leadId"], sep="-")
            data["Driver"] = data["Driver"].str.cat(data["seg"], sep="-")
        case "astazero":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("astazero_processed.csv"))
            data["Driver"] = data["Driver"].astype(str)
            data["Driver"] = data["File"].str.cat(data["Driver"], sep="-")
        case "zalazone":
            zalazoneD = pd.read_csv(base_dir.joinpath("data").joinpath("ZalaZone_Dyn.csv"), index_col=0)
            zalazoneH = pd.read_csv(base_dir.joinpath("data").joinpath("ZalaZone_Hdl.csv"), index_col=0)
            zalazoneD["Driver"] = zalazoneD["Driver"].astype(str)
            zalazoneH["Driver"] = zalazoneH["Driver"].astype(str)
            zalazoneD["Driver"] = zalazoneD["File"].str.cat(
                zalazoneD["Driver"], sep="-"
            )
            zalazoneH["Driver"] = zalazoneH["File"].str.cat(
                zalazoneH["Driver"], sep="-"
            )

            data = pd.concat([zalazoneD, zalazoneH], ignore_index=True)
        case "bjtu":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("bjtu.csv"))
            data["Driver"] = data["Driver"].astype(str)
            data["Driver"] = data["File"].str.cat(data["Driver"], sep="-")
        case "stern":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("setern2v_processed.csv"))
            data["Driver"] = data["File"].str.cat(data["Driver"], sep="-")
        case "all":
            astazero = pd.read_csv(base_dir.joinpath("data").joinpath("astazero_processed.csv"))
            zalazoneD = pd.read_csv(base_dir.joinpath("data").joinpath("ZalaZone_Dyn.csv"), index_col=0)
            zalazoneH = pd.read_csv(base_dir.joinpath("data").joinpath("ZalaZone_Hdl.csv"), index_col=0)
            stern2v = pd.read_csv(base_dir.joinpath("data").joinpath("setern2v_processed.csv"))
            waymo = pd.read_csv(base_dir.joinpath("data").joinpath("waymo_processed.csv"))
            bjtu = pd.read_csv(base_dir.joinpath("data").joinpath("bjtu.csv"))

            astazero["Driver"] = astazero["Driver"].astype(str)
            bjtu["Driver"] = bjtu["Driver"].astype(str)
            zalazoneD["Driver"] = zalazoneD["Driver"].astype(str)
            zalazoneH["Driver"] = zalazoneH["Driver"].astype(str)

            waymo["followerId"] = waymo["followerId"].astype(str)
            waymo["leadId"] = waymo["leadId"].astype(str)
            waymo["seg"] = waymo["seg"].astype(str)

            astazero["Driver"] = astazero["File"].str.cat(astazero["Driver"], sep="-")
            bjtu["Driver"] = bjtu["File"].str.cat(bjtu["Driver"], sep="-")
            zalazoneD["Driver"] = zalazoneD["File"].str.cat(
                zalazoneD["Driver"], sep="-"
            )
            zalazoneH["Driver"] = zalazoneH["File"].str.cat(
                zalazoneH["Driver"], sep="-"
            )
            stern2v["Driver"] = stern2v["File"].str.cat(stern2v["Driver"], sep="-")
            waymo["Driver"] = waymo["followerId"].str.cat(waymo["leadId"], sep="-")
            waymo["Driver"] = waymo["Driver"].str.cat(waymo["seg"], sep="-")

            data = pd.concat([astazero, zalazoneD, zalazoneH, stern2v, bjtu], ignore_index=True)

    return data


def _read_li_data(dataset: str) -> pd.DataFrame:
    base_dir = Path(__file__).parent
    match dataset:
        case "waymo":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("waymo_processed_Li_et_al.csv"))
            data["followerId"] = data["followerId"].astype(str)
            data["leadId"] = data["leadId"].astype(str)
            data["seg"] = data["seg"].astype(str)
            data["Driver"] = data["followerId"].str.cat(data["leadId"], sep="-")
            data["Driver"] = data["Driver"].str.cat(data["seg"], sep="-")
        case "astazero":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("AstaZero_data_processed_Li_et_al.csv"))
            data["Driver"] = data["Driver"].astype(str)
            data["Driver"] = data["File"].str.cat(data["Driver"], sep="-")
        case "zalazone":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("zalazone_processed_Li_et_al.csv"))
        case "bjtu":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("Bjtu_data_processed_Li_et_al.csv"))
            data["Driver"] = data["Driver"].astype(str)
            data["Driver"] = data["File"].str.cat(data["Driver"], sep="-")
        case "stern":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("setern2v_processed_Li_et_al.csv"))
            data["Driver"] = data["File"].str.cat(data["Driver"], sep="-")
        case "all":
            astazero = pd.read_csv(base_dir.joinpath("data").joinpath("AstaZero_data_processed_Li_et_al.csv"))
            astazero["Driver"] = astazero["Driver"].astype(str)
            astazero["Driver"] = astazero["File"].str.cat(astazero["Driver"], sep="-")
            zalazone = pd.read_csv(base_dir.joinpath("data").joinpath("zalazone_processed_Li_et_al.csv"))
            stern2v = pd.read_csv(base_dir.joinpath("data").joinpath("setern2v_processed_Li_et_al.csv"))
            stern2v["Driver"] = stern2v["File"].str.cat(stern2v["Driver"], sep="-")
            waymo = pd.read_csv(base_dir.joinpath("data").joinpath("waymo_processed_Li_et_al.csv"))
            waymo["followerId"] = waymo["followerId"].astype(str)
            waymo["leadId"] = waymo["leadId"].astype(str)
            waymo["seg"] = waymo["seg"].astype(str)
            waymo["Driver"] = waymo["followerId"].str.cat(waymo["leadId"], sep="-")
            waymo["Driver"] = waymo["Driver"].str.cat(waymo["seg"], sep="-")
            bjtu = pd.read_csv(base_dir.joinpath("data").joinpath("Bjtu_data_processed_Li_et_al.csv"))
            bjtu["Driver"] = bjtu["Driver"].astype(str)
            bjtu["Driver"] = bjtu["File"].str.cat(bjtu["Driver"], sep="-")

            data = pd.concat([astazero, waymo, stern2v, bjtu], ignore_index=True)
            
    return data

def _read_clean_data(dataset: str) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    dataset : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    base_dir = Path(__file__).parent
    match dataset:
        case "waymo":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("waymo_processed_cleaned.csv"))
        case "astazero":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("astazero_processed_cleaned.csv"))
        case "zalazone":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("zalazone_processed_cleaned.csv"))
        case "bjtu":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("bjtu_processed_cleaned.csv"))
        case "stern":
            data = pd.read_csv(base_dir.joinpath("data").joinpath("stern_processed_cleaned.csv"))
        case "all":
            astazero = pd.read_csv(base_dir.joinpath("data").joinpath("astazero_processed_cleaned.csv"))
            zalazone = pd.read_csv(base_dir.joinpath("data").joinpath("zalazone_processed_cleaned.csv"))
            stern2v = pd.read_csv(base_dir.joinpath("data").joinpath("stern_processed_cleaned.csv"))
            waymo = pd.read_csv(base_dir.joinpath("data").joinpath("waymo_processed_cleaned.csv"))
            bjtu = pd.read_csv(base_dir.joinpath("data").joinpath("bjtu_processed_cleaned.csv"))

            data = pd.concat([zalazone, astazero, stern2v, bjtu], ignore_index=True)
            
    return data


def data_scaler(data: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # data_s = pd.DataFrame()

    scaler = StandardScaler()

    data_s = scaler.fit_transform(data[COLUMNS])
    data_s = pd.DataFrame(data_s, columns=COLUMNS)
    data_s["ACC"] = data["ACC"].values

    data_s["Driver"] = data["Driver"].values

    return data_s, scaler
    
    
def evaluation(mode: int, model, scaler, dataset: str, window_size: int):
    """_summary_

    Parameters
    ----------
    mode : int
        _description_
    model : _type_
        _description_
    """
    data_eval = _read_clean_data(dataset)
    data_eval_HV = data_eval[data_eval["ACC"] == mode]
    
    data_eval_HV, scaler = data_scaler(data_eval_HV)
    
    data_s = scaler.fit_transform(data_eval_HV[COLUMNS])
    data_s = pd.DataFrame(data_s, columns=COLUMNS)
    data_s["ACC"] = data_eval_HV["ACC"].values

    data_s["Driver"] = data_eval_HV["Driver"].values
    
    data_s = data_eval_HV
    
    carID = data_s.groupby("Driver")

    dfs = [carID.get_group(car) for car in data_s["Driver"].unique()]
    
    # dfs = instant_fourier(dfs)
    
    labels = []
    X = []
    input_trips = _create_trip_windows(windows_size=window_size, list_dfs=dfs, stride=25)
    for trip in input_trips:
        # X.append(trip[["SpeedF", "SpeedL", "Acceleration", "Spacing"]].values)
        X.append(
            trip[
                COLUMNS
            ].values
        )
        if trip["ACC"].sum() == window_size:
            labels.append(1)
        if trip["ACC"].sum() == 0:
            labels.append(0)

    X = np.array(X)
    labels = list(map(int, labels))
    
    if mode == 1:
        print(f"The scores for AVs in {dataset} dataset:")
        print(np.shape(X))
        tn, fp, fn, tp = model.evaluate(X, labels)
        print(f"Total: {tn + fp + fn + tp}, TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        
    if mode == 0:
        print(f"The scores for HVs in {dataset} dataset:")
        print(np.shape(X))
        tn, fp, fn, tp = model.evaluate(X, labels)
        print(f"Total: {tn + fp + fn + tp}, TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        
def instant_fourier(list_df: list) -> list:
    """_summary_

    Parameters
    ----------
    list_df : list
        _description_

    Returns
    -------
    list
        _description_
    """
    sample_rate = 10
    
    final_data = []
    
    for df in list_df:
        final_df = pd.DataFrame()
        final_df["Driver"] = df["Driver"].values
        final_df["ACC"] = df["ACC"].values
        # final_df["StdSpeed"] = df["StdSpeed"].values
        # final_df["StdAcceleration"] = df["StdAcceleration"].values
        # final_df["StdTimegap"] = df["StdTimegap"].values
        # "StdSpeed", "StdAcceleration", "StdTimegap"
        for c in COLUMNS:
            # Compute frequency statistics
            # IP, IF, IA = emd.spectra.frequency_transform(df[c].values, sample_rate, 'hilbert')
            imf = emd.sift.mask_sift(df[c].values,  verbose='CRITICAL')
            himf = [emd.spectra.frequency_transform(imf[:, n], sample_rate, 'hilbert')[2] for n in range(0, np.shape(imf)[1])]
            himf = np.array(himf)
            thimf = himf.sum(axis=0)
            final_df[c] = thimf
        
        # df = df.iloc[1: , :]
        
        final_data.append(final_df)
    
    return final_data



def fit_classifier(name: str, dataset:str="all"):
    """_summary_

    Parameters
    ----------
    name : str
        _description_
    dataset : str
        _description_
    """
    data = _read_clean_data(dataset)
    # data = _read_li_data(dataset)
    
    # data["Distancegap"] = data["Timegap"] * data["Speed"]

    data, scaler = data_scaler(data)

    carID = data.groupby("Driver")

    dfs = [carID.get_group(car) for car in data["Driver"].unique()]
    
    dfs = instant_fourier(dfs)

    columns_used = COLUMNS.copy()
    
    columns_used.extend(["ACC", "Driver"])

    list_df = [df[columns_used] for df in dfs]
    
    window_size = 50

    labels = []
    X = []
    input_trips = _create_trip_windows(windows_size=window_size, list_dfs=list_df, stride=25)
    for trip in input_trips:
        X.append(
            trip[
                COLUMNS
            ].values
        )
        if trip["ACC"].sum() == window_size:
            labels.append(1)
        if trip["ACC"].sum() == 0:
            labels.append(0)


    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), np.array(labels), test_size=0.4, random_state=42, shuffle=True
    )

    X_validate, X_test, y_validate, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=7, shuffle=True
    )
    
    y_train = list(map(int, y_train))
    y_validate = list(map(int, y_validate))
    
    if name == "mcdcnn":
        model = Classifier_CNN("src/final_models/", (window_size, len(columns_used)-1), 2)
    if name == "fcn":
        model = Classifier_FCN("src/final_models/", (window_size, len(columns_used)-1), 2)
    if name == "resnet":
        model = Classifier_RESNET("src/final_models/", (window_size, len(columns_used)-1), 2)
    if name == "biLSTM":
        model = Classifier_biLSTM("src/final_models/", (window_size, len(columns_used)-2), 2)
    if name == "Li_et_al":
        model = Classifier_LSTM("src/final_models/", (window_size, len(columns_used)-2), 2)
    if name == "dcnn":
        model = Classifier_dCNN("src/final_models/", (window_size, len(columns_used)-1), 2)
    
    model.fit(X_train, y_train, X_validate, y_validate)
    
    # Evaluation in a test partition
    X_test_0 = []
    X_test_1 = []
    y_test_0 = []
    y_test_1 = []

    for x,y in zip(X_test, y_test):
        if y == 0:
            X_test_0.append(x)
            y_test_0.append(y)
        else:
            X_test_1.append(x)
            y_test_1.append(y)
    X_test_0 = np.array(X_test_0)
    X_test_1 = np.array(X_test_1)
    y_test_0 = list(map(int, y_test_0))
    y_test_1 = list(map(int, y_test_1))
    
    print(f"The scores for HVs:")
    print(X_test_0.shape)
    tn, fp, fn, tp = model.evaluate(X_test_0, y_test_0)
    print(f"Total: {tn + fp + fn + tp}, TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    print(f"The scores for AVs:")
    print(X_test_1.shape)
    tn, fp, fn, tp = model.evaluate(X_test_1, y_test_1)
    print(f"Total: {tn + fp + fn + tp}, TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    # evaluation(1, model, scaler, "waymo", window_size)
    # evaluation(0, model, scaler, "waymo", window_size)

    
    
if __name__ == "__main__":
    fit_classifier("biLSTM", dataset="all")


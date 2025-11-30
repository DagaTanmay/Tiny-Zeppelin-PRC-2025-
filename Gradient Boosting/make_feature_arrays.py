import pandas as pd
import numpy as np

def get_x_y_from_df(df, phase, airframe_encoder = None):

    if phase == 'CR':
        column_list = [
            "GTD_total_start",
            "GTD_total_flight",
            "total_time",
            "min_altitude",
            "max_altitude",
            "mean_altitude",
            "mean_groundspeed",
            "mean_roc",
            # "mean_longitude",
            # "mean_latitude",
            # "mean_track",
            # "mean_track_45",
            # "mean_track_90",
            "MZFW"
        ]
    elif phase == 'CL':
        column_list = [
            "GTD_total_start",
            "GTD_total_flight",
            "total_time",
            "min_altitude",
            "max_altitude",
            "mean_altitude",
            "mean_groundspeed",
            "mean_roc",
            "MZFW",
            ]
    elif phase == 'DE':
        column_list = [
            "GTD_total_start",
            "GTD_total_flight",
            "total_time",
            "min_altitude",
            "max_altitude",
            "mean_altitude",
            "mean_groundspeed",
            "mean_roc",
            "MZFW",
        ]
    else:
        column_list = [
            "GTD_total_start",
            "GTD_total_flight",
            "total_time",
            "min_altitude",
            "max_altitude",
            "mean_altitude",
            "mean_groundspeed",
            "mean_roc",
            # "mean_longitude",
            # "mean_latitude",
            # "mean_track",
            # "mean_track_45",
            # "mean_track_90",
        ]

    X = np.array([df[col].values for col in column_list]).transpose()

    if airframe_encoder is not None:
        airframe_enc = airframe_encoder.transform(np.array(df['airframe']).reshape(-1, 1)).toarray()
        X = np.concat([X, airframe_enc], axis=1)

    y = np.array(df["fuel_kg"])
    return X, y
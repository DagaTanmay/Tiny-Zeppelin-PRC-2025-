import numpy as np
from pathlib import Path
import pickle

models_folder = Path(__file__).parent.parent / "gradient_boost"

one_hot_encoders = {
    'light': models_folder / "one_hot_encoder_light.pkl",
    'heavy': models_folder / "one_hot_encoder_heavy.pkl",
}

for c, path  in one_hot_encoders.items():
    with open(path, 'rb') as f:
        one_hot_encoders[c] = pickle.load(f)

with open(models_folder / "one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder = pickle.load(f)

def get_features(model_name, df, flight_df):

    # if "MZFW" in df.columns:
    #     if df["MZFW"].values[0] > 100:
    #         airframe_enc = one_hot_encoders['heavy'].transform(np.array(df['typecode'].values[0]).reshape(-1, 1)).toarray()
    #     else:
    #         airframe_enc = one_hot_encoders['light'].transform(np.array(df['typecode'].values[0]).reshape(-1, 1)).toarray()
    # airframe_enc = one_hot_encoder.transform(np.array(df['typecode'].values[0]).reshape(-1, 1)).toarray()

    gtd_so_far = df["GTD_total"].min()
    gtd_flight = flight_df["GTD_total"].max()
    snippet_duration = df.index.max() - df.index.min()
    altitude_mean = df["altitude"].mean()
    speed_mean = df["groundspeed"].mean()
    roc_mean = df["vertical_rate"].mean()
    lon_mean = df["longitude"].mean()
    lat_mean = df["latitude"].mean()
    track_mean = df["track"].mean()
    altitude_min = df["altitude"].min()
    altitude_max = df["altitude"].max()
    if "CR_" in model_name:
        features = np.array([
            gtd_so_far,
            gtd_flight,
            snippet_duration,
            altitude_min,
            altitude_max,
            altitude_mean,
            speed_mean,
            roc_mean,
            # lon_mean,
            # lat_mean,
            # track_mean,
            df["MZFW"].values[0]
        ]).reshape(1, -1)
        # features = np.concat([features, airframe_enc], axis=1)

    # TODO discuss if speed & roc mean is enough
    elif "CL_" in model_name:
        features = np.array([
            gtd_so_far,
            gtd_flight,
            snippet_duration,
            altitude_min,
            altitude_max,
            altitude_mean,
            speed_mean,
            roc_mean,
            df["MZFW"].values[0]
        ]).reshape(1, -1)
        # features = np.concat([features, airframe_enc], axis=1)
    elif "DE_" in model_name:
        features = np.array([
            gtd_so_far,
            gtd_flight,
            snippet_duration,
            altitude_min,
            altitude_max,
            altitude_mean,
            speed_mean,
            roc_mean,
            df["MZFW"].values[0]
        ]).reshape(1, -1)
    else:
        features = np.array([
            gtd_so_far,
            gtd_flight,
            snippet_duration,
            altitude_min,
            altitude_max,
            altitude_mean,
            speed_mean,
            roc_mean,
            # lon_mean,
            # lat_mean,
            # track_mean,
            ]).reshape(1, -1)
    return features

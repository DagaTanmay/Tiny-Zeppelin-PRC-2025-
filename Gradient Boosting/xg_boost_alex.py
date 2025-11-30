from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import parse_version

import xgboost as xgb

input_file = Path("G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\labeled_data_distribution.csv")



input_df = pd.read_csv(input_file)

#assign heavy category
l = ["B78","B77","B76","B74","A38","A35","A34","A33","MD1"]
#l = ["B78","A33"]
#l = ["B77","A34"]
# Neue Spalte erstellen basierend auf den ersten drei Zeichen
input_df['heavy_class'] = input_df['airframe'].apply(lambda x: 1 if any(x[:3] == prefix for prefix in l) else 0)

#input_df = input_df[input_df["heavy_class"] == 0]
#input_df = input_df[input_df["DE"] == 1.0]
input_df.dropna(axis=0, inplace=True)

features = ["GDT_total_start","total_time","mean_altitude","mean_groundspeed","mean_roc","min_altitude","max_altitude"]

X = np.array([input_df["GDT_total_start"].values,
     input_df["total_time"].values,
     input_df["mean_altitude"].values,
     input_df["mean_groundspeed"].values,
     input_df["mean_roc"].values,
     input_df["min_altitude"].values,
     input_df["max_altitude"].values]).transpose()

y = np.array(input_df["fuel_kg"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)

xgb_model = xgb.XGBRegressor(n_jobs=1).fit(X_train, y_train)

rmse = root_mean_squared_error(y_test, xgb_model.predict(X_test))
print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))
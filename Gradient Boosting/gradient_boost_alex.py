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

input_file = Path("G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\labeled_data_distribution.csv")



input_df = pd.read_csv(input_file)

#assign heavy category
l = ["B78","B77","B76","B74","A38","A35","A34","A33","MD1"]
l = ["B78","A33"]
#l = ["B77","A34"]
# Neue Spalte erstellen basierend auf den ersten drei Zeichen
input_df['heavy_class'] = input_df['airframe'].apply(lambda x: 1 if any(x[:3] == prefix for prefix in l) else 0)

input_df = input_df[input_df["heavy_class"] == 1]
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

params = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 5,
    "learning_rate": 0.1,
    "loss": "squared_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

rmse = root_mean_squared_error(y_test, reg.predict(X_test))
print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))

test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(features)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
fig.tight_layout()
plt.show()

print()
"""
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()

print()
"""
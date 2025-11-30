from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.extend([str(Path(__file__).parent.parent / "utils")])

from add_airframe_info import add_airframe_info
from sklearn import datasets, ensemble, linear_model
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from make_feature_arrays import get_x_y_from_df

from pickle import dump

input_file = Path("G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\train_no_interpolation_19features.csv")

df = pd.read_csv(input_file)
df.dropna(axis=0, inplace=True)

# pos_alt_df = input_df[input_df["mean_altitude"] > 0]
#
# total_time = np.array(np.append(pos_alt_df["total_time"].values, [0, 0]))[:, np.newaxis]
# fuel_kg = np.append(pos_alt_df["fuel_kg"].values, [0, 0])
#
# ransac = linear_model.RANSACRegressor()
# ransac.fit(total_time, fuel_kg)
#
#
#
# line_X = np.arange(total_time.min(), total_time.max())[:, np.newaxis]
# line_y = np.array(ransac.predict(line_X))
# line_y_idle = np.array(line_X*0.3)
#
# plt.scatter(total_time, fuel_kg, color="yellowgreen", marker=".")
# plt.plot(
#     line_X,
#     line_y,
#     color="cornflowerblue",
#     label="RANSAC regressor",
# )
# # plt.plot(line_X, line_y_idle, color="navy", label="idling fuel")
# plt.legend()
# plt.show()
#
# line_01 = np.array([0, 1])[:, np.newaxis]
# line_y01 = np.array(ransac.predict(line_01))

# print(line_y01)

flight_ids = df['flight_id'].unique()

train_ids = np.random.choice(flight_ids, int(len(flight_ids) * 0.8))
train_df = df[df['flight_id'].isin(train_ids)]
val_df = df[~df['flight_id'].isin(train_ids)]

X_train, y_train = get_x_y_from_df(train_df, phase="all")
X_valid, y_valid = get_x_y_from_df(val_df, phase="all")

params = {
    "n_estimators": 800,
    "max_depth": 8,
    "min_samples_split": 20,
    "learning_rate": 0.01,
    "loss": "squared_error",
    "n_iter_no_change": 10,
    "tol": 1e-2,
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

val_preds = reg.predict(X_valid)

rmse = root_mean_squared_error(y_valid, val_preds)
print("The root mean squared error (RMSE) on validation set: {:.4f}".format(rmse))


test_score = np.zeros((len(reg.train_score_),), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_valid)):
    test_score[i] = mean_squared_error(y_valid, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("All phases")
plt.plot(
    np.arange(len(reg.train_score_)) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(len(reg.train_score_)) + 1, test_score, "r-", label="Validation Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()

with open("all_phase_model.pkl", "wb") as f:
    dump(reg, f, protocol=5)

print()
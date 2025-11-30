from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from pickle import dump

input_file = Path("G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\Fuel Consumption\\fuel_train.csv")


input_df = pd.read_csv(input_file)
input_df.dropna(axis=0, inplace=True)

start_time = pd.to_datetime(input_df["start"])
end_time = pd.to_datetime(input_df["end"])



X = np.array([(end_time - start_time).dt.total_seconds(),
     ]).transpose()

y = np.array(input_df["fuel_kg"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=13
)

params = {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_split": 20,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

val_preds = reg.predict(X_valid)

rmse = root_mean_squared_error(y_valid, val_preds)
print("The root mean squared error (RMSE) on validation set: {:.4f}".format(rmse))


test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_valid)):
    test_score[i] = mean_squared_error(y_valid, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("All phases")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Validation Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()

with open("duration_model.pkl", "wb") as f:
    dump(reg, f, protocol=5)

print()
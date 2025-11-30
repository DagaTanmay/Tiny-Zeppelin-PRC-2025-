from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.extend([str(Path(__file__).parent.parent / "utils")])

from add_airframe_info import add_airframe_info
from make_feature_arrays import get_x_y_from_df
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from pickle import dump, load

input_file = Path("G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\train_no_interpolation_19features.csv")

original_input_df = pd.read_csv(input_file)
original_input_df = original_input_df[original_input_df["CL"] == 1.0]
original_input_df.dropna(axis=0, inplace=True)
original_input_df = add_airframe_info(original_input_df)

total_rmse = []
original_input_df['weight_class'] = 'light'
original_input_df.loc[original_input_df['MZFW'] > 100, 'weight_class'] = 'heavy'
#
# input_df["airframe_class"] = input_df['age_class'] + "_" + input_df['weight_class']
#
# # cl = ""
# # df = input_df

for cl, df in original_input_df.groupby('weight_class'):
    print(f"Class {cl} has {df.shape[0]} points")
    model_name = f'climb_model_{cl}'

    with open(f'one_hot_encoder_{cl}.pkl', "rb") as f:
        one_hot_encoder = load(f)

# if True:
#     with open(f'one_hot_encoder.pkl', "rb") as f:
#         one_hot_encoder = load(f)
#
#     df = input_df

    flight_ids = df['flight_id'].unique()

    train_ids = np.random.choice(flight_ids, int(len(flight_ids) * 0.8))
    train_df = df[df['flight_id'].isin(train_ids)]
    val_df = df[~df['flight_id'].isin(train_ids)]

    X_train, y_train = get_x_y_from_df(train_df, phase="CL")
    X_valid, y_valid = get_x_y_from_df(val_df, phase="CL")

    # X_train, X_valid, y_train, y_valid = train_test_split(
    #     X, y, test_size=0.2, random_state=13
    # )

    params = {
        "n_estimators": 1000,
        "max_depth": 5,
        "min_samples_split": 10,
        "learning_rate": 0.005,
        "loss": "squared_error",
        "n_iter_no_change": 10,
        "tol": 1e-2,
    }

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)

    val_preds = reg.predict(X_valid)

    rmse = root_mean_squared_error(y_valid, val_preds)
    print("The root mean squared error (RMSE) on validation set: {:.4f}".format(rmse))
    total_rmse.append(rmse * df.shape[0])

    test_score = np.zeros((len(reg.train_score_),), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_valid)):
        test_score[i] = mean_squared_error(y_valid, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title(model_name)
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

    with open(f"climb_model_{cl}.pkl", "wb") as f:
        dump(reg, f, protocol=5)

    # with open(f"climb_model.pkl", "wb") as f:
    #     dump(reg, f, protocol=5)

avg_rmse = np.sum(total_rmse) / original_input_df.shape[0]
print(f"Weighted average RMSE: {round(avg_rmse, 2)}")
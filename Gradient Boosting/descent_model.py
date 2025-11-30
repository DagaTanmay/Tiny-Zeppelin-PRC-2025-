from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor

import sys
sys.path.extend([str(Path(__file__).parent.parent / "utils")])

from add_airframe_info import add_airframe_info
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from make_feature_arrays import get_x_y_from_df

from pickle import dump, load

input_file = Path("G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\train_no_interpolation_19features.csv")

input_df = pd.read_csv(input_file)
input_df = input_df[input_df["DE"] == 1.0]
input_df.dropna(axis=0, inplace=True)
input_df = add_airframe_info(input_df)
input_df['weight_class'] = 'light'
input_df.loc[input_df['MZFW'] > 100, 'weight_class'] = 'heavy'

input_df["airframe_class"] = input_df['age_class'] + "_" + input_df['weight_class']
total_rmse = []

# cl = ""
# df = input_df

# with open('one_hot_encoder.pkl', "rb") as f:
#    one_hot_encoder = load(f)

for cl, df in input_df.groupby('weight_class'):
    print(f"Class {cl} has {df.shape[0]} points")
    model_name = f'descent_model_{cl}'

    with open(f'one_hot_encoder_{cl}.pkl', "rb") as f:
        one_hot_encoder = load(f)

# if True:
#     df = input_df
#     with open(f'one_hot_encoder.pkl', "rb") as f:
#         one_hot_encoder = load(f)
#     model_name = "single_descent"
    flight_ids = df['flight_id'].unique()

    train_ids = np.random.choice(flight_ids, int(len(flight_ids) * 0.8))
    train_df = df[df['flight_id'].isin(train_ids)]
    val_df = df[~df['flight_id'].isin(train_ids)]

    X_train, y_train = get_x_y_from_df(train_df, phase="DE")
    X_valid, y_valid = get_x_y_from_df(val_df, phase="DE")

    params = {
        "n_estimators": 600,
        "max_depth": 5,
        "min_samples_split": 12,
        "learning_rate": 0.005,
        "loss": "squared_error",
        "n_iter_no_change": 10,
        "tol": 1e-2,
    }

    XGB_PARAMS = dict(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=25,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=0
    )

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)
    # reg = XGBRegressor(**XGB_PARAMS)
    # reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid,y_valid)], verbose=False)

    val_preds = reg.predict(X_valid)

    rmse = root_mean_squared_error(y_valid, val_preds)
    print(f"The root mean squared error (RMSE) for {model_name} on validation set: {round(rmse, 2)}")
    total_rmse.append(rmse * df.shape[0])

    ## Learning curve XGBoost

    # results = reg.evals_result()
    # # plot learning curves
    # plt.plot(results['validation_0']['rmse'], label='train')
    # plt.plot(results['validation_1']['rmse'], label='test')
    # # show the legend
    # plt.title(model_name)
    # plt.legend()
    # # show the plot
    # plt.show()

    ## Learning curve gradient boosting
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

    with open(f"descent_model_{cl}.pkl", "wb") as f:
        dump(reg, f, protocol=5)
    # with open(f"descent_model.pkl", "wb") as f:
    #     dump(reg, f, protocol=5)

avg_rmse = np.sum(total_rmse) / input_df.shape[0]
print(f"Weighted average RMSE: {round(avg_rmse, 2)}")
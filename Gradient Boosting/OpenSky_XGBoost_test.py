
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib


# config
DATA_FILE = r"G:\MO-PLM\Transfer\Open Sky Data Challenge\labeled_data_distribution.csv"

# features
NUMERIC_FEATURES = [
    "total_time", "GDT_total_start", "mean_altitude", "mean_groundspeed",
    "mean_roc", "min_altitude", "max_altitude", "CL", "CR", "DE", "LVL"
]


CATEGORICAL_FEATURES = ["airframe"]
TARGET = "fuel_kg"

# XGBoost hyperparameters
XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=0
)

TEST_SIZE = 0.2
RANDOM_STATE = 42



def main():
   # load data
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"File not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"Data loaded: {df.shape[0]:,} lines, {df.shape[1]} columns")

    #split up test and training data (80% train, 20% test)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(float).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    #preprocessing

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])

    # build model

    model = XGBRegressor(**XGB_PARAMS)
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("xgb", model)
    ])

    print("Training started")
    pipe.fit(X_train, y_train)
    print("Training done")

    # evaluation
    y_pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    baseline_rmse = mean_squared_error(y_test, np.full_like(y_test, y_train.mean()), squared=False)

    print(f"Results:")
    print(f"   RMSE (fuel_kg): {rmse:,.3f}")
    print(f"   Baseline (mean): {baseline_rmse:,.3f}")

    # feature importance

    try:
        cat_encoder: OneHotEncoder = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
        feature_names = NUMERIC_FEATURES + cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()

        booster = pipe.named_steps["xgb"]
        importances = booster.feature_importances_

        top_idx = np.argsort(importances)[::-1][:10]
        print("Top 10 Feature Importances:")
        for i in top_idx:
            print(f"   {feature_names[i]:35s} {importances[i]:.4f}")
    except Exception as e:
        warnings.warn(f"Feature Importances could not be calculated: {e}")

    # save model as .pkl

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/xgb_fuel_pipeline.pkl")
    print("Model saved: models/xgb_fuel_pipeline.pkl")


# run
if __name__ == "__main__":
    main()

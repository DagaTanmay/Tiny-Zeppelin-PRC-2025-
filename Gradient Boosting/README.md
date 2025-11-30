# Project Utilities and Models

This repository contains Python scripts and pre-trained machine learning models related to flight phase prediction, climb/cruise/descent modeling, and fuel estimation.

## Contents

### 1. Python Scripts (`.py`)

These scripts contain the code for training, evaluating, or testing machine learning models:

- **Model Training & Testing**
  - `all_phase_model.py` — Model for predicting all flight phases.
  - `climb_model.py` — Climb phase model.
  - `cruise_model.py` — Cruise phase model.
  - `descent_model.py` — Descent phase model.
  - `duration_model.py` — Predicts flight durations.
  - `gradient_boost_alex.py` — Gradient Boosting-based models.
  - `xg_boost_alex.py` — XGBoost-based models.

- **Feature and Data Utilities**
  - `make_feature_arrays.py` — Generates feature arrays from raw data.
  - `one_hot_encoder.py` — Performs one-hot encoding of categorical features.
  - `OpenSky_XGBoost_test.py` — Script for testing XGBoost models on OpenSky data.

---

### 2. Pre-trained Model Files (`.pkl`)

These files are serialized versions of trained models for direct use without retraining:

- **All Phase**
  - `all_phase_model.pkl`

- **Climb Phase**
  - `climb_model.pkl`
  - `climb_model_heavy.pkl`
  - `climb_model_light.pkl`
  - `climb_model_neo.pkl`
  - `climb_model_old.pkl`

- **Cruise Phase**
  - `cruise_model.pkl`
  - `cruise_model_heavy.pkl`
  - `cruise_model_light.pkl`
  - `cruise_model_neo.pkl`
  - `cruise_model_old.pkl`

- **Descent Phase**
  - `descent_model.pkl`
  - `descent_model_heavy.pkl`
  - `descent_model_light.pkl`
  - `descent_model_neo.pkl`
  - `descent_model_old.pkl`

- **Duration**
  - `duration_model.pkl`

- **Fuel Pipeline**
  - `xgb_fuel_pipeline.pkl`

---

### 3. README

- `README.md` — This file.

---

### Notes

- Python scripts (`.py`) are used for **training, testing, or feature engineering**.  
- Pickle files (`.pkl`) are **pre-trained models** ready for inference.  
- Model names like `*_heavy`, `*_light`, `*_neo`, `*_old` differentiate variants based on aircraft type or configuration.

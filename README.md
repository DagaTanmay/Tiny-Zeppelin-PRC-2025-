# PRC Data Challenge 2025 – Flight Fuel Prediction

This repository contains code and models developed for the **PRC Data Challenge 2025**, organized to predict fuel consumption at specific intervals of a flight using Machine Learning (ML).  

The repository is structured to separate model files, orchestration logic, and preprocessing utilities.

---

## Overview

The **Performance Review Commission (PRC)**, with support from the **OpenSky Network**, engages data scientists to solve aviation-related data challenges using open data.  

The 2025 edition focuses on building ML models to predict **fuel burnt** during flight intervals. The dataset contains flight segments and corresponding fuel consumption, and participants are encouraged to openly share solutions.

---

## Repository Structure

### 1. `grad_boost/`
Contains **Machine Learning models** and related scripts:  

- **Python scripts (`.py`)** — For training and testing models.
- **Pre-trained model files (`.pkl`)** — Ready-to-use serialized models.
- Focused on **Gradient Boosting** and **XGBoost** approaches for predicting fuel consumption and flight phases.

### 2. `orchestrator/`
Contains scripts to **orchestrate the workflow**:  

- Data ingestion, preprocessing, model training, and evaluation pipelines.
- Automates experiment runs and prediction workflows.

### 3. `utils/`
Contains **preprocessing and utility scripts**:  

- Data cleaning, feature engineering, and transformations.
- File operations for handling OpenSky datasets, coordinate transformations, and other helper functions.

### 3. `Final Model`
Contains the **Final Model**

---

## Notes

- Scripts in `grad_boost/` handle ML model creation, testing, and inference.  
- Pickle files (`.pkl`) are pre-trained models ready for inference.  
- `orchestrator/` manages workflow automation.  
- `utils/` contains standalone helper scripts used across the project.

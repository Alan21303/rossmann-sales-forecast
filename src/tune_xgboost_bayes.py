# src/tune_xgboost_bayes.py

import os
import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# ------------------------------
# Metrics
# ------------------------------
def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error"""
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

# ------------------------------
# Load dataset
# ------------------------------
def load_data(data_path="data/processed_train.csv"):
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    X = df.drop(columns=["Sales"])
    y = df["Sales"]
    return X, y

# ------------------------------
# Objective function for Optuna
# ------------------------------
def objective(trial, X, y):
    # Define search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "tree_method": "hist",
        "objective": "reg:squarederror",
    }

    # TimeSeries Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    rmspe_scores = []

    for train_idx, valid_idx in tscv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        preds = model.predict(X_valid)
        score = rmspe(y_valid.values, preds)
        rmspe_scores.append(score)

    # Return mean RMSPE across folds
    return np.mean(rmspe_scores)

# ------------------------------
# Main
# ------------------------------
def main():
    print(" Loading processed dataset...")
    X, y = load_data()
    print(f" Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")

    # ------------------------------
    # Optuna study
    # ------------------------------
    print("🔹 Starting Bayesian Optimization with Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=30, show_progress_bar=True)

    print("🎯 Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    print(f"Best CV RMSPE: {study.best_value:.4f}")

    # ------------------------------
    # Train final model with best params
    # ------------------------------
    print("🔹 Training final XGBoost model with best hyperparameters...")
    best_params = study.best_params
    best_params.update({
        "random_state": 42,
        "tree_method": "hist",
        "objective": "reg:squarederror"
    })
    final_model = XGBRegressor(**best_params)
    final_model.fit(X, y, verbose=True)

    # ------------------------------
    # Save model
    # ------------------------------
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "xgboost_bayes_opt.pkl")
    joblib.dump(final_model, model_path)
    print(f" Final model saved at '{model_path}'")

if __name__ == "__main__":
    main()

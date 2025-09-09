# src/train_models.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error (safe version)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0  # avoid division by zero
    if mask.sum() == 0:
        return np.nan

    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


def evaluate_model(name, model, X_train, y_train, X_valid, y_valid):
    """Train, predict and print evaluation metrics."""
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)

    # Training predictions
    train_preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_rmspe = rmspe(y_train, train_preds)

    # Validation predictions
    valid_preds = model.predict(X_valid)
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
    valid_rmspe = rmspe(y_valid, valid_preds)

    print(f"\n {name} Performance:")
    print(f"  - Train RMSE : {train_rmse:.2f}")
    print(f"  - Train RMSPE: {train_rmspe:.4f}")
    print(f"  - Valid RMSE : {valid_rmse:.2f}")
    print(f"  - Valid RMSPE: {valid_rmspe:.4f}")

    return model


def build_random_forest():
    """Build Random Forest Regressor."""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )


def build_xgboost():
    """Build XGBoost Regressor."""
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        eval_metric="rmse"
    )


def main():
    # Load processed dataset
    print(" Loading processed dataset...")
    data_path = os.path.join("data", "processed_train.csv")
    df = pd.read_csv(data_path)

    # Drop string/date columns that models can't use
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # Features and target
    X = df.drop(columns=["Sales"])
    y = df["Sales"]


    # Train/validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f" Dataset split: {X_train.shape[0]} train rows, {X_valid.shape[0]} validation rows")

    # Train Random Forest
    rf_model = evaluate_model("Random Forest", build_random_forest(), X_train, y_train, X_valid, y_valid)

    # Train XGBoost
    xgb_model = evaluate_model("XGBoost", build_xgboost(), X_train, y_train, X_valid, y_valid)

    # Save trained models
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(model_dir, "random_forest.pkl"))
    joblib.dump(xgb_model, os.path.join(model_dir, "xgboost.pkl"))

    print("\n Models saved in 'models/' folder.")


if __name__ == "__main__":
    main()

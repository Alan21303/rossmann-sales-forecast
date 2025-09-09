# src/ensemble_xgb_lgbm_rf.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
import lightgbm as lgb


# ---------- Metrics ----------
def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error (safe version)"""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


def evaluate_model(model, X_train, y_train, X_valid, y_valid, name="Model"):
    """Train & evaluate a model"""
    model.fit(X_train, y_train)

    preds_train = model.predict(X_train)
    preds_valid = model.predict(X_valid)

    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, preds_train)),
        "train_rmspe": rmspe(y_train, preds_train),
        "valid_rmse": np.sqrt(mean_squared_error(y_valid, preds_valid)),
        "valid_rmspe": rmspe(y_valid, preds_valid),
    }

    print(f"\n  {name} Performance:")
    print(f"  - Train RMSE : {metrics['train_rmse']:.2f}")
    print(f"  - Train RMSPE: {metrics['train_rmspe']:.4f}")
    print(f"  - Valid RMSE : {metrics['valid_rmse']:.2f}")
    print(f"  - Valid RMSPE: {metrics['valid_rmspe']:.4f}")

    return model, preds_train, preds_valid, metrics


# ---------- Main ----------
def main():
    print("\n  Loading processed dataset...")
    data_path = os.path.join("data", "processed_train.csv")
    df = pd.read_csv(data_path)

    # Drop non-numeric columns
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    X = df.drop(columns=["Sales"])
    y = df["Sales"]

    # Train/Validation Split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"  Dataset split: {X_train.shape[0]} train rows, {X_valid.shape[0]} validation rows")

    # ---------- RandomForest ----------
    print("\n 🔹 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )
    rf, rf_preds_train, rf_preds_valid, _ = evaluate_model(
        rf, X_train, y_train, X_valid, y_valid, name="Random Forest"
    )

    # ---------- XGBoost ----------
    print("\n  Loading Optimized XGBoost from Step 1...")
    xgb_path = os.path.join("models", "xgboost_bayes_opt.pkl")

    if os.path.exists(xgb_path):
        xgb = joblib.load(xgb_path)
        # retrain on current split for fair comparison
        xgb, xgb_preds_train, xgb_preds_valid, _ = evaluate_model(
            xgb, X_train, y_train, X_valid, y_valid, name="XGBoost (Optimized)"
        )
    else:
        print("  Optimized XGBoost not found. Training fresh one...")
        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            eval_metric="rmse"
        )
        xgb, xgb_preds_train, xgb_preds_valid, _ = evaluate_model(
            xgb, X_train, y_train, X_valid, y_valid, name="XGBoost"
        )

    # ---------- LightGBM ----------
    print("\n  Training LightGBM...")
    lgbm = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgbm, lgbm_preds_train, lgbm_preds_valid, _ = evaluate_model(
        lgbm, X_train, y_train, X_valid, y_valid, name="LightGBM"
    )

    # ---------- Ensemble ----------
    print("\n  Creating Ensemble (XGBoost + LightGBM + RandomForest)...")
    w_xgb, w_lgbm, w_rf = 0.5, 0.3, 0.2  # can be tuned later

    ensemble_preds_train = (
        w_xgb * xgb_preds_train +
        w_lgbm * lgbm_preds_train +
        w_rf * rf_preds_train
    )
    ensemble_preds_valid = (
        w_xgb * xgb_preds_valid +
        w_lgbm * lgbm_preds_valid +
        w_rf * rf_preds_valid
    )

    ensemble_metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, ensemble_preds_train)),
        "train_rmspe": rmspe(y_train, ensemble_preds_train),
        "valid_rmse": np.sqrt(mean_squared_error(y_valid, ensemble_preds_valid)),
        "valid_rmspe": rmspe(y_valid, ensemble_preds_valid),
    }

    print("\n  Ensemble (XGB + LGBM + RF) Performance:")
    print(f"  - Train RMSE : {ensemble_metrics['train_rmse']:.2f}")
    print(f"  - Train RMSPE: {ensemble_metrics['train_rmspe']:.4f}")
    print(f"  - Valid RMSE : {ensemble_metrics['valid_rmse']:.2f}")
    print(f"  - Valid RMSPE: {ensemble_metrics['valid_rmspe']:.4f}")

    # ---------- Save Models ----------
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))
    joblib.dump(xgb, os.path.join(model_dir, "xgboost_bayes_opt.pkl"))
    joblib.dump(lgbm, os.path.join(model_dir, "lightgbm.pkl"))

    np.save(os.path.join(model_dir, "ensemble_preds_valid.npy"), ensemble_preds_valid)

    print("\n  Models and ensemble predictions saved in 'models/' folder.")


if __name__ == "__main__":
    main()

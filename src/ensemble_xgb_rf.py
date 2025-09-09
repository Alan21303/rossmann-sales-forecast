# ensemble_xgb_rf.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --------------------------
# Metrics
# --------------------------
def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error"""
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

def evaluate_model(name, y_train, train_pred, y_valid, valid_pred):
    """Print metrics for train and validation sets"""
    print(f"\n {name} Performance:")
    print(f"  - Train RMSE : {np.sqrt(mean_squared_error(y_train, train_pred)):.2f}")
    print(f"  - Train RMSPE: {rmspe(y_train, train_pred):.4f}")
    print(f"  - Valid RMSE : {np.sqrt(mean_squared_error(y_valid, valid_pred)):.2f}")
    print(f"  - Valid RMSPE: {rmspe(y_valid, valid_pred):.4f}")

# --------------------------
# Load Data
# --------------------------
print(" Loading processed dataset...")
data_path = os.path.join("data", "processed_train.csv")
df = pd.read_csv(data_path)

# Drop string columns
if "Date" in df.columns:
    df = df.drop(columns=["Date"])

X = df.drop(columns=["Sales"])
y = df["Sales"]

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f" Dataset split: {X_train.shape[0]} train rows, {X_valid.shape[0]} validation rows")

# --------------------------
# Train RandomForest
# --------------------------
print("\n Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)
rf_valid_pred = rf_model.predict(X_valid)

evaluate_model("Random Forest", y_train, rf_train_pred, y_valid, rf_valid_pred)

# --------------------------
# Load Optimized XGBoost
# --------------------------
print("\n Loading Optimized XGBoost from Step 1...")
xgb_model_path = os.path.join("models", "xgboost_bayes_opt.pkl")
xgb_model = joblib.load(xgb_model_path)
xgb_train_pred = xgb_model.predict(X_train)
xgb_valid_pred = xgb_model.predict(X_valid)

evaluate_model("XGBoost (Optimized)", y_train, xgb_train_pred, y_valid, xgb_valid_pred)

# --------------------------
# Ensemble: Weighted Average
# --------------------------
print("\n Creating Ensemble (XGBoost + RandomForest)...")

# Example: 0.6 XGBoost + 0.4 RF (you can tune weights)
weight_xgb = 0.6
weight_rf = 0.4

ensemble_train_pred = weight_xgb * xgb_train_pred + weight_rf * rf_train_pred
ensemble_valid_pred = weight_xgb * xgb_valid_pred + weight_rf * rf_valid_pred

evaluate_model("Ensemble (XGB + RF)", y_train, ensemble_train_pred, y_valid, ensemble_valid_pred)

# --------------------------
# Save models and predictions
# --------------------------
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(rf_model, os.path.join(model_dir, "random_forest.pkl"))
# Ensemble is just a combination, so no direct model save
# Optionally save ensemble predictions
np.save(os.path.join(model_dir, "ensemble_valid_pred.npy"), ensemble_valid_pred)

print("\n Models and ensemble predictions saved in 'models/' folder.")

# app.py
"""
Rossmann Sales Forecasting Streamlit App

This app:
- Loads processed historical sales data
- Detects available trained models in the models/ folder
- Lets the user select a store, date range, and model
- Compares actual vs predicted sales for that slice
- Displays metrics (RMSE, RMSPE), plots, tables, and allows CSV download
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rossmann Sales Forecast", layout="wide")

@st.cache_data
def load_processed_data(path="data/processed_train.csv"):
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def list_models(model_dir="models"):
    models = {}
    if not os.path.exists(model_dir):
        return models
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".pkl"):
            name = os.path.splitext(fname)[0]
            models[name] = os.path.join(model_dir, fname)
    return models

def safe_rmspe(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

def safe_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_model(path):
    return joblib.load(path)

def make_predictions(model, X):
    return model.predict(X)

def prepare_feature_slice(df, store_id, date_from, date_to):
    mask = (df["Store"] == store_id) & (df["Date"] >= date_from) & (df["Date"] <= date_to)
    return df.loc[mask].copy().reset_index(drop=True)

st.sidebar.title("Controls")
st.sidebar.markdown("Select a store, date range, and model.")

df = load_processed_data()
models_available = list_models()

stores = sorted(df["Store"].unique().tolist())
store_sel = st.sidebar.selectbox("Select Store", stores, index=0 if stores else None)

min_date, max_date = df["Date"].min(), df["Date"].max()
date_from = st.sidebar.date_input("From", min_date.date() if pd.notna(min_date) else None)
date_to = st.sidebar.date_input("To", max_date.date() if pd.notna(max_date) else None)

if models_available:
    model_names = list(models_available.keys())
    model_sel = st.sidebar.selectbox("Choose model", model_names, index=0)
else:
    model_sel = None

st.sidebar.markdown("---")
show_table = st.sidebar.checkbox("Show raw predictions table", value=True)
download_preds = st.sidebar.checkbox("Enable CSV download", value=True)
run_btn = st.sidebar.button("Run predictions")

st.title("Rossmann — Sales Forecast Explorer")
st.markdown("Pick a store, date range, and model to compare actual vs predicted sales.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows in data", f"{len(df):,}")
with col2:
    st.metric("Date range", f"{min_date.date()} → {max_date.date()}")
with col3:
    st.metric("Stores", f"{len(stores):,}")

st.markdown("---")

if run_btn:
    if store_sel is None:
        st.error("No store selected.")
    elif model_sel is None:
        st.error("No model found in `models/`. Please ensure `.pkl` files exist.")
    else:
        slice_df = prepare_feature_slice(df, store_sel, pd.to_datetime(date_from), pd.to_datetime(date_to))
        if slice_df.empty:
            st.warning("No data found for this selection.")
        else:
            st.success(f"Found {len(slice_df)} rows for Store {store_sel} between {date_from} and {date_to}.")

            feature_cols = [c for c in slice_df.columns if c not in ("Sales", "Date")]
            X_slice, y_true = slice_df[feature_cols], slice_df["Sales"].values

            try:
                model_obj = load_model(models_available[model_sel])
                preds = make_predictions(model_obj, X_slice)
            except Exception as e:
                st.error(f"Model failed: {e}")
                st.stop()

            results = slice_df[["Store", "Date"]].copy()
            results["ActualSales"], results["PredictedSales"] = y_true, preds

            rmse_val, rmspe_val = safe_rmse(y_true, preds), safe_rmspe(y_true, preds)

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Rows in slice", f"{len(results):,}")
            mcol2.metric("RMSE", f"{rmse_val:.2f}")
            mcol3.metric("RMSPE", f"{rmspe_val:.4f}")

            st.markdown("**Actual vs Predicted Sales**")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(results["Date"], results["ActualSales"], label="Actual", linewidth=2)
            ax.plot(results["Date"], results["PredictedSales"], label=f"Predicted ({model_sel})", linewidth=2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            results["Diff"] = results["PredictedSales"] - results["ActualSales"]
            results["AbsDiff"] = results["Diff"].abs()

            st.markdown("#### Aggregate summary")
            summary = {
                "Mean Actual": [results["ActualSales"].mean()],
                "Mean Predicted": [results["PredictedSales"].mean()],
                "Mean Absolute Error": [results["AbsDiff"].mean()],
                "Median Absolute Error": [results["AbsDiff"].median()],
            }
            st.table(pd.DataFrame(summary).T.rename(columns={0: "Value"}))

            if show_table:
                st.markdown("#### Daily predictions")
                st.dataframe(results.sort_values("Date").reset_index(drop=True), use_container_width=True)

            if download_preds:
                st.download_button(
                    label="Download predictions (CSV)",
                    data=results.to_csv(index=False).encode(),
                    file_name=f"predictions_store_{store_sel}_{date_from}_{date_to}.csv",
                    mime="text/csv",
                )

            st.markdown("---")
            st.info("This compares predictions to historical data. Forecasting future dates requires generating features for those dates.")
else:
    st.info("Click 'Run predictions' to evaluate models for a store and date range.")

st.markdown("---")
st.markdown("Rossmann Sales Forecasting — Streamlit App")


import pandas as pd
import numpy as np
import os

def load_and_merge_data(data_path="data"):
    """Load train and store data, then merge."""
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    store = pd.read_csv(os.path.join(data_path, "store.csv"))
    
    # Merge datasets on Store
    data = pd.merge(train, store, on="Store", how="left")
    
    # Convert Date column to datetime
    data["Date"] = pd.to_datetime(data["Date"])
    
    # Sort by store and date (important for lag features)
    data = data.sort_values(["Store", "Date"]).reset_index(drop=True)
    
    return data

def add_temporal_features(df):
    """Add time-based features for seasonality and trends."""
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    return df

def add_lag_features(df, lags=[7, 14, 28]):
    """Add lag features for past sales values."""
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)
    return df

def add_rolling_features(df, windows=[7, 14, 28]):
    """Add rolling average sales features."""
    for window in windows:
        df[f"roll_mean_{window}"] = (
            df.groupby("Store")["Sales"].shift(1).rolling(window=window).mean()
        )
    return df

def add_promo_holiday_features(df):
    """Add binary flags for promotions and holidays."""
    df["is_promo"] = df["Promo"].astype(int)
    df["is_school_holiday"] = df["SchoolHoliday"].astype(int)
    df["is_state_holiday"] = (df["StateHoliday"] != "0").astype(int)
    return df

def add_store_features(df):
    """Process store-related categorical/numeric features."""
    # Fill missing CompetitionDistance with median
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    
    # Competition open features
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0).astype(int)
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0).astype(int)
    
    # Promo2 handling
    df["Promo2"] = df["Promo2"].fillna(0).astype(int)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0).astype(int)
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0).astype(int)

    # Handle PromoInterval (convert month names to binary flags)
    months_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    for m, num in months_map.items():
        df[f"promo_{m.lower()}"] = df["PromoInterval"].fillna("").apply(lambda x: 1 if m in x else 0)

    # Drop original string column
    if "PromoInterval" in df.columns:
        df = df.drop(columns=["PromoInterval"])

    # Encode categorical features
    df = pd.get_dummies(df, columns=["StoreType", "Assortment", "StateHoliday"], drop_first=True)
    
    return df

def build_features(data_path="data", output_path="data/processed_train.csv"):
    """Main function to build features and save processed data."""
    print("Loading and merging data...")
    df = load_and_merge_data(data_path)
    
    print("Adding temporal features...")
    df = add_temporal_features(df)
    
    print("Adding lag features...")
    df = add_lag_features(df)
    
    print("Adding rolling features...")
    df = add_rolling_features(df)
    
    print("Adding promotion and holiday features...")
    df = add_promo_holiday_features(df)
    
    print("Adding store-specific features...")
    df = add_store_features(df)
    
    # Drop rows with NaNs introduced by lag/rolling
    df = df.dropna().reset_index(drop=True)
    
    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved at {output_path}")
    
    return df


if __name__ == "__main__":
    processed_df = build_features()
    print(processed_df.head())
    print(processed_df.shape)

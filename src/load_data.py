import pandas as pd

train = pd.read_csv('../data/train.csv')
store = pd.read_csv('../data/store.csv')

# Merge on Store column
data = pd.merge(train, store, on='Store', how='left')

# Quick check
print(data.head())
print(data.info())
print(data.isnull().sum())

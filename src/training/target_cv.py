# target_cv.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from metrics import smape, to_log1p, from_log1p

def create_price_bins(df, n_bins=5):
    """
    Bin price into quantiles for stratified K-Fold splitting.
    Returns a new column 'price_bin' (integer 0..n_bins-1)
    """
    prices = df['price'].fillna(0)
    # Use qcut with duplicates handled
    try:
        bins = pd.qcut(prices, q=n_bins, labels=False, duplicates='drop')
    except:
        bins = pd.cut(prices, bins=n_bins, labels=False)
    return bins

def get_stratified_folds(df, target_col='price', n_splits=5, random_state=42):
    """
    Returns a list of (train_idx, val_idx) tuples for CV.
    Stratified on price bins.
    """
    df = df.copy()
    df['price_bin'] = create_price_bins(df, n_bins=n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, val_idx in skf.split(df, df['price_bin']):
        folds.append((train_idx, val_idx))
    df.drop(columns=['price_bin'], inplace=True)
    return folds

# New helper for other scripts
def get_folds(df, target_col='price', n_splits=5, seed=42):
    """
    Simple wrapper that returns CV folds (train_idx, val_idx) list.
    """
    return get_stratified_folds(df, target_col=target_col, n_splits=n_splits, random_state=seed)

def evaluate_smape(y_true, y_pred, use_log=False):
    """
    Compute SMAPE on original price scale.
    If predictions were trained on log1p scale, set use_log=True
    """
    if use_log:
        y_true = from_log1p(y_true)
        y_pred = from_log1p(y_pred)
    return smape(y_true, y_pred)

# Example usage
if __name__ == "__main__":
    df = pd.read_csv(r"D:\Amazon\dataset\raw\train.csv")
    folds = get_stratified_folds(df, n_splits=5)
    print("Created", len(folds), "folds for CV.")

    # Example: log-transform target
    df['y_log'] = to_log1p(df['price'])
    # Sanity check
    print(df[['price', 'y_log']].head())

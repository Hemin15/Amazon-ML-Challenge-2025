import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from target_cv import get_stratified_folds, to_log1p, from_log1p
from metrics import smape
import lightgbm as lgb
from catboost import CatBoostRegressor
from utils import set_seed, safe_read_csv
import matplotlib.pyplot as plt
import os

from lightgbm import early_stopping, log_evaluation

set_seed(42)

# ---------------------------
# Paths
# ---------------------------
NUMERIC_FEATURES = r"D:\Amazon\output\baseline_numeric_features.csv"
EMBEDDINGS_FILE = r"D:\Amazon\output\baseline_embeddings.npy"
TRAIN_CSV = r"D:\Amazon\dataset\raw\train.csv"
OUTPUT_DIR = r"D:\Amazon\output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load features
# ---------------------------
print("Loading numeric features...")
X_num = pd.read_csv(NUMERIC_FEATURES)
print("Numeric shape:", X_num.shape)

print("Loading TF-IDF embeddings...")
X_text = np.load(EMBEDDINGS_FILE)
print("Text embedding shape:", X_text.shape)

# ---------------------------
# Load CSV for target + categorical
# ---------------------------
df = safe_read_csv(TRAIN_CSV)
y = to_log1p(df['price'].values)

# ---------------------------
# Feature engineering: brand target encoding
# ---------------------------
if 'brand' in df.columns:
    df['brand'] = df['brand'].fillna('unknown')
    oof_brand = pd.Series(np.zeros(len(df)), index=df.index)
    folds = get_stratified_folds(df, n_splits=5)
    for tr_idx, val_idx in folds:
        means = df.loc[tr_idx].groupby('brand')['price'].mean()
        oof_brand.iloc[val_idx] = df.loc[val_idx, 'brand'].map(means).fillna(df['price'].mean())
    brand_feat = to_log1p(oof_brand.values).reshape(-1, 1)
else:
    brand_feat = np.zeros((len(df), 1))

# ---------------------------
# Feature engineering: weight/volume normalization
# ---------------------------
for col in ['weight', 'volume']:
    if col in X_num.columns:
        X_num[col] = np.log1p(X_num[col].fillna(0))  # log-transform skewed numeric

# ---------------------------
# Feature engineering: IPQ (item pack quantity)
# ---------------------------
if 'catalog_content' in df.columns:
    import re
    ipq = []
    for text in df['catalog_content'].astype(str):
        match = re.findall(r'(\d+)[\s]*([pP][cC][sS]?|pack|pcs)', text)
        if match:
            ipq.append(int(match[0][0]))
        else:
            ipq.append(1)
    ipq = np.array(ipq).reshape(-1, 1)
else:
    ipq = np.ones((len(df), 1))

# ---------------------------
# Feature engineering: multipack flag
# ---------------------------
multipack_flag = (ipq > 1).astype(int)

# ---------------------------
# Feature engineering: catalog embeddings (TF-IDF + SVD)
# ---------------------------
if 'catalog_content' in df.columns:
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_cat_tfidf = tfidf.fit_transform(df['catalog_content'].astype(str))
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_cat_emb = svd.fit_transform(X_cat_tfidf)
    print("Catalog embeddings shape:", X_cat_emb.shape)
else:
    X_cat_emb = np.zeros((len(df), 50))

# ---------------------------
# Combine all features
# ---------------------------
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)
X = np.hstack([X_num_scaled, X_text, brand_feat, ipq, multipack_flag, X_cat_emb])
print("Combined feature matrix shape:", X.shape)

feature_names = list(X_num.columns) + \
                [f'tfidf_{i}' for i in range(X_text.shape[1])] + \
                ['brand_enc', 'ipq', 'multipack'] + \
                [f'svd_{i}' for i in range(X_cat_emb.shape[1])]

# ---------------------------
# CV setup
# ---------------------------
folds = get_stratified_folds(df, n_splits=5)
oof_lgb = np.zeros(len(df))
oof_cat = np.zeros(len(df))

# ---------------------------
# LightGBM Regressor parameters
# ---------------------------
lgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 128,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# ---------------------------
# CatBoost parameters
# ---------------------------
cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'early_stopping_rounds': 50,
    'verbose': 50
}

# ---------------------------
# Store feature importances
# ---------------------------
lgb_importances = np.zeros(X.shape[1])
cat_importances = np.zeros(X.shape[1])

# ---------------------------
# Training loop
# ---------------------------
for fold, (tr_idx, val_idx) in enumerate(folds):
    print(f"\nFold {fold+1}/5")
    
    X_train, X_val = X[tr_idx], X[val_idx]
    y_train, y_val = y[tr_idx], y[val_idx]
    
    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(100)]
    )
    val_pred_lgb = lgb_model.predict(X_val)
    oof_lgb[val_idx] = val_pred_lgb
    fold_smape_lgb = smape(from_log1p(y_val), from_log1p(val_pred_lgb))
    print(f"Fold {fold+1} LightGBM SMAPE: {fold_smape_lgb:.4f}%")
    lgb_importances += lgb_model.feature_importances_

    # --- CatBoost ---
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    val_pred_cat = cat_model.predict(X_val)
    oof_cat[val_idx] = val_pred_cat
    fold_smape_cat = smape(from_log1p(y_val), from_log1p(val_pred_cat))
    print(f"Fold {fold+1} CatBoost SMAPE: {fold_smape_cat:.4f}%")
    cat_importances += cat_model.get_feature_importance()

# ---------------------------
# Average feature importances
# ---------------------------
lgb_importances /= len(folds)
cat_importances /= len(folds)

fi_df = pd.DataFrame({
    'feature': feature_names,
    'lgb_importance': lgb_importances,
    'cat_importance': cat_importances
}).sort_values(by='lgb_importance', ascending=False)
fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)
print("Saved feature importances ->", os.path.join(OUTPUT_DIR, "feature_importances.csv"))

# ---------------------------
# Plot top 20 features
# ---------------------------
top_n = 20
plt.figure(figsize=(12, 8))
plt.barh(fi_df['feature'][:top_n][::-1], fi_df['lgb_importance'][:top_n][::-1], color='skyblue')
plt.title("Top 20 LightGBM Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lgb_feature_importance.png"))
plt.show()

plt.figure(figsize=(12, 8))
plt.barh(fi_df['feature'][:top_n][::-1], fi_df['cat_importance'][:top_n][::-1], color='orange')
plt.title("Top 20 CatBoost Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cat_feature_importance.png"))
plt.show()

# ---------------------------
# OOF evaluation
# ---------------------------
oof_ensemble = (oof_lgb + oof_cat) / 2
overall_smape_lgb = smape(from_log1p(y), from_log1p(oof_lgb))
overall_smape_cat = smape(from_log1p(y), from_log1p(oof_cat))
overall_smape_ens = smape(from_log1p(y), from_log1p(oof_ensemble))

print("\nOverall OOF SMAPE LightGBM:", overall_smape_lgb)
print("Overall OOF SMAPE CatBoost:", overall_smape_cat)
print("Overall OOF SMAPE Ensemble:", overall_smape_ens)

# ---------------------------
# Save OOF predictions
# ---------------------------
oof_df = pd.DataFrame({
    'sample_id': df['sample_id'],
    'y_log_lgb': oof_lgb,
    'y_log_cat': oof_cat,
    'y_log_ens': oof_ensemble,
    'price_lgb': from_log1p(oof_lgb),
    'price_cat': from_log1p(oof_cat),
    'price_ens': from_log1p(oof_ensemble)
})
oof_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_oof_predictions_full_features_with_fi.csv"), index=False)
print("Saved OOF predictions ->", os.path.join(OUTPUT_DIR, "baseline_oof_predictions_full_features_with_fi.csv"))

# src/multimodal_train.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from metrics import smape, to_log1p, from_log1p
from target_cv import get_stratified_folds
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

# -------- CONFIG & PATHS --------
if Path("data/raw/train.csv").exists():
    DATA_DIR = Path("data/raw")
elif Path("dataset/raw/train.csv").exists():
    DATA_DIR = Path("dataset/raw")
else:
    raise FileNotFoundError("Can't find train.csv in data/raw or dataset/raw")

OUTPUT_DIR = Path("outputs") if Path("outputs").exists() else Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

FEATURES_FILE = OUTPUT_DIR / "multimodal_features.npy"
FEATURE_NAMES_FILE = OUTPUT_DIR / "multimodal_feature_names.npy"

OOF_OUT = OUTPUT_DIR / "multimodal_oof_predictions.csv"
TEST_RAW_OUT = OUTPUT_DIR / "multimodal_test_raw.csv"
FINAL_SUBMIT = OUTPUT_DIR / "test_out_full.csv"
FINAL_SUBMIT_CAL = OUTPUT_DIR / "test_out_full_calibrated.csv"

N_FOLDS = 5
SEED = 42

# -------- Load data & features --------
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

if not FEATURES_FILE.exists() or not FEATURE_NAMES_FILE.exists():
    raise FileNotFoundError("Run multimodal_features.py first to produce features and names.")

# load features as float32 (avoid float16 overflow)
features = np.load(FEATURES_FILE).astype(np.float32)
feat_names = np.load(FEATURE_NAMES_FILE).astype(str)

n_train = len(train_df)
n_test = len(test_df)

if features.shape[0] != n_train + n_test:
    raise ValueError(f"Feature rows ({features.shape[0]}) != train+test ({n_train+n_test})")

X_train = features[:n_train].astype(np.float32)
X_test = features[n_train:n_train + n_test].astype(np.float32)
y = train_df['price'].fillna(0).values.astype(np.float32)
y_log = to_log1p(y)

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

# -------- Clean non-finite & clip extremes (robust) --------
# Replace NaN/inf and clip to safe range
X_train[~np.isfinite(X_train)] = 0.0
X_test[~np.isfinite(X_test)] = 0.0

# Clip to a large but finite bound (adjust CLIP_VAL if needed)
CLIP_VAL = 1e6
np.clip(X_train, -CLIP_VAL, CLIP_VAL, out=X_train)
np.clip(X_test, -CLIP_VAL, CLIP_VAL, out=X_test)

# If any remaining non-finite values (unlikely), zero them
X_train[~np.isfinite(X_train)] = 0.0
X_test[~np.isfinite(X_test)] = 0.0

# -------- Manual column-wise scaling (stable, avoids sklearn dtype checks) --------
# Compute means/std on train only
col_means = np.mean(X_train, axis=0)
col_std = np.std(X_train, axis=0)
# avoid zero std
col_std[col_std == 0.0] = 1.0
# apply scaling (float32)
X_train = ((X_train - col_means) / col_std).astype(np.float32)
X_test = ((X_test - col_means) / col_std).astype(np.float32)

# Save scaler params for reproducibility
np.save(OUTPUT_DIR / "scaler_means.npy", col_means.astype(np.float32))
np.save(OUTPUT_DIR / "scaler_stds.npy", col_std.astype(np.float32))

# -------- CV folds (stratified on price bins preferred) --------
try:
    folds = get_stratified_folds(train_df, target_col='price', n_splits=N_FOLDS, random_state=SEED)
except Exception as e:
    print("get_stratified_folds failed, falling back to KFold:", e)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(X_train))

# -------- containers --------
oof_lgb = np.zeros(n_train, dtype=np.float32)
oof_cat = np.zeros(n_train, dtype=np.float32)
oof_ridge = np.zeros(n_train, dtype=np.float32)

test_lgb = np.zeros(n_test, dtype=np.float32)
test_cat = np.zeros(n_test, dtype=np.float32)
test_ridge = np.zeros(n_test, dtype=np.float32)

lgb_importances = np.zeros(X_train.shape[1], dtype=np.float64)
cat_importances = np.zeros(X_train.shape[1], dtype=np.float64)

# -------- Training per fold --------
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    print(f"\n=== Fold {fold_idx+1}/{len(folds)} ===")
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]

    # LightGBM
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 128,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'n_jobs': -1,
        'seed': SEED
    }
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    try:
        booster = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_val],
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=200)]
        )
        val_pred_l = booster.predict(X_val, num_iteration=booster.best_iteration)
        t_pred_l = booster.predict(X_test, num_iteration=booster.best_iteration)
        oof_lgb[val_idx] = val_pred_l
        test_lgb += t_pred_l / len(folds)
        try:
            lgb_importances += booster.feature_importance(importance_type='gain')
        except Exception:
            pass
    except Exception as e:
        print("LightGBM training failed on fold:", fold_idx+1, "error:", e)
        # keep zeros for this fold (will hurt ensemble but won't crash)

    # CatBoost
    try:
        cb = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.05,
            depth=8,
            eval_metric='RMSE',
            random_seed=SEED,
            verbose=200,
            early_stopping_rounds=50
        )
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        val_pred_c = cb.predict(X_val)
        t_pred_c = cb.predict(X_test)
        oof_cat[val_idx] = val_pred_c
        test_cat += t_pred_c / len(folds)
        try:
            cat_importances += cb.get_feature_importance()
        except Exception:
            pass
    except Exception as e:
        print("CatBoost training failed on fold:", fold_idx+1, "error:", e)

    # Ridge (linear on log-target)
    try:
        ridge = Ridge(alpha=10)
        ridge.fit(X_tr, y_tr)
        val_pred_r = ridge.predict(X_val)
        t_pred_r = ridge.predict(X_test)
        oof_ridge[val_idx] = val_pred_r
        test_ridge += t_pred_r / len(folds)
    except Exception as e:
        print("Ridge failed on fold:", fold_idx+1, "error:", e)

    # Per-fold SMAPE (de-transformed)
    # protect missing predictions by filling zeros -> small effect
    val_l = from_log1p(oof_lgb[val_idx]) if np.any(oof_lgb[val_idx]) else np.expm1(y_val)
    try:
        sm_l = smape(from_log1p(y_val), val_l)
    except Exception:
        sm_l = float('nan')
    val_c = from_log1p(oof_cat[val_idx]) if np.any(oof_cat[val_idx]) else np.expm1(y_val)
    try:
        sm_c = smape(from_log1p(y_val), val_c)
    except Exception:
        sm_c = float('nan')
    val_r = from_log1p(oof_ridge[val_idx]) if np.any(oof_ridge[val_idx]) else np.expm1(y_val)
    try:
        sm_r = smape(from_log1p(y_val), val_r)
    except Exception:
        sm_r = float('nan')
    print(f"fold {fold_idx+1} SMAPE - LGB: {sm_l:.4f}%, CAT: {sm_c:.4f}%, RIDGE: {sm_r:.4f}%")

# -------- Feature importances save --------
try:
    fi_df = pd.DataFrame({
        'feature': feat_names,
        'lgb_gain': (lgb_importances / len(folds)).tolist(),
        'cat_gain': (cat_importances / len(folds)).tolist()
    }).sort_values('lgb_gain', ascending=False)
    fi_df.to_csv(OUTPUT_DIR / "multimodal_feature_importances.csv", index=False)
    print("Saved feature importances.")
except Exception as e:
    print("Failed saving feature importances:", e)

# -------- Ensemble coarse grid search --------
best_smape = 1e9
best_weights = (0.33, 0.33, 0.34)
for w1 in [0.0, 0.25, 0.5, 0.75, 1.0]:
    for w2 in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue
        ensemble_oof = (w1 * np.expm1(oof_lgb) + w2 * np.expm1(oof_cat) + w3 * np.expm1(oof_ridge))
        s = smape(y, ensemble_oof)
        if s < best_smape:
            best_smape = s
            best_weights = (w1, w2, w3)
print("Best ensemble weights (w_lgb,w_cat,w_ridge):", best_weights, "OOF SMAPE:", best_smape)

# -------- Final OOF & test ensemble --------
oof_ens = best_weights[0] * np.expm1(oof_lgb) + best_weights[1] * np.expm1(oof_cat) + best_weights[2] * np.expm1(oof_ridge)
test_ens = best_weights[0] * np.expm1(test_lgb) + best_weights[1] * np.expm1(test_cat) + best_weights[2] * np.expm1(test_ridge)

# clip final preds to positive
oof_ens = np.clip(oof_ens, 0.01, None)
test_ens = np.clip(test_ens, 0.01, None)

print("Final OOF SMAPE (ensemble):", smape(y, oof_ens))

# -------- Calibration (Isotonic) --------
try:
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_ens, y)  # fit on OOF predictions -> actual prices
    oof_ens_cal = iso.transform(oof_ens)
    test_ens_cal = iso.transform(test_ens)
    oof_ens_cal = np.clip(oof_ens_cal, 0.01, None)
    test_ens_cal = np.clip(test_ens_cal, 0.01, None)
    print("OOF SMAPE after isotonic calibration:", smape(y, oof_ens_cal))
except Exception as e:
    print("Isotonic calibration failed:", e)
    oof_ens_cal = oof_ens.copy()
    test_ens_cal = test_ens.copy()

# -------- Save OOF details --------
oof_df = pd.DataFrame({
    'sample_id': train_df['sample_id'],
    'price_true': y,
    'price_ens': oof_ens,
    'price_ens_cal': oof_ens_cal
})
oof_df.to_csv(OOF_OUT, index=False)
print("Saved OOF ->", OOF_OUT)

# -------- Save test raw preds --------
test_df_out = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price_ens': test_ens,
    'price_ens_cal': test_ens_cal
})
test_df_out.to_csv(TEST_RAW_OUT, index=False)
print("Saved test raw preds ->", TEST_RAW_OUT)

# -------- Final submission files (full 75k) --------
sub = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': test_ens})
sub.to_csv(FINAL_SUBMIT, index=False)
print("Saved full submission ->", FINAL_SUBMIT)

sub_cal = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': test_ens_cal})
sub_cal.to_csv(FINAL_SUBMIT_CAL, index=False)
print("Saved full calibrated submission ->", FINAL_SUBMIT_CAL)

# optionally save models/weights
np.save(OUTPUT_DIR / "ensemble_weights.npy", np.array(best_weights))

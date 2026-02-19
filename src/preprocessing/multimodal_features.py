# scripts/multimodal_features.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import torch

# ---------- Directories ----------
DATA_DIR = Path("dataset/raw") if Path("dataset/raw/train.csv").exists() else Path("data/raw")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
NUMERIC_CSV = OUTPUT_DIR / "baseline_numeric_features.csv"
TEXT_FILE = OUTPUT_DIR / "baseline_embeddings.npy"
CLIP_FILE = OUTPUT_DIR / "clip_embeddings.npy"

OUT_FEATURES = OUTPUT_DIR / "multimodal_features.npy"
OUT_NAMES = OUTPUT_DIR / "multimodal_feature_names.npy"
OUT_MISSING = OUTPUT_DIR / "missing_clip_rows.txt"

CLIP_PCA_DIM = 128
TEXT_PCA_DIM = 64

# ---------- Load CSVs ----------
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
n_train, n_test = len(train_df), len(test_df)

# ---------- Numeric ----------
if NUMERIC_CSV.exists():
    num_df = pd.read_csv(NUMERIC_CSV)
    num_df_train = num_df.set_index('sample_id').reindex(train_df['sample_id']).reset_index(drop=True)
    X_num_train = num_df_train.fillna(0).values.astype(np.float32)
    X_num_test = np.zeros((n_test, X_num_train.shape[1]), dtype=np.float32)
    num_col_names = [f"num_{i}" for i in range(X_num_train.shape[1])]
else:
    X_num_train = np.zeros((n_train, 1), dtype=np.float32)
    X_num_test = np.zeros((n_test, 1), dtype=np.float32)
    num_col_names = ["num_0"]

# ---------- Text ----------
if TEXT_FILE.exists():
    X_text_all = np.load(TEXT_FILE).astype(np.float32)
    if X_text_all.shape[0] == n_train:
        X_text_train = X_text_all
        X_text_test = np.zeros((n_test, X_text_all.shape[1]), dtype=np.float32)
    else:
        X_text_train = X_text_all[:n_train]
        X_text_test = np.zeros((n_test, X_text_all.shape[1]), dtype=np.float32)
else:
    X_text_train = np.zeros((n_train, TEXT_PCA_DIM), dtype=np.float32)
    X_text_test = np.zeros((n_test, TEXT_PCA_DIM), dtype=np.float32)

if X_text_train.shape[1] > TEXT_PCA_DIM:
    pca = IncrementalPCA(n_components=TEXT_PCA_DIM, batch_size=1024)
    combined = np.vstack([X_text_train, X_text_test])
    X_text_reduced = pca.fit_transform(combined)
    X_text_train = X_text_reduced[:n_train]
    X_text_test = X_text_reduced[n_train:]

# ---------- CLIP ----------
clip = np.load(CLIP_FILE).astype(np.float32)
X_clip_train = np.zeros((n_train, clip.shape[1]), dtype=np.float32)
X_clip_test = np.zeros((n_test, clip.shape[1]), dtype=np.float32)
missing = []

preload_dir = Path("dataset/preloaded_images")
if preload_dir.exists():
    pt_map = {}
    for f in os.listdir(preload_dir):
        if f.endswith(".pt"):
            sid = int(Path(f).stem)
            tensor = torch.load(preload_dir / f, map_location="cpu")
            if isinstance(tensor, dict) and "data" in tensor:
                tensor = tensor["data"]
            pt_map[sid] = tensor.numpy().astype(np.float32) if hasattr(tensor, "numpy") else np.array(tensor, dtype=np.float32)
    for i, sid in enumerate(train_df["sample_id"]):
        if sid in pt_map:
            X_clip_train[i] = pt_map[sid]
        else:
            missing.append(("train", i, sid))
    for i, sid in enumerate(test_df["sample_id"]):
        if sid in pt_map:
            X_clip_test[i] = pt_map[sid]
        else:
            missing.append(("test", i, sid))
else:
    m = min(clip.shape[0], n_train)
    X_clip_train[:m] = clip[:m]
    n2 = min(clip.shape[0]-n_train, n_test)
    if n2>0:
        X_clip_test[:n2] = clip[n_train:n_train+n2]

if X_clip_train.shape[1] > CLIP_PCA_DIM:
    pca = IncrementalPCA(n_components=CLIP_PCA_DIM, batch_size=512)
    combined = np.vstack([X_clip_train, X_clip_test])
    X_clip_reduced = pca.fit_transform(combined)
    X_clip_train = X_clip_reduced[:n_train]
    X_clip_test = X_clip_reduced[n_train:]

# ---------- Concatenate ----------
X_train = np.hstack([X_num_train, X_text_train, X_clip_train])
X_test = np.hstack([X_num_test, X_text_test, X_clip_test])

# ---------- Scaling ----------
scaler = StandardScaler()
combined = np.vstack([X_train, X_test])
combined_scaled = scaler.fit_transform(combined)
X_train = combined_scaled[:n_train]
X_test = combined_scaled[n_train:]

# ---------- Save ----------
feat_names = num_col_names + [f"text_{i}" for i in range(X_text_train.shape[1])] + [f"clip_{i}" for i in range(X_clip_train.shape[1])]
np.save(OUT_FEATURES, np.vstack([X_train, X_test]))
np.save(OUT_NAMES, np.array(feat_names))

if missing:
    with open(OUT_MISSING, "w") as f:
        for rec in missing:
            f.write(",".join(map(str, rec)) + "\n")

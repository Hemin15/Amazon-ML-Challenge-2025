# baseline_features.py
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
from utils import safe_read_csv, ensure_dirs

# ----------------------
# Paths
# ----------------------
ROOT = Path(r"D:\Amazon")
RAW = ROOT / "dataset/raw/train.csv"
OUTPUT_DIR = ROOT / "output"
ensure_dirs(OUTPUT_DIR)

# ----------------------
# Load data
# ----------------------
df = safe_read_csv(str(RAW))

# ----------------------
# Step 3.1: Numeric / tabular features
# ----------------------
df['text_len'] = df['catalog_content'].astype(str).apply(len)
df['num_words'] = df['catalog_content'].astype(str).apply(lambda x: len(x.split()))
df['num_digits'] = df['catalog_content'].astype(str).apply(lambda x: sum(c.isdigit() for c in x))

# Extract item pack quantity (ipq) using regex like "Pack of 3", "3 pcs", "6x"
def extract_ipq(text):
    if pd.isna(text):
        return np.nan
    m = re.search(r'(\d+)\s*(pcs|pack|x|units?)', text.lower())
    return int(m.group(1)) if m else np.nan

df['ipq'] = df['catalog_content'].astype(str).apply(extract_ipq)

# first token as candidate brand
df['first_token'] = df['catalog_content'].astype(str).apply(lambda x: x.split()[0] if len(x.split())>0 else '')

# Numeric specs: weight (g, kg, oz), volume (ml, l, fl oz)
def extract_weight(text):
    if pd.isna(text):
        return np.nan
    m = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g|oz)', text.lower())
    if not m:
        return np.nan
    val, unit = float(m.group(1)), m.group(2)
    if unit == 'kg':
        val *= 1000
    elif unit == 'oz':
        val *= 28.3495
    return val

def extract_volume(text):
    if pd.isna(text):
        return np.nan
    m = re.search(r'(\d+(?:\.\d+)?)\s*(ml|l|fl oz)', text.lower())
    if not m:
        return np.nan
    val, unit = float(m.group(1)), m.group(2)
    if unit == 'l':
        val *= 1000
    elif unit == 'fl oz':
        val *= 29.5735
    return val

df['weight_g'] = df['catalog_content'].astype(str).apply(extract_weight)
df['volume_ml'] = df['catalog_content'].astype(str).apply(extract_volume)

# Save numeric features
num_features = ['text_len','num_words','num_digits','ipq','weight_g','volume_ml']
df[num_features + ['sample_id']].to_csv(OUTPUT_DIR / "baseline_numeric_features.csv", index=False)
print("Saved numeric features ->", OUTPUT_DIR / "baseline_numeric_features.csv")

# ----------------------
# Step 3.2: TF-IDF text embeddings
# ----------------------
texts = df['catalog_content'].astype(str).fillna('')

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    analyzer='word',
)
X_tfidf = tfidf.fit_transform(texts)

# Dimensionality reduction
svd = TruncatedSVD(n_components=128, random_state=42)
X_emb = svd.fit_transform(X_tfidf)

# Save embeddings + vectorizer + SVD
np.save(OUTPUT_DIR / "baseline_embeddings.npy", X_emb)
joblib.dump(tfidf, OUTPUT_DIR / "baseline_tfidf.pkl")
joblib.dump(svd, OUTPUT_DIR / "baseline_svd.pkl")
print("Saved TF-IDF embeddings ->", OUTPUT_DIR / "baseline_embeddings.npy")

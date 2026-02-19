# eda.py
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from utils import set_seed, safe_read_csv, ensure_dirs

set_seed(42)

# Paths (edit if you use different)
ROOT = Path(r"D:\Amazon")
RAW = ROOT / "dataset" / "raw" / "train.csv"
TEST_RAW = ROOT / "dataset" / "raw" / "test.csv"
IMAGES_DIR = ROOT / "dataset" / "images" / "train"
OUTPUT_DIR = ROOT / "output"

ensure_dirs(OUTPUT_DIR)

print("Loading CSV:", RAW)
df = safe_read_csv(str(RAW))
print("Loaded rows:", len(df))

# Basic stats
report = {}
report['rows'] = int(len(df))
report['columns'] = df.columns.tolist()

# Price distribution (if present)
if 'price' in df.columns:
    prices = pd.to_numeric(df['price'], errors='coerce')
    report['price_count'] = int(prices.notna().sum())
    report['price_min'] = float(prices.min())
    report['price_median'] = float(prices.median())
    report['price_max'] = float(prices.max())
    report['price_mean'] = float(prices.mean())
    report['price_std'] = float(prices.std())
    report['price_skew'] = float(prices.skew())

# Missing counts
report['missing_counts'] = df.isna().sum().to_dict()

# catalog_content checks
if 'catalog_content' in df.columns:
    report['catalog_content_missing'] = int(df['catalog_content'].isna().sum())
    # very short entries
    df['catalog_len'] = df['catalog_content'].astype(str).str.len()
    report['catalog_len_min'] = int(df['catalog_len'].min())
    report['catalog_len_median'] = int(df['catalog_len'].median())

# image_link checks
if 'image_link' in df.columns:
    report['image_link_missing'] = int(df['image_link'].isna().sum())
    # extract basename for comparison with local files
    df['image_file'] = df['image_link'].astype(str).apply(lambda x: Path(x).name if pd.notna(x) else x)

# Duplicates
report['duplicate_rows'] = int(df.duplicated(keep=False).sum())
report['duplicate_sample_id'] = 0
if 'sample_id' in df.columns:
    report['duplicate_sample_id'] = int(df['sample_id'].duplicated().sum())

# Local images check
local_files = []
if IMAGES_DIR.exists():
    for root, _, files in os.walk(IMAGES_DIR):
        for f in files:
            local_files.append(f)
local_files_set = set(local_files)
report['local_image_count_walk'] = len(local_files)
# how many csv filenames match local files
if 'image_file' in df.columns:
    df['exists_local_exact'] = df['image_file'].apply(lambda x: bool(x in local_files_set))
    report['csv_image_unique_filenames'] = int(df['image_file'].nunique(dropna=True))
    report['csv_rows_with_local_file'] = int(df['exists_local_exact'].sum())
    report['csv_rows_missing_local_file'] = int(len(df) - report['csv_rows_with_local_file'])
else:
    df['exists_local_exact'] = False

# Save lists: missing rows and duplicates and sample of issues
missing_local = df[~df['exists_local_exact']].copy()
missing_local[['sample_id','image_link','image_file']].to_csv(OUTPUT_DIR / "missing_links.csv", index=False)

# Duplicates in CSV (same image filename used in multiple rows)
if 'image_file' in df.columns:
    dup_image_usage = df.groupby('image_file').size().reset_index(name='count').query("count>1")
    dup_image_usage.to_csv(OUTPUT_DIR / "duplicates_in_csv.csv", index=False)
else:
    pd.DataFrame().to_csv(OUTPUT_DIR / "duplicates_in_csv.csv", index=False)

# Save report
with open(OUTPUT_DIR / "eda_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("EDA done.")
print(json.dumps(report, indent=2))
print("Saved missing rows ->", OUTPUT_DIR / "missing_links.csv")
print("Saved duplicates ->", OUTPUT_DIR / "duplicates_in_csv.csv")
print("Saved report ->", OUTPUT_DIR / "eda_report.json")

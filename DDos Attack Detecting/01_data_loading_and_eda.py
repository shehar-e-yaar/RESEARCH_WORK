# %% [markdown]
# # Reproducing: DDoS Attacks Detection with ML, Kafka & Spark
# ## Part 1: Fast Data Loading, Cleaning, and Exploratory Data Analysis
#
# **Paper:** J. Ahmad et al., IEEE Access, vol. 14, 2026
# **Dataset:** CICDDoS2019 (Sharafaldin et al., ICCST 2019)

# %% Imports and Configuration
import os
import gc
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)

# Ensure console handles output cleanly
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ──
BASE_DIR = r'c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting'
DATA_DIR = os.path.join(BASE_DIR, 'DataSets')
TRAIN_DAY_DIR = os.path.join(DATA_DIR, '01-12')   # Training day: Jan 12 (11 CSVs)
TEST_DAY_DIR  = os.path.join(DATA_DIR, '03-11')    # Test day: Mar 11 (7 CSVs)
OUTPUT_DIR    = os.path.join(BASE_DIR, 'processed_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Paper's 20 Selected Features (Table 2) ──
PAPER_20_FEATURES = [
    'Flow Duration',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Packet Length Mean',
    'Packet Length Std',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Destination Port',
    'ACK Flag Count',
]

RANDOM_STATE = 42
TARGET_ATTACK_PER_FILE = 25_000  # Sample up to 25K attack flows per file to preserve memory & balance

# %% [markdown]
# ## 1. Optimized Fast CSV Loader
# Reads only required columns, preserves ALL rare BENIGN flows, and samples attacks.

# %%
def load_csv_fast(filepath, target_attacks=25_000, random_state=42):
    """
    Optimized loader for massive CICDDoS2019 files:
    1. Uses usecols to skip unneeded columns (5-10x speedup)
    2. Collects ALL rare BENIGN samples
    3. Randomly samples attack flows up to target_attacks
    4. Handles Infinity, -Infinity, and NaNs
    """
    fname = os.path.basename(filepath)
    t0 = time.time()
    
    # Read header to map column names
    header = pd.read_csv(filepath, nrows=1)
    raw_to_clean = {c: c.strip() for c in header.columns}
    clean_to_raw = {c.strip(): c for c in header.columns}
    
    # Identify raw column names matching paper features + Label
    cols_to_read = [clean_to_raw[f] for f in PAPER_20_FEATURES if f in clean_to_raw]
    label_raw = clean_to_raw['Label']
    cols_to_read.append(label_raw)
    
    benign_chunks = []
    attack_chunks = []
    total_rows = 0
    
    # Process in 150,000-row chunks
    for chunk in pd.read_csv(filepath, usecols=cols_to_read, chunksize=150_000, low_memory=False):
        total_rows += len(chunk)
        chunk.rename(columns=raw_to_clean, inplace=True)
        chunk['Label'] = chunk['Label'].astype(str).str.strip()
        
        # Split benign and attack
        b = chunk[chunk['Label'].str.upper() == 'BENIGN']
        a = chunk[chunk['Label'].str.upper() != 'BENIGN']
        
        if len(b) > 0:
            benign_chunks.append(b)
        if len(a) > 0:
            # Subsample attack rows per chunk
            n_sample = min(len(a), 3_000)
            attack_chunks.append(a.sample(n=n_sample, random_state=random_state))
    
    df_b = pd.concat(benign_chunks, ignore_index=True) if benign_chunks else pd.DataFrame()
    df_a = pd.concat(attack_chunks, ignore_index=True) if attack_chunks else pd.DataFrame()
    
    # Limit attack rows to target_attacks
    if len(df_a) > target_attacks:
        df_a = df_a.sample(n=target_attacks, random_state=random_state)
        
    df = pd.concat([df_b, df_a], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # ── Numeric conversions & inf cleaning ──
    for col in PAPER_20_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    elapsed = time.time() - t0
    n_b = len(df_b)
    n_a = len(df_a)
    print(f"  {fname:<20} | Total: {total_rows:>10,} -> Sampled: {len(df):>6,} (Benign: {n_b:>5,}, Attack: {n_a:>5,}) [{elapsed:.1f}s]")
    return df

# %% [markdown]
# ## 2. Load Training Day (01-12) & Test Day (03-11)

# %%
print("=" * 80)
print("LOADING TRAINING DAY (01-12) - 11 ATTACK FILES")
print("=" * 80)

train_dfs = []
for fname in sorted(os.listdir(TRAIN_DAY_DIR)):
    if fname.endswith('.csv'):
        fpath = os.path.join(TRAIN_DAY_DIR, fname)
        df = load_csv_fast(fpath, target_attacks=TARGET_ATTACK_PER_FILE, random_state=RANDOM_STATE)
        df['source_file'] = fname
        df['day'] = 'train_01-12'
        train_dfs.append(df)

train_df = pd.concat(train_dfs, ignore_index=True)
del train_dfs
gc.collect()

print(f"\nTraining Day Total: {len(train_df):,} flows")
print(f"Training Day Class Counts:\n{train_df['Label'].value_counts()}")

# %%
print("\n" + "=" * 80)
print("LOADING TEST DAY (03-11) - 7 ATTACK FILES")
print("=" * 80)

test_dfs = []
for fname in sorted(os.listdir(TEST_DAY_DIR)):
    if fname.endswith('.csv'):
        fpath = os.path.join(TEST_DAY_DIR, fname)
        df = load_csv_fast(fpath, target_attacks=TARGET_ATTACK_PER_FILE, random_state=RANDOM_STATE)
        df['source_file'] = fname
        df['day'] = 'test_03-11'
        test_dfs.append(df)

test_df = pd.concat(test_dfs, ignore_index=True)
del test_dfs
gc.collect()

print(f"\nTest Day Total: {len(test_df):,} flows")
print(f"Test Day Class Counts:\n{test_df['Label'].value_counts()}")

# %% [markdown]
# ## 3. Binary Label Encoding & Quality Checks

# %%
def prepare_binary_dataset(df):
    df = df.copy()
    df['binary_label'] = (df['Label'].str.upper() != 'BENIGN').astype(int)
    return df

train_df = prepare_binary_dataset(train_df)
test_df  = prepare_binary_dataset(test_df)

print("\n" + "=" * 80)
print("DATASET SANITY CHECKS")
print("=" * 80)

# Check missing values
train_nan = train_df[PAPER_20_FEATURES].isnull().sum()
test_nan  = test_df[PAPER_20_FEATURES].isnull().sum()
print("NaN counts in Training features:")
print(train_nan[train_nan > 0])
print("NaN counts in Test features:")
print(test_nan[test_nan > 0])

# Check constant columns
train_constants = [col for col in PAPER_20_FEATURES if train_df[col].nunique(dropna=True) <= 1]
print(f"Constant features in Train: {train_constants}")

# Check class distributions
print("\nClass Distribution:")
print(f"Train - BENIGN: {(train_df['binary_label']==0).sum():,} ({100*(train_df['binary_label']==0).mean():.1f}%), Attack: {(train_df['binary_label']==1).sum():,}")
print(f"Test  - BENIGN: {(test_df['binary_label']==0).sum():,} ({100*(test_df['binary_label']==0).mean():.1f}%), Attack: {(test_df['binary_label']==1).sum():,}")

# %% [markdown]
# ## 4. Save Clean Parquet Files

# %%
print("\n" + "=" * 80)
print("SAVING PROCESSED PARQUET DATA")
print("=" * 80)

# Convert string columns explicitly
for col in ['Label', 'source_file', 'day']:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

train_out_path = os.path.join(OUTPUT_DIR, 'train_day_sampled.parquet')
test_out_path  = os.path.join(OUTPUT_DIR, 'test_day_sampled.parquet')

train_df.to_parquet(train_out_path, index=False)
test_df.to_parquet(test_out_path, index=False)

# Save feature metadata
meta = {
    'features': PAPER_20_FEATURES,
    'n_train': len(train_df),
    'n_test': len(test_df),
    'train_benign': int((train_df['binary_label']==0).sum()),
    'train_attack': int((train_df['binary_label']==1).sum()),
    'test_benign': int((test_df['binary_label']==0).sum()),
    'test_attack': int((test_df['binary_label']==1).sum()),
    'train_classes': {str(k): int(v) for k, v in train_df['Label'].value_counts().items()},
    'test_classes': {str(k): int(v) for k, v in test_df['Label'].value_counts().items()},
}

with open(os.path.join(OUTPUT_DIR, 'feature_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"[OK] Saved {len(train_df):,} train records to: {train_out_path}")
print(f"[OK] Saved {len(test_df):,} test records to: {test_out_path}")
print(f"[OK] Saved metadata to: {os.path.join(OUTPUT_DIR, 'feature_metadata.json')}")

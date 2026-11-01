"""
Deduplication Experiment on CIC-DDoS2019
Objective:
    Test what happens to accuracy when duplicate records are completely removed:
    Case 1: All features (with drop_duplicates)
    Case 2: Paper's Table 3 features (with drop_duplicates)
"""

import os
import glob
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROCESSED_DATA_DIR = r"data\processed"

PAPER_FEATURES = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 'Packet Length Mean',
    'Packet Length Std', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Destination Port', 'ACK Flag Count'
]

def collect_deduplicated_data(pattern, target_per_class=10000, max_chunks=5):
    files = glob.glob(os.path.join(PROCESSED_DATA_DIR, f"*{pattern}*_cleaned.csv"))
    if not files:
        files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*_cleaned.csv"))
        
    benign_list = []
    attack_list = []
    total_raw_rows = 0
    
    print(f"\n[+] Collecting and deduplicating data for pattern '{pattern}'...")
    
    for f in files:
        filename = os.path.basename(f)
        chunks_read = 0
        for chunk in pd.read_csv(f, chunksize=50000, low_memory=False):
            total_raw_rows += len(chunk)
            
            # Separate before deduplicating
            b_chunk = chunk[chunk['Label'] == 0]
            a_chunk = chunk[chunk['Label'] == 1]
            
            if len(b_chunk) > 0:
                benign_list.append(b_chunk)
            if len(a_chunk) > 0:
                attack_list.append(a_chunk)
                
            chunks_read += 1
            if chunks_read >= max_chunks:
                break

    all_b = pd.concat(benign_list, ignore_index=True)
    all_a = pd.concat(attack_list, ignore_index=True)
    
    # DEDUPLICATION STEP: Drop duplicates across all feature columns
    feature_cols = [c for c in all_b.columns if c != 'Label']
    b_dedup = all_b.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    a_dedup = all_a.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    
    print(f"  Raw Benign: {len(all_b):,} -> Unique Benign: {len(b_dedup):,} (Dropped {len(all_b)-len(b_dedup):,} duplicates)")
    print(f"  Raw Attack: {len(all_a):,} -> Unique Attack: {len(a_dedup):,} (Dropped {len(all_a)-len(a_dedup):,} duplicates)")
    
    # Balance 50/50
    n = min(len(b_dedup), len(a_dedup), target_per_class)
    b_sample = b_dedup.sample(n=n, random_state=42)
    a_sample = a_dedup.sample(n=n, random_state=42)
    
    final_df = pd.concat([b_sample, a_sample], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"  Final 50/50 Balanced & Deduplicated Pool: {len(final_df):,} rows (Normal: {n:,}, Attack: {n:,})")
    return final_df

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  FPR:       {fpr * 100:.2f}%")
    print(f"  Matrix: [TN={tn}, FP={fp}] | [FN={fn}, TP={tp}]")
    return acc

def run_experiment(features_subset=None, exp_name="All Features"):
    print("\n" + "="*65)
    print(f" EXPERIMENT: {exp_name.upper()} (DEDUPLICATED 50/50)")
    print("="*65)
    
    # 1. Collect
    train_df = collect_deduplicated_data("01-12", target_per_class=10000, max_chunks=4)
    test_df = collect_deduplicated_data("03-11", target_per_class=4000, max_chunks=4)
    
    # 2. Select Features
    if features_subset:
        cols = [c for c in features_subset if c in train_df.columns]
        X_train = train_df[cols]
        X_test = test_df[cols]
    else:
        X_train = train_df.drop(columns=['Label'])
        X_test = test_df.drop(columns=['Label'])
        
    y_train = train_df['Label'].values
    y_test = test_df['Label'].values
    
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    print(f"\nFeatures used: {X_train.shape[1]}")
    
    # 3. Scaling
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    
    # 4. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_train)
    rf_preds = rf.predict(X_te_s)
    evaluate(f"Random Forest ({exp_name})", y_test, rf_preds)
    
    # 5. XGBoost
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_tr_s, y_train)
    xgb_preds = xgb.predict(X_te_s)
    evaluate(f"XGBoost ({exp_name})", y_test, xgb_preds)

def main():
    # Test 1: All features (deduplicated)
    run_experiment(features_subset=None, exp_name="All 80 Features")
    
    # Test 2: Paper Table 3 features (deduplicated)
    run_experiment(features_subset=PAPER_FEATURES, exp_name="Paper Table 3 (20 Features)")

if __name__ == "__main__":
    main()

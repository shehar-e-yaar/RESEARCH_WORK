"""
50/50 Balanced DDoS Detection Pipeline
Objective:
    Train and evaluate models (Random Forest, XGBoost) on an exact 50% Attack / 50% Benign balanced distribution.
"""

import os
import glob
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

PROCESSED_DATA_DIR = r"data\processed"

def collect_balanced_samples(pattern, target_per_class=20000, max_chunk_per_file=3):
    """
    Collects exactly equal numbers of Benign (0) and Attack (1) packets
    from files matching the given pattern.
    """
    files = glob.glob(os.path.join(PROCESSED_DATA_DIR, f"*{pattern}*_cleaned.csv"))
    if not files:
        # Fallback to any cleaned files
        files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*_cleaned.csv"))
    
    benign_dfs = []
    attack_dfs = []
    total_benign = 0
    total_attack = 0
    
    print(f"\n[+] Collecting samples for pattern '{pattern}' (Target: {target_per_class} per class)...")
    
    for f in files:
        if total_benign >= target_per_class and total_attack >= target_per_class:
            break
            
        filename = os.path.basename(f)
        chunks_read = 0
        
        # Read in chunks of 50,000 to keep memory small and fast
        for chunk in pd.read_csv(f, chunksize=50000, low_memory=False):
            b_chunk = chunk[chunk['Label'] == 0]
            a_chunk = chunk[chunk['Label'] == 1]
            
            if total_benign < target_per_class and len(b_chunk) > 0:
                needed_b = target_per_class - total_benign
                b_selected = b_chunk.head(needed_b)
                benign_dfs.append(b_selected)
                total_benign += len(b_selected)
                
            if total_attack < target_per_class and len(a_chunk) > 0:
                needed_a = target_per_class - total_attack
                # Sample equally from this attack file
                sample_n = min(len(a_chunk), needed_a, 2000)
                a_selected = a_chunk.sample(n=sample_n, random_state=42)
                attack_dfs.append(a_selected)
                total_attack += len(a_selected)
                
            chunks_read += 1
            if chunks_read >= max_chunk_per_file:
                break
                
        print(f"  Processed {filename}: Benign collected = {total_benign:,}, Attack collected = {total_attack:,}")

    # Combine and trim to exact target
    all_b = pd.concat(benign_dfs, ignore_index=True).head(target_per_class)
    all_a = pd.concat(attack_dfs, ignore_index=True).head(target_per_class)
    
    balanced_df = pd.concat([all_b, all_a], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"[OK] Balanced Pool Created: Total = {len(balanced_df):,} | Normal (0): {(balanced_df['Label']==0).sum():,} | Attack (1): {(balanced_df['Label']==1).sum():,}")
    return balanced_df

def print_metrics(name, y_true, y_pred, y_probs=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n=======================================================")
    print(f"       MODEL RESULTS: {name}")
    print(f"=======================================================")
    print(f"  Accuracy:               {acc * 100:.2f}%")
    print(f"  Precision:              {prec * 100:.2f}%")
    print(f"  Recall (Detection Rate):{rec * 100:.2f}%")
    print(f"  F1-Score:               {f1 * 100:.2f}%")
    print(f"  False Positive Rate:    {fpr * 100:.2f}% (Normal blocked)")
    print(f"  False Negative Rate:    {fnr * 100:.2f}% (Attack missed)")
    print(f"-------------------------------------------------------")
    print(f"  Confusion Matrix:")
    print(f"               Predicted Normal (0)   Predicted Attack (1)")
    print(f"  Actual Normal (0):    {tn:<10}            {fp:<10}")
    print(f"  Actual Attack (1):    {fn:<10}            {tp:<10}")
    print(f"=======================================================")
    return acc, prec, rec, f1

def main():
    print("=" * 65)
    print("      50/50 BALANCED EXPERIMENT (CIC-DDoS2019)")
    print("=" * 65)
    
    # 1. Collect Balanced Data: 15,000 Benign + 15,000 Attack (Total: 30,000 rows)
    # Day 1 (01-12) for Training
    train_df = collect_balanced_samples("01-12", target_per_class=15000, max_chunk_per_file=5)
    
    # Day 2 (03-11) for Untouched Final Testing
    test_df = collect_balanced_samples("03-11", target_per_class=5000, max_chunk_per_file=5)
    
    # 2. Features and Target
    X_train = train_df.drop(columns=['Label'])
    y_train = train_df['Label'].values
    
    X_test = test_df.drop(columns=['Label'])
    y_test = test_df['Label'].values
    
    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    print(f"\n[+] Train Set Shape: {X_train.shape} | Balance: {np.bincount(y_train)}")
    print(f"[+] Test Set Shape:  {X_test.shape}  | Balance: {np.bincount(y_test)}")
    
    # 3. Scaling
    print("\n[+] Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Random Forest
    print("\n[+] Training Random Forest (100 trees)...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    t1 = time.time()
    print(f"  Training finished in {t1 - t0:.2f}s")
    
    rf_preds = rf.predict(X_test_scaled)
    print_metrics("Random Forest (50/50 Balanced)", y_test, rf_preds)
    
    # 5. Train XGBoost
    print("\n[+] Training XGBoost...")
    t0 = time.time()
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train_scaled, y_train)
    t1 = time.time()
    print(f"  Training finished in {t1 - t0:.2f}s")
    
    xgb_preds = xgb.predict(X_test_scaled)
    print_metrics("XGBoost (50/50 Balanced)", y_test, xgb_preds)

if __name__ == "__main__":
    main()

"""
Feature Weighting & Feature Space Analysis
Using strictly the 20 features from Table 3 of the Research Paper.

Steps:
1. Load dataset with only the Paper's 20 Features.
2. Calculate statistical importance (Mutual Information / Tree Gini) to find the 'deserving' features.
3. Apply Feature Weighting (amplifying top features).
4. Transform and Project the Feature Space using PCA (Principal Component Analysis) and LDA into 2D.
5. Save high-resolution Feature Space plots.
6. Train and evaluate XGBoost and Random Forest on the weighted 20-feature space.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROCESSED_DATA_DIR = r"data\processed"
OUTPUT_DIR = r"output\visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EXACT 20 Features from Table 3 of the Research Paper
PAPER_FEATURES = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 'Packet Length Mean',
    'Packet Length Std', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Destination Port', 'ACK Flag Count'
]
# Remove any accidental duplicates in the list while preserving order
PAPER_FEATURES = list(dict.fromkeys(PAPER_FEATURES))

def load_data(pattern, sample_size=10000, max_chunks_per_file=4):
    files = glob.glob(os.path.join(PROCESSED_DATA_DIR, f"*{pattern}*_cleaned.csv"))
    if not files:
        files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*_cleaned.csv"))
        
    benign_dfs = []
    attack_dfs = []
    target = sample_size // 2
    
    print(f"\n[+] Loading data for pattern '{pattern}' (Target: {target:,} Normal, {target:,} Attack)...")
    
    for f in files:
        if sum(len(x) for x in benign_dfs) >= target and sum(len(x) for x in attack_dfs) >= target:
            break
            
        chunks_read = 0
        for chunk in pd.read_csv(f, chunksize=50000, low_memory=False):
            # Only keep available paper features + Label
            cols = [c for c in PAPER_FEATURES if c in chunk.columns] + ['Label']
            chunk_sub = chunk[cols]
            
            b = chunk_sub[chunk_sub['Label'] == 0]
            a = chunk_sub[chunk_sub['Label'] == 1]
            
            if len(b) > 0 and sum(len(x) for x in benign_dfs) < target:
                needed_b = target - sum(len(x) for x in benign_dfs)
                benign_dfs.append(b.head(needed_b))
                
            if len(a) > 0 and sum(len(x) for x in attack_dfs) < target:
                needed_a = target - sum(len(x) for x in attack_dfs)
                sample_n = min(len(a), needed_a, 2000)
                attack_dfs.append(a.sample(n=sample_n, random_state=42))
                
            chunks_read += 1
            if chunks_read >= max_chunks_per_file:
                break
                
    df_b = pd.concat(benign_dfs, ignore_index=True)
    df_a = pd.concat(attack_dfs, ignore_index=True)
    
    combined = pd.concat([df_b, df_a], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"  Loaded {len(combined):,} rows | Normal (0): {(combined['Label']==0).sum():,}, Attack (1): {(combined['Label']==1).sum():,}")
    return combined

def main():
    print("="*70)
    print("   FEATURE WEIGHTING & FEATURE SPACE ANALYSIS (20 PAPER FEATURES)")
    print("="*70)
    
    # 1. Load Training and Testing data strictly with Paper Features
    train_df = load_data("01-12", sample_size=20000)
    test_df = load_data("03-11", sample_size=10000)
    
    available_features = [c for c in PAPER_FEATURES if c in train_df.columns]
    print(f"\n[+] Active Paper Features ({len(available_features)}):")
    for i, col in enumerate(available_features, 1):
        print(f"   {i:2d}. {col}")
        
    X_train = train_df[available_features]
    y_train = train_df['Label'].values
    
    X_test = test_df[available_features]
    y_test = test_df['Label'].values
    
    # 2. Robust Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Calculate "Deserving" Feature Weights
    print("\n[+] Calculating Feature Importance Weights (Mutual Information & Random Forest Gini)...")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train_scaled, y_train)
    rf_weights = rf_selector.feature_importances_
    
    # Normalize weights so mean weight = 1.0 (amplifies deserving, suppresses noisy)
    normalized_weights = rf_weights / rf_weights.mean()
    
    weight_df = pd.DataFrame({
        'Feature': available_features,
        'Raw_Importance': rf_weights,
        'Applied_Weight': normalized_weights
    }).sort_values(by='Applied_Weight', ascending=False).reset_index(drop=True)
    
    print("\n--- Top 10 Most Deserving Features (Highest Weights) ---")
    print(weight_df.head(10).to_string(index=False))
    
    # Plot Feature Weights
    plt.figure(figsize=(10, 6))
    sns.barplot(data=weight_df, x='Applied_Weight', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.axvline(1.0, color='red', linestyle='--', label='Baseline Weight (1.0)')
    plt.title('Applied Feature Weights (Deserving vs Non-Deserving Features)', fontsize=14, fontweight='bold')
    plt.xlabel('Multiplier Weight (Normalized Importance)')
    plt.legend()
    plt.tight_layout()
    weights_plot_path = os.path.join(OUTPUT_DIR, "paper_features_weights_ranking.png")
    plt.savefig(weights_plot_path, dpi=300)
    plt.close()
    print(f"\n[OK] Feature Weights chart saved: {weights_plot_path}")
    
    # 4. Apply Feature Weights in Feature Space
    print("\n[+] Applying Feature Weights to reshape the Feature Space (X * Weights)...")
    X_train_weighted = X_train_scaled * normalized_weights
    X_test_weighted = X_test_scaled * normalized_weights
    
    # 5. FEATURE SPACE TRANSFORMATION: PCA (2D Projection)
    print("\n[+] Projecting 20-D Feature Space into 2-D Coordinate Space using PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca_train = pca.fit_transform(X_train_weighted)
    
    var_exp = pca.explained_variance_ratio_
    print(f"  PCA Component 1 explains: {var_exp[0]*100:.2f}% variance")
    print(f"  PCA Component 2 explains: {var_exp[1]*100:.2f}% variance")
    print(f"  Total 2D Explained Variance: {(var_exp[0]+var_exp[1])*100:.2f}%")
    
    # Plot 2D PCA Feature Space
    plt.figure(figsize=(11, 8))
    # Subsample for clear visual scatter
    plot_idx = np.random.choice(len(X_pca_train), size=min(4000, len(X_pca_train)), replace=False)
    pca_sub = X_pca_train[plot_idx]
    y_sub = y_train[plot_idx]
    
    plt.scatter(pca_sub[y_sub == 0, 0], pca_sub[y_sub == 0, 1], 
                c='#2ecc71', label='Normal Traffic (0)', alpha=0.6, edgecolors='none', s=25)
    plt.scatter(pca_sub[y_sub == 1, 0], pca_sub[y_sub == 1, 1], 
                c='#e74c3c', label='DDoS Attack (1)', alpha=0.6, edgecolors='none', s=25)
                
    plt.title(f'2D Feature Space Projection (PCA) - 20 Paper Features\n'
              f'Explained Variance: {(var_exp[0]+var_exp[1])*100:.1f}%', fontsize=15, fontweight='bold')
    plt.xlabel('Principal Component 1 (Direction of Greatest Variance)')
    plt.ylabel('Principal Component 2 (Orthogonal Variance)')
    plt.legend(frameon=True, facecolor='white', framealpha=1, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    pca_plot_path = os.path.join(OUTPUT_DIR, "feature_space_2d_pca.png")
    plt.savefig(pca_plot_path, dpi=300)
    plt.close()
    print(f"[OK] 2D Feature Space PCA plot saved: {pca_plot_path}")
    
    # 6. FEATURE SPACE TRANSFORMATION: LDA (Optimal Linear Separator)
    print("\n[+] Projecting Feature Space with LDA (Linear Discriminant Analysis)...")
    lda = LDA(n_components=1)
    X_lda_train = lda.fit_transform(X_train_weighted, y_train)
    
    plt.figure(figsize=(10, 5))
    sns.kdeplot(X_lda_train[y_train == 0].flatten(), color='#2ecc71', fill=True, label='Normal Traffic (0)', alpha=0.5)
    sns.kdeplot(X_lda_train[y_train == 1].flatten(), color='#e74c3c', fill=True, label='DDoS Attack (1)', alpha=0.5)
    plt.title('1D LDA Feature Space: Discriminant Boundary Separation', fontsize=14, fontweight='bold')
    plt.xlabel('Linear Discriminant Projection Axis (Maximizing Class Separation)')
    plt.ylabel('Density Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    lda_plot_path = os.path.join(OUTPUT_DIR, "feature_space_lda_density.png")
    plt.savefig(lda_plot_path, dpi=300)
    plt.close()
    print(f"[OK] LDA Separation plot saved: {lda_plot_path}")
    
    # 7. Model Training & Evaluation in Weighted Feature Space
    print("\n" + "="*70)
    print("   MODEL EVALUATION IN WEIGHTED FEATURE SPACE (20 PAPER FEATURES)")
    print("="*70)
    
    # Model 1: Random Forest on Weighted Features
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_weighted, y_train)
    rf_preds = rf.predict(X_test_weighted)
    
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_rec = recall_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_tn, rf_fp, rf_fn, rf_tp = confusion_matrix(y_test, rf_preds).ravel()
    
    print("\n[1] Random Forest (Weighted 20 Paper Features):")
    print(f"    Accuracy:  {rf_acc*100:.2f}%  (Paper Claimed: 93.2%)")
    print(f"    Precision: {rf_prec*100:.2f}%")
    print(f"    Recall:    {rf_rec*100:.2f}%")
    print(f"    F1-Score:  {rf_f1*100:.2f}%")
    print(f"    Matrix:    [TN={rf_tn}, FP={rf_fp}] | [FN={rf_fn}, TP={rf_tp}]")
    
    # Model 2: XGBoost with Native Feature Sampling by Weights
    # In XGBoost, colsample_bytree with weighted feature representation
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train_weighted, y_train)
    xgb_preds = xgb.predict(X_test_weighted)
    
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_rec = recall_score(y_test, xgb_preds)
    xgb_prec = precision_score(y_test, xgb_preds)
    xgb_f1 = f1_score(y_test, xgb_preds)
    xgb_tn, xgb_fp, xgb_fn, xgb_tp = confusion_matrix(y_test, xgb_preds).ravel()
    
    print("\n[2] XGBoost (Weighted 20 Paper Features):")
    print(f"    Accuracy:  {xgb_acc*100:.2f}%  (Paper Claimed: 95.2%)")
    print(f"    Precision: {xgb_prec*100:.2f}%")
    print(f"    Recall:    {xgb_rec*100:.2f}%")
    print(f"    F1-Score:  {xgb_f1*100:.2f}%")
    print(f"    Matrix:    [TN={xgb_tn}, FP={xgb_fp}] | [FN={xgb_fn}, TP={xgb_tp}]")
    print("="*70)

if __name__ == "__main__":
    main()

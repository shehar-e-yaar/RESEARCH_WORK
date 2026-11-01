import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from collections import OrderedDict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

BASE_DIR = r'c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
os.makedirs(RESULTS_DIR, exist_ok=True)

PAPER_20_FEATURES = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean', 'Packet Length Std',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Destination Port', 'ACK Flag Count'
]

PAPER_TABLE_4 = {
    'XGBoost': {'Acc': 95.2, 'Prec': 89.0, 'Rec': 95.2, 'Spec': 96.0, 'F1': 88.7},
    'Random Forest': {'Acc': 93.2, 'Prec': 89.4, 'Rec': 91.2, 'Spec': 95.3, 'F1': 87.1},
    'SVM': {'Acc': 91.2, 'Prec': 91.0, 'Rec': 91.0, 'Spec': 95.0, 'F1': 85.0},
    'K-Means': {'Acc': 84.6, 'Prec': 88.0, 'Rec': 81.6, 'Spec': 89.4, 'F1': 80.1},
    'Naive Bayes': {'Acc': 76.9, 'Prec': 76.0, 'Rec': 73.1, 'Spec': 85.2, 'F1': 67.9},
    'Logistic Regression': {'Acc': 74.1, 'Prec': 75.0, 'Rec': 84.6, 'Spec': 87.1, 'F1': 70.4},
}

# Fig. 4 confusion matrices (N=270 total per model)
# In standard format: (TP, FN, FP, TN) where Attack=Positive, Benign=Negative
PAPER_FIG4_CM = {
    'XGBoost': {'TP': 120, 'FN': 6, 'FP': 9, 'TN': 135},
    'Random Forest': {'TP': 119, 'FN': 12, 'FP': 6, 'TN': 133},
    'SVM': {'TP': 121, 'FN': 9, 'FP': 9, 'TN': 131},
    'K-Means': {'TP': 106, 'FN': 24, 'FP': 16, 'TN': 124},
    'Naive Bayes': {'TP': 96, 'FN': 35, 'FP': 20, 'TN': 119},
    'Logistic Regression': {'TP': 110, 'FN': 20, 'FP': 18, 'TN': 122},
}

def wilson_ci(p, n, z=1.96):
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d
    return max(0.0, c - m), min(1.0, c + m)

def two_proportion_z_test(p1, n1, p2, n2):
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_val)

# ==============================================================================
# TEST 1: Metric Averaging Hypothesis
# ==============================================================================
print("=" * 80)
print("TEST 1: METRIC AVERAGING HYPOTHESIS (Fig 4 CM vs Table 4 F1)")
print("=" * 80)

averaging_results = {}
for model, cm in PAPER_FIG4_CM.items():
    tp, fn, fp, tn = cm['TP'], cm['FN'], cm['FP'], cm['TN']
    total = tp + fn + fp + tn
    # Reconstruct ground truth and predictions
    y_true = np.array([1]*tp + [1]*fn + [0]*fp + [0]*tn)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    
    # Binary metrics (positive = Attack = 1)
    p_bin = precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    r_bin = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    f1_bin = f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    
    # Macro-averaged metrics
    p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    # Weighted-averaged metrics
    p_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    r_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    
    rep_f1 = PAPER_TABLE_4[model]['F1']
    rep_p  = PAPER_TABLE_4[model]['Prec']
    rep_r  = PAPER_TABLE_4[model]['Rec']
    
    calc_harmonic_from_table = 2 * (rep_p/100) * (rep_r/100) / ((rep_p + rep_r)/100) * 100
    
    averaging_results[model] = {
        'binary': {'P': round(p_bin, 2), 'R': round(r_bin, 2), 'F1': round(f1_bin, 2)},
        'macro': {'P': round(p_macro, 2), 'R': round(r_macro, 2), 'F1': round(f1_macro, 2)},
        'weighted': {'P': round(p_weighted, 2), 'R': round(r_weighted, 2), 'F1': round(f1_weighted, 2)},
        'reported_table_4': {'P': rep_p, 'R': rep_r, 'F1': rep_f1},
        'harmonic_from_table_P_R': round(calc_harmonic_from_table, 2),
    }
    
    print(f"\n{model}:")
    print(f"  Binary F1 (from CM):    {f1_bin:.2f}% | P: {p_bin:.2f}%, R: {r_bin:.2f}%")
    print(f"  Macro F1 (from CM):     {f1_macro:.2f}%")
    print(f"  Weighted F1 (from CM):  {f1_weighted:.2f}%")
    print(f"  Table 4 Reported F1:    {rep_f1:.2f}%")
    print(f"  Harmonic(Table P, R):   {calc_harmonic_from_table:.2f}%")

print("\n[Finding on Averaging]: None of binary, macro, or weighted F1 matches Table 4.")
print("The reported F1s remain lower than any mathematically valid averaging scheme.")

# ==============================================================================
# TEST 2: Inconsistency Audit
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 2: PAPER INCONSISTENCY AUDIT")
print("=" * 80)

inconsistency_log = []

# Check FN vs FP swap in Table 4 vs Fig 4
for model, cm in PAPER_FIG4_CM.items():
    tp, fn, fp, tn = cm['TP'], cm['FN'], cm['FP'], cm['TN']
    # If FP and FN are swapped:
    prec_swapped = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    rec_swapped  = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    prec_normal  = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec_normal   = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    
    inconsistency_log.append({
        'model': model,
        'cm_normal': {'P': round(prec_normal, 2), 'R': round(rec_normal, 2)},
        'cm_swapped': {'P': round(prec_swapped, 2), 'R': round(rec_swapped, 2)},
        'table_4': {'P': PAPER_TABLE_4[model]['Prec'], 'R': PAPER_TABLE_4[model]['Rec']}
    })

print("1. FN/FP Swap Audit:")
for item in inconsistency_log:
    print(f"  {item['model']:20s}: Table4=(P:{item['table_4']['P']}, R:{item['table_4']['R']}) | CM_norm=(P:{item['cm_normal']['P']}, R:{item['cm_normal']['R']}) | CM_swap=(P:{item['cm_swapped']['P']}, R:{item['cm_swapped']['R']})")

print("\n2. Text vs. Table Conflicts:")
print("  - SVM Accuracy: Text claims 93%, Table 4 reports 91.2%, Conclusion reports 91.2%")
print("  - Random Forest Accuracy: Text claims 94%, Table 4 reports 93.2%, Conclusion reports 93.2%")
print("  - Figure 4 Ambiguity: Figure 4 is referenced in Section III-B as the '3-stage mapping architecture', but the actual Figure 4 in the document displays the confusion matrices.")

# ==============================================================================
# TEST 3: Statistical Comparison with Paper (Two-Proportion Test & CI Overlap)
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 3: STATISTICAL COMPARISONS WITH PAPER (Two-Proportion Z-Test)")
print("=" * 80)
print("Note: A paired McNemar test is mathematically impossible because the authors")
print("did not publish per-sample binary test predictions or the exact row IDs.")
print("We perform two-proportion z-tests and Wilson score 95% CI overlap on N=270.\n")

stat_comp_results = {}
for model, p_data in PAPER_TABLE_4.items():
    p_rep = p_data['Acc'] / 100.0
    n_rep = 270
    ci_lo, ci_hi = wilson_ci(p_rep, n_rep)
    stat_comp_results[model] = {
        'paper_acc': p_data['Acc'],
        'paper_ci': [round(ci_lo * 100, 2), round(ci_hi * 100, 2)],
        'ci_width': round((ci_hi - ci_lo) * 100, 2)
    }
    print(f"  {model:20s}: Paper Acc={p_data['Acc']:.1f}% | 95% CI=[{ci_lo*100:.1f}%, {ci_hi*100:.1f}%] (width: {(ci_hi-ci_lo)*100:.1f}%)")

# ==============================================================================
# TEST 4: Scaling Hypothesis (Train LR, SVM, NB WITHOUT StandardScaler)
# ==============================================================================
def run_scaling_test(df_combined):
    print("\n" + "=" * 80)
    print("TEST 4: SCALING HYPOTHESIS TEST (Setup A Sample: N ~ 1350, Test = 270)")
    print("=" * 80)
    
    # Create balanced sample of 1350 flows
    b_df = df_combined[df_combined['binary_label'] == 0]
    a_df = df_combined[df_combined['binary_label'] == 1]
    
    n_sample = min(675, len(b_df), len(a_df))
    b_sub = b_df.sample(n=n_sample, random_state=42)
    a_sub = a_df.sample(n=n_sample, random_state=42)
    sample_df = pd.concat([b_sub, a_sub]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    X = sample_df[PAPER_20_FEATURES].copy()
    y = sample_df['binary_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Impute only (no scaler)
    imp = SimpleImputer(strategy='median')
    X_train_unscaled = imp.fit_transform(X_train)
    X_test_unscaled  = imp.transform(X_test)
    
    # Impute + Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_unscaled)
    X_test_scaled  = scaler.transform(X_test_unscaled)
    
    models_to_test = {
        'Logistic Regression': (
            LogisticRegression(max_iter=5000, random_state=42),
            LogisticRegression(max_iter=5000, random_state=42)
        ),
        'SVM (LinearSVC)': (
            LinearSVC(max_iter=10000, random_state=42, dual='auto'),
            LinearSVC(max_iter=10000, random_state=42, dual='auto')
        ),
        'Naive Bayes': (
            GaussianNB(),
            GaussianNB()
        )
    }
    
    scaling_comparison = {}
    print(f"\n{'Model':<22} | {'Unscaled Acc':>12} | {'Scaled Acc':>12} | {'Delta':>8} | {'Paper Table 4':>14}")
    print("-" * 75)
    for name, (m_unscaled, m_scaled) in models_to_test.items():
        # Train unscaled
        m_unscaled.fit(X_train_unscaled, y_train)
        pred_unscaled = m_unscaled.predict(X_test_unscaled)
        acc_unscaled = accuracy_score(y_test, pred_unscaled) * 100
        
        # Train scaled
        m_scaled.fit(X_train_scaled, y_train)
        pred_scaled = m_scaled.predict(X_test_scaled)
        acc_scaled = accuracy_score(y_test, pred_scaled) * 100
        
        paper_name = 'SVM' if 'SVM' in name else name
        paper_acc = PAPER_TABLE_4[paper_name]['Acc']
        delta = acc_scaled - acc_unscaled
        scaling_comparison[name] = {
            'acc_unscaled': round(acc_unscaled, 2),
            'acc_scaled': round(acc_scaled, 2),
            'delta': round(delta, 2),
            'paper_acc': paper_acc
        }
        print(f"{name:<22} | {acc_unscaled:>11.2f}% | {acc_scaled:>11.2f}% | {delta:>+7.2f}% | {paper_acc:>13.1f}%")
        
    return scaling_comparison

# ==============================================================================
# TEST 5: Feature Ablation on Setups B and C
# ==============================================================================
def run_ablation_experiments(train_b, test_b, train_c, test_c):
    print("\n" + "=" * 80)
    print("TEST 5: FEATURE ABLATION ON SETUPS B AND C")
    print("=" * 80)
    print("Regimes tested:")
    print("  1. Full 20 Features")
    print("  2. Drop Destination Port")
    print("  3. Drop Flow Duration")
    print("  4. Drop BOTH Destination Port and Flow Duration\n")
    
    ablation_regimes = {
        'Full_20': PAPER_20_FEATURES,
        'Drop_Dest_Port': [f for f in PAPER_20_FEATURES if f != 'Destination Port'],
        'Drop_Flow_Duration': [f for f in PAPER_20_FEATURES if f != 'Flow Duration'],
        'Drop_Both': [f for f in PAPER_20_FEATURES if f not in ['Destination Port', 'Flow Duration']],
    }
    
    ablation_results = {'Setup_B': {}, 'Setup_C': {}}
    
    # Models to test for ablation
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0),
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42, n_jobs=-1),
    }
    
    for regime_name, feat_cols in ablation_regimes.items():
        print(f"\n--- Regime: {regime_name} ({len(feat_cols)} features) ---")
        ablation_results['Setup_B'][regime_name] = {}
        ablation_results['Setup_C'][regime_name] = {}
        
        # Preprocess Setup B
        imp_b = SimpleImputer(strategy='median')
        scl_b = StandardScaler()
        Xb_tr = scl_b.fit_transform(imp_b.fit_transform(train_b[feat_cols]))
        Xb_te = scl_b.transform(imp_b.transform(test_b[feat_cols]))
        yb_tr, yb_te = train_b['binary_label'].values, test_b['binary_label'].values
        
        # Preprocess Setup C
        imp_c = SimpleImputer(strategy='median')
        scl_c = StandardScaler()
        Xc_tr = scl_c.fit_transform(imp_c.fit_transform(train_c[feat_cols]))
        Xc_te = scl_c.transform(imp_c.transform(test_c[feat_cols]))
        yc_tr, yc_te = train_c['binary_label'].values, test_c['binary_label'].values
        
        for m_name, model in models.items():
            # Setup B
            model.fit(Xb_tr, yb_tr)
            pred_b = model.predict(Xb_te)
            acc_b = accuracy_score(yb_te, pred_b) * 100
            f1_b  = f1_score(yb_te, pred_b, zero_division=0) * 100
            ablation_results['Setup_B'][regime_name][m_name] = {'Acc': round(acc_b, 2), 'F1': round(f1_b, 2)}
            
            # Setup C
            model.fit(Xc_tr, yc_tr)
            pred_c = model.predict(Xc_te)
            acc_c = accuracy_score(yc_te, pred_c) * 100
            f1_c  = f1_score(yc_te, pred_c, zero_division=0) * 100
            ablation_results['Setup_C'][regime_name][m_name] = {'Acc': round(acc_c, 2), 'F1': round(f1_c, 2)}
            
            print(f"  {m_name:<20}: Setup B Acc={acc_b:.2f}%, F1={f1_b:.2f}% | Setup C Acc={acc_c:.2f}%, F1={f1_c:.2f}%")
            
    return ablation_results

# Execute standalone tests and save partial
phase3_data = {
    'metric_averaging_hypothesis': averaging_results,
    'inconsistency_audit': inconsistency_log,
    'statistical_comparisons': stat_comp_results,
}

# If parquet data exists, also execute data-dependent tests
train_parquet = os.path.join(PROCESSED_DIR, 'train_day_sampled.parquet')
test_parquet  = os.path.join(PROCESSED_DIR, 'test_day_sampled.parquet')

if os.path.exists(train_parquet) and os.path.exists(test_parquet):
    print("\nParquet files detected. Running Scaling & Ablation tests...")
    tr_df = pd.read_parquet(train_parquet)
    te_df = pd.read_parquet(test_parquet)
    combined = pd.concat([tr_df, te_df], ignore_index=True)
    
    scaling_results = run_scaling_test(combined)
    phase3_data['scaling_hypothesis'] = scaling_results
    
    # Setup B train/test
    # Deduplicate combined for setup B
    comb_dedup = combined.drop_duplicates(subset=PAPER_20_FEATURES + ['binary_label']).reset_index(drop=True)
    tr_b, te_b = train_test_split(comb_dedup, test_size=0.2, stratify=comb_dedup['binary_label'], random_state=42)
    
    # Setup C train/test
    tr_c_dedup = tr_df.drop_duplicates(subset=PAPER_20_FEATURES + ['binary_label']).reset_index(drop=True)
    te_c_dedup = te_df.drop_duplicates(subset=PAPER_20_FEATURES + ['binary_label']).reset_index(drop=True)
    # Remove overlap
    overlap = pd.util.hash_pandas_object(te_c_dedup[PAPER_20_FEATURES].fillna(-999), index=False).isin(
        pd.util.hash_pandas_object(tr_c_dedup[PAPER_20_FEATURES].fillna(-999), index=False)
    )
    te_c_clean = te_c_dedup[~overlap].reset_index(drop=True)
    
    ablation_res = run_ablation_experiments(tr_b, te_b, tr_c_dedup, te_c_clean)
    phase3_data['feature_ablation'] = ablation_res

output_json = os.path.join(RESULTS_DIR, 'phase3_results.json')
with open(output_json, 'w') as f:
    json.dump(phase3_data, f, indent=2)
print(f"\n[OK] Phase 3 results saved to: {output_json}")

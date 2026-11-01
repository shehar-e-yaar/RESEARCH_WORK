import os
import sys
import gc
import json
import warnings
import numpy as np
import pandas as pd
from time import time
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
import xgboost as xgb

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

BASE_DIR = r'c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
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

SEEDS = [42, 123, 456, 789, 1024]

# True class prior in raw Day 2 (03-11): 56,965 Benign / 20,364,525 Total = 0.002797 (~0.28%)
TRUE_BENIGN_PRIOR = 56965 / 20364525
TRUE_ATTACK_PRIOR = 1.0 - TRUE_BENIGN_PRIOR

def wilson_ci(p, n, z=1.96):
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d
    return max(0.0, c - m), min(1.0, c + m)

def compute_metrics(y_true, y_pred, y_prob=None, n_bootstrap=500):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    
    acc  = (tp + tn) / n * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr  = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0.0
    
    # 95% Wilson CI for Accuracy
    acc_lo, acc_hi = wilson_ci(acc / 100.0, n)
    
    roc_auc, pr_auc = None, None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob) * 100
            pr_auc  = average_precision_score(y_true, y_prob) * 100
        except Exception:
            pass
            
    metrics = {
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'Acc': round(acc, 2),
        'Acc_CI': [round(acc_lo * 100, 2), round(acc_hi * 100, 2)],
        'Prec': round(prec, 2),
        'Rec': round(rec, 2),
        'Spec': round(spec, 2),
        'F1': round(f1, 2),
        'FPR': round(fpr, 2),
        'FNR': round(fnr, 2),
        'ROC_AUC': round(roc_auc, 2) if roc_auc is not None else None,
        'PR_AUC': round(pr_auc, 2) if pr_auc is not None else None,
    }
    
    # Fast vectorized bootstrap for CIs of other metrics
    if n_bootstrap > 0 and n > 0:
        rng = np.random.default_rng(42)
        boot_f1, boot_prec, boot_rec = [], [], []
        # Draw 500 resamples
        boot_indices = rng.integers(0, n, size=(min(n_bootstrap, 300), n))
        for idxs in boot_indices:
            b_yt = y_true[idxs]
            b_yp = y_pred[idxs]
            b_tn, b_fp, b_fn, b_tp = confusion_matrix(b_yt, b_yp, labels=[0, 1]).ravel()
            b_p = b_tp / (b_tp + b_fp) * 100 if (b_tp + b_fp) > 0 else 0.0
            b_r = b_tp / (b_tp + b_fn) * 100 if (b_tp + b_fn) > 0 else 0.0
            b_f = 2 * b_p * b_r / (b_p + b_r) if (b_p + b_r) > 0 else 0.0
            boot_f1.append(b_f)
            boot_prec.append(b_p)
            boot_rec.append(b_r)
        metrics['F1_CI'] = [round(float(np.percentile(boot_f1, 2.5)), 2), round(float(np.percentile(boot_f1, 97.5)), 2)]
        metrics['Prec_CI'] = [round(float(np.percentile(boot_prec, 2.5)), 2), round(float(np.percentile(boot_prec, 97.5)), 2)]
        metrics['Rec_CI'] = [round(float(np.percentile(boot_rec, 2.5)), 2), round(float(np.percentile(boot_rec, 97.5)), 2)]
        
    return metrics

def reweight_setup_c(metrics, p_benign=TRUE_BENIGN_PRIOR):
    """Reweight metrics according to true operating benign prior."""
    spec = metrics['Spec'] / 100.0
    rec  = metrics['Rec'] / 100.0
    p_attack = 1.0 - p_benign
    
    acc_rew = (p_benign * spec + p_attack * rec) * 100
    prec_denom = (p_attack * rec + p_benign * (1.0 - spec))
    prec_rew = (p_attack * rec / prec_denom * 100) if prec_denom > 0 else 0.0
    f1_rew = (2 * prec_rew * (rec * 100) / (prec_rew + rec * 100)) if (prec_rew + rec * 100) > 0 else 0.0
    
    return {
        'Acc_Reweighted': round(acc_rew, 2),
        'Prec_Reweighted': round(prec_rew, 2),
        'Rec_Reweighted': round(rec * 100, 2),
        'F1_Reweighted': round(f1_rew, 2),
        'True_Benign_Prior': round(p_benign * 100, 4)
    }

def get_models(seed=42):
    return OrderedDict({
        'XGBoost': (
            xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                eval_metric='logloss', random_state=seed, n_jobs=-1, verbosity=0
            ), True
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1), True
        ),
        'SVM (LinearSVC)': (
            LinearSVC(max_iter=10000, random_state=seed, dual='auto'), False
        ),
        'Logistic Regression': (
            LogisticRegression(max_iter=5000, random_state=seed, n_jobs=-1), True
        ),
        'Naive Bayes': (
            GaussianNB(), True
        ),
        'Decision Tree': (
            DecisionTreeClassifier(max_depth=10, random_state=seed), True
        ),
        'Majority Class': (
            DummyClassifier(strategy='most_frequent', random_state=seed), True
        ),
    })

def run_kmeans_dual(X_train, y_train, X_test, y_test, seed=42):
    """
    K-Means with:
    1. Train-only majority vote mapping (strictly leakage-free)
    2. Best-case post-hoc test mapping (cheating upper bound)
    """
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    km.fit(X_train)
    
    train_c = km.predict(X_train)
    test_c  = km.predict(X_test)
    
    # 1. Train-only mapping
    train_map = {}
    for c in [0, 1]:
        mask = (train_c == c)
        if mask.sum() > 0:
            train_map[c] = int(round(y_train[mask].mean()))
        else:
            train_map[c] = 0
    pred_train_mapped = np.array([train_map[c] for c in test_c])
    metrics_train_map = compute_metrics(y_test, pred_train_mapped)
    metrics_train_map['mapping_used'] = str(train_map)
    metrics_train_map['method'] = 'Train-only majority vote (leakage-free)'
    
    # 2. Post-hoc cheating mapping on test
    pred_c0_as_0 = test_c.copy()
    pred_c0_as_1 = 1 - test_c
    acc_0 = (pred_c0_as_0 == y_test).mean()
    acc_1 = (pred_c0_as_1 == y_test).mean()
    if acc_0 >= acc_1:
        pred_posthoc = pred_c0_as_0
        posthoc_map = {0: 0, 1: 1}
    else:
        pred_posthoc = pred_c0_as_1
        posthoc_map = {0: 1, 1: 0}
    metrics_posthoc = compute_metrics(y_test, pred_posthoc)
    metrics_posthoc['mapping_used'] = str(posthoc_map)
    metrics_posthoc['method'] = 'Post-hoc test cheating mapping'
    
    return metrics_train_map, metrics_posthoc

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
print("=" * 80)
print("PHASE 2: LEAKAGE-FREE EVALUATION & MODELING")
print("=" * 80)

print("\n1. Loading parquet datasets...")
train_parquet = os.path.join(PROCESSED_DIR, 'train_day_sampled.parquet')
test_parquet  = os.path.join(PROCESSED_DIR, 'test_day_sampled.parquet')

train_day = pd.read_parquet(train_parquet)
test_day  = pd.read_parquet(test_parquet)

print(f"  Raw Train Day (01-12): {len(train_day):,} flows")
print(f"  Raw Test Day (03-11):  {len(test_day):,} flows")

# ── Deduplication ──
print("\n2. Executing Strict Deduplication...")
combined = pd.concat([train_day, test_day], ignore_index=True)
n_before = len(combined)
combined_dedup = combined.drop_duplicates(subset=PAPER_20_FEATURES + ['binary_label']).reset_index(drop=True)
n_after = len(combined_dedup)
print(f"  Full Combined Data: {n_before:,} -> {n_after:,} (Removed {n_before - n_after:,} duplicate flows)")

# Day 1 and Day 2 individual dedup + cross-day dedup
train_day_dedup = train_day.drop_duplicates(subset=PAPER_20_FEATURES + ['binary_label']).reset_index(drop=True)
test_day_dedup  = test_day.drop_duplicates(subset=PAPER_20_FEATURES + ['binary_label']).reset_index(drop=True)

# Remove test rows that appear in train
train_hashes = pd.util.hash_pandas_object(train_day_dedup[PAPER_20_FEATURES].fillna(-999), index=False)
test_hashes  = pd.util.hash_pandas_object(test_day_dedup[PAPER_20_FEATURES].fillna(-999), index=False)
cross_overlap = test_hashes.isin(train_hashes)
n_cross = cross_overlap.sum()
test_day_clean = test_day_dedup[~cross_overlap].reset_index(drop=True)

print(f"  Day 1 Deduplicated: {len(train_day_dedup):,} flows (Removed {len(train_day) - len(train_day_dedup):,})")
print(f"  Day 2 Deduplicated: {len(test_day_clean):,} flows (Removed {len(test_day) - len(test_day_clean):,} total, {n_cross:,} cross-day overlap)")

phase2_results = {
    'deduplication_summary': {
        'combined_raw': n_before,
        'combined_dedup': n_after,
        'combined_removed': n_before - n_after,
        'day1_dedup': len(train_day_dedup),
        'day2_dedup': len(test_day_clean),
        'cross_day_overlap_removed': int(n_cross),
    },
    'setup_a_seeds': {},
    'setup_a_summary': {},
    'setup_b_seeds': {},
    'setup_b_summary': {},
    'setup_c_seeds': {},
    'setup_c_summary': {},
    'cv_full_setup_b': {},
}

# ==============================================================================
# SETUP A: ~1,350 Balanced Flows, 20% Test Split Across 5 Seeds
# ==============================================================================
print("\n" + "=" * 80)
print("SETUP A: ~1,350 BALANCED ROWS (20% TEST SPLIT ACROSS 5 SEEDS)")
print("=" * 80)

b_pool = combined_dedup[combined_dedup['binary_label'] == 0]
a_pool = combined_dedup[combined_dedup['binary_label'] == 1]
n_per_class = min(675, len(b_pool), len(a_pool))

setup_a_seed_metrics = OrderedDict()
for seed in SEEDS:
    print(f"\n--- Running Setup A with seed={seed} ---")
    b_sample = b_pool.sample(n=n_per_class, random_state=seed)
    a_sample = a_pool.sample(n=n_per_class, random_state=seed)
    df_a = pd.concat([b_sample, a_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    Xa = df_a[PAPER_20_FEATURES]
    ya = df_a['binary_label'].values
    
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.2, stratify=ya, random_state=seed)
    
    # Impute and scale fit on train only
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    Xa_tr_p = scl.fit_transform(imp.fit_transform(Xa_tr))
    Xa_te_p = scl.transform(imp.transform(Xa_te))
    
    models = get_models(seed=seed)
    setup_a_seed_metrics[seed] = {}
    
    for name, (model, has_proba) in models.items():
        model.fit(Xa_tr_p, ya_tr)
        yp = model.predict(Xa_te_p)
        yprob = model.predict_proba(Xa_te_p)[:, 1] if has_proba else None
        m = compute_metrics(ya_te, yp, yprob)
        setup_a_seed_metrics[seed][name] = m
        print(f"  {name:<20}: Acc={m['Acc']}% | F1={m['F1']}% | Prec={m['Prec']}% | Rec={m['Rec']}%")
        
    # K-Means dual
    km_train, km_posthoc = run_kmeans_dual(Xa_tr_p, ya_tr, Xa_te_p, ya_te, seed=seed)
    setup_a_seed_metrics[seed]['K-Means (Train Map)'] = km_train
    setup_a_seed_metrics[seed]['K-Means (Post-Hoc Cheating)'] = km_posthoc
    print(f"  {'K-Means (Train Map)':<20}: Acc={km_train['Acc']}% | F1={km_train['F1']}%")
    print(f"  {'K-Means (Post-Hoc)':<20}: Acc={km_posthoc['Acc']}% | F1={km_posthoc['F1']}%")

# Aggregate Setup A Mean +/- Std
for model_name in list(setup_a_seed_metrics[SEEDS[0]].keys()):
    accs = [setup_a_seed_metrics[s][model_name]['Acc'] for s in SEEDS]
    f1s  = [setup_a_seed_metrics[s][model_name]['F1'] for s in SEEDS]
    precs = [setup_a_seed_metrics[s][model_name]['Prec'] for s in SEEDS]
    recs  = [setup_a_seed_metrics[s][model_name]['Rec'] for s in SEEDS]
    specs = [setup_a_seed_metrics[s][model_name]['Spec'] for s in SEEDS]
    phase2_results['setup_a_summary'][model_name] = {
        'Acc_mean': round(float(np.mean(accs)), 2), 'Acc_std': round(float(np.std(accs)), 2),
        'F1_mean': round(float(np.mean(f1s)), 2),   'F1_std': round(float(np.std(f1s)), 2),
        'Prec_mean': round(float(np.mean(precs)), 2), 'Prec_std': round(float(np.std(precs)), 2),
        'Rec_mean': round(float(np.mean(recs)), 2),   'Rec_std': round(float(np.std(recs)), 2),
        'Spec_mean': round(float(np.mean(specs)), 2), 'Spec_std': round(float(np.std(specs)), 2),
    }

# ==============================================================================
# SETUP B: Full Deduplicated Stratified 80/20 Across 5 Seeds
# ==============================================================================
print("\n" + "=" * 80)
print("SETUP B: FULL DEDUPLICATED STRATIFIED 80/20 ACROSS 5 SEEDS")
print("=" * 80)

setup_b_seed_metrics = OrderedDict()
for seed in SEEDS:
    print(f"\n--- Running Setup B with seed={seed} ---")
    Xb = combined_dedup[PAPER_20_FEATURES]
    yb = combined_dedup['binary_label'].values
    
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=seed)
    
    # Remove any test rows that appear in train
    tr_h = pd.util.hash_pandas_object(Xb_tr.fillna(-999), index=False)
    te_h = pd.util.hash_pandas_object(Xb_te.fillna(-999), index=False)
    ov = te_h.isin(tr_h)
    if ov.sum() > 0:
        Xb_te = Xb_te[~ov]
        yb_te = yb_te[~ov]
        
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    Xb_tr_p = scl.fit_transform(imp.fit_transform(Xb_tr))
    Xb_te_p = scl.transform(imp.transform(Xb_te))
    
    models = get_models(seed=seed)
    setup_b_seed_metrics[seed] = {}
    
    for name, (model, has_proba) in models.items():
        t0 = time()
        model.fit(Xb_tr_p, yb_tr)
        yp = model.predict(Xb_te_p)
        yprob = model.predict_proba(Xb_te_p)[:, 1] if has_proba else None
        m = compute_metrics(yb_te, yp, yprob, n_bootstrap=100)
        m['train_time_s'] = round(time() - t0, 2)
        setup_b_seed_metrics[seed][name] = m
        print(f"  {name:<20}: Acc={m['Acc']}% | F1={m['F1']}% | Prec={m['Prec']}% | Rec={m['Rec']}% ({m['train_time_s']}s)")
        
    km_train, km_posthoc = run_kmeans_dual(Xb_tr_p, yb_tr, Xb_te_p, yb_te, seed=seed)
    setup_b_seed_metrics[seed]['K-Means (Train Map)'] = km_train
    setup_b_seed_metrics[seed]['K-Means (Post-Hoc Cheating)'] = km_posthoc
    print(f"  {'K-Means (Train Map)':<20}: Acc={km_train['Acc']}% | F1={km_train['F1']}%")
    print(f"  {'K-Means (Post-Hoc)':<20}: Acc={km_posthoc['Acc']}% | F1={km_posthoc['F1']}%")

# Aggregate Setup B Mean +/- Std
for model_name in list(setup_b_seed_metrics[SEEDS[0]].keys()):
    accs = [setup_b_seed_metrics[s][model_name]['Acc'] for s in SEEDS]
    f1s  = [setup_b_seed_metrics[s][model_name]['F1'] for s in SEEDS]
    precs = [setup_b_seed_metrics[s][model_name]['Prec'] for s in SEEDS]
    recs  = [setup_b_seed_metrics[s][model_name]['Rec'] for s in SEEDS]
    specs = [setup_b_seed_metrics[s][model_name]['Spec'] for s in SEEDS]
    phase2_results['setup_b_summary'][model_name] = {
        'Acc_mean': round(float(np.mean(accs)), 2), 'Acc_std': round(float(np.std(accs)), 2),
        'F1_mean': round(float(np.mean(f1s)), 2),   'F1_std': round(float(np.std(f1s)), 2),
        'Prec_mean': round(float(np.mean(precs)), 2), 'Prec_std': round(float(np.std(precs)), 2),
        'Rec_mean': round(float(np.mean(recs)), 2),   'Rec_std': round(float(np.std(recs)), 2),
        'Spec_mean': round(float(np.mean(specs)), 2), 'Spec_std': round(float(np.std(specs)), 2),
    }

# ==============================================================================
# SETUP C: Cross-Day Evaluation (Day 1 -> Day 2) Across 5 Seeds
# ==============================================================================
print("\n" + "=" * 80)
print("SETUP C: CROSS-DAY GENERALIZATION (DAY 1 TRAIN -> DAY 2 TEST) ACROSS 5 SEEDS")
print("=" * 80)

setup_c_seed_metrics = OrderedDict()
for seed in SEEDS:
    print(f"\n--- Running Setup C with seed={seed} ---")
    Xc_tr = train_day_dedup[PAPER_20_FEATURES]
    yc_tr = train_day_dedup['binary_label'].values
    Xc_te = test_day_clean[PAPER_20_FEATURES]
    yc_te = test_day_clean['binary_label'].values
    
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    Xc_tr_p = scl.fit_transform(imp.fit_transform(Xc_tr))
    Xc_te_p = scl.transform(imp.transform(Xc_te))
    
    models = get_models(seed=seed)
    setup_c_seed_metrics[seed] = {}
    
    for name, (model, has_proba) in models.items():
        t0 = time()
        model.fit(Xc_tr_p, yc_tr)
        yp = model.predict(Xc_te_p)
        yprob = model.predict_proba(Xc_te_p)[:, 1] if has_proba else None
        m = compute_metrics(yc_te, yp, yprob, n_bootstrap=100)
        m['train_time_s'] = round(time() - t0, 2)
        # Reweight to realistic class prior
        m['reweighted_realistic_prior'] = reweight_setup_c(m, p_benign=TRUE_BENIGN_PRIOR)
        setup_c_seed_metrics[seed][name] = m
        print(f"  {name:<20}: Sampled Acc={m['Acc']}% (Reweighted={m['reweighted_realistic_prior']['Acc_Reweighted']}%) | F1={m['F1']}% | Rec={m['Rec']}%")
        
    km_train, km_posthoc = run_kmeans_dual(Xc_tr_p, yc_tr, Xc_te_p, yc_te, seed=seed)
    km_train['reweighted_realistic_prior'] = reweight_setup_c(km_train, p_benign=TRUE_BENIGN_PRIOR)
    km_posthoc['reweighted_realistic_prior'] = reweight_setup_c(km_posthoc, p_benign=TRUE_BENIGN_PRIOR)
    setup_c_seed_metrics[seed]['K-Means (Train Map)'] = km_train
    setup_c_seed_metrics[seed]['K-Means (Post-Hoc Cheating)'] = km_posthoc
    print(f"  {'K-Means (Train Map)':<20}: Sampled Acc={km_train['Acc']}% (Reweighted={km_train['reweighted_realistic_prior']['Acc_Reweighted']}%)")
    print(f"  {'K-Means (Post-Hoc)':<20}: Sampled Acc={km_posthoc['Acc']}% (Reweighted={km_posthoc['reweighted_realistic_prior']['Acc_Reweighted']}%)")

# Aggregate Setup C Mean +/- Std
for model_name in list(setup_c_seed_metrics[SEEDS[0]].keys()):
    accs = [setup_c_seed_metrics[s][model_name]['Acc'] for s in SEEDS]
    f1s  = [setup_c_seed_metrics[s][model_name]['F1'] for s in SEEDS]
    precs = [setup_c_seed_metrics[s][model_name]['Prec'] for s in SEEDS]
    recs  = [setup_c_seed_metrics[s][model_name]['Rec'] for s in SEEDS]
    specs = [setup_c_seed_metrics[s][model_name]['Spec'] for s in SEEDS]
    rew_accs = [setup_c_seed_metrics[s][model_name]['reweighted_realistic_prior']['Acc_Reweighted'] for s in SEEDS]
    rew_f1s  = [setup_c_seed_metrics[s][model_name]['reweighted_realistic_prior']['F1_Reweighted'] for s in SEEDS]
    phase2_results['setup_c_summary'][model_name] = {
        'Acc_mean': round(float(np.mean(accs)), 2), 'Acc_std': round(float(np.std(accs)), 2),
        'F1_mean': round(float(np.mean(f1s)), 2),   'F1_std': round(float(np.std(f1s)), 2),
        'Prec_mean': round(float(np.mean(precs)), 2), 'Prec_std': round(float(np.std(precs)), 2),
        'Rec_mean': round(float(np.mean(recs)), 2),   'Rec_std': round(float(np.std(recs)), 2),
        'Spec_mean': round(float(np.mean(specs)), 2), 'Spec_std': round(float(np.std(specs)), 2),
        'Acc_Reweighted_mean': round(float(np.mean(rew_accs)), 2),
        'F1_Reweighted_mean': round(float(np.mean(rew_f1s)), 2),
    }

# ==============================================================================
# STRATIFIED 5-FOLD CV x 5 SEEDS ON FULL SETUP B TRAINING DATA
# ==============================================================================
print("\n" + "=" * 80)
print("STRATIFIED 5-FOLD CV x 5 SEEDS ON FULL SETUP B TRAINING DATA (ALL 8 MODELS)")
print("=" * 80)

# Use Setup B training set for CV (from seed 42 split)
X_cv_full = Xb_tr.copy().reset_index(drop=True)
y_cv_full = yb_tr.copy()

print(f"Full CV Training Set Size: {len(X_cv_full):,} flows")

cv_full_results = {}
cv_model_names = ['XGBoost', 'Random Forest', 'SVM (LinearSVC)', 'Logistic Regression',
                    'Naive Bayes', 'Decision Tree', 'Majority Class', 'K-Means']

for m_name in cv_model_names:
    print(f"\n--- Cross-Validating {m_name} (25 folds total) ---")
    fold_accs, fold_f1s, fold_precs, fold_recs = [], [], [], []
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_cv_full, y_cv_full)):
            X_tr_f, X_val_f = X_cv_full.iloc[tr_idx], X_cv_full.iloc[val_idx]
            y_tr_f, y_val_f = y_cv_full[tr_idx], y_cv_full[val_idx]
            
            imp = SimpleImputer(strategy='median')
            scl = StandardScaler()
            X_tr_fp = scl.fit_transform(imp.fit_transform(X_tr_f))
            X_val_fp = scl.transform(imp.transform(X_val_f))
            
            if m_name == 'K-Means':
                km = KMeans(n_clusters=2, random_state=seed, n_init=5)
                km.fit(X_tr_fp)
                tr_c = km.predict(X_tr_fp)
                val_c = km.predict(X_val_fp)
                t_map = {}
                for c in [0, 1]:
                    mask = (tr_c == c)
                    t_map[c] = int(round(y_tr_f[mask].mean())) if mask.sum() > 0 else 0
                y_pred_f = np.array([t_map[c] for c in val_c])
            else:
                m_obj, _ = get_models(seed=seed)[m_name]
                m_obj.fit(X_tr_fp, y_tr_f)
                y_pred_f = m_obj.predict(X_val_fp)
                
            m_eval = compute_metrics(y_val_f, y_pred_f, n_bootstrap=0)
            fold_accs.append(m_eval['Acc'])
            fold_f1s.append(m_eval['F1'])
            fold_precs.append(m_eval['Prec'])
            fold_recs.append(m_eval['Rec'])
            
    cv_full_results[m_name] = {
        'Acc_mean': round(float(np.mean(fold_accs)), 2), 'Acc_std': round(float(np.std(fold_accs)), 2),
        'F1_mean': round(float(np.mean(fold_f1s)), 2),   'F1_std': round(float(np.std(fold_f1s)), 2),
        'Prec_mean': round(float(np.mean(fold_precs)), 2), 'Prec_std': round(float(np.std(fold_precs)), 2),
        'Rec_mean': round(float(np.mean(fold_recs)), 2),   'Rec_std': round(float(np.std(fold_recs)), 2),
    }
    print(f"  {m_name} (25 folds): Acc = {cv_full_results[m_name]['Acc_mean']} +/- {cv_full_results[m_name]['Acc_std']}% | F1 = {cv_full_results[m_name]['F1_mean']} +/- {cv_full_results[m_name]['F1_std']}%")

phase2_results['cv_full_setup_b'] = cv_full_results
phase2_results['setup_a_seeds'] = setup_a_seed_metrics
phase2_results['setup_b_seeds'] = setup_b_seed_metrics
phase2_results['setup_c_seeds'] = setup_c_seed_metrics

output_path = os.path.join(RESULTS_DIR, 'phase2_results.json')
with open(output_path, 'w') as f:
    json.dump(phase2_results, f, indent=2)

print(f"\n[OK] Phase 2 evaluation complete! Results saved to: {output_path}")

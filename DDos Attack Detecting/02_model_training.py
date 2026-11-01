import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
# %% [markdown]
# # Reproducing: DDoS Detection with ML — Part 2: Model Training & Evaluation
# ## Three Reproduction Setups + Validation + Gap Analysis
#
# **Setup A**: Paper-like (small balanced sample, test~270)
# **Setup B**: Full sampled data, stratified random 80/20 split
# **Setup C**: Cross-day split (train on 01-12, test on 03-11)

# %% Imports
import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
from time import time
from collections import OrderedDict

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

# ── Paths ──
BASE_DIR = r'c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting'
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Paper's 20 Features ──
PAPER_20_FEATURES = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean', 'Packet Length Std',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Destination Port', 'ACK Flag Count',
]

# ── Paper's reported results (Table 4) ──
PAPER_RESULTS = {
    'XGBoost':    {'Acc': 95.2, 'Prec': 89.0, 'Rec': 95.2, 'Spec': 96.0, 'F1': 88.7},
    'Random Forest': {'Acc': 93.2, 'Prec': 89.4, 'Rec': 91.2, 'Spec': 95.3, 'F1': 87.1},
    'SVM (LinearSVC)': {'Acc': 91.2, 'Prec': 91.0, 'Rec': 91.0, 'Spec': 95.0, 'F1': 85.0},
    'K-Means':    {'Acc': 84.6, 'Prec': 88.0, 'Rec': 81.6, 'Spec': 89.4, 'F1': 80.1},
    'Naive Bayes':{'Acc': 76.9, 'Prec': 76.0, 'Rec': 73.1, 'Spec': 85.2, 'F1': 67.9},
    'Logistic Regression': {'Acc': 74.1, 'Prec': 75.0, 'Rec': 84.6, 'Spec': 87.1, 'F1': 70.4},
}

# %% [markdown]
# ## 1. Load Processed Data

# %%
print("Loading processed data...")
train_day_df = pd.read_parquet(os.path.join(OUTPUT_DIR, 'train_day_sampled.parquet'))
test_day_df  = pd.read_parquet(os.path.join(OUTPUT_DIR, 'test_day_sampled.parquet'))

# Combine for setups A and B
full_df = pd.concat([train_day_df, test_day_df], ignore_index=True)

print(f"Train day: {len(train_day_df):,} rows")
print(f"Test day:  {len(test_day_df):,} rows")
print(f"Combined:  {len(full_df):,} rows")

# Check which paper features are available
available_features = [f for f in PAPER_20_FEATURES if f in full_df.columns]
missing_features = [f for f in PAPER_20_FEATURES if f not in full_df.columns]
if missing_features:
    print(f"[!] Missing features: {missing_features}")
print(f"Using {len(available_features)} of 20 paper features")

# %% [markdown]
# ## 2. Preprocessing Pipeline (Leakage-Free)

# %%
def preprocess_features(X_train, X_test, feature_cols):
    """
    Leakage-free preprocessing:
    1. Impute NaN/inf with median (fit on train only)
    2. Standard scale (fit on train only)
    
    Returns transformed X_train, X_test as numpy arrays.
    """
    X_train = X_train[feature_cols].copy()
    X_test  = X_test[feature_cols].copy()
    
    # Replace inf with NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)
    
    # Impute (fit on train only)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)
    
    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)
    
    return X_train_scaled, X_test_scaled, imputer, scaler


def deduplicate_across_splits(train_df, test_df, feature_cols):
    """
    Remove any rows from the test set that also appear in the training set
    (prevents data leakage from duplicate flows).
    """
    # Create hash of feature values for dedup
    train_hashes = pd.util.hash_pandas_object(
        train_df[feature_cols].fillna(-999), index=False
    )
    test_hashes = pd.util.hash_pandas_object(
        test_df[feature_cols].fillna(-999), index=False
    )
    
    overlap = test_hashes.isin(train_hashes)
    n_overlap = overlap.sum()
    if n_overlap > 0:
        print(f"  [!] Removing {n_overlap:,} duplicate rows from test set ({n_overlap/len(test_df)*100:.2f}%)")
        test_df = test_df[~overlap].reset_index(drop=True)
    else:
        print(f"  [OK] No duplicate rows between train and test")
    
    return train_df, test_df

# %% [markdown]
# ## 3. Metrics Functions

# %%
def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all metrics from the confusion matrix with correct formulas.
    Returns dict with: Acc, Prec, Rec, Spec, F1, FPR, FNR, and optionally ROC-AUC, PR-AUC.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    acc  = (tp + tn) / (tp + tn + fp + fn) * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0  # Sensitivity / TPR
    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0  # Specificity / TNR
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr  = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0.0
    
    metrics = OrderedDict({
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'Acc': round(acc, 2),
        'Prec': round(prec, 2),
        'Rec': round(rec, 2),
        'Spec': round(spec, 2),
        'F1': round(f1, 2),
        'FPR': round(fpr, 2),
        'FNR': round(fnr, 2),
        'N_test': len(y_true),
    })
    
    if y_prob is not None:
        try:
            metrics['ROC_AUC'] = round(roc_auc_score(y_true, y_prob) * 100, 2)
            metrics['PR_AUC']  = round(average_precision_score(y_true, y_prob) * 100, 2)
        except Exception:
            metrics['ROC_AUC'] = None
            metrics['PR_AUC'] = None
    
    return metrics


def wilson_ci(p, n, z=1.96):
    """
    Wilson score 95% confidence interval for a proportion.
    p: observed proportion (0-1 scale)
    n: sample size
    z: z-score for confidence level (1.96 for 95%)
    """
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denom
    return max(0, centre - margin), min(1, centre + margin)


def add_confidence_intervals(metrics_dict, n_test):
    """Add 95% Wilson CI to accuracy."""
    p = metrics_dict['Acc'] / 100
    lo, hi = wilson_ci(p, n_test)
    metrics_dict['Acc_CI_lo'] = round(lo * 100, 2)
    metrics_dict['Acc_CI_hi'] = round(hi * 100, 2)
    return metrics_dict

# %% [markdown]
# ## 4. Define Models
# 
# Notes:
# - Spark MLlib has NO kernel SVM -> we use LinearSVC (same as Spark)
# - Spark MLlib has NO native XGBoost -> we use xgboost.XGBClassifier
# - K-Means is unsupervised -> we do cluster-to-label majority mapping
# - All models use fixed random_state=42

# %%
def get_models():
    """
    Return dict of model name -> (model, supports_proba).
    K-Means is handled separately.
    """
    models = OrderedDict({
        # ── Paper models ──
        'XGBoost': (
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
            True,  # supports predict_proba
        ),
        'Random Forest': (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            True,
        ),
        'SVM (LinearSVC)': (
            LinearSVC(
                max_iter=10000,
                random_state=RANDOM_STATE,
                dual='auto',
            ),
            False,  # LinearSVC has decision_function, not predict_proba
        ),
        'Naive Bayes': (
            GaussianNB(),
            True,
        ),
        'Logistic Regression': (
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            True,
        ),
        # ── Baselines ──
        'Majority Class': (
            DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE),
            True,
        ),
        'Decision Tree': (
            DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
            True,
        ),
    })
    return models


def kmeans_predict(X_train, y_train, X_test, n_clusters=2, random_state=42):
    """
    K-Means with cluster-to-label majority mapping.
    
    The paper reports K-Means with TP/FP/TN/FN, which requires mapping
    unsupervised clusters to ground-truth labels. The standard approach:
    1. Fit K-Means on training data
    2. For each cluster, assign the majority label
    3. Predict using the cluster-to-label mapping
    
    NOTE: This is a DUBIOUS approach because K-Means is not designed for
    classification. The results depend heavily on feature scaling and
    the cluster-to-label mapping can be arbitrary for k != number of classes.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(X_train)
    
    # Map clusters to labels using training data
    train_clusters = km.predict(X_train)
    cluster_label_map = {}
    for c in range(n_clusters):
        mask = train_clusters == c
        if mask.sum() > 0:
            # Majority vote
            labels_in_cluster = y_train[mask]
            majority = int(np.round(labels_in_cluster.mean()))
            cluster_label_map[c] = majority
        else:
            cluster_label_map[c] = 0
    
    # Predict on test
    test_clusters = km.predict(X_test)
    y_pred = np.array([cluster_label_map[c] for c in test_clusters])
    
    return y_pred, cluster_label_map

# %% [markdown]
# ## 5. Setup A: Paper-Like (Small Balanced Sample, Test~270)

# %%
print("=" * 80)
print("SETUP A: Paper-Like (small balanced sample, test ~ 270)")
print("=" * 80)
print()
print("Rationale: The paper's confusion matrices sum to 270.")
print("If they used a 20% test split -> total ~ 1,350 rows.")
print("We create a balanced binary dataset of ~1,350 rows, split 80/20.")

# Create balanced binary sample
benign_mask = full_df['binary_label'] == 0
attack_mask = full_df['binary_label'] == 1

n_benign = benign_mask.sum()
n_attack = attack_mask.sum()
print(f"Available benign: {n_benign:,}, attack: {n_attack:,}")

# Sample ~675 of each class for a total of ~1,350
SETUP_A_PER_CLASS = 675
benign_sample = full_df[benign_mask].sample(
    n=min(SETUP_A_PER_CLASS, n_benign), random_state=RANDOM_STATE
)
attack_sample = full_df[attack_mask].sample(
    n=min(SETUP_A_PER_CLASS, n_attack), random_state=RANDOM_STATE
)
setup_a_df = pd.concat([benign_sample, attack_sample]).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

print(f"Setup A dataset: {len(setup_a_df)} rows, balanced")
print(f"Label distribution: {dict(setup_a_df['binary_label'].value_counts())}")

# 80/20 stratified split
X_a = setup_a_df[available_features]
y_a = setup_a_df['binary_label'].values

X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(
    X_a, y_a, test_size=0.2, stratify=y_a, random_state=RANDOM_STATE
)
print(f"Train: {len(X_a_train)}, Test: {len(X_a_test)}")
print(f"Test label dist: {dict(zip(*np.unique(y_a_test, return_counts=True)))}")

# Preprocess
X_a_train_p, X_a_test_p, _, _ = preprocess_features(
    X_a_train, X_a_test, available_features
)

# Train all models
setup_a_results = {}
models = get_models()

for name, (model, has_proba) in models.items():
    print(f"\n  Training {name}...", end=' ')
    t0 = time()
    model.fit(X_a_train_p, y_a_train)
    train_time = time() - t0
    
    y_pred = model.predict(X_a_test_p)
    y_prob = None
    if has_proba:
        try:
            y_prob = model.predict_proba(X_a_test_p)[:, 1]
        except Exception:
            try:
                y_prob = model.decision_function(X_a_test_p)
            except Exception:
                pass
    
    metrics = compute_all_metrics(y_a_test, y_pred, y_prob)
    metrics['Train_Time_s'] = round(train_time, 3)
    metrics = add_confidence_intervals(metrics, len(y_a_test))
    setup_a_results[name] = metrics
    print(f"Acc={metrics['Acc']}% [CI: {metrics['Acc_CI_lo']}-{metrics['Acc_CI_hi']}%] "
          f"F1={metrics['F1']}% ({train_time:.2f}s)")

# K-Means
print(f"\n  Training K-Means...", end=' ')
t0 = time()
km_pred, km_map = kmeans_predict(X_a_train_p, y_a_train, X_a_test_p, 
                                  n_clusters=2, random_state=RANDOM_STATE)
train_time = time() - t0
metrics = compute_all_metrics(y_a_test, km_pred)
metrics['Train_Time_s'] = round(train_time, 3)
metrics['Cluster_Map'] = str(km_map)
metrics = add_confidence_intervals(metrics, len(y_a_test))
setup_a_results['K-Means'] = metrics
print(f"Acc={metrics['Acc']}% [CI: {metrics['Acc_CI_lo']}-{metrics['Acc_CI_hi']}%] "
      f"F1={metrics['F1']}% ({train_time:.2f}s)")

# %% [markdown]
# ## 6. Setup B: Full Sampled Data, Stratified Random Split

# %%
print("\n" + "=" * 80)
print("SETUP B: Full sampled data, stratified 80/20 split")
print("=" * 80)

X_b = full_df[available_features]
y_b = full_df['binary_label'].values

X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
    X_b, y_b, test_size=0.2, stratify=y_b, random_state=RANDOM_STATE
)

# Deduplicate
train_b_df = pd.DataFrame(X_b_train, columns=available_features)
train_b_df['binary_label'] = y_b_train
test_b_df = pd.DataFrame(X_b_test, columns=available_features)
test_b_df['binary_label'] = y_b_test

train_b_df, test_b_df = deduplicate_across_splits(train_b_df, test_b_df, available_features)
X_b_train = train_b_df[available_features]
y_b_train = train_b_df['binary_label'].values
X_b_test = test_b_df[available_features]
y_b_test = test_b_df['binary_label'].values

print(f"Train: {len(X_b_train):,}, Test: {len(X_b_test):,}")
print(f"Test label dist: {dict(zip(*np.unique(y_b_test, return_counts=True)))}")
print(f"[!] WARNING: Random splits on CICDDoS2019 often yield 99%+ accuracy")
print(f"  because identical/near-identical flows leak across train and test.")

X_b_train_p, X_b_test_p, _, _ = preprocess_features(
    X_b_train, X_b_test, available_features
)

setup_b_results = {}
models = get_models()

for name, (model, has_proba) in models.items():
    print(f"\n  Training {name}...", end=' ')
    t0 = time()
    model.fit(X_b_train_p, y_b_train)
    train_time = time() - t0
    
    y_pred = model.predict(X_b_test_p)
    y_prob = None
    if has_proba:
        try:
            y_prob = model.predict_proba(X_b_test_p)[:, 1]
        except Exception:
            try:
                y_prob = model.decision_function(X_b_test_p)
            except Exception:
                pass
    
    metrics = compute_all_metrics(y_b_test, y_pred, y_prob)
    metrics['Train_Time_s'] = round(train_time, 3)
    metrics = add_confidence_intervals(metrics, len(y_b_test))
    setup_b_results[name] = metrics
    print(f"Acc={metrics['Acc']}% F1={metrics['F1']}% ({train_time:.2f}s)")

# K-Means
print(f"\n  Training K-Means...", end=' ')
t0 = time()
km_pred, km_map = kmeans_predict(X_b_train_p, y_b_train, X_b_test_p,
                                  n_clusters=2, random_state=RANDOM_STATE)
train_time = time() - t0
metrics = compute_all_metrics(y_b_test, km_pred)
metrics['Train_Time_s'] = round(train_time, 3)
metrics = add_confidence_intervals(metrics, len(y_b_test))
setup_b_results['K-Means'] = metrics
print(f"Acc={metrics['Acc']}% F1={metrics['F1']}% ({train_time:.2f}s)")

# %% [markdown]
# ## 7. Setup C: Cross-Day Generalization (Train 01-12, Test 03-11)

# %%
print("\n" + "=" * 80)
print("SETUP C: Cross-day split (train on 01-12, test on 03-11)")
print("=" * 80)
print("This is the OFFICIAL intended evaluation. It tests generalization")
print("because different attack subtypes appear on different days.")

X_c_train = train_day_df[available_features]
y_c_train = train_day_df['binary_label'].values
X_c_test  = test_day_df[available_features]
y_c_test  = test_day_df['binary_label'].values

print(f"Train (01-12): {len(X_c_train):,}, Test (03-11): {len(X_c_test):,}")
print(f"Train labels: {dict(zip(*np.unique(y_c_train, return_counts=True)))}")
print(f"Test labels:  {dict(zip(*np.unique(y_c_test, return_counts=True)))}")

# Check label overlap
train_attacks = set(train_day_df['Label'].unique())
test_attacks  = set(test_day_df['Label'].unique())
print(f"\nTrain attack types: {train_attacks}")
print(f"Test attack types:  {test_attacks}")
print(f"Overlap: {train_attacks & test_attacks}")
print(f"Test-only (unseen): {test_attacks - train_attacks}")

X_c_train_p, X_c_test_p, _, _ = preprocess_features(
    X_c_train, X_c_test, available_features
)

setup_c_results = {}
models = get_models()

for name, (model, has_proba) in models.items():
    print(f"\n  Training {name}...", end=' ')
    t0 = time()
    model.fit(X_c_train_p, y_c_train)
    train_time = time() - t0
    
    y_pred = model.predict(X_c_test_p)
    y_prob = None
    if has_proba:
        try:
            y_prob = model.predict_proba(X_c_test_p)[:, 1]
        except Exception:
            try:
                y_prob = model.decision_function(X_c_test_p)
            except Exception:
                pass
    
    metrics = compute_all_metrics(y_c_test, y_pred, y_prob)
    metrics['Train_Time_s'] = round(train_time, 3)
    metrics = add_confidence_intervals(metrics, len(y_c_test))
    setup_c_results[name] = metrics
    print(f"Acc={metrics['Acc']}% F1={metrics['F1']}% ({train_time:.2f}s)")

# K-Means
print(f"\n  Training K-Means...", end=' ')
t0 = time()
km_pred, km_map = kmeans_predict(X_c_train_p, y_c_train, X_c_test_p,
                                  n_clusters=2, random_state=RANDOM_STATE)
train_time = time() - t0
metrics = compute_all_metrics(y_c_test, km_pred)
metrics['Train_Time_s'] = round(train_time, 3)
metrics = add_confidence_intervals(metrics, len(y_c_test))
setup_c_results['K-Means'] = metrics
print(f"Acc={metrics['Acc']}% F1={metrics['F1']}% ({train_time:.2f}s)")

# %% [markdown]
# ## 8. Cross-Validation (Setup B with 5-fold, 5 seeds)

# %%
print("\n" + "=" * 80)
print("STRATIFIED 5-FOLD CV with 5 RANDOM SEEDS (Setup B data)")
print("=" * 80)

# Leakage-free CV with Pipeline and stratified sample
from sklearn.pipeline import make_pipeline

X_cv = full_df[available_features].copy()
X_cv = X_cv.replace([np.inf, -np.inf], np.nan)
y_cv = full_df['binary_label'].values

# Stratified sample of up to 25,000 rows for fast 25-fold evaluation (5 folds x 5 seeds)
cv_sample_n = min(len(y_cv), 25000)
cv_indices, _ = train_test_split(
    np.arange(len(y_cv)), train_size=cv_sample_n, stratify=y_cv, random_state=42
)
X_cv_sub = X_cv.iloc[cv_indices].reset_index(drop=True)
y_cv_sub = y_cv[cv_indices]
print(f"Running 5-fold CV across 5 random seeds (25 folds per model) on {len(y_cv_sub):,} stratified samples...")

SEEDS = [42, 123, 456, 789, 1024]
cv_results = {}

cv_models = {
    'XGBoost': lambda seed: xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss',
        random_state=seed, n_jobs=-1, verbosity=0,
    ),
    'Random Forest': lambda seed: RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1,
    ),
    'SVM (LinearSVC)': lambda seed: LinearSVC(
        max_iter=10000, random_state=seed, dual='auto',
    ),
    'Logistic Regression': lambda seed: LogisticRegression(
        max_iter=5000, random_state=seed, n_jobs=-1,
    ),
    'Naive Bayes': lambda seed: GaussianNB(),
}

scoring = ['accuracy', 'precision', 'recall', 'f1']

for model_name, model_fn in cv_models.items():
    print(f"\n  CV: {model_name}")
    all_scores = {s: [] for s in scoring}
    
    for seed in SEEDS:
        model = model_fn(seed)
        pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            model
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_validate(
            pipe, X_cv_sub, y_cv_sub, cv=cv, scoring=scoring, n_jobs=-1
        )
        for s in scoring:
            all_scores[s].extend(scores[f'test_{s}'])
    
    cv_results[model_name] = {}
    for s in scoring:
        vals = np.array(all_scores[s]) * 100
        cv_results[model_name][s] = {
            'mean': round(vals.mean(), 2),
            'std': round(vals.std(), 2),
            'min': round(vals.min(), 2),
            'max': round(vals.max(), 2),
        }
        print(f"    {s}: {vals.mean():.2f} +/- {vals.std():.2f}%")

# %% [markdown]
# ## 9. Feature Importance and Shortcut Analysis

# %%
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE & SHORTCUT RISK ANALYSIS")
print("=" * 80)

# Train a full RF on Setup B to get feature importance
rf_for_importance = RandomForestClassifier(
    n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
)
rf_for_importance.fit(X_b_train_p, y_b_train)

importances = pd.Series(
    rf_for_importance.feature_importances_, 
    index=available_features
).sort_values(ascending=False)

print("\nRandom Forest feature importances (Setup B):")
for feat, imp in importances.items():
    bar = '#' * int(imp * 50)
    print(f"  {feat:30s} {imp:.4f} {bar}")

# ── Ablation: Remove top feature and retrain ──
print("\n--- Ablation: Remove top feature ---")
top_feat = importances.index[0]
print(f"Removing: {top_feat} (importance={importances.iloc[0]:.4f})")

ablation_features = [f for f in available_features if f != top_feat]
X_abl_train, X_abl_test, _, _ = preprocess_features(
    pd.DataFrame(X_b_train, columns=available_features),
    pd.DataFrame(X_b_test, columns=available_features),
    ablation_features
)
rf_ablation = RandomForestClassifier(
    n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
)
rf_ablation.fit(X_abl_train, y_b_train)
y_abl_pred = rf_ablation.predict(X_abl_test)
abl_acc = accuracy_score(y_b_test, y_abl_pred) * 100
full_acc = setup_b_results['Random Forest']['Acc']
print(f"RF accuracy WITH {top_feat}: {full_acc}%")
print(f"RF accuracy WITHOUT {top_feat}: {abl_acc:.2f}%")
print(f"Delta = {full_acc - abl_acc:+.2f}%")

# Shortcut features
print("\n--- Shortcut Risk Features ---")
shortcut_features = ['Destination Port', 'Flow Duration']
for feat in shortcut_features:
    if feat in available_features:
        single_features = [feat]
        X_s_train, X_s_test, _, _ = preprocess_features(
            pd.DataFrame(X_b_train, columns=available_features),
            pd.DataFrame(X_b_test, columns=available_features),
            single_features
        )
        rf_single = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_single.fit(X_s_train, y_b_train)
        y_s_pred = rf_single.predict(X_s_test)
        s_acc = accuracy_score(y_b_test, y_s_pred) * 100
        print(f"  RF with ONLY '{feat}': {s_acc:.2f}%")
        if s_acc > 90:
            print(f"  [!] HIGH SHORTCUT RISK: '{feat}' alone gives {s_acc:.1f}% accuracy!")

# %% [markdown]
# ## 10. Comparison: Paper vs Our Results

# %%
print("\n" + "=" * 80)
print("COMPARISON: PAPER (Table 4) vs OUR RESULTS")
print("=" * 80)

comparison_models = ['XGBoost', 'Random Forest', 'SVM (LinearSVC)', 'K-Means', 
                     'Naive Bayes', 'Logistic Regression']

# Build comparison table
print(f"\n{'Model':<25} {'Metric':>8} {'Paper':>8} {'Setup A':>10} {'Setup B':>10} {'Setup C':>10}")
print("-" * 75)

for model in comparison_models:
    paper_name = model
    paper = PAPER_RESULTS.get(paper_name, {})
    a = setup_a_results.get(model, {})
    b = setup_b_results.get(model, {})
    c = setup_c_results.get(model, {})
    
    for metric in ['Acc', 'Prec', 'Rec', 'F1', 'Spec']:
        p_val = paper.get(metric, '-')
        a_val = a.get(metric, '-')
        b_val = b.get(metric, '-')
        c_val = c.get(metric, '-')
        
        label = model if metric == 'Acc' else ''
        print(f"  {label:<23} {metric:>8} {str(p_val):>8} {str(a_val):>10} {str(b_val):>10} {str(c_val):>10}")
    print()

# %% [markdown]
# ## 11. Save All Results

# %%
all_results = {
    'setup_a': setup_a_results,
    'setup_b': setup_b_results,
    'setup_c': setup_c_results,
    'cv_results': cv_results,
    'paper_results': PAPER_RESULTS,
    'feature_importances': dict(importances),
    'config': {
        'features_used': available_features,
        'n_features': len(available_features),
        'random_state': RANDOM_STATE,
        'cv_seeds': SEEDS,
        'sample_per_file': 50000,
    },
}

# Convert numpy types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

results_path = os.path.join(RESULTS_DIR, 'reproduction_results.json')
with open(results_path, 'w') as f:
    json.dump(convert_numpy(all_results), f, indent=2)

print(f"\n[OK] All results saved to {results_path}")
print("\n" + "=" * 80)
print("REPRODUCTION COMPLETE")
print("=" * 80)

"""
reproduce.py

One script that reproduces and stress-tests the paper's central claim
("Ransomware Resilience: A Real-Time Detection Framework using Kafka and
Machine Learning", Table 1: XGBoost = 95.2% accuracy) against the actual
CIC-MalMem-2022 dataset.

What it does, in order:
  1. Load + clean the raw CSV, remove duplicates.
  2. Create a DOCUMENTED, reproducible 80/20 train/test split -- unlike the
     paper, which never explains how it got from 58,596 records down to a
     270-record test set. Saves data/train.csv + data/test.csv, used later
     by kafka_stream.py and spark_consumer.py for the live pipeline.
  3. Train all 6 models the paper used, compare each against its Table 1
     accuracy, save the trained XGBoost model (model_artifacts/xgb_ransomware.json).
  4. Hypothesis tests: does a smaller feature set, or a smaller/unlucky test
     set, explain the gap between reported and reproduced accuracy?
  5. Leakage check: the dataset's 'Category' field encodes a malware SAMPLE
     identity plus a snapshot number (e.g. 'Ransomware-Maze-<hash>-1.raw',
     '...-2.raw', ...). A plain random row split can leak -- different
     snapshots of the SAME sample landing in both train and test. This step
     measures that leakage, then re-evaluates XGBoost with a leakage-free,
     sample-grouped split across 5 seeds.

Run:
    python reproduce.py
    (place Obfuscated-MalMem2022.csv in this same folder first)
"""

import re
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

RAW_CSV = "Obfuscated-MalMem2022.csv"
DATA_DIR = "data"
MODEL_DIR = "model_artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# The paper's own reported numbers, Table 1
PAPER_REPORTED = {
    "XGBoost": 95.2,
    "Random Forest": 93.1,
    "SVM": 91.4,
    "K-Means": 84.6,
    "Naive Bayes": 76.9,
    "Logistic Regression": 74.1,
}


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------- Steps 1-2
def load_and_split():
    section("STEP 1-2: Load, clean, and split the data")
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows -> {len(df)} rows remain")

    # Confirm the dataset has no IP/port fields (the paper's Algorithm 1 groups
    # by source_ip/source_port -- fields that don't exist in this dataset).
    assert "source_ip" not in df.columns and "source_port" not in df.columns

    le = LabelEncoder()
    y = le.fit_transform(df["Class"])
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    X = df.drop(columns=["Category", "Class"])
    record_ids = df["Category"]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, record_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}  (documented 80/20 stratified split)")

    os.makedirs(DATA_DIR, exist_ok=True)
    X_train.assign(Class=y_train, record_id=id_train.values).to_csv(f"{DATA_DIR}/train.csv", index=False)
    X_test.assign(Class=y_test, record_id=id_test.values).to_csv(f"{DATA_DIR}/test.csv", index=False)
    joblib.dump(le, f"{DATA_DIR}/label_encoder.joblib")
    print(f"Saved {DATA_DIR}/train.csv, {DATA_DIR}/test.csv (used later by kafka_stream.py)")

    return df, X_train, X_test, y_train, y_test


# --------------------------------------------------------------- Step 3
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred) * 100
    rec = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) * 100 if (tn + fp) else 0.0
    print(f"{name:<20} Acc:{acc:6.2f}%  Prec:{prec:6.2f}%  Rec:{rec:6.2f}%  "
          f"Spec:{spec:6.2f}%  F1:{f1:6.2f}%  (TP:{tp} FN:{fn} FP:{fp} TN:{tn})")
    return acc


def train_and_compare(X_train, X_test, y_train, y_test):
    section("STEP 3: Train all 6 models from the paper, compare vs Table 1")
    os.makedirs(MODEL_DIR, exist_ok=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

    reproduced = {}

    xgb = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    reproduced["XGBoost"] = evaluate_model("XGBoost", y_test, xgb.predict(X_test))
    xgb.save_model(f"{MODEL_DIR}/xgb_ransomware.json")  # frozen artifact for the live pipeline

    rf = RandomForestClassifier(random_state=42, n_estimators=200)
    rf.fit(X_train, y_train)
    reproduced["Random Forest"] = evaluate_model("Random Forest", y_test, rf.predict(X_test))

    svm = SVC(kernel="rbf", random_state=42)
    svm.fit(X_train_s, y_train)
    reproduced["SVM"] = evaluate_model("SVM", y_test, svm.predict(X_test_s))

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    train_clusters = km.fit_predict(X_train_s)
    mapping = {c: np.bincount(y_train[train_clusters == c]).argmax() for c in [0, 1]}
    test_clusters = km.predict(X_test_s)
    km_preds = np.array([mapping[c] for c in test_clusters])
    reproduced["K-Means"] = evaluate_model("K-Means", y_test, km_preds)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    reproduced["Naive Bayes"] = evaluate_model("Naive Bayes", y_test, nb.predict(X_test))

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train_s, y_train)
    reproduced["Logistic Regression"] = evaluate_model("Logistic Regression", y_test, lr.predict(X_test_s))

    print("\n--- Comparison: Paper Reported vs Reproduced ---")
    rows = []
    for model_name, paper_acc in PAPER_REPORTED.items():
        repro_acc = reproduced[model_name]
        diff = repro_acc - paper_acc
        rows.append([model_name, paper_acc, round(repro_acc, 2), round(diff, 2)])
        print(f"{model_name:<20} Paper:{paper_acc:6.2f}%  Reproduced:{repro_acc:6.2f}%  Diff:{diff:+6.2f}%")

    pd.DataFrame(rows, columns=["Model", "Paper_Reported_%", "Reproduced_%", "Difference_%"]) \
      .to_csv("results_comparison.csv", index=False)
    print("\nSaved results_comparison.csv")
    print(f"Saved frozen model artifact: {MODEL_DIR}/xgb_ransomware.json  (used by spark_consumer.py)")


# --------------------------------------------------------------- Step 4
def hypothesis_tests(df):
    section("STEP 4: Hypothesis tests -- why doesn't 95.2% reproduce?")
    feature_cols = [c for c in df.columns if c not in ("Class", "Category")]
    X = df[feature_cols]
    y = LabelEncoder().fit_transform(df["Class"])

    print("-- Hypothesis A: does using fewer random features explain ~95%? --")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rng = np.random.default_rng(42)
    for k in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, len(feature_cols)]:
        n_trials = 5 if k < len(feature_cols) else 1
        accs = []
        for _ in range(n_trials):
            cols = rng.choice(feature_cols, size=k, replace=False)
            model = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
            model.fit(X_train[cols], y_train)
            accs.append(accuracy_score(y_test, model.predict(X_test[cols])) * 100)
        print(f"  Features={k:>3}  Mean Acc={np.mean(accs):6.2f}%  Min={np.min(accs):6.2f}%  Max={np.max(accs):6.2f}%")

    print("\n-- Hypothesis B: does a tiny/unlucky ~270-row test set explain ~95%? --")
    accs = []
    for seed in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=2000, test_size=270, random_state=seed, stratify=y
        )
        model = XGBClassifier(eval_metric="logloss", random_state=seed, verbosity=0)
        model.fit(X_train, y_train)
        accs.append(accuracy_score(y_test, model.predict(X_test)) * 100)
    print(f"  20 runs -> Mean={np.mean(accs):.2f}%  Min={np.min(accs):.2f}%  Max={np.max(accs):.2f}%  Std={np.std(accs):.2f}")

    print("\nConclusion: if neither hypothesis lands near 95.2%, the gap is most likely an "
          "undocumented feature subset or preprocessing step in the paper, not something "
          "explainable by feature count or test-set size alone.")


# --------------------------------------------------------------- Step 5
def base_sample(category, row_idx):
    """Strip the trailing '-N.raw' snapshot number to get the malware
    sample's group identity. Each Benign row gets its own unique group id
    (there's no shared 'sample' concept for benign snapshots)."""
    if category == "Benign":
        return f"Benign-{row_idx}"
    m = re.match(r"(.+)-\d+\.raw$", category)
    return m.group(1) if m else category


def leakage_check(df):
    section("STEP 5: Sample-level leakage check + leakage-free re-test")

    groups = pd.Series([base_sample(c, i) for i, c in enumerate(df["Category"])])
    X = df.drop(columns=["Category", "Class"])
    y = LabelEncoder().fit_transform(df["Class"])

    # Measure leakage in the SAME split style used in Step 1-2
    _, _, _, _, g_train, g_test = train_test_split(
        X, y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    train_ns = {g for g in g_train if not g.startswith("Benign-")}
    test_ns_list = [g for g in g_test if not g.startswith("Benign-")]
    leaked = train_ns & set(test_ns_list)
    leaked_rows = sum(1 for g in test_ns_list if g in leaked)
    print(f"Malware samples in both train and test: {len(leaked)}  "
          f"({leaked_rows}/{len(test_ns_list)} = {100 * leaked_rows / len(test_ns_list):.1f}% of malware test rows)")

    print("\nRe-testing with a LEAKAGE-FREE split (GroupShuffleSplit, 5 seeds):")
    accs = []
    for seed in range(5):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))
        assert len(set(groups.iloc[tr_idx]) & set(groups.iloc[te_idx])) == 0  # sanity check

        model = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        acc = accuracy_score(y[te_idx], model.predict(X.iloc[te_idx])) * 100
        print(f"  seed={seed}  Accuracy={acc:.2f}%")
        accs.append(acc)

    print(f"\nMean accuracy across 5 leakage-free splits: {np.mean(accs):.2f}%  (paper reported: 95.2%)")
    print("If this is still far above 95.2%, the leakage does NOT explain the gap -- the task "
          "is simply very separable on this dataset regardless of split strategy.")


def main():
    df, X_train, X_test, y_train, y_test = load_and_split()
    train_and_compare(X_train, X_test, y_train, y_test)
    hypothesis_tests(df)
    leakage_check(df)
    section("DONE")
    print("results_comparison.csv + everything printed above is your reproduction evidence.")


if __name__ == "__main__":
    main()

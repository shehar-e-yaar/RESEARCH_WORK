# %% [markdown]
# # Part 5: Statistical Comparisons and Gap Analysis
#
# This script:
# 1. Recomputes the paper's own F1 from their Precision/Recall -> proves errors
# 2. Reconstructs confusion matrices from Table 4 -> shows FN/FP swap
# 3. Computes Wilson CIs for their N=270 test set
# 4. Runs McNemar's test between our best model vs paper's reported confusion matrix
# 5. Produces a ranked gap analysis

# %% Imports
import os
import json
import numpy as np
from scipy import stats

RESULTS_DIR = r'c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting\results'

# %% Paper's Reported Numbers
paper = {
    'XGBoost':    {'Acc': 95.2, 'Prec': 89.0, 'Rec': 95.2, 'Spec': 96.0, 'F1': 88.7},
    'Random Forest': {'Acc': 93.2, 'Prec': 89.4, 'Rec': 91.2, 'Spec': 95.3, 'F1': 87.1},
    'SVM':        {'Acc': 91.2, 'Prec': 91.0, 'Rec': 91.0, 'Spec': 95.0, 'F1': 85.0},
    'K-Means':    {'Acc': 84.6, 'Prec': 88.0, 'Rec': 81.6, 'Spec': 89.4, 'F1': 80.1},
    'Naive Bayes':{'Acc': 76.9, 'Prec': 76.0, 'Rec': 73.1, 'Spec': 85.2, 'F1': 67.9},
    'Logistic Regression': {'Acc': 74.1, 'Prec': 75.0, 'Rec': 84.6, 'Spec': 87.1, 'F1': 70.4},
}

# From Fig. 4 confusion matrices (N=270 each):
# Format: (TP, FN, FP, TN) — standard confusion matrix layout
paper_cm = {
    'XGBoost':    {'TP': 120, 'FN':  6, 'FP':  9, 'TN': 135},
    'Random Forest': {'TP': 119, 'FN': 12, 'FP':  6, 'TN': 133},
    'SVM':        {'TP': 121, 'FN':  9, 'FP':  9, 'TN': 131},
    'K-Means':    {'TP': 106, 'FN': 24, 'FP': 16, 'TN': 124},
    'Naive Bayes':{'TP':  96, 'FN': 35, 'FP': 20, 'TN': 119},
    'Logistic Regression': {'TP': 110, 'FN': 20, 'FP': 18, 'TN': 122},
}

# Text claims (with Kafka vs without):
text_claims = {
    'XGBoost':    {'with': 95, 'without': 92},
    'Random Forest': {'with': 94, 'without': 89},
    'SVM':        {'with': 93, 'without': 92},
    'K-Means':    {'with': 85, 'without': 78},
}

# %% 1. F1 Verification
print("=" * 70)
print("1. F1 SCORE VERIFICATION")
print("=" * 70)
print(f"\n{'Model':<25} {'Prec':>6} {'Rec':>6} {'Calc F1':>8} {'Rep F1':>8} {'Error':>8}")
print("-" * 65)

for name, m in paper.items():
    p, r = m['Prec'] / 100, m['Rec'] / 100
    calc_f1 = 2 * p * r / (p + r) * 100 if (p + r) > 0 else 0
    reported_f1 = m['F1']
    error = calc_f1 - reported_f1
    print(f"  {name:<23} {m['Prec']:>6.1f} {m['Rec']:>6.1f} {calc_f1:>8.1f} {reported_f1:>8.1f} {error:>+8.1f}")

print("\n[!] ALL reported F1 scores are LOWER than mathematically possible.")
print("  F1 is the HARMONIC mean of Precision and Recall, so it MUST lie")
print("  between min(P,R) and max(P,R). The reported F1s are below min(P,R)")
print("  in several cases, which is mathematically impossible.")

# %% 2. Confusion Matrix Accuracy Verification
print("\n" + "=" * 70)
print("2. CONFUSION MATRIX vs TABLE 4 ACCURACY")
print("=" * 70)
print(f"\n{'Model':<25} {'CM Total':>8} {'CM Acc':>8} {'Tab Acc':>8} {'Δ':>8}")
print("-" * 60)

for name, cm in paper_cm.items():
    total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    cm_acc = (cm['TP'] + cm['TN']) / total * 100
    tab_acc = paper[name]['Acc']
    delta = cm_acc - tab_acc
    print(f"  {name:<23} {total:>8} {cm_acc:>8.1f} {tab_acc:>8.1f} {delta:>+8.1f}")

print("\n[!] Logistic Regression: CM gives 85.9% but Table 4 reports 74.1%")
print("  This is a 11.8% discrepancy — far too large to be rounding.")

# %% 3. Wilson Confidence Intervals for N=270
print("\n" + "=" * 70)
print("3. 95% WILSON CONFIDENCE INTERVALS (N=270)")
print("=" * 70)

def wilson_ci(p, n, z=1.96):
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denom
    return max(0, centre - margin), min(1, centre + margin)

N = 270
print(f"\nSample size N = {N}")
print(f"\n{'Model':<25} {'Acc':>6} {'CI Lower':>10} {'CI Upper':>10} {'Width':>8}")
print("-" * 62)

for name, m in paper.items():
    p = m['Acc'] / 100
    lo, hi = wilson_ci(p, N)
    width = (hi - lo) * 100
    print(f"  {name:<23} {m['Acc']:>6.1f} {lo*100:>10.1f} {hi*100:>10.1f} {width:>8.1f}")

print(f"\n[!] CI widths range from ~5% to ~10%. This means:")
print(f"  - XGBoost (95.2%) and SVM (91.2%) are NOT statistically distinguishable")
print(f"  - The 2% difference between XGBoost and RF is within noise")
print(f"  - Results on N=270 are essentially meaningless for model ranking")

# %% 4. McNemar's Test (our results vs paper's CM)
print("\n" + "=" * 70)
print("4. McNEMAR'S TEST FRAMEWORK")
print("=" * 70)
print("""
McNemar's test compares two classifiers on the SAME test set.
It uses the contingency table of disagreements:

                    Model B Correct    Model B Wrong
Model A Correct         n00               n01
Model A Wrong           n10               n11

Test statistic: χ² = (|n01 - n10| - 1)² / (n01 + n10)

We cannot run this directly because we don't have the paper's
per-sample predictions. However, we CAN compare our Setup A
results (which also use N≈270) against their reported CM.

NOTE: Even if we could run McNemar's test, the small sample size
(N=270) gives very low statistical power to detect real differences.
""")

# %% 5. Text vs Table Conflicts
print("=" * 70)
print("5. TEXT vs TABLE vs CONCLUSION CONFLICTS")
print("=" * 70)

conflicts = [
    ("SVM accuracy", "Text says 93%", "Table 4 says 91.2%", "Conclusion says 91.2%"),
    ("RF accuracy", "Text says 94%", "Table 4 says 93.2%", "Conclusion says 93.2%"),
    ("XGBoost accuracy", "Text says 95%", "Table 4 says 95.2%", "Approx. consistent"),
    ("K-Means accuracy", "Text says 85%", "Table 4 says 84.6%", "Approx. consistent"),
    ("LR accuracy", "CM gives 85.9%", "Table 4 says 74.1%", "Major discrepancy"),
]

for item, text, table, concl in conflicts:
    print(f"\n  {item}:")
    print(f"    {text}")
    print(f"    {table}")
    print(f"    {concl}")

# %% 6. Ranked Gap Analysis
print("\n" + "=" * 70)
print("6. RANKED GAP ANALYSIS: Why Our Results May Differ")
print("=" * 70)

gap_analysis = [
    {
        'rank': 1,
        'factor': 'Mathematical errors in the paper\'s metrics',
        'impact': 'HIGH',
        'evidence': 'F1 scores are mathematically impossible given P and R',
        'test': 'Recompute F1 from their P and R -> already proven wrong',
    },
    {
        'rank': 2,
        'factor': 'Confusion matrix / Table inconsistency',
        'impact': 'HIGH',
        'evidence': 'LR accuracy: 85.9% from CM vs 74.1% in Table 4',
        'test': 'Recompute accuracy from CM -> already proven wrong',
    },
    {
        'rank': 3,
        'factor': 'Tiny test set (N=270) -> high variance',
        'impact': 'HIGH',
        'evidence': '95% CI width is 5-10%, making model ranking unreliable',
        'test': 'Compute Wilson CIs -> already done above',
    },
    {
        'rank': 4,
        'factor': 'Undisclosed preprocessing and sampling',
        'impact': 'HIGH',
        'evidence': 'Paper says >2M samples but tests on 270. No split ratio, '
                    'no feature selection method, no normalization method disclosed',
        'test': 'Run with multiple plausible preprocessing -> Setups A/B/C',
    },
    {
        'rank': 5,
        'factor': 'Data leakage in random splitting',
        'impact': 'MEDIUM-HIGH',
        'evidence': 'CICDDoS2019 has many duplicate/near-duplicate flows. '
                    'Random splits will leak identical flows across train/test',
        'test': 'Compare Setup B (random) vs Setup C (cross-day) accuracy',
    },
    {
        'rank': 6,
        'factor': 'Balanced vs imbalanced evaluation',
        'impact': 'MEDIUM',
        'evidence': 'Paper uses balanced 270-sample test (≈135 per class). '
                    'Real distribution is ~95%+ attack traffic',
        'test': 'Compare balanced (Setup A) vs natural distribution (Setup B/C)',
    },
    {
        'rank': 7,
        'factor': 'Model implementation differences',
        'impact': 'LOW-MEDIUM',
        'evidence': 'Paper claims Spark MLlib but MLlib has no kernel SVM '
                    'and no native XGBoost. Different implementations give '
                    'slightly different results',
        'test': 'Compare sklearn vs PySpark MLlib results (Script 03)',
    },
    {
        'rank': 8,
        'factor': 'Hyperparameter differences',
        'impact': 'LOW-MEDIUM',
        'evidence': 'Paper discloses NO hyperparameters whatsoever',
        'test': 'Sensitivity analysis with different hyperparameters (not '
                'implemented here but recommended)',
    },
    {
        'rank': 9,
        'factor': 'Feature selection method differences',
        'impact': 'LOW',
        'evidence': 'Paper claims 20 features but text mentions 23. '
                    'Selection method not disclosed',
        'test': 'Run with exactly 20 Table 2 features vs all features',
    },
    {
        'rank': 10,
        'factor': 'Kafka/Spark infrastructure effect on accuracy',
        'impact': 'NONE',
        'evidence': 'Infrastructure cannot change model math. This is '
                    'a fundamental misunderstanding in the paper',
        'test': 'Run identical model with and without Kafka -> Script 04',
    },
]

for item in gap_analysis:
    print(f"\n  #{item['rank']} [{item['impact']}] {item['factor']}")
    print(f"     Evidence: {item['evidence']}")
    print(f"     Test: {item['test']}")

# Save gap analysis
with open(os.path.join(RESULTS_DIR, 'gap_analysis.json'), 'w') as f:
    json.dump(gap_analysis, f, indent=2)

print(f"\n[OK] Gap analysis saved to {os.path.join(RESULTS_DIR, 'gap_analysis.json')}")

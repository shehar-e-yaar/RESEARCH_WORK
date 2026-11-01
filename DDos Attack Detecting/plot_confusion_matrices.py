"""
Generate high-resolution, publication-grade confusion matrix heatmaps
for our research results (Setup C Cross-Day Benchmark and Literature Audit).
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure style
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# Setup C Real Data Counts (N = 173,454 flows from results/phase2_results.json)
# Format: [[TN, FP], [FN, TP]]
# Class 0: Benign (Normal Traffic)
# Class 1: DDoS Attack

# 1. Random Forest (Setup C - 98.70% Accuracy)
cm_rf = np.array([
    [42868, 2195],   # Benign: TN=42,868, FP=2,195
    [76, 128315]     # Attack: FN=76,     TP=128,315
])

# 2. XGBoost (Setup C - 98.50% Accuracy)
cm_xgb = np.array([
    [42546, 2517],   # Benign: TN=42,546, FP=2,517
    [82, 128309]     # Attack: FN=82,     TP=128,309
])

# 3. Ahmad et al. (IEEE Access 2026 Fig. 4 Audit - Discarded N = 270)
cm_paper_xgb = np.array([
    [135, 9],        # Benign: TN=135, FP=9
    [6, 120]         # Attack: FN=6,   TP=120
])

matrices = [
    (cm_rf, "Random Forest (Setup C - Cross-Day)", "98.70% Acc | 99.94% Recall | N = 173,454", "Blues"),
    (cm_xgb, "XGBoost (Setup C - Cross-Day)", "98.50% Acc | 99.94% Recall | N = 173,454", "Greens"),
    (cm_paper_xgb, "Ahmad et al. XGBoost (Published Fig. 4)", "94.44% Acc | 95.24% Recall | N = 270 Only!", "Oranges")
]

labels = ['Benign (0)', 'DDoS Attack (1)']

for idx, (cm, title, subtitle, cmap) in enumerate(matrices):
    ax = axes[idx]
    
    # Calculate percentages relative to true class totals
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Annotate with both Count and Percentage
    annot = np.empty_like(cm, dtype=object)
    for r in range(2):
        for c in range(2):
            count = cm[r, c]
            pct = cm_norm[r, c]
            metric_tag = ""
            if r == 0 and c == 0:
                metric_tag = "\n(True Negative)"
            elif r == 0 and c == 1:
                metric_tag = "\n(False Alarm - FP)"
            elif r == 1 and c == 0:
                metric_tag = "\n(Missed Attack - FN)"
            elif r == 1 and c == 1:
                metric_tag = "\n(Detected - TP)"
                
            annot[r, c] = f"{count:,}\n({pct:.2f}%){metric_tag}"

    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"fontsize": 11, "fontweight": "bold"},
                linewidths=1.5, linecolor='white')
    
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('True Ground Truth Label', fontsize=11, fontweight='bold', labelpad=8)

plt.tight_layout()
out_dir = r"c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting\results"
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "confusion_matrices_publication.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] High-resolution confusion matrices saved to: {out_png}")

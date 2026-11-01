import pandas as pd
import numpy as np
import os
import glob
import joblib
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)


PROCESSED_DATA_DIR = r"data**\p**rocessed"
MODEL_SAVE_DIR = r"models"
METRICS_DIR = r"metrics"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ============================================================
# Dataset groups
# ============================================================

TRAIN_FILES_PATTERN = [
    "01-12",
    "DrDoS",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday"
]

TEST_FILES_PATTERN = [
    "03-11",
    "DNS",
    "LDAP",
    "MSSQL",
    "NTP",
    "NetBIOS",
    "Portmap",
    "SNMP",
    "Syn",
    "TFTP",
    "UDP",
    "UDPLag"
]


# ============================================================
# MEMORY-EFFICIENT DATA LOADING
# ============================================================

def load_grouped_data(
    patterns,
    sample_size=None,
    exclude_patterns=None,
    chunksize=100_000
):
    """
    Load cleaned CSV files without loading entire huge files
    into RAM.

    If sample_size is provided:
        - CSV is read in chunks.
        - Each chunk is sampled.
        - Final result is limited to sample_size rows per file.

    This prevents MemoryError on machines with limited RAM.
    """

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    all_files = []

    for pattern in patterns:
        matches = glob.glob(
            os.path.join(
                PROCESSED_DATA_DIR,
                f"*{pattern}*_cleaned.csv"
            )
        )

        all_files.extend(matches)

    # --------------------------------------------------------
    # Exclude files if requested
    # --------------------------------------------------------

    if exclude_patterns:
        excluded = set()

        for pattern in exclude_patterns:
            matches = glob.glob(
                os.path.join(
                    PROCESSED_DATA_DIR,
                    f"*{pattern}*_cleaned.csv"
                )
            )

            excluded.update(matches)

        all_files = [
            f for f in all_files
            if f not in excluded
        ]

    # --------------------------------------------------------
    # Remove duplicate files
    # --------------------------------------------------------

    all_files = list(dict.fromkeys(all_files))

    if not all_files:
        return pd.DataFrame()

    print(f"Found {len(all_files)} files.")

    # --------------------------------------------------------
    # Read files
    # --------------------------------------------------------

    dfs = []

    for filepath in all_files:

        filename = os.path.basename(filepath)

        print(f"\nLoading: {filename}")

        # ====================================================
        # SAMPLE MODE
        # ====================================================

        if sample_size is not None:

            sampled_chunks = []

            for chunk in pd.read_csv(
                filepath,
                chunksize=chunksize,
                low_memory=False
            ):

                # If chunk contains more rows than required,
                # randomly sample from the chunk.
                if len(chunk) > sample_size:
                    chunk = chunk.sample(
                        n=sample_size,
                        random_state=42
                    )

                sampled_chunks.append(chunk)

            if not sampled_chunks:
                continue

            df = pd.concat(
                sampled_chunks,
                ignore_index=True
            )

            # Final limit for this file
            if len(df) > sample_size:
                df = df.sample(
                    n=sample_size,
                    random_state=42
                ).reset_index(drop=True)

            print(
                f"  Selected {len(df):,} rows "
                f"from this file."
            )

            dfs.append(df)

        # ====================================================
        # FULL MODE
        # ====================================================

        else:

            for chunk in pd.read_csv(
                filepath,
                chunksize=chunksize,
                low_memory=False
            ):
                dfs.append(chunk)

    # --------------------------------------------------------
    # Combine selected data
    # --------------------------------------------------------

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(
        dfs,
        ignore_index=True
    )

    print(
        f"\nCombined dataset shape: "
        f"{combined_df.shape}"
    )

    return combined_df


# ============================================================
# METRICS
# ============================================================

def calculate_detailed_metrics(
    model,
    X_test,
    y_test,
    model_prefix,
    stage_name
):

    print(
        f"\n--- {model_prefix} "
        f"({stage_name}) Evaluation ---"
    )

    # --------------------------------------------------------
    # Inference time
    # --------------------------------------------------------

    start_time = time.time()

    y_probs = model.predict_proba(X_test)[:, 1]

    y_pred = (
        y_probs >= 0.50
    ).astype(int)

    end_time = time.time()

    inference_time = (
        end_time - start_time
    ) / len(X_test)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    acc = accuracy_score(
        y_test,
        y_pred
    )

    prec = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    rec = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probs
    )

    pr_auc = average_precision_score(
        y_test,
        y_probs
    )

    metrics = {
        "model": model_prefix,
        "stage": stage_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "inference_time_per_sample": float(
            inference_time
        )
    }

    # --------------------------------------------------------
    # Save JSON metrics
    # --------------------------------------------------------

    json_path = os.path.join(
        METRICS_DIR,
        f"{model_prefix.lower()}_"
        f"{stage_name.lower()}_metrics.json"
    )

    with open(json_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 4)
    )

    sns.heatmap(
        [[tn, fp], [fn, tp]],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Normal",
            "Attack"
        ],
        yticklabels=[
            "Normal",
            "Attack"
        ]
    )

    plt.title(
        f"Conf Matrix: "
        f"{model_prefix} ({stage_name})"
    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    conf_path = os.path.join(
        METRICS_DIR,
        f"{model_prefix.lower()}_"
        f"{stage_name.lower()}_conf.png"
    )

    plt.savefig(conf_path)
    plt.close()

    print(
        f"Metrics saved to: {json_path}"
    )

    print(
        f"Accuracy:  {acc * 100:.2f}%"
    )

    print(
        f"Precision: {prec * 100:.2f}%"
    )

    print(
        f"Recall:    {rec * 100:.2f}%"
    )

    print(
        f"F1:        {f1 * 100:.2f}%"
    )

    print(
        f"FPR:       {fpr * 100:.2f}%"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "DDoS ATTACK DETECTION - ML TRAINING"
    )
    print("=" * 60)

    # ========================================================
    # 1. Load Training Data
    # ========================================================

    print(
        "\nStep 1: Loading Training Data "
        "(Mixed 2017 & 2019)..."
    )

    train_data_2017 = load_grouped_data(
        TRAIN_FILES_PATTERN,
        sample_size=40_000
    )

    # ========================================================
    # Load 2019 data
    # ========================================================

    data_2019 = load_grouped_data(
        TEST_FILES_PATTERN,
        sample_size=15_000,
        exclude_patterns=TRAIN_FILES_PATTERN
    )

    # ========================================================
    # Column mapping
    # ========================================================

    column_mapping = {

        "Fwd Packets Length Total":
            "Total Length of Fwd Packets",

        "Bwd Packets Length Total":
            "Total Length of Bwd Packets",

        "Packet Length Min":
            "Min Packet Length",

        "Packet Length Max":
            "Max Packet Length",

        "Avg Packet Size":
            "Average Packet Size",

        "Init Fwd Win Bytes":
            "Init_Win_bytes_forward",

        "Init Bwd Win Bytes":
            "Init_Win_bytes_backward",

        "Fwd Act Data Packets":
            "act_data_pkt_fwd",

        "Fwd Seg Size Min":
            "min_seg_size_forward"
    }

    data_2019.rename(
        columns=column_mapping,
        inplace=True
    )

    # ========================================================
    # Prepare train/test datasets
    # ========================================================

    if (
        not train_data_2017.empty
        and not data_2019.empty
    ):

        data_2019 = data_2019.reindex(
            columns=train_data_2017.columns,
            fill_value=0
        )

        strat = (
            data_2019["Label"]
            if (
                data_2019["Label"].nunique() > 1
                and
                data_2019["Label"]
                .value_counts()
                .min() >= 2
            )
            else None
        )

        test_size = min(
            10_000,
            int(len(data_2019) * 0.5)
        )

        train_data_2019, final_test_2019 = (
            train_test_split(
                data_2019,
                test_size=test_size,
                random_state=42,
                stratify=strat
            )
        )

        train_data = pd.concat(
            [
                train_data_2017,
                train_data_2019
            ],
            ignore_index=True
        )

    elif not train_data_2017.empty:

        strat = (
            train_data_2017["Label"]
            if (
                train_data_2017["Label"].nunique() > 1
                and
                train_data_2017["Label"]
                .value_counts()
                .min() >= 2
            )
            else None
        )

        train_data, final_test_2019 = (
            train_test_split(
                train_data_2017,
                test_size=0.2,
                random_state=42,
                stratify=strat
            )
        )

    elif not data_2019.empty:

        strat = (
            data_2019["Label"]
            if (
                data_2019["Label"].nunique() > 1
                and
                data_2019["Label"]
                .value_counts()
                .min() >= 2
            )
            else None
        )

        train_data, final_test_2019 = (
            train_test_split(
                data_2019,
                test_size=0.2,
                random_state=42,
                stratify=strat
            )
        )

    else:

        raise ValueError(
            f"No cleaned data found in "
            f"{PROCESSED_DATA_DIR}. "
            f"Please run preprocessing first."
        )

    # ========================================================
    # 2. Separate features and labels
    # ========================================================

    X = train_data.drop(
        columns=["Label"]
    )

    y = train_data["Label"].values

    train_columns = X.columns

    # Save column names
    joblib.dump(
        train_columns,
        os.path.join(
            MODEL_SAVE_DIR,
            "train_columns.pkl"
        )
    )

    # ========================================================
    # 3. Train / Validation Split
    # ========================================================

    strat_train = (
        y
        if (
            len(np.unique(y)) > 1
            and
            pd.Series(y)
            .value_counts()
            .min() >= 2
        )
        else None
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=strat_train
        )
    )

    print(
        f"\nTraining rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_val):,}"
    )

    # ========================================================
    # 4. Scaling
    # ========================================================

    print(
        "\nStep 2: Scaling features "
        "(Fitting only on X_train)..."
    )

    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    # Save scaler
    joblib.dump(
        scaler,
        os.path.join(
            MODEL_SAVE_DIR,
            "scaler.pkl"
        )
    )

    # ========================================================
    # 5. Random Forest
    # ========================================================

    print(
        "\nStep 3: Training Random Forest..."
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    rf_model.fit(
        X_train_scaled,
        y_train
    )

    joblib.dump(
        rf_model,
        os.path.join(
            MODEL_SAVE_DIR,
            "random_forest_model.pkl"
        )
    )

    calculate_detailed_metrics(
        rf_model,
        X_val_scaled,
        y_val,
        "RF",
        "Val"
    )

    # ========================================================
    # 6. XGBoost
    # ========================================================

    print(
        "\nStep 4: Training XGBoost..."
    )

    num_neg = (
        y_train == 0
    ).sum()

    num_pos = (
        y_train == 1
    ).sum()

    ratio = (
        num_neg / num_pos
        if num_pos > 0
        else 1.0
    )

    xgb_model = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=ratio,
        random_state=42
    )

    xgb_model.fit(
        X_train_scaled,
        y_train
    )

    joblib.dump(
        xgb_model,
        os.path.join(
            MODEL_SAVE_DIR,
            "xgboost_model.pkl"
        )
    )

    calculate_detailed_metrics(
        xgb_model,
        X_val_scaled,
        y_val,
        "XGB",
        "Val"
    )

    # ========================================================
    # 7. Final Test with CICDDoS2019
    # ========================================================

    print(
        "\nStep 5: Final Test with "
        "unseen CICDDoS2019 data..."
    )

    if not final_test_2019.empty:

        X_final_test = (
            final_test_2019
            .drop(columns=["Label"])
        )

        y_final_test = (
            final_test_2019["Label"]
            .values
        )

        X_final_test = (
            X_final_test.reindex(
                columns=train_columns,
                fill_value=0
            )
        )

        X_final_test_scaled = (
            scaler.transform(
                X_final_test
            )
        )

        calculate_detailed_metrics(
            rf_model,
            X_final_test_scaled,
            y_final_test,
            "RF",
            "Test2019"
        )

        calculate_detailed_metrics(
            xgb_model,
            X_final_test_scaled,
            y_final_test,
            "XGB",
            "Test2019"
        )

    print(
        "\n[SUCCESS] Detailed metrics, "
        "models, and scaler saved."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import os
import glob
import time
import json
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, recall_score, 
    precision_score, accuracy_score, roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PROCESSED_DATA_DIR = r"data\processed"
MODEL_SAVE_DIR = r"models"
METRICS_DIR = r"metrics"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

TRAIN_FILES_PATTERN = ["01-12", "DrDoS", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TEST_FILES_PATTERN = ["03-11", "DNS", "LDAP", "MSSQL", "NTP", "NetBIOS", "Portmap", "SNMP", "Syn", "TFTP", "UDP", "UDPLag"]

def load_grouped_data(patterns, sample_size=None, exclude_patterns=None):
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(PROCESSED_DATA_DIR, f"*{pattern}*_cleaned.csv")))
    
    if exclude_patterns:
        excluded = set()
        for p in exclude_patterns:
            excluded.update(glob.glob(os.path.join(PROCESSED_DATA_DIR, f"*{p}*_cleaned.csv")))
        all_files = [f for f in all_files if f not in excluded]

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    all_files = unique_files

    if not all_files:
        return pd.DataFrame()

    dfs = []
    for f in all_files:
        file_size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"Loading: {os.path.basename(f)} ({file_size_mb:.0f} MB)")
        
        # For large files: stream in chunks to avoid MemoryError on low-RAM systems
        if sample_size and file_size_mb > 200:
            rng = np.random.default_rng(42)
            est_rows = max(int((file_size_mb * 1024 * 1024) / 360), sample_size * 2)
            prob = min(1.0, (sample_size * 1.3) / est_rows)
            
            chunks = []
            for chunk in pd.read_csv(f, chunksize=100_000, low_memory=False):
                mask = rng.random(len(chunk)) < prob
                selected = chunk[mask]
                if len(selected) > 0:
                    chunks.append(selected)
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
            else:
                df = pd.DataFrame()
        else:
            df = pd.read_csv(f, low_memory=False)
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
        
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def calculate_detailed_metrics_cnn(model, X_test, y_test, stage_name):
    print(f"\n--- CNN ({stage_name}) Evaluation ---")
    
    start_time = time.time()
    y_probs = model.predict(X_test).flatten()
    end_time = time.time()
    inference_time = (end_time - start_time) / len(X_test)
    
    y_pred = (y_probs >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    metrics = {
        "model": "CNN",
        "stage": stage_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "inference_time_per_sample": float(inference_time)
    }

    # Save JSON metrics
    json_path = os.path.join(METRICS_DIR, f"cnn_{stage_name.lower()}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion Matrix Plot
    plt.figure(figsize=(6,4))
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'Conf Matrix: CNN ({stage_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(METRICS_DIR, f"cnn_{stage_name.lower()}_conf.png"))
    plt.close()

    return metrics

def main():
   # --- STEP 1: Mixed dataset for better accuracy ---
    print("Step 1: Loading Training Data (Mixed 2017 & 2019)...")
    
    # 1.A: 2017 / Day 1 Data is loading 
    train_data_2017 = load_grouped_data(TRAIN_FILES_PATTERN, sample_size=40000) 
    
    # 1.B: 2019 / Day 2 Data is loading 
    data_2019 = load_grouped_data(TEST_FILES_PATTERN, sample_size=15000, exclude_patterns=TRAIN_FILES_PATTERN)
    
    # Column mapping for backward compatibility if needed
    column_mapping = {
        'Fwd Packets Length Total': 'Total Length of Fwd Packets',
        'Bwd Packets Length Total': 'Total Length of Bwd Packets',
        'Packet Length Min': 'Min Packet Length',
        'Packet Length Max': 'Max Packet Length',
        'Avg Packet Size': 'Average Packet Size',
        'Init Fwd Win Bytes': 'Init_Win_bytes_forward',
        'Init Bwd Win Bytes': 'Init_Win_bytes_backward',
        'Fwd Act Data Packets': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward'
    }
    data_2019.rename(columns=column_mapping, inplace=True)
    
    if not train_data_2017.empty and not data_2019.empty:
        data_2019 = data_2019.reindex(columns=train_data_2017.columns, fill_value=0)
        strat = data_2019['Label'] if data_2019['Label'].nunique() > 1 and data_2019['Label'].value_counts().min() >= 2 else None
        test_size = min(10000, int(len(data_2019) * 0.5)) if len(data_2019) > 1 else 0.5
        train_data_2019, final_test_2019 = train_test_split(data_2019, test_size=test_size, random_state=42, stratify=strat)
        train_data = pd.concat([train_data_2017, train_data_2019], ignore_index=True)
    elif not train_data_2017.empty:
        strat = train_data_2017['Label'] if train_data_2017['Label'].nunique() > 1 and train_data_2017['Label'].value_counts().min() >= 2 else None
        train_data, final_test_2019 = train_test_split(train_data_2017, test_size=0.2, random_state=42, stratify=strat)
    elif not data_2019.empty:
        strat = data_2019['Label'] if data_2019['Label'].nunique() > 1 and data_2019['Label'].value_counts().min() >= 2 else None
        train_data, final_test_2019 = train_test_split(data_2019, test_size=0.2, random_state=42, stratify=strat)
    else:
        raise ValueError(f"No cleaned data found in {PROCESSED_DATA_DIR}. Please run preprocessing first.")
    
    # Saved columns for model recognition
    X = train_data.drop(columns=['Label'])
    y = train_data['Label'].values
    train_columns = X.columns
    joblib.dump(train_columns, os.path.join(MODEL_SAVE_DIR, "train_columns.pkl"))
    # --------------------------------------------------------

    # 2. Split
    strat_train = y if len(np.unique(y)) > 1 and pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat_train)

    # 3. Scaling 
    print("Step 2: Scaling features for CNN (Fit ONLY on X_train)...")
    scaler = MinMaxScaler() 
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler_cnn.pkl"))

    # 4. Reshape for CNN
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)

    # 5. Build and Train
    print("\nStep 3: Building and Training 1D-CNN (with Early Stopping)...")
    input_shape = (X_train_reshaped.shape[1], 1)
    model = build_cnn(input_shape)

    counts = np.bincount(y_train)
    if len(counts) > 1 and counts[0] > 0 and counts[1] > 0:
        class_weights = {0: (1.0/counts[0])*(len(y_train)/2.0), 1: (1.0/counts[1])*(len(y_train)/2.0)}
    else:
        class_weights = None

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train_reshaped, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val_reshaped, y_val),
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # Save
    model.save(os.path.join(MODEL_SAVE_DIR, "cnn_model.h5"))
    with open(os.path.join(METRICS_DIR, "cnn_history.json"), "w") as f:
        json.dump(history.history, f)

    # Plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss'); plt.title('CNN Loss'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history.history['accuracy'], label='Train Acc'); plt.plot(history.history['val_accuracy'], label='Val Acc'); plt.title('CNN Accuracy'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(METRICS_DIR, "cnn_training_plots.png")); plt.close()

    # Evaluation
    calculate_detailed_metrics_cnn(model, X_val_reshaped, y_val, "Val")

    # 6. Final Test with CICDDoS2019 (Alignment & Scaling)
    print("\nStep 4: Final Test with unseen CICDDoS2019 data (using pre-split data to avoid leakage)...")
    test_data = final_test_2019
    
    if not test_data.empty:
        X_final_test = test_data.drop(columns=['Label'])
        y_final_test = test_data['Label'].values
        X_final_test = X_final_test.reindex(columns=train_columns, fill_value=0)
        
        # SCALING (Already aligned in Step 1)
        X_final_test_scaled = scaler.transform(X_final_test)
        
        # RESHAPE
        X_final_test_reshaped = X_final_test_scaled.reshape(X_final_test_scaled.shape[0], X_final_test_scaled.shape[1], 1)
        
        calculate_detailed_metrics_cnn(model, X_final_test_reshaped, y_final_test, "Test2019")

    print("\n[SUCCESS] CNN restored and fixed.")

if __name__ == "__main__":
    main()

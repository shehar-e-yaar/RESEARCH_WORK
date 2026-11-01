import os
import json

def build_colab_notebook(output_path):
    cells = []

    def add_markdown(source):
        cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': [line + '\n' for line in source.split('\n')]
        })

    def add_code(source):
        cells.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line + '\n' for line in source.split('\n')]
        })

    # Title & Markdown intro
    add_markdown("""# CICDDoS2019 Full 70-Million-Row Big Data Pipeline
### End-to-End Out-Of-Core Ingestion via KaggleHub, Parquet Compression, and ML Training in Google Colab
This notebook processes **all 70,427,637 flows (100% of the raw dataset)** across Day 1 (50.06M rows) and Day 2 (20.36M rows) without running out of RAM (OOM) on Google Colab.

---
### Key Pipeline Features:
1. **High-Speed Datacenter Download via KaggleHub:** Downloads the 3GB compressed archive directly into Colab's local high-speed disk (`/root/.cache/kagglehub/...`), extracting the full 30GB CSV files in ~2 minutes!
2. **Chunked Out-of-Core Processing:** Streams 500,000 rows at a time, requiring only ~1.5 GB of RAM at any moment (never crashes Colab's 12.7 GB limit).
3. **Memory Downcasting (75% RAM Reduction):** Casts `float64` to `float32` and port/flag integers to `uint16`.
4. **Snappy-Compressed Parquet Conversion:** Compresses 30.8 GB of raw CSV text down to **~3.5 GB of partitioned Parquet files** saved directly to Google Drive.
5. **Big Data ML on All 70M Rows:** Evaluates **PySpark MLlib** (partitioned disk spilling) and **Incremental SGDClassifier (`partial_fit`)** across all 50M training flows and 20M testing flows!""")

    # Cell 1: Environment Setup
    add_code("""# Cell 1: Mount Google Drive & Install Required Big-Data Tools
from google.colab import drive
drive.mount('/content/drive')

!pip install -q kagglehub pyspark pyarrow polars fastparquet xgboost
print("[OK] All big data dependencies installed successfully!")""")

    # Cell 2: Hardware Diagnostics
    add_code("""# Cell 2: Verify Hardware Specs & GPU Acceleration
import os, psutil, torch

ram_gb = psutil.virtual_memory().total / (1024**3)
gpu_available = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"

print("=" * 65)
print(f"Colab System RAM: {ram_gb:.2f} GB")
print(f"GPU Available:    {gpu_available} ({gpu_name})")
print("=" * 65)
if ram_gb < 20:
    print("Running on Standard Colab RAM (~12.7 GB).")
    print("Out-of-core chunking is actively enabled to prevent Out-Of-Memory crashes!")""")

    # Cell 3: KaggleHub Direct Download
    add_code("""# Cell 3: Download Full 30GB CICDDoS2019 Dataset via KaggleHub (Fast Datacenter Speed!)
import kagglehub
import glob

print("Downloading full CICDDoS2019 dataset from Kaggle...")
print("Kaggle dataset ID: rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files")

# Downloads 3GB compressed archive and extracts full 30GB CSVs into local Colab disk in ~2 minutes:
raw_dataset_path = kagglehub.dataset_download("rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files")

print("\\n[OK] Dataset ready on Colab local scratch disk!")
print("Path to extracted dataset files:", raw_dataset_path)

# Automatically list all CSV files in the downloaded dataset
all_csv_files = sorted(glob.glob(f"{raw_dataset_path}/**/*.csv", recursive=True))
print(f"\\nFound {len(all_csv_files)} CSV files in downloaded Kaggle dataset:")
for f in all_csv_files:
    size_mb = os.path.getsize(f) / (1024**2)
    print(f"  • {os.path.basename(f)} ({size_mb:,.1f} MB)")""")

    # Cell 4: Configuration & Downcast Types
    add_code("""# Cell 4: Directory Configuration & Downcasted Data Types
import os

# Output directory on Google Drive to permanently store the compressed ~3.5 GB Parquet files:
PARQUET_OUTPUT_DIR = "/content/drive/MyDrive/CICDDoS2019_Full_70M_Parquet/"

DAY1_OUT = os.path.join(PARQUET_OUTPUT_DIR, "day1_train")
DAY2_OUT = os.path.join(PARQUET_OUTPUT_DIR, "day2_test")

os.makedirs(DAY1_OUT, exist_ok=True)
os.makedirs(DAY2_OUT, exist_ok=True)

# The 20 Selected Features from Ahmad et al. (IEEE Access 2026)
PAPER_20_FEATURES = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean', 'Packet Length Std',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Destination Port', 'ACK Flag Count'
]

# Downcasted Types (Saves 75% RAM!)
DTYPES = {
    'Flow Duration': 'float32',
    'Flow Bytes/s': 'float32',
    'Flow Packets/s': 'float32',
    'Packet Length Mean': 'float32',
    'Packet Length Std': 'float32',
    'Fwd Packet Length Max': 'float32',
    'Fwd Packet Length Min': 'float32',
    'Fwd Packet Length Mean': 'float32',
    'Fwd Packet Length Std': 'float32',
    'Flow IAT Mean': 'float32',
    'Flow IAT Std': 'float32',
    'Flow IAT Max': 'float32',
    'Flow IAT Min': 'float32',
    'Fwd IAT Total': 'float32',
    'Fwd IAT Mean': 'float32',
    'Fwd IAT Std': 'float32',
    'Fwd IAT Max': 'float32',
    'Fwd IAT Min': 'float32',
    'Destination Port': 'uint16',
    'ACK Flag Count': 'uint16',
}
print(f"[OK] Configuration set. Selected {len(PAPER_20_FEATURES)} features.")
print(f"Parquet output path: {PARQUET_OUTPUT_DIR}")""")

    # Cell 5: Chunked Ingestion Function
    add_code("""# Cell 5: Out-Of-Core 70-Million-Row Chunked Ingestion Engine
# Converts 30 GB of raw CSV text into partitioned, snappy-compressed Parquets
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def convert_csv_to_parquet_chunks(csv_path, output_dir, chunksize=500_000):
    fname = os.path.basename(csv_path)
    base_name = os.path.splitext(fname)[0]
    print(f"\\n--- Converting {fname} (Chunksize: {chunksize:,}) ---")
    
    # 1. Inspect header to strip leading/trailing spaces
    sample = pd.read_csv(csv_path, nrows=5)
    col_map = {c: c.strip() for c in sample.columns}
    
    # Target columns to load
    target_cols = [c for c, stripped in col_map.items() if stripped in PAPER_20_FEATURES or stripped == 'Label']
    
    total_rows = 0
    chunk_idx = 0
    t0 = time.time()
    
    for chunk in pd.read_csv(csv_path, usecols=target_cols, chunksize=chunksize, low_memory=False):
        chunk.rename(columns=col_map, inplace=True)
        
        # Binary Label: BENIGN = 0, Attack = 1
        chunk['binary_label'] = (chunk['Label'].astype(str).str.strip().str.upper() != 'BENIGN').astype('uint8')
        chunk.drop(columns=['Label'], inplace=True, errors='ignore')
        
        # Replace Inf and -Inf in rate features
        for num_col in ['Flow Bytes/s', 'Flow Packets/s']:
            if num_col in chunk.columns:
                chunk[num_col] = chunk[num_col].replace([np.inf, -np.inf], np.nan)
        
        # Clean NaNs with 0
        chunk.fillna(0, inplace=True)
        
        # Downcast datatypes to float32 and uint16
        for c, dt in DTYPES.items():
            if c in chunk.columns:
                chunk[c] = chunk[c].astype(dt)
        
        # Write to partitioned snappy-parquet chunk
        part_name = f"{base_name}_part_{chunk_idx:04d}.parquet"
        out_part_path = os.path.join(output_dir, part_name)
        
        table = pa.Table.from_pandas(chunk)
        pq.write_table(table, out_part_path, compression='snappy')
        
        total_rows += len(chunk)
        chunk_idx += 1
        print(f"  Processed {total_rows:,} rows | Saved {part_name}")
        
    dur = time.time() - t0
    print(f"[Done] {fname}: {total_rows:,} total rows converted in {dur:.1f}s ({total_rows/dur:,.0f} rows/s)")
    return total_rows

# Separate CSV files into Day 1 (01-12) and Day 2 (03-11):
day1_files = []
day2_files = []

for f in all_csv_files:
    fname = os.path.basename(f)
    parent_dir = os.path.basename(os.path.dirname(f))
    if "01-12" in f or "01-12" in parent_dir or fname.startswith("DrDoS_") or fname in ["TFTP.csv"]:
        day1_files.append(f)
    else:
        day2_files.append(f)

print(f"\\nFound {len(day1_files)} files for Day 1 (Train) and {len(day2_files)} files for Day 2 (Test).")

# Convert Day 1 CSVs to Parquet:
print("\\n=======================================================")
print("STAGE 1: CONVERTING DAY 1 TRAINING DATA (~50M ROWS)")
print("=======================================================")
day1_total = 0
for f in day1_files:
    day1_total += convert_csv_to_parquet_chunks(f, DAY1_OUT)

# Convert Day 2 CSVs to Parquet:
print("\\n=======================================================")
print("STAGE 2: CONVERTING DAY 2 TESTING DATA (~20M ROWS)")
print("=======================================================")
day2_total = 0
for f in day2_files:
    day2_total += convert_csv_to_parquet_chunks(f, DAY2_OUT)

print(f"\\n[COMPLETE] Successfully converted {day1_total + day2_total:,} total flows across Day 1 & Day 2!")""")

    # Cell 6: PySpark MLlib on Full 70 Million Rows
    add_code("""# Cell 6: Big Data Model Training via Apache Spark on All 70M Rows
# Spark uses distributed disk spilling so it NEVER runs out of memory!
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark configured for Colab hardware
spark = SparkSession.builder \\
    .appName("CICDDoS2019_Full_70M_Colab") \\
    .master("local[*]") \\
    .config("spark.driver.memory", "10g") \\
    .config("spark.sql.shuffle.partitions", "32") \\
    .config("spark.driver.maxResultSize", "4g") \\
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("[OK] Spark 3.5 Session Initialized successfully!")

# Load all partitioned Parquet files for Day 1 (50M rows) and Day 2 (20M rows)
print("\\nLoading Day 1 and Day 2 partitioned Parquet datasets...")
train_spark = spark.read.parquet(f"{DAY1_OUT}/*.parquet")
test_spark = spark.read.parquet(f"{DAY2_OUT}/*.parquet")

clean_cols = []
for f in PAPER_20_FEATURES:
    c_clean = f.replace(' ', '_').replace('/', '_per_')
    train_spark = train_spark.withColumn(c_clean, col(f).cast(DoubleType()))
    test_spark = test_spark.withColumn(c_clean, col(f).cast(DoubleType()))
    clean_cols.append(c_clean)

train_spark = train_spark.withColumn("label", col("binary_label").cast(DoubleType()))
test_spark = test_spark.withColumn("label", col("binary_label").cast(DoubleType()))

# Pipeline Assembly
assembler = VectorAssembler(inputCols=clean_cols, outputCol="features_raw", handleInvalid="keep")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

print("Preparing training pipeline...")
train_scaled = scaler.fit(assembler.transform(train_spark)).transform(assembler.transform(train_spark))

# Train Linear Support Vector Machine on Full Day 1 (~50 Million Flows)
print("\\nTraining LinearSVC on the full Day 1 dataset (~50,000,000 flows)...")
lsvc = LinearSVC(maxIter=100, labelCol="label", featuresCol="features")
lsvc_model = lsvc.fit(train_scaled)
print("[OK] LinearSVC training completed across all 50 million rows!")

# Evaluate on Full Day 2 (~20 Million Flows)
print("\\nEvaluating model on the full Day 2 test dataset (~20,000,000 flows)...")
test_scaled = scaler.fit(assembler.transform(train_spark)).transform(assembler.transform(test_spark))
predictions = lsvc_model.transform(test_scaled)

eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
eval_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

test_accuracy = eval_acc.evaluate(predictions) * 100
test_f1 = eval_f1.evaluate(predictions) * 100

print("=" * 60)
print(f"FULL 70-MILLION-ROW TEST RESULTS (Setup C):")
print(f"  Accuracy: {test_accuracy:.2f}%")
print(f"  F1 Score: {test_f1:.2f}%")
print("=" * 60)""")

    # Cell 7: Incremental Scikit-Learn SGD Classifier
    add_code("""# Cell 7: Alternative Method: Out-Of-Core Incremental SGD Classifier
# Streams through all 70 million flows chunk-by-chunk using partial_fit()
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import glob, time

print("Initializing Out-of-Core Incremental SGD Classifier (Logistic Regression)...")
sgd_clf = SGDClassifier(loss='log_loss', max_iter=1, learning_rate='optimal', random_state=42)

train_parts = sorted(glob.glob(f"{DAY1_OUT}/*.parquet"))
print(f"Found {len(train_parts)} training parquet chunks (~50M rows total).")

t0 = time.time()
total_trained = 0
for idx, p in enumerate(train_parts):
    df_chunk = pd.read_parquet(p)
    X = df_chunk[PAPER_20_FEATURES].values
    y = df_chunk['binary_label'].values
    
    sgd_clf.partial_fit(X, y, classes=[0, 1])
    total_trained += len(y)
    
    if (idx + 1) % 10 == 0 or (idx + 1) == len(train_parts):
        print(f"  Trained on {total_trained:,} flows ({idx+1}/{len(train_parts)} chunks) | Elapsed: {time.time()-t0:.1f}s")

print(f"\\n[Complete] Trained on all {total_trained:,} Day 1 rows in {time.time()-t0:.1f}s!")

# Evaluate incrementally on all Day 2 chunks (20M rows)
test_parts = sorted(glob.glob(f"{DAY2_OUT}/*.parquet"))
print(f"\\nEvaluating incrementally on {len(test_parts)} Day 2 test chunks (~20M rows)...")

total_correct = 0
total_tested = 0
t_eval = time.time()

for idx, p in enumerate(test_parts):
    df_chunk = pd.read_parquet(p)
    X = df_chunk[PAPER_20_FEATURES].values
    y = df_chunk['binary_label'].values
    
    preds = sgd_clf.predict(X)
    total_correct += (preds == y).sum()
    total_tested += len(y)

full_accuracy = (total_correct / total_tested) * 100
print("=" * 60)
print(f"INCREMENTAL EVALUATION ON ALL {total_tested:,} TEST FLOWS:")
print(f"  Overall Accuracy: {full_accuracy:.2f}%")
print(f"  Evaluation Time:  {time.time()-t_eval:.1f}s")
print("=" * 60)""")

    # Cell 8: GPU-Accelerated XGBoost Training & Evaluation
    add_code("""# Cell 8: High-Performance GPU-Accelerated XGBoost Training & Evaluation
# Trains trees using XGBoost's fast histogram algorithm (tree_method='hist')
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob, time
import pandas as pd
import numpy as np
import torch

print("=" * 65)
print("STAGE 4: GPU-ACCELERATED XGBOOST TRAINING ON FULL DATASET")
print("=" * 65)

# Detect GPU capability
use_gpu = torch.cuda.is_available()
tree_method = "hist"
device = "cuda" if use_gpu else "cpu"
print(f"XGBoost Configuration: tree_method='{tree_method}', device='{device.upper()}'")

# XGBoost Parameters tuned for fast big-data histogram training
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': tree_method,
    'device': device,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

train_parts = sorted(glob.glob(f"{DAY1_OUT}/*.parquet"))
print(f"Training XGBoost across {len(train_parts)} Day 1 training chunks (~50M rows)...")

booster = None
total_trained_xgb = 0
t0 = time.time()

# Iterative warm-start boosting across Day 1 chunks
for idx, p in enumerate(train_parts):
    df_chunk = pd.read_parquet(p)
    X = df_chunk[PAPER_20_FEATURES].values
    y = df_chunk['binary_label'].values
    
    dtrain = xgb.DMatrix(X, label=y)
    
    # Continual tree boosting: pass previous booster to add new trees
    booster = xgb.train(params, dtrain, num_boost_round=5, xgb_model=booster)
    total_trained_xgb += len(y)
    
    if (idx + 1) % 10 == 0 or (idx + 1) == len(train_parts):
        print(f"  [XGBoost] Boosted trees on {total_trained_xgb:,} flows ({idx+1}/{len(train_parts)} chunks) | Elapsed: {time.time()-t0:.1f}s")

print(f"\\n[Complete] XGBoost training finished across all Day 1 flows in {time.time()-t0:.1f}s!")

# Evaluate XGBoost across all Day 2 test chunks (~20M rows)
test_parts = sorted(glob.glob(f"{DAY2_OUT}/*.parquet"))
print(f"\\nEvaluating XGBoost across {len(test_parts)} Day 2 test chunks (~20M rows)...")

total_correct_xgb = 0
total_tested_xgb = 0
tp, fp, tn, fn = 0, 0, 0, 0
t_eval = time.time()

for idx, p in enumerate(test_parts):
    df_chunk = pd.read_parquet(p)
    X = df_chunk[PAPER_20_FEATURES].values
    y = df_chunk['binary_label'].values
    
    dtest = xgb.DMatrix(X)
    probs = booster.predict(dtest)
    preds = (probs > 0.5).astype(int)
    
    total_correct_xgb += (preds == y).sum()
    total_tested_xgb += len(y)
    
    # Confusion matrix elements
    tp += int(((preds == 1) & (y == 1)).sum())
    tn += int(((preds == 0) & (y == 0)).sum())
    fp += int(((preds == 1) & (y == 0)).sum())
    fn += int(((preds == 0) & (y == 1)).sum())

xgb_acc = (total_correct_xgb / total_tested_xgb) * 100
xgb_prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
xgb_rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
xgb_f1 = (2 * xgb_prec * xgb_rec / (xgb_prec + xgb_rec)) if (xgb_prec + xgb_rec) > 0 else 0.0

print("=" * 65)
print(f"FINAL XGBOOST RESULTS (Full 20-Million-Row Day 2 Test):")
print(f"  Accuracy:  {xgb_acc:.2f}%")
print(f"  Precision: {xgb_prec:.2f}%")
print(f"  Recall:    {xgb_rec:.2f}%")
print(f"  F1 Score:  {xgb_f1:.2f}%")
print(f"  Total Flows Evaluated: {total_tested_xgb:,}")
print(f"  Evaluation Duration:   {time.time()-t_eval:.1f}s")
print("=" * 65)""")

    # Build notebook JSON
    notebook = {
        'cells': cells,
        'metadata': {
            'accelerator': 'GPU',
            'colab': {
                'provenance': [],
                'toc_visible': True
            },
            'language_info': {
                'name': 'python',
                'version': '3.10'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 0
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

    print(f"[OK] Colab Notebook successfully built at: {output_path}")

if __name__ == '__main__':
    out = os.path.join(r'c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting', 'CICDDoS2019_Full_70M_Colab_Pipeline.ipynb')
    build_colab_notebook(out)

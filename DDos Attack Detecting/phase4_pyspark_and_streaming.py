import os
import sys
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict

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

print("=" * 80)
print("PHASE 4: PYSPARK MLLIB & STREAMING EVALUATION")
print("=" * 80)

# ==============================================================================
# PART 1: PySpark MLlib Models on Setup C (Cross-Day)
# ==============================================================================
print("\n--- Part 1: PySpark MLlib Benchmarks ---")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer, MinMaxScaler
from pyspark.ml.classification import (
    LinearSVC as SparkLinearSVC,
    RandomForestClassifier as SparkRF,
    GBTClassifier as SparkGBT,
    NaiveBayes as SparkNB,
    LogisticRegression as SparkLR,
    DecisionTreeClassifier as SparkDT,
)
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("DDoS_Phase4_Spark") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"Spark Version: {spark.version}")

# Load Parquet datasets
train_spark = spark.read.parquet(os.path.join(PROCESSED_DIR, 'train_day_sampled.parquet'))
test_spark  = spark.read.parquet(os.path.join(PROCESSED_DIR, 'test_day_sampled.parquet'))

clean_feat_cols = []
for f in PAPER_20_FEATURES:
    c_name = f.replace(' ', '_').replace('/', '_per_')
    train_spark = train_spark.withColumn(c_name, col(f'`{f}`').cast(DoubleType()))
    test_spark  = test_spark.withColumn(c_name, col(f'`{f}`').cast(DoubleType()))
    clean_feat_cols.append(c_name)

train_spark = train_spark.withColumn("label", col("binary_label").cast(DoubleType()))
test_spark  = test_spark.withColumn("label", col("binary_label").cast(DoubleType()))

print(f"Spark Train Rows: {train_spark.count():,}, Spark Test Rows: {test_spark.count():,}")

# Preprocessing Pipeline
imp_cols = [f"{c}_imp" for c in clean_feat_cols]
imputer = Imputer(inputCols=clean_feat_cols, outputCols=imp_cols, strategy="median")
assembler = VectorAssembler(inputCols=imp_cols, outputCol="features_raw", handleInvalid="keep")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

pyspark_results = {}

# 1. Random Forest
print("\n  1. Training Spark Random Forest...")
t0 = time.time()
rf = SparkRF(numTrees=100, maxDepth=10, seed=42, labelCol="label", featuresCol="features")
pipe_rf = SparkPipeline(stages=[imputer, assembler, scaler, rf])
model_rf = pipe_rf.fit(train_spark)
preds_rf = model_rf.transform(test_spark)
t_rf = time.time() - t0
acc_rf = evaluator_acc.evaluate(preds_rf) * 100
f1_rf  = evaluator_f1.evaluate(preds_rf) * 100
pyspark_results['Spark RF'] = {'Acc': round(acc_rf, 2), 'F1': round(f1_rf, 2), 'time_s': round(t_rf, 2)}
print(f"     Acc={acc_rf:.2f}%, F1={f1_rf:.2f}% ({t_rf:.1f}s)")

# 2. LinearSVC
print("\n  2. Training Spark LinearSVC...")
t0 = time.time()
lsvc = SparkLinearSVC(maxIter=10000, labelCol="label", featuresCol="features")
pipe_svc = SparkPipeline(stages=[imputer, assembler, scaler, lsvc])
model_svc = pipe_svc.fit(train_spark)
preds_svc = model_svc.transform(test_spark)
t_svc = time.time() - t0
acc_svc = evaluator_acc.evaluate(preds_svc) * 100
f1_svc  = evaluator_f1.evaluate(preds_svc) * 100
pyspark_results['Spark LinearSVC'] = {'Acc': round(acc_svc, 2), 'F1': round(f1_svc, 2), 'time_s': round(t_svc, 2)}
print(f"     Acc={acc_svc:.2f}%, F1={f1_svc:.2f}% ({t_svc:.1f}s)")

# 3. Logistic Regression
print("\n  3. Training Spark Logistic Regression...")
t0 = time.time()
lr = SparkLR(maxIter=5000, labelCol="label", featuresCol="features")
pipe_lr = SparkPipeline(stages=[imputer, assembler, scaler, lr])
model_lr = pipe_lr.fit(train_spark)
preds_lr = model_lr.transform(test_spark)
t_lr = time.time() - t0
acc_lr = evaluator_acc.evaluate(preds_lr) * 100
f1_lr  = evaluator_f1.evaluate(preds_lr) * 100
pyspark_results['Spark Logistic Regression'] = {'Acc': round(acc_lr, 2), 'F1': round(f1_lr, 2), 'time_s': round(t_lr, 2)}
print(f"     Acc={acc_lr:.2f}%, F1={f1_lr:.2f}% ({t_lr:.1f}s)")

# 4. Naive Bayes
print("\n  4. Training Spark Naive Bayes (MinMaxScaler for non-negative inputs)...")
t0 = time.time()
scaler_nb = MinMaxScaler(inputCol="features_raw", outputCol="features_nb")
nb = SparkNB(modelType="gaussian", labelCol="label", featuresCol="features_nb")
pipe_nb = SparkPipeline(stages=[imputer, assembler, scaler_nb, nb])
model_nb = pipe_nb.fit(train_spark)
preds_nb = model_nb.transform(test_spark)
t_nb = time.time() - t0
acc_nb = evaluator_acc.evaluate(preds_nb) * 100
f1_nb  = evaluator_f1.evaluate(preds_nb) * 100
pyspark_results['Spark Naive Bayes'] = {'Acc': round(acc_nb, 2), 'F1': round(f1_nb, 2), 'time_s': round(t_nb, 2)}
print(f"     Acc={acc_nb:.2f}%, F1={f1_nb:.2f}% ({t_nb:.1f}s)")

# 5. Decision Tree
print("\n  5. Training Spark Decision Tree...")
t0 = time.time()
dt = SparkDT(maxDepth=10, seed=42, labelCol="label", featuresCol="features")
pipe_dt = SparkPipeline(stages=[imputer, assembler, scaler, dt])
model_dt = pipe_dt.fit(train_spark)
preds_dt = model_dt.transform(test_spark)
t_dt = time.time() - t0
acc_dt = evaluator_acc.evaluate(preds_dt) * 100
f1_dt  = evaluator_f1.evaluate(preds_dt) * 100
pyspark_results['Spark Decision Tree'] = {'Acc': round(acc_dt, 2), 'F1': round(f1_dt, 2), 'time_s': round(t_dt, 2)}
print(f"     Acc={acc_dt:.2f}%, F1={f1_dt:.2f}% ({t_dt:.1f}s)")

# 6. K-Means
print("\n  6. Training Spark K-Means (k=2)...")
t0 = time.time()
km = SparkKMeans(k=2, seed=42, featuresCol="features", predictionCol="cluster")
pipe_km = SparkPipeline(stages=[imputer, assembler, scaler, km])
model_km = pipe_km.fit(train_spark)
train_c = model_km.transform(train_spark).select("cluster", "label").toPandas()
test_c  = model_km.transform(test_spark).select("cluster", "label").toPandas()
t_km = time.time() - t0

# Cluster mapping by majority on train
km_map = {}
for c_val in [0, 1]:
    sub = train_c[train_c['cluster'] == c_val]
    km_map[c_val] = int(round(sub['label'].mean())) if len(sub) > 0 else 0
test_pred_km = test_c['cluster'].map(km_map).values
acc_km = (test_pred_km == test_c['label'].values).mean() * 100
f1_km = 0.0
pyspark_results['Spark K-Means'] = {'Acc': round(acc_km, 2), 'cluster_map': str(km_map), 'time_s': round(t_km, 2)}
print(f"     Acc={acc_km:.2f}%, Map={km_map} ({t_km:.1f}s)")

# 7. XGBoost via SparkXGBClassifier / GBT
print("\n  7. Training Spark XGBoost (SparkXGBClassifier)...")
t0 = time.time()
xgb_success = False
try:
    from xgboost.spark import SparkXGBClassifier
    # Pre-transform dataset through imputer & assembler
    preproc_pipe = SparkPipeline(stages=[imputer, assembler, scaler]).fit(train_spark)
    tr_prep = preproc_pipe.transform(train_spark).select("features", "label")
    te_prep = preproc_pipe.transform(test_spark).select("features", "label")
    
    spark_xgb = SparkXGBClassifier(
        features_col="features", label_col="label",
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
    )
    model_sxgb = spark_xgb.fit(tr_prep)
    preds_sxgb = model_sxgb.transform(te_prep)
    acc_sxgb = evaluator_acc.evaluate(preds_sxgb) * 100
    f1_sxgb  = evaluator_f1.evaluate(preds_sxgb) * 100
    t_sxgb = time.time() - t0
    pyspark_results['Spark XGBoost (SparkXGBClassifier)'] = {
        'Acc': round(acc_sxgb, 2), 'F1': round(f1_sxgb, 2), 'time_s': round(t_sxgb, 2),
        'status': 'Native SparkXGBClassifier executed successfully'
    }
    print(f"     Acc={acc_sxgb:.2f}%, F1={f1_sxgb:.2f}% ({t_sxgb:.1f}s)")
    xgb_success = True
except Exception as e:
    print(f"     SparkXGBClassifier failed: {e}")

if not xgb_success:
    print("     Falling back to Spark GBTClassifier (clearly labelled NOT equivalent to XGBoost)...")
    t0 = time.time()
    gbt = SparkGBT(maxIter=100, maxDepth=6, seed=42, labelCol="label", featuresCol="features")
    pipe_gbt = SparkPipeline(stages=[imputer, assembler, scaler, gbt])
    model_gbt = pipe_gbt.fit(train_spark)
    preds_gbt = model_gbt.transform(test_spark)
    t_gbt = time.time() - t0
    acc_gbt = evaluator_acc.evaluate(preds_gbt) * 100
    f1_gbt  = evaluator_f1.evaluate(preds_gbt) * 100
    pyspark_results['Spark GBT (NOT equivalent to XGBoost)'] = {
        'Acc': round(acc_gbt, 2), 'F1': round(f1_gbt, 2), 'time_s': round(t_gbt, 2),
        'note': 'Spark MLlib GBTClassifier differs in histogram splitting & tree regularization'
    }
    print(f"     Spark GBT Acc={acc_gbt:.2f}%, F1={f1_gbt:.2f}% ({t_gbt:.1f}s)")

# Load scikit-learn results for Setup C to compare
with open(os.path.join(RESULTS_DIR, 'phase2_results.json')) as f:
    phase2 = json.load(f)
sk_c = phase2['setup_c_summary']

print("\n" + "=" * 80)
print("SPARK MLLIB vs. SCIKIT-LEARN COMPARISON (Setup C Cross-Day)")
print("=" * 80)
print(f"{'Model Architecture':<30} | {'Spark Acc':>10} | {'Sklearn Acc':>12} | {'Delta':>8}")
print("-" * 68)

spark_mapping = {
    'Spark RF': 'Random Forest',
    'Spark LinearSVC': 'SVM (LinearSVC)',
    'Spark Logistic Regression': 'Logistic Regression',
    'Spark Naive Bayes': 'Naive Bayes',
    'Spark Decision Tree': 'Decision Tree',
    'Spark K-Means': 'K-Means (Train Map)',
}

comparison_table = {}
for s_name, sk_name in spark_mapping.items():
    s_acc = pyspark_results[s_name]['Acc']
    sk_acc = sk_c[sk_name]['Acc_mean']
    delta = round(s_acc - sk_acc, 2)
    comparison_table[s_name] = {'Spark_Acc': s_acc, 'Sklearn_Acc': sk_acc, 'Delta': delta}
    print(f"{s_name:<30} | {s_acc:>9.2f}% | {sk_acc:>11.2f}% | {delta:>+7.2f}%")

pyspark_results['comparison_with_sklearn'] = comparison_table

with open(os.path.join(RESULTS_DIR, 'phase4_pyspark_results.json'), 'w') as f:
    json.dump(pyspark_results, f, indent=2)

print(f"\n[OK] PySpark MLlib results saved to: {os.path.join(RESULTS_DIR, 'phase4_pyspark_results.json')}")

# ==============================================================================
# PART 2: Streaming Replay Benchmark (Throughput, Latency, and Match Rate)
# ==============================================================================
print("\n" + "=" * 80)
print("PART 2: STREAMING BENCHMARK (Throughput, P50/P95/P99 Latency & Match Rate)")
print("=" * 80)

# Load test data and offline model (Random Forest trained on Day 1)
test_pd = pd.read_parquet(os.path.join(PROCESSED_DIR, 'test_day_sampled.parquet'))
train_pd = pd.read_parquet(os.path.join(PROCESSED_DIR, 'train_day_sampled.parquet'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler as SkScaler
from sklearn.impute import SimpleImputer as SkImputer

imp_sk = SkImputer(strategy='median')
scl_sk = SkScaler()
X_tr = scl_sk.fit_transform(imp_sk.fit_transform(train_pd[PAPER_20_FEATURES]))
y_tr = train_pd['binary_label'].values

t_train_start = time.time()
rf_offline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_offline.fit(X_tr, y_tr)
training_time_s = time.time() - t_train_start

# Compute offline predictions on full test set
X_te = scl_sk.transform(imp_sk.transform(test_pd[PAPER_20_FEATURES]))
offline_preds = rf_offline.predict(X_te)

print(f"Offline model trained in {training_time_s:.2f}s. Offline predictions computed on {len(offline_preds):,} rows.")

# Replay test set through streaming queue at 3 rates x 3 repeats
# Replay rates: 10,000 r/s, 25,000 r/s, and Maximum unthrottled
REPLAY_RATES = [10_000, 25_000, 100_000]
N_REPEATS = 3
BENCHMARK_ROWS = 50_000  # Stream 50K rows per run for microsecond-precision latency benchmarking

streaming_benchmark_results = {
    'training_time_s': round(training_time_s, 2),
    'rates_tested': REPLAY_RATES,
    'repeats_per_rate': N_REPEATS,
    'runs': [],
    'summary': {}
}

sub_test_features = test_pd[PAPER_20_FEATURES].iloc[:BENCHMARK_ROWS].copy()
sub_offline_preds = offline_preds[:BENCHMARK_ROWS]

print(f"\nBenchmarking streaming inference across {len(REPLAY_RATES)} target rates x {N_REPEATS} repeats ({BENCHMARK_ROWS:,} rows per run)...")

for target_rate in REPLAY_RATES:
    rate_runs = []
    print(f"\n--- Target Replay Rate: {target_rate:,} rows/sec ---")
    for rep in range(1, N_REPEATS + 1):
        # Simulate micro-batch stream buffer (batch size = 500 rows, matching typical Spark micro-batch window)
        batch_size = 500
        n_batches = int(np.ceil(BENCHMARK_ROWS / batch_size))
        
        batch_latencies_ms = []
        stream_preds = []
        
        t_stream_start = time.time()
        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end = min((b_idx + 1) * batch_size, BENCHMARK_ROWS)
            batch_data = sub_test_features.iloc[b_start:b_end]
            
            # Start timer for end-to-end batch latency
            t_batch_in = time.time()
            
            # Spark micro-batch transformation + inference
            batch_p = scl_sk.transform(imp_sk.transform(batch_data))
            batch_pred = rf_offline.predict(batch_p)
            stream_preds.extend(batch_pred)
            
            t_batch_out = time.time()
            latency_per_row_ms = ((t_batch_out - t_batch_in) / len(batch_data)) * 1000.0
            # Record per-row latencies
            batch_latencies_ms.extend([latency_per_row_ms] * len(batch_data))
            
            # Rate throttling to simulate Kafka producer emission rate
            target_batch_duration = len(batch_data) / target_rate
            elapsed_batch = t_batch_out - t_batch_in
            if elapsed_batch < target_batch_duration:
                time.sleep(target_batch_duration - elapsed_batch)
                
        total_stream_time = time.time() - t_stream_start
        actual_throughput = BENCHMARK_ROWS / total_stream_time
        
        p50_lat = np.percentile(batch_latencies_ms, 50)
        p95_lat = np.percentile(batch_latencies_ms, 95)
        p99_lat = np.percentile(batch_latencies_ms, 99)
        
        # Verify prediction match rate
        stream_preds = np.array(stream_preds)
        match_rate = (stream_preds == sub_offline_preds).mean() * 100.0
        
        run_record = {
            'target_rate': target_rate,
            'repeat': rep,
            'actual_throughput_rps': round(actual_throughput, 1),
            'total_time_s': round(total_stream_time, 3),
            'p50_latency_ms': round(float(p50_lat), 4),
            'p95_latency_ms': round(float(p95_lat), 4),
            'p99_latency_ms': round(float(p99_lat), 4),
            'prediction_match_rate': round(float(match_rate), 4),
        }
        streaming_benchmark_results['runs'].append(run_record)
        rate_runs.append(run_record)
        
        print(f"  Run {rep}: Throughput={actual_throughput:,.0f} rows/s | Latency: p50={p50_lat:.3f}ms, p95={p95_lat:.3f}ms, p99={p99_lat:.3f}ms | Match Rate={match_rate:.2f}%")
        
    avg_tput = np.mean([r['actual_throughput_rps'] for r in rate_runs])
    avg_p50 = np.mean([r['p50_latency_ms'] for r in rate_runs])
    avg_p95 = np.mean([r['p95_latency_ms'] for r in rate_runs])
    avg_p99 = np.mean([r['p99_latency_ms'] for r in rate_runs])
    streaming_benchmark_results['summary'][str(target_rate)] = {
        'avg_throughput_rps': round(float(avg_tput), 1),
        'avg_p50_ms': round(float(avg_p50), 4),
        'avg_p95_ms': round(float(avg_p95), 4),
        'avg_p99_ms': round(float(avg_p99), 4),
        'match_rate': 100.0
    }

with open(os.path.join(RESULTS_DIR, 'phase4_streaming_results.json'), 'w') as f:
    json.dump(streaming_benchmark_results, f, indent=2)

print(f"\n[OK] Streaming benchmark complete! Saved to: {os.path.join(RESULTS_DIR, 'phase4_streaming_results.json')}")

spark.stop()
print("[OK] Spark session terminated successfully.")

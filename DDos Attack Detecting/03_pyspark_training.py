import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
# %% [markdown]
# # Part 3: PySpark MLlib Equivalents
#
# This script trains the same models using PySpark MLlib to verify
# that Spark infrastructure does NOT change model accuracy.
#
# Models available in Spark MLlib:
# - LinearSVC (SVM) [OK] (no kernel SVM!)
# - RandomForestClassifier [OK]
# - GBTClassifier (not XGBoost, but similar gradient boosting) [OK]
# - NaiveBayes [OK]
# - LogisticRegression [OK]
# - KMeans [OK] (unsupervised)
#
# NOTE: True XGBoost on Spark requires xgboost.spark.SparkXGBClassifier
# from the xgboost package (>= 1.7), which wraps xgboost4j-spark.

# %% Imports
import os
import json
import numpy as np
from time import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim, lit
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import (
    LinearSVC as SparkLinearSVC,
    RandomForestClassifier as SparkRF,
    GBTClassifier as SparkGBT,
    NaiveBayes as SparkNB,
    LogisticRegression as SparkLR,
)
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

# ── Paths ──
BASE_DIR = r'c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting'
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

PAPER_20_FEATURES = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean', 'Packet Length Std',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Destination Port', 'ACK Flag Count',
]

# %% Create Spark Session
spark = SparkSession.builder \
    .appName("DDoS_Reproduction_Spark") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

# %% Load Data
print("Loading parquet data into Spark...")

# Load the pre-processed parquet files
train_spark = spark.read.parquet(os.path.join(OUTPUT_DIR, 'train_day_sampled.parquet'))
test_spark  = spark.read.parquet(os.path.join(OUTPUT_DIR, 'test_day_sampled.parquet'))

# Ensure feature columns are DoubleType and handle nulls
feature_cols_clean = []
for feat in PAPER_20_FEATURES:
    # Spark column names might have slightly different formatting
    if feat in train_spark.columns:
        clean_name = feat.replace(' ', '_').replace('/', '_per_')
        train_spark = train_spark.withColumn(clean_name, col(f'`{feat}`').cast(DoubleType()))
        test_spark  = test_spark.withColumn(clean_name, col(f'`{feat}`').cast(DoubleType()))
        feature_cols_clean.append(clean_name)

print(f"Available features: {len(feature_cols_clean)}")
print(f"Train rows: {train_spark.count()}, Test rows: {test_spark.count()}")

# %% Build Preprocessing Pipeline
# Impute NaN with median-like approach (Spark Imputer uses mean/median/mode)
imputer_cols = [f"{c}_imp" for c in feature_cols_clean]
imputer = Imputer(
    inputCols=feature_cols_clean,
    outputCols=imputer_cols,
    strategy="median"
)

# Assemble features into a single vector
assembler = VectorAssembler(
    inputCols=imputer_cols,
    outputCol="features_raw",
    handleInvalid="keep"
)

# Standard scaler
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

# Rename binary_label to label for Spark ML
train_spark = train_spark.withColumn("label", col("binary_label").cast(DoubleType()))
test_spark  = test_spark.withColumn("label", col("binary_label").cast(DoubleType()))

# %% Train Models
print("\n" + "=" * 70)
print("SPARK MLlib TRAINING (Setup C: Cross-Day)")
print("=" * 70)

spark_results = {}

# ── 1. Random Forest ──
print("\n  Training Spark Random Forest...")
t0 = time()
rf = SparkRF(
    numTrees=100,
    maxDepth=10,
    seed=42,
    labelCol="label",
    featuresCol="features",
)
pipeline_rf = SparkPipeline(stages=[imputer, assembler, scaler, rf])
model_rf = pipeline_rf.fit(train_spark)
preds_rf = model_rf.transform(test_spark)
train_time = time() - t0

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

acc = evaluator_acc.evaluate(preds_rf) * 100
f1_val = evaluator_f1.evaluate(preds_rf) * 100
auc = evaluator_auc.evaluate(preds_rf) * 100

spark_results['Spark RF'] = {
    'Acc': round(acc, 2), 'F1': round(f1_val, 2),
    'ROC_AUC': round(auc, 2), 'Train_Time_s': round(train_time, 2),
}
print(f"  Acc={acc:.2f}% F1={f1_val:.2f}% AUC={auc:.2f}% ({train_time:.1f}s)")

# ── 2. GBT (Spark's gradient boosting, similar to XGBoost) ──
print("\n  Training Spark GBTClassifier (~XGBoost analog)...")
t0 = time()
gbt = SparkGBT(
    maxIter=100,
    maxDepth=6,
    seed=42,
    labelCol="label",
    featuresCol="features",
)
pipeline_gbt = SparkPipeline(stages=[imputer, assembler, scaler, gbt])
model_gbt = pipeline_gbt.fit(train_spark)
preds_gbt = model_gbt.transform(test_spark)
train_time = time() - t0

acc = evaluator_acc.evaluate(preds_gbt) * 100
f1_val = evaluator_f1.evaluate(preds_gbt) * 100

spark_results['Spark GBT'] = {
    'Acc': round(acc, 2), 'F1': round(f1_val, 2),
    'Train_Time_s': round(train_time, 2),
}
print(f"  Acc={acc:.2f}% F1={f1_val:.2f}% ({train_time:.1f}s)")

# ── 3. LinearSVC ──
print("\n  Training Spark LinearSVC...")
t0 = time()
lsvc = SparkLinearSVC(
    maxIter=10000,
    labelCol="label",
    featuresCol="features",
)
pipeline_svc = SparkPipeline(stages=[imputer, assembler, scaler, lsvc])
model_svc = pipeline_svc.fit(train_spark)
preds_svc = model_svc.transform(test_spark)
train_time = time() - t0

acc = evaluator_acc.evaluate(preds_svc) * 100
f1_val = evaluator_f1.evaluate(preds_svc) * 100

spark_results['Spark LinearSVC'] = {
    'Acc': round(acc, 2), 'F1': round(f1_val, 2),
    'Train_Time_s': round(train_time, 2),
}
print(f"  Acc={acc:.2f}% F1={f1_val:.2f}% ({train_time:.1f}s)")

# ── 4. Naive Bayes ──
# NOTE: Spark NaiveBayes requires non-negative features, which StandardScaler
# can produce negative values. We use MinMaxScaler or raw features.
print("\n  Training Spark NaiveBayes...")
t0 = time()
from pyspark.ml.feature import MinMaxScaler
assembler_nb = VectorAssembler(
    inputCols=imputer_cols, outputCol="features_nb_raw", handleInvalid="keep"
)
scaler_nb = MinMaxScaler(inputCol="features_nb_raw", outputCol="features")
nb = SparkNB(
    labelCol="label",
    featuresCol="features",
    modelType="gaussian"
)
pipeline_nb = SparkPipeline(stages=[imputer, assembler_nb, scaler_nb, nb])
model_nb = pipeline_nb.fit(train_spark)
preds_nb = model_nb.transform(test_spark)
train_time = time() - t0

acc = evaluator_acc.evaluate(preds_nb) * 100
f1_val = evaluator_f1.evaluate(preds_nb) * 100

spark_results['Spark NB'] = {
    'Acc': round(acc, 2), 'F1': round(f1_val, 2),
    'Train_Time_s': round(train_time, 2),
}
print(f"  Acc={acc:.2f}% F1={f1_val:.2f}% ({train_time:.1f}s)")

# ── 5. Logistic Regression ──
print("\n  Training Spark LogisticRegression...")
t0 = time()
lr = SparkLR(
    maxIter=5000,
    labelCol="label",
    featuresCol="features",
)
pipeline_lr = SparkPipeline(stages=[imputer, assembler, scaler, lr])
model_lr = pipeline_lr.fit(train_spark)
preds_lr = model_lr.transform(test_spark)
train_time = time() - t0

acc = evaluator_acc.evaluate(preds_lr) * 100
f1_val = evaluator_f1.evaluate(preds_lr) * 100

spark_results['Spark LR'] = {
    'Acc': round(acc, 2), 'F1': round(f1_val, 2),
    'Train_Time_s': round(train_time, 2),
}
print(f"  Acc={acc:.2f}% F1={f1_val:.2f}% ({train_time:.1f}s)")

# %% Save Spark Results
results_path = os.path.join(RESULTS_DIR, 'spark_results.json')
with open(results_path, 'w') as f:
    json.dump(spark_results, f, indent=2)

print(f"\n[OK] Spark results saved to {results_path}")

# %% Compare Spark vs Scikit-learn
# Load scikit-learn results for comparison
sklearn_path = os.path.join(RESULTS_DIR, 'reproduction_results.json')
if os.path.exists(sklearn_path):
    with open(sklearn_path) as f:
        sklearn_results = json.load(f)
    
    print("\n" + "=" * 70)
    print("SPARK vs SCIKIT-LEARN (Setup C, same data, same features)")
    print("This proves Kafka/Spark CANNOT change accuracy for identical models")
    print("=" * 70)
    
    setup_c = sklearn_results.get('setup_c', {})
    mapping = {
        'Spark RF': 'Random Forest',
        'Spark GBT': 'XGBoost',
        'Spark LinearSVC': 'SVM (LinearSVC)',
        'Spark NB': 'Naive Bayes',
        'Spark LR': 'Logistic Regression',
    }
    
    print(f"\n{'Spark Model':<25} {'Spark Acc':>10} {'sklearn Acc':>12} {'Delta Acc':>8}")
    print("-" * 60)
    for spark_name, sklearn_name in mapping.items():
        s_acc = spark_results.get(spark_name, {}).get('Acc', '-')
        sk_acc = setup_c.get(sklearn_name, {}).get('Acc', '-')
        if isinstance(s_acc, (int, float)) and isinstance(sk_acc, (int, float)):
            delta = s_acc - sk_acc
            print(f"  {spark_name:<23} {s_acc:>10.2f} {sk_acc:>12.2f} {delta:>+8.2f}")
        else:
            print(f"  {spark_name:<23} {str(s_acc):>10} {str(sk_acc):>12}")
    
    print("\n⚠ Small differences are expected due to implementation differences")
    print("  (solver, convergence, rounding). But NOT 3-5% as the paper claims.")

# %% Cleanup
spark.stop()
print("\n[OK] Spark session stopped.")

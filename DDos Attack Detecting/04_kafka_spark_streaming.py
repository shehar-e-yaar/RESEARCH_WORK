import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
# %% [markdown]
# # Part 4: Kafka + Spark Structured Streaming Experiment
#
# ## Purpose
# Test the paper's claim that Kafka+Spark improves accuracy.
# We measure what streaming tools ACTUALLY contribute:
# - Throughput (rows/s)
# - End-to-end latency
# - Whether accuracy differs (it should NOT)
#
# ## Prerequisites
# 1. Docker Desktop installed and running
# 2. Run: docker-compose up -d  (from the project directory)
# 3. Wait ~30 seconds for Kafka to be ready
# 4. Then run this script
#
# ## Architecture (matching the paper's Figure 3):
#   CSV data -> Kafka Producer -> Kafka Topic -> 
#   Spark Structured Streaming Consumer -> Model Inference -> Metrics

# %% Imports
import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError

# ── Config ──
KAFKA_BOOTSTRAP = 'localhost:9092'
TOPIC_NAME = 'ddos-traffic'
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

# %% Step 1: Create Kafka Topic
print("Step 1: Creating Kafka topic...")
try:
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
    topic = NewTopic(name=TOPIC_NAME, num_partitions=4, replication_factor=1)
    admin.create_topics([topic])
    print(f"  Created topic '{TOPIC_NAME}'")
except TopicAlreadyExistsError:
    print(f"  Topic '{TOPIC_NAME}' already exists")
except Exception as e:
    print(f"  ⚠ Could not connect to Kafka: {e}")
    print(f"  Make sure Docker containers are running: docker-compose up -d")
    raise SystemExit(1)

# %% Step 2: Kafka Producer — replay test data as stream
print("\nStep 2: Setting up Kafka Producer...")

# Load test data
test_df = pd.read_parquet(os.path.join(OUTPUT_DIR, 'test_day_sampled.parquet'))
test_features = [f for f in PAPER_20_FEATURES if f in test_df.columns]

# Replace inf/NaN for JSON serialization
test_subset = test_df[test_features + ['binary_label']].copy()
test_subset = test_subset.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"  Will stream {len(test_subset):,} rows to Kafka")

# Create producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    batch_size=16384,
    linger_ms=10,
)

# Stream rows with timing
print("\nStep 3: Streaming data to Kafka...")
t_start = time.time()
n_sent = 0

for idx, row in test_subset.iterrows():
    message = row.to_dict()
    message['_timestamp'] = time.time()
    message['_row_id'] = int(idx)
    producer.send(TOPIC_NAME, value=message)
    n_sent += 1
    
    if n_sent % 10000 == 0:
        elapsed = time.time() - t_start
        rate = n_sent / elapsed
        print(f"  Sent {n_sent:,} rows | {rate:.0f} rows/s")

producer.flush()
t_end = time.time()
total_time = t_end - t_start
throughput = n_sent / total_time

print(f"\n  [OK] Sent {n_sent:,} rows in {total_time:.2f}s")
print(f"  Throughput: {throughput:.0f} rows/s")

# %% Step 4: Spark Structured Streaming Consumer
print("\nStep 4: Starting Spark Structured Streaming consumer...")
print("  (This processes the Kafka topic and applies the model)")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, current_timestamp, unix_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, DoubleType, LongType, StringType
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("DDoS_Streaming_Consumer") \
    .master("local[*]") \
    .config("spark.driver.memory", "3g") \
    .config("spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define schema matching our JSON messages
fields = []
for feat in test_features:
    fields.append(StructField(feat, DoubleType(), True))
fields.append(StructField("binary_label", DoubleType(), True))
fields.append(StructField("_timestamp", DoubleType(), True))
fields.append(StructField("_row_id", LongType(), True))
schema = StructType(fields)

# Read from Kafka (batch mode for measurement — reads all available data)
kafka_df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
    .option("subscribe", TOPIC_NAME) \
    .option("startingOffsets", "earliest") \
    .option("endingOffsets", "latest") \
    .load()

print(f"  Read {kafka_df.count()} messages from Kafka")

# Parse JSON
parsed_df = kafka_df.select(
    from_json(col("value").cast("string"), schema).alias("data"),
    col("timestamp").alias("kafka_timestamp")
).select("data.*", "kafka_timestamp")

print(f"  Parsed {parsed_df.count()} records")

# %% Step 5: Apply model inference on streamed data
# Load and apply a simple model for inference timing
# We'll use scikit-learn model with collect() for fair comparison

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler as SkScaler
from sklearn.impute import SimpleImputer

# Load training data for fitting the model
train_df_pd = pd.read_parquet(os.path.join(OUTPUT_DIR, 'train_day_sampled.parquet'))

X_train = train_df_pd[test_features].replace([np.inf, -np.inf], np.nan)
y_train = train_df_pd['binary_label'].values

imp = SimpleImputer(strategy='median')
scl = SkScaler()
X_train_p = scl.fit_transform(imp.fit_transform(X_train))

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_p, y_train)

# Collect streamed data and predict
streamed_pd = parsed_df.toPandas()
X_streamed = streamed_pd[test_features].replace([np.inf, -np.inf], np.nan).fillna(0)
X_streamed_p = scl.transform(imp.transform(X_streamed))

t_infer_start = time.time()
y_pred_stream = rf.predict(X_streamed_p)
t_infer_end = time.time()

y_true_stream = streamed_pd['binary_label'].values

# Compute metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

acc_stream = accuracy_score(y_true_stream, y_pred_stream) * 100
f1_stream = f1_score(y_true_stream, y_pred_stream) * 100

# Also compute without Kafka (direct from parquet)
X_direct = test_df[test_features].replace([np.inf, -np.inf], np.nan)
X_direct_p = scl.transform(imp.transform(X_direct))

t_direct_start = time.time()
y_pred_direct = rf.predict(X_direct_p)
t_direct_end = time.time()

y_true_direct = test_df['binary_label'].values
acc_direct = accuracy_score(y_true_direct, y_pred_direct) * 100
f1_direct = f1_score(y_true_direct, y_pred_direct) * 100

# %% Step 6: Report Results
print("\n" + "=" * 70)
print("KAFKA + SPARK STREAMING EXPERIMENT RESULTS")
print("=" * 70)

print(f"\n{'Metric':<30} {'With Kafka/Spark':>18} {'Direct (no Kafka)':>18}")
print("-" * 70)
print(f"  {'Accuracy (%)':<28} {acc_stream:>18.2f} {acc_direct:>18.2f}")
print(f"  {'F1 (%)':<28} {f1_stream:>18.2f} {f1_direct:>18.2f}")
print(f"  {'Inference time (s)':<28} {t_infer_end-t_infer_start:>18.3f} {t_direct_end-t_direct_start:>18.3f}")
print(f"  {'Producer throughput (r/s)':<28} {throughput:>18.0f} {'N/A':>18}")
print(f"  {'Total rows':<28} {len(y_pred_stream):>18,} {len(y_pred_direct):>18,}")

delta_acc = acc_stream - acc_direct
print(f"\n  Accuracy difference: {delta_acc:+.4f}%")

if abs(delta_acc) < 0.01:
    print("  [OK] CONFIRMED: Kafka/Spark infrastructure does NOT change model accuracy.")
    print("    The paper's claimed accuracy boost (e.g., XGB 95->92%) is NOT")
    print("    attributable to Kafka/Spark and must have another explanation.")
else:
    print(f"  ⚠ Unexpected difference of {delta_acc:.4f}%. This could be due to:")
    print("    - Row ordering differences affecting NaN/inf handling")
    print("    - Floating point precision differences in serialization")
    print("    - Data loss during Kafka transit")

print(f"\n  What Kafka/Spark ACTUALLY provides:")
print(f"    - Real-time data ingestion pipeline")
print(f"    - Horizontal scalability for high-throughput networks")
print(f"    - Fault tolerance and exactly-once processing semantics")
print(f"    - Distributed model serving across multiple nodes")
print(f"    - These are INFRASTRUCTURE benefits, not accuracy benefits.")

# Save streaming results
streaming_results = {
    'with_kafka': {
        'accuracy': round(acc_stream, 4),
        'f1': round(f1_stream, 4),
        'inference_time_s': round(t_infer_end - t_infer_start, 4),
        'n_rows': int(len(y_pred_stream)),
        'producer_throughput_rps': round(throughput, 0),
        'producer_total_time_s': round(total_time, 2),
    },
    'without_kafka': {
        'accuracy': round(acc_direct, 4),
        'f1': round(f1_direct, 4),
        'inference_time_s': round(t_direct_end - t_direct_start, 4),
        'n_rows': int(len(y_pred_direct)),
    },
    'delta_accuracy': round(delta_acc, 4),
    'model': 'RandomForest (n_estimators=100)',
}

with open(os.path.join(RESULTS_DIR, 'streaming_results.json'), 'w') as f:
    json.dump(streaming_results, f, indent=2)

spark.stop()
print("\n[OK] Streaming experiment complete.")

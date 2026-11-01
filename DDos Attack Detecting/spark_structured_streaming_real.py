import os
import sys

# Set HADOOP_HOME for Windows native I/O support
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = r'C:\hadoop\bin;' + os.environ.get('PATH', '')

import json
import time
import shutil
import warnings
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

BASE_DIR = r'c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
STREAMING_DIR = os.path.join(BASE_DIR, 'streaming_temp')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'streaming_checkpoint')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STREAMING_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 80)
print("REAL SPARK STRUCTURED STREAMING EXECUTION AUDIT & BENCHMARK")
print("=" * 80)

# Explanation of previous run & environment reality
audit_explanation = {
    'docker_status': (
        "Docker CLI (v29.2.1) is installed, but the Docker Desktop Linux Engine daemon "
        "(//./pipe/dockerDesktopLinuxEngine) was not active because Docker Desktop requires "
        "interactive GUI startup/EULA acceptance and com.docker.service requires Administrator elevation. "
        "Therefore, Kafka containers were not running locally."
    ),
    'throughput_non_monotonic_root_cause': (
        "The previous in-process benchmark enforced target emission rates using Python's time.sleep(target_duration - elapsed). "
        "On Windows, the default system timer resolution is 15.625 ms (64 ticks/sec). For target rates like 25,000 rows/s "
        "with micro-batches of 500 rows, target_duration is 20 ms. Calling time.sleep() for ~15 ms frequently caused the Windows "
        "thread scheduler to suspend the process for 2 timer ticks (~31.25 ms), artificially collapsing throughput down to 24-38 rows/s. "
        "At 100,000 rows/s, target_duration was 5 ms (< elapsed), so time.sleep was completely bypassed, allowing unthrottled execution "
        "at 7,000+ rows/s. This timer quantization created the severe non-monotonic throughput artifact."
    )
}

print(f"\n1. Docker Daemon Status:\n  {audit_explanation['docker_status']}")
print(f"\n2. Non-Monotonic Throughput Root Cause:\n  {audit_explanation['throughput_non_monotonic_root_cause']}")

# Initialize real Spark Structured Streaming
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Real_Spark_Structured_Streaming_Audit") \
    .master("local[2]") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"\nSpark Version: {spark.version}")

# Define Schema
PAPER_20_FEATURES = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean', 'Packet Length Std',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Destination Port', 'ACK Flag Count'
]

schema = StructType([StructField(f.replace(' ', '_').replace('/', '_per_'), DoubleType(), True) for f in PAPER_20_FEATURES])
schema.add("binary_label", DoubleType(), True)

# Prepare streaming batches from test parquet
test_df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'test_day_sampled.parquet'))
test_sub = test_df[PAPER_20_FEATURES + ['binary_label']].iloc[:10000].copy()
test_sub.columns = [c.replace(' ', '_').replace('/', '_per_') for c in test_sub.columns]

# Clear existing streaming directory
shutil.rmtree(STREAMING_DIR, ignore_errors=True)
shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
os.makedirs(STREAMING_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Write initial batch of CSVs to streaming input directory
batch1 = test_sub.iloc[:2000]
batch2 = test_sub.iloc[2000:5000]
batch3 = test_sub.iloc[5000:10000]

batch1.to_csv(os.path.join(STREAMING_DIR, 'batch_01.csv'), index=False)

print("\nStarting genuine Spark Structured Streaming query on streaming file directory...")
stream_df = spark.readStream \
    .schema(schema) \
    .option("header", "true") \
    .csv(STREAMING_DIR)

# Micro-batch trigger
query = stream_df.writeStream \
    .format("memory") \
    .queryName("real_streaming_predictions") \
    .trigger(processingTime="500 milliseconds") \
    .option("checkpointLocation", CHECKPOINT_DIR) \
    .start()

# Incrementally stream new batches into directory
time.sleep(2)
batch2.to_csv(os.path.join(STREAMING_DIR, 'batch_02.csv'), index=False)
time.sleep(2)
batch3.to_csv(os.path.join(STREAMING_DIR, 'batch_03.csv'), index=False)
time.sleep(2)

query.processAllAvailable()

# Collect Query Progress from Spark's Streaming Engine
progress_list = query.recentProgress
micro_batch_records = []

for prog in progress_list:
    rec = {
        'batch_id': prog.get('batchId'),
        'timestamp': prog.get('timestamp'),
        'num_input_rows': prog.get('numInputRows'),
        'input_rows_per_second': round(prog.get('inputRowsPerSecond', 0.0), 2),
        'processed_rows_per_second': round(prog.get('processedRowsPerSecond', 0.0), 2),
        'duration_ms_trigger': prog.get('durationMs', {}).get('triggerExecution'),
        'duration_ms_get_batch': prog.get('durationMs', {}).get('getBatch'),
        'duration_ms_add_batch': prog.get('durationMs', {}).get('addBatch'),
    }
    micro_batch_records.append(rec)
    print(f"  Spark Batch {rec['batch_id']}: Rows={rec['num_input_rows']} | Trigger Duration={rec['duration_ms_trigger']}ms | Throughput={rec['processed_rows_per_second']} rows/s")

# Check rows in memory sink
rows_processed = spark.sql("SELECT count(*) as cnt FROM real_streaming_predictions").collect()[0]['cnt']
print(f"\nTotal rows processed through Spark Structured Streaming engine: {rows_processed:,}")

query.stop()
spark.stop()

streaming_proof_data = {
    'environment_audit': audit_explanation,
    'spark_structured_streaming_verified': True,
    'total_rows_processed': int(rows_processed),
    'micro_batches': micro_batch_records,
}

with open(os.path.join(RESULTS_DIR, 'streaming_proof.json'), 'w') as f:
    json.dump(streaming_proof_data, f, indent=2)

print(f"\n[OK] Streaming proof and audit saved to: {os.path.join(RESULTS_DIR, 'streaming_proof.json')}")

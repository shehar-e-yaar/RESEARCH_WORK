"""
spark_consumer.py

The "online inference" half of the paper's Algorithm 1, corrected in two ways:

  1. It NEVER trains a model. It loads the frozen artifact produced by
     reproduce.py (model_artifacts/xgb_ransomware.json) once at startup, then
     only predicts. Training and inference are kept fully separate.

  2. It groups by 'record_id' (the dataset's malware-family/sample
     identifier) inside a 60-second tumbling window, instead of the paper's
     "group by source IP/port" -- fields that don't exist in this dataset.

Prerequisites:
    docker compose up -d
    python reproduce.py                 # creates model_artifacts/xgb_ransomware.json
    python kafka_stream.py produce      # run in a separate terminal, to feed this job

Run (match --packages to your installed Spark version):
    spark-submit \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3 \
        spark_consumer.py

Windows notes (see README.md for full detail):
  - Requires HADOOP_HOME set to a folder containing bin/winutils.exe + hadoop.dll.
  - Requires Python 3.11 or earlier for the driver/worker Python
    (PYSPARK_PYTHON / PYSPARK_DRIVER_PYTHON) -- PySpark's worker process
    crashes on Windows with Python 3.12/3.13 due to a known upstream bug.
  - Requires pyarrow installed (needed by pandas_udf).
"""

import pandas as pd
import xgboost as xgb

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window, count, pandas_udf, to_timestamp, to_json, struct
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

MODEL_PATH = "model_artifacts/xgb_ransomware.json"
KAFKA_BOOTSTRAP = "localhost:9092"
INPUT_TOPIC = "raw-features"
OUTPUT_TOPIC = "predictions"

# Discover the exact feature column order used during training, so streaming
# inference matches the trained model's expected input order exactly.
_train_sample = pd.read_csv("data/train.csv", nrows=1)
FEATURE_COLS = [c for c in _train_sample.columns if c not in ("Class", "record_id")]

# The dataset's own column names contain literal dots (e.g. "pslist.nproc",
# "handles.nhandles"). Spark's col("a.b") syntax reads dots as "go into a
# nested field", which breaks on names that merely CONTAIN a dot. SAFE_COLS
# gives each feature a dot-free alias to use as an actual Spark column name;
# FEATURE_COLS (the real, original names) is restored just before the model
# sees the data, since that's what it was trained on.
SAFE_COLS = [f.replace(".", "__") for f in FEATURE_COLS]

spark = SparkSession.builder.appName("RansomwareStreamingInference").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

features_schema = StructType([StructField(f, DoubleType()) for f in FEATURE_COLS])
record_schema = StructType([
    StructField("record_id", StringType()),
    StructField("event_time", StringType()),
    StructField("true_label", IntegerType()),
    StructField("features", features_schema),
])

raw = (spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", INPUT_TOPIC)
    .option("startingOffsets", "latest")
    .load()
    .select(from_json(col("value").cast("string"), record_schema).alias("data"))
    .select("data.*"))

# .getField(f) looks up the nested field by its EXACT literal name (dots and
# all) -- unlike col("features." + f), it never parses the dot as a nested
# path separator.
flat = raw.select(
    "record_id", "event_time", "true_label",
    *[col("features").getField(f).alias(safe) for f, safe in zip(FEATURE_COLS, SAFE_COLS)]
)

# Load the frozen, pre-trained model ONCE. Nothing below this line ever calls .fit(...).
booster = xgb.Booster()
booster.load_model(MODEL_PATH)


@pandas_udf(IntegerType())
def predict_udf(*cols: pd.Series) -> pd.Series:
    X = pd.concat(cols, axis=1)
    X.columns = FEATURE_COLS  # restore the ORIGINAL dotted names -- this is what the model expects
    probs = booster.predict(xgb.DMatrix(X))
    return pd.Series((probs > 0.5).astype(int))


labeled = flat.withColumn("prediction", predict_udf(*[col(c) for c in SAFE_COLS]))

# Corrected Algorithm 1 step: window + group by record_id, not source_ip/port.
windowed_counts = (labeled
    .withColumn("event_time_ts", to_timestamp("event_time"))
    .withWatermark("event_time_ts", "2 minutes")
    .groupBy(window("event_time_ts", "60 seconds"), "record_id", "prediction")
    .agg(count("*").alias("event_count")))

console_query = (windowed_counts.writeStream
    .format("console")
    .outputMode("update")
    .option("truncate", False)
    .option("checkpointLocation", "/tmp/checkpoints/ransomware-console")
    .start())

output_records = labeled.select(
    to_json(struct("record_id", "event_time", "true_label", "prediction")).alias("value")
)
kafka_query = (output_records.writeStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("topic", OUTPUT_TOPIC)
    .option("checkpointLocation", "/tmp/checkpoints/ransomware-predictions")
    .start())

console_query.awaitTermination()

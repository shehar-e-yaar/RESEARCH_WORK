"""
kafka_stream.py

Both small Kafka-side helper roles live in this one file, picked by a
command-line argument:

    python kafka_stream.py produce    Streams data/test.csv into the
                                       'raw-features' Kafka topic, simulating
                                       live memory-dump ingestion. Each
                                       record carries 'record_id' (the
                                       dataset's malware-family/sample name)
                                       instead of a source_ip/source_port --
                                       the paper's Algorithm 1 groups by
                                       IP/port, fields that don't exist in
                                       this dataset.

    python kafka_stream.py evaluate   Listens on the 'predictions' topic
                                       (written by spark_consumer.py) and
                                       prints a running accuracy.

Run 'produce' and 'evaluate' in separate terminals, alongside spark_consumer.py
(started via spark-submit, in a third terminal). See README.md for the full
run order.

Prerequisites:
    docker compose up -d
    python reproduce.py
"""

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd
from kafka import KafkaProducer, KafkaConsumer

BOOTSTRAP_SERVERS = "localhost:9092"
RAW_TOPIC = "raw-features"
PREDICTIONS_TOPIC = "predictions"
TEST_FILE = "data/test.csv"
DELAY_SECONDS = 0.05  # time between records, to simulate a live stream


def produce():
    df = pd.read_csv(TEST_FILE)
    feature_cols = [c for c in df.columns if c not in ("Class", "record_id")]

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Streaming {len(df)} records to Kafka topic '{RAW_TOPIC}' ...")
    for i, row in df.iterrows():
        record = {
            "record_id": row["record_id"],
            "event_time": datetime.now(timezone.utc).isoformat(),
            "true_label": int(row["Class"]),  # kept only for optional accuracy checking downstream
            "features": {col: float(row[col]) for col in feature_cols},
        }
        producer.send(RAW_TOPIC, value=record)
        if i % 500 == 0:
            print(f"  sent {i} records...")
        time.sleep(DELAY_SECONDS)

    producer.flush()
    producer.close()
    print("Done streaming.")


def evaluate():
    consumer = KafkaConsumer(
        PREDICTIONS_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    correct = 0
    total = 0
    print(f"Listening on '{PREDICTIONS_TOPIC}' for live predictions... (Ctrl+C to stop)\n")
    for msg in consumer:
        record = msg.value
        total += 1
        if record["prediction"] == record["true_label"]:
            correct += 1
        if total % 100 == 0:
            print(f"[{total} records processed]  running accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["produce", "evaluate"])
    args = parser.parse_args()
    (produce if args.mode == "produce" else evaluate)()

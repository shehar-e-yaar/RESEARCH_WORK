import os
import sys
import json
import time
import argparse
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

BASE_DIR = r'c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
TOPIC_NAME = 'ddos-network-traffic'
BROKER = 'localhost:9092'

PAPER_20_FEATURES = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean', 'Packet Length Std',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Destination Port', 'ACK Flag Count'
]

def main():
    parser = argparse.ArgumentParser(description="Real-Time Kafka DDoS Flow Producer")
    parser.add_argument('--rate', type=int, default=2000, help="Target emission rate in flows/second")
    parser.add_argument('--count', type=int, default=10000, help="Total number of flows to replay")
    parser.add_argument('--topic', type=str, default=TOPIC_NAME, help="Kafka topic name")
    parser.add_argument('--broker', type=str, default=BROKER, help="Kafka broker address")
    args = parser.parse_args()

    print("=" * 70)
    print("REAL-TIME NETWORK FLOW PRODUCER (KAFKA)")
    print(f"Broker: {args.broker} | Topic: {args.topic}")
    print(f"Target Rate: {args.rate:,} flows/s | Total Flows: {args.count:,}")
    print("=" * 70)

    try:
        producer = KafkaProducer(
            bootstrap_servers=[args.broker],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            linger_ms=5,
            batch_size=16384,
            acks=1
        )
        print("[OK] Connected successfully to Kafka broker.")
    except NoBrokersAvailable:
        print(f"\n[ERROR] No Kafka broker available at {args.broker}!")
        print("Please start Docker Desktop and run: docker compose up -d")
        sys.exit(1)

    test_file = os.path.join(PROCESSED_DIR, 'test_day_sampled.parquet')
    if not os.path.exists(test_file):
        print(f"[ERROR] Test parquet file not found: {test_file}")
        sys.exit(1)

    print(f"Loading test flows from {test_file}...")
    df = pd.read_parquet(test_file)
    cols = [c for c in PAPER_20_FEATURES if c in df.columns] + ['binary_label', 'Label']
    df_stream = df[cols].iloc[:args.count].copy()
    total_rows = len(df_stream)

    print(f"Starting flow emission ({total_rows:,} records)...")
    start_time = time.time()
    batch_size = max(10, min(500, args.rate // 10))
    target_batch_duration = batch_size / float(args.rate)
    sent_count = 0

    for i in range(0, total_rows, batch_size):
        chunk = df_stream.iloc[i:i + batch_size]
        t0 = time.time()
        for idx, row in chunk.iterrows():
            record = row.to_dict()
            record['timestamp_sent'] = time.time()
            record['flow_id'] = int(idx)
            producer.send(args.topic, value=record)
            sent_count += 1

        elapsed = time.time() - t0
        sleep_time = target_batch_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        if sent_count % 2000 == 0 or sent_count == total_rows:
            curr_rate = sent_count / (time.time() - start_time)
            print(f"  Sent: {sent_count:,}/{total_rows:,} flows | Effective Rate: {curr_rate:,.1f} flows/s")

    producer.flush()
    total_duration = time.time() - start_time
    print("=" * 70)
    print(f"[DONE] Completed streaming {sent_count:,} flows in {total_duration:.2f}s ({sent_count/total_duration:,.1f} flows/s)")
    print("=" * 70)

if __name__ == '__main__':
    main()

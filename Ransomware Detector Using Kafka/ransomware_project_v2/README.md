# Ransomware Detection — Paper Reproduction + Corrected Kafka/Spark Pipeline

6 files total. Two parts: **Part A** reproduces the paper's experiment (no Kafka
needed). **Part B** runs a corrected version of its Kafka/Spark architecture live.

## Files

| File | Purpose |
|---|---|
| `reproduce.py` | Everything for Part A: cleans data, splits, trains all 6 models, compares vs the paper's Table 1, runs 2 hypothesis tests, checks for data leakage, re-tests with a leakage-free split. Saves the model + data used by Part B. |
| `kafka_stream.py` | Part B helper: `produce` streams test data into Kafka; `evaluate` prints live accuracy. |
| `spark_consumer.py` | Part B: loads the frozen model, classifies the live Kafka stream in 60-second windows. |
| `docker-compose.yml` | Starts a local Kafka broker. |
| `requirements.txt` | Python dependencies. |
| `README.md` | This file. |

## Setup

```bash
pip install -r requirements.txt
```
Place `Obfuscated-MalMem2022.csv` in this same folder.

## Part A — Reproduce the paper (no Kafka needed)

```bash
python reproduce.py
```

Prints, in order: the train/test split, all 6 models' metrics vs. the paper's
reported numbers, two hypothesis tests explaining *why* they differ, a
data-leakage check, and a leakage-free re-test. Saves `results_comparison.csv`
and `model_artifacts/xgb_ransomware.json` (needed for Part B).

## Part B — Run the corrected live pipeline

Three terminals, all in this folder.

**Terminal 1 — Kafka:**
```bash
docker compose up -d
```

**Terminal 2 — Spark job (start BEFORE the producer, or it'll miss data):**
```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3 spark_consumer.py
```

**Terminal 3 — stream data in:**
```bash
python kafka_stream.py produce
```

Watch Terminal 2 for windowed prediction tables, printed every ~60 seconds.

**Optional Terminal 4 — live accuracy:**
```bash
python kafka_stream.py evaluate
```

**Stop everything:**
```bash
docker compose down
```
(Ctrl+C in the other terminals.)

## Windows-specific fixes (if Part B fails)

These were needed to get `spark_consumer.py` running on Windows — check these
first if you hit an error:

1. **`HADOOP_HOME and hadoop.home.dir are unset`** — Spark needs a Windows-only
   helper file. Download `winutils.exe` + `hadoop.dll` (matching Hadoop 3.3.x,
   e.g. from `github.com/cdarlint/winutils`) into `C:\hadoop\bin`, then:
   ```powershell
   $env:HADOOP_HOME = "C:\hadoop"
   $env:PATH = "$env:HADOOP_HOME\bin;$env:PATH"
   ```

2. **`Python worker exited unexpectedly (crashed)` + `EOFException`** — a known
   PySpark bug on Windows with Python 3.12/3.13. Fix: create a virtual
   environment using **Python 3.11** specifically, install
   `requirements.txt` into it, and point Spark at it:
   ```powershell
   py -3.11 -m venv venv311
   .\venv311\Scripts\Activate.ps1
   pip install -r requirements.txt
   $env:PYSPARK_PYTHON = "python"
   $env:PYSPARK_DRIVER_PYTHON = "python"
   ```

3. **`ModuleNotFoundError: No module named 'pyarrow'`** — `pip install pyarrow`
   (already in `requirements.txt`, but easy to miss if installed by hand).

4. **`Failed to resolve reference "docker.io/bitnami/kafka..."`** — Bitnami
   paywalled that image in 2025. This project already uses the official
   `apache/kafka` image instead, so this shouldn't occur — but if you swap
   images, keep that in mind.

## How the pieces connect

- `reproduce.py` saves the model to `model_artifacts/xgb_ransomware.json` →
  `spark_consumer.py` loads that exact path. Trained once, loaded read-only
  forever after.
- `kafka_stream.py produce` and `spark_consumer.py` are linked only by the
  Kafka topic **name** `raw-features` — no shared file. Same for
  `spark_consumer.py` → `predictions` topic → `kafka_stream.py evaluate`.
- Each record uses `record_id` (the dataset's malware-family/sample name)
  instead of the paper's `source_ip`/`source_port`, since those fields don't
  exist in this dataset.

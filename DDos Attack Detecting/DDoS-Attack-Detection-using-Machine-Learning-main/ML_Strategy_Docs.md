
-------------------------------------------------------------------------------

# Project Strategy and Machine Learning (ML) Roadmap 🧠🛡️

This document provides a detailed explanation of the project's engineering and data science perspectives, answering questions like "Why are we following these steps?" and "How does the system work?". Our project covers the processes of **data preprocessing, visualization tools, machine learning (ML), deep learning (DL), and a live monitoring interface.**

-------------------------------------------------------------------------------

## 1. Phase 1: Data Pipeline and Why It Is Essential

There is a golden rule in machine learning: **"Garbage in, Garbage out."**
Network traffic is inherently "dirty" data in its raw form. Our pipeline (`src/preprocessing.py`) performs these critical tasks:

*   **Infinity and NaN Cleaning:** Infinite division errors or missing (NaN) packets in the data cause the model to crash during mathematical calculations. These are automatically detected and filtered.

*   **Anti-Memorization "Feature Selection":** Columns like Source IP Address, Port Number, or Timestamp are not patterns that define DDoS; they are merely the identity of the packet. If we provide these to the model, it memorizes **specific IPs** (Overfitting) instead of learning the **behavior**. We forced the model to focus on the statistical behavior of the packets.

*   **Dynamic Scaling:**
    *   **RobustScaler:** Used for traditional ML models (RF, XGB) due to its resilience against outliers.
    *   **MinMaxScaler:** Preferred for the CNN pipeline to ensure neural networks train more stably within the 0-1 range.

----------------------------------------------------------------------------------

### Batch Processing Architecture
When the datasets (2017 and 2019) are combined, they reach massive sizes. Attempting to load all the data into RAM results in an **Out of Memory (OOM)** error. Instead, we built a **Batch Processing** system that handles each file sequentially. This architecture allows us to process data at Big Data standards even on low-end hardware.

----------------------------------------------------------------------------------

## 2. Visualization Strategy (3 Main Layers)

Before feeding the data into an algorithm, we built a visualization layer to make the process transparent:

1.  **`preprocessing_summary_visual.py` (Pipeline Analysis):** Reports how much data was discarded during cleaning and whether the system suffered any data loss.
2.  **`master_visualization.py` (Bird's Eye Comparison):** Compares the class distributions (Normal vs. Attack) of all datasets in a giant bar chart.
3.  **`visualization.py` (Detailed EDA):** Extracts "Correlation Heatmaps," identifies the **Top 10 Most Suspicious Features** for detecting DDoS, and plots Boxplot analyses.

----------------------------------------------------------------------------------

## 3. Data Splitting and "Zero-Day" Test Strategy

We measure the success of our models not just with training data, but with real-world scenarios:

*   **📚 Train Set:** All of CICIDS2017 data and 33% of CICDDoS2019 data are combined to ensure the model sees a "broad spectrum of attacks."
*   **🤔 Validation Set:** A 20% portion used during training to monitor whether the model is overfitting.
*   **📝 Zero-Day Test:** The remaining 67% of CICDDoS2019 data, which the model has **never seen** during training. If the model can correctly predict these "next-generation" attacks, its ability to sense real-world threats (Generalization) is proven.

----------------------------------------------------------------------------------

## 4. System Architecture and Engineering Decisions

### Why was CICIDS2017 Chosen for Training?
In cybersecurity, evolution flows forward. The 2017 attacks (SYN Flood, etc.) represent the "alphabet" (ancestors) of modern attacks. The 2019 attacks are more complex derivatives. Teaching the AI the alphabet first and then expecting it to solve difficult texts (2019) is the foundation of our cybersecurity vision.

### Why Do We Persist with .CSV Output?
Although our code can read both Parquet and CSV, it outputs processed data as CSV. The reasons are:
1.  **Transparency:** You can visually inspect the data in VS Code to ensure NaN values are properly handled.
2.  **Universality:** CSV is the common language of data science; it can be analyzed in any language from Java to R.
3.  **Debug Speed:** Finding and fixing a faulty row takes seconds in a CSV format.

----------------------------------------------------------------------------------

## 5. PHASE 2 & 3: Model Implementations (ML & DL)

### ⚖️ Handling Class Imbalance
To break the dominance of "Normal" packets in the data:
*   **Random Forest:** Used `class_weight='balanced'` to increase the penalty for missing attack packets.
*   **XGBoost:** Used `scale_pos_weight` to mathematically weight the minority class.
*   **Future Strategy:** If these methods prove insufficient, **SMOTE (Synthetic Minority Over-sampling Technique)** can be used to generate synthetic twins of the attack packets.

### 🧠 1D-CNN (Deep Learning)
Beyond traditional models, a two-layer Conv1D structure was established to filter the abstract and hidden connections between network packets. `EarlyStopping` was implemented to prevent overfitting, resulting in the most stable model.

----------------------------------------------------------------------------------

## 6. PHASE 4: Deployment and Dashboard (Streamlit)

As the final stage of the project, a live monitoring panel was provided via `src/app_streamlit.py`:
*   **Live Analysis:** Instantly scans traffic files (CSV) uploaded by the user.
*   **XAI (Explainable AI):** Visualizes why the model flagged a "Threat" and which network features were the primary triggers.
*   **Performance Archive:** Transparently lists all metrics (Precision, Recall, F1, Confusion Matrix) from the models' training phase.

----------------------------------------------------------------------------------

## 7. Performance Metrics

The system is evaluated using the following metrics:
*   **Recall:** The rate of catching actual attacks (Critical).
*   **FPR (False Positive Rate):** The rate of blocking normal traffic (Must be minimized).
*   **Inference Time:** The speed of analyzing a single packet (Vital for live systems).

*(The project is currently running stably across all phases. Models are stored in `models/`, and visual results are kept in the `metrics/` folder.)*
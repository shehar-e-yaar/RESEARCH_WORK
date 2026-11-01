# 🚀 Installation & Execution Guide

This document provides step-by-step instructions to set up the environment and run the DDoS Detection System.

## 1. Environment Setup

### Prerequisites
- Python 3.9+ installed.
- (Optional but recommended) A virtual environment.

### Install Dependencies
Run the following command to install all required libraries (TensorFlow, XGBoost, Streamlit, Plotly, etc.):
```bash
pip install -r requirements.txt
```

## 2. Data Preparation & Preprocessing

### 2.1. Prepare Raw Data
1. Create a folder named `data/raw` in the project root if it doesn't exist.
2. Place your raw `.csv` or `.parquet` datasets (e.g., CICIDS2017 or CICDDoS2019) into `data/raw/`.

### 2.2. Run Preprocessing
This script cleans column names, handles infinity/NaN values, encodes labels, and removes identity columns to prevent data leakage.
```bash
python src/preprocessing.py
```
*Note: Cleaned files will be saved to `data/processed/` automatically.*

## 3. Directory Structure
Ensure your project folder looks like this:
```text
DDos_attack/
├── data/
│   ├── raw/           # Place raw .csv / .parquet files here
│   └── processed/     # Cleaned files are generated here
├── metrics/           # Performance JSONs and PNGs
├── models/            # Trained .h5, .pkl and scalers
├── output/
│   └── visualizations/# Preprocessing and comparison charts
├── src/
│   ├── preprocessing.py # Data cleaning script
│   ├── train_ml.py      # Traditional ML training script
│   ├── train_cnn.py     # Deep Learning training script
│   ├── master_visualization.py # Global dataset comparison
│   ├── preprocessing_summary_visual.py # Data loss report
│   └── app_streamlit.py # Interactive GUI
└── ddos.png           # Dashboard Sidebar Icon
```

## 3. Running the Models

### Training ML Models (XGBoost & Random Forest)
This script handles the hybrid data mixing and trains the tree-based models:
```bash
python src/train_ml.py
```

### Training Deep Learning (1D-CNN)
This script applies MinMaxScaler and trains the convolutional architecture:
```bash
python src/train_cnn.py
```

## 5. Visualizing Data & Results

### Preprocessing Report
To see how much data was dropped and the feature reduction stats:
```bash
python src/preprocessing_summary_visual.py
```

### Master Dataset Comparison
To see the distribution of Normal vs DDoS packets across all processed datasets:
```bash
python src/master_visualization.py
```

## 6. Launching the Interactive GUI (Sunum İçin)
To show the project to the professor or for testing, use the Streamlit dashboard:
```bash
streamlit run src/app_streamlit.py
```

## 7. Key File Descriptions
- **`models/scaler.pkl`**: Scaler used for ML models.
- **`models/scaler_cnn.pkl`**: Scaler used for the CNN model.
- **`models/train_columns.pkl`**: The exact feature list needed for inference (ensures column alignment).
- **`metrics/`**: Every time you train, this folder updates with the latest Accuracy/Recall scores and Confusion Matrices.
- **`output/visualizations/`**: Contains global comparison charts and preprocessing dashboards.

## ⚠️ Important Notes for Group Members
- **Data Alignment:** If you use a new dataset, the system will automatically try to map the 2019 column names to the 2017 schema.
- **Leakage Prevention:** Do not modify the `train_test_split` logic in Step 1 of the scripts; it is crucial for ensuring the 2019 test data remains "unseen".
- **Hardware:** CNN training is optimized for CPU on Windows. If you have a GPU, ensure you are running via WSL2 for hardware acceleration.

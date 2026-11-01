import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import json

RAW_DATA_DIR = r"data\raw"
PROCESSED_DATA_DIR = r"data\processed"

def load_data(filepath):
    print(f"[1/5] Data is loading: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data not found! Please make sure the file is at {filepath} location.")
    
    if filepath.lower().endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath, low_memory=False)
        
    print(f"Data loaded. Shape: {df.shape}")
    return df

def clean_columns_and_values(df):
    print("[2/5] Column names are cleaned and values are cleaned...")
    df.columns = df.columns.str.strip()
    
    # Standardizing column names and handling anomalous values
    df.replace([np.inf, -np.inf, 'Infinity', 'infinity', 'NaN', 'nan'], np.nan, inplace=True)
    
    # Drop rows with NaN
    initial_shape = df.shape
    df.dropna(inplace=True)
    print(f"NaN and Infinity containing rows are deleted. {initial_shape[0] - df.shape[0]} rows are deleted.")
    return df

def feature_selection(df):
    print("[3/5] Identity and time columns that will cause memorization are removed...")
    # NOTE: Destination Port and Protocol are preserved as required network flow features for DDoS detection
    cols_to_drop = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Timestamp', 'SimillarHTTP']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    df.drop(columns=existing_cols_to_drop, inplace=True)
    print(f"Removed columns: {existing_cols_to_drop}")
    return df

def encode_labels(df):
    print("[4/5] Label column is encoded (BENIGN -> 0, DDoS -> 1)...")
    if 'Label' not in df.columns:
        raise ValueError("'Label' column not found!")
    
    print(f"Original Label Distribution:\n{df['Label'].value_counts()}")
    # Using str.contains to match benign
    df['Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
    
    print(f"Encoded Label Distribution (0: Normal, 1: Attack):\n{df['Label'].value_counts()}")
    
    # Ensure all other columns are numeric
    print("[5/5] Converting features to numeric...")
    cols_except_label = [col for col in df.columns if col != 'Label']
    df[cols_except_label] = df[cols_except_label].apply(pd.to_numeric, errors='coerce')
    
    initial_rows = len(df)
    df.dropna(inplace=True) # Drop any rows that became NaN after coercion
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows due to non-numeric values.")
        
    return df

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Find all raw CSV and Parquet files recursively (including subdirectories like 01-12 and 03-11)
    raw_files = (
        glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.csv"), recursive=True) +
        glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.parquet"), recursive=True)
    )
    
    if len(raw_files) == 0:
        print(f"No CSV or Parquet files found in {RAW_DATA_DIR}")
        return
        
    summary_data = []
        
    for filepath in raw_files:
        filename = os.path.basename(filepath)
        
        # Include parent subdirectory name (e.g. 01-12 or 03-11) as prefix to avoid filename collisions
        rel_dir = os.path.relpath(os.path.dirname(filepath), RAW_DATA_DIR)
        prefix = f"{rel_dir}_" if rel_dir and rel_dir != "." else ""
        base_name = os.path.splitext(filename)[0]
        processed_filename = f"{prefix}{base_name}_cleaned.csv"
            
        processed_filepath = os.path.join(PROCESSED_DATA_DIR, processed_filename)
        
        print("\n" + "="*50)
        print(f"Processing File: {filename} (from {rel_dir if rel_dir else 'root'})")
        print("="*50)
        
        try:
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            # Process large files in chunks to avoid Out-Of-Memory errors
            if file_size_mb > 150 and not filepath.lower().endswith(".parquet"):
                print(f"[1/5] Large file detected ({file_size_mb:.1f} MB). Processing in chunks...")
                first_chunk = True
                rows_before = 0
                rows_after = 0
                cols_before = 0
                cols_after = 0
                
                for chunk in pd.read_csv(filepath, chunksize=100000, low_memory=False):
                    if first_chunk:
                        cols_before = chunk.shape[1]
                    rows_before += chunk.shape[0]
                    
                    chunk = clean_columns_and_values(chunk)
                    chunk = feature_selection(chunk)
                    chunk = encode_labels(chunk)
                    
                    if first_chunk:
                        cols_after = chunk.shape[1]
                        chunk.to_csv(processed_filepath, index=False, mode='w')
                        first_chunk = False
                    else:
                        chunk.to_csv(processed_filepath, index=False, mode='a', header=False)
                        
                    rows_after += chunk.shape[0]
                    
                removed_nan_inf_rows = rows_before - rows_after
                print(f"Total processed: {rows_after:,} rows (from {rows_before:,} initial rows).")
            else:
                df = load_data(filepath)
                rows_before = df.shape[0]
                cols_before = df.shape[1]
                
                df = clean_columns_and_values(df)
                df = feature_selection(df)
                df = encode_labels(df)
                           
                rows_after = df.shape[0]
                cols_after = df.shape[1]
                removed_nan_inf_rows = rows_before - rows_after
                
                df.to_csv(processed_filepath, index=False)
            
            summary_data.append({
                "file_short": f"{prefix}{filename}",
                "rows_before": rows_before,
                "rows_after": rows_after,
                "cols_before": cols_before,
                "cols_after": cols_after,
                "removed_nan_inf_rows": removed_nan_inf_rows
            })
            print(f"Successfully! Cleaned data saved to: {processed_filepath}")
            
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
            
    if summary_data:
        summary_path = os.path.join(PROCESSED_DATA_DIR, "preprocessing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"\n[+] Preprocessing summary saved to {summary_path}")

if __name__ == "__main__":
    main()

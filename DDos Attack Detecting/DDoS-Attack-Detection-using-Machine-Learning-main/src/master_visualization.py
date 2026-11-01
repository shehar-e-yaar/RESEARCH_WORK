import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants defining the directory paths
PROCESSED_DATA_DIR = r"data\processed"
BASE_OUTPUT_DIR = r"output\visualizations"

def main():
    # Find all processed CSV files ending with '_cleaned.csv' in the processed directory
    processed_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*_cleaned.csv"))
    
    # If no files are found, gracefully exit with a warning
    if len(processed_files) == 0:
        print(f"No processed CSV files found in {PROCESSED_DATA_DIR}")
        return
        
    print(f"\n[{len(processed_files)}] Datasets found. Drawing Master Comparison Chart (Fast reading labels only)...")
    
    summary_data = []
    
    # Iterate through each processed file to count labels
    for filepath in processed_files:
        filename = os.path.basename(filepath)
        dataset_name = filename.replace("_cleaned.csv", "")
        # Truncate very long names for the X-axis so they fit nicely without overlapping
        short_name = dataset_name[:20] + ".." if len(dataset_name) > 20 else dataset_name
        
        try:
            # Memory optimization: Only load the 'Label' column to make it lightning-fast for huge datasets!
            df_label = pd.read_csv(filepath, usecols=['Label'])
            
            normal_count = (df_label['Label'] == 0).sum()
            ddos_count = (df_label['Label'] == 1).sum()
            
            summary_data.append({
                'Dataset': short_name,
                'Normal Traffic (0)': normal_count,
                'DDoS Attack (1)': ddos_count,
                'Total': normal_count + ddos_count
            })
        except Exception as e:
            print(f"Error reading {filename}: {e}")
        
    # Build a DataFrame containing the summary of all datasets
    summary_df = pd.DataFrame(summary_data)
    
    # Sort them by total size so the chart looks structured (descending stairs format)
    summary_df = summary_df.sort_values(by='Total', ascending=False)
    
    # Display the summary table in the Terminal
    print("\n" + "="*70)
    print("MASTER DATASET COMPARISON TABLE (Summary of All Data)")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)
    
    # Create the output directory mapping safely
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Restructure the DataFrame for Seaborn's grouped barplot (using Melt function)
    melted_df = summary_df.melt(id_vars='Dataset', value_vars=['Normal Traffic (0)', 'DDoS Attack (1)'], 
                                var_name='Traffic Type', value_name='Packet Count')
    
    plt.figure(figsize=(16, 8))
    # Elegant color palette reflecting 'Safe' (Green) vs 'Danger' (Red)
    sns.barplot(data=melted_df, x='Dataset', y='Packet Count', hue='Traffic Type', palette=['#2ecc71', '#e74c3c'])
    
    plt.title("Master Comparison: Normal vs DDoS Packets Across All Datasets", fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylabel("Number of Packets (Count)", fontsize=12, fontweight='bold')
    plt.xlabel("Dataset File Name", fontsize=12, fontweight='bold')
    
    # Add horizontal grid lines to compare bar heights easier visually
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    master_chart_path = os.path.join(BASE_OUTPUT_DIR, "MASTER_dataset_comparison.png")
    # Save the huge high-quality comparative image explicitly
    plt.savefig(master_chart_path, dpi=300)
    plt.close()
    
    print(f"\n✅ Master Chart successfully saved to -> {master_chart_path}")

if __name__ == "__main__":
    main()

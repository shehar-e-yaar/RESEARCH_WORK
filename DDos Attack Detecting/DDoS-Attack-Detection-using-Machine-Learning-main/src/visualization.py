import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse

# Constants defining the directory paths 
PROCESSED_DATA_DIR = r"data\processed"
BASE_OUTPUT_DIR = r"output\visualizations"

def main():
    # Setup argument parser to read inputs from the terminal 
    parser = argparse.ArgumentParser(description="Visualize processed DDoS datasets.")
    parser.add_argument("--file", type=str, default="all", help="Specific filename to visualize (e.g., 'Syn-training_cleaned.csv'). Default is 'all'.")
    args = parser.parse_args()

    # Determine which file(s) we are visualizing based on user input
    if args.file != "all":
        target_path = os.path.join(PROCESSED_DATA_DIR, args.file)
        if os.path.exists(target_path):
            processed_files = [target_path]
        else:
            print(f"Error: The specific file '{args.file}' was not found in {PROCESSED_DATA_DIR}")
            return
    else:
        # Find all processed CSV files ending with '_cleaned.csv' in the processed directory
        processed_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*_cleaned.csv"))
    
    # If no files are found, gracefully exit with a warning
    if len(processed_files) == 0:
        print(f"No processed CSV files found in {PROCESSED_DATA_DIR}")
        return
        
    # Iterate through each processed file to generate separate EDA charts
    for filepath in processed_files:
        filename = os.path.basename(filepath)
        
        # Extract the base dataset name by removing the suffix
        dataset_name = filename.replace("_cleaned.csv", "")
        
        # Create a specific output directory for this dataset's plots
        output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print(f"[{dataset_name}] Analyzing and plotting data...")
        print("="*80)
        
        # Load the processed dataset into a Pandas DataFrame
        df = pd.read_csv(filepath)
        
        # Configure Pandas to display more columns and print the first 5 rows to console
        pd.set_option('display.max_columns', 10)
        print("\nFIRST 5 ROWS (Table View):")
        print(df.head())
        
        # Print a short high-level summary of the dataset's size
        print("\nDATA SCIENCE SUMMARY:")
        print(f"- Total Rows: {len(df):,}")
        print(f"- Total Columns (Features): {len(df.columns)}")
        
        # ---------------------------------------------------------
        # Plot 1: Label Distribution (Bar Chart)
        # Shows the distribution balance between Normal Traffic (0) and DDoS (1)
        # ---------------------------------------------------------
        print("\nPlot 1: Drawing Attack Distribution...")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Label', palette='viridis', hue='Label', legend=False)
        plt.title(f"Attack vs Normal Traffic Distribution (0: Normal, 1: DDoS)\nDataset: {dataset_name}")
        plt.savefig(os.path.join(output_dir, "1_label_distribution.png"))
        plt.close()
        
        # ---------------------------------------------------------
        # Plot 2: Correlation Heatmap (Top 10 Features)
        # Identifies and graphs the top 10 features most strongly correlated with the Label
        # ---------------------------------------------------------
        print("Plot 2: Drawing Correlation Heatmap (Top 10 Features)...")
        # Calculate absolute correlations against 'Label' and sort them descending
        correlations = df.corr(numeric_only=True)['Label'].abs().sort_values(ascending=False)
        
        # Extract the top 10 most correlated features (excluding the Label itself at index 0)
        top_features = correlations.index[1:11]
        
        if len(top_features) > 0:
            plt.figure(figsize=(12, 10))
            sns.heatmap(df[top_features].corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f"Top 10 Features Correlated with DDoS\nDataset: {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "2_top_10_features_correlation.png"))
            plt.close()
        
        # ---------------------------------------------------------
        # Plot 3: Boxplot of the Most Defining Feature
        # Highlights how the #1 most correlated feature varies between Normal and DDoS
        # ---------------------------------------------------------
        if len(top_features) > 0:
            top_feature_name = top_features[0] 
            print(f"Plot 3: Drawing boxplot for the most defining feature '{top_feature_name}'...")
            
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Label', y=top_feature_name, data=df, palette='Set2', hue='Label', legend=False)
        plt.title(f"{top_feature_name} : Normal (0) vs DDoS (1)\nDataset: {dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_top_feature_boxplot.png"))
        plt.close()
        
        print(f"\n[{dataset_name}] Visualizations successfully saved to -> '{output_dir}'")
        
    print("\nAnalysis of all files completed successfully!")

if __name__ == "__main__":
    main()

# fonde i csv e stampiamo delle statistiche utili

import pandas as pd
import re
import os

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    return text  # Return original value if not a string

def calculate_dataset_stats(df, source_name):
    """Calculate and print statistics for individual dataset"""
    # Convert days_resolution to numeric
    df['days_resolution'] = pd.to_numeric(df['days_resolution'], errors='coerce')
    df = df.dropna(subset=['days_resolution'])
    
    # Calculate percentiles
    percentile_75 = df['days_resolution'].quantile(0.75)
    
    # Assign labels
    df['label'] = df['days_resolution'].apply(lambda x: 0 if x <= percentile_75 else 1)
    
    print(f"\nStatistics for {source_name}:")
    print(f"Total rows: {len(df)}")
    print(f"75th percentile: {percentile_75:.2f} days")
    print(f"Label 0 (≤{percentile_75:.2f} days): {sum(df['label'] == 0)}")
    print(f"Label 1 (>{percentile_75:.2f} days): {sum(df['label'] == 1)}")
    print("\nDays Resolution Statistics:")
    print(df['days_resolution'].describe())
    print("-" * 50)
    
    return df, percentile_75

# List of file names
files = [
    'Eclipse', 'FreeDesktop', 'Gentoo', 'KDE', 'LibreOffice',
    'LiveCode', 'NetBeans', 'Novell', 'OpenOffice', 'OpenXchange', 'W3C'
]

# Initialize an empty list to store dataframes and a dict for percentiles
dfs = []
percentiles = {}

# Read and process each file
for file in files:
    csv_file = f'csv_output_from_json/{file}_data.csv'
    if os.path.exists(csv_file):
        try:
            print(f"\nProcessing {file}...")
            df = pd.read_csv(csv_file)
            
            # Clean text columns
            text_columns = ['short_desc', 'comments', 'product', 'priority']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].apply(clean_text)
            
            # Remove rows where all columns are NaN
            df = df.dropna(how='all')
            
            # Add source column
            df['source'] = file
            
            # Calculate statistics for this dataset
            processed_df, p75 = calculate_dataset_stats(df, file)
            
            # Store results
            dfs.append(processed_df)
            percentiles[file] = p75
            
            print(f"Successfully processed {file} - {len(df)} rows")
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    else:
        print(f"File not found: {csv_file}")

if not dfs:
    print("No data frames were created. Check your input files and paths.")
    exit()

# Concatenate all dataframes
print("\nMerging dataframes...")
merged_df = pd.concat(dfs, ignore_index=True)
merged_df = merged_df.drop_duplicates()

# Calculate merged dataset statistics
merged_df['days_resolution'] = pd.to_numeric(merged_df['days_resolution'], errors='coerce')
merged_df = merged_df.dropna(subset=['days_resolution'])
percentile_75_merged = merged_df['days_resolution'].quantile(0.75)

# Assign labels for merged dataset
merged_df['label'] = merged_df['days_resolution'].apply(lambda x: 0 if x <= percentile_75_merged else 1)

# Save the cleaned and merged data
output_file = 'merged_processed_labeled.csv'
merged_df.to_csv(output_file, index=False)

# Print final statistics
print("\nFINAL STATISTICS FOR ALL DATASETS:")
print("\nIndividual Dataset 75th Percentiles:")
for file, perc in percentiles.items():
    print(f"{file}: {perc:.2f} days")

print(f"\nMerged Dataset 75th Percentile: {percentile_75_merged:.2f} days")
print(f"Total rows in merged dataset: {len(merged_df)}")
print(f"Label 0 (≤{percentile_75_merged:.2f} days): {sum(merged_df['label'] == 0)}")
print(f"Label 1 (>{percentile_75_merged:.2f} days): {sum(merged_df['label'] == 1)}")

print("\nRows per source in merged dataset:")
print(merged_df['source'].value_counts())

print("\nMerged Dataset Days Resolution Statistics:")
print(merged_df['days_resolution'].describe())

# Create a dictionary to store row counts
row_counts = {df['source'].iloc[0]: len(df) for df in dfs}

# Save percentiles and row counts to a separate CSV for reference
percentiles_df = pd.DataFrame([
    {
        "source": source,
        "percentile_75": perc,
        "total_rows": row_counts[source],
        "rows_label_0": sum(dfs[i]['label'] == 0) if source == dfs[i]['source'].iloc[0] else None,
        "rows_label_1": sum(dfs[i]['label'] == 1) if source == dfs[i]['source'].iloc[0] else None,
    } 
    for i, (source, perc) in enumerate(percentiles.items())
])

# Add percentage columns
percentiles_df['percent_label_0'] = (percentiles_df['rows_label_0'] / percentiles_df['total_rows'] * 100).round(2)
percentiles_df['percent_label_1'] = (percentiles_df['rows_label_1'] / percentiles_df['total_rows'] * 100).round(2)

# Reorder columns for better readability
percentiles_df = percentiles_df[[
    'source', 
    'total_rows', 
    'percentile_75',
    'rows_label_0',
    'percent_label_0',
    'rows_label_1',
    'percent_label_1'
]]

# Save to CSV
percentiles_df.to_csv('dataset_percentiles.csv', index=False)

# Print the detailed statistics
print("\nDetailed statistics per dataset saved to dataset_percentiles.csv:")
print("\nSummary of datasets:")
print(percentiles_df.to_string(index=False))
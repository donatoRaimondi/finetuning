import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_balanced_datasets(input_dir="", merged_file="merged_processed_labeled.csv", sizes=[1000, 2000, 5000, 10000]):
    """
    Create multiple balanced training datasets of different sizes,
    along with a single fixed validation and test dataset.
    """
    output_dir = "balanced_datasets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Reading merged dataset...")
    df = pd.read_csv(os.path.join(input_dir, merged_file))
    
    # Rimuovere le fonti con troppi pochi dati (W3C e OpenXchange)
    df = df[~df['source'].isin(['W3C', 'OpenXchange'])]
    
    min_samples_per_source_label = float('inf')
    sources = df['source'].unique()
    
    print("\nMinimi campioni per source e label:")
    for source in sources:
        for label in [0, 1]:
            samples = len(df[(df['source'] == source) & (df['label'] == label)])
            print(f"Source: {source}, Label: {label}, Samples: {samples}")
            min_samples_per_source_label = min(min_samples_per_source_label, samples)
    
    # Imposta un limite minimo per evitare dataset troppo piccoli
    min_samples_per_source_label = max(500, min_samples_per_source_label)
    
    max_balanced_size = min_samples_per_source_label * 2 * len(sources)
    valid_sizes = [size for size in sizes if size <= max_balanced_size]
    if max_balanced_size not in valid_sizes:
        valid_sizes.append(max_balanced_size)
    
    print("Creating fixed validation and test sets...")
    samples_per_source_label = min_samples_per_source_label // 2
    validation_dfs, test_dfs = [], []
    
    for source in sources:
        source_df = df[df['source'] == source]
        source_val_test = []
        
        for label in [0, 1]:
            label_df = source_df[source_df['label'] == label]
            sampled_df = label_df.sample(n=samples_per_source_label, random_state=42, replace=True)
            source_val_test.append(sampled_df)
        
        val_test_df = pd.concat(source_val_test)
        val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42, stratify=val_test_df[['source', 'label']])
        
        validation_dfs.append(val_df)
        test_dfs.append(test_df)
    
    validation_df = pd.concat(validation_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    validation_df.to_csv(os.path.join(output_dir, "balanced_validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "balanced_test.csv"), index=False)
    
    for target_size in valid_sizes:
        print(f"Creating balanced training dataset of size {target_size}...")
        samples_per_source = target_size // len(sources)
        samples_per_source_label = samples_per_source // 2
        balanced_dfs = []
        
        for source in sources:
            source_df = df[df['source'] == source]
            source_balanced = []
            
            for label in [0, 1]:
                label_df = source_df[source_df['label'] == label]
                sampled_df = label_df.sample(n=samples_per_source_label, random_state=42, replace=True)
                source_balanced.append(sampled_df)
            
            balanced_dfs.append(pd.concat(source_balanced))
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        balanced_df.to_csv(os.path.join(output_dir, f"balanced_train_{target_size}.csv"), index=False)
    
    print("✅ Balanced datasets created successfully!")

if __name__ == "__main__":
    create_balanced_datasets(sizes=[1000, 2000, 5000, 10000])

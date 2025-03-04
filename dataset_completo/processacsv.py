import pandas as pd
import numpy as np
import re
import os

def clean_text(text):
    """Clean text data removing URLs, special characters and extra spaces"""
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+', '', text)
        #text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else np.nan
    return np.nan

def handle_missing_values(df, strategy='conservative'):
    """Handle missing values based on different strategies"""
    original_size = len(df)
    changes = {}
    df = df.copy()
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    text_columns = df.select_dtypes(include=['object']).columns
    
    if strategy == 'conservative':
        for col in numeric_columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                if col == 'days_resolution':
                    quantile_75 = df[col].quantile(0.75)
                    df[col].fillna(quantile_75, inplace=True)
                    changes[col] = f"Filled {missing_count} NaN with median ({quantile_75:.2f})"
                else:
                    df[col].fillna(0, inplace=True)
                    changes[col] = f"Filled {missing_count} NaN with 0"
        
        for col in text_columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col].fillna('Unknown', inplace=True)
                changes[col] = f"Filled {missing_count} NaN with 'Unknown'"
    else:
        missing_threshold = len(df.columns) * 0.5
        before_rows = len(df)
        df = df.dropna(thresh=missing_threshold)
        after_rows = len(df)
        changes['removed_rows'] = f"Removed {before_rows - after_rows} rows with >50% missing values"
        
        for col in numeric_columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                quantile_75 = df[col].quantile(0.75)
                df[col].fillna(quantile_75, inplace=True)
                changes[col] = f"Filled {missing_count} NaN with mean ({quantile_75:.2f})"
        
        for col in text_columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                changes[col] = f"Filled {missing_count} NaN with mode value"
    
    duplicates = df.duplicated()
    if duplicates.any():
        df = df.drop_duplicates()
        changes['duplicates'] = f"Removed {duplicates.sum()} duplicate rows"
    
    final_size = len(df)
    rows_removed = original_size - final_size
    
    return df, {
        'original_rows': original_size,
        'final_rows': final_size,
        'rows_removed': rows_removed,
        'changes': changes
    }

def process_and_label_datasets(input_dir='csv_output_from_json', strategy='conservative'):
    """Process datasets and assign labels based on merged dataset's 75th percentile"""
    output_dir = "processed_labeled_datasets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prima fase: raccolta e pulizia di tutti i dataset
    print("Phase 1: Collecting and cleaning datasets...")
    cleaned_dfs = []
    processing_reports = {}
    
    files = [f for f in os.listdir(input_dir) if f.endswith('_data.csv')]
    
    # Primo passaggio: pulizia e raccolta dati
    for file in files:
        source_name = file.replace('_data.csv', '')
        try:
            print(f"\nProcessing {source_name}...")
            
            # Legge e pulisce il dataset
            df = pd.read_csv(os.path.join(input_dir, file))
            
            # Pulisce le colonne di testo
            text_columns = ['short_desc', 'comments', 'product', 'priority']
            for col in text_columns:
                if col in df.columns:
                    print(f"Cleaning text in column: {col}")
                    df[col] = df[col].apply(clean_text)
            
            # Gestisce i valori mancanti
            df, report = handle_missing_values(df, strategy)
            processing_reports[source_name] = report
            
            # Converte days_resolution a numerico
            df['days_resolution'] = pd.to_numeric(df['days_resolution'], errors='coerce')
            df = df.dropna(subset=['days_resolution'])
            
            # Aggiunge la colonna source
            df['source'] = source_name
            
            cleaned_dfs.append(df)
            
        except Exception as e:
            print(f"Error processing {source_name}: {str(e)}")
    
    if not cleaned_dfs:
        print("No data frames were created. Check your input files and paths.")
        return
    
    # Seconda fase: calcolo del 75° percentile sul dataset merged
    print("\nPhase 2: Calculating global 75th percentile...")
    merged_df = pd.concat(cleaned_dfs, ignore_index=True)
    global_75th_percentile = merged_df['days_resolution'].quantile(0.75)
    print(f"Global 75th percentile: {global_75th_percentile:.2f} days")
    
    # Terza fase: assegnazione labels e salvataggio
    print("\nPhase 3: Assigning labels and saving datasets...")
    final_dfs = []
    
    # Processa ogni dataset usando la soglia globale
    for df in cleaned_dfs:
        source_name = df['source'].iloc[0]
        
        # Assegna labels usando la soglia globale
        df['label'] = (df['days_resolution'] > global_75th_percentile).astype(int)
        
        # Salva il dataset singolo
        output_file = os.path.join(output_dir, f"{source_name}_processed_labeled.csv")
        df.to_csv(output_file, index=False)
        
        print(f"\nStatistics for {source_name}:")
        print(f"Total samples: {len(df)}")
        print(f"Label 0 (≤{global_75th_percentile:.2f} days): {sum(df['label'] == 0)}")
        print(f"Label 1 (>{global_75th_percentile:.2f} days): {sum(df['label'] == 1)}")
        
        final_dfs.append(df)
    
    # Salva il dataset merged con le labels
    final_merged_df = pd.concat(final_dfs, ignore_index=True)
    merged_output = os.path.join(output_dir, "merged_processed_labeled.csv")
    final_merged_df.to_csv(merged_output, index=False)
    
    # Crea e salva il report finale
    stats_data = []
    for df in final_dfs:
        source = df['source'].iloc[0]
        total_samples = len(df)
        samples_label_0 = sum(df['label'] == 0)
        samples_label_1 = sum(df['label'] == 1)

        stats_data.append({
            'source': source,
            'total_samples': len(df),
            'samples_label_0': sum(df['label'] == 0),
            'samples_label_1': sum(df['label'] == 1),
            'percent_label_0': round((samples_label_0 / total_samples * 100), 2),  # Corretto qui
            'percent_label_1': round((samples_label_1 / total_samples * 100), 2),  # Corretto qui
            'min_days': df['days_resolution'].min(),
            'max_days': df['days_resolution'].max(),
            '75th_percentile': df['days_resolution'].mean().round(2)
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(output_dir, "dataset_statistics.csv"), index=False)
    
    print("\nFinal Statistics for Merged Dataset:")
    print(f"Total samples: {len(final_merged_df)}")
    print(f"Global 75th percentile: {global_75th_percentile:.2f} days")
    print(f"Label 0 (≤{global_75th_percentile:.2f} days): {sum(final_merged_df['label'] == 0)}")
    print(f"Label 1 (>{global_75th_percentile:.2f} days): {sum(final_merged_df['label'] == 1)}")
    print("\nSamples per source:")
    print(final_merged_df['source'].value_counts())

if __name__ == "__main__":
    process_and_label_datasets(strategy='conservative')
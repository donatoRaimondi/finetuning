import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_balanced_datasets(input_dir="", merged_file="merged_processed_labeled.csv", sizes=[1000, 2000, 5000, 10000]):
    """
    Crea dataset bilanciati per il training in diverse dimensioni e genera set di validazione e test fissi.
    
    Args:
        input_dir (str): Cartella in cui si trova il dataset unificato.
        merged_file (str): Nome del file CSV contenente il dataset completo.
        sizes (list): Dimensioni desiderate per i dataset di training bilanciati.
    """
    
    output_dir = "balanced_datasets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Crea la cartella se non esiste
    
    print("📂 Lettura del dataset unificato...")
    df = pd.read_csv(os.path.join(input_dir, merged_file))
    
    # Rimuove fonti con troppi pochi dati per garantire bilanciamento
    df = df[~df['source'].isin(['W3C', 'OpenXchange'])]
    
    # Trova il minimo numero di campioni per ogni coppia (source, label)
    min_samples_per_source_label = float('inf')
    sources = df['source'].unique()
    
    print("\n🔍 Minimo numero di campioni per ogni source e label:")
    for source in sources:
        for label in [0, 1]:
            samples = len(df[(df['source'] == source) & (df['label'] == label)])
            print(f"Source: {source}, Label: {label}, Samples: {samples}")
            min_samples_per_source_label = min(min_samples_per_source_label, samples)
    
    # Imposta un minimo di sicurezza per evitare dataset troppo piccoli
    min_samples_per_source_label = max(500, min_samples_per_source_label)
    
    # Calcola la dimensione massima possibile di un dataset bilanciato
    max_balanced_size = min_samples_per_source_label * 2 * len(sources)
    valid_sizes = [size for size in sizes if size <= max_balanced_size]
    if max_balanced_size not in valid_sizes:
        valid_sizes.append(max_balanced_size)
    
    # Creazione dei set di validazione e test
    print("📊 Creazione dei set di validazione e test...")
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
        
        # Divide il dataset in validation e test, mantenendo la stratificazione per source e label
        val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42, stratify=val_test_df[['source', 'label']])
        
        validation_dfs.append(val_df)
        test_dfs.append(test_df)
    
    validation_df = pd.concat(validation_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    validation_df.to_csv(os.path.join(output_dir, "balanced_validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "balanced_test.csv"), index=False)
    
    # Creazione dei dataset di training bilanciati
    for target_size in valid_sizes:
        print(f"⚖️ Creazione del dataset bilanciato di training di dimensione {target_size}...")
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
    
    print("✅ Dataset bilanciati creati con successo!")

if __name__ == "__main__":
    create_balanced_datasets(sizes=[1000, 2000, 5000, 10000])

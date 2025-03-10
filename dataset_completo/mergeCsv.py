import pandas as pd
import numpy as np
import re
import os

def clean_text(text):
    """
    Pulisce il testo rimuovendo spazi extra e normalizzando i caratteri speciali.
    
    Args:
        text (str): Testo da pulire.
    
    Returns:
        str o np.nan: Testo pulito o NaN se vuoto.
    """
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text).strip()  # Rimuove spazi extra
        return text if text else np.nan
    return np.nan

def handle_missing_values(df):
    """
    Gestisce i valori mancanti in modo conservativo:
    - Riempe `days_resolution` con il 75° percentile del dataset.
    - Sostituisce altri valori numerici mancanti con 0.
    - Riempe colonne testuali con "Unknown".
    - Rimuove duplicati.

    Args:
        df (pd.DataFrame): DataFrame da elaborare.
    
    Returns:
        pd.DataFrame: DataFrame con valori mancanti gestiti.
    """
    df = df.copy()
    
    # Identifica colonne numeriche e testuali
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    text_columns = df.select_dtypes(include=['object']).columns
    
    # Sostituisce i valori mancanti nelle colonne numeriche
    for col in numeric_columns:
        if col == 'days_resolution':
            df[col].fillna(df[col].quantile(0.75), inplace=True)  # Usa il 75° percentile
        else:
            df[col].fillna(0, inplace=True)  # Usa 0 per altre colonne numeriche
    
    # Sostituisce valori mancanti nelle colonne testuali con 'Unknown'
    for col in text_columns:
        df[col].fillna('Unknown', inplace=True)
    
    # Rimuove righe duplicate
    df.drop_duplicates(inplace=True)
    
    return df

def process_and_merge_datasets(input_dir='csv_output_from_json', output_dir='processed_labeled_datasets'):
    """
    Processa e unisce i dataset:
    - Pulisce i dati e gestisce i valori mancanti.
    - Converte `days_resolution` in numerico e rimuove righe con valori non validi.
    - Assegna etichette (`label`) basate sul 75° percentile globale.
    - Salva i dataset processati e il dataset unificato.

    Args:
        input_dir (str): Cartella contenente i file CSV di input.
        output_dir (str): Cartella in cui salvare i file processati.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Crea la cartella di output se non esiste
    
    files = [f for f in os.listdir(input_dir) if f.endswith('_data.csv')]
    cleaned_dfs = []  # Lista per i dataset puliti
    
    print("Elaborazione dei dataset in corso...")
    for file in files:
        source_name = file.replace('_data.csv', '')  # Estrae il nome della fonte dal file
        try:
            # Carica il file CSV
            df = pd.read_csv(os.path.join(input_dir, file))
            
            # Pulisce le colonne di testo
            for col in ['short_desc', 'comments', 'product', 'priority']:
                if col in df.columns:
                    df[col] = df[col].apply(clean_text)
            
            # Gestisce i valori mancanti
            df = handle_missing_values(df)
            
            # Converte `days_resolution` in numerico e rimuove righe non valide
            df['days_resolution'] = pd.to_numeric(df['days_resolution'], errors='coerce')
            df.dropna(subset=['days_resolution'], inplace=True)
            
            # Aggiunge il nome della fonte
            df['source'] = source_name
            
            cleaned_dfs.append(df)
        
        except Exception as e:
            print(f"Errore durante l'elaborazione di {source_name}: {str(e)}")
    
    # Se non ci sono dataset validi, termina l'esecuzione
    if not cleaned_dfs:
        print("Nessun dataset valido trovato. Uscita.")
        return
    
    # Unisce i dataset e calcola il 75° percentile globale
    print("Unione dei dataset e calcolo del 75° percentile globale...")
    merged_df = pd.concat(cleaned_dfs, ignore_index=True)
    global_75th_percentile = merged_df['days_resolution'].quantile(0.75)
    
    # Assegna etichette ai dataset singoli e li salva
    final_dfs = []
    for df in cleaned_dfs:
        df['label'] = (df['days_resolution'] > global_75th_percentile).astype(int)
        output_file = os.path.join(output_dir, f"{df['source'].iloc[0]}_processed_labeled.csv")
        df.to_csv(output_file, index=False)
        final_dfs.append(df)
    
    # Unisce e salva il dataset finale
    final_merged_df = pd.concat(final_dfs, ignore_index=True)
    merged_output = os.path.join(output_dir, "merged_processed_labeled.csv")
    final_merged_df.to_csv(merged_output, index=False)
    
    print("Elaborazione completata. Dataset unificato salvato.")

if __name__ == "__main__":
    process_and_merge_datasets()

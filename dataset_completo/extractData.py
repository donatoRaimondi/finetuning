import json
import pandas as pd
import os

# Funzione per estrarre attributi specifici da un oggetto JSON
# e inserirli in formato tabellare.
def extract_attributes(entry):
    bug = entry.get('bug', {})
    attributes = {
        'short_desc': bug.get('short_desc', ''), # Descrizione breve del bug
        'product': bug.get('product', ''), # Prodotto associato al bug
        'priority': bug.get('priority', ''), # Priorità assegnata al bug 
        'bug_severity': bug.get('bug_severity', ''),  # Gravità del bug
        'days_resolution': bug.get('days_resolution', '') # Giorni impiegati per la risoluzione
    }
    
     # Estrarre tutti i commenti (long_desc) e concatenarli
    long_desc_list = bug.get("long_desc", [])
    all_long_desc = "\n\n".join(comment.get("thetext", "") for comment in long_desc_list)
    # mettiamo i commenti dio un singolo bug report 
    # in una singola colonna chiamata comments
    attributes['comments'] = all_long_desc 
    
    return attributes

# Funzione per leggere un file JSON, supportando sia file con un singolo oggetto
# JSON che file con più JSON separati per riga.
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        try:
            # Tentativo di caricare il file come un unico JSON
            data = json.load(file)
            if isinstance(data, dict):
                return [data]  # Wrap single object in a list
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            file.seek(0)  # Reset del puntatore del file per la lettura linea per linea
            data = []
            for line in file:
                try:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
    return data


# Funzione per convertire un file JSON in CSV.
def json_to_csv(json_file, csv_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Il file {json_file} non esiste.")

    # Lettura del file JSON
    data = read_json_file(json_file)

    if not data:
        raise ValueError("Nessun oggetto JSON valido trovato nel file.")

    # Estrazione degli attributi desiderati
    extracted_data = [extract_attributes(entry) for entry in data]

    # Creazione del DataFrame
    df = pd.DataFrame(extracted_data)

    # Sostituzione dei valori NaN con stringhe vuote
    df = df.fillna('')

    # Scrittura su file CSV
    df.to_csv(csv_file, index=False)

    print(f"File CSV '{csv_file}' creato con successo.")

# Elenco dei file da processare
files = ['Eclipse','FreeDesktop','Gentoo','KDE','LibreOffice','LiveCode','NetBeans','Novell','OpenOffice','OpenXchange','W3C']

# Elaborazione dei file JSON e conversione in CSV 
for file in files:
    json_file = 'Dataset/'+ file +'_original.json'  # Percorso del file JSON
    csv_file = 'csv_output_from_json/'+file + '_data.csv'  # Percorso del file CSV in output
    try:
        json_to_csv(json_file, csv_file)
    except Exception as e:
        print(f"Errore durante la conversione di {json_file}: {str(e)}")
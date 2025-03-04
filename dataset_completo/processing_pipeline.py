import os
import subprocess

def run_script(script_name, description):
    """Esegue uno script Python e gestisce gli errori."""
    print(f"\n[STEP] {description}...")
    if not os.path.exists(script_name):
        print(f"[ERRORE] Lo script {script_name} non esiste! Controlla il nome del file.")
        return False
    
    show_output = (script_name == "verifica_bilanciamento.py")  # Stampa solo l'ultimo step
    result = subprocess.run(["python3", script_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] {description} completato con successo!\n")
        return True
    else:
        print(f"[ERRORE] {description} fallito!\n")
        print(result.stderr)
        return False

if __name__ == "__main__":
    print("\n=== PIPELINE DI PROCESSAMENTO DEL DATASET ===\n")
    
    # Step 1: Estrazione JSON → CSV
    if not run_script("extractData.py", "Estrazione dei dataset JSON in CSV"):
        exit(1)
    
    # Step 2: Fusione e pulizia CSV
    if not run_script("mergeCsv.py", "Unione e pulizia dei file CSV"):
        exit(1)
    
    # Step 3: Processing e assegnazione labels
    if not run_script("processacsv.py", "Processamento e assegnazione labels"):
        exit(1)

    # Step 4: Creazione dataset bilanciati
    if not run_script("split_into_balanced_datasets.py", "Creazione di dataset bilanciati per training e valutazione"):
        exit(1)
    # Step 5: Verifica il bilanciamento dei dataset
    if not run_script("verifica_bilanciamento.py", "Verifica il bilanciamento dei dataset"):
        exit(1)

    print("\nPipeline completata con successo!")

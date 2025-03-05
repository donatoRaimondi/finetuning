import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def plot_metrics_trend(results_files, system_files, training_results_file, output_dir="metrics_trend_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Crea la cartella se non esiste

    results = {}
    
    # Carica le metriche di sistema e performance
    for size, metric_file in results_files.items():
        system_file = system_files.get(size)  # Ottieni il file delle metriche di sistema corrispondente

        if os.path.exists(metric_file) and os.path.exists(system_file):  # Controlla se entrambi i file esistono
            df_metrics = pd.read_csv(metric_file)
            df_system = pd.read_csv(system_file)

            # Uniamo i due DataFrame per la stessa dataset size
            df_combined = pd.concat([df_metrics, df_system], axis=1)
            results[size] = df_combined
        else:
            print(f"⚠️ File mancanti per Dataset Size {size}:")
            if not os.path.exists(metric_file):
                print(f"   ❌ {metric_file} non trovato.")
            if not os.path.exists(system_file):
                print(f"   ❌ {system_file} non trovato.")

    if not results:
        print("❌ Nessun file trovato, impossibile generare il trend.")
        return None

    # **Definizione delle metriche**
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1']
    resource_metrics = ['cpu', 'ram', 'gpu', 'time']

    trend_data = []
    for size, df in results.items():
        row = {'Dataset Size': size}
        for metric in performance_metrics + resource_metrics:
            if metric in df.columns:
                row[metric] = df[metric].iloc[0]  # Prendiamo il primo valore disponibile
        trend_data.append(row)

    trend_df = pd.DataFrame(trend_data)
    
    # Carica i risultati di training dal file CSV
    for size, training_file in training_results_file.items():
        if os.path.exists(training_file):
            df_training = pd.read_csv(training_file)
            # Aggiungiamo le informazioni sui risultati di training (ad esempio Training Loss, Train Time)
            for column in df_training.columns:
                if column not in trend_df.columns:
                    trend_df[column] = df_training[column].iloc[0]
        else:
            # Gestiamo il caso in cui il file di training non esista (ad esempio, per il modello non fine-tuned)
            print(f"⚠️ Nessun dato di training trovato per Dataset Size {size} ({training_file})")
            # Puoi aggiungere valori NaN o 0 se i dati di training non sono disponibili
            trend_df['Training Loss'] = trend_df.get('Training Loss', pd.NA)
            trend_df['Training Time'] = trend_df.get('Training Time', pd.NA)
    
    trend_df.sort_values(by='Dataset Size', inplace=True)

    # **📈 Plot delle metriche di performance**
    plt.figure(figsize=(12, 6))
    for metric in performance_metrics:
        if metric in trend_df.columns:
            sns.lineplot(data=trend_df, x='Dataset Size', y=metric, marker='o', label=metric.capitalize())

    plt.title('Performance Metric Trend by Dataset Size')
    plt.xlabel('Dataset Size')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics_trend.png', dpi=300)
    plt.close()

    # **📊 Plot delle risorse hardware**
    plt.figure(figsize=(12, 6))
    for metric in resource_metrics:
        if metric in trend_df.columns:
            sns.lineplot(data=trend_df, x='Dataset Size', y=metric, marker='o', label=metric.upper())

    plt.title('Resource Usage Trend by Dataset Size')
    plt.xlabel('Dataset Size')
    plt.ylabel('Usage')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'resource_usage_trend.png', dpi=300)
    plt.close()

    # **🔥 Heatmap combinata**
    plt.figure(figsize=(12, 6))
    sns.heatmap(trend_df.set_index('Dataset Size').T, annot=True, cmap='coolwarm', fmt='.4f')
    plt.title('Overall Metric Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_metrics_heatmap.png', dpi=300)
    plt.close()

    # **📄 Salvataggio tabella riassuntiva**
    trend_df.to_csv(output_dir / 'metrics_trend_summary.csv', index=False)

    print("\nTrend visualization saved in:", output_dir)
    print("\nSummary:")
    print(trend_df)

    return trend_df


# Esempio di utilizzo
dictionary_path = {
    1: {"path": "isola_esperimento_llm/meta-llama/", "model": "Llama-3.1-8b-Instruct"},
    2: {"path": "isola_classificazione_distilbert/", "model": "distilbert-base-uncased"}
}

# Mostra il menu all'utente
print("\n📌 Seleziona un modello disponibile:")
for key, info in dictionary_path.items():
    print(f"{key}: {info['model']} ({info['path']})")

# Input dell'utente con validazione
while True:
    try:
        selected_key = int(input("\n🔹 Inserisci il numero del modello: "))
        if selected_key in dictionary_path:
            break
        else:
            print("❌ Scelta non valida! Scegli un numero corretto.")
    except ValueError:
        print("⚠️ Inserisci un numero valido.")

# Recupera percorso e modello selezionato
selected_info = dictionary_path[selected_key]
partial_path = selected_info["path"]
model = selected_info["model"]

# Output della selezione
print(f"\n✅ Hai selezionato: {model}")
print(f"📂 Percorso parziale di selezione dei risultati: {partial_path}")

results_files = {
    0: os.path.join(partial_path, f"{model}_not_fine_tuned", "metrics.csv"),
    1000: os.path.join(partial_path, f"{model}_fine_tuned_on_1000", "metrics.csv"),
    #2000: os.path.join(partial_path, f"{model}_fine_tuned_on_2000", "metrics.csv"),
    #5000: os.path.join(partial_path, f"{model}_fine_tuned_on_5000", "metrics.csv"),
    #9000: os.path.join(partial_path, f"{model}_fine_tuned_on_9000", "metrics.csv"),
}

system_files = {
    0: os.path.join(partial_path, f"{model}_not_fine_tuned", "avg_system_metrics.csv"),
    1000: os.path.join(partial_path, f"{model}_fine_tuned_on_1000", "avg_system_metrics.csv"),
    #2000: os.path.join(partial_path, f"{model}_fine_tuned_on_2000", "avg_system_metrics.csv"),
    #5000: os.path.join(partial_path, f"{model}_fine_tuned_on_5000", "avg_system_metrics.csv"),
    #9000: os.path.join(partial_path, f"{model}_fine_tuned_on_9000", "avg_system_metrics.csv"),
}

# Carica i file dei risultati di training
training_results_file = {
    0: os.path.join(),
    1000: os.path.join(partial_path, f"{model}_fine_tuned_on_1000/training_comparison.csv"),
    #2000: os.path.join(partial_path, f"{model}_fine_tuned_on_2000/training_comparison.csv"),
    #5000: os.path.join(partial_path, f"{model}_fine_tuned_on_5000/training_comparison.csv"),
    #9000: os.path.join(partial_path, f"{model}_fine_tuned_on_9000/training_comparison.csv"),
} 

# Creazione cartella e salvataggio risultati
output_path = os.path.join("comparisons", model)
trend_summary = plot_metrics_trend(results_files, system_files, training_results_file, output_dir=output_path)
# Creazione cartella e salvataggio risultati
output_path = os.path.join("comparisons", model)
trend_summary = plot_metrics_trend(results_files, system_files, output_dir=output_path)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def plot_metrics_trend(results_files, system_files, training_results_files, output_dir="metrics_trend_results"):
    """
    Analizza e visualizza le metriche di performance, risorse e training per diversi dataset sizes.
    
    :param results_files: Dizionario con i percorsi dei file delle metriche di performance.
    :param system_files: Dizionario con i percorsi dei file delle metriche di sistema.
    :param training_results_files: Dizionario con i percorsi dei file delle metriche di training.
    :param output_dir: Cartella di output per salvare i risultati.
    :return: DataFrame con i risultati aggregati.
    """
    # Crea la cartella di output se non esiste
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lista per memorizzare i risultati aggregati
    trend_data = []

    # Itera su ogni dataset size
    for size in results_files.keys():
        metric_file = results_files[size]
        system_file = system_files.get(size)
        training_file = training_results_files.get(size)

        # Verifica se tutti i file esistono
        if not all(os.path.exists(f) for f in [metric_file, system_file, training_file]):
            print(f"⚠️ File mancanti per Dataset Size {size}:")
            for file, path in zip(["Metriche", "Sistema", "Training"], [metric_file, system_file, training_file]):
                if not os.path.exists(path):
                    print(f"   ❌ {file}: {path} non trovato.")
            continue

        # Carica i file CSV
        df_metrics = pd.read_csv(metric_file)
        df_system = pd.read_csv(system_file)
        df_training = pd.read_csv(training_file)

        # Debug: stampa le prime righe dei DataFrame
        print(f"\n📥 Dataset Size {size}:")
        print(f"Metriche:\n{df_metrics.head()}")
        print(f"Metriche di sistema:\n{df_system.head()}")
        print(f"Metriche di training:\n{df_training.head()}")

        # Combina i DataFrame
        df_combined = pd.concat([df_metrics, df_system, df_training], axis=1)
        df_combined['Dataset Size'] = size  # Aggiungi la colonna Dataset Size
        trend_data.append(df_combined)

    # Se non ci sono dati, termina la funzione
    if not trend_data:
        print("❌ Nessun file trovato, impossibile generare il trend.")
        return None

    # Crea il DataFrame finale
    trend_df = pd.concat(trend_data, ignore_index=True)

    # Salva il DataFrame in un file CSV
    trend_df.to_csv(output_dir / 'metrics_trend_summary.csv', index=False)

    # **Plot delle metriche di performance**
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1']
    plot_trend(trend_df, performance_metrics, 'Performance Metric Trend by Dataset Size', 'Score', output_dir / 'performance_metrics_trend.png')

    # **Plot delle risorse hardware**
    resource_metrics = ['cpu', 'ram', 'gpu', 'time']
    plot_trend(trend_df, resource_metrics, 'Resource Usage Trend by Dataset Size', 'Usage', output_dir / 'resource_usage_trend.png')

    # **Heatmap combinata**
    plt.figure(figsize=(12, 6))
    sns.heatmap(trend_df.set_index('Dataset Size').T, annot=True, cmap='coolwarm', fmt='.4f')
    plt.title('Overall Metric Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_metrics_heatmap.png', dpi=300)
    plt.close()

    print("\nTrend visualization saved in:", output_dir)
    print("\nSummary:")
    print(trend_df)

    return trend_df


def plot_trend(df, metrics, title, ylabel, output_path):
    """
    Crea un grafico a linee per le metriche specificate.
    
    :param df: DataFrame contenente i dati.
    :param metrics: Lista delle metriche da plottare.
    :param title: Titolo del grafico.
    :param ylabel: Etichetta dell'asse Y.
    :param output_path: Percorso di salvataggio del grafico.
    """
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        if metric in df.columns:
            sns.lineplot(data=df, x='Dataset Size', y=metric, marker='o', label=metric.capitalize())
    plt.title(title)
    plt.xlabel('Dataset Size')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


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

training_results_file = {
    0: os.path.join(partial_path, f"{model}_not_fine_tuned", "training_comparison.csv"),
    1000: os.path.join(partial_path, f"{model}_fine_tuned_on_1000", "training_comparison.csv"),
    #2000: os.path.join(partial_path, f"{model}_fine_tuned_on_2000", "training_comparison.csv"),
    #5000: os.path.join(partial_path, f"{model}_fine_tuned_on_5000", "training_comparison.csv"),
    #9000: os.path.join(partial_path, f"{model}_fine_tuned_on_9000", "training_comparison.csv"),
}

# Creazione cartella e salvataggio risultati
output_path = os.path.join("comparisons", model)
trend_summary = plot_metrics_trend(results_files, system_files, training_results_file, output_dir=output_path)
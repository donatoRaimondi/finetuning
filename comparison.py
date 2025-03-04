import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_metrics_trend(results_files, output_dir="metrics_trend_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Caricamento dei file
    results = {size: pd.read_csv(file) for size, file in results_files.items()}
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Creazione di DataFrame aggregato
    trend_data = []
    for size, df in results.items():
        row = {'Dataset Size': size}
        for metric in metrics:
            row[metric] = df[metric].iloc[0]
        trend_data.append(row)

    trend_df = pd.DataFrame(trend_data)
    trend_df.sort_values(by='Dataset Size', inplace=True)

    # Plot lineare per le metriche
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        sns.lineplot(data=trend_df, x='Dataset Size', y=metric, marker='o', label=metric.capitalize())

    plt.title('Metric Trend by Dataset Size')
    plt.xlabel('Dataset Size')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_trend.png', dpi=300)
    plt.close()

    # Heatmap delle metriche
    plt.figure(figsize=(10, 6))
    sns.heatmap(trend_df.set_index('Dataset Size').T, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title('Metric Heatmap by Dataset Size')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300)
    plt.close()

    # Salvataggio tabella riassuntiva
    trend_df.to_csv(output_dir / 'metrics_trend_summary.csv', index=False)

    print("\nTrend visualization saved in:", output_dir)
    print("\nSummary:")
    print(trend_df)

    return trend_df

# Esempio di utilizzo
dictionary_path = {
    1: {"path": "isola_esperimento_llm/meta-llama/", "model": "Llama-3.1-8b-Instruct"},
    2: {"path": "isola_classificazione_distilbert/", "model": "distilber-base-uncased"}
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
    0: f"{partial_path}/{model}_not_fine_tuned/metrics.csv",
    1000: f"{partial_path}/{model}_fine_tuned_on_1000/metrics.csv",
    #2000: f"{partial_path}/{model}_fine_tuned_on_2000/metrics.csv",
    #5000: f"{partial_path}/{model}_fine_tuned_on_5000/metrics.csv",
    9000: f"{partial_path}/{model}_fine_tuned_on_9000/metrics.csv",
}

#dove salviamo i risultati della comparazione
trend_summary = plot_metrics_trend(results_files, output_dir=f"comparisons/{model}") 

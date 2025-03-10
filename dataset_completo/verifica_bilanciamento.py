import pandas as pd

# Lista delle dimensioni dei dataset di training da analizzare
nums = [1000, 2000, 5000, 9000]

for num in nums:
    print("=" * 50)
    print(f"📊 ANALISI DATASET TRAINING: {num} CAMPIONI")
    print("=" * 50)
    
    # Carica i dataset bilanciati
    train_df = pd.read_csv(f"balanced_datasets/balanced_train_{num}.csv")
    val_df = pd.read_csv("balanced_datasets/balanced_validation.csv")
    test_df = pd.read_csv("balanced_datasets/balanced_test.csv")

    # Verifica presenza di valori mancanti
    print("\n🔍 Controllo valori mancanti:")
    print(f"Train {num}: {train_df.isnull().sum().sum()} NaN")
    print(f"Validation: {val_df.isnull().sum().sum()} NaN")
    print(f"Test: {test_df.isnull().sum().sum()} NaN")
    
    # Distribuzione delle etichette (label)
    print("\n🟢 Distribuzione etichette (label):")
    print(f"Train {num}:\n", train_df['label'].value_counts(normalize=True).round(3))
    print("\nValidation:\n", val_df['label'].value_counts(normalize=True).round(3))
    print("\nTest:\n", test_df['label'].value_counts(normalize=True).round(3))

    # Distribuzione delle fonti (source)
    print("\n🔵 Distribuzione fonti (source):")
    print(f"Train {num}:\n", train_df['source'].value_counts(normalize=True).round(3))
    print("\nValidation:\n", val_df['source'].value_counts(normalize=True).round(3))
    print("\nTest:\n", test_df['source'].value_counts(normalize=True).round(3))
    
    print("\n" + "=" * 50 + "\n")

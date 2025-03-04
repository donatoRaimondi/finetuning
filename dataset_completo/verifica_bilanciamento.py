import pandas as pd

# Carica i dataset finali
nums = [1000,2000,5000,9000]
for num in nums:
    train_df = pd.read_csv(f"balanced_datasets/balanced_train_{num}.csv")
    val_df = pd.read_csv("balanced_datasets/balanced_validation.csv")
    test_df = pd.read_csv("balanced_datasets/balanced_test.csv")

    # Controllo distribuzione labels
    print(f"Train Label {num} Distribution:\n", train_df['label'].value_counts(normalize=True))
    print("\nValidation Label Distribution:\n", val_df['label'].value_counts(normalize=True))
    print("\nTest Label Distribution:\n", test_df['label'].value_counts(normalize=True))

    # Controllo distribuzione per source
    print(f"\nTrain {num} Source Distribution:\n", train_df['source'].value_counts(normalize=True))
    print("\nValidation Source Distribution:\n", val_df['source'].value_counts(normalize=True))
    print("\nTest Source Distribution:\n", test_df['source'].value_counts(normalize=True))

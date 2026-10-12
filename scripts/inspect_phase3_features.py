import pandas as pd


TRAIN_PATH = "data/processed/splits/train.csv"
VALIDATION_PATH = "data/processed/splits/validation.csv"


def inspect_dataset(df, name):
    print("\n" + "=" * 70)
    print(f"{name} DATASET")
    print("=" * 70)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nTransaction ID duplicates:")
    print(df["TRANSACTION_ID"].duplicated().sum())

    print("\nFraud distribution:")
    print(df["TX_FRAUD"].value_counts())
    print("\nFraud percentage:")
    print(df["TX_FRAUD"].value_counts(normalize=True) * 100)

    print("\nFraud scenario distribution:")
    print(
        pd.crosstab(
            df["TX_FRAUD"],
            df["TX_FRAUD_SCENARIO"]
        )
    )

    print("\nDate range:")
    dates = pd.to_datetime(df["TX_DATETIME"])
    print("Start:", dates.min())
    print("End:  ", dates.max())

    print("\nTX_TIME_DAYS range:")
    print("Min:", df["TX_TIME_DAYS"].min())
    print("Max:", df["TX_TIME_DAYS"].max())

    print("\nUnique customers:")
    print(df["CUSTOMER_ID"].nunique())

    print("\nUnique terminals:")
    print(df["TERMINAL_ID"].nunique())

    print("\nCustomer transaction frequency:")
    print(df["CUSTOMER_ID"].value_counts().describe())

    print("\nTerminal transaction frequency:")
    print(df["TERMINAL_ID"].value_counts().describe())

    print("\nTransaction amount statistics:")
    print(df["TX_AMOUNT"].describe())

    print("\nTransactions per TX_TIME_DAYS:")
    print(df["TX_TIME_DAYS"].value_counts().sort_index())


def main():

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    inspect_dataset(train, "TRAIN")
    inspect_dataset(validation, "VALIDATION")

    print("\n" + "=" * 70)
    print("TRAIN → VALIDATION CUSTOMER OVERLAP")
    print("=" * 70)

    common_customers = set(train["CUSTOMER_ID"]) & set(
        validation["CUSTOMER_ID"]
    )

    print("Train unique customers:", train["CUSTOMER_ID"].nunique())
    print("Validation unique customers:", validation["CUSTOMER_ID"].nunique())
    print("Customers appearing in both:", len(common_customers))

    print("\n" + "=" * 70)
    print("TRAIN → VALIDATION TERMINAL OVERLAP")
    print("=" * 70)

    common_terminals = set(train["TERMINAL_ID"]) & set(
        validation["TERMINAL_ID"]
    )

    print("Train unique terminals:", train["TERMINAL_ID"].nunique())
    print("Validation unique terminals:", validation["TERMINAL_ID"].nunique())
    print("Terminals appearing in both:", len(common_terminals))


if __name__ == "__main__":
    main()
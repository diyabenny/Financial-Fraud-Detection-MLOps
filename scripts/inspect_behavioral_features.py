"""
Phase 3 - Behavioral Feature Investigation.

Inspects the Phase 2.5 training and validation datasets before
creating customer- and terminal-level behavioral features.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------

TRAIN_PATH = Path("data/processed/splits/train.csv")
VALIDATION_PATH = Path("data/processed/splits/validation.csv")


# ---------------------------------------------------------------------------
# Columns required for behavioral feature investigation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_DAYS",
    "TX_FRAUD",
]


def inspect_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Print basic information needed for behavioral feature design."""

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} BEHAVIORAL FEATURE INSPECTION")
    print("=" * 70)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(df[REQUIRED_COLUMNS].isna().sum())

    print("\nDuplicate TRANSACTION_ID:")
    print(df["TRANSACTION_ID"].duplicated().sum())

    print("\nTX_DATETIME information:")

    dt = pd.to_datetime(
        df["TX_DATETIME"],
        errors="raise",
    )

    print(f"  Minimum: {dt.min()}")
    print(f"  Maximum: {dt.max()}")
    print(f"  Invalid timestamps: {dt.isna().sum()}")

    print("\nUnique customers:")
    print(df["CUSTOMER_ID"].nunique())

    print("\nUnique terminals:")
    print(df["TERMINAL_ID"].nunique())

    print("\nTX_AMOUNT:")
    print(f"  Minimum: {df['TX_AMOUNT'].min()}")
    print(f"  Maximum: {df['TX_AMOUNT'].max()}")
    print(f"  Mean:    {df['TX_AMOUNT'].mean():.4f}")
    print(f"  Median:  {df['TX_AMOUNT'].median():.4f}")

    print("\nNegative TX_AMOUNT values:")
    print((df["TX_AMOUNT"] < 0).sum())

    print("\nTransactions per customer:")
    customer_counts = df["CUSTOMER_ID"].value_counts()
    print(customer_counts.describe())

    print("\nTransactions per terminal:")
    terminal_counts = df["TERMINAL_ID"].value_counts()
    print(terminal_counts.describe())

    print("\nTX_TIME_DAYS range:")
    print(
        f"  Minimum: {df['TX_TIME_DAYS'].min()}"
    )
    print(
        f"  Maximum: {df['TX_TIME_DAYS'].max()}"
    )


def main() -> None:
    """Inspect training and validation datasets."""

    print("=" * 70)
    print("PHASE 3 - BEHAVIORAL FEATURE INVESTIGATION")
    print("=" * 70)

    for path, dataset_name in [
        (TRAIN_PATH, "Training dataset"),
        (VALIDATION_PATH, "Validation dataset"),
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

        df = pd.read_csv(path)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing required columns: "
                f"{missing_columns}"
            )

        inspect_dataset(
            df,
            dataset_name,
        )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
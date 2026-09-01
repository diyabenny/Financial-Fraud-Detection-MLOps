"""
Phase 2.5 - Train / Validation Data Split

Creates a chronological train/validation split from the April
transaction dataset for model development.

Split strategy:
    - Training: April 1-24 (TX_TIME_DAYS 0-23)
    - Validation: April 25-30 (TX_TIME_DAYS 24-29)
    - May 1-19 is reserved as unseen future test data and is
      intentionally not processed by this script.

Input:
    data/incoming/april/converted_april_file.csv

Outputs:
    data/processed/splits/train.csv
    data/processed/splits/validation.csv

Usage:
    python scripts/split_data_phase2.py
"""

from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APRIL_DATA_PATH = (
    PROJECT_ROOT / "data" / "incoming" / "april" / "converted_april_file.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_OUTPUT_PATH = OUTPUT_DIR / "validation.csv"

# Chronological split boundaries
TRAIN_END_DAY = 23
VALIDATION_START_DAY = 24


def load_april_data(file_path: Path) -> pd.DataFrame:
    """
    Load the April transaction dataset.

    Args:
        file_path: Path to the April CSV file.

    Returns:
        DataFrame containing April transactions.

    Raises:
        FileNotFoundError: If the April dataset does not exist.
        ValueError: If required columns are missing.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"April dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = {
        "TX_DATETIME",
        "TX_TIME_DAYS",
        "TX_FRAUD",
        "TRANSACTION_ID",
    }
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def split_train_validation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create chronological training and validation datasets.

    Training contains TX_TIME_DAYS 0-23.
    Validation contains TX_TIME_DAYS 24-29.

    Args:
        df: April transaction DataFrame.

    Returns:
        Tuple containing training and validation DataFrames.
    """
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)

    train_df = df[df["TX_TIME_DAYS"] <= TRAIN_END_DAY].copy()
    validation_df = df[
        df["TX_TIME_DAYS"] >= VALIDATION_START_DAY
    ].copy()

    return train_df, validation_df


def validate_split(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    """
    Validate the chronological train/validation split.

    Checks:
        - Neither split is empty.
        - Original April transaction IDs are unique.
        - Training stays within the training boundary.
        - Validation starts at the validation boundary.
        - Training and validation periods do not overlap.
        - No April rows are lost during splitting.
        - No duplicate transaction IDs exist across the splits.

    Args:
        original_df: Original April DataFrame.
        train_df: Training DataFrame.
        validation_df: Validation DataFrame.

    Raises:
        ValueError: If any validation check fails.
    """
    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")

    if original_df["TRANSACTION_ID"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs found in the original April dataset."
        )

    if train_df["TX_TIME_DAYS"].max() > TRAIN_END_DAY:
        raise ValueError(
            "Training data contains transactions beyond the training boundary."
        )

    if validation_df["TX_TIME_DAYS"].min() < VALIDATION_START_DAY:
        raise ValueError(
            "Validation data contains transactions before the validation boundary."
        )

    if train_df["TX_TIME_DAYS"].max() >= validation_df["TX_TIME_DAYS"].min():
        raise ValueError("Training and validation periods overlap.")

    total_split_rows = len(train_df) + len(validation_df)

    if total_split_rows != len(original_df):
        raise ValueError(
            "Split completeness check failed: "
            f"original={len(original_df):,}, "
            f"train+validation={total_split_rows:,}."
        )

    train_ids = set(train_df["TRANSACTION_ID"])
    validation_ids = set(validation_df["TRANSACTION_ID"])
    duplicate_ids = train_ids.intersection(validation_ids)

    if duplicate_ids:
        raise ValueError(
            "Transaction ID overlap detected between training "
            f"and validation datasets: {len(duplicate_ids):,} IDs."
        )


def print_split_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    """
    Print summary statistics for the generated datasets.

    Args:
        train_df: Training DataFrame.
        validation_df: Validation DataFrame.
    """
    print("\nPhase 2.5 Split Summary")
    print("-" * 60)

    print(
        f"Training:    {len(train_df):,} transactions | "
        f"Days {train_df['TX_TIME_DAYS'].min()}-"
        f"{train_df['TX_TIME_DAYS'].max()} | "
        f"Fraud: {train_df['TX_FRAUD'].sum():,}"
    )

    print(
        f"Validation:  {len(validation_df):,} transactions | "
        f"Days {validation_df['TX_TIME_DAYS'].min()}-"
        f"{validation_df['TX_TIME_DAYS'].max()} | "
        f"Fraud: {validation_df['TX_FRAUD'].sum():,}"
    )

    print(
        f"Total:       "
        f"{len(train_df) + len(validation_df):,} transactions"
    )

    print("-" * 60)


def main() -> None:
    """Run the Phase 2.5 train/validation split."""
    print("Loading April dataset...")

    april_df = load_april_data(APRIL_DATA_PATH)

    print(f"Loaded {len(april_df):,} April transactions.")

    train_df, validation_df = split_train_validation(april_df)

    validate_split(
        april_df,
        train_df,
        validation_df,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        TRAIN_OUTPUT_PATH,
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print_split_summary(
        train_df,
        validation_df,
    )

    print("\nValidation checks: PASSED")
    print("No April rows were lost.")
    print("No duplicate transaction IDs found in April.")
    print("No transaction IDs overlap between splits.")

    print("\nOutput files:")
    print(f"Training:   {TRAIN_OUTPUT_PATH}")
    print(f"Validation: {VALIDATION_OUTPUT_PATH}")

    print("\nMay data was not modified.")


if __name__ == "__main__":
    main()
"""
Phase 2.6 - Dataset Metadata Management

Generates validated metadata summaries for the April source dataset
and the May unseen future test dataset.

The metadata is calculated directly from the actual input CSV files.
Reference values from previous EDA or validation work are not
hard-coded into this script.

Inputs:
    data/incoming/april/converted_april_file.csv
    data/incoming/may/converted_may_file.csv

Outputs:
    reports/metadata/april_metadata.csv
    reports/metadata/may_metadata.csv

Usage:
    python scripts/metadata_phase2.py
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS AND DATASET ROLES
# ============================================================

# Resolve the project root from the location of this script.
# The script is expected to be stored in the project's scripts/
# directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Store generated metadata reports separately from the input
# transaction datasets.
OUTPUT_DIR = PROJECT_ROOT / "reports" / "metadata"

# Define the Phase 2 datasets and their intended roles.
#
# April is the historical source dataset used for development.
# May is reserved as an unseen future-test dataset and must not
# be used during model development or train/validation splitting.
DATASETS = {
    "april": (
        PROJECT_ROOT
        / "data"
        / "incoming"
        / "april"
        / "converted_april_file.csv",
        "source",
    ),
    "may": (
        PROJECT_ROOT
        / "data"
        / "incoming"
        / "may"
        / "converted_may_file.csv",
        "unseen_future_test",
    ),
}


# ============================================================
# REQUIRED COLUMNS
# ============================================================

# These columns are required to calculate the Phase 2.6
# metadata. Their presence is validated before processing.
REQUIRED_COLUMNS = {
    "TRANSACTION_ID",
    "TX_DATETIME",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "CUSTOMER_ID",
    "TERMINAL_ID",
}


# ============================================================
# DATASET LOADING AND VALIDATION
# ============================================================

def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load a Phase 2 dataset and validate its required structure."""

    # Check that the expected input file exists before reading it.
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Read the actual transaction data from the CSV file.
    # No metadata values are hard-coded.
    df = pd.read_csv(file_path)

    # An empty dataset cannot provide meaningful metadata.
    if df.empty:
        raise ValueError(
            f"Dataset is empty: {file_path}"
        )

    # Check that all columns required for Phase 2.6 are present.
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Validate transaction identifiers
    # --------------------------------------------------------

    # Transaction IDs are required for unique-ID and duplicate-ID
    # metadata. Missing IDs are therefore treated as invalid input.
    if df["TRANSACTION_ID"].isna().any():
        raise ValueError(
            "TRANSACTION_ID contains missing values."
        )

    # --------------------------------------------------------
    # Validate the fraud target
    # --------------------------------------------------------

    # Fraud statistics depend on TX_FRAUD, so missing target
    # values must be rejected.
    if df["TX_FRAUD"].isna().any():
        raise ValueError(
            "TX_FRAUD contains missing values."
        )

    # The fraud target must be numeric before checking its values.
    if not pd.api.types.is_numeric_dtype(
        df["TX_FRAUD"]
    ):
        raise ValueError(
            "TX_FRAUD must be numeric."
        )

    # The fraud target is expected to contain only:
    # 0 = legitimate transaction.
    # 1 = fraudulent transaction.
    fraud_values = set(df["TX_FRAUD"].unique())

    if not fraud_values.issubset({0, 1}):
        raise ValueError(
            f"TX_FRAUD must contain only 0 and 1. "
            f"Found: {sorted(fraud_values)}"
        )

    # --------------------------------------------------------
    # Validate transaction timestamps
    # --------------------------------------------------------

    # Convert timestamps once and reject invalid date values.
    # The converted values are used later for date-range metadata.
    df["TX_DATETIME"] = pd.to_datetime(
        df["TX_DATETIME"],
        errors="raise",
    )

    # Missing timestamps would make the reported date coverage
    # incomplete, so reject them explicitly.
    if df["TX_DATETIME"].isna().any():
        raise ValueError(
            "TX_DATETIME contains missing values."
        )

    # --------------------------------------------------------
    # Validate transaction-day information
    # --------------------------------------------------------

    # TX_TIME_DAYS is used to report the temporal coverage of
    # each Phase 2 dataset.
    if df["TX_TIME_DAYS"].isna().any():
        raise ValueError(
            "TX_TIME_DAYS contains missing values."
        )

    # TX_TIME_DAYS must be numeric so that minimum and maximum
    # values can be calculated correctly.
    if not pd.api.types.is_numeric_dtype(
        df["TX_TIME_DAYS"]
    ):
        raise ValueError(
            "TX_TIME_DAYS must be numeric."
        )

    # --------------------------------------------------------
    # Validate customer and terminal identifiers
    # --------------------------------------------------------

    # Customer and terminal IDs are required for unique-entity
    # metadata, so missing values are rejected.
    if df["CUSTOMER_ID"].isna().any():
        raise ValueError(
            "CUSTOMER_ID contains missing values."
        )

    if df["TERMINAL_ID"].isna().any():
        raise ValueError(
            "TERMINAL_ID contains missing values."
        )

    return df


# ============================================================
# METADATA GENERATION
# ============================================================

def generate_metadata(
    df: pd.DataFrame,
    dataset_name: str,
    dataset_role: str,
) -> pd.DataFrame:
    """Generate dataset-level metadata from the supplied DataFrame."""

    # Calculate the total number of transactions once and reuse it.
    total_transactions = len(df)

    # Calculate fraud and legitimate transaction counts directly
    # from the actual dataset.
    fraud_transactions = int(
        (df["TX_FRAUD"] == 1).sum()
    )

    legitimate_transactions = int(
        (df["TX_FRAUD"] == 0).sum()
    )

    # Calculate missing values for every column.
    missing_counts = df.isna().sum()

    # Record missing-value counts by column.
    # "None" is used when no column contains missing values.
    missing_summary = "; ".join(
        f"{column}:{int(count)}"
        for column, count in missing_counts.items()
        if count > 0
    ) or "None"

    # Capture the dataset schema directly from the DataFrame.
    columns_list = ", ".join(df.columns)

    data_types = "; ".join(
        f"{column}:{df[column].dtype}"
        for column in df.columns
    )

    # Build one machine-readable metadata record.
    #
    # All values are calculated from the actual DataFrame.
    # No previously observed reference values are hard-coded.
    metadata = {
        "Dataset": dataset_name,
        "Dataset_Role": dataset_role,
        "Rows": total_transactions,
        "Columns": len(df.columns),
        "Columns_List": columns_list,
        "Data_Types": data_types,
        "Missing_Cells": int(missing_counts.sum()),
        "Missing_Value_Counts": missing_summary,
        "Duplicate_Rows": int(
            df.duplicated().sum()
        ),
        "Unique_Transaction_IDs": int(
            df["TRANSACTION_ID"].nunique()
        ),
        "Duplicate_Transaction_IDs": int(
            df["TRANSACTION_ID"].duplicated().sum()
        ),
        "Start_Date": df["TX_DATETIME"].min(),
        "End_Date": df["TX_DATETIME"].max(),
        "Min_TX_TIME_DAYS": df["TX_TIME_DAYS"].min(),
        "Max_TX_TIME_DAYS": df["TX_TIME_DAYS"].max(),
        "Fraud_Transactions": fraud_transactions,
        "Legitimate_Transactions": legitimate_transactions,
        "Fraud_Percentage": round(
            fraud_transactions / total_transactions * 100,
            4,
        ),
        "Unique_Customers": int(
            df["CUSTOMER_ID"].nunique()
        ),
        "Unique_Terminals": int(
            df["TERMINAL_ID"].nunique()
        ),
    }

    # Return the metadata as a one-row DataFrame so it can be
    # directly written to a CSV report.
    return pd.DataFrame([metadata])


# ============================================================
# METADATA VALIDATION
# ============================================================

def validate_metadata(
    metadata: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """Validate that generated metadata matches the source DataFrame."""

    # A dataset-level metadata report should contain exactly
    # one summary row.
    if len(metadata) != 1:
        raise ValueError(
            "Metadata must contain exactly one summary row."
        )

    row = metadata.iloc[0]

    # --------------------------------------------------------
    # Basic size validation
    # --------------------------------------------------------

    # Verify that the metadata row and column counts match
    # the actual DataFrame.
    if row["Rows"] != len(df):
        raise ValueError(
            "Metadata row count does not match the dataset."
        )

    if row["Columns"] != len(df.columns):
        raise ValueError(
            "Metadata column count does not match the dataset."
        )

    if row["Rows"] <= 0 or row["Columns"] <= 0:
        raise ValueError(
            "Invalid row or column count."
        )

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    # Verify that the recorded column list matches the source
    # DataFrame exactly and in the same order.
    expected_columns = ", ".join(df.columns)

    if row["Columns_List"] != expected_columns:
        raise ValueError(
            "Metadata column list does not match the dataset."
        )

    # Verify the recorded data types against the source DataFrame.
    expected_data_types = "; ".join(
        f"{column}:{df[column].dtype}"
        for column in df.columns
    )

    if row["Data_Types"] != expected_data_types:
        raise ValueError(
            "Metadata data types do not match the dataset."
        )

    # --------------------------------------------------------
    # Transaction-ID validation
    # --------------------------------------------------------

    # Recalculate the expected number of unique transaction IDs.
    expected_unique_ids = int(
        df["TRANSACTION_ID"].nunique()
    )

    if row["Unique_Transaction_IDs"] != expected_unique_ids:
        raise ValueError(
            "Metadata unique transaction-ID count does not "
            "match the dataset."
        )

    if row["Unique_Transaction_IDs"] <= 0:
        raise ValueError(
            "No unique transaction IDs found."
        )

    # --------------------------------------------------------
    # Missing-value validation
    # --------------------------------------------------------

    # Verify the total number of missing cells.
    expected_missing_cells = int(
        df.isna().sum().sum()
    )

    if row["Missing_Cells"] != expected_missing_cells:
        raise ValueError(
            "Metadata missing-cell count does not match "
            "the dataset."
        )

    # Verify the per-column missing-value summary.
    missing_counts = df.isna().sum()

    expected_missing_summary = "; ".join(
        f"{column}:{int(count)}"
        for column, count in missing_counts.items()
        if count > 0
    ) or "None"

    if row["Missing_Value_Counts"] != expected_missing_summary:
        raise ValueError(
            "Metadata missing-value summary does not match "
            "the dataset."
        )

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    # Verify the number of completely duplicated rows.
    expected_duplicate_rows = int(
        df.duplicated().sum()
    )

    if row["Duplicate_Rows"] != expected_duplicate_rows:
        raise ValueError(
            "Metadata duplicate-row count does not match "
            "the dataset."
        )

    # Verify the number of repeated transaction IDs.
    expected_duplicate_ids = int(
        df["TRANSACTION_ID"].duplicated().sum()
    )

    if row["Duplicate_Transaction_IDs"] != expected_duplicate_ids:
        raise ValueError(
            "Metadata duplicate transaction-ID count does not "
            "match the dataset."
        )

    # --------------------------------------------------------
    # Date and time-range validation
    # --------------------------------------------------------

    # Verify the transaction date range.
    expected_start_date = df["TX_DATETIME"].min()
    expected_end_date = df["TX_DATETIME"].max()

    if row["Start_Date"] != expected_start_date:
        raise ValueError(
            "Metadata start date does not match the dataset."
        )

    if row["End_Date"] != expected_end_date:
        raise ValueError(
            "Metadata end date does not match the dataset."
        )

    if row["Start_Date"] > row["End_Date"]:
        raise ValueError(
            "Invalid transaction date range."
        )

    # Verify the TX_TIME_DAYS range.
    expected_min_days = df["TX_TIME_DAYS"].min()
    expected_max_days = df["TX_TIME_DAYS"].max()

    if row["Min_TX_TIME_DAYS"] != expected_min_days:
        raise ValueError(
            "Metadata minimum TX_TIME_DAYS does not match "
            "the dataset."
        )

    if row["Max_TX_TIME_DAYS"] != expected_max_days:
        raise ValueError(
            "Metadata maximum TX_TIME_DAYS does not match "
            "the dataset."
        )

    if row["Min_TX_TIME_DAYS"] > row["Max_TX_TIME_DAYS"]:
        raise ValueError(
            "Invalid TX_TIME_DAYS range."
        )

    # --------------------------------------------------------
    # Fraud-count validation
    # --------------------------------------------------------

    # Recalculate fraud and legitimate counts from the source.
    expected_fraud = int(
        (df["TX_FRAUD"] == 1).sum()
    )

    expected_legitimate = int(
        (df["TX_FRAUD"] == 0).sum()
    )

    if row["Fraud_Transactions"] != expected_fraud:
        raise ValueError(
            "Metadata fraud count does not match the dataset."
        )

    if row["Legitimate_Transactions"] != expected_legitimate:
        raise ValueError(
            "Metadata legitimate count does not match "
            "the dataset."
        )

    # Because TX_FRAUD has been validated as a complete binary
    # target, fraud and legitimate transactions must account
    # for every row.
    if (
        row["Fraud_Transactions"]
        + row["Legitimate_Transactions"]
        != row["Rows"]
    ):
        raise ValueError(
            "Fraud and legitimate transaction counts do not "
            "match total rows."
        )

    # Recalculate the fraud percentage and compare it with
    # the generated metadata.
    expected_fraud_percentage = round(
        expected_fraud / len(df) * 100,
        4,
    )

    if row["Fraud_Percentage"] != expected_fraud_percentage:
        raise ValueError(
            "Metadata fraud percentage does not match "
            "the dataset."
        )

    if not 0 <= row["Fraud_Percentage"] <= 100:
        raise ValueError(
            "Invalid fraud percentage."
        )

    # --------------------------------------------------------
    # Entity-count validation
    # --------------------------------------------------------

    # Verify the number of unique customers.
    expected_customers = int(
        df["CUSTOMER_ID"].nunique()
    )

    if row["Unique_Customers"] != expected_customers:
        raise ValueError(
            "Metadata unique-customer count does not match "
            "the dataset."
        )

    # Verify the number of unique terminals.
    expected_terminals = int(
        df["TERMINAL_ID"].nunique()
    )

    if row["Unique_Terminals"] != expected_terminals:
        raise ValueError(
            "Metadata unique-terminal count does not match "
            "the dataset."
        )


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main() -> None:
    """Generate and save validated metadata for Phase 2 datasets."""

    # Create the metadata output directory if it does not exist.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Process April and May independently.
    #
    # This script only reads the datasets and creates metadata.
    # It does not create, modify, or overwrite the train/validation
    # split.
    for dataset_name, (
        file_path,
        dataset_role,
    ) in DATASETS.items():

        print(
            f"\nProcessing {dataset_name.upper()} dataset..."
        )

        # Load the actual dataset and perform structural validation.
        df = load_dataset(file_path)

        # Generate metadata directly from the validated DataFrame.
        metadata = generate_metadata(
            df,
            dataset_name,
            dataset_role,
        )

        # Verify that the generated metadata agrees with
        # the original DataFrame.
        validate_metadata(
            metadata,
            df,
        )

        # Save the validated metadata summary as a CSV report.
        output_path = (
            OUTPUT_DIR
            / f"{dataset_name}_metadata.csv"
        )

        metadata.to_csv(
            output_path,
            index=False,
        )

        # Print a concise processing summary.
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"Role: {dataset_role}")
        print(f"Metadata saved: {output_path}")

    print(
        "\nPhase 2.6 metadata generation completed successfully."
    )


if __name__ == "__main__":
    main()
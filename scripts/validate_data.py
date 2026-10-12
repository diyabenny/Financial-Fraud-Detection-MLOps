import io
import os
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# PHASE 2.3 - DATA VALIDATION
# ============================================================

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not BUCKET_NAME:
    raise ValueError("S3_BUCKET_NAME is not set in .env")


DATASETS = {
    "april": {
        "s3_key": "raw/april/converted_april_file.csv",
        "expected_min_day": 0,
        "expected_max_day": 29,
        "expected_month": 4,
    },
    "may": {
        "s3_key": "raw/may/converted_may_file.csv",
        "expected_min_day": 30,
        "expected_max_day": 48,
        "expected_month": 5,
    },
}


REQUIRED_COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
]


OUTPUT_DIR = Path("reports/validation")


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(s3, s3_key):
    """
    Load a CSV dataset directly from the configured Amazon S3 bucket.

    Parameters:
        s3 (boto3.client): Configured Amazon S3 client.
        s3_key (str): S3 object key identifying the CSV dataset.

    Returns:
        pandas.DataFrame: Dataset loaded from the S3 object.

    Raises:
        botocore.exceptions.ClientError: If the S3 object cannot be
            accessed or does not exist.
        pandas.errors.ParserError: If the CSV file cannot be parsed.

    Side Effects:
        Reads data from Amazon S3. The source dataset is not modified.
    """

    response = s3.get_object(
        Bucket=BUCKET_NAME,
        Key=s3_key
    )

    return pd.read_csv(
        io.BytesIO(response["Body"].read())
    )


# ============================================================
# Validation 1 - Schema
# ============================================================

def validate_schema(df):
    """
    Validate that all required columns are present in the dataset.

    Parameters:
        df (pandas.DataFrame): Dataset to validate.

    Returns:
        tuple:
            - bool: True if all required columns are present.
            - list: Names of columns missing from the dataset.

    Raises:
        None.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    passed = len(missing_columns) == 0

    return passed, missing_columns


# ============================================================
# Validation 2 - Missing Values
# ============================================================

def validate_missing_values(df):
    """
    Check the dataset for missing or null values.

    Parameters:
        df (pandas.DataFrame): Dataset to validate.

    Returns:
        tuple:
            - bool: True if no missing values are found.
            - int: Total number of missing values.

    Raises:
        None.
    """

    missing_by_column = df.isnull().sum()
    total_missing = int(missing_by_column.sum())

    passed = total_missing == 0

    return passed, total_missing


# ============================================================
# Validation 3 - Duplicates
# ============================================================

def validate_duplicates(df):
    """
    Check for duplicate rows and duplicate transaction IDs.

    Parameters:
        df (pandas.DataFrame): Dataset to validate.

    Returns:
        tuple:
            - bool: True if no duplicate rows or transaction IDs exist.
            - int: Number of duplicate complete rows.
            - int: Number of duplicate transaction IDs.

    Raises:
        KeyError: If TRANSACTION_ID is not present in the dataset.
    """

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_transaction_ids = int(
        df["TRANSACTION_ID"].duplicated().sum()
    )

    passed = (
        duplicate_rows == 0
        and duplicate_transaction_ids == 0
    )

    return (
        passed,
        duplicate_rows,
        duplicate_transaction_ids,
    )


# ============================================================
# Validation 4 - Fraud Values
# ============================================================

def validate_fraud_values(df):
    """
    Validate that TX_FRAUD contains only the values 0 and 1.

    A value of 0 represents a normal transaction and a value of 1
    represents a fraudulent transaction.

    Parameters:
        df (pandas.DataFrame): Dataset to validate.

    Returns:
        tuple:
            - bool: True if only 0 and 1 are present.
            - list: Invalid values found in TX_FRAUD.

    Raises:
        KeyError: If TX_FRAUD is not present in the dataset.
    """

    actual_values = set(
        df["TX_FRAUD"].dropna().unique()
    )

    invalid_values = actual_values - {0, 1}

    passed = len(invalid_values) == 0

    return passed, sorted(invalid_values)


# ============================================================
# Validation 5 - Transaction Amount
# ============================================================

def validate_transaction_amount(df):
    """
    Validate that transaction amounts are not negative.

    Parameters:
        df (pandas.DataFrame): Dataset to validate.

    Returns:
        tuple:
            - bool: True if no negative transaction amounts exist.
            - int: Number of transactions with negative amounts.

    Raises:
        KeyError: If TX_AMOUNT is not present in the dataset.
    """

    negative_amounts = int(
        (df["TX_AMOUNT"] < 0).sum()
    )

    passed = negative_amounts == 0

    return passed, negative_amounts


# ============================================================
# Validation 6 - TX_TIME_DAYS Range
# ============================================================

def validate_time_days(
    df,
    expected_min_day,
    expected_max_day
):
    """
    Validate that TX_TIME_DAYS falls within the expected range.

    Parameters:
        df (pandas.DataFrame): Dataset to validate.
        expected_min_day (int): Minimum valid transaction day.
        expected_max_day (int): Maximum valid transaction day.

    Returns:
        tuple:
            - bool: True if the actual range matches the expected range
              and no rows fall outside it.
            - int or float: Actual minimum TX_TIME_DAYS value.
            - int or float: Actual maximum TX_TIME_DAYS value.
            - int: Number of rows outside the expected range.

    Raises:
        KeyError: If TX_TIME_DAYS is not present in the dataset.
    """

    actual_min = df["TX_TIME_DAYS"].min()
    actual_max = df["TX_TIME_DAYS"].max()

    invalid_rows = int(
        (
            (df["TX_TIME_DAYS"] < expected_min_day)
            |
            (df["TX_TIME_DAYS"] > expected_max_day)
        ).sum()
    )

    passed = (
        actual_min == expected_min_day
        and actual_max == expected_max_day
        and invalid_rows == 0
    )

    return (
        passed,
        actual_min,
        actual_max,
        invalid_rows,
    )


# ============================================================
# Validation 7 - Datetime
# ============================================================

def validate_datetime(df, expected_month):
    """
    Validate transaction datetime values and their expected month.

    The function checks whether datetime values can be parsed and
    whether all valid transactions belong to the expected month.

    Parameters:
        df (pandas.DataFrame): Dataset to validate.
        expected_month (int): Expected calendar month number.

    Returns:
        tuple:
            - bool: True if all dates are valid and belong to the
              expected month.
            - int: Number of invalid datetime values.
            - list: Calendar months found in valid datetime values.
            - int: Number of rows belonging to an unexpected month.

    Raises:
        KeyError: If TX_DATETIME is not present in the dataset.
    """

    datetime_values = pd.to_datetime(
        df["TX_DATETIME"],
        errors="coerce"
    )

    invalid_dates = int(
        datetime_values.isna().sum()
    )

    actual_months = sorted(
        datetime_values
        .dropna()
        .dt.month
        .unique()
        .tolist()
    )

    wrong_month_rows = int(
        (
            datetime_values.notna()
            &
            (datetime_values.dt.month != expected_month)
        ).sum()
    )

    passed = (
        invalid_dates == 0
        and wrong_month_rows == 0
    )

    return (
        passed,
        invalid_dates,
        actual_months,
        wrong_month_rows,
    )


# ============================================================
# Validation 8 - Fraud Scenarios
# ============================================================

def validate_fraud_scenarios(df):
    """
    Validate TX_FRAUD_SCENARIO according to TX_FRAUD.

    Normal transactions must have TX_FRAUD_SCENARIO equal to 0.
    Fraudulent transactions must have a scenario value of 1, 2, or 3.

    Parameters:
        df (pandas.DataFrame): Dataset to validate.

    Returns:
        tuple:
            - bool: True if all fraud scenario rules are satisfied.
            - int: Number of fraudulent transactions with invalid
              scenario values.
            - int: Number of normal transactions with invalid
              scenario values.

    Raises:
        KeyError: If TX_FRAUD or TX_FRAUD_SCENARIO is not present.
    """

    fraud_scenarios = df.loc[
        df["TX_FRAUD"] == 1,
        "TX_FRAUD_SCENARIO"
    ]

    normal_scenarios = df.loc[
        df["TX_FRAUD"] == 0,
        "TX_FRAUD_SCENARIO"
    ]

    invalid_fraud_scenarios = int(
        (~fraud_scenarios.isin([1, 2, 3])).sum()
    )

    invalid_normal_scenarios = int(
        (~normal_scenarios.isin([0])).sum()
    )

    passed = (
        invalid_fraud_scenarios == 0
        and invalid_normal_scenarios == 0
    )

    return (
        passed,
        invalid_fraud_scenarios,
        invalid_normal_scenarios,
    )


# ============================================================
# Validation 9 - Data Types
# ============================================================

def validate_data_types(df):
    """
    Validate expected data types for numerical dataset columns.

    Integer columns are checked using pandas integer dtype validation,
    while TX_AMOUNT is checked as a numeric column.

    Parameters:
        df (pandas.DataFrame): Dataset to validate.

    Returns:
        tuple:
            - bool: True if all expected columns have valid data types.
            - list: Names of columns with invalid data types.

    Raises:
        KeyError: If an expected column is missing from the dataset.
    """

    integer_columns = [
        "TRANSACTION_ID",
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TX_TIME_SECONDS",
        "TX_TIME_DAYS",
        "TX_FRAUD",
        "TX_FRAUD_SCENARIO",
    ]

    numeric_columns = [
        "TX_AMOUNT"
    ]

    invalid_columns = []

    for column in integer_columns:
        if not pd.api.types.is_integer_dtype(df[column]):
            invalid_columns.append(column)

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            invalid_columns.append(column)

    passed = len(invalid_columns) == 0

    return passed, invalid_columns


# ============================================================
# Validation 10 - Time Consistency
# ============================================================

def validate_time_consistency(df):
    """
    Validate consistency between TX_TIME_SECONDS and TX_TIME_DAYS.

    The expected relationship is:
        TX_TIME_SECONDS // 86400 == TX_TIME_DAYS

    Parameters:
        df (pandas.DataFrame): Dataset to validate.

    Returns:
        tuple:
            - bool: True if all rows satisfy the time relationship.
            - int: Number of rows with inconsistent time values.

    Raises:
        KeyError: If TX_TIME_SECONDS or TX_TIME_DAYS is missing.
    """

    inconsistent_rows = int(
        (
            df["TX_TIME_SECONDS"] // 86400
            != df["TX_TIME_DAYS"]
        ).sum()
    )

    passed = inconsistent_rows == 0

    return passed, inconsistent_rows


# ============================================================
# Validate One Dataset
# ============================================================

def validate_dataset(dataset_name, config, s3):
    """
    Run all Phase 2.3 validation checks for one dataset.

    Parameters:
        dataset_name (str): Name of the dataset being validated,
            such as "april" or "may".
        config (dict): Dataset configuration containing the S3 key,
            expected day range, and expected month.
        s3 (boto3.client): Configured Amazon S3 client.

    Returns:
        str: Validation status, either "ACCEPTED" or "REJECTED".

    Raises:
        botocore.exceptions.ClientError: If the dataset cannot be
            retrieved from S3.
        KeyError: If required dataset columns are missing during
            validation.
        pandas.errors.ParserError: If the CSV cannot be parsed.

    Side Effects:
        Creates the validation report directory if it does not exist
        and writes validation_results.csv and validation_summary.csv.
    """

    print("\n" + "=" * 70)
    print(f"VALIDATING {dataset_name.upper()}")
    print("=" * 70)

    print("S3 bucket:", BUCKET_NAME)
    print("S3 key:", config["s3_key"])

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_dataset(
        s3,
        config["s3_key"]
    )

    print("\nDataset loaded successfully.")

    results = []

    # --------------------------------------------------------
    # Dataset size
    # --------------------------------------------------------

    rows = len(df)
    columns = len(df.columns)

    print("\n[1] Dataset Size")
    print("Rows:", rows)
    print("Columns:", columns)

    results.append({
        "Check": "Dataset Size",
        "Status": "PASS",
        "Details": f"{rows} rows, {columns} columns",
    })

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    schema_pass, missing_columns = validate_schema(df)

    print("\n[2] Schema Validation")
    print("Status:", "PASS" if schema_pass else "FAIL")

    if missing_columns:
        print("Missing columns:", missing_columns)

    results.append({
        "Check": "Schema",
        "Status": "PASS" if schema_pass else "FAIL",
        "Details": (
            "All required columns present"
            if schema_pass
            else f"Missing columns: {missing_columns}"
        ),
    })

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_pass, total_missing = (
        validate_missing_values(df)
    )

    print("\n[3] Missing Value Validation")
    print("Status:", "PASS" if missing_pass else "FAIL")
    print("Total missing values:", total_missing)

    results.append({
        "Check": "Missing Values",
        "Status": "PASS" if missing_pass else "FAIL",
        "Details": f"Total missing values: {total_missing}",
    })

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    (
        duplicate_pass,
        duplicate_rows,
        duplicate_ids,
    ) = validate_duplicates(df)

    print("\n[4] Duplicate Validation")
    print("Status:", "PASS" if duplicate_pass else "FAIL")
    print("Duplicate rows:", duplicate_rows)
    print("Duplicate transaction IDs:", duplicate_ids)

    results.append({
        "Check": "Duplicates",
        "Status": "PASS" if duplicate_pass else "FAIL",
        "Details": (
            f"Duplicate rows: {duplicate_rows}; "
            f"Duplicate transaction IDs: {duplicate_ids}"
        ),
    })

    # --------------------------------------------------------
    # Fraud values
    # --------------------------------------------------------

    fraud_pass, invalid_fraud_values = (
        validate_fraud_values(df)
    )

    print("\n[5] TX_FRAUD Validation")
    print("Status:", "PASS" if fraud_pass else "FAIL")
    print("Invalid values:", invalid_fraud_values)

    results.append({
        "Check": "TX_FRAUD Values",
        "Status": "PASS" if fraud_pass else "FAIL",
        "Details": (
            "Only valid values 0 and 1 found"
            if fraud_pass
            else f"Invalid values: {invalid_fraud_values}"
        ),
    })

    # --------------------------------------------------------
    # Transaction amount
    # --------------------------------------------------------

    amount_pass, negative_amounts = (
        validate_transaction_amount(df)
    )

    print("\n[6] TX_AMOUNT Validation")
    print("Status:", "PASS" if amount_pass else "FAIL")
    print("Negative amounts:", negative_amounts)

    results.append({
        "Check": "TX_AMOUNT",
        "Status": "PASS" if amount_pass else "FAIL",
        "Details": f"Negative amounts: {negative_amounts}",
    })

    # --------------------------------------------------------
    # TX_TIME_DAYS
    # --------------------------------------------------------

    (
        time_days_pass,
        actual_min,
        actual_max,
        invalid_time_rows,
    ) = validate_time_days(
        df,
        config["expected_min_day"],
        config["expected_max_day"],
    )

    print("\n[7] TX_TIME_DAYS Validation")
    print(
        "Status:",
        "PASS" if time_days_pass else "FAIL"
    )
    print(
        "Expected range:",
        config["expected_min_day"],
        "to",
        config["expected_max_day"],
    )
    print(
        "Actual range:",
        actual_min,
        "to",
        actual_max,
    )
    print("Invalid rows:", invalid_time_rows)

    results.append({
        "Check": "TX_TIME_DAYS Range",
        "Status": "PASS" if time_days_pass else "FAIL",
        "Details": (
            f"Expected: "
            f"{config['expected_min_day']}-"
            f"{config['expected_max_day']}; "
            f"Actual: {actual_min}-{actual_max}; "
            f"Invalid rows: {invalid_time_rows}"
        ),
    })

    # --------------------------------------------------------
    # Datetime
    # --------------------------------------------------------

    (
        datetime_pass,
        invalid_dates,
        actual_months,
        wrong_month_rows,
    ) = validate_datetime(
        df,
        config["expected_month"],
    )

    print("\n[8] TX_DATETIME Validation")
    print(
        "Status:",
        "PASS" if datetime_pass else "FAIL"
    )
    print(
        "Invalid dates:",
        invalid_dates
    )
    print(
        "Actual months:",
        actual_months
    )
    print(
        "Wrong-month rows:",
        wrong_month_rows
    )

    results.append({
        "Check": "TX_DATETIME",
        "Status": "PASS" if datetime_pass else "FAIL",
        "Details": (
            f"Invalid dates: {invalid_dates}; "
            f"Actual months: {actual_months}; "
            f"Wrong-month rows: {wrong_month_rows}"
        ),
    })

    # --------------------------------------------------------
    # Fraud scenarios
    # --------------------------------------------------------

    (
        scenario_pass,
        invalid_fraud_scenarios,
        invalid_normal_scenarios,
    ) = validate_fraud_scenarios(df)

    print("\n[9] Fraud Scenario Validation")
    print(
        "Status:",
        "PASS" if scenario_pass else "FAIL"
    )
    print(
        "Invalid fraud scenarios:",
        invalid_fraud_scenarios
    )
    print(
        "Invalid normal scenarios:",
        invalid_normal_scenarios
    )

    results.append({
        "Check": "TX_FRAUD_SCENARIO",
        "Status": "PASS" if scenario_pass else "FAIL",
        "Details": (
            f"Invalid fraud scenarios: "
            f"{invalid_fraud_scenarios}; "
            f"Invalid normal scenarios: "
            f"{invalid_normal_scenarios}"
        ),
    })

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    type_pass, invalid_type_columns = (
        validate_data_types(df)
    )

    print("\n[10] Data Type Validation")
    print(
        "Status:",
        "PASS" if type_pass else "FAIL"
    )

    if invalid_type_columns:
        print(
            "Invalid type columns:",
            invalid_type_columns
        )

    results.append({
        "Check": "Data Types",
        "Status": "PASS" if type_pass else "FAIL",
        "Details": (
            "All expected data types are valid"
            if type_pass
            else f"Invalid columns: {invalid_type_columns}"
        ),
    })

    # --------------------------------------------------------
    # Time consistency
    # --------------------------------------------------------

    (
        consistency_pass,
        inconsistent_rows,
    ) = validate_time_consistency(df)

    print("\n[11] Time Consistency Validation")
    print(
        "Status:",
        "PASS" if consistency_pass else "FAIL"
    )
    print(
        "Inconsistent rows:",
        inconsistent_rows
    )

    results.append({
        "Check": "Time Consistency",
        "Status": "PASS" if consistency_pass else "FAIL",
        "Details": (
            f"Inconsistent rows: {inconsistent_rows}"
        ),
    })

    # ========================================================
    # VALIDATION GATE
    # ========================================================

    results_df = pd.DataFrame(results)

    all_passed = (
        results_df["Status"] == "PASS"
    ).all()

    if all_passed:
        validation_status = "ACCEPTED"
    else:
        validation_status = "REJECTED"

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_DIR / dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_dir / "validation_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame({
        "Dataset": [dataset_name],
        "Rows": [rows],
        "Columns": [columns],
        "Validation_Status": [validation_status],
        "Failed_Checks": [
            int(
                (results_df["Status"] == "FAIL").sum()
            )
        ],
    })

    summary.to_csv(
        output_dir / "validation_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Print final decision
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    if validation_status == "ACCEPTED":

        print(
            f"{dataset_name.upper()} → DATASET ACCEPTED"
        )
        print(
            "All mandatory validation checks passed."
        )

    else:

        print(
            f"{dataset_name.upper()} → DATASET REJECTED"
        )
        print(
            "One or more mandatory validation checks failed."
        )

    print("-" * 70)

    print(
        "Validation report:",
        output_dir / "validation_results.csv"
    )

    print(
        "Validation summary:",
        output_dir / "validation_summary.csv"
    )

    return validation_status


# ============================================================
# Main
# ============================================================

def main():
    """
    Execute Phase 2.3 validation for all configured datasets.

    Parameters:
        None.

    Returns:
        None.

    Raises:
        ValueError: If S3_BUCKET_NAME is not configured.
        botocore.exceptions.ClientError: If a dataset cannot be
            retrieved from Amazon S3.

    Side Effects:
        Reads April and May datasets from S3 and creates local
        validation reports under reports/validation/.
    """

    print("=" * 70)
    print("PHASE 2.3 - DATA VALIDATION")
    print("=" * 70)

    print("\nS3 bucket:", BUCKET_NAME)

    s3 = boto3.client("s3")

    validation_results = {}

    # --------------------------------------------------------
    # Validate April and May
    # --------------------------------------------------------

    for dataset_name, config in DATASETS.items():

        status = validate_dataset(
            dataset_name,
            config,
            s3
        )

        validation_results[dataset_name] = status

    # ========================================================
    # FINAL VALIDATION GATE
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL VALIDATION GATE")
    print("=" * 70)

    for dataset_name, status in validation_results.items():

        print(
            f"{dataset_name.upper():<10} → {status}"
        )

    if all(
        status == "ACCEPTED"
        for status in validation_results.values()
    ):

        print("\nALL DATASETS ACCEPTED")
        print(
            "PHASE 2.3 DATA VALIDATION PASSED"
        )

    else:

        print("\nONE OR MORE DATASETS REJECTED")
        print(
            "PHASE 2.3 DATA VALIDATION FAILED"
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
import io
import os
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# PHASE 2.2 - EXPLORATORY DATA ANALYSIS
# ============================================================

# ------------------------------------------------------------
# S3 Configuration
# ------------------------------------------------------------

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not BUCKET_NAME:
    raise ValueError("S3_BUCKET_NAME is not set")


DATASETS = {
    "april": "raw/april/converted_april_file.csv",
    "may": "raw/may/converted_may_file.csv",
}


# ------------------------------------------------------------
# Output Configuration
# ------------------------------------------------------------

BASE_OUTPUT_DIR = Path("reports/eda")


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(s3, key):
    """
    Load a CSV transaction dataset directly from Amazon S3.

    Args:
        s3: Boto3 S3 client used to access the bucket.
        key (str): S3 object key of the CSV dataset.

    Returns:
        pandas.DataFrame: Transaction dataset loaded from S3.

    Raises:
        botocore.exceptions.ClientError: If the S3 object
            cannot be retrieved.
        pandas.errors.ParserError: If the CSV cannot be parsed.
    """
    response = s3.get_object(
        Bucket=BUCKET_NAME,
        Key=key
    )

    return pd.read_csv(
        io.BytesIO(response["Body"].read())
    )


# ============================================================
# Basic Information
# ============================================================

def basic_information(df, output_dir):
    """
    Generate basic information about the transaction dataset.

    The analysis includes dataset size, column names, data types,
    missing values, duplicate rows, and duplicate transaction IDs.
    The results are saved as CSV files in the specified output
    directory.

    Args:
        df (pandas.DataFrame): Transaction dataset to analyze.
        output_dir (Path): Directory where analysis results
            are saved.

    Returns:
        None: Results are printed and saved to CSV files.
    """
    print("\n--- Basic Information ---")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())
    print(
        "Duplicate transaction IDs:",
        df["TRANSACTION_ID"].duplicated().sum()
    )

    # Save basic information
    basic_results = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing Cells",
            "Duplicate Rows",
            "Duplicate Transaction IDs",
        ],
        "Value": [
            len(df),
            len(df.columns),
            int(df.isnull().sum().sum()),
            int(df.duplicated().sum()),
            int(df["TRANSACTION_ID"].duplicated().sum()),
        ],
    })

    basic_results.to_csv(
        output_dir / "basic_information.csv",
        index=False
    )

    # Save column names
    column_results = pd.DataFrame({
        "Column": df.columns
    })

    column_results.to_csv(
        output_dir / "columns.csv",
        index=False
    )

    # Save data types
    dtype_results = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": df.dtypes.astype(str).values
    })

    dtype_results.to_csv(
        output_dir / "data_types.csv",
        index=False
    )


# ============================================================
# Date Analysis
# ============================================================

def date_analysis(df, output_dir):
    """
    Analyze transaction date and time information.

    The analysis identifies the earliest and latest transaction
    dates and the minimum and maximum TX_TIME_DAYS values.
    The results are saved as a CSV file.

    Args:
        df (pandas.DataFrame): Transaction dataset to analyze.
        output_dir (Path): Directory where analysis results
            are saved.

    Returns:
        None: Results are printed and saved to a CSV file.
    """
    df = df.copy()
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

    start_date = df["TX_DATETIME"].min()
    end_date = df["TX_DATETIME"].max()

    min_days = df["TX_TIME_DAYS"].min()
    max_days = df["TX_TIME_DAYS"].max()

    print("\n--- Date Information ---")
    print("Start:", start_date)
    print("End:", end_date)
    print("TX_TIME_DAYS:", min_days, "to", max_days)

    date_results = pd.DataFrame({
        "Metric": [
            "Start Date",
            "End Date",
            "Minimum TX_TIME_DAYS",
            "Maximum TX_TIME_DAYS",
        ],
        "Value": [
            start_date,
            end_date,
            min_days,
            max_days,
        ],
    })

    date_results.to_csv(
        output_dir / "date_information.csv",
        index=False
    )


# ============================================================
# Fraud Analysis
# ============================================================

def fraud_analysis(df, output_dir):
    """
    Analyze fraud distribution and fraud scenarios.

    The analysis calculates the number and percentage of normal
    and fraudulent transactions and examines the distribution
    of fraud scenarios. Corresponding CSV files and plots are
    saved in the specified output directory.

    Args:
        df (pandas.DataFrame): Transaction dataset to analyze.
        output_dir (Path): Directory where analysis results
            and plots are saved.

    Returns:
        None: Results are printed and saved to CSV and image files.
    """
    fraud_counts = df["TX_FRAUD"].value_counts().sort_index()

    fraud_percentage = df["TX_FRAUD"].mean() * 100

    print("\n--- Fraud Distribution ---")
    print(fraud_counts)
    print("Fraud percentage:", round(fraud_percentage, 4), "%")

    fraud_results = pd.DataFrame({
        "Fraud_Status": fraud_counts.index,
        "Count": fraud_counts.values,
    })

    fraud_results["Percentage"] = (
        fraud_results["Count"] / len(df) * 100
    )

    fraud_results.to_csv(
        output_dir / "fraud_distribution.csv",
        index=False
    )

    # Fraud distribution plot
    plt.figure()
    fraud_counts.plot(kind="bar")
    plt.title("Fraud Distribution")
    plt.xlabel("Fraud Status")
    plt.ylabel("Number of Transactions")
    plt.xticks(
        ticks=[0, 1],
        labels=["Normal", "Fraud"],
        rotation=0
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "fraud_distribution.png"
    )
    plt.close()

    # Fraud scenarios
    fraud_scenarios = (
        df.loc[df["TX_FRAUD"] == 1, "TX_FRAUD_SCENARIO"]
        .value_counts()
        .sort_index()
    )

    print("\n--- Fraud Scenarios ---")
    print(fraud_scenarios)

    scenario_results = pd.DataFrame({
        "Fraud_Scenario": fraud_scenarios.index,
        "Count": fraud_scenarios.values,
    })

    scenario_results.to_csv(
        output_dir / "fraud_scenarios.csv",
        index=False
    )

    # Fraud scenario plot
    plt.figure()
    fraud_scenarios.plot(kind="bar")
    plt.title("Fraud Scenario Distribution")
    plt.xlabel("Fraud Scenario")
    plt.ylabel("Number of Fraudulent Transactions")
    plt.tight_layout()
    plt.savefig(
        output_dir / "fraud_scenarios.png"
    )
    plt.close()


# ============================================================
# Transaction Amount Analysis
# ============================================================

def amount_analysis(df, output_dir):
    """
    Analyze transaction amount statistics and distributions.

    The analysis calculates minimum, mean, median, and maximum
    transaction amounts and compares transaction amounts by
    fraud status. Histograms and boxplots are also generated.

    Args:
        df (pandas.DataFrame): Transaction dataset to analyze.
        output_dir (Path): Directory where analysis results
            and plots are saved.

    Returns:
        None: Results are printed and saved to CSV and image files.
    """
    print("\n--- Transaction Amount Statistics ---")

    amount_statistics = pd.DataFrame({
        "Statistic": [
            "Minimum",
            "Mean",
            "Median",
            "Maximum",
        ],
        "Value": [
            df["TX_AMOUNT"].min(),
            df["TX_AMOUNT"].mean(),
            df["TX_AMOUNT"].median(),
            df["TX_AMOUNT"].max(),
        ],
    })

    print(amount_statistics)

    amount_statistics.to_csv(
        output_dir / "amount_statistics.csv",
        index=False
    )

    # Overall transaction amount distribution
    plt.figure()
    df["TX_AMOUNT"].plot(
        kind="hist",
        bins=50
    )
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        output_dir / "transaction_amount_distribution.png"
    )
    plt.close()

    # Amount comparison by fraud status
    amount_by_fraud = (
        df.groupby("TX_FRAUD")["TX_AMOUNT"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    amount_by_fraud.to_csv(
        output_dir / "amount_by_fraud.csv",
        index=False
    )

    print("\n--- Amount Statistics by Fraud Status ---")
    print(amount_by_fraud)

    # Boxplot comparing normal and fraud amounts
    normal_amounts = df.loc[
        df["TX_FRAUD"] == 0,
        "TX_AMOUNT"
    ]

    fraud_amounts = df.loc[
        df["TX_FRAUD"] == 1,
        "TX_AMOUNT"
    ]

    plt.figure()
    plt.boxplot(
        [normal_amounts, fraud_amounts],
        tick_labels=["Normal", "Fraud"]
    )
    plt.title("Transaction Amount by Fraud Status")
    plt.xlabel("Fraud Status")
    plt.ylabel("Transaction Amount")
    plt.tight_layout()
    plt.savefig(
        output_dir / "amount_by_fraud.png"
    )
    plt.close()


# ============================================================
# Customer and Terminal Analysis
# ============================================================

def customer_terminal_analysis(df, output_dir):
    """
    Analyze the number of unique customers and terminals.

    Args:
        df (pandas.DataFrame): Transaction dataset to analyze.
        output_dir (Path): Directory where analysis results
            are saved.

    Returns:
        None: Results are printed and saved to a CSV file.
    """
    unique_customers = df["CUSTOMER_ID"].nunique()
    unique_terminals = df["TERMINAL_ID"].nunique()

    print("\n--- Customer and Terminal Information ---")
    print("Unique customers:", unique_customers)
    print("Unique terminals:", unique_terminals)

    results = pd.DataFrame({
        "Metric": [
            "Unique Customers",
            "Unique Terminals",
        ],
        "Value": [
            unique_customers,
            unique_terminals,
        ],
    })

    results.to_csv(
        output_dir / "customer_terminal_information.csv",
        index=False
    )


# ============================================================
# Daily Analysis
# ============================================================

def daily_analysis(df, output_dir):
    """
    Analyze daily transaction activity and fraud activity.

    The analysis calculates the total number of transactions,
    fraudulent transactions, and fraud percentage for each day.
    Daily transaction and fraud activity plots are also generated.

    Args:
        df (pandas.DataFrame): Transaction dataset to analyze.
        output_dir (Path): Directory where analysis results
            and plots are saved.

    Returns:
        None: Results are printed and saved to CSV and image files.
    """
    df = df.copy()
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df["DATE"] = df["TX_DATETIME"].dt.date

    daily_activity = (
        df.groupby("DATE")
        .agg(
            Total_Transactions=("TRANSACTION_ID", "size"),
            Fraud_Transactions=("TX_FRAUD", "sum"),
        )
        .reset_index()
    )

    daily_activity["Fraud_Percentage"] = (
        daily_activity["Fraud_Transactions"]
        / daily_activity["Total_Transactions"]
        * 100
    )

    print("\n--- Daily Activity ---")
    print(daily_activity)

    daily_activity.to_csv(
        output_dir / "daily_activity.csv",
        index=False
    )

    # Daily transactions
    plt.figure()
    plt.plot(
        daily_activity["DATE"],
        daily_activity["Total_Transactions"]
    )
    plt.title("Daily Transaction Activity")
    plt.xlabel("Date")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        output_dir / "daily_transactions.png"
    )
    plt.close()

    # Daily fraud
    plt.figure()
    plt.plot(
        daily_activity["DATE"],
        daily_activity["Fraud_Transactions"]
    )
    plt.title("Daily Fraud Activity")
    plt.xlabel("Date")
    plt.ylabel("Number of Fraudulent Transactions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        output_dir / "daily_fraud.png"
    )
    plt.close()


# ============================================================
# Correlation Analysis
# ============================================================

def correlation_analysis(df, output_dir):
    """
    Calculate correlations between selected numerical variables.

    The analysis examines the relationship between transaction
    amount, transaction time, transaction days, and fraud status.
    The resulting correlation matrix is saved as a CSV file.

    Args:
        df (pandas.DataFrame): Transaction dataset to analyze.
        output_dir (Path): Directory where the correlation
            matrix is saved.

    Returns:
        None: The correlation matrix is printed and saved to CSV.
    """
    correlation_columns = [
        "TX_AMOUNT",
        "TX_TIME_SECONDS",
        "TX_TIME_DAYS",
        "TX_FRAUD",
    ]

    correlation_matrix = df[
        correlation_columns
    ].corr()

    print("\n--- Correlation Matrix ---")
    print(correlation_matrix)

    correlation_matrix.to_csv(
        output_dir / "correlation_matrix.csv"
    )


# ============================================================
# Main
# ============================================================

def main():
    """
    Run the Phase 2.2 exploratory data analysis workflow.

    The workflow loads the April and May transaction datasets
    from S3 and performs basic information, date, fraud,
    transaction amount, customer and terminal, daily activity,
    and correlation analyses. Results are saved under the
    reports/eda directory.

    Returns:
        None: EDA results are printed and saved to files.

    Raises:
        ValueError: If S3_BUCKET_NAME is not configured.
        botocore.exceptions.ClientError: If an S3 dataset
            cannot be retrieved.
    """
    s3 = boto3.client("s3")

    for dataset_name, s3_key in DATASETS.items():

        print("\n" + "=" * 60)
        print(f"EDA FOR {dataset_name.upper()}")
        print("=" * 60)

        output_dir = BASE_OUTPUT_DIR / dataset_name
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print("\nReading:", s3_key)

        df = load_dataset(
            s3,
            s3_key
        )

        basic_information(
            df,
            output_dir
        )

        date_analysis(
            df,
            output_dir
        )

        fraud_analysis(
            df,
            output_dir
        )

        amount_analysis(
            df,
            output_dir
        )

        customer_terminal_analysis(
            df,
            output_dir
        )

        daily_analysis(
            df,
            output_dir
        )

        correlation_analysis(
            df,
            output_dir
        )

        print(
            f"\nCompleted EDA for {dataset_name.upper()}"
        )
        print(
            "Results saved to:",
            output_dir
        )

    print("\n" + "=" * 60)
    print("PHASE 2.2 EDA COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
import io
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import pandas as pd


# S3 configuration
BUCKET_NAME = "financial-fraud-mlops-avani-2026"

DATASETS = {
    "april": "raw/april/converted_april_file.csv",
    "may": "raw/may/converted_may_file.csv",
}

# Output directory
BASE_OUTPUT_DIR = Path("reports/eda")


def load_dataset(s3, key):
    """Read a CSV file directly from S3."""
    response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_csv(io.BytesIO(response["Body"].read()))


def basic_information(df, output_dir):
    """Generate basic dataset information."""

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


def date_analysis(df, output_dir):
    """Analyze transaction date and time information."""

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


def fraud_analysis(df, output_dir):
    """Analyze fraud distribution and fraud scenarios."""

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


def amount_analysis(df, output_dir):
    """Analyze transaction amounts."""

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


def customer_terminal_analysis(df, output_dir):
    """Analyze customers and terminals."""

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


def daily_analysis(df, output_dir):
    """Analyze daily transaction and fraud activity."""

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


def correlation_analysis(df, output_dir):
    """Calculate exploratory numerical correlations."""

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


def main():
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
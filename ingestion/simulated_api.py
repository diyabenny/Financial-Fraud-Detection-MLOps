import io
import os
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Original May dataset in S3
SOURCE_S3_KEY = "raw/may/converted_may_file.csv"

# Destination for simulated incoming batches
INCOMING_PREFIX = "incoming/may/"

# Expected transaction columns
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


def validate_configuration():
    """Check that the S3 bucket is configured."""

    if not BUCKET_NAME:
        raise ValueError(
            "S3_BUCKET_NAME is not set in the environment or .env file."
        )


def read_source_from_s3():
    """Read the original May CSV directly from S3."""

    print(f"Reading source from S3:")
    print(f"  s3://{BUCKET_NAME}/{SOURCE_S3_KEY}")

    try:
        s3 = boto3.client("s3")

        response = s3.get_object(
            Bucket=BUCKET_NAME,
            Key=SOURCE_S3_KEY
        )

        file_content = response["Body"].read()

        if not file_content:
            raise ValueError("The S3 file is empty.")

        if not SOURCE_S3_KEY.lower().endswith(".csv"):
            raise ValueError("The source file is not a CSV file.")

        df = pd.read_csv(io.BytesIO(file_content))

        print(f"Successfully read {len(df)} rows from S3.")

        return df

    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            f"Failed to read source file from S3: {exc}"
        ) from exc


def validate_dataframe(df):
    """Perform basic ingestion checks."""

    print("\nRunning basic ingestion checks...")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The incoming dataset contains no rows.")

    print("[OK] File format: CSV")
    print("[OK] Required columns: present")
    print(f"[OK] Rows: {len(df)}")
    print(f"[OK] Columns: {len(df.columns)}")

    return True


def create_batches(df, batch_size=10000):
    """Create batches without changing transaction contents."""

    batches = []

    total_rows = len(df)

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)

        batch = df.iloc[start:end].copy()

        batches.append(batch)

    print(f"\nCreated {len(batches)} simulated batches.")
    print(f"Batch size: {batch_size} rows maximum")

    return batches


def upload_batches_to_s3(batches):
    """Upload simulated incoming batches to S3."""

    s3 = boto3.client("s3")

    print("\nUploading simulated batches to S3...")

    for index, batch in enumerate(batches, start=1):

        filename = f"batch_{index:03d}.csv"

        s3_key = f"{INCOMING_PREFIX}{filename}"

        csv_buffer = io.StringIO()

        batch.to_csv(
            csv_buffer,
            index=False
        )

        try:
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=csv_buffer.getvalue().encode("utf-8"),
                ContentType="text/csv"
            )

        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(
                f"Failed to upload {filename}: {exc}"
            ) from exc

        print(
            f"[OK] Uploaded {filename}  "
            f"({len(batch)} rows)"
        )


def main():

    print("=" * 60)
    print("SIMULATED MAY DATA INGESTION")
    print("=" * 60)

    validate_configuration()

    # Step 1: Read original May data from S3
    df = read_source_from_s3()

    # Step 2: Perform basic ingestion checks
    validate_dataframe(df)

    # Step 3: Create simulated incoming batches
    batches = create_batches(
        df,
        batch_size=10000
    )

    # Step 4: Upload batches to incoming/ in S3
    upload_batches_to_s3(batches)

    print("\n" + "=" * 60)
    print("SIMULATED INGESTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
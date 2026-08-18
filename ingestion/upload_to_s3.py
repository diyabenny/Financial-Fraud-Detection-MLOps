import boto3
from pathlib import Path


# AWS S3 configuration
BUCKET_NAME = "financial-fraud-mlops-avani-2026"

# Local dataset location
DATASET_PATH = Path("data/raw/Financial_fraud_dataset.csv")

# Location inside S3
S3_KEY = "raw/Financial_fraud_dataset.csv"


def upload_file_to_s3():
    s3 = boto3.client("s3")

    print(f"Uploading {DATASET_PATH} to S3...")

    s3.upload_file(
        str(DATASET_PATH),
        BUCKET_NAME,
        S3_KEY
    )

    print("Upload completed successfully!")


if __name__ == "__main__":
    upload_file_to_s3()
import os
from pathlib import Path
import boto3
from dotenv import load_dotenv

# Load variables from local .env file
load_dotenv()

# AWS S3 configuration dynamically fetched from environment
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Local dataset location
DATASET_PATH = Path("data/raw/Financial_fraud_dataset.csv")

# Location inside S3
S3_KEY = "raw/Financial_fraud_dataset.csv"


def upload_file_to_s3():
    if not BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME is not set in environment or .env file.")

    s3 = boto3.client("s3")

    print(f"Uploading {DATASET_PATH} to S3 bucket '{BUCKET_NAME}'...")

    s3.upload_file(
        str(DATASET_PATH),
        BUCKET_NAME,
        S3_KEY
    )

    print("Upload completed successfully!")


if __name__ == "__main__":
    upload_file_to_s3()
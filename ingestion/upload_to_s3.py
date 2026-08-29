import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


# Load variables from local .env file
load_dotenv()

# Read the S3 bucket name from .env
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


def upload_file_to_s3(local_path: str, s3_key: str) -> None:
    """
    Upload one local file to the specified S3 object key.
    """

    if not BUCKET_NAME:
        raise ValueError(
            "S3_BUCKET_NAME is not set in the environment or .env file."
        )

    file_path = Path(local_path)

    # Check that the local file actually exists
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Local file does not exist: {file_path}"
        )

    print(f"Local file: {file_path}")
    print(f"S3 bucket: {BUCKET_NAME}")
    print(f"S3 destination: {s3_key}")
    print("Starting upload...")

    try:
        s3 = boto3.client("s3")

        s3.upload_file(
            str(file_path),
            BUCKET_NAME,
            s3_key
        )

    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            f"S3 upload failed: {exc}"
        ) from exc

    print("Upload completed successfully!")


def main() -> None:
    """
    Command-line entry point.

    Usage:
        python ingestion/upload_to_s3.py <local_file> <s3_key>
    """

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python ingestion/upload_to_s3.py "
            "<local_file> <s3_key>"
        )
        sys.exit(1)

    local_file = sys.argv[1]
    s3_key = sys.argv[2]

    upload_file_to_s3(local_file, s3_key)


if __name__ == "__main__":
    main()
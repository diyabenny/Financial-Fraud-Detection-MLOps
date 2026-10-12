import os
import pandas as pd


TRAIN_INPUT_PATH = "data/phase3/train_combined_features.csv"
TRAIN_OUTPUT_PATH = (
    "feast_repo/fraud_features/feature_repo/data/train_combined_features.parquet"
)

VALIDATION_INPUT_PATH = "data/phase3/validation_combined_features.csv"
VALIDATION_OUTPUT_PATH = (
    "feast_repo/fraud_features/feature_repo/data/validation_combined_features.parquet"
)


def convert_csv_to_parquet(input_path, output_path, dataset_name):
    print(f"Reading {dataset_name} features...")

    df = pd.read_csv(input_path)

    print(f"Input shape: {df.shape}")

    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_parquet(output_path, index=False)

    saved_df = pd.read_parquet(output_path)

    assert saved_df.shape == df.shape
    assert list(saved_df.columns) == list(df.columns)
    assert saved_df["TX_DATETIME"].dtype == "datetime64[ns]"

    print(f"Output shape: {saved_df.shape}")
    print(f"Output path: {output_path}")
    print(f"{dataset_name} CSV -> Parquet conversion successful.")


def main():
    convert_csv_to_parquet(
        TRAIN_INPUT_PATH,
        TRAIN_OUTPUT_PATH,
        "training",
    )

    convert_csv_to_parquet(
        VALIDATION_INPUT_PATH,
        VALIDATION_OUTPUT_PATH,
        "validation",
    )


if __name__ == "__main__":
    main()
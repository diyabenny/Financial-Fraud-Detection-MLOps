import os
import pandas as pd


INPUT_PATH = "data/phase3/train_combined_features.csv"
OUTPUT_PATH = "feast_repo/fraud_features/feature_repo/data/train_combined_features.parquet"


def main():
    print("Reading combined training features...")
    df = pd.read_csv(INPUT_PATH)

    print(f"Input shape: {df.shape}")

    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    saved_df = pd.read_parquet(OUTPUT_PATH)

    assert saved_df.shape == df.shape
    assert list(saved_df.columns) == list(df.columns)
    assert saved_df["TX_DATETIME"].dtype == "datetime64[ns]"

    print(f"Output shape: {saved_df.shape}")
    print(f"Output path: {OUTPUT_PATH}")
    print("CSV -> Parquet conversion successful.")


if __name__ == "__main__":
    main()
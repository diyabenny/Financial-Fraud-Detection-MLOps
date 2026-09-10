import pandas as pd
import numpy as np
from feast import FeatureStore


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
MAY_PATH = r"data\phase3\may_processed_features.csv"

BATCH_SIZE = 10000


# ---------------------------------------------------------
# Features registered in Feast
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    "TX_AMOUNT",
    "AMOUNT_LOG",
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
    "TX_IS_WEEKEND",
    "TERMINAL_TX_COUNT_1D",
    "TERMINAL_TX_COUNT_7D",
    "TERMINAL_AVG_AMOUNT_7D",
    "CUSTOMER_TX_COUNT_1D",
    "CUSTOMER_TX_COUNT_7D",
]


# ---------------------------------------------------------
# Load processed May data
# ---------------------------------------------------------
df = pd.read_csv(MAY_PATH)

df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])


# ---------------------------------------------------------
# Calculate original Phase 3 features
# ---------------------------------------------------------
df["AMOUNT_LOG"] = np.log1p(df["TX_AMOUNT"])

df["TX_HOUR"] = df["TX_DATETIME"].dt.hour

df["TX_DAY_OF_WEEK"] = df["TX_DATETIME"].dt.dayofweek

df["TX_IS_WEEKEND"] = (
    df["TX_DAY_OF_WEEK"] >= 5
).astype(int)


# ---------------------------------------------------------
# Connect to Feast
# ---------------------------------------------------------
store = FeatureStore(
    repo_path=r"feast_repo\fraud_features\feature_repo"
)


# ---------------------------------------------------------
# Push May transactions batch by batch
# ---------------------------------------------------------
total_rows = len(df)

total_batches = (
    total_rows + BATCH_SIZE - 1
) // BATCH_SIZE

print(f"Total May transactions: {total_rows}")
print(f"Total batches: {total_batches}")
print()


for batch_number, start in enumerate(
    range(0, total_rows, BATCH_SIZE),
    start=1,
):
    end = min(
        start + BATCH_SIZE,
        total_rows
    )

    batch = df.iloc[start:end]

    push_df = batch[
        ["TRANSACTION_ID", "TX_DATETIME"] + FEATURE_COLUMNS
    ].copy()

    store.push(
        push_source_name="fraud_push_source",
        df=push_df,
    )

    print(
        f"Batch {batch_number}/{total_batches} pushed: "
        f"{len(push_df)} transactions"
    )


print()
print("All May batches pushed successfully.")
print(f"Total transactions pushed: {total_rows}")
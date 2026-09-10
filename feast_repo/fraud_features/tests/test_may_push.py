import pandas as pd
import numpy as np
from feast import FeatureStore


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
MAY_PATH = r"..\..\..\data\phase3\may_processed_features.csv"


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
# Load first May batch
# ---------------------------------------------------------
# Only the first 10,000 transactions are used for this test.
df = pd.read_csv(MAY_PATH, nrows=10000)

df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])


# ---------------------------------------------------------
# Calculate original Phase 3 features
# ---------------------------------------------------------
# These features are not stored in may_processed_features.csv,
# so calculate them here before pushing to Feast.

df["AMOUNT_LOG"] = np.log1p(df["TX_AMOUNT"])

df["TX_HOUR"] = df["TX_DATETIME"].dt.hour

df["TX_DAY_OF_WEEK"] = df["TX_DATETIME"].dt.dayofweek

df["TX_IS_WEEKEND"] = (
    df["TX_DAY_OF_WEEK"] >= 5
).astype(int)


# ---------------------------------------------------------
# Prepare data for Feast
# ---------------------------------------------------------
push_df = df[
    ["TRANSACTION_ID", "TX_DATETIME"] + FEATURE_COLUMNS
].copy()


# ---------------------------------------------------------
# Connect to Feast
# ---------------------------------------------------------
store = FeatureStore(repo_path=".")


# ---------------------------------------------------------
# Push May Batch 1
# ---------------------------------------------------------
store.push(
    push_source_name="fraud_push_source",
    df=push_df,
)

print("May Batch 1 push successful.")
print(f"Transactions pushed: {len(push_df)}")


# ---------------------------------------------------------
# Test online retrieval
# ---------------------------------------------------------
# Retrieve the features for the first May transaction.

test_id = int(push_df.iloc[0]["TRANSACTION_ID"])

result = store.get_online_features(
    features=[
        f"fraud_features:{feature}"
        for feature in FEATURE_COLUMNS
    ],
    entity_rows=[
        {"TRANSACTION_ID": test_id}
    ],
).to_df()


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------
print("\nOnline retrieval for first May transaction:")
print(result)
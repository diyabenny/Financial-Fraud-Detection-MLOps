import pandas as pd
import numpy as np
from feast import FeatureStore


MAY_PATH = r"data\phase3\may_processed_features.csv"

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
# Load May data
# ---------------------------------------------------------
df = pd.read_csv(MAY_PATH)

df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])


# ---------------------------------------------------------
# Calculate the original Phase 3 features
# ---------------------------------------------------------
df["AMOUNT_LOG"] = np.log1p(df["TX_AMOUNT"])

df["TX_HOUR"] = df["TX_DATETIME"].dt.hour

df["TX_DAY_OF_WEEK"] = df["TX_DATETIME"].dt.dayofweek

df["TX_IS_WEEKEND"] = (
    df["TX_DAY_OF_WEEK"] >= 5
).astype(int)


# ---------------------------------------------------------
# Select transactions from different points in May
# ---------------------------------------------------------
test_rows = pd.concat([
    df.iloc[[0]],
    df.iloc[[9999]],
    df.iloc[[50000]],
    df.iloc[[100000]],
    df.iloc[[-1]],
])


# ---------------------------------------------------------
# Connect to Feast
# ---------------------------------------------------------
store = FeatureStore(
    repo_path=r"feast_repo\fraud_features\feature_repo"
)


# ---------------------------------------------------------
# Retrieve features
# ---------------------------------------------------------
entity_rows = [
    {"TRANSACTION_ID": int(row["TRANSACTION_ID"])}
    for _, row in test_rows.iterrows()
]

result = store.get_online_features(
    features=[
        f"fraud_features:{feature}"
        for feature in FEATURE_COLUMNS
    ],
    entity_rows=entity_rows,
).to_df()


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------
print("Online retrieval result:")
print(result.to_string(index=False))

print()
print(f"Transactions requested: {len(entity_rows)}")
print(f"Transactions returned: {len(result)}")


if len(result) == len(entity_rows):
    print("RESULT: PASS")
    print("✓ All selected May transactions were retrieved from Feast.")
else:
    print("RESULT: FAIL")
    print("✗ Some May transactions were not retrieved.")
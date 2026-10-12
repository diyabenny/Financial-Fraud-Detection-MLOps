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
# Load May processed data
# ---------------------------------------------------------
df = pd.read_csv(MAY_PATH)

df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])


# ---------------------------------------------------------
# Calculate the four original Phase 3 features
# ---------------------------------------------------------
df["AMOUNT_LOG"] = np.log1p(df["TX_AMOUNT"])

df["TX_HOUR"] = df["TX_DATETIME"].dt.hour

df["TX_DAY_OF_WEEK"] = df["TX_DATETIME"].dt.dayofweek

df["TX_IS_WEEKEND"] = (
    df["TX_DAY_OF_WEEK"] >= 5
).astype(int)


# ---------------------------------------------------------
# Select 5 transactions from different points in May
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
# Retrieve the same transactions from Feast
# ---------------------------------------------------------
entity_rows = [
    {"TRANSACTION_ID": int(row["TRANSACTION_ID"])}
    for _, row in test_rows.iterrows()
]

feast_result = store.get_online_features(
    features=[
        f"fraud_features:{feature}"
        for feature in FEATURE_COLUMNS
    ],
    entity_rows=entity_rows,
).to_df()


# ---------------------------------------------------------
# Prepare expected and actual values
# ---------------------------------------------------------
expected = test_rows[
    ["TRANSACTION_ID"] + FEATURE_COLUMNS
].copy()

actual = feast_result[
    ["TRANSACTION_ID"] + FEATURE_COLUMNS
].copy()


# ---------------------------------------------------------
# Merge expected and Feast values
# ---------------------------------------------------------
merged = expected.merge(
    actual,
    on="TRANSACTION_ID",
    suffixes=("_expected", "_feast"),
)


# ---------------------------------------------------------
# Compare feature values
# ---------------------------------------------------------
all_match = True

for feature in FEATURE_COLUMNS:

    expected_values = merged[
        f"{feature}_expected"
    ]

    feast_values = merged[
        f"{feature}_feast"
    ]

    # Compare numerical values rather than Pandas dtypes.
    match = np.allclose(
        expected_values.astype(float),
        feast_values.astype(float),
        rtol=1e-5,
        atol=1e-5,
    )

    if not match:
        all_match = False
        print(f"FAIL: {feature}")


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------
print()
print("Transactions compared:", len(merged))

if all_match and len(merged) == len(test_rows):
    print("RESULT: PASS")
    print("✓ Feast values exactly match the processed May feature values.")
else:
    print("RESULT: FAIL")
    print("✗ Feast values do not completely match the processed May values.")
import pandas as pd
import numpy as np
from feast import FeatureStore


# ---------------------------------------------------------
# 1. Load one known-good April/train development transaction
# ---------------------------------------------------------
df = pd.read_parquet(
    "feast_repo/fraud_features/feature_repo/data/train_combined_features.parquet"
)

row = df.iloc[0]


# ---------------------------------------------------------
# 2. Features registered in Feast
# ---------------------------------------------------------
feature_columns = [
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
# 3. Build the row to push into Feast
# ---------------------------------------------------------
push_df = pd.DataFrame([{
    "TRANSACTION_ID": row["TRANSACTION_ID"],
    "TX_DATETIME": row["TX_DATETIME"],
    "TX_AMOUNT": row["TX_AMOUNT"],
    "AMOUNT_LOG": row["AMOUNT_LOG"],
    "TX_HOUR": row["TX_HOUR"],
    "TX_DAY_OF_WEEK": row["TX_DAY_OF_WEEK"],
    "TX_IS_WEEKEND": row["TX_IS_WEEKEND"],
    "TERMINAL_TX_COUNT_1D": row["TERMINAL_TX_COUNT_1D"],
    "TERMINAL_TX_COUNT_7D": row["TERMINAL_TX_COUNT_7D"],
    "TERMINAL_AVG_AMOUNT_7D": row["TERMINAL_AVG_AMOUNT_7D"],
    "CUSTOMER_TX_COUNT_1D": row["CUSTOMER_TX_COUNT_1D"],
    "CUSTOMER_TX_COUNT_7D": row["CUSTOMER_TX_COUNT_7D"],
}])


# ---------------------------------------------------------
# 4. Connect to Feast
# ---------------------------------------------------------
store = FeatureStore(
    repo_path="feast_repo/fraud_features/feature_repo"
)


# ---------------------------------------------------------
# 5. Push feature values
# ---------------------------------------------------------
store.push(
    push_source_name="fraud_push_source",
    df=push_df,
)

print("Push successful.")


# ---------------------------------------------------------
# 6. Retrieve features from Online Store
# ---------------------------------------------------------
online_features = store.get_online_features(
    features=[
        f"fraud_features:{feature}"
        for feature in feature_columns
    ],
    entity_rows=[
        {
            "TRANSACTION_ID": int(row["TRANSACTION_ID"])
        }
    ],
).to_df()


# ---------------------------------------------------------
# 7. Display retrieved values
# ---------------------------------------------------------
print("\nOnline retrieval:")
print(online_features.to_string(index=False))


# ---------------------------------------------------------
# 8. Compare source values with Feast values
# ---------------------------------------------------------
all_match = True

for feature in feature_columns:

    expected_value = float(row[feature])
    actual_value = float(
        online_features.iloc[0][feature]
    )

    if not np.isclose(
        expected_value,
        actual_value,
        rtol=1e-5,
        atol=1e-5,
    ):
        all_match = False

        print(
            f"MISMATCH: {feature} | "
            f"expected={expected_value} | "
            f"actual={actual_value}"
        )


# ---------------------------------------------------------
# 9. Final result
# ---------------------------------------------------------
print()

if all_match and len(online_features) == 1:
    print("RESULT: PASS")
    print("✓ Feast online retrieval matches the source values.")
else:
    print("RESULT: FAIL")
    print("✗ Feast online retrieval does not match the source values.")
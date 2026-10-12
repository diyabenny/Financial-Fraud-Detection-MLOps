import pandas as pd
from feast import FeatureStore


# ---------------------------------------------------------
# 1. Load Avani's validated feature values
# ---------------------------------------------------------
df = pd.read_parquet(
    "feast_repo/fraud_features/feature_repo/data/train_combined_features.parquet"
)

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
# 2. Select transactions for testing
# ---------------------------------------------------------
reference = df[
    ["TRANSACTION_ID", "TX_DATETIME"] + feature_columns
].head(5).copy()

entity_df = reference[
    ["TRANSACTION_ID", "TX_DATETIME"]
].rename(
    columns={"TX_DATETIME": "event_timestamp"}
)


# ---------------------------------------------------------
# 3. Retrieve the same features from Feast
# ---------------------------------------------------------
store = FeatureStore(
    repo_path="feast_repo/fraud_features/feature_repo"
)

retrieval = store.get_historical_features(
    entity_df=entity_df,
    features=[
        f"fraud_features:{feature}"
        for feature in feature_columns
    ],
).to_df()


# ---------------------------------------------------------
# 4. Compare Feast values with Avani's values
# ---------------------------------------------------------
merged = reference.merge(
    retrieval,
    on="TRANSACTION_ID",
    suffixes=("_avani", "_feast"),
)


all_match = True

for feature in feature_columns:
    avani_values = merged[f"{feature}_avani"]
    feast_values = merged[f"{feature}_feast"]

    if pd.api.types.is_numeric_dtype(avani_values):
        matches = (
            avani_values.astype(float)
            .round(5)
            .eq(feast_values.astype(float).round(5))
        )
    else:
        matches = avani_values.eq(feast_values)

    if not matches.all():
        all_match = False
        print(f"❌ MISMATCH: {feature}")

        print(
            merged.loc[
                ~matches,
                [
                    "TRANSACTION_ID",
                    f"{feature}_avani",
                    f"{feature}_feast",
                ],
            ].to_string(index=False)
        )


# ---------------------------------------------------------
# 5. Final result
# ---------------------------------------------------------
if all_match:
    print("\nRESULT: PASS")
    print("✓ Feast historical retrieval matches Avani's values.")
else:
    print("\nRESULT: FAIL")
    print("✗ Some Feast values do not match Avani's values.")
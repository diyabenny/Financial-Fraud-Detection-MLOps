from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float32, Int64


# ---------------------------------------------------------
# Entity
# ---------------------------------------------------------
# Each transaction is uniquely identified by TRANSACTION_ID.
transaction = Entity(
    name="transaction",
    join_keys=["TRANSACTION_ID"],
)


# ---------------------------------------------------------
# Historical / batch source
# ---------------------------------------------------------
# This contains Avani's already-calculated, validated
# Phase 3 training feature values.
fraud_training_source = FileSource(
    name="fraud_training_source",
    path="data/train_combined_features.parquet",
    timestamp_field="TX_DATETIME",
)


# ---------------------------------------------------------
# Push source for newly arriving May transactions
# ---------------------------------------------------------
# Python calculates the leakage-safe features first.
# The calculated values are then pushed into Feast.
fraud_push_source = PushSource(
    name="fraud_push_source",
    batch_source=fraud_training_source,
)


# ---------------------------------------------------------
# Feature View
# ---------------------------------------------------------
# These are the 10 features used by the fraud model.
fraud_features = FeatureView(
    name="fraud_features",
    entities=[transaction],
    ttl=None,
    schema=[
        Field(name="TX_AMOUNT", dtype=Float32),
        Field(name="AMOUNT_LOG", dtype=Float32),
        Field(name="TX_HOUR", dtype=Int64),
        Field(name="TX_DAY_OF_WEEK", dtype=Int64),
        Field(name="TX_IS_WEEKEND", dtype=Int64),
        Field(name="TERMINAL_TX_COUNT_1D", dtype=Float32),
        Field(name="TERMINAL_TX_COUNT_7D", dtype=Float32),
        Field(name="TERMINAL_AVG_AMOUNT_7D", dtype=Float32),
        Field(name="CUSTOMER_TX_COUNT_1D", dtype=Float32),
        Field(name="CUSTOMER_TX_COUNT_7D", dtype=Float32),
    ],
    online=True,
    source=fraud_push_source,
)
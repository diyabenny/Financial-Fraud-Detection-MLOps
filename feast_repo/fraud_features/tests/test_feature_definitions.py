import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "feature_repo"))

from feature_definitions import fraud_features


# ---------------------------------------------------------
# Expected model features
# ---------------------------------------------------------
EXPECTED_FEATURES = [
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
# Get features registered in Feast
# ---------------------------------------------------------
REGISTERED_FEATURES = [
    field.name
    for field in fraud_features.schema
]


# ---------------------------------------------------------
# Compare feature definitions
# ---------------------------------------------------------
if set(REGISTERED_FEATURES) == set(EXPECTED_FEATURES):
    print("RESULT: PASS")
    print("✓ Feast feature definitions match the approved 10 features.")
else:
    print("RESULT: FAIL")
    print("✗ Feast feature definitions do not match.")

    print("\nExpected features:")
    print(EXPECTED_FEATURES)

    print("\nRegistered features:")
    print(REGISTERED_FEATURES)
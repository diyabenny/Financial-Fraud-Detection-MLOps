"""
Day 5 Step 12: Feature Pipeline -> Feast.

Controlled technical test for transaction 288062.

Scope:
    5-field validated transaction
        -> existing April history
        -> existing leakage-safe feature calculation
        -> 10 approved features
        -> Feast PushSource
        -> Feast Online Store
        -> retrieval
        -> generated/retrieved value verification

This script intentionally stops after Feast verification.
It does not perform RF-300 inference or prediction storage.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_REPO = (
    PROJECT_ROOT
    / "feast_repo"
    / "fraud_features"
    / "feature_repo"
)

PROCESSOR_DIR = (
    PROJECT_ROOT
    / "feast_repo"
    / "fraud_features"
    / "scripts"
)

if str(PROCESSOR_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESSOR_DIR))


from may_online_processor import (  # noqa: E402
    load_april_history,
    calculate_features_for_transaction,
)


# ---------------------------------------------------------------------
# CONTROLLED TRANSACTION
# ---------------------------------------------------------------------

TRANSACTION = {
    "TRANSACTION_ID": 288062,
    "TX_DATETIME": pd.Timestamp("2018-05-01 00:01:21"),
    "CUSTOMER_ID": 3546,
    "TERMINAL_ID": 2944,
    "TX_AMOUNT": 18.71,
}


# ---------------------------------------------------------------------
# APPROVED 10 FEATURES
# ---------------------------------------------------------------------

APPROVED_FEATURES = [
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


# ---------------------------------------------------------------------
# BUILD FEATURES
# ---------------------------------------------------------------------

def build_feature_row():
    """
    Build the ten approved features for transaction 288062.

    Returns:
        tuple[pd.DataFrame, dict]:
            Feast push row and generated feature values.
    """

    transaction = TRANSACTION.copy()

    # Load only the established April historical context.
    history = load_april_history()

    # Reuse the existing leakage-safe behavioral feature function.
    behavioral_features = calculate_features_for_transaction(
        transaction,
        history,
    )

    current_time = transaction["TX_DATETIME"]

    # Existing project time/amount feature definitions.
    amount_log = float(np.log1p(transaction["TX_AMOUNT"]))
    tx_hour = int(current_time.hour)
    tx_day_of_week = int(current_time.dayofweek)
    tx_is_weekend = int(tx_day_of_week >= 5)

    generated = {
        "TRANSACTION_ID": int(transaction["TRANSACTION_ID"]),
        "TX_DATETIME": current_time,
        "TX_AMOUNT": float(transaction["TX_AMOUNT"]),
        "AMOUNT_LOG": amount_log,
        "TX_HOUR": tx_hour,
        "TX_DAY_OF_WEEK": tx_day_of_week,
        "TX_IS_WEEKEND": tx_is_weekend,
        "TERMINAL_TX_COUNT_1D": float(
            behavioral_features["TERMINAL_TX_COUNT_1D"]
        ),
        "TERMINAL_TX_COUNT_7D": float(
            behavioral_features["TERMINAL_TX_COUNT_7D"]
        ),
        "TERMINAL_AVG_AMOUNT_7D": float(
            behavioral_features["TERMINAL_AVG_AMOUNT_7D"]
        ),
        "CUSTOMER_TX_COUNT_1D": float(
            behavioral_features["CUSTOMER_TX_COUNT_1D"]
        ),
        "CUSTOMER_TX_COUNT_7D": float(
            behavioral_features["CUSTOMER_TX_COUNT_7D"]
        ),
    }

    push_df = pd.DataFrame([generated])

    return push_df, generated


# ---------------------------------------------------------------------
# PUSH TO FEAST
# ---------------------------------------------------------------------

def push_to_feast(push_df):
    """
    Push the generated feature row to the existing Feast PushSource.

    Args:
        push_df: DataFrame containing the controlled feature row.

    Returns:
        None.
    """

    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    store.push(
        push_source_name="fraud_push_source",
        df=push_df,
    )


# ---------------------------------------------------------------------
# RETRIEVE FROM FEAST
# ---------------------------------------------------------------------

def retrieve_from_feast():
    """
    Retrieve the controlled transaction's ten approved features.

    Returns:
        dict: Retrieved feature values.
    """

    store = FeatureStore(
        repo_path=str(FEATURE_REPO)
    )

    result = store.get_online_features(
        features=[
            "fraud_features:TX_AMOUNT",
            "fraud_features:AMOUNT_LOG",
            "fraud_features:TX_HOUR",
            "fraud_features:TX_DAY_OF_WEEK",
            "fraud_features:TX_IS_WEEKEND",
            "fraud_features:TERMINAL_TX_COUNT_1D",
            "fraud_features:TERMINAL_TX_COUNT_7D",
            "fraud_features:TERMINAL_AVG_AMOUNT_7D",
            "fraud_features:CUSTOMER_TX_COUNT_1D",
            "fraud_features:CUSTOMER_TX_COUNT_7D",
        ],
        entity_rows=[
            {
                "TRANSACTION_ID": int(
                    TRANSACTION["TRANSACTION_ID"]
                )
            }
        ],
    )

    row = result.to_dict()

    return {
        feature: row[feature][0]
        for feature in APPROVED_FEATURES
    }


# ---------------------------------------------------------------------
# VALUE COMPARISON
# ---------------------------------------------------------------------

def compare_values(generated, retrieved):
    """
    Compare generated feature values with Feast-retrieved values.

    Args:
        generated: Generated feature dictionary.
        retrieved: Feast-retrieved feature dictionary.

    Returns:
        None.

    Raises:
        AssertionError: If any approved feature does not match.
    """

    print()
    print("FEATURE VERIFICATION")
    print("-" * 70)

    for feature in APPROVED_FEATURES:
        expected = generated[feature]
        actual = retrieved[feature]

        if isinstance(expected, (float, np.floating)):
            matches = np.isclose(
                float(expected),
                float(actual),
                rtol=1e-6,
                atol=1e-6,
            )
        else:
            matches = expected == actual

        print(
            f"{feature}: "
            f"generated={expected!r}, "
            f"feast={actual!r}, "
            f"match={matches}"
        )

        if not matches:
            raise AssertionError(
                f"Feature mismatch for {feature}: "
                f"generated={expected!r}, "
                f"feast={actual!r}"
            )

    print("-" * 70)
    print("PASS: All 10 generated feature values match Feast.")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DAY 5 STEP 12 - FEATURE PIPELINE -> FEAST")
    print("=" * 70)

    print()
    print("Controlled transaction:")
    print(TRANSACTION)

    print()
    print("Loading April historical context...")
    push_df, generated = build_feature_row()

    print("April history loaded.")
    print()
    print("Generated feature row:")
    print(
        push_df[
            [
                "TRANSACTION_ID",
                "TX_DATETIME",
                *APPROVED_FEATURES,
            ]
        ].to_string(index=False)
    )

    print()
    print("Pushing generated features to Feast...")
    push_to_feast(push_df)
    print("PASS: Feast push completed.")

    print()
    print("Retrieving transaction 288062 from Feast...")
    retrieved = retrieve_from_feast()

    print()
    print("Retrieved Feast values:")
    for feature in APPROVED_FEATURES:
        print(f"{feature}: {retrieved[feature]!r}")

    compare_values(generated, retrieved)

    print()
    print("=" * 70)
    print("STEP 12 PASS")
    print("=" * 70)
    print("Feature generation -> Feast push -> Feast retrieval -> verification")
    print("completed successfully for transaction 288062.")
    print()
    print("STOP HERE.")
    print("No RF-300 inference or prediction storage is performed by Step 12.")


if __name__ == "__main__":
    main()

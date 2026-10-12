import sys
from pathlib import Path

import pandas as pd
import numpy as np
import skops.io as sio
from feast import FeatureStore


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Existing project functions
# ---------------------------------------------------------
from feast_repo.fraud_features.scripts.may_online_processor import (
    load_april_history,
    calculate_features_for_transaction,
)

from src.prediction_storage import (
    initialize_database,
    store_prediction,
    get_prediction,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
FEAST_REPO_PATH = Path("feast_repo") / "fraud_features" / "feature_repo"
MODEL_PATH = Path("artifacts") / "rf300_champion" / "model.skops"
THRESHOLD = 0.65


# ---------------------------------------------------------
# Controlled Step 13 transaction
# ---------------------------------------------------------
CONTROLLED_TRANSACTION = {
    "TRANSACTION_ID": 288062,
    "TX_DATETIME": pd.Timestamp("2018-05-01 00:01:21"),
    "CUSTOMER_ID": 3546,
    "TERMINAL_ID": 2944,
    "TX_AMOUNT": 18.71,
}


# ---------------------------------------------------------
# Features used by the model
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
# VALIDATE FEATURES
# ---------------------------------------------------------
def validate_transaction_features(features):
    """Validate the 10 approved features before pushing to Feast."""

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    feature_values = [
        features[feature]
        for feature in FEATURE_COLUMNS
    ]

    if not np.all(np.isfinite(feature_values)):
        raise ValueError(
            "Feature values contain NaN or infinite values."
        )

    return True


# ---------------------------------------------------------
# PUSH FEATURES TO FEAST
# ---------------------------------------------------------
def push_to_feast(transaction, features):
    """Push one transaction's approved features to Feast."""

    store = FeatureStore(
        repo_path=FEAST_REPO_PATH
    )

    push_data = {
        "TRANSACTION_ID": int(
            transaction["TRANSACTION_ID"]
        ),
        "TX_DATETIME": transaction["TX_DATETIME"],
        **features,
    }

    push_df = pd.DataFrame(
        [push_data]
    )

    store.push(
        push_source_name="fraud_push_source",
        df=push_df,
    )

    return push_df


# ---------------------------------------------------------
# RETRIEVE FEATURES FROM FEAST
# ---------------------------------------------------------
def retrieve_from_feast(transaction_id):
    """Retrieve the 10 approved features from Feast Online Store."""

    store = FeatureStore(
        repo_path=FEAST_REPO_PATH
    )

    result = store.get_online_features(
        features=[
            f"fraud_features:{feature}"
            for feature in FEATURE_COLUMNS
        ],
        entity_rows=[
            {
                "TRANSACTION_ID": int(
                    transaction_id
                )
            }
        ],
    ).to_df()

    return result


# ---------------------------------------------------------
# LOAD CHAMPION MODEL
# ---------------------------------------------------------
def load_champion_model():
    """Load the already-verified frozen RF-300 Champion artifact."""

    model = sio.load(
        MODEL_PATH,
        trusted=sio.get_untrusted_types(
            file=MODEL_PATH
        ),
    )

    return model


# ---------------------------------------------------------
# RUN INFERENCE
# ---------------------------------------------------------
def predict_transaction(result, model):
    """Run RF-300 inference using the 10 Feast-retrieved features."""

    X = result[FEATURE_COLUMNS]

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    return probability, prediction


# ---------------------------------------------------------
# PROCESS ONE CONTROLLED TRANSACTION
# ---------------------------------------------------------
def process_new_transaction(
    transaction,
    history,
    model,
):
    """
    Process one controlled transaction through the Step 13 flow.

    Flow:
    1. Calculate behavioral features using the existing Step 12 function.
    2. Build transaction/time features from the current transaction.
    3. Keep exactly the 10 approved model features.
    4. Validate the features.
    5. Push features to Feast.
    6. Retrieve features from Feast Online Store.
    7. Run frozen RF-300 inference.
    8. Store prediction.
    9. Read prediction back.
    10. Verify exact prediction match.
    """

    # -----------------------------------------------------
    # Calculate leakage-safe behavioral features
    # using existing Step 12 logic
    # -----------------------------------------------------
    calculated_features = calculate_features_for_transaction(
        transaction,
        history,
    )

    # -----------------------------------------------------
    # Build exactly the 10 approved model features
    #
    # Transaction/time features come directly from the
    # controlled transaction.
    #
    # Behavioral features come from the existing
    # Step 12 calculation function.
    # -----------------------------------------------------
    features = {
        "TX_AMOUNT": float(
            transaction["TX_AMOUNT"]
        ),
        "AMOUNT_LOG": float(
            np.log1p(transaction["TX_AMOUNT"])
        ),
        "TX_HOUR": int(
            transaction["TX_DATETIME"].hour
        ),
        "TX_DAY_OF_WEEK": int(
            transaction["TX_DATETIME"].dayofweek
        ),
        "TX_IS_WEEKEND": int(
            transaction["TX_DATETIME"].dayofweek >= 5
        ),
        "TERMINAL_TX_COUNT_1D": calculated_features[
            "TERMINAL_TX_COUNT_1D"
        ],
        "TERMINAL_TX_COUNT_7D": calculated_features[
            "TERMINAL_TX_COUNT_7D"
        ],
        "TERMINAL_AVG_AMOUNT_7D": calculated_features[
            "TERMINAL_AVG_AMOUNT_7D"
        ],
        "CUSTOMER_TX_COUNT_1D": calculated_features[
            "CUSTOMER_TX_COUNT_1D"
        ],
        "CUSTOMER_TX_COUNT_7D": calculated_features[
            "CUSTOMER_TX_COUNT_7D"
        ],
    }

    # -----------------------------------------------------
    # Validate features
    # -----------------------------------------------------
    validate_transaction_features(
        features
    )

    # -----------------------------------------------------
    # Push features to Feast
    # -----------------------------------------------------
    push_to_feast(
        transaction,
        features,
    )

    # -----------------------------------------------------
    # Retrieve features from Feast
    # -----------------------------------------------------
    result = retrieve_from_feast(
        transaction["TRANSACTION_ID"]
    )

    # -----------------------------------------------------
    # Run frozen RF-300 inference
    # -----------------------------------------------------
    probability, prediction = predict_transaction(
        result,
        model,
    )

    # -----------------------------------------------------
    # Store prediction
    # -----------------------------------------------------
    store_prediction(
        transaction["TRANSACTION_ID"],
        prediction,
    )

    # -----------------------------------------------------
    # Read prediction back
    # -----------------------------------------------------
    stored_prediction = get_prediction(
        transaction["TRANSACTION_ID"]
    )

    # -----------------------------------------------------
    # Exact-match verification
    # -----------------------------------------------------
    if stored_prediction is None:
        raise AssertionError(
            "Prediction storage verification failed: "
            "no prediction was returned."
        )

    if int(stored_prediction) != int(prediction):
        raise AssertionError(
            "Prediction storage verification failed: "
            f"inference={prediction}, "
            f"stored={stored_prediction}"
        )

    return (
        result,
        probability,
        prediction,
        stored_prediction,
    )


# ---------------------------------------------------------
# MAIN STEP 13 AUTOMATION
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("Step 13 - Online Prediction Automation")
    print("=" * 60)

    # -----------------------------------------------------
    # Initialize prediction storage
    # -----------------------------------------------------
    initialize_database()

    print("\nPrediction storage initialization: PASS")

    # -----------------------------------------------------
    # Load April history
    # -----------------------------------------------------
    history = load_april_history()

    print(
        f"April historical transactions available: "
        f"{len(history)}"
    )

    # -----------------------------------------------------
    # Controlled transaction
    # -----------------------------------------------------
    transaction = CONTROLLED_TRANSACTION.copy()

    print("\nControlled transaction:")
    print(
        pd.Series(transaction).to_string()
    )

    # -----------------------------------------------------
    # Verify no future/same-time history is used
    # -----------------------------------------------------
    history_before_transaction = history[
        history["TX_DATETIME"]
        < transaction["TX_DATETIME"]
    ]

    if len(history_before_transaction) != len(history):
        raise AssertionError(
            "History contains transactions at or after "
            "the controlled transaction timestamp."
        )

    print(
        "\nLeakage check: PASS"
    )

    # -----------------------------------------------------
    # Load frozen Champion
    # -----------------------------------------------------
    model = load_champion_model()

    print("Champion model loading: PASS")

    # -----------------------------------------------------
    # Process controlled transaction
    # -----------------------------------------------------
    (
        result,
        probability,
        prediction,
        stored_prediction,
    ) = process_new_transaction(
        transaction,
        history,
        model,
    )

    # -----------------------------------------------------
    # Verification output
    # -----------------------------------------------------
    print("\nFeature validation: PASS")
    print("Feast push: PASS")

    print("\nRetrieved features:")
    print(
        result.to_string(index=False)
    )

    print("\nOnline retrieval: PASS")

    print(
        f"\nChampion probability: "
        f"{probability:.6f}"
    )

    print(
        f"Champion threshold: "
        f"{THRESHOLD}"
    )

    print(
        f"Champion prediction: "
        f"{prediction}"
    )

    print(
        f"Stored prediction: "
        f"{stored_prediction}"
    )

    print(
        "\nPrediction storage write/read: PASS"
    )

    print(
        "Exact prediction match: PASS"
    )

    print("=" * 60)
    print(
        "Step 13 controlled inference completed successfully."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
import os
import argparse
import pandas as pd
import skops.io as sio

from feast import FeatureStore

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAY_PATH = r"data\phase3\may_processed_features.csv"

FEAST_REPO_PATH = r"feast_repo\fraud_features\feature_repo"

MODEL_PATH = r"artifacts\rf300_champion\model.skops"

# Final May evaluation evidence is stored separately
# from the previous smoke-test evidence.
OUTPUT_DIR = r"reports\phase4\may_champion_evaluation_final"

PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "may_champion_predictions.csv",
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "may_champion_metrics.txt",
)

THRESHOLD = 0.65


# ============================================================
# APPROVED MODEL FEATURES
# ============================================================

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

TARGET = "TX_FRAUD"


# ============================================================
# LOAD MAY IDS AND LABELS ONLY
# ============================================================

def load_may_data():
    df = pd.read_csv(
        MAY_PATH,
        usecols=[
            "TRANSACTION_ID",
            TARGET,
        ],
    )

    if df["TRANSACTION_ID"].duplicated().any():
        raise ValueError(
            "Duplicate TRANSACTION_ID values found in May data."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Missing TX_FRAUD values found in May data."
        )

    return df


# ============================================================
# RETRIEVE FEATURES FROM EXISTING FEAST ONLINE STORE
# ============================================================

def retrieve_may_features(transaction_ids):

    store = FeatureStore(
        repo_path=FEAST_REPO_PATH
    )

    entity_rows = [
        {
            "TRANSACTION_ID": int(transaction_id)
        }
        for transaction_id in transaction_ids
    ]

    result = store.get_online_features(
        features=[
            f"fraud_features:{feature}"
            for feature in FEATURE_COLUMNS
        ],
        entity_rows=entity_rows,
    ).to_df()

    return result


# ============================================================
# LOAD FROZEN CHAMPION
# ============================================================

def load_champion_model():

    model = sio.load(
        MODEL_PATH,
        trusted=sio.get_untrusted_types(
            file=MODEL_PATH
        ),
    )

    # Frozen champion safeguards
    if model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError(
            "Champion model feature count does not match "
            "the approved 10-feature schema."
        )

    params = model.get_params()

    expected_params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    for parameter, expected_value in expected_params.items():

        actual_value = params.get(parameter)

        if actual_value != expected_value:
            raise ValueError(
                f"Champion parameter mismatch: "
                f"{parameter}={actual_value}, "
                f"expected {expected_value}"
            )

    return model


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn)

    return {
        "threshold": threshold,

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "fpr": fpr,

        "true_positives": tp,

        "false_positives": fp,

        "true_negatives": tn,

        "false_negatives": fn,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Optional smoke-test limit
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to the first N May transactions.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MAY - FROZEN CHAMPION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load only May transaction IDs and labels
    # --------------------------------------------------------

    may_df = load_may_data()

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than 0."
            )

        may_df = may_df.head(
            args.limit
        ).copy()

        print(
            f"\nSMOKE TEST MODE: first "
            f"{args.limit} May transactions"
        )

    print(
        "\nMay dataset shape:",
        may_df.shape,
    )

    print(
        "May transaction count:",
        len(may_df),
    )

    print(
        "May fraud count:",
        int(may_df[TARGET].sum()),
    )

    # --------------------------------------------------------
    # Retrieve approved features from existing Feast store
    # --------------------------------------------------------

    transaction_ids = may_df[
        "TRANSACTION_ID"
    ].tolist()

    feast_df = retrieve_may_features(
        transaction_ids
    )

    print(
        "\nFeast rows retrieved:",
        len(feast_df),
    )

    if len(feast_df) != len(may_df):
        raise ValueError(
            "Feast returned a different number of "
            "transactions than the May dataset."
        )

    # --------------------------------------------------------
    # Validate Feast feature columns
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in feast_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing Feast features: {missing_features}"
        )

    # --------------------------------------------------------
    # Validate Feast transaction IDs
    # --------------------------------------------------------

    if feast_df["TRANSACTION_ID"].duplicated().any():
        raise ValueError(
            "Duplicate TRANSACTION_ID values returned by Feast."
        )

    may_ids = set(
        may_df["TRANSACTION_ID"]
    )

    feast_ids = set(
        feast_df["TRANSACTION_ID"]
    )

    missing_ids = may_ids - feast_ids
    extra_ids = feast_ids - may_ids

    if missing_ids:
        raise ValueError(
            f"May transaction IDs missing from Feast: "
            f"{len(missing_ids)}"
        )

    if extra_ids:
        raise ValueError(
            f"Unexpected transaction IDs returned by Feast: "
            f"{len(extra_ids)}"
        )

    # --------------------------------------------------------
    # Align labels with Feast features
    # --------------------------------------------------------

    evaluation_df = may_df.merge(
        feast_df[
            ["TRANSACTION_ID"] + FEATURE_COLUMNS
        ],
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )

    if len(evaluation_df) != len(may_df):
        raise ValueError(
            "Transaction alignment failed."
        )

    # --------------------------------------------------------
    # Check for missing Feast feature values
    # --------------------------------------------------------

    missing_value_count = (
        evaluation_df[FEATURE_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Missing Feast feature values:",
        int(missing_value_count),
    )

    if missing_value_count != 0:
        raise ValueError(
            "Missing Feast feature values detected."
        )

    # --------------------------------------------------------
    # Load frozen champion
    # --------------------------------------------------------

    model = load_champion_model()

    print(
        "\nChampion model:",
        type(model).__name__,
    )

    print(
        "Champion n_features_in_:",
        model.n_features_in_,
    )

    print(
        "Champion threshold:",
        THRESHOLD,
    )

    # --------------------------------------------------------
    # Prepare EXACT approved feature order
    # --------------------------------------------------------

    X = evaluation_df[
        FEATURE_COLUMNS
    ]

    y_true = evaluation_df[
        TARGET
    ].astype(int)

    # --------------------------------------------------------
    # Generate fraud probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X
    )[:, 1]

    # --------------------------------------------------------
    # Apply frozen threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Calculate May metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        probabilities,
        THRESHOLD,
    )

    # --------------------------------------------------------
    # Create prediction evidence
    # --------------------------------------------------------

    evidence_df = evaluation_df[
        ["TRANSACTION_ID", TARGET]
    ].copy()

    evidence_df["fraud_probability"] = probabilities

    evidence_df["prediction"] = predictions

    # --------------------------------------------------------
    # Save separate FINAL evidence
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    evidence_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save FINAL metrics evidence
    # --------------------------------------------------------

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "MAY FROZEN CHAMPION EVALUATION\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"May transactions: {len(may_df)}\n"
        )

        file.write(
            f"May fraud transactions: "
            f"{int(y_true.sum())}\n"
        )

        file.write(
            f"Model: {type(model).__name__}\n"
        )

        file.write(
            f"Threshold: {THRESHOLD}\n"
        )

        file.write(
            "Feature order:\n"
        )

        for feature in FEATURE_COLUMNS:
            file.write(
                f"  - {feature}\n"
            )

        file.write(
            "\nMetrics\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for name, value in metrics.items():
            file.write(
                f"{name}: {value}\n"
            )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MAY CHAMPION EVALUATION RESULTS")
    print("=" * 70)

    for name, value in metrics.items():
        print(
            f"{name}: {value}"
        )

    print(
        "\nPrediction evidence:",
        PREDICTIONS_PATH,
    )

    print(
        "Metrics evidence:",
        METRICS_PATH,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
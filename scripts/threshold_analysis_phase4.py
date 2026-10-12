"""
Run Phase 4 threshold analysis using a controlled temporal split.
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.train_phase4 import (
    MODEL_FEATURES,
    TARGET,
    calculate_metrics,
    prepare_features,
    train_model,
    predict_probabilities,
)


TUNING_DATASET = "data/phase3/train_feast_features.csv"


CANDIDATES = [
    {
        "name": "LR_Original_C_0.01",
        "model_type": "logistic_regression",
        "imbalance_strategy": "original",
        "C": 0.01,
    },
    {
        "name": "RF_Balanced_100_None_1",
        "model_type": "random_forest",
        "imbalance_strategy": "balanced",
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    },
    {
        "name": "LR_Balanced_C_10",
        "model_type": "logistic_regression",
        "imbalance_strategy": "balanced",
        "C": 10,
    },
]


def create_threshold_split(df: pd.DataFrame):
    """Create the temporal split used for threshold selection.

    Args:
        df: Feast training feature dataset containing event timestamps.

    Returns:
        Tuple containing fitting data from Apr 1–12 and
        threshold-selection data from Apr 13–18.

    Raises:
        ValueError: If the required event_timestamp column is missing.
    """
    if "event_timestamp" not in df.columns:
        raise ValueError("Required column 'event_timestamp' is missing.")

    data = df.copy()

    data["event_timestamp"] = pd.to_datetime(
        data["event_timestamp"],
        utc=True,
    )

    fitting_data = data[
        data["event_timestamp"].dt.day <= 12
    ].copy()

    threshold_data = data[
        (data["event_timestamp"].dt.day >= 13)
        & (data["event_timestamp"].dt.day <= 18)
    ].copy()

    return fitting_data, threshold_data


def prepare_threshold_data(fitting_data, threshold_data):
    """Prepare model features and targets for threshold analysis.

    Args:
        fitting_data: Data from Apr 1–12 used for model fitting.
        threshold_data: Data from Apr 13–18 used only for threshold selection.

    Returns:
        Tuple containing prepared fitting features, fitting target,
        threshold features, and threshold target.
    """
    X_fit, y_fit, X_threshold, y_threshold = prepare_features(
        fitting_data,
        threshold_data,
    )

    return X_fit, y_fit, X_threshold, y_threshold


def train_threshold_candidate(
    candidate,
    X_fit,
    y_fit,
    X_threshold,
):
    """Train one approved candidate and generate threshold-set probabilities.

    Args:
        candidate: Candidate model configuration.
        X_fit: Model features from Apr 1–12.
        y_fit: Target values from Apr 1–12.
        X_threshold: Model features from Apr 13–18.

    Returns:
        Tuple containing the trained model and predicted fraud probabilities
        for the Apr 13–18 threshold-selection data.
    """
    model = train_model(
        X_fit,
        y_fit,
        candidate["model_type"],
        candidate["imbalance_strategy"],
        {
            key: value
            for key, value in candidate.items()
            if key not in {"name", "model_type", "imbalance_strategy"}
        },
    )

    threshold_probabilities = predict_probabilities(
        model,
        X_threshold,
    )

    return model, threshold_probabilities


def evaluate_thresholds(
    y_threshold,
    threshold_probabilities,
):
    """Evaluate multiple probability thresholds using F1.

    Args:
        y_threshold: Fraud labels from Apr 13–18.
        threshold_probabilities: Predicted fraud probabilities for
            Apr 13–18.

    Returns:
        A DataFrame containing threshold-dependent metrics.
    """
    thresholds = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    results = []

    for threshold in thresholds:
        metrics = calculate_metrics(
            y_threshold,
            threshold_probabilities,
            threshold=threshold,
        )

        results.append(
            {
                "threshold": threshold,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fpr": metrics["fpr"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    data = pd.read_csv(TUNING_DATASET)

    fitting_data, threshold_data = create_threshold_split(data)

    X_fit, y_fit, X_threshold, y_threshold = prepare_threshold_data(
        fitting_data,
        threshold_data,
    )

    print("\nPREPARED FEATURE VERIFICATION")
    print("X_fit shape:", X_fit.shape)
    print("y_fit shape:", y_fit.shape)
    print("X_threshold shape:", X_threshold.shape)
    print("y_threshold shape:", y_threshold.shape)

    print("\nModel features:")
    print(list(X_fit.columns))

    print("\nTarget values in fitting data:")
    print(sorted(y_fit.unique()))

    print("Target values in threshold data:")
    print(sorted(y_threshold.unique()))

    print("=" * 60)
    print("THRESHOLD SPLIT VERIFICATION")
    print("=" * 60)

    print("\nApr 1–12 FITTING DATA")
    print("Rows:", len(fitting_data))
    print("Min timestamp:", fitting_data["event_timestamp"].min())
    print("Max timestamp:", fitting_data["event_timestamp"].max())
    print("Fraud count:", fitting_data["TX_FRAUD"].sum())
    print("Unique dates:")
    print(sorted(fitting_data["event_timestamp"].dt.day.unique()))

    print("\nApr 13–18 THRESHOLD DATA")
    print("Rows:", len(threshold_data))
    print("Min timestamp:", threshold_data["event_timestamp"].min())
    print("Max timestamp:", threshold_data["event_timestamp"].max())
    print("Fraud count:", threshold_data["TX_FRAUD"].sum())
    print("Unique dates:")
    print(sorted(threshold_data["event_timestamp"].dt.day.unique()))

    fitting_ids = set(fitting_data["TRANSACTION_ID"])
    threshold_ids = set(threshold_data["TRANSACTION_ID"])

    overlap = fitting_ids.intersection(threshold_ids)

    print("\nOVERLAP CHECK")
    print("Overlapping transaction IDs:", len(overlap))

    print("\nFIXED CANDIDATES")

    for candidate in CANDIDATES:
        print(candidate)

    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS FOR APPROVED CANDIDATES")
    print("=" * 60)

    all_candidate_results = []

    for candidate in CANDIDATES:
        print("\n" + "-" * 60)
        print("Candidate:", candidate["name"])
        print("-" * 60)

        model, threshold_probabilities = train_threshold_candidate(
            candidate,
            X_fit,
            y_fit,
            X_threshold,
        )

        print("Model type:", type(model).__name__)
        print("Number of fitting rows:", len(X_fit))
        print("Number of threshold rows:", len(X_threshold))
        print("Probability count:", len(threshold_probabilities))
        print(
            "Probability range:",
            float(threshold_probabilities.min()),
            "to",
            float(threshold_probabilities.max()),
        )

        threshold_results = evaluate_thresholds(
            y_threshold,
            threshold_probabilities,
        )

        print("\nTHRESHOLD EVALUATION")
        print(threshold_results.to_string(index=False))

        best_threshold_row = threshold_results.loc[
            threshold_results["f1"].idxmax()
        ]

        print("\nBEST THRESHOLD BY F1")
        print(best_threshold_row.to_string())

        all_candidate_results.append(
            {
                "candidate": candidate["name"],
                "threshold": best_threshold_row["threshold"],
                "f1": best_threshold_row["f1"],
                "precision": best_threshold_row["precision"],
                "recall": best_threshold_row["recall"],
                "fpr": best_threshold_row["fpr"],
            }
        )

    print("\n" + "=" * 60)
    print("BEST THRESHOLD PER APPROVED CANDIDATE")
    print("=" * 60)

    print(
        pd.DataFrame(all_candidate_results).to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETED")
    print("=" * 60)

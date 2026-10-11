"""
Evaluate the Step 17 RF300 candidate on the fixed Apr 25–30 validation set.
This step evaluates only; it does not retrain or promote the Champion.
"""

import sys
from pathlib import Path

import pandas as pd
import skops.io as sio
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_phase4 import MODEL_FEATURES


# ---------------------------------------------------------------------
# Fixed Step 18 inputs
# ---------------------------------------------------------------------

EVALUATION_DATASET = (
    PROJECT_ROOT
    / "data"
    / "phase4"
    / "april_25_30_evaluation_features.csv"
)

CANDIDATE_ARTIFACT = (
    PROJECT_ROOT
    / "artifacts"
    / "candidates"
    / "rf300_candidate"
    / "model.skops"
)

CHAMPION_ARTIFACT = (
    PROJECT_ROOT
    / "artifacts"
    / "rf300_champion"
    / "model.skops"
)

EVIDENCE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase4"
    / "step18_candidate_evaluation_rerun_20261010"
)

METRICS_FILE = EVIDENCE_DIR / "candidate_evaluation_metrics.txt"
PREDICTIONS_FILE = EVIDENCE_DIR / "candidate_predictions.csv"


# ---------------------------------------------------------------------
# Fixed project contracts
# ---------------------------------------------------------------------

TARGET = "TX_FRAUD"
THRESHOLD = 0.65

EXPECTED_ROWS = 57767
EXPECTED_FRAUD_COUNT = 526

CHAMPION_IDENTITY = "RF_Balanced_300_None_1"
CANDIDATE_IDENTITY = "RF300_STEP17_CANDIDATE"

# Step 17 MLflow run ID recorded for reference; direct artifact provenance is unverified.
STEP17_MLFLOW_RUN_ID = "ca4b5285656c4906a01502115774347d"

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

EXPECTED_RF_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------
# Historical RF300 benchmark
# ---------------------------------------------------------------------

HISTORICAL_CHAMPION_BENCHMARK = {
    "threshold": 0.65,
    "pr_auc": 0.2574410304,
    "roc_auc": 0.6370331816,
    "precision": 1.0,
    "recall": 0.2338403042,
    "f1": 0.3790446841,
    "fpr": 0.0,
    "true_positives": 123,
    "false_positives": 0,
    "true_negatives": 57241,
    "false_negatives": 403,
}


def validate_contract():
    """Validate the fixed Step 18 evaluation contract."""

    if EVALUATION_DATASET.name != "april_25_30_evaluation_features.csv":
        raise ValueError("Unexpected Step 18 evaluation dataset.")

    if CANDIDATE_ARTIFACT == CHAMPION_ARTIFACT:
        raise ValueError("Candidate artifact must differ from Champion artifact.")

    if not CANDIDATE_ARTIFACT.is_relative_to(
        PROJECT_ROOT / "artifacts" / "candidates"
    ):
        raise ValueError("Candidate artifact is outside the candidate directory.")

    if not CHAMPION_ARTIFACT.is_relative_to(
        PROJECT_ROOT / "artifacts" / "rf300_champion"
    ):
        raise ValueError("Champion artifact is outside the Champion directory.")

    if CANDIDATE_IDENTITY == CHAMPION_IDENTITY:
        raise ValueError("Candidate and Champion identities must differ.")

    if EXPECTED_FEATURES != MODEL_FEATURES:
        raise ValueError("Step 18 features do not match MODEL_FEATURES.")

    if THRESHOLD != 0.65:
        raise ValueError("Step 18 threshold must remain 0.65.")

    if not STEP17_MLFLOW_RUN_ID:
        raise ValueError("Step 17 MLflow run ID is missing.")


def load_candidate_model():
    """Load and validate the Step 17 candidate model."""

    if not CANDIDATE_ARTIFACT.exists():
        raise FileNotFoundError(
            f"Candidate artifact not found: {CANDIDATE_ARTIFACT}"
        )

    if not CHAMPION_ARTIFACT.exists():
        raise FileNotFoundError(
            f"Champion artifact not found: {CHAMPION_ARTIFACT}"
        )

    model = sio.load(
        CANDIDATE_ARTIFACT,
        trusted=sio.get_untrusted_types(file=CANDIDATE_ARTIFACT),
    )

    if getattr(model, "n_features_in_", None) != len(EXPECTED_FEATURES):
        raise ValueError(
            "Candidate model does not contain the expected number of features."
        )

    for parameter, expected_value in EXPECTED_RF_PARAMETERS.items():
        actual_value = model.get_params().get(parameter)

        if actual_value != expected_value:
            raise ValueError(
                f"Candidate parameter mismatch: "
                f"{parameter}={actual_value!r}, "
                f"expected {expected_value!r}"
            )

    return model


def calculate_metrics(labels, probabilities):
    """Calculate fixed-threshold and ranking metrics."""

    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "threshold": THRESHOLD,
        "pr_auc": average_precision_score(labels, probabilities),
        "roc_auc": roc_auc_score(labels, probabilities),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    return metrics, predictions


def main():
    # Never overwrite previously generated rerun evidence.
    existing_outputs = [path for path in (METRICS_FILE, PREDICTIONS_FILE) if path.exists()]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Refusing to overwrite existing Step 18 evidence file(s): {paths}")

    validate_contract()

    if not EVALUATION_DATASET.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_DATASET}"
        )

    evaluation_df = pd.read_csv(EVALUATION_DATASET)

    print("Step 18 candidate evaluation")
    print(f"Evaluation dataset: {EVALUATION_DATASET}")
    print(f"Evaluation shape: {evaluation_df.shape}")
    print(f"Expected rows: {EXPECTED_ROWS}")

    if len(evaluation_df) != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected evaluation row count: {len(evaluation_df)}"
        )

    if TARGET not in evaluation_df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    actual_fraud_count = int(evaluation_df[TARGET].sum())

    print(f"Fraud count: {actual_fraud_count}")
    print(f"Expected fraud count: {EXPECTED_FRAUD_COUNT}")

    if actual_fraud_count != EXPECTED_FRAUD_COUNT:
        raise ValueError(
            f"Unexpected fraud count: {actual_fraud_count}"
        )

    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in evaluation_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing expected features: {missing_features}"
        )

    if evaluation_df["TRANSACTION_ID"].duplicated().any():
        raise ValueError("Duplicate TRANSACTION_ID values found.")

    candidate_model = load_candidate_model()

    X = evaluation_df[EXPECTED_FEATURES]
    y = evaluation_df[TARGET].astype(int)

    probabilities = candidate_model.predict_proba(X)[:, 1]

    candidate_metrics, predictions = calculate_metrics(
        y,
        probabilities,
    )

    print()
    print("Candidate metrics")
    print("-----------------")

    for key, value in candidate_metrics.items():
        print(f"{key}: {value}")

    print()
    print("Historical RF300 Champion benchmark")
    print("------------------------------------")

    for key, value in HISTORICAL_CHAMPION_BENCHMARK.items():
        print(f"{key}: {value}")

    print()
    print("Metric differences")
    print("------------------")

    metric_keys = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "fpr",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ]

    differences = {}

    for key in metric_keys:
        candidate_value = candidate_metrics[key]
        benchmark_value = HISTORICAL_CHAMPION_BENCHMARK[key]

        difference = candidate_value - benchmark_value
        differences[key] = difference

        print(f"{key}: {difference}")

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    evidence_lines = [
        "STEP 18 CANDIDATE EVALUATION",
        "============================",
        "",
        f"Candidate identity: {CANDIDATE_IDENTITY}",
        f"Step 17 MLflow run ID (recorded reference; direct candidate-artifact provenance unverified): {STEP17_MLFLOW_RUN_ID}",
        f"Candidate artifact: {CANDIDATE_ARTIFACT}",
        f"Champion artifact protected: {CHAMPION_ARTIFACT}",
        f"Evaluation dataset: {EVALUATION_DATASET}",
        f"Evaluation rows: {len(evaluation_df)}",
        f"Fraud count: {actual_fraud_count}",
        f"Threshold: {THRESHOLD}",
        "",
        "Candidate metrics:",
    ]

    for key, value in candidate_metrics.items():
        evidence_lines.append(f"{key}: {value}")

    evidence_lines.extend(
        [
            "",
            "Historical RF300 Champion benchmark:",
        ]
    )

    for key, value in HISTORICAL_CHAMPION_BENCHMARK.items():
        evidence_lines.append(f"{key}: {value}")

    evidence_lines.extend(
        [
            "",
            "Metric differences:",
        ]
    )

    for key, value in differences.items():
        evidence_lines.append(f"{key}: {value}")

    evidence_lines.extend(
        [
            "",
            "Candidate only: no promotion performed.",
            "Champion artifact was not modified.",
            "No retraining performed during Step 18.",
        ]
    )

    METRICS_FILE.write_text(
        "\n".join(evidence_lines),
        encoding="utf-8",
    )

    prediction_output = evaluation_df[
        ["TRANSACTION_ID", TARGET]
    ].copy()

    prediction_output["candidate_probability"] = probabilities
    prediction_output["candidate_prediction"] = predictions

    prediction_output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print()
    print("Evidence written:")
    print(f"Metrics: {METRICS_FILE}")
    print(f"Predictions: {PREDICTIONS_FILE}")
    print()
    print(f"Step 17 MLflow run ID (recorded reference; direct candidate-artifact provenance unverified): {STEP17_MLFLOW_RUN_ID}")
    print("Candidate only: no promotion performed.")
    print("Champion artifact protected.")


if __name__ == "__main__":
    main()

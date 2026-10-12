"""
Controlled Phase 4 XGBoost experiment.

Frozen methodology:
1. Stage 1: train on Apr 1-12.
2. Select threshold on Apr 13-18 using Stage-1 predictions and max F1.
3. Freeze that threshold.
4. Stage 2: retrain on Apr 1-24 with a newly calculated scale_pos_weight.
5. Evaluate Stage-2 model on Apr 25-30 using the frozen Stage-1 threshold.
6. May is not used.

Use --validate-only before the actual experiment.
Validation mode must not train a model or create an MLflow run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

TRAIN_PATH = Path("data/phase3/train_feast_features.csv")

EVALUATION_PATH = Path(
    "data/phase4/april_25_30_evaluation_features.csv"
)

OUTPUT_DIR = Path(
    "reports/phase4/controlled_xgboost"
)

MLFLOW_EXPERIMENT = (
    "financial_fraud_detection_phase4_controlled_xgboost"
)


# ---------------------------------------------------------------------
# Frozen model configuration
# ---------------------------------------------------------------------

MODEL_FEATURES = [
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

TARGET_COLUMN = "TX_FRAUD"
ID_COLUMN = "TRANSACTION_ID"

N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.05
OBJECTIVE = "binary:logistic"
RANDOM_STATE = 42
N_JOBS = -1
EVAL_METRIC = "logloss"


# ---------------------------------------------------------------------
# Temporal protocol
# ---------------------------------------------------------------------

STAGE1_START_DAY = 1
STAGE1_END_DAY = 12

THRESHOLD_START_DAY = 13
THRESHOLD_END_DAY = 18

STAGE2_START_DAY = 1
STAGE2_END_DAY = 24


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_training_data():
    """Load the verified Feast training dataset."""

    df = pd.read_csv(TRAIN_PATH)

    required_columns = [
        ID_COLUMN,
        "event_timestamp",
        *MODEL_FEATURES,
        TARGET_COLUMN,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Training dataset is missing columns: {missing}"
        )

    df["event_timestamp"] = pd.to_datetime(
        df["event_timestamp"],
        utc=True,
    )

    return df


def load_evaluation_data():
    """Load the established Apr 25-30 candidate evaluation dataset."""

    df = pd.read_csv(EVALUATION_PATH)

    required_columns = [
        ID_COLUMN,
        "TX_DATETIME",
        *MODEL_FEATURES,
        TARGET_COLUMN,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Evaluation dataset is missing columns: {missing}"
        )

    df["TX_DATETIME"] = pd.to_datetime(
        df["TX_DATETIME"],
        utc=True,
    )

    return df


# ---------------------------------------------------------------------
# Temporal splits
# ---------------------------------------------------------------------

def split_training_data(df):
    """Create Stage-1, threshold-selection, and Stage-2 datasets."""

    stage1 = df[
        (df["event_timestamp"].dt.day >= STAGE1_START_DAY)
        & (df["event_timestamp"].dt.day <= STAGE1_END_DAY)
    ].copy()

    threshold = df[
        (df["event_timestamp"].dt.day >= THRESHOLD_START_DAY)
        & (df["event_timestamp"].dt.day <= THRESHOLD_END_DAY)
    ].copy()

    stage2 = df[
        (df["event_timestamp"].dt.day >= STAGE2_START_DAY)
        & (df["event_timestamp"].dt.day <= STAGE2_END_DAY)
    ].copy()

    return stage1, threshold, stage2


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def class_counts(df):
    """Return row, legitimate, and fraud counts."""

    legitimate = int(
        (df[TARGET_COLUMN] == 0).sum()
    )

    fraud = int(
        (df[TARGET_COLUMN] == 1).sum()
    )

    return {
        "rows": int(len(df)),
        "legitimate": legitimate,
        "fraud": fraud,
    }


def calculate_scale_pos_weight(y):
    """Calculate legitimate/fraud ratio from supplied training labels."""

    negative_count = int(
        (y == 0).sum()
    )

    positive_count = int(
        (y == 1).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "Cannot calculate scale_pos_weight because "
            "the training data contains no fraud examples."
        )

    return negative_count / positive_count


def verify_unique_ids(stage1, threshold):
    """Verify Stage-1 and threshold-selection IDs do not overlap."""

    overlap = set(
        stage1[ID_COLUMN]
    ).intersection(
        set(threshold[ID_COLUMN])
    )

    if overlap:
        raise ValueError(
            "Stage-1 and threshold-selection data overlap "
            f"on {len(overlap)} transaction IDs."
        )

    return len(overlap)


def verify_stage2_evaluation_ids(stage2, evaluation_df):
    """
    Verify that Stage-2 training IDs do not overlap with
    Apr 25-30 evaluation IDs.
    """

    overlap = set(
        stage2[ID_COLUMN]
    ).intersection(
        set(evaluation_df[ID_COLUMN])
    )

    if overlap:
        raise ValueError(
            "Stage-2 training and Apr 25-30 evaluation data "
            f"overlap on {len(overlap)} transaction IDs."
        )

    return len(overlap)


# ---------------------------------------------------------------------
# DVC provenance
# ---------------------------------------------------------------------

def get_dvc_status():
    """
    Execute `dvc status` and return the actual command result.

    This is recorded as provenance. It is not hard-coded.
    """

    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
            check=False,
        )

        return {
            "return_code": int(result.returncode),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

    except Exception as exc:
        return {
            "return_code": None,
            "stdout": "",
            "stderr": str(exc),
        }


def get_dvc_pointer_status(path):
    """
    Check whether the exact dataset has a corresponding .dvc pointer.

    This does not claim DVC tracking when a pointer is absent.
    """

    dvc_pointer = Path(
        f"{path}.dvc"
    )

    return {
        "dataset": str(path),
        "pointer": str(dvc_pointer),
        "exists": dvc_pointer.exists(),
    }


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def create_model(scale_pos_weight):
    """Create the frozen XGBoost pipeline."""

    xgb_model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        objective=OBJECTIVE,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        eval_metric=EVAL_METRIC,
        scale_pos_weight=scale_pos_weight,
    )

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", xgb_model),
        ]
    )


def prepare_xy(df):
    """Prepare the approved feature matrix and target."""

    X = df[MODEL_FEATURES].copy()

    y = df[TARGET_COLUMN].astype(int).copy()

    return X, y


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    """Calculate the same Phase 4 metric definitions."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------

def select_threshold(
    model,
    threshold_df,
):
    """
    Select threshold using only Stage-1 model predictions
    on Apr 13-18.
    """

    X_threshold, y_threshold = prepare_xy(
        threshold_df
    )

    probabilities = (
        model.predict_proba(
            X_threshold
        )[:, 1]
    )

    threshold_values = np.round(
        np.arange(
            0.01,
            0.96,
            0.01,
        ),
        2,
    )

    results = []

    for threshold in threshold_values:

        metrics = calculate_metrics(
            y_threshold,
            probabilities,
            threshold,
        )

        results.append(metrics)

    threshold_results = pd.DataFrame(
        results
    )

    best_index = (
        threshold_results["f1"].idxmax()
    )

    best_row = threshold_results.loc[
        best_index
    ]

    selected_threshold = float(
        best_row["threshold"]
    )

    return (
        selected_threshold,
        threshold_results,
        probabilities,
        y_threshold,
    )


# ---------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------

def get_git_commit():
    """Return the current Git commit when available."""

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return "UNAVAILABLE"


# ---------------------------------------------------------------------
# Validation mode
# ---------------------------------------------------------------------

def validate_only():
    """
    Validate configuration, datasets, temporal splits,
    class counts, expected weights, DVC status, and
    ID separation without training or creating an MLflow run.
    """

    print("=" * 70)
    print(
        "CONTROLLED XGBOOST - VALIDATION ONLY"
    )
    print("=" * 70)

    print("\nFrozen configuration:")

    print(
        f"n_estimators   = {N_ESTIMATORS}"
    )

    print(
        f"max_depth      = {MAX_DEPTH}"
    )

    print(
        f"learning_rate  = {LEARNING_RATE}"
    )

    print(
        f"objective      = {OBJECTIVE}"
    )

    print(
        f"random_state   = {RANDOM_STATE}"
    )

    print(
        f"n_jobs         = {N_JOBS}"
    )

    print(
        f"eval_metric    = {EVAL_METRIC}"
    )

    print("\nApproved model features:")

    for feature in MODEL_FEATURES:
        print(
            f"  - {feature}"
        )

    train_df = load_training_data()

    evaluation_df = load_evaluation_data()

    (
        stage1,
        threshold,
        stage2,
    ) = split_training_data(
        train_df
    )

    stage1_counts = class_counts(
        stage1
    )

    threshold_counts = class_counts(
        threshold
    )

    stage2_counts = class_counts(
        stage2
    )

    evaluation_counts = class_counts(
        evaluation_df
    )

    stage1_threshold_overlap = (
        verify_unique_ids(
            stage1,
            threshold,
        )
    )

    stage2_evaluation_overlap = (
        verify_stage2_evaluation_ids(
            stage2,
            evaluation_df,
        )
    )

    stage1_weight = (
        calculate_scale_pos_weight(
            stage1[TARGET_COLUMN]
        )
    )

    stage2_weight = (
        calculate_scale_pos_weight(
            stage2[TARGET_COLUMN]
        )
    )

    dvc_status = get_dvc_status()

    train_dvc_pointer = (
        get_dvc_pointer_status(
            TRAIN_PATH
        )
    )

    evaluation_dvc_pointer = (
        get_dvc_pointer_status(
            EVALUATION_PATH
        )
    )

    print("\nStage 1 - Apr 1-12:")

    print(
        f"  Rows       = {stage1_counts['rows']}"
    )

    print(
        f"  Legitimate = {stage1_counts['legitimate']}"
    )

    print(
        f"  Fraud      = {stage1_counts['fraud']}"
    )

    print(
        "  scale_pos_weight = "
        f"{stage1_weight:.12f}"
    )

    print(
        "\nThreshold selection - Apr 13-18:"
    )

    print(
        f"  Rows       = {threshold_counts['rows']}"
    )

    print(
        f"  Legitimate = "
        f"{threshold_counts['legitimate']}"
    )

    print(
        f"  Fraud      = "
        f"{threshold_counts['fraud']}"
    )

    print(
        "\nStage 2 - Apr 1-24:"
    )

    print(
        f"  Rows       = {stage2_counts['rows']}"
    )

    print(
        f"  Legitimate = "
        f"{stage2_counts['legitimate']}"
    )

    print(
        f"  Fraud      = "
        f"{stage2_counts['fraud']}"
    )

    print(
        "  scale_pos_weight = "
        f"{stage2_weight:.12f}"
    )

    print(
        "\nApr 25-30 evaluation:"
    )

    print(
        f"  Rows       = "
        f"{evaluation_counts['rows']}"
    )

    print(
        f"  Legitimate = "
        f"{evaluation_counts['legitimate']}"
    )

    print(
        f"  Fraud      = "
        f"{evaluation_counts['fraud']}"
    )

    print(
        "\nStage-1 / threshold ID overlap:"
    )

    print(
        f"  Overlap = "
        f"{stage1_threshold_overlap}"
    )

    print(
        "\nStage-2 / Apr 25-30 ID overlap:"
    )

    print(
        f"  Overlap = "
        f"{stage2_evaluation_overlap}"
    )

    print(
        "\nDVC status:"
    )

    print(
        f"  Return code = "
        f"{dvc_status['return_code']}"
    )

    print(
        f"  Output = "
        f"{dvc_status['stdout']}"
    )

    if dvc_status["stderr"]:
        print(
            f"  Error  = "
            f"{dvc_status['stderr']}"
        )

    print(
        "\nDVC pointer status:"
    )

    print(
        f"  Training dataset pointer: "
        f"{train_dvc_pointer['pointer']} "
        f"exists={train_dvc_pointer['exists']}"
    )

    print(
        f"  Evaluation dataset pointer: "
        f"{evaluation_dvc_pointer['pointer']} "
        f"exists={evaluation_dvc_pointer['exists']}"
    )

    expected = {
        "stage1_rows": 114978,
        "stage1_legitimate": 114610,
        "stage1_fraud": 368,
        "threshold_rows": 57525,
        "threshold_legitimate": 57176,
        "threshold_fraud": 349,
        "stage2_rows": 230295,
        "stage2_legitimate": 229119,
        "stage2_fraud": 1176,
        "evaluation_rows": 57767,
        "evaluation_legitimate": 57241,
        "evaluation_fraud": 526,
        "stage1_overlap": 0,
        "stage2_evaluation_overlap": 0,
    }

    actual = {
        "stage1_rows": stage1_counts["rows"],
        "stage1_legitimate": (
            stage1_counts["legitimate"]
        ),
        "stage1_fraud": (
            stage1_counts["fraud"]
        ),
        "threshold_rows": (
            threshold_counts["rows"]
        ),
        "threshold_legitimate": (
            threshold_counts["legitimate"]
        ),
        "threshold_fraud": (
            threshold_counts["fraud"]
        ),
        "stage2_rows": stage2_counts["rows"],
        "stage2_legitimate": (
            stage2_counts["legitimate"]
        ),
        "stage2_fraud": (
            stage2_counts["fraud"]
        ),
        "evaluation_rows": (
            evaluation_counts["rows"]
        ),
        "evaluation_legitimate": (
            evaluation_counts["legitimate"]
        ),
        "evaluation_fraud": (
            evaluation_counts["fraud"]
        ),
        "stage1_overlap": (
            stage1_threshold_overlap
        ),
        "stage2_evaluation_overlap": (
            stage2_evaluation_overlap
        ),
    }

    print("\nExpected vs actual:")

    validation_passed = True

    for key in expected:

        status = (
            "PASS"
            if expected[key] == actual[key]
            else "FAIL"
        )

        if status == "FAIL":
            validation_passed = False

        print(
            f"  {key}: "
            f"expected={expected[key]} "
            f"actual={actual[key]} "
            f"[{status}]"
        )

    print("\nValidation result:")

    if validation_passed:

        print(
            "  PASS - no model training performed."
        )

        print(
            "  PASS - no MLflow run created."
        )

    else:

        print(
            "  FAIL - do not run the experiment."
        )

        raise SystemExit(1)


# ---------------------------------------------------------------------
# Actual experiment
# ---------------------------------------------------------------------

def run_experiment():
    """
    Execute the single frozen controlled XGBoost experiment.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = load_training_data()

    evaluation_df = load_evaluation_data()

    (
        stage1,
        threshold_df,
        stage2,
    ) = split_training_data(
        train_df
    )

    stage1_threshold_overlap = (
        verify_unique_ids(
            stage1,
            threshold_df,
        )
    )

    stage2_evaluation_overlap = (
        verify_stage2_evaluation_ids(
            stage2,
            evaluation_df,
        )
    )

    # -------------------------------------------------------------
    # DVC provenance
    # -------------------------------------------------------------

    dvc_status = get_dvc_status()

    train_dvc_pointer = (
        get_dvc_pointer_status(
            TRAIN_PATH
        )
    )

    evaluation_dvc_pointer = (
        get_dvc_pointer_status(
            EVALUATION_PATH
        )
    )

    # -------------------------------------------------------------
    # Stage 1
    # -------------------------------------------------------------

    X_stage1, y_stage1 = prepare_xy(
        stage1
    )

    stage1_weight = (
        calculate_scale_pos_weight(
            y_stage1
        )
    )

    stage1_model = create_model(
        stage1_weight
    )

    print(
        "\nTraining Stage-1 model on Apr 1-12..."
    )

    stage1_model.fit(
        X_stage1,
        y_stage1,
    )

    # -------------------------------------------------------------
    # Threshold selection
    # -------------------------------------------------------------

    print(
        "Selecting threshold using Stage-1 "
        "predictions on Apr 13-18..."
    )

    (
        selected_threshold,
        threshold_results,
        _,
        _,
    ) = select_threshold(
        stage1_model,
        threshold_df,
    )

    print(
        f"Frozen threshold = "
        f"{selected_threshold:.2f}"
    )

    # -------------------------------------------------------------
    # Stage 2
    # -------------------------------------------------------------

    X_stage2, y_stage2 = prepare_xy(
        stage2
    )

    stage2_weight = (
        calculate_scale_pos_weight(
            y_stage2
        )
    )

    stage2_model = create_model(
        stage2_weight
    )

    print(
        "\nTraining Stage-2 model on Apr 1-24..."
    )

    stage2_model.fit(
        X_stage2,
        y_stage2,
    )

    # -------------------------------------------------------------
    # Apr 25-30 evaluation
    # -------------------------------------------------------------

    X_evaluation, y_evaluation = (
        prepare_xy(
            evaluation_df
        )
    )

    evaluation_probabilities = (
        stage2_model.predict_proba(
            X_evaluation
        )[:, 1]
    )

    evaluation_metrics = (
        calculate_metrics(
            y_evaluation,
            evaluation_probabilities,
            selected_threshold,
        )
    )

    print(
        "\nApr 25-30 evaluation:"
    )

    for key, value in (
        evaluation_metrics.items()
    ):
        print(
            f"  {key}: {value}"
        )

    # -------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------

    git_commit = get_git_commit()

    provenance = {
        "experiment": MLFLOW_EXPERIMENT,

        "git_commit": git_commit,

        "train_dataset": str(
            TRAIN_PATH
        ),

        "evaluation_dataset": str(
            EVALUATION_PATH
        ),

        "dvc_status": dvc_status,

        "train_dataset_dvc_tracking": (
            train_dvc_pointer
        ),

        "evaluation_dataset_dvc_tracking": (
            evaluation_dvc_pointer
        ),

        "features": MODEL_FEATURES,

        "target": TARGET_COLUMN,

        "preprocessing": (
            "StandardScaler inside sklearn Pipeline"
        ),

        "model": "XGBoost",

        "n_estimators": N_ESTIMATORS,

        "max_depth": MAX_DEPTH,

        "learning_rate": LEARNING_RATE,

        "objective": OBJECTIVE,

        "random_state": RANDOM_STATE,

        "n_jobs": N_JOBS,

        "eval_metric": EVAL_METRIC,

        "stage1": {
            **class_counts(stage1),

            "scale_pos_weight": stage1_weight,

            "date_range": (
                "2018-04-01 to 2018-04-12"
            ),
        },

        "threshold_selection": {
            **class_counts(
                threshold_df
            ),

            "date_range": (
                "2018-04-13 to 2018-04-18"
            ),

            "selection_method": (
                "maximum F1"
            ),

            "selected_threshold": (
                selected_threshold
            ),
        },

        "stage2": {
            **class_counts(stage2),

            "scale_pos_weight": stage2_weight,

            "date_range": (
                "2018-04-01 to 2018-04-24"
            ),

            "training_evaluation_id_overlap": (
                stage2_evaluation_overlap
            ),
        },

        "evaluation": {
            **class_counts(
                evaluation_df
            ),

            "date_range": (
                "2018-04-25 to 2018-04-30"
            ),

            "metrics": evaluation_metrics,
        },

        "id_overlap_checks": {
            "stage1_threshold": (
                stage1_threshold_overlap
            ),

            "stage2_evaluation": (
                stage2_evaluation_overlap
            ),
        },

        "may_usage": "Not used",
    }

    provenance_path = (
        OUTPUT_DIR / "provenance.json"
    )

    with provenance_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            provenance,
            file,
            indent=2,
        )

    threshold_results_path = (
        OUTPUT_DIR
        / "threshold_results.csv"
    )

    threshold_results.to_csv(
        threshold_results_path,
        index=False,
    )

    # -------------------------------------------------------------
    # MLflow
    # -------------------------------------------------------------

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    with mlflow.start_run(
        run_name="controlled_xgboost_phase4"
    ) as run:

        mlflow.log_params(
            {
                "model": "XGBoost",

                "n_estimators": (
                    N_ESTIMATORS
                ),

                "max_depth": MAX_DEPTH,

                "learning_rate": (
                    LEARNING_RATE
                ),

                "objective": OBJECTIVE,

                "random_state": (
                    RANDOM_STATE
                ),

                "n_jobs": N_JOBS,

                "eval_metric": (
                    EVAL_METRIC
                ),

                "stage1_scale_pos_weight": (
                    stage1_weight
                ),

                "stage2_scale_pos_weight": (
                    stage2_weight
                ),

                "selected_threshold": (
                    selected_threshold
                ),
            }
        )

        mlflow.log_metrics(
            {
                key: value
                for key, value in (
                    evaluation_metrics.items()
                )
                if key != "threshold"
                and isinstance(
                    value,
                    (int, float),
                )
            }
        )

        mlflow.log_params(
            {
                "train_dataset": str(
                    TRAIN_PATH
                ),

                "evaluation_dataset": str(
                    EVALUATION_PATH
                ),

                "git_commit": git_commit,

                "preprocessing": (
                    "StandardScaler inside "
                    "sklearn Pipeline"
                ),

                "stage1_rows": (
                    class_counts(
                        stage1
                    )["rows"]
                ),

                "stage1_legitimate": (
                    class_counts(
                        stage1
                    )["legitimate"]
                ),

                "stage1_fraud": (
                    class_counts(
                        stage1
                    )["fraud"]
                ),

                "stage2_rows": (
                    class_counts(
                        stage2
                    )["rows"]
                ),

                "stage2_legitimate": (
                    class_counts(
                        stage2
                    )["legitimate"]
                ),

                "stage2_fraud": (
                    class_counts(
                        stage2
                    )["fraud"]
                ),

                "stage1_threshold_id_overlap": (
                    stage1_threshold_overlap
                ),

                "stage2_evaluation_id_overlap": (
                    stage2_evaluation_overlap
                ),
            }
        )

        mlflow.log_artifact(
            str(provenance_path)
        )

        mlflow.log_artifact(
            str(threshold_results_path)
        )

        mlflow.sklearn.log_model(
            stage2_model,
            name="controlled_xgboost_model",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        print(
            "\nMLflow run ID:"
        )

        print(
            run.info.run_id
        )

    print(
        "\nControlled XGBoost experiment complete."
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate datasets and configuration "
            "without training or creating an "
            "MLflow run."
        ),
    )

    args = parser.parse_args()

    if args.validate_only:

        validate_only()

    else:

        run_experiment()


if __name__ == "__main__":
    main()
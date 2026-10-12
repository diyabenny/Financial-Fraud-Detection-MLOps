"""
Evaluate fixed Phase 4 candidate models on a later temporal
evaluation set.
"""

import sys
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.train_phase4 import (
    MODEL_FEATURES,
    TARGET,
    predict_probabilities,
    prepare_features,
    train_model,
)


# ============================================================
# INPUT DATASETS
# ============================================================

RETRAINING_DATASET = (
    "data/phase3/train_feast_features.csv"
)

EVALUATION_DATASET = (
    "data/phase4/april_25_30_evaluation_features.csv"
)


# ============================================================
# FIXED CANDIDATES
# ============================================================

CANDIDATES = [
    {
        "name": "LR_Original_C_0.01",
        "model_type": "logistic_regression",
        "imbalance_strategy": "original",
        "model_parameters": {
            "C": 0.01
        },
        "threshold": 0.10,
    },
    {
        "name": "RF_Balanced_100_None_1",
        "model_type": "random_forest",
        "imbalance_strategy": "balanced",
        "model_parameters": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        "threshold": 0.65,
    },
    {
        "name": "RF_Balanced_300_None_1",
        "model_type": "random_forest",
        "imbalance_strategy": "balanced",
        "model_parameters": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        "threshold": 0.65,
    },
    {
        "name": "LR_Balanced_C_10",
        "model_type": "logistic_regression",
        "imbalance_strategy": "balanced",
        "model_parameters": {
            "C": 10
        },
        "threshold": 0.95,
    },
]


def load_retraining_data():
    """Load the Apr 1-24 Feast retraining dataset.

    Returns:
        DataFrame containing the retraining features and target.
    """
    return pd.read_csv(RETRAINING_DATASET)


def load_evaluation_data():
    """Load the Apr 25-30 evaluation feature dataset.

    Returns:
        DataFrame containing the evaluation features and target.
    """
    return pd.read_csv(EVALUATION_DATASET)


def prepare_evaluation_data():
    """Prepare separate Apr 1-24 retraining and Apr 25-30 evaluation data.

    Returns:
        A tuple containing:
        X_retrain, y_retrain, X_evaluation, y_evaluation.
    """
    retraining_df = load_retraining_data()
    evaluation_df = load_evaluation_data()

    X_retrain, y_retrain = prepare_features(
        retraining_df,
        retraining_df,
    )[0:2]

    X_evaluation, y_evaluation = prepare_features(
        evaluation_df,
        evaluation_df,
    )[2:4]

    return (
        X_retrain,
        y_retrain,
        X_evaluation,
        y_evaluation,
    )


def calculate_fixed_threshold_metrics(
    y_true,
    probabilities,
    threshold,
):
    """Calculate evaluation metrics using a fixed fraud threshold.

    Parameters:
        y_true: True fraud labels.
        probabilities: Predicted fraud probabilities.
        threshold: Previously selected candidate-specific threshold.

    Returns:
        Dictionary containing threshold-dependent and
        threshold-independent evaluation metrics.
    """
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    fpr = fp / (fp + tn)

    return {
        "threshold": threshold,
        "recall": recall_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
        ),
        "f1": f1_score(
            y_true,
            predictions,
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


def main():
    """Retrain fixed candidates on Apr 1-24 and evaluate on Apr 25-30."""
    print("=" * 70)
    print(
        "PHASE 4 - FIXED CANDIDATE EVALUATION"
    )
    print("=" * 70)

    (
        X_retrain,
        y_retrain,
        X_evaluation,
        y_evaluation,
    ) = prepare_evaluation_data()

    print(
        "\nRetraining feature shape:",
        X_retrain.shape,
    )

    print(
        "Retraining target shape:",
        y_retrain.shape,
    )

    print(
        "Evaluation feature shape:",
        X_evaluation.shape,
    )

    print(
        "Evaluation target shape:",
        y_evaluation.shape,
    )

    print(
        "\nRetraining fraud count:",
        int(y_retrain.sum()),
    )

    print(
        "Evaluation fraud count:",
        int(y_evaluation.sum()),
    )

    results = []

    for candidate in CANDIDATES:
        print(
            "\n" + "-" * 70
        )

        print(
            "Candidate:",
            candidate["name"],
        )

        print(
            "Threshold:",
            candidate["threshold"],
        )

        model = train_model(
            X_retrain,
            y_retrain,
            model_type=candidate["model_type"],
            imbalance_strategy=candidate[
                "imbalance_strategy"
            ],
            model_parameters=candidate[
                "model_parameters"
            ],
        )

        probabilities = predict_probabilities(
            model,
            X_evaluation,
        )

        metrics = calculate_fixed_threshold_metrics(
            y_evaluation,
            probabilities,
            candidate["threshold"],
        )

        metrics["candidate"] = candidate["name"]

        results.append(metrics)

        print(
            "PR-AUC:",
            metrics["pr_auc"],
        )

        print(
            "ROC-AUC:",
            metrics["roc_auc"],
        )

        print(
            "Threshold:",
            metrics["threshold"],
        )

        print(
            "Recall:",
            metrics["recall"],
        )

        print(
            "Precision:",
            metrics["precision"],
        )

        print(
            "F1:",
            metrics["f1"],
        )

        print(
            "Accuracy:",
            metrics["accuracy"],
        )

        print(
            "FPR:",
            metrics["fpr"],
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL FIXED-THRESHOLD EVALUATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        results_df[
            [
                "candidate",
                "threshold",
                "pr_auc",
                "roc_auc",
                "recall",
                "precision",
                "f1",
                "accuracy",
                "fpr",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

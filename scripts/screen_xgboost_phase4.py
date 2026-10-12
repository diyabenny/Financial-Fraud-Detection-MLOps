"""
Run predefined XGBoost screening configurations for Phase 4.

This script uses Apr 1-18 for training and Apr 19-24
for internal validation. The screening threshold is fixed
at 0.50 and is not a final operating threshold.
"""

import sys
from pathlib import Path

import mlflow
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
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DATASET
# ============================================================

DATASET_PATH = (
    "data/phase3/train_feast_features.csv"
)

TARGET = "TX_FRAUD"

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


# ============================================================
# SCREENING SETTINGS
# ============================================================

SCREENING_THRESHOLD = 0.50

EXPERIMENT_NAME = (
    "financial_fraud_detection_phase4_xgboost_screening"
)


# ============================================================
# FROZEN XGBOOST CONFIGURATIONS
# ============================================================

CONFIGURATIONS = [
    {
        "name": "X01",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
    },
    {
        "name": "X02",
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
    },
    {
        "name": "X03",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.10,
    },
]


def load_data():
    """Load the Feast-generated Phase 4 training dataset.

    Returns:
        DataFrame containing Apr 1-24 feature data.
    """
    return pd.read_csv(DATASET_PATH)


def create_temporal_split(df):
    """Create the fixed Apr 1-18 and Apr 19-24 temporal split.

    Parameters:
        df: DataFrame containing event_timestamp and TX_FRAUD.

    Returns:
        Tuple containing training and validation DataFrames.
    """
    data = df.copy()

    if "event_timestamp" not in data.columns:
        raise ValueError(
            "event_timestamp column is required."
        )

    data["event_timestamp"] = pd.to_datetime(
        data["event_timestamp"],
        utc=True,
    )

    train_df = data[
        data["event_timestamp"].dt.day <= 18
    ].copy()

    validation_df = data[
        data["event_timestamp"].dt.day.between(19, 24)
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if validation_df.empty:
        raise ValueError(
            "Validation dataset is empty."
        )

    return train_df, validation_df


def prepare_features(df):
    """Extract the approved model features and target.

    Parameters:
        df: DataFrame containing model features and TX_FRAUD.

    Returns:
        Tuple containing feature matrix and target vector.
    """
    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET}"
        )

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


def calculate_scale_pos_weight(y_train):
    """Calculate XGBoost class weighting from training data only.

    Parameters:
        y_train: Training target labels.

    Returns:
        Ratio of legitimate transactions to fraud transactions.
    """
    legitimate_count = int(
        (y_train == 0).sum()
    )

    fraud_count = int(
        (y_train == 1).sum()
    )

    if fraud_count == 0:
        raise ValueError(
            "Training data contains no fraud cases."
        )

    return legitimate_count / fraud_count


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    """Calculate screening metrics at a fixed threshold.

    Parameters:
        y_true: True fraud labels.
        probabilities: Predicted fraud probabilities.
        threshold: Screening threshold.

    Returns:
        Dictionary containing classification metrics.
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
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "recall": recall_score(
            y_true,
            predictions,
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


def run_configuration(
    configuration,
    X_train,
    y_train,
    X_validation,
    y_validation,
    scale_pos_weight,
):
    """Train and evaluate one frozen XGBoost configuration.

    Parameters:
        configuration: Frozen XGBoost hyperparameters.
        X_train: Apr 1-18 training features.
        y_train: Apr 1-18 training labels.
        X_validation: Apr 19-24 validation features.
        y_validation: Apr 19-24 validation labels.
        scale_pos_weight: Weight calculated from training data.

    Returns:
        Tuple containing the trained model and screening metrics.
    """
    model = XGBClassifier(
        n_estimators=configuration["n_estimators"],
        max_depth=configuration["max_depth"],
        learning_rate=configuration["learning_rate"],
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    metrics = calculate_metrics(
        y_validation,
        probabilities,
        SCREENING_THRESHOLD,
    )

    return model, {
        "configuration": configuration["name"],
        "n_estimators": configuration["n_estimators"],
        "max_depth": configuration["max_depth"],
        "learning_rate": configuration["learning_rate"],
        "scale_pos_weight": scale_pos_weight,
        **metrics,
    }


def main():
    """Run all three predefined XGBoost screening configurations."""
    print("=" * 70)
    print("PHASE 4 - XGBOOST SCREENING")
    print("=" * 70)

    df = load_data()

    train_df, validation_df = (
        create_temporal_split(df)
    )

    X_train, y_train = prepare_features(
        train_df
    )

    X_validation, y_validation = (
        prepare_features(validation_df)
    )

    scale_pos_weight = (
        calculate_scale_pos_weight(y_train)
    )

    print("\nTemporal split:")
    print(
        "Training period: Apr 1-18"
    )
    print(
        "Validation period: Apr 19-24"
    )

    print(
        "\nTraining shape:",
        X_train.shape,
    )

    print(
        "Validation shape:",
        X_validation.shape,
    )

    print(
        "Training fraud:",
        int(y_train.sum()),
    )

    print(
        "Training legitimate:",
        int((y_train == 0).sum()),
    )

    print(
        "Validation fraud:",
        int(y_validation.sum()),
    )

    print(
        "Validation legitimate:",
        int((y_validation == 0).sum()),
    )

    print(
        "scale_pos_weight:",
        scale_pos_weight,
    )

    print(
        "Screening threshold:",
        SCREENING_THRESHOLD,
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    results = []

    for configuration in CONFIGURATIONS:
        print(
            "\n" + "-" * 70
        )

        print(
            "Running:",
            configuration["name"],
        )

        with mlflow.start_run(
            run_name=configuration["name"]
        ):
            model, metrics = run_configuration(
                configuration,
                X_train,
                y_train,
                X_validation,
                y_validation,
                scale_pos_weight,
            )

            mlflow.xgboost.log_model(
                model,
                name="model",
            )
            mlflow.log_params(
                {
                    "model_type": "xgboost",
                    "configuration": configuration[
                        "name"
                    ],
                    "n_estimators": configuration[
                        "n_estimators"
                    ],
                    "max_depth": configuration[
                        "max_depth"
                    ],
                    "learning_rate": configuration[
                        "learning_rate"
                    ],
                    "random_state": 42,
                    "n_jobs": -1,
                    "eval_metric": "logloss",
                    "scale_pos_weight": scale_pos_weight,
                    "evaluation_threshold": SCREENING_THRESHOLD,
                    "threshold_status": "screening_only",
                    "training_period": "Apr 1-18",
                    "validation_period": "Apr 19-24",
                }
            )

            mlflow.log_metrics(
                {
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "fpr": metrics["fpr"],
                }
            )

            metrics["mlflow_run_id"] = (
                mlflow.active_run().info.run_id
            )

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
        "XGBOOST SCREENING RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        results_df[
            [
                "configuration",
                "scale_pos_weight",
                "threshold",
                "pr_auc",
                "roc_auc",
                "recall",
                "precision",
                "f1",
                "accuracy",
                "fpr",
                "mlflow_run_id",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

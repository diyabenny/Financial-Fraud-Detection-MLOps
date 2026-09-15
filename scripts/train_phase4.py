"""
Reusable Phase 4 training pipeline for financial fraud detection.
"""

import sys
from pathlib import Path

import pandas as pd
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlflow_utils import setup_mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

TARGET = "TX_FRAUD"


def load_datasets(train_path, validation_path):
    """Load the training and validation datasets.

    Parameters:
        train_path: Path to the Feast training dataset.
        validation_path: Path to the Feast validation dataset.

    Returns:
        A tuple containing the training and validation DataFrames.
    """
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    return train_df, validation_df


def prepare_features(train_df, validation_df):
    """Separate model features and target from the datasets.

    Parameters:
        train_df: Training DataFrame.
        validation_df: Validation DataFrame.

    Returns:
        A tuple containing training features, training target,
        validation features, and validation target.
    """
    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[MODEL_FEATURES]
    y_validation = validation_df[TARGET]

    return X_train, y_train, X_validation, y_validation


def train_model(
    X_train,
    y_train,
    model_type,
    imbalance_strategy,
    model_parameters=None,
):
    """Train a fraud detection model using the selected configuration.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target.
        model_type: Model family to train.
        imbalance_strategy: Strategy used to handle class imbalance.
        model_parameters: Optional model hyperparameters.

    Returns:
        A fitted machine learning model.
    Raises:
        ValueError: If an unsupported model type is provided.
    """
    if model_parameters is None:
        model_parameters = {}

    if model_type == "logistic_regression":
        class_weight = None

        if imbalance_strategy == "balanced":
            class_weight = "balanced"

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight=class_weight,
                        random_state=42,
                        max_iter=1000,
                        **model_parameters,
                    ),
                ),
            ]
        )

    elif model_type == "random_forest":
        class_weight = None

        if imbalance_strategy == "balanced":
            class_weight = "balanced"

        model = RandomForestClassifier(
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            **model_parameters,
        )

    elif model_type == "xgboost":
        model = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            **model_parameters,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)

    return model


def predict_probabilities(model, X_validation):
    """Generate fraud probabilities for validation data.

    Parameters:
        model: Fitted machine learning model.
        X_validation: Validation feature matrix.

    Returns:
        Fraud probability for each validation transaction.
    """
    probabilities = model.predict_proba(X_validation)[:, 1]

    return probabilities


def calculate_metrics(
    y_validation,
    validation_probabilities,
    threshold=0.5,
):
    """Calculate validation metrics from fraud probabilities.

    Parameters:
        y_validation: True validation fraud labels.
        validation_probabilities: Predicted fraud probabilities.
        threshold: Probability threshold used for classification.

    Returns:
        A dictionary containing validation metrics.
    """
    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        validation_predictions,
    ).ravel()

    fpr = fp / (fp + tn)

    metrics = {
        "recall": recall_score(y_validation, validation_predictions),
        "precision": precision_score(y_validation, validation_predictions),
        "f1": f1_score(y_validation, validation_predictions),
        "pr_auc": average_precision_score(
            y_validation,
            validation_probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_validation,
            validation_probabilities,
        ),
        "accuracy": accuracy_score(
            y_validation,
            validation_predictions,
        ),
        "fpr": fpr,
    }

    return metrics


def run_experiment(
    run_name,
    model_type,
    imbalance_strategy,
    model_parameters=None,
    threshold=0.5,
    train_path="data/phase3/train_feast_features.csv",
    validation_path="data/phase3/validation_feast_features.csv",
):
    """Run one Phase 4 model experiment and log results to MLflow.

    Parameters:
        run_name: Name assigned to the MLflow run.
        model_type: Model family to train.
        imbalance_strategy: Strategy used to handle class imbalance.
        model_parameters: Optional model hyperparameters.
        threshold: Probability threshold used for classification.
        train_path: Path to the training dataset.
        validation_path: Path to the validation dataset.

    Returns:
        A dictionary containing validation metrics.
    Side Effects:
        Creates an MLflow run and logs parameters, metrics,
        and the trained model artifact.
    """
    setup_mlflow()

    with mlflow.start_run(run_name=run_name):
        train_df, validation_df = load_datasets(
            train_path,
            validation_path,
        )

        X_train, y_train, X_validation, y_validation = prepare_features(
            train_df,
            validation_df,
        )

        model_parameters = dict(model_parameters or {})

        if model_type == "xgboost":
            negative_count = (y_train == 0).sum()
            positive_count = (y_train == 1).sum()

            model_parameters["scale_pos_weight"] = (
                negative_count / positive_count
            )

        mlflow.log_params(
            {
                "model_type": model_type,
                "imbalance_strategy": imbalance_strategy,
                "random_state": 42,
                "n_features": len(MODEL_FEATURES),
                "threshold": threshold,
                "train_dataset": str(train_path),
                "validation_dataset": str(validation_path),
                **model_parameters,
            }
        )

        model = train_model(
            X_train,
            y_train,
            model_type=model_type,
            imbalance_strategy=imbalance_strategy,
            model_parameters=model_parameters,
        )

        validation_probabilities = predict_probabilities(
            model,
            X_validation,
        )

        metrics = calculate_metrics(
            y_validation,
            validation_probabilities,
            threshold=threshold,
        )

        mlflow.log_metrics(metrics)

        if model_type == "xgboost":
            mlflow.xgboost.log_model(
                model,
                name="model",
            )
        else:
            mlflow.sklearn.log_model(
                model,
                name="model",
            )
        print("Validation metrics:", metrics)
        print("Model trained:", type(model).__name__)
        print("Training feature shape:", X_train.shape)
        print("Training target shape:", y_train.shape)
        print("Validation feature shape:", X_validation.shape)
        print("Validation target shape:", y_validation.shape)

        return metrics


def main():
    """Run the Phase 4 screening experiments.

    Parameters:
        None.

    Returns:
        None.

    Side Effects:
        Runs the configured Phase 4 screening experiments
        and logs their results to MLflow.
    """
    experiments = [
        {
            "run_name": "E08_LR_original",
            "model_type": "logistic_regression",
            "imbalance_strategy": "original",
            "model_parameters": {},
        },
        {
            "run_name": "E09_LR_balanced",
            "model_type": "logistic_regression",
            "imbalance_strategy": "balanced",
            "model_parameters": {},
        },
        {
            "run_name": "E10_RF_balanced",
            "model_type": "random_forest",
            "imbalance_strategy": "balanced",
            "model_parameters": {},
        },
        {
            "run_name": "E11_XGB_scale_pos_weight",
            "model_type": "xgboost",
            "imbalance_strategy": "scale_pos_weight",
            "model_parameters": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
            },
        },
    ]

    for experiment in experiments:
        run_experiment(**experiment)


if __name__ == "__main__":
    main()

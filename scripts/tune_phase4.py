"""
Run Phase 4 Round 1 hyperparameter tuning using a temporal validation split.
"""

import sys
from pathlib import Path

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_phase4 import (
    MODEL_FEATURES,
    TARGET,
    calculate_metrics,
    predict_probabilities,
    prepare_features,
    setup_mlflow,
    train_model,
)

TUNING_DATASET = "data/phase3/train_feast_features.csv"
TUNING_THRESHOLD = 0.5


def create_temporal_split(df):
    """Split the April training data into temporal tuning subsets.

    Parameters:
        df: Feast training feature dataframe covering April 1 to April 24.

    Returns:
        A tuple containing the tuning training dataframe and
        internal validation dataframe.

    Raises:
        ValueError: If the dataframe does not contain event_timestamp.
    """
    if "event_timestamp" not in df.columns:
        raise ValueError("event_timestamp column is required for temporal splitting.")

    df = df.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

    tuning_train_df = df[df["event_timestamp"].dt.day <= 18].copy()
    tuning_validation_df = df[
        df["event_timestamp"].dt.day.between(19, 24)
    ].copy()

    return tuning_train_df, tuning_validation_df

TUNING_CONFIGURATIONS = []

# Logistic Regression — original class weighting
for c_value in [0.01, 0.1, 1, 10]:
    TUNING_CONFIGURATIONS.append(
        {
            "run_name": f"T01_LR_original_C_{c_value}",
            "model_type": "logistic_regression",
            "imbalance_strategy": "original",
            "model_parameters": {"C": c_value},
        }
    )

# Logistic Regression — balanced class weighting
for c_value in [0.01, 0.1, 1, 10]:
    TUNING_CONFIGURATIONS.append(
        {
            "run_name": f"T01_LR_balanced_C_{c_value}",
            "model_type": "logistic_regression",
            "imbalance_strategy": "balanced",
            "model_parameters": {"C": c_value},
        }
    )

# Random Forest — balanced class weighting
for n_estimators in [100, 300]:
    for max_depth in [None, 10, 20]:
        for min_samples_leaf in [1, 5]:
            TUNING_CONFIGURATIONS.append(
                {
                    "run_name": (
                        f"T01_RF_balanced_"
                        f"n{n_estimators}_"
                        f"depth{max_depth}_"
                        f"leaf{min_samples_leaf}"
                    ),
                    "model_type": "random_forest",
                    "imbalance_strategy": "balanced",
                    "model_parameters": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    },
                }
            )


def load_tuning_data():
    """Load the April training feature dataset for Round 1 tuning.

    Returns:
        A pandas dataframe containing the April 1 to April 24
        Feast training features.

    Raises:
        FileNotFoundError: If the tuning dataset does not exist.
    """
    df = pd.read_csv(TUNING_DATASET)

    if df.empty:
        raise ValueError("Tuning dataset is empty.")

    return df

def prepare_tuning_data():
    """Load and prepare the temporal tuning datasets.

    Returns:
        A tuple containing the tuning training features, tuning training
        target, validation features, and validation target.

    Raises:
        ValueError: If either temporal subset is empty.
    """
    df = load_tuning_data()

    tuning_train_df, tuning_validation_df = create_temporal_split(df)

    if tuning_train_df.empty or tuning_validation_df.empty:
        raise ValueError("Temporal tuning split produced an empty subset.")

    X_train, y_train, X_val, y_val = prepare_features(
        tuning_train_df,
        tuning_validation_df,
    )

    return X_train, y_train, X_val, y_val

def run_tuning_configuration(config, X_train, y_train, X_val, y_val):
    """Train and evaluate one Round 1 tuning configuration.

    Parameters:
        config: Dictionary containing the model type, imbalance strategy,
            model parameters, and MLflow run name.
        X_train: Training feature matrix.
        y_train: Training target values.
        X_val: Internal validation feature matrix.
        y_val: Internal validation target values.

    Returns:
        A dictionary containing the configuration name and evaluation metrics.

    Side Effects:
        Creates one MLflow run containing the configuration parameters
        and validation metrics.

    Raises:
        ValueError: If the configuration is missing required fields.
    """
    required_keys = {
        "run_name",
        "model_type",
        "imbalance_strategy",
        "model_parameters",
    }

    if not required_keys.issubset(config):
        raise ValueError("Tuning configuration is missing required fields.")

    model = train_model(
        X_train,
        y_train,
        model_type=config["model_type"],
        imbalance_strategy=config["imbalance_strategy"],
        model_parameters=config["model_parameters"],
    )

    probabilities = predict_probabilities(model, X_val)

    metrics = calculate_metrics(
        y_val,
        probabilities,
        threshold=TUNING_THRESHOLD,
    )

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params(
            {
                "model_type": config["model_type"],
                "imbalance_strategy": config["imbalance_strategy"],
                "threshold": TUNING_THRESHOLD,
                "tuning_round": "Round 1",
                "evaluation_type": "internal_temporal_validation",
                **config["model_parameters"],
            }
        )

        mlflow.log_metrics(metrics)

    return {
        "run_name": config["run_name"],
        **metrics,
    }

def main():
    """Run all fixed Round 1 tuning configurations.

    Returns:
        None.

    Side Effects:
        Trains each configured model, creates one MLflow run per
        configuration, and prints the tuning results.
    """
    setup_mlflow()

    X_train, y_train, X_val, y_val = prepare_tuning_data()

    results = []

    print("=" * 60)
    print("Phase 4 Round 1 Hyperparameter Tuning")
    print("=" * 60)
    print("Tuning training: April 1-18")
    print("Tuning validation: April 19-24")
    print("Total configurations:", len(TUNING_CONFIGURATIONS))

    for index, config in enumerate(TUNING_CONFIGURATIONS, start=1):
        print(
            f"\n[{index}/{len(TUNING_CONFIGURATIONS)}] "
            f"Running {config['run_name']}..."
        )

        result = run_tuning_configuration(
            config,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        results.append(result)

        print(
            f"PR-AUC: {result['pr_auc']:.6f} | "
            f"Recall: {result['recall']:.6f} | "
            f"Precision: {result['precision']:.6f} | "
            f"F1: {result['f1']:.6f}"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="pr_auc",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("Round 1 Results — sorted by PR-AUC")
    print("=" * 60)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

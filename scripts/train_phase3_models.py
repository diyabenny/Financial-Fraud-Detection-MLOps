"""
Phase 3.2 - Model Development and Training.

This script establishes the first Phase 3.2 modeling experiment (E01)
using the approved Phase 3.1 feature datasets.

E01:
    Logistic Regression
    No class-imbalance adjustment

The script:
    1. Loads the DVC-managed Phase 3.1 feature datasets.
    2. Validates the approved model feature contract.
    3. Explicitly separates model features (X) from the target (y).
    4. Fits preprocessing only on the training data.
    5. Trains the E01 Logistic Regression model.
    6. Generates validation fraud probabilities.
    7. Performs basic prediction sanity checks.

Important project boundaries:
    - Phase 2.5 train/validation data is not modified or recreated.
    - May/future test data is not used.
    - TX_FRAUD is the target and is not used as a model feature.
    - TX_FRAUD_SCENARIO is excluded because of leakage risk.
    - Raw identifiers are not used as model features.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------

TRAIN_FEATURES_PATH = Path(
    "data/phase3/train_features.csv"
)

VALIDATION_FEATURES_PATH = Path(
    "data/phase3/validation_features.csv"
)

# ---------------------------------------------------------------------------
# E01 output path
# ---------------------------------------------------------------------------

E01_OUTPUT_PATH = Path(
    "reports/phase3/experiments/E01_validation_predictions.csv"
)

E02_OUTPUT_PATH = Path(
    "reports/phase3/experiments/E02_validation_predictions.csv"
)
# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "E01"

MODEL_FEATURES = [
    "TX_AMOUNT",
    "AMOUNT_LOG",
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
    "TX_IS_WEEKEND",
]

TARGET_COLUMN = "TX_FRAUD"

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Columns that must never enter the model
# ---------------------------------------------------------------------------

FORBIDDEN_MODEL_COLUMNS = {
    "TRANSACTION_ID",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_DATETIME",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
}


def load_feature_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the approved Phase 3.1 training and validation feature datasets.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training and validation feature datasets.

    Raises
    ------
    FileNotFoundError
        If either required feature dataset is not available.
    """
    if not TRAIN_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Training feature dataset not found: "
            f"{TRAIN_FEATURES_PATH}"
        )

    if not VALIDATION_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Validation feature dataset not found: "
            f"{VALIDATION_FEATURES_PATH}"
        )

    train_features = pd.read_csv(TRAIN_FEATURES_PATH)
    validation_features = pd.read_csv(
        VALIDATION_FEATURES_PATH
    )

    return train_features, validation_features


def validate_model_inputs(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
) -> None:
    """
    Validate the input datasets against the Phase 3.2 model contract.

    Parameters
    ----------
    train_features : pd.DataFrame
        Phase 3.1 training feature dataset.
    validation_features : pd.DataFrame
        Phase 3.1 validation feature dataset.

    Raises
    ------
    ValueError
        If required columns are missing, feature columns differ,
        target values are invalid, or model features contain missing values.
    """
    required_columns = set(MODEL_FEATURES)
    required_columns.add(TARGET_COLUMN)

    for dataset_name, df in (
        ("Training", train_features),
        ("Validation", validation_features),
    ):
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{dataset_name} dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if df[MODEL_FEATURES].isna().any().any():
            raise ValueError(
                f"{dataset_name} dataset contains missing model-feature "
                "values."
            )

        if not df[TARGET_COLUMN].isin([0, 1]).all():
            raise ValueError(
                f"{dataset_name} dataset contains TX_FRAUD values "
                "other than 0 and 1."
            )

    # Training and validation must use exactly the same
    # model feature columns and order.
    if list(train_features[MODEL_FEATURES].columns) != list(
        validation_features[MODEL_FEATURES].columns
    ):
        raise ValueError(
            "Training and validation model-feature columns do not match."
        )

    # Confirm no forbidden column can accidentally enter the model
    # through the explicit model-feature contract.
    forbidden_in_model = (
        set(MODEL_FEATURES) & FORBIDDEN_MODEL_COLUMNS
    )

    if forbidden_in_model:
        raise ValueError(
            "Forbidden columns found in MODEL_FEATURES: "
            f"{sorted(forbidden_in_model)}"
        )


def prepare_model_data(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    Explicitly construct training and validation X/y datasets.

    Parameters
    ----------
    train_features : pd.DataFrame
        Phase 3.1 training feature dataset.
    validation_features : pd.DataFrame
        Phase 3.1 validation feature dataset.

    Returns
    -------
    tuple
        X_train, y_train, X_validation, y_validation.
    """
    X_train = train_features[MODEL_FEATURES]
    y_train = train_features[TARGET_COLUMN]

    X_validation = validation_features[MODEL_FEATURES]
    y_validation = validation_features[TARGET_COLUMN]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


def build_e01_model() -> Pipeline:
    """
    Build the E01 Logistic Regression pipeline.

    Returns
    -------
    Pipeline
        A pipeline containing StandardScaler followed by
        Logistic Regression.

    Notes
    -----
    The scaler is fitted only when the pipeline is fitted on
    the training data. Validation data is only transformed using
    the fitted scaler.
    """
    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


def build_e02_model() -> Pipeline:
    """
    Build the E02 Logistic Regression pipeline.

    Returns
    -------
    Pipeline
        A pipeline containing StandardScaler followed by
        Logistic Regression with class-weight balancing.

    Notes
    -----
    E02 introduces class_weight="balanced" to give greater
    importance to the minority fraud class during training.
    """
    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model

def save_validation_predictions(
    validation_features: pd.DataFrame,
    validation_probabilities,
    output_path: Path,
) -> None:
    """
    Save E01 validation probabilities for downstream evaluation.

    Parameters
    ----------
    validation_features : pd.DataFrame
        Phase 3.1 validation feature dataset containing
        TRANSACTION_ID and TX_FRAUD.
    validation_probabilities : array-like
        Fraud probabilities generated by the E01 model.

    Raises
    ------
    ValueError
        If the number of probabilities does not match the
        validation dataset length.
    """
    if len(validation_probabilities) != len(validation_features):
        raise ValueError(
            "Number of validation probabilities does not match "
            "the validation dataset length."
        )

    prediction_artifact = pd.DataFrame(
        {
            "TRANSACTION_ID": validation_features["TRANSACTION_ID"],
            "TX_FRAUD": validation_features["TX_FRAUD"],
            "fraud_probability": validation_probabilities,
        }
    )

    output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
    )

    prediction_artifact.to_csv(
    output_path,
    index=False,
    )

    print(
    "\nPrediction artifact saved to: "
    f"{output_path}"
    )


def validate_saved_predictions(
    expected_rows: int,
    output_path: Path,
) -> None:
    """
    Re-read and validate the saved E01 prediction artifact.

    Raises
    ------
    FileNotFoundError
        If the prediction artifact does not exist.
    ValueError
        If required columns are missing, the row count is incorrect,
        or fraud probabilities fall outside the range [0, 1].
    """
    if not output_path.exists():
        raise FileNotFoundError(
            f"E01 prediction artifact not found: "
            f"{output_path}"
        )

    saved_predictions = pd.read_csv(
        output_path
    )

    required_columns = {
        "TRANSACTION_ID",
        "TX_FRAUD",
        "fraud_probability",
    }

    missing_columns = (
        required_columns - set(saved_predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "E01 prediction artifact is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(saved_predictions) != expected_rows:
        raise ValueError(
            "E01 prediction artifact row count does not match "
            f"the expected validation row count of {expected_rows}."
        )

    if not (
        saved_predictions["fraud_probability"].between(
            0,
            1,
        ).all()
    ):
        raise ValueError(
            "E01 prediction artifact contains probabilities "
            "outside the range [0, 1]."
        )

    print(
    "Prediction artifact validation passed."
    )


def main() -> None:
    """
    Run Phase 3.2 experiment E01.

    The experiment trains Logistic Regression without an explicit
    class-imbalance adjustment and generates fraud probabilities
    for the approved validation dataset.
    """
    print("=" * 70)
    print("PHASE 3.2 - EXPERIMENT E01")
    print("=" * 70)

    print("\nLoading Phase 3.1 feature datasets...")

    train_features, validation_features = load_feature_data()

    print(f"Training dataset shape: {train_features.shape}")
    print(
        "Validation dataset shape: "
        f"{validation_features.shape}"
    )

    print("\nValidating model inputs...")

    validate_model_inputs(
        train_features,
        validation_features,
    )

    print("Model input validation passed.")

    print("\nPreparing X and y...")

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = prepare_model_data(
        train_features,
        validation_features,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_validation shape: {X_validation.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_validation shape: {y_validation.shape}")

    print("\nModel features:")
    for feature in MODEL_FEATURES:
        print(f"  - {feature}")

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nValidation target distribution:")
    print(y_validation.value_counts().sort_index())

    print("\nBuilding E01 model...")

    model = build_e01_model()

    print(
        "\nE01 configuration:"
        "\n  Model: Logistic Regression"
        "\n  Imbalance strategy: None"
        f"\n  Random state: {RANDOM_STATE}"
    )

    print("\nTraining E01...")

    model.fit(
        X_train,
        y_train,
    )

    print("E01 training completed.")

    print("\nGenerating validation fraud probabilities...")

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    print(
        "Number of validation probabilities: "
        f"{len(validation_probabilities)}"
    )

    print(
        "Minimum probability: "
        f"{validation_probabilities.min():.6f}"
    )

    print(
        "Maximum probability: "
        f"{validation_probabilities.max():.6f}"
    )

    # Basic probability sanity checks.
    if len(validation_probabilities) != len(X_validation):
        raise ValueError(
            "Number of validation probabilities does not match "
            "the number of validation rows."
        )

    if not (
        (validation_probabilities >= 0).all()
        and (validation_probabilities <= 1).all()
    ):
        raise ValueError(
            "Validation probabilities contain values outside "
            "the range [0, 1]."
        )

    print("\nE01 probability sanity checks passed.")
    save_validation_predictions(
        validation_features=validation_features,
        validation_probabilities=validation_probabilities,
        output_path=E01_OUTPUT_PATH,
    )
    validate_saved_predictions(
        expected_rows=len(validation_features),
        output_path=E01_OUTPUT_PATH,
    )
    print("\nPhase 3.2 E01 completed successfully.")

    print("\nBuilding E02 model...")
    e02_model = build_e02_model()

    print(
        "\nE02 configuration:"
        "\n  Model: Logistic Regression"
        "\n  Imbalance strategy: class_weight='balanced'"
        f"\n  Random state: {RANDOM_STATE}"
    )

    print("\nTraining E02...")

    e02_model.fit(
        X_train,
        y_train,
    )

    print("E02 training completed.")

    print("\nGenerating E02 validation fraud probabilities...")

    e02_validation_probabilities = e02_model.predict_proba(
        X_validation
    )[:, 1]

    print(
        "Number of E02 validation probabilities: "
        f"{len(e02_validation_probabilities)}"
    )

    print(
        "Minimum E02 probability: "
        f"{e02_validation_probabilities.min():.6f}"
    )

    print(
        "Maximum E02 probability: "
        f"{e02_validation_probabilities.max():.6f}"
    )

    if len(e02_validation_probabilities) != len(X_validation):
        raise ValueError(
            "Number of E02 validation probabilities does not match "
            "the number of validation rows."
        )

    if not (
        (e02_validation_probabilities >= 0).all()
        and (e02_validation_probabilities <= 1).all()
    ):
        raise ValueError(
            "E02 validation probabilities contain values outside "
            "the range [0, 1]."
        )

    print("\nE02 probability sanity checks passed.")

    save_validation_predictions(
        validation_features=validation_features,
        validation_probabilities=e02_validation_probabilities,
        output_path=E02_OUTPUT_PATH,
    )

    validate_saved_predictions(
        expected_rows=len(validation_features),
        output_path=E02_OUTPUT_PATH,
    )

    print("\nPhase 3.2 E02 completed successfully.")


if __name__ == "__main__":
    main()
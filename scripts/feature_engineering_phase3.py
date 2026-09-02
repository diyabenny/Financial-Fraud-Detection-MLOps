"""
Phase 3.1 - Feature Engineering.

Creates reproducible transaction-level features from the validated
Phase 2.5 training and validation datasets.

Model features:
    - TX_AMOUNT
    - AMOUNT_LOG
    - TX_HOUR
    - TX_DAY_OF_WEEK
    - TX_IS_WEEKEND

The target TX_FRAUD and leakage-sensitive TX_FRAUD_SCENARIO are excluded
from the model feature set.

Identifier columns are retained in the output for traceability but are
not used as model features.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------

TRAIN_INPUT_PATH = Path("data/processed/splits/train.csv")
VALIDATION_INPUT_PATH = Path("data/processed/splits/validation.csv")


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("data/phase3")

TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train_features.csv"
VALIDATION_OUTPUT_PATH = OUTPUT_DIR / "validation_features.csv"


# ---------------------------------------------------------------------------
# Source columns expected from Phase 2.5
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
}


# ---------------------------------------------------------------------------
# Model feature contract
# ---------------------------------------------------------------------------

MODEL_FEATURES = [
    "TX_AMOUNT",
    "AMOUNT_LOG",
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
    "TX_IS_WEEKEND",
]


def validate_input_columns(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Validate that the input dataset contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction dataset.
    dataset_name : str
        Name used in validation error messages.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def validate_input_values(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Validate important source values before feature generation.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction dataset.
    dataset_name : str
        Name used in validation error messages.

    Raises
    ------
    ValueError
        If required values are missing, transaction IDs are duplicated,
        transaction amounts are invalid, or the target contains values
        other than 0 and 1.
    """
    columns_to_check = [
        "TRANSACTION_ID",
        "TX_DATETIME",
        "TX_AMOUNT",
        "TX_FRAUD",
    ]

    missing_counts = df[columns_to_check].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        raise ValueError(
            f"{dataset_name} contains missing required values:\n"
            f"{missing_counts.to_string()}"
        )

    # Every transaction must have a unique identifier.
    if not df["TRANSACTION_ID"].is_unique:
        raise ValueError(
            f"{dataset_name} contains duplicate TRANSACTION_ID values."
        )

    # Transaction amounts must be non-negative before applying log1p().
    if (df["TX_AMOUNT"] < 0).any():
        raise ValueError(
            f"{dataset_name}: TX_AMOUNT contains negative values."
        )

    # TX_FRAUD is the binary target and must contain only 0 and 1.
    if not df["TX_FRAUD"].isin([0, 1]).all():
        raise ValueError(
            f"{dataset_name}: TX_FRAUD contains values other than 0 and 1."
        )


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create deterministic transaction-level features.

    The original TX_DATETIME column is preserved. A temporary parsed
    datetime series is used only for feature extraction.

    Parameters
    ----------
    df : pd.DataFrame
        Validated transaction dataset.

    Returns
    -------
    pd.DataFrame
        Dataset containing original traceability columns, target,
        and generated model features.
    """
    result = df.copy()

    # Parse timestamp without replacing the original TX_DATETIME column.
    dt = pd.to_datetime(
        result["TX_DATETIME"],
        errors="raise",
    )

    # Log-transformed transaction amount.
    #
    # log1p(x) = log(1 + x)
    #
    # This safely handles TX_AMOUNT = 0.
    result["AMOUNT_LOG"] = np.log1p(result["TX_AMOUNT"])

    # Hour of transaction.
    result["TX_HOUR"] = dt.dt.hour

    # Monday = 0, Sunday = 6.
    result["TX_DAY_OF_WEEK"] = dt.dt.dayofweek

    # Weekend indicator:
    # Saturday = 5
    # Sunday   = 6
    result["TX_IS_WEEKEND"] = (
        result["TX_DAY_OF_WEEK"] >= 5
    ).astype(int)

    return result


def validate_generated_features(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Validate generated Phase 3.1 features.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset.
    dataset_name : str
        Name used in validation error messages.

    Raises
    ------
    ValueError
        If generated features are invalid or inconsistent with
        their source values.
    """
    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{dataset_name} is missing generated features: "
            f"{missing_features}"
        )

    # No missing values are allowed in model features.
    missing_values = df[MODEL_FEATURES].isna().sum()

    if missing_values.any():
        raise ValueError(
            f"{dataset_name} contains missing feature values:\n"
            f"{missing_values[missing_values > 0].to_string()}"
        )

    # TX_HOUR must be between 0 and 23.
    if not df["TX_HOUR"].between(0, 23).all():
        raise ValueError(
            f"{dataset_name}: TX_HOUR contains invalid values."
        )

    # TX_DAY_OF_WEEK must be between 0 and 6.
    if not df["TX_DAY_OF_WEEK"].between(0, 6).all():
        raise ValueError(
            f"{dataset_name}: TX_DAY_OF_WEEK contains invalid values."
        )

    # Weekend flag must contain only 0 or 1.
    if not df["TX_IS_WEEKEND"].isin([0, 1]).all():
        raise ValueError(
            f"{dataset_name}: TX_IS_WEEKEND contains invalid values."
        )

    # Transaction amount must not be negative.
    if (df["TX_AMOUNT"] < 0).any():
        raise ValueError(
            f"{dataset_name}: TX_AMOUNT contains negative values."
        )

    # Log transformation should not create invalid values.
    if not np.isfinite(df["AMOUNT_LOG"]).all():
        raise ValueError(
            f"{dataset_name}: AMOUNT_LOG contains non-finite values."
        )

    # Verify that AMOUNT_LOG is mathematically consistent with TX_AMOUNT.
    expected_amount_log = np.log1p(df["TX_AMOUNT"])

    if not np.allclose(
        df["AMOUNT_LOG"],
        expected_amount_log,
    ):
        raise ValueError(
            f"{dataset_name}: AMOUNT_LOG does not match TX_AMOUNT."
        )


def validate_identity_preservation(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Verify that feature engineering preserves transaction identity
    and the target variable.

    Parameters
    ----------
    input_df : pd.DataFrame
        Original input dataset.
    output_df : pd.DataFrame
        Feature-engineered dataset.
    dataset_name : str
        Name used in validation error messages.

    Raises
    ------
    ValueError
        If transaction IDs, target values, row count, or row order change.
    """
    # Row count must remain unchanged.
    if len(input_df) != len(output_df):
        raise ValueError(
            f"{dataset_name}: row count changed during feature engineering."
        )

    # Transaction IDs must remain unchanged and in the same order.
    if not output_df["TRANSACTION_ID"].equals(
        input_df["TRANSACTION_ID"]
    ):
        raise ValueError(
            f"{dataset_name}: TRANSACTION_ID values changed during "
            "feature engineering."
        )

    # Target values must remain unchanged and in the same order.
    if not output_df["TX_FRAUD"].equals(
        input_df["TX_FRAUD"]
    ):
        raise ValueError(
            f"{dataset_name}: TX_FRAUD changed during feature engineering."
        )


def validate_saved_output(
    input_df: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Re-read the saved feature artifact and validate it.

    This ensures that the actual CSV written to disk is valid for
    downstream Phase 3.2 consumption.

    Parameters
    ----------
    input_df : pd.DataFrame
        Original input dataset.
    output_path : Path
        Path of the saved feature CSV.
    dataset_name : str
        Name used in validation error messages.

    Returns
    -------
    pd.DataFrame
        Validated saved feature dataset.

    Raises
    ------
    FileNotFoundError
        If the saved output does not exist.
    ValueError
        If the saved artifact fails validation.
    """
    if not output_path.exists():
        raise FileNotFoundError(
            f"{dataset_name}: saved output not found: {output_path}"
        )

    saved_df = pd.read_csv(output_path)

    print(f"Reloaded output shape: {saved_df.shape}")

    # Verify that the saved artifact still contains all required
    # source columns.
    validate_input_columns(
        saved_df,
        dataset_name,
    )

    # Verify source identity and target preservation.
    validate_identity_preservation(
        input_df,
        saved_df,
        dataset_name,
    )

    # Verify generated features in the actual saved artifact.
    validate_generated_features(
        saved_df,
        dataset_name,
    )

    # Verify the target is still binary after saving and reloading.
    if not saved_df["TX_FRAUD"].isin([0, 1]).all():
        raise ValueError(
            f"{dataset_name}: saved TX_FRAUD contains values "
            "other than 0 and 1."
        )

    return saved_df


def process_dataset(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Process one Phase 2.5 dataset and save its Phase 3.1 features.

    Parameters
    ----------
    input_path : Path
        Input CSV path.
    output_path : Path
        Output CSV path.
    dataset_name : str
        Dataset name used in logs and validation.

    Returns
    -------
    pd.DataFrame
        Validated feature-engineered dataset reloaded from disk.
    """
    print("\n" + "=" * 70)
    print(f"PROCESSING {dataset_name.upper()}")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    # Load source dataset.
    df = pd.read_csv(input_path)

    print(f"Input path: {input_path}")
    print(f"Input shape: {df.shape}")

    # Validate source dataset before transformation.
    validate_input_columns(
        df,
        dataset_name,
    )

    validate_input_values(
        df,
        dataset_name,
    )

    # Create deterministic features.
    features_df = create_features(df)

    # Validate generated features in memory.
    validate_generated_features(
        features_df,
        dataset_name,
    )

    # Verify identity and target preservation before saving.
    validate_identity_preservation(
        df,
        features_df,
        dataset_name,
    )

    # Create output directory if it does not already exist.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save feature artifact.
    features_df.to_csv(
        output_path,
        index=False,
    )

    print(f"Output path: {output_path}")
    print(f"Output shape: {features_df.shape}")

    # Re-read and validate the actual artifact written to disk.
    print("\nValidating saved output...")

    saved_features_df = validate_saved_output(
        input_df=df,
        output_path=output_path,
        dataset_name=dataset_name,
    )

    print(
        f"{dataset_name}: saved output validation passed."
    )

    return saved_features_df


def main() -> None:
    """
    Run the complete Phase 3.1 feature-engineering pipeline.
    """
    print("=" * 70)
    print("PHASE 3.1 - FEATURE ENGINEERING")
    print("=" * 70)

    # Process training dataset.
    train_features = process_dataset(
        input_path=TRAIN_INPUT_PATH,
        output_path=TRAIN_OUTPUT_PATH,
        dataset_name="Training dataset",
    )

    # Process validation dataset.
    validation_features = process_dataset(
        input_path=VALIDATION_INPUT_PATH,
        output_path=VALIDATION_OUTPUT_PATH,
        dataset_name="Validation dataset",
    )

    print("\n" + "=" * 70)
    print("FINAL FEATURE CHECK")
    print("=" * 70)

    print("\nModel features:")
    for feature in MODEL_FEATURES:
        print(f"  - {feature}")

    train_model_features = train_features[MODEL_FEATURES]
    validation_model_features = validation_features[MODEL_FEATURES]

    print("\nTraining model feature matrix:")
    print(train_model_features.shape)

    print("\nValidation model feature matrix:")
    print(validation_model_features.shape)

    # Training and validation must have exactly the same
    # feature columns and feature order.
    if list(train_model_features.columns) != list(
        validation_model_features.columns
    ):
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    print(
        "\nTraining and validation feature columns match."
    )

    # Target must not be included in model features.
    if "TX_FRAUD" in MODEL_FEATURES:
        raise ValueError(
            "TX_FRAUD must not be included in MODEL_FEATURES."
        )

    # Fraud scenario must not be included in model features.
    if "TX_FRAUD_SCENARIO" in MODEL_FEATURES:
        raise ValueError(
            "TX_FRAUD_SCENARIO must not be included in MODEL_FEATURES."
        )

    print("TX_FRAUD excluded from model features.")
    print("TX_FRAUD_SCENARIO excluded from model features.")

    print("\nPhase 3.1 feature engineering completed successfully.")


if __name__ == "__main__":
    main()
    
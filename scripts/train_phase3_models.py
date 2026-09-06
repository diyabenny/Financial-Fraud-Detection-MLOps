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
import math
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier



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

E03_OUTPUT_PATH = Path(
    "reports/phase3/experiments/"
    "E03_90_10_validation_predictions.csv"
)

E04_OUTPUT_PATH = Path(
    "reports/phase3/experiments/"
    "E04_80_20_validation_predictions.csv"
)

E05_OUTPUT_PATH = Path(
    "reports/phase3/experiments/"
    "E05_70_30_validation_predictions.csv"
)

E06_OUTPUT_PATH = Path(
    "reports/phase3/experiments/"
    "E06_validation_predictions.csv"
)

E07_OUTPUT_PATH = Path(
    "reports/phase3/experiments/"
    "E07_validation_predictions.csv"
)

EXPERIMENTS = {
    "E03": {
        "sampling_strategy": 10 / 90,
        "output_path": E03_OUTPUT_PATH,
    },
    "E04": {
        "sampling_strategy": 20 / 80,
        "output_path": E04_OUTPUT_PATH,
    },
    "E05": {
        "sampling_strategy": 30 / 70,
        "output_path": E05_OUTPUT_PATH,
    },
}
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

SMOTENC_BASE_FEATURES = [
    "TX_AMOUNT",
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
]

SMOTENC_CATEGORICAL_FEATURES = [
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
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

def resample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTENC to the non-redundant base training features.

    SMOTENC operates only on:
        - TX_AMOUNT
        - TX_HOUR
        - TX_DAY_OF_WEEK

    After resampling, the two deterministic derived features are
    reconstructed:
        - AMOUNT_LOG from TX_AMOUNT
        - TX_IS_WEEKEND from TX_DAY_OF_WEEK

    Parameters
    ----------
    X_train : pd.DataFrame
        Original Phase 3.1 training feature matrix.
    y_train : pd.Series
        Training target.
    sampling_strategy : float
        SMOTENC minority/majority target ratio.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Resampled five-feature training matrix and target.
    """

    # Select only the non-redundant base features for SMOTENC.
    X_base = X_train[SMOTENC_BASE_FEATURES].copy()

    # Convert categorical feature names to column positions.
    categorical_indices = [
        SMOTENC_BASE_FEATURES.index(feature)
        for feature in SMOTENC_CATEGORICAL_FEATURES
    ]

    sampler = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X_base,
        y_train,
    )

    X_resampled = pd.DataFrame(
        X_resampled,
        columns=SMOTENC_BASE_FEATURES,
    )

    y_resampled = pd.Series(
        y_resampled,
        name=TARGET_COLUMN,
    )

    # Build the actual Phase 3.1 day -> weekend mapping from training data.
    day_to_weekend = (
        X_train.groupby("TX_DAY_OF_WEEK")["TX_IS_WEEKEND"]
        .first()
        .astype(int)
        .to_dict()
    )

    # Reconstruct deterministic derived features.
    X_resampled["AMOUNT_LOG"] = (
        X_resampled["TX_AMOUNT"].apply(math.log1p)
    )

    X_resampled["TX_IS_WEEKEND"] = (
        X_resampled["TX_DAY_OF_WEEK"]
        .map(day_to_weekend)
    )

    if X_resampled["TX_IS_WEEKEND"].isna().any():
        raise ValueError(
            "Generated TX_DAY_OF_WEEK contains a value without "
            "a known TX_IS_WEEKEND mapping."
        )

    X_resampled["TX_IS_WEEKEND"] = (
        X_resampled["TX_IS_WEEKEND"].astype(int)
    )

    # -----------------------------------------------------------------------
    # Validate reconstructed and discrete features.
    # -----------------------------------------------------------------------

    # TX_HOUR must remain a valid whole-hour value from 0 through 23.
    if not X_resampled["TX_HOUR"].between(0, 23).all():
        raise ValueError(
            "Resampled TX_HOUR contains values outside the valid range 0–23."
        )

    if X_resampled["TX_HOUR"].mod(1).ne(0).any():
        raise ValueError(
            "Resampled TX_HOUR contains fractional values."
        )

    # TX_DAY_OF_WEEK must remain a valid whole-day value from the
    # observed Phase 3.1 mapping.
    valid_days = set(
        X_train["TX_DAY_OF_WEEK"].dropna().unique()
    )

    generated_days = set(
        X_resampled["TX_DAY_OF_WEEK"].dropna().unique()
    )

    invalid_days = generated_days - valid_days

    if invalid_days:
        raise ValueError(
            "Resampled TX_DAY_OF_WEEK contains invalid values: "
            f"{sorted(invalid_days)}"
        )

    if X_resampled["TX_DAY_OF_WEEK"].mod(1).ne(0).any():
        raise ValueError(
            "Resampled TX_DAY_OF_WEEK contains fractional values."
        )

    # TX_IS_WEEKEND must remain binary.
    if not X_resampled["TX_IS_WEEKEND"].isin([0, 1]).all():
        raise ValueError(
            "Resampled TX_IS_WEEKEND contains values other than 0 or 1."
        )

    # AMOUNT_LOG must exactly follow TX_AMOUNT.
    expected_amount_log = X_resampled["TX_AMOUNT"].apply(
        math.log1p
    )

    amount_log_mismatches = (
        X_resampled["AMOUNT_LOG"]
        .sub(expected_amount_log)
        .abs()
        .gt(1e-10)
        .sum()
    )

    if amount_log_mismatches != 0:
        raise ValueError(
            "AMOUNT_LOG is inconsistent with TX_AMOUNT."
        )

    # TX_IS_WEEKEND must follow the original Phase 3.1 mapping.
    expected_weekend = (
        X_resampled["TX_DAY_OF_WEEK"]
        .map(day_to_weekend)
        .astype(int)
    )

    weekend_mismatches = (
        X_resampled["TX_IS_WEEKEND"]
        != expected_weekend
    ).sum()

    if weekend_mismatches != 0:
        raise ValueError(
            "TX_IS_WEEKEND is inconsistent with TX_DAY_OF_WEEK."
        )

    # Restore the final approved five-feature order.
    X_resampled = X_resampled[MODEL_FEATURES]

    return X_resampled, y_resampled
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

def build_e03_model() -> Pipeline:
    """
    Build the Logistic Regression pipeline used for E03-E05.

    The same model configuration is used for all oversampling
    experiments. Only the SMOTENC sampling ratio changes.
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

def build_e06_model() -> RandomForestClassifier:
    """
    Build the E06 Random Forest model.

    E06 uses class weighting to give greater importance
    to the minority fraud class.

    Returns
    -------
    RandomForestClassifier
        Configured Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model

def calculate_scale_pos_weight(
    y_train: pd.Series,
) -> float:
    """
    Calculate XGBoost class-imbalance weight from the training target.

    The ratio is calculated as:

        negative_count / positive_count

    Only the training target is used.

    Parameters
    ----------
    y_train : pd.Series
        Training target containing TX_FRAUD values.

    Returns
    -------
    float
        Training-derived scale_pos_weight.

    Raises
    ------
    ValueError
        If no positive fraud samples are present.
    """

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError(
            "Training target contains no positive fraud samples."
        )

    scale_pos_weight = (
        negative_count / positive_count
    )

    return float(scale_pos_weight)

def build_e07_model(
    scale_pos_weight: float,
) -> XGBClassifier:
    """
    Build the E07 XGBoost classifier.

    Parameters
    ----------
    scale_pos_weight : float
        Class-imbalance weight calculated from the training target.

    Returns
    -------
    XGBClassifier
        Configured XGBoost model.
    """

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
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

def run_oversampling_experiment(
    experiment_id: str,
    sampling_strategy: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    validation_features: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Run one Phase 3.2 synthetic oversampling experiment.

    The oversampling step is applied only to the training data.
    Validation data remains untouched.
    """

    print("\n" + "=" * 70)
    print(f"PHASE 3.2 - EXPERIMENT {experiment_id}")
    print("=" * 70)

    print(f"\nSampling strategy: {sampling_strategy:.6f}")

    print("\nOriginal training class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nResampling training data...")

    X_resampled, y_resampled = resample_training_data(
        X_train=X_train,
        y_train=y_train,
        sampling_strategy=sampling_strategy,
    )

    print("\nResampled training class distribution:")
    print(y_resampled.value_counts().sort_index())

    print(
        "\nResampled training shape: "
        f"{X_resampled.shape}"
    )

    print("\nBuilding model...")

    model = build_e03_model()

    print("\nTraining model...")

    model.fit(
        X_resampled,
        y_resampled,
    )

    print(
        f"{experiment_id} training completed."
    )

    print(
        "\nGenerating validation fraud probabilities..."
    )

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

    if len(validation_probabilities) != len(
        X_validation
    ):
        raise ValueError(
            f"{experiment_id} validation probability count "
            "does not match the validation row count."
        )

    if not (
        (validation_probabilities >= 0).all()
        and (validation_probabilities <= 1).all()
    ):
        raise ValueError(
            f"{experiment_id} validation probabilities "
            "contain values outside [0, 1]."
        )

    print(
        f"\n{experiment_id} probability sanity checks passed."
    )

    save_validation_predictions(
        validation_features=validation_features,
        validation_probabilities=validation_probabilities,
        output_path=output_path,
    )

    validate_saved_predictions(
        expected_rows=len(validation_features),
        output_path=output_path,
    )

    print(
        f"\nPhase 3.2 {experiment_id} completed successfully."
    )

def run_e06_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    validation_features: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Run Phase 3.2 experiment E06 using Random Forest.

    E06 uses the original imbalanced training distribution with
    class weighting. No SMOTENC or other resampling is applied.

    Parameters
    ----------
    X_train : pd.DataFrame
        Phase 3.1 training features.

    y_train : pd.Series
        Phase 3.1 training target.

    X_validation : pd.DataFrame
        Untouched validation features.

    validation_features : pd.DataFrame
        Validation dataset containing TRANSACTION_ID and TX_FRAUD.

    output_path : Path
        Path for the E06 validation prediction artifact.
    """

    print("\n" + "=" * 70)
    print("PHASE 3.2 - EXPERIMENT E06")
    print("=" * 70)

    print("\nE06 configuration:")
    print("  Model: Random Forest")
    print("  Imbalance strategy: class_weight='balanced'")
    print("  Number of trees: 300")
    print(f"  Random state: {RANDOM_STATE}")
    print("  Feature scaling: None")
    print("  SMOTENC: None")

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTraining E06...")

    model = build_e06_model()

    model.fit(
        X_train,
        y_train,
    )

    print("E06 training completed.")

    print("\nGenerating E06 validation fraud probabilities...")

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    print(
        "Number of E06 validation probabilities: "
        f"{len(validation_probabilities)}"
    )

    print(
        "Minimum E06 probability: "
        f"{validation_probabilities.min():.6f}"
    )

    print(
        "Maximum E06 probability: "
        f"{validation_probabilities.max():.6f}"
    )

    if len(validation_probabilities) != len(
        X_validation
    ):
        raise ValueError(
            "Number of E06 validation probabilities does not "
            "match the validation dataset length."
        )

    if not (
        (validation_probabilities >= 0).all()
        and (validation_probabilities <= 1).all()
    ):
        raise ValueError(
            "E06 validation probabilities contain values "
            "outside the range [0, 1]."
        )

    print("\nE06 probability sanity checks passed.")

    save_validation_predictions(
        validation_features=validation_features,
        validation_probabilities=validation_probabilities,
        output_path=output_path,
    )

    validate_saved_predictions(
        expected_rows=len(validation_features),
        output_path=output_path,
    )

    print("\nPhase 3.2 E06 completed successfully.")

def run_e07_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    validation_features: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Run Phase 3.2 experiment E07 using XGBoost.

    E07 uses the original imbalanced training distribution and
    a training-derived scale_pos_weight. No SMOTENC or resampling
    is applied.

    Parameters
    ----------
    X_train : pd.DataFrame
        Phase 3.1 training features.

    y_train : pd.Series
        Phase 3.1 training target.

    X_validation : pd.DataFrame
        Untouched validation features.

    validation_features : pd.DataFrame
        Validation dataset containing TRANSACTION_ID and TX_FRAUD.

    output_path : Path
        Path for the E07 validation prediction artifact.
    """

    print("\n" + "=" * 70)
    print("PHASE 3.2 - EXPERIMENT E07")
    print("=" * 70)

    print("\nE07 configuration:")
    print("  Model: XGBoost")
    print("  Imbalance strategy: scale_pos_weight")
    print("  SMOTENC: None")
    print("  Feature scaling: None")
    print("  Number of trees: 300")
    print("  Max depth: 6")
    print("  Learning rate: 0.05")
    print(f"  Random state: {RANDOM_STATE}")
    print("  Evaluation metric: logloss")

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nCalculating scale_pos_weight...")

    scale_pos_weight = calculate_scale_pos_weight(
        y_train
    )

    print(
        "Training-derived scale_pos_weight: "
        f"{scale_pos_weight:.12f}"
    )

    print("\nTraining E07...")

    model = build_e07_model(
        scale_pos_weight=scale_pos_weight
    )

    model.fit(
        X_train,
        y_train,
    )

    print("E07 training completed.")

    print(
        "\nGenerating E07 validation fraud probabilities..."
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    print(
        "Number of E07 validation probabilities: "
        f"{len(validation_probabilities)}"
    )

    print(
        "Minimum E07 probability: "
        f"{validation_probabilities.min():.6f}"
    )

    print(
        "Maximum E07 probability: "
        f"{validation_probabilities.max():.6f}"
    )

    if len(validation_probabilities) != len(
        X_validation
    ):
        raise ValueError(
            "Number of E07 validation probabilities does not "
            "match the validation dataset length."
        )

    if not (
        (validation_probabilities >= 0).all()
        and (validation_probabilities <= 1).all()
    ):
        raise ValueError(
            "E07 validation probabilities contain values "
            "outside the range [0, 1]."
        )

    print("\nE07 probability sanity checks passed.")

    save_validation_predictions(
        validation_features=validation_features,
        validation_probabilities=validation_probabilities,
        output_path=output_path,
    )

    validate_saved_predictions(
        expected_rows=len(validation_features),
        output_path=output_path,
    )

    print("\nPhase 3.2 E07 completed successfully.")
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

    # -----------------------------------------------------------------------
    # E03-E05: SMOTENC oversampling experiments
    # -----------------------------------------------------------------------

    for experiment_id, config in EXPERIMENTS.items():
        run_oversampling_experiment(
            experiment_id=experiment_id,
            sampling_strategy=config["sampling_strategy"],
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            validation_features=validation_features,
            output_path=config["output_path"],
        )

if __name__ == "__main__":
    main()
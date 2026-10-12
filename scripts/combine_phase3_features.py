import os
import pandas as pd


# ============================================================
# INPUT FILES
# ============================================================

TRAIN_BASE_PATH = "data/phase3/train_features.csv"
VALIDATION_BASE_PATH = "data/phase3/validation_features.csv"

TRAIN_BEHAVIOR_PATH = "data/phase3/train_behavioral_features.csv"
VALIDATION_BEHAVIOR_PATH = "data/phase3/validation_behavioral_features.csv"


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_DIR = "data/phase3"

TRAIN_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "train_combined_features.csv",
)

VALIDATION_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "validation_combined_features.csv",
)


# ============================================================
# FINAL MODEL FEATURES
# ============================================================

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
# TRACEABILITY / NON-MODEL COLUMNS
# ============================================================

TRACEABILITY_COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading Phase 3.1 features...")

    train_base = pd.read_csv(TRAIN_BASE_PATH)
    validation_base = pd.read_csv(VALIDATION_BASE_PATH)

    print("Loading behavioral features...")

    train_behavior = pd.read_csv(TRAIN_BEHAVIOR_PATH)
    validation_behavior = pd.read_csv(VALIDATION_BEHAVIOR_PATH)

    return (
        train_base,
        validation_base,
        train_behavior,
        validation_behavior,
    )


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

def validate_input_data(
    train_base,
    validation_base,
    train_behavior,
    validation_behavior,
):

    print("\n" + "=" * 70)
    print("INPUT VALIDATION")
    print("=" * 70)

    print("\nTraining base shape:", train_base.shape)
    print("Training behavioral shape:", train_behavior.shape)

    print("\nValidation base shape:", validation_base.shape)
    print("Validation behavioral shape:", validation_behavior.shape)

    # Check transaction IDs.
    assert train_base["TRANSACTION_ID"].is_unique
    assert train_behavior["TRANSACTION_ID"].is_unique

    assert validation_base["TRANSACTION_ID"].is_unique
    assert validation_behavior["TRANSACTION_ID"].is_unique

    # Check that both datasets contain exactly the same transaction IDs.
    assert set(train_base["TRANSACTION_ID"]) == set(
        train_behavior["TRANSACTION_ID"]
    )

    assert set(validation_base["TRANSACTION_ID"]) == set(
        validation_behavior["TRANSACTION_ID"]
    )

    print("\nTraining transaction IDs match.")
    print("Validation transaction IDs match.")

    # Check expected row counts.
    assert len(train_base) == len(train_behavior)
    assert len(validation_base) == len(validation_behavior)

    print("\nInput row counts match.")


# ============================================================
# COMBINE FEATURES
# ============================================================

def combine_features(base_df, behavioral_df):

    # Keep only the behavioral features and transaction ID.
    behavioral_subset = behavioral_df[
    [
        "TRANSACTION_ID",
        "TERMINAL_TX_COUNT_1D",
        "TERMINAL_TX_COUNT_7D",
        "TERMINAL_AVG_AMOUNT_7D",
        "CUSTOMER_TX_COUNT_1D",
        "CUSTOMER_TX_COUNT_7D",
    ]
].copy()

    # The behavioral file contains the original source columns too.
    # Only behavioral features are taken from it here.

    combined_df = base_df.merge(
        behavioral_subset,
        on="TRANSACTION_ID",
        how="inner",
        validate="one_to_one",
    )

    return combined_df


# ============================================================
# VALIDATE FINAL DATASET
# ============================================================

def validate_combined_data(
    combined_df,
    expected_rows,
    dataset_name,
):

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} COMBINED DATA VALIDATION")
    print("=" * 70)

    print("\nShape:", combined_df.shape)

    # Row count must remain unchanged.
    assert len(combined_df) == expected_rows

    # Transaction ID must remain unique.
    assert combined_df["TRANSACTION_ID"].is_unique

    # All model features must exist.
    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in combined_df.columns
    ]

    assert not missing_features, (
        f"Missing model features: {missing_features}"
    )

    # Check missing values.
    missing_values = (
        combined_df[MODEL_FEATURES]
        .isna()
        .sum()
    )

    print("\nMissing model feature values:")
    print(missing_values)

    assert missing_values.sum() == 0

    # Check finite numerical values.
    assert (
        combined_df[MODEL_FEATURES]
        .apply(lambda column: column.map(pd.notna).all())
        .all()
    )

    # Check behavioral features are non-negative.
    behavioral_features = [
        "TERMINAL_TX_COUNT_1D",
        "TERMINAL_TX_COUNT_7D",
        "TERMINAL_AVG_AMOUNT_7D",
        "CUSTOMER_TX_COUNT_1D",
        "CUSTOMER_TX_COUNT_7D",
    ]

    for feature in behavioral_features:

        assert (
            combined_df[feature] >= 0
        ).all()

    # Check AMOUNT_LOG relationship.
    expected_amount_log = (
        __import__("numpy").log1p(
            combined_df["TX_AMOUNT"]
        )
    )

    assert (
        (combined_df["AMOUNT_LOG"] - expected_amount_log)
        .abs()
        .max()
        < 1e-10
    )

    print("\nAMOUNT_LOG consistency check passed.")

    # Check chronological order.
    combined_df["TX_DATETIME"] = pd.to_datetime(
        combined_df["TX_DATETIME"],
        errors="raise",
    )

    assert combined_df["TX_DATETIME"].is_monotonic_increasing

    print("Chronological order check passed.")

    # Print model features.
    print("\nFinal model features:")
    for feature in MODEL_FEATURES:
        print(" -", feature)

    print("\nCombined dataset validation passed.")


# ============================================================
# TRAIN / VALIDATION BOUNDARY CHECK
# ============================================================

def validate_train_validation_boundary(
    train_df,
    validation_df,
):

    print("\n" + "=" * 70)
    print("TRAIN / VALIDATION BOUNDARY VALIDATION")
    print("=" * 70)

    train_end = train_df["TX_DATETIME"].max()
    validation_start = validation_df["TX_DATETIME"].min()

    print("\nTraining end:", train_end)
    print("Validation start:", validation_start)

    assert train_end < validation_start

    train_ids = set(train_df["TRANSACTION_ID"])
    validation_ids = set(validation_df["TRANSACTION_ID"])

    overlap = train_ids.intersection(validation_ids)

    print("\nTrain/validation ID overlap:", len(overlap))

    assert len(overlap) == 0

    print("\nTrain/validation boundary validation passed.")


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    train_df,
    validation_df,
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    train_df.to_csv(
        TRAIN_OUTPUT_PATH,
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)

    print("\nTraining:")
    print(TRAIN_OUTPUT_PATH)

    print("\nValidation:")
    print(VALIDATION_OUTPUT_PATH)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3 - COMBINE FINAL CANDIDATE FEATURES")
    print("=" * 70)

    (
        train_base,
        validation_base,
        train_behavior,
        validation_behavior,
    ) = load_data()

    validate_input_data(
        train_base,
        validation_base,
        train_behavior,
        validation_behavior,
    )

    print("\nCombining training features...")

    train_combined = combine_features(
        train_base,
        train_behavior,
    )

    print("Combining validation features...")

    validation_combined = combine_features(
        validation_base,
        validation_behavior,
    )

    # Sort chronologically.
    train_combined = train_combined.sort_values(
        "TX_DATETIME"
    ).reset_index(drop=True)

    validation_combined = validation_combined.sort_values(
        "TX_DATETIME"
    ).reset_index(drop=True)

    validate_combined_data(
        train_combined,
        len(train_base),
        "training",
    )

    validate_combined_data(
        validation_combined,
        len(validation_base),
        "validation",
    )

    validate_train_validation_boundary(
        train_combined,
        validation_combined,
    )

    save_outputs(
        train_combined,
        validation_combined,
    )

    print("\n" + "=" * 70)
    print("PHASE 3 FEATURE COMBINATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
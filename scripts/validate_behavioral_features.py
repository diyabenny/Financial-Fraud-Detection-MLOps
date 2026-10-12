import numpy as np
import pandas as pd


# ============================================================
# INPUT FILES
# ============================================================

TRAIN_PATH = "data/phase3/train_behavioral_features.csv"
VALIDATION_PATH = "data/phase3/validation_behavioral_features.csv"


# ============================================================
# FEATURES
# ============================================================

BEHAVIORAL_FEATURES = [
    "TERMINAL_TX_COUNT_1D",
    "TERMINAL_TX_COUNT_7D",
    "TERMINAL_AVG_AMOUNT_7D",
    "CUSTOMER_TX_COUNT_1D",
    "CUSTOMER_TX_COUNT_7D",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load generated behavioral feature datasets."""

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    train_df["TX_DATETIME"] = pd.to_datetime(
        train_df["TX_DATETIME"],
        errors="raise",
    )

    validation_df["TX_DATETIME"] = pd.to_datetime(
        validation_df["TX_DATETIME"],
        errors="raise",
    )

    return train_df, validation_df


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_basic_structure(train_df, validation_df):

    print("\n" + "=" * 70)
    print("1. BASIC STRUCTURE VALIDATION")
    print("=" * 70)

    print("\nTraining shape:", train_df.shape)
    print("Validation shape:", validation_df.shape)

    print("\nTraining columns:")
    print(list(train_df.columns))

    print("\nValidation columns:")
    print(list(validation_df.columns))

    # Required behavioral features must exist.
    for feature in BEHAVIORAL_FEATURES:

        assert feature in train_df.columns, (
            f"{feature} missing from training data."
        )

        assert feature in validation_df.columns, (
            f"{feature} missing from validation data."
        )

    print("\nAll behavioral features are present.")


# ============================================================
# MISSING VALUES
# ============================================================

def validate_missing_values(train_df, validation_df):

    print("\n" + "=" * 70)
    print("2. MISSING VALUE CHECK")
    print("=" * 70)

    train_missing = (
        train_df[BEHAVIORAL_FEATURES]
        .isna()
        .sum()
    )

    validation_missing = (
        validation_df[BEHAVIORAL_FEATURES]
        .isna()
        .sum()
    )

    print("\nTraining missing values:")
    print(train_missing)

    print("\nValidation missing values:")
    print(validation_missing)

    assert train_missing.sum() == 0
    assert validation_missing.sum() == 0

    print("\nMissing-value check passed.")


# ============================================================
# DUPLICATE TRANSACTION IDS
# ============================================================

def validate_duplicate_ids(train_df, validation_df):

    print("\n" + "=" * 70)
    print("3. DUPLICATE TRANSACTION ID CHECK")
    print("=" * 70)

    train_duplicates = (
        train_df["TRANSACTION_ID"]
        .duplicated()
        .sum()
    )

    validation_duplicates = (
        validation_df["TRANSACTION_ID"]
        .duplicated()
        .sum()
    )

    overlap = len(
        set(train_df["TRANSACTION_ID"])
        .intersection(
            set(validation_df["TRANSACTION_ID"])
        )
    )

    print("\nTraining duplicate IDs:", train_duplicates)
    print("Validation duplicate IDs:", validation_duplicates)
    print("Train/validation ID overlap:", overlap)

    assert train_duplicates == 0
    assert validation_duplicates == 0
    assert overlap == 0

    print("\nTransaction ID validation passed.")


# ============================================================
# DATETIME VALIDATION
# ============================================================

def validate_datetime(train_df, validation_df):

    print("\n" + "=" * 70)
    print("4. DATETIME VALIDATION")
    print("=" * 70)

    print("\nTraining:")
    print(
        train_df["TX_DATETIME"].min(),
        "to",
        train_df["TX_DATETIME"].max(),
    )

    print("\nValidation:")
    print(
        validation_df["TX_DATETIME"].min(),
        "to",
        validation_df["TX_DATETIME"].max(),
    )

    assert (
        train_df["TX_DATETIME"].max()
        < validation_df["TX_DATETIME"].min()
    )

    print(
        "\nTraining occurs completely before validation."
    )


# ============================================================
# FEATURE RANGE VALIDATION
# ============================================================

def validate_feature_ranges(train_df, validation_df):

    print("\n" + "=" * 70)
    print("5. FEATURE RANGE VALIDATION")
    print("=" * 70)

    combined = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    for feature in BEHAVIORAL_FEATURES:

        minimum = combined[feature].min()
        maximum = combined[feature].max()

        print(
            f"\n{feature}"
        )
        print(f"  Minimum: {minimum}")
        print(f"  Maximum: {maximum}")

        assert np.isfinite(
            combined[feature].to_numpy()
        ).all()

        assert (
            combined[feature] >= 0
        ).all()

    print("\nFeature range validation passed.")


# ============================================================
# TRAIN VS VALIDATION DISTRIBUTION
# ============================================================

def compare_distributions(train_df, validation_df):

    print("\n" + "=" * 70)
    print("6. TRAINING VS VALIDATION DISTRIBUTION")
    print("=" * 70)

    rows = []

    for feature in BEHAVIORAL_FEATURES:

        train_mean = train_df[feature].mean()
        validation_mean = validation_df[feature].mean()

        train_median = train_df[feature].median()
        validation_median = validation_df[feature].median()

        rows.append(
            {
                "Feature": feature,
                "Train Mean": train_mean,
                "Validation Mean": validation_mean,
                "Train Median": train_median,
                "Validation Median": validation_median,
            }
        )

    comparison = pd.DataFrame(rows)

    print()
    print(comparison.to_string(index=False))


# ============================================================
# FRAUD RELATIONSHIP
# ============================================================

def analyze_fraud_relationship(train_df, validation_df):

    print("\n" + "=" * 70)
    print("7. RELATIONSHIP WITH FRAUD")
    print("=" * 70)

    print("\nTraining feature means by TX_FRAUD:")

    print(
        train_df
        .groupby("TX_FRAUD")[BEHAVIORAL_FEATURES]
        .mean()
        .to_string()
    )

    print("\nValidation feature means by TX_FRAUD:")

    print(
        validation_df
        .groupby("TX_FRAUD")[BEHAVIORAL_FEATURES]
        .mean()
        .to_string()
    )


# ============================================================
# FRAUD SCENARIO ANALYSIS
# ============================================================

def analyze_fraud_scenarios(validation_df):

    print("\n" + "=" * 70)
    print("8. VALIDATION FRAUD SCENARIO ANALYSIS")
    print("=" * 70)

    fraud_validation = validation_df[
        validation_df["TX_FRAUD"] == 1
    ].copy()

    print(
        "\nFraud transactions by scenario:"
    )

    print(
        fraud_validation["TX_FRAUD_SCENARIO"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nBehavioral feature means by fraud scenario:"
    )

    print(
        fraud_validation
        .groupby("TX_FRAUD_SCENARIO")[BEHAVIORAL_FEATURES]
        .mean()
        .to_string()
    )


# ============================================================
# ZERO-HISTORY ANALYSIS
# ============================================================

def analyze_zero_history(train_df, validation_df):

    print("\n" + "=" * 70)
    print("9. ZERO-HISTORY ANALYSIS")
    print("=" * 70)

    combined = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    for feature in BEHAVIORAL_FEATURES:

        zero_count = (
            combined[feature] == 0
        ).sum()

        zero_percentage = (
            zero_count / len(combined)
        ) * 100

        print(
            f"\n{feature}:"
        )
        print(
            f"  Zero values: {zero_count}"
        )
        print(
            f"  Percentage: {zero_percentage:.2f}%"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(train_df, validation_df):

    print("\n" + "=" * 70)
    print("10. FINAL VALIDATION SUMMARY")
    print("=" * 70)

    checks = {
        "Training rows preserved": len(train_df) == 230295,
        "Validation rows preserved": len(validation_df) == 57767,
        "No missing behavioral features": (
            train_df[BEHAVIORAL_FEATURES].isna().sum().sum() == 0
            and
            validation_df[BEHAVIORAL_FEATURES].isna().sum().sum() == 0
        ),
        "No duplicate training IDs": (
            train_df["TRANSACTION_ID"].duplicated().sum() == 0
        ),
        "No duplicate validation IDs": (
            validation_df["TRANSACTION_ID"].duplicated().sum() == 0
        ),
        "No train/validation ID overlap": (
            len(
                set(train_df["TRANSACTION_ID"])
                .intersection(
                    set(validation_df["TRANSACTION_ID"])
                )
            ) == 0
        ),
        "Chronological split preserved": (
            train_df["TX_DATETIME"].max()
            < validation_df["TX_DATETIME"].min()
        ),
    }

    for check, result in checks.items():

        status = "PASS" if result else "FAIL"

        print(
            f"{status}: {check}"
        )

    assert all(checks.values())

    print(
        "\nAll behavioral feature validation checks passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3 - BEHAVIORAL FEATURE VALIDATION")
    print("=" * 70)

    train_df, validation_df = load_data()

    validate_basic_structure(
        train_df,
        validation_df,
    )

    validate_missing_values(
        train_df,
        validation_df,
    )

    validate_duplicate_ids(
        train_df,
        validation_df,
    )

    validate_datetime(
        train_df,
        validation_df,
    )

    validate_feature_ranges(
        train_df,
        validation_df,
    )

    compare_distributions(
        train_df,
        validation_df,
    )

    analyze_fraud_relationship(
        train_df,
        validation_df,
    )

    analyze_fraud_scenarios(
        validation_df,
    )

    analyze_zero_history(
        train_df,
        validation_df,
    )

    final_summary(
        train_df,
        validation_df,
    )

    print("\n" + "=" * 70)
    print("BEHAVIORAL FEATURE VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
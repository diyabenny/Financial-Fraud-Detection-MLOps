import os
import numpy as np
import pandas as pd


# ============================================================
# INPUT FILES
# ============================================================

TRAIN_PATH = "data/processed/splits/train.csv"
VALIDATION_PATH = "data/processed/splits/validation.csv"


# ============================================================
# OUTPUT FILES
# ============================================================

TRAIN_OUTPUT_PATH = "data/phase3/train_behavioral_features.csv"
VALIDATION_OUTPUT_PATH = "data/phase3/validation_behavioral_features.csv"


# ============================================================
# ORIGINAL FEATURES
# ============================================================

ORIGINAL_FEATURES = [
    "TX_AMOUNT",
    "AMOUNT_LOG",
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
    "TX_IS_WEEKEND",
]


# ============================================================
# BEHAVIORAL FEATURES
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
    """Load the approved Phase 2.5 train and validation datasets."""

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
# CALCULATE TERMINAL FEATURES
# ============================================================

def calculate_terminal_features(df):
    """
    Calculate leakage-safe terminal behavioral features.

    For every transaction at time T:
    - only previous transactions with timestamp < T are used
    - the current transaction is excluded
    - future transactions are excluded
    """

    base = df[
        [
            "TRANSACTION_ID",
            "TERMINAL_ID",
            "TX_DATETIME",
            "TX_AMOUNT",
        ]
    ].copy()

    base = base.sort_values(
        ["TERMINAL_ID", "TX_DATETIME"]
    ).reset_index(drop=True)

    results = []

    for terminal_id, group in base.groupby(
        "TERMINAL_ID",
        sort=False,
    ):
        group = group.sort_values("TX_DATETIME").copy()

        times = group["TX_DATETIME"].to_numpy()
        amounts = group["TX_AMOUNT"].to_numpy()

        count_1d = np.zeros(len(group), dtype=float)
        count_7d = np.zeros(len(group), dtype=float)
        avg_7d = np.zeros(len(group), dtype=float)

        for i in range(len(group)):
            current_time = times[i]

            start_1d = current_time - np.timedelta64(1, "D")
            start_7d = current_time - np.timedelta64(7, "D")

            # Strictly earlier than current transaction.
            previous_mask_1d = (
                (times < current_time)
                & (times >= start_1d)
            )

            previous_mask_7d = (
                (times < current_time)
                & (times >= start_7d)
            )

            amounts_1d = amounts[previous_mask_1d]
            amounts_7d = amounts[previous_mask_7d]

            count_1d[i] = len(amounts_1d)
            count_7d[i] = len(amounts_7d)

            if len(amounts_7d) > 0:
                avg_7d[i] = amounts_7d.mean()

        terminal_result = pd.DataFrame(
            {
                "TRANSACTION_ID": group[
                    "TRANSACTION_ID"
                ].values,
                "TERMINAL_TX_COUNT_1D": count_1d,
                "TERMINAL_TX_COUNT_7D": count_7d,
                "TERMINAL_AVG_AMOUNT_7D": avg_7d,
            }
        )

        results.append(terminal_result)

    return pd.concat(
        results,
        ignore_index=True,
    )


# ============================================================
# CALCULATE CUSTOMER FEATURES
# ============================================================

def calculate_customer_features(df):
    """
    Calculate leakage-safe customer behavioral features.

    For every transaction at time T:
    - only previous transactions with timestamp < T are used
    - the current transaction is excluded
    - future transactions are excluded
    """

    base = df[
        [
            "TRANSACTION_ID",
            "CUSTOMER_ID",
            "TX_DATETIME",
        ]
    ].copy()

    base = base.sort_values(
        ["CUSTOMER_ID", "TX_DATETIME"]
    ).reset_index(drop=True)

    results = []

    for customer_id, group in base.groupby(
        "CUSTOMER_ID",
        sort=False,
    ):
        group = group.sort_values("TX_DATETIME").copy()

        times = group["TX_DATETIME"].to_numpy()

        count_1d = np.zeros(len(group), dtype=float)
        count_7d = np.zeros(len(group), dtype=float)

        for i in range(len(group)):
            current_time = times[i]

            start_1d = current_time - np.timedelta64(1, "D")
            start_7d = current_time - np.timedelta64(7, "D")

            previous_mask_1d = (
                (times < current_time)
                & (times >= start_1d)
            )

            previous_mask_7d = (
                (times < current_time)
                & (times >= start_7d)
            )

            count_1d[i] = previous_mask_1d.sum()
            count_7d[i] = previous_mask_7d.sum()

        customer_result = pd.DataFrame(
            {
                "TRANSACTION_ID": group[
                    "TRANSACTION_ID"
                ].values,
                "CUSTOMER_TX_COUNT_1D": count_1d,
                "CUSTOMER_TX_COUNT_7D": count_7d,
            }
        )

        results.append(customer_result)

    return pd.concat(
        results,
        ignore_index=True,
    )


# ============================================================
# CALCULATE ALL BEHAVIORAL FEATURES
# ============================================================

def calculate_behavioral_features(df):
    """Calculate all leakage-safe behavioral features."""

    print("\nCalculating terminal behavioral features...")

    terminal_features = calculate_terminal_features(df)

    print("Terminal features completed.")

    print("\nCalculating customer behavioral features...")

    customer_features = calculate_customer_features(df)

    print("Customer features completed.")

    # Merge using TRANSACTION_ID.
    result = df.merge(
        terminal_features,
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        customer_features,
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )

    # Transactions with no previous history have zero counts.
    count_features = [
        "TERMINAL_TX_COUNT_1D",
        "TERMINAL_TX_COUNT_7D",
        "CUSTOMER_TX_COUNT_1D",
        "CUSTOMER_TX_COUNT_7D",
    ]

    result[count_features] = (
        result[count_features].fillna(0)
    )

    # No previous terminal history = average represented as 0.
    result["TERMINAL_AVG_AMOUNT_7D"] = (
        result["TERMINAL_AVG_AMOUNT_7D"].fillna(0)
    )

    return result


# ============================================================
# VALIDATE BEHAVIORAL FEATURES
# ============================================================

def validate_behavioral_features(df):
    """Validate generated behavioral features."""

    print("\n" + "=" * 70)
    print("BEHAVIORAL FEATURE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = (
        df[BEHAVIORAL_FEATURES].isna().sum()
    )

    print("\nMissing behavioral feature values:")
    print(missing_values)

    # --------------------------------------------------------
    # Duplicate transaction IDs
    # --------------------------------------------------------

    duplicate_ids = (
        df["TRANSACTION_ID"].duplicated().sum()
    )

    print("\nDuplicate TRANSACTION_ID:")
    print(duplicate_ids)

    # --------------------------------------------------------
    # Minimum values
    # --------------------------------------------------------

    print("\nMinimum behavioral feature values:")

    for feature in BEHAVIORAL_FEATURES:
        print(
            f"  {feature}: "
            f"{df[feature].min()}"
        )

    # --------------------------------------------------------
    # Maximum values
    # --------------------------------------------------------

    print("\nMaximum behavioral feature values:")

    for feature in BEHAVIORAL_FEATURES:
        print(
            f"  {feature}: "
            f"{df[feature].max()}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nBehavioral feature summary:")

    print(
        df[BEHAVIORAL_FEATURES].describe()
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert missing_values.sum() == 0, (
        "Behavioral features contain missing values."
    )

    assert duplicate_ids == 0, (
        "Duplicate TRANSACTION_ID found."
    )

    for feature in BEHAVIORAL_FEATURES:

        assert np.isfinite(
            df[feature].to_numpy()
        ).all(), (
            f"Non-finite values found in {feature}."
        )

        assert (
            df[feature] >= 0
        ).all(), (
            f"Negative values found in {feature}."
        )

    print("\nBehavioral feature validation passed.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3 - BEHAVIORAL FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD TRAINING AND VALIDATION
    # --------------------------------------------------------

    train_df, validation_df = load_data()

    print("\nTraining data shape:")
    print(train_df.shape)

    print("\nValidation data shape:")
    print(validation_df.shape)

    print("\nTraining date range:")
    print(
        train_df["TX_DATETIME"].min(),
        "to",
        train_df["TX_DATETIME"].max(),
    )

    print("\nValidation date range:")
    print(
        validation_df["TX_DATETIME"].min(),
        "to",
        validation_df["TX_DATETIME"].max(),
    )

    # --------------------------------------------------------
    # 2. COMBINE TRAIN + VALIDATION
    # --------------------------------------------------------

    combined_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    print("\nCombined April data shape:")
    print(combined_df.shape)

    # --------------------------------------------------------
    # 3. CALCULATE BEHAVIORAL FEATURES
    # --------------------------------------------------------

    combined_df = calculate_behavioral_features(
        combined_df
    )

    # --------------------------------------------------------
    # 4. VALIDATE
    # --------------------------------------------------------

    validate_behavioral_features(
        combined_df
    )

    # --------------------------------------------------------
    # 5. SPLIT BACK INTO TRAIN AND VALIDATION
    # --------------------------------------------------------

    train_ids = set(
        train_df["TRANSACTION_ID"]
    )

    train_features = combined_df[
        combined_df["TRANSACTION_ID"].isin(train_ids)
    ].copy()

    validation_features = combined_df[
        ~combined_df["TRANSACTION_ID"].isin(train_ids)
    ].copy()

    # Restore chronological order.
    train_features = (
        train_features
        .sort_values("TX_DATETIME")
        .reset_index(drop=True)
    )

    validation_features = (
        validation_features
        .sort_values("TX_DATETIME")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 6. CHECK FINAL SHAPES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL FEATURE DATASETS")
    print("=" * 70)

    print("\nTraining feature shape:")
    print(train_features.shape)

    print("\nValidation feature shape:")
    print(validation_features.shape)

    # --------------------------------------------------------
    # 7. CHECK DATE RANGES
    # --------------------------------------------------------

    print("\nTraining date range:")
    print(
        train_features["TX_DATETIME"].min(),
        "to",
        train_features["TX_DATETIME"].max(),
    )

    print("\nValidation date range:")
    print(
        validation_features["TX_DATETIME"].min(),
        "to",
        validation_features["TX_DATETIME"].max(),
    )

    # --------------------------------------------------------
    # 8. CHECK ID OVERLAP
    # --------------------------------------------------------

    train_id_set = set(
        train_features["TRANSACTION_ID"]
    )

    validation_id_set = set(
        validation_features["TRANSACTION_ID"]
    )

    overlap = train_id_set.intersection(
        validation_id_set
    )

    print(
        "\nTrain/validation TRANSACTION_ID overlap:"
    )
    print(len(overlap))

    assert len(overlap) == 0, (
        "Train and validation transaction IDs overlap."
    )

    # --------------------------------------------------------
    # 9. CHECK ROW COUNTS
    # --------------------------------------------------------

    assert len(train_features) == len(train_df), (
        "Training row count changed."
    )

    assert len(validation_features) == len(validation_df), (
        "Validation row count changed."
    )

    print("\nRow-count validation passed.")

    # --------------------------------------------------------
    # 10. DISPLAY SAMPLE
    # --------------------------------------------------------

    display_columns = [
        "TRANSACTION_ID",
        "TX_DATETIME",
        "TERMINAL_ID",
        "CUSTOMER_ID",
        "TX_AMOUNT",
    ] + BEHAVIORAL_FEATURES

    print("\nSample training behavioral features:")

    print(
        train_features[
            display_columns
        ].head(10)
    )

    print("\nSample validation behavioral features:")

    print(
        validation_features[
            display_columns
        ].head(10)
    )

    # --------------------------------------------------------
    # 11. SAVE OUTPUTS
    # --------------------------------------------------------

    os.makedirs(
        "data/phase3",
        exist_ok=True,
    )

    train_features.to_csv(
        TRAIN_OUTPUT_PATH,
        index=False,
    )

    validation_features.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FEATURE FILES SAVED")
    print("=" * 70)

    print("\nTraining features:")
    print(TRAIN_OUTPUT_PATH)

    print("\nValidation features:")
    print(VALIDATION_OUTPUT_PATH)

    print(
        "\nPHASE 3 BEHAVIORAL FEATURE ENGINEERING COMPLETED."
    )


if __name__ == "__main__":
    main()
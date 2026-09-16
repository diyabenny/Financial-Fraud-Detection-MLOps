"""
Prepare leakage-safe final model features for the April 25-30
Phase 4 evaluation checkpoint.
"""

import os

import numpy as np
import pandas as pd


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_PATH = "data/incoming/april/converted_april_file.csv"

OUTPUT_PATH = (
    "data/phase4/april_25_30_evaluation_features.csv"
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

TARGET = "TX_FRAUD"


# ============================================================
# SOURCE COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_FRAUD",
]


def load_april_data():
    """Load and validate the corrected April source dataset.

    Returns:
        The April transaction DataFrame with TX_DATETIME parsed
        as a UTC-aware datetime column.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(INPUT_PATH)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["TX_DATETIME"] = pd.to_datetime(
        df["TX_DATETIME"],
        errors="raise",
        utc=True,
    )

    return df


def create_basic_features(df):
    """Create the four deterministic transaction-time features.

    Parameters:
        df: April transaction DataFrame.

    Returns:
        DataFrame containing the original columns plus the
        four derived features AMOUNT_LOG, TX_HOUR,
        TX_DAY_OF_WEEK, and TX_IS_WEEKEND.
    """
    data = df.copy()

    data["AMOUNT_LOG"] = np.log1p(data["TX_AMOUNT"])

    data["TX_HOUR"] = data["TX_DATETIME"].dt.hour

    data["TX_DAY_OF_WEEK"] = (
        data["TX_DATETIME"].dt.dayofweek
    )

    data["TX_IS_WEEKEND"] = (
        data["TX_DAY_OF_WEEK"] >= 5
    ).astype(int)

    return data


def calculate_terminal_features(df):
    """Calculate leakage-safe terminal behavioral features.

    Only transactions strictly earlier than the current transaction
    are included in the behavioral windows.

    Parameters:
        df: April transaction DataFrame.

    Returns:
        DataFrame containing TRANSACTION_ID and the three terminal
        behavioral features.
    """
    data = df[
        [
            "TRANSACTION_ID",
            "TERMINAL_ID",
            "TX_DATETIME",
            "TX_AMOUNT",
        ]
    ].copy()

    data = data.sort_values(
        [
            "TERMINAL_ID",
            "TX_DATETIME",
            "TRANSACTION_ID",
        ]
    )

    results = []

    for terminal_id, group in data.groupby(
        "TERMINAL_ID",
        sort=False,
    ):
        group = group.reset_index(drop=True)

        times = (
            group["TX_DATETIME"]
            .astype("int64")
            .to_numpy()
        )

        amounts = group["TX_AMOUNT"].to_numpy()

        for i in range(len(group)):
            current_time = times[i]

            start_1d = (
                current_time
                - pd.Timedelta(days=1).value
            )

            start_7d = (
                current_time
                - pd.Timedelta(days=7).value
            )

            previous_1d = (
                (times < current_time)
                & (times >= start_1d)
            )

            previous_7d = (
                (times < current_time)
                & (times >= start_7d)
            )

            count_1d = int(
                previous_1d.sum()
            )

            count_7d = int(
                previous_7d.sum()
            )

            if previous_7d.any():
                avg_amount_7d = float(
                    amounts[previous_7d].mean()
                )
            else:
                avg_amount_7d = 0.0

            results.append(
                {
                    "TRANSACTION_ID": group.loc[
                        i,
                        "TRANSACTION_ID",
                    ],
                    "TERMINAL_TX_COUNT_1D": count_1d,
                    "TERMINAL_TX_COUNT_7D": count_7d,
                    "TERMINAL_AVG_AMOUNT_7D": (
                        avg_amount_7d
                    ),
                }
            )

    return pd.DataFrame(results)


def calculate_customer_features(df):
    """Calculate leakage-safe customer behavioral features.

    Only transactions strictly earlier than the current transaction
    are included in the behavioral windows.

    Parameters:
        df: April transaction DataFrame.

    Returns:
        DataFrame containing TRANSACTION_ID and the two customer
        behavioral features.
    """
    data = df[
        [
            "TRANSACTION_ID",
            "CUSTOMER_ID",
            "TX_DATETIME",
        ]
    ].copy()

    data = data.sort_values(
        [
            "CUSTOMER_ID",
            "TX_DATETIME",
            "TRANSACTION_ID",
        ]
    )

    results = []

    for customer_id, group in data.groupby(
        "CUSTOMER_ID",
        sort=False,
    ):
        group = group.reset_index(drop=True)

        times = (
            group["TX_DATETIME"]
            .astype("int64")
            .to_numpy()
        )

        for i in range(len(group)):
            current_time = times[i]

            start_1d = (
                current_time
                - pd.Timedelta(days=1).value
            )

            start_7d = (
                current_time
                - pd.Timedelta(days=7).value
            )

            previous_1d = (
                (times < current_time)
                & (times >= start_1d)
            )

            previous_7d = (
                (times < current_time)
                & (times >= start_7d)
            )

            count_1d = int(
                previous_1d.sum()
            )

            count_7d = int(
                previous_7d.sum()
            )

            results.append(
                {
                    "TRANSACTION_ID": group.loc[
                        i,
                        "TRANSACTION_ID",
                    ],
                    "CUSTOMER_TX_COUNT_1D": count_1d,
                    "CUSTOMER_TX_COUNT_7D": count_7d,
                }
            )

    return pd.DataFrame(results)


def combine_features(
    df,
    terminal_features,
    customer_features,
):
    """Combine basic, terminal, and customer features.

    Parameters:
        df: April transaction DataFrame containing basic features.
        terminal_features: Terminal behavioral feature DataFrame.
        customer_features: Customer behavioral feature DataFrame.

    Returns:
        DataFrame containing all final model features and the target.

    Raises:
        ValueError: If transaction IDs are duplicated or required
            features are missing.
    """
    data = df.copy()

    data = data.merge(
        terminal_features,
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )

    data = data.merge(
        customer_features,
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )

    behavioral_features = [
        "TERMINAL_TX_COUNT_1D",
        "TERMINAL_TX_COUNT_7D",
        "TERMINAL_AVG_AMOUNT_7D",
        "CUSTOMER_TX_COUNT_1D",
        "CUSTOMER_TX_COUNT_7D",
    ]

    for column in behavioral_features:
        data[column] = data[column].fillna(0)

    missing_features = [
        column
        for column in MODEL_FEATURES
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    if data["TRANSACTION_ID"].duplicated().any():
        raise ValueError(
            "Duplicate TRANSACTION_ID values found."
        )

    return data


def select_evaluation_period(df):
    """Select transactions from April 25 through April 30.

    Parameters:
        df: Fully feature-engineered April transaction DataFrame.

    Returns:
        DataFrame containing only Apr 25-30 transactions.

    Raises:
        ValueError: If the evaluation period is empty.
    """
    data = df.copy()

    evaluation_mask = (
        data["TX_DATETIME"]
        .dt.day
        .between(25, 30)
    )

    evaluation_df = data.loc[
        evaluation_mask
    ].copy()

    if evaluation_df.empty:
        raise ValueError(
            "Apr 25-30 evaluation dataset is empty."
        )

    return evaluation_df


def validate_evaluation_data(df):
    """Validate the Apr 25-30 evaluation feature dataset.

    Parameters:
        df: Apr 25-30 feature-engineered DataFrame.

    Raises:
        ValueError: If required features, targets, IDs, timestamps,
            or feature values fail validation.
    """
    if df["TRANSACTION_ID"].duplicated().any():
        raise ValueError(
            "Duplicate TRANSACTION_ID values found."
        )

    missing_features = [
        column
        for column in MODEL_FEATURES
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Missing TX_FRAUD target values found."
        )

    if df[MODEL_FEATURES].isna().any().any():
        raise ValueError(
            "Missing model feature values found."
        )

    if not (
        df["TX_DATETIME"]
        .dt.day
        .between(25, 30)
        .all()
    ):
        raise ValueError(
            "Evaluation dataset contains dates "
            "outside Apr 25-30."
        )

    if not set(
        df[TARGET].unique()
    ).issubset({0, 1}):
        raise ValueError(
            "TX_FRAUD contains values other than 0 and 1."
        )

    if (df["TX_AMOUNT"] < 0).any():
        raise ValueError(
            "Negative TX_AMOUNT values found."
        )

    if not np.isfinite(
        df[MODEL_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite model feature values found."
        )

    expected_amount_log = np.log1p(
        df["TX_AMOUNT"]
    )

    if not np.allclose(
        df["AMOUNT_LOG"],
        expected_amount_log,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(
            "AMOUNT_LOG does not match "
            "np.log1p(TX_AMOUNT)."
        )

    if not df["TX_HOUR"].between(
        0,
        23,
    ).all():
        raise ValueError(
            "TX_HOUR contains invalid values."
        )

    if not df["TX_DAY_OF_WEEK"].between(
        0,
        6,
    ).all():
        raise ValueError(
            "TX_DAY_OF_WEEK contains invalid values."
        )

    if not df["TX_IS_WEEKEND"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "TX_IS_WEEKEND contains invalid values."
        )

    behavioral_features = [
        "TERMINAL_TX_COUNT_1D",
        "TERMINAL_TX_COUNT_7D",
        "TERMINAL_AVG_AMOUNT_7D",
        "CUSTOMER_TX_COUNT_1D",
        "CUSTOMER_TX_COUNT_7D",
    ]

    if (
        df[behavioral_features] < 0
    ).any().any():
        raise ValueError(
            "Negative behavioral feature values found."
        )

    print(
        "\nEvaluation data validation passed."
    )


def main():
    """Prepare and validate the Apr 25-30 evaluation dataset.

    The full Apr 1-30 dataset is used to calculate leakage-safe
    behavioral features before the Apr 25-30 evaluation period
    is selected.
    """
    print("=" * 70)
    print(
        "PREPARING APRIL 25-30 EVALUATION FEATURES"
    )
    print("=" * 70)

    print(
        "\n[1/6] Loading corrected April source..."
    )

    df = load_april_data()

    print(
        "Rows loaded:",
        len(df),
    )

    print(
        "Date range:",
        df["TX_DATETIME"].min(),
        "to",
        df["TX_DATETIME"].max(),
    )

    print(
        "\n[2/6] Creating basic features..."
    )

    df = create_basic_features(df)

    print(
        "\n[3/6] Calculating terminal "
        "behavioral features..."
    )

    terminal_features = calculate_terminal_features(
        df
    )

    print(
        "\n[4/6] Calculating customer "
        "behavioral features..."
    )

    customer_features = calculate_customer_features(
        df
    )

    print(
        "\n[5/6] Combining all model features..."
    )

    combined_df = combine_features(
        df,
        terminal_features,
        customer_features,
    )

    print(
        "Combined rows:",
        len(combined_df),
    )

    print(
        "\nSelecting Apr 25-30 evaluation period..."
    )

    evaluation_df = select_evaluation_period(
        combined_df
    )

    print(
        "Evaluation rows:",
        len(evaluation_df),
    )

    print(
        "\n[6/6] Validating evaluation dataset..."
    )

    validate_evaluation_data(
        evaluation_df
    )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    evaluation_columns = [
        "TRANSACTION_ID",
        "TX_DATETIME",
        *MODEL_FEATURES,
        TARGET,
    ]

    evaluation_df = evaluation_df[
        evaluation_columns
    ].copy()

    evaluation_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nEvaluation dataset saved to:"
    )

    print(OUTPUT_PATH)

    print(
        "\nFinal shape:",
        evaluation_df.shape,
    )

    print(
        "Fraud count:",
        int(evaluation_df[TARGET].sum()),
    )

    print("\n" + "=" * 70)
    print(
        "APRIL 25-30 EVALUATION DATASET READY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

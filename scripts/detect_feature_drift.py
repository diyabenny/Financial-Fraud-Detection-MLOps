import numpy as np
import pandas as pd


REFERENCE_PATH = (
    "feast_repo/fraud_features/feature_repo/data/train_combined_features.parquet"
)
CURRENT_PATH = "data/phase3/may_processed_features.csv"

EPSILON = 1e-6
DRIFT_THRESHOLD = 0.25
NUMBER_OF_BINS = 10

FEATURE_COLUMNS = [
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


def load_populations():
    reference_df = pd.read_parquet(REFERENCE_PATH)
    current_df = pd.read_csv(CURRENT_PATH)

    return reference_df, current_df


def add_deterministic_features(current_df):
    required_source_columns = ["TX_AMOUNT", "TX_DATETIME"]

    missing_columns = [
        column
        for column in required_source_columns
        if column not in current_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Current population is missing required source columns: {missing_columns}"
        )

    current_df = current_df.copy()

    current_df["AMOUNT_LOG"] = np.log1p(current_df["TX_AMOUNT"])

    timestamps = pd.to_datetime(
        current_df["TX_DATETIME"],
        errors="raise",
    )

    current_df["TX_HOUR"] = timestamps.dt.hour
    current_df["TX_DAY_OF_WEEK"] = timestamps.dt.dayofweek
    current_df["TX_IS_WEEKEND"] = (timestamps.dt.dayofweek >= 5).astype(int)

    return current_df


def validate_features(reference_df, current_df):
    for population_name, dataframe in [
        ("reference", reference_df),
        ("current", current_df),
    ]:
        missing_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                f"{population_name.capitalize()} population is missing "
                f"approved features: {missing_features}"
            )

        null_features = [
            feature
            for feature in FEATURE_COLUMNS
            if dataframe[feature].isna().any()
        ]

        if null_features:
            raise ValueError(
                f"{population_name.capitalize()} population contains null values "
                f"in approved features: {null_features}"
            )


def create_reference_bins(reference_values):
    if reference_values.nunique() < 2:
        raise ValueError(
            "Reference feature has fewer than 2 unique values; "
            "cannot create drift bins."
        )

    quantiles = np.linspace(0, 1, NUMBER_OF_BINS + 1)

    edges = np.quantile(
        reference_values,
        quantiles,
    )

    edges = np.unique(edges)

    if len(edges) < 2:
        raise ValueError(
            "Reference feature has fewer than 2 unique bin edges; "
            "cannot create drift bins."
        )

    return edges


def assign_bins(values, edges):
    bin_indices = np.searchsorted(
        edges,
        values,
        side="right",
    ) - 1

    return np.clip(
        bin_indices,
        0,
        len(edges) - 2,
    )


def calculate_psi(reference_values, current_values, edges):
    reference_bins = assign_bins(reference_values, edges)
    current_bins = assign_bins(current_values, edges)

    number_of_actual_bins = len(edges) - 1

    reference_counts = np.bincount(
        reference_bins,
        minlength=number_of_actual_bins,
    )

    current_counts = np.bincount(
        current_bins,
        minlength=number_of_actual_bins,
    )

    reference_proportions = reference_counts / len(reference_values)
    current_proportions = current_counts / len(current_values)

    reference_proportions = np.maximum(
        reference_proportions,
        EPSILON,
    )

    current_proportions = np.maximum(
        current_proportions,
        EPSILON,
    )

    psi = np.sum(
        (current_proportions - reference_proportions)
        * np.log(current_proportions / reference_proportions)
    )

    return float(psi)


def calculate_feature_drift(reference_df, current_df, feature):
    reference_values = reference_df[feature].to_numpy()
    current_values = current_df[feature].to_numpy()

    edges = create_reference_bins(
        pd.Series(reference_values)
    )

    psi = calculate_psi(
        reference_values,
        current_values,
        edges,
    )

    drift_detected = psi > DRIFT_THRESHOLD

    return {
        "feature": feature,
        "psi": psi,
        "threshold": DRIFT_THRESHOLD,
        "drift_detected": drift_detected,
    }


def calculate_drift(reference_df, current_df):
    results = []

    for feature in FEATURE_COLUMNS:
        result = calculate_feature_drift(
            reference_df,
            current_df,
            feature,
        )

        results.append(result)

    overall_drift_detected = any(
        result["drift_detected"]
        for result in results
    )

    crossing_features = [
        result["feature"]
        for result in results
        if result["drift_detected"]
    ]

    return results, overall_drift_detected, crossing_features


def main():
    reference_df, current_df = load_populations()

    current_df = add_deterministic_features(current_df)

    validate_features(
        reference_df,
        current_df,
    )

    results, overall_drift_detected, crossing_features = calculate_drift(
        reference_df,
        current_df,
    )

    print("Feature Drift Results")
    print("=====================")

    for result in results:
        print(
            f"{result['feature']}: "
            f"PSI={result['psi']:.6f}, "
            f"threshold={result['threshold']}, "
            f"drift_detected={result['drift_detected']}"
        )

    print()
    print(f"Overall drift_detected: {overall_drift_detected}")
    print(f"Crossing features: {crossing_features}")


if __name__ == "__main__":
    main()

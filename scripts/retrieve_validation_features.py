import os

import pandas as pd
from feast import FeatureStore


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

VALIDATION_INPUT_PATH = "data/phase3/validation_combined_features.csv"
FEAST_REPO_PATH = "feast_repo/fraud_features/feature_repo"
OUTPUT_PATH = "data/phase3/validation_feast_features.csv"

# ---------------------------------------------------------
# Feast feature definitions
# ---------------------------------------------------------

FEATURE_REFS = [
    "fraud_validation_features:TX_AMOUNT",
    "fraud_validation_features:AMOUNT_LOG",
    "fraud_validation_features:TX_HOUR",
    "fraud_validation_features:TX_DAY_OF_WEEK",
    "fraud_validation_features:TX_IS_WEEKEND",
    "fraud_validation_features:TERMINAL_TX_COUNT_1D",
    "fraud_validation_features:TERMINAL_TX_COUNT_7D",
    "fraud_validation_features:TERMINAL_AVG_AMOUNT_7D",
    "fraud_validation_features:CUSTOMER_TX_COUNT_1D",
    "fraud_validation_features:CUSTOMER_TX_COUNT_7D",
]

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

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_validation_data(input_path):
    """
    Load the combined validation feature dataset from a CSV file.

    Args:
        input_path (str): Path to the validation feature CSV file.

    Returns:
        pandas.DataFrame: Loaded validation dataset.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        pandas.errors.EmptyDataError: If the input file is empty.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Validation input file not found: {input_path}"
        )

    return pd.read_csv(input_path)


# ---------------------------------------------------------
# Feast entity preparation
# ---------------------------------------------------------

def create_entity_dataframe(df):
    """
    Create the entity DataFrame required for Feast historical
    feature retrieval.

    Args:
        df (pandas.DataFrame): Validation dataset containing
            TRANSACTION_ID and TX_DATETIME columns.

    Returns:
        pandas.DataFrame: DataFrame containing TRANSACTION_ID
            and event_timestamp for Feast retrieval.

    Raises:
        KeyError: If required columns are missing from the
            validation dataset.
    """
    required_columns = ["TRANSACTION_ID", "TX_DATETIME"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    entity_df = df[required_columns].copy()

    entity_df["event_timestamp"] = pd.to_datetime(
        entity_df["TX_DATETIME"]
    )

    return entity_df[["TRANSACTION_ID", "event_timestamp"]]


# ---------------------------------------------------------
# Feast connection
# ---------------------------------------------------------

def create_feature_store(repo_path):
    """
    Create a Feast FeatureStore object using the specified
    feature repository.

    Args:
        repo_path (str): Path to the Feast feature repository.

    Returns:
        feast.FeatureStore: Initialized Feast FeatureStore.

    Raises:
        FileNotFoundError: If the Feast repository path does not exist.
    """
    if not os.path.exists(repo_path):
        raise FileNotFoundError(
            f"Feast repository not found: {repo_path}"
        )

    return FeatureStore(repo_path=repo_path)


# ---------------------------------------------------------
# Historical feature retrieval
# ---------------------------------------------------------

def retrieve_historical_features(store, entity_df, feature_refs):
    """
    Retrieve historical validation feature values from Feast
    using the supplied transaction entities and timestamps.

    Args:
        store (feast.FeatureStore): Initialized Feast FeatureStore.
        entity_df (pandas.DataFrame): DataFrame containing
            TRANSACTION_ID and event_timestamp.
        feature_refs (list): List of Feast feature references
            to retrieve.

    Returns:
        pandas.DataFrame: Historical feature values retrieved
            from Feast.

    Raises:
        Exception: If Feast historical feature retrieval fails.
    """
    historical_features = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()

    return historical_features


# ---------------------------------------------------------
# Feature validation
# ---------------------------------------------------------

def validate_retrieved_features(
    validation_df,
    historical_features,
    feature_columns,
):
    """
    Compare Feast-retrieved validation feature values with the
    original validation feature values.

    Args:
        validation_df (pandas.DataFrame): Original validation
            dataset containing the calculated feature values.
        historical_features (pandas.DataFrame): Features
            retrieved from Feast.
        feature_columns (list): Names of the 10 model features.

    Returns:
        None: Prints the maximum absolute difference for each
            feature.

    Raises:
        ValueError: If a retrieved feature differs from the
            original validation feature values.
    """
    original_features = validation_df[
        ["TRANSACTION_ID"] + feature_columns
    ].copy()

    retrieved_features = historical_features[
        ["TRANSACTION_ID"] + feature_columns
    ].copy()

    comparison = original_features.merge(
        retrieved_features,
        on="TRANSACTION_ID",
        suffixes=("_original", "_feast"),
        validate="one_to_one",
    )

    print("\nFeature comparison:")

    for feature in feature_columns:
        difference = (
            comparison[f"{feature}_original"]
            - comparison[f"{feature}_feast"]
        )

        max_difference = difference.abs().max()

        print(
            f"{feature}: "
            f"max absolute difference = {max_difference}"
        )

        if max_difference != 0:
            raise ValueError(
                f"Feature validation failed for {feature}. "
                f"Maximum absolute difference: {max_difference}"
            )


# ---------------------------------------------------------
# Missing-value validation
# ---------------------------------------------------------

def validate_missing_values(historical_features, feature_columns):
    """
    Check the Feast-retrieved validation feature columns for
    missing values.

    Args:
        historical_features (pandas.DataFrame): Features
            retrieved from Feast.
        feature_columns (list): Names of the 10 model features.

    Returns:
        None: Prints the missing-value count for each feature.

    Raises:
        ValueError: If any retrieved feature contains missing values.
    """
    missing_values = historical_features[
        feature_columns
    ].isna().sum()

    print("\nMissing values in retrieved features:")
    print(missing_values)

    if missing_values.sum() > 0:
        raise ValueError(
            "Missing values were found in Feast-retrieved features."
        )


# ---------------------------------------------------------
# Final validation feature dataset
# ---------------------------------------------------------

def create_validation_feature_dataset(
    historical_features,
    validation_df,
    feature_columns,
):
    """
    Create the final validation dataset using features retrieved
    from Feast and the original fraud target.

    Args:
        historical_features (pandas.DataFrame): Features retrieved
            from Feast.
        validation_df (pandas.DataFrame): Original validation
            dataset containing the TX_FRAUD target.
        feature_columns (list): Names of the 10 model features.

    Returns:
        pandas.DataFrame: Final validation dataset containing
            TRANSACTION_ID, event_timestamp, model features,
            and TX_FRAUD.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If transaction IDs do not match between
            the datasets.
    """
    required_feast_columns = [
        "TRANSACTION_ID",
        "event_timestamp",
    ] + feature_columns

    missing_feast_columns = [
        column
        for column in required_feast_columns
        if column not in historical_features.columns
    ]

    if missing_feast_columns:
        raise KeyError(
            "Missing columns in Feast output: "
            f"{missing_feast_columns}"
        )

    if "TX_FRAUD" not in validation_df.columns:
        raise KeyError(
            "TX_FRAUD column is missing from validation data."
        )

    feature_dataset = historical_features[
        ["TRANSACTION_ID", "event_timestamp"] + feature_columns
    ].copy()

    target_dataset = validation_df[
        ["TRANSACTION_ID", "TX_FRAUD"]
    ].copy()

    output_df = feature_dataset.merge(
        target_dataset,
        on="TRANSACTION_ID",
        how="inner",
        validate="one_to_one",
    )

    if len(output_df) != len(validation_df):
        raise ValueError(
            "Transaction ID count mismatch after merging "
            "Feast features with TX_FRAUD."
        )

    return output_df

def main():
    validation_df = load_validation_data(VALIDATION_INPUT_PATH)
    print("Validation data shape:", validation_df.shape)

    entity_df = create_entity_dataframe(validation_df)

    print("\nEntity DataFrame shape:", entity_df.shape)

    print("\nFirst 5 entity rows:")
    print(entity_df.head())

    store = create_feature_store(FEAST_REPO_PATH)

    print("\nFeast FeatureStore created successfully.")

    historical_features = retrieve_historical_features(
        store=store,
        entity_df=entity_df,
        feature_refs=FEATURE_REFS,
    )

    print("\nHistorical validation features retrieved successfully.")

    print("Retrieved shape:", historical_features.shape)

    print("\nFirst 5 retrieved rows:")
    print(historical_features.head())

    print("\nRetrieved columns:")
    print(historical_features.columns.tolist())

    validate_retrieved_features(
        validation_df=validation_df,
        historical_features=historical_features,
        feature_columns=FEATURE_COLUMNS,
    )

    validate_missing_values(
        historical_features=historical_features,
        feature_columns=FEATURE_COLUMNS,
    )

    validation_feature_dataset = create_validation_feature_dataset(
        historical_features=historical_features,
        validation_df=validation_df,
        feature_columns=FEATURE_COLUMNS,
    )

    print("\nFinal validation feature dataset created successfully.")

    print("Final output shape:", validation_feature_dataset.shape)

    print("\nFinal output columns:")
    print(validation_feature_dataset.columns.tolist())

    validation_feature_dataset.to_csv(OUTPUT_PATH, index=False)
    print("\nValidation Feast feature dataset saved successfully.")
    print("Output path:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
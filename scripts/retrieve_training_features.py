import os

import pandas as pd
from feast import FeatureStore


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

TRAIN_INPUT_PATH = "data/phase3/train_combined_features.csv"
FEAST_REPO_PATH = "feast_repo/fraud_features/feature_repo"
OUTPUT_PATH = "data/phase3/train_feast_features.csv"


# ---------------------------------------------------------
# Feast feature definitions
# ---------------------------------------------------------

FEATURE_REFS = [
    "fraud_features:TX_AMOUNT",
    "fraud_features:AMOUNT_LOG",
    "fraud_features:TX_HOUR",
    "fraud_features:TX_DAY_OF_WEEK",
    "fraud_features:TX_IS_WEEKEND",
    "fraud_features:TERMINAL_TX_COUNT_1D",
    "fraud_features:TERMINAL_TX_COUNT_7D",
    "fraud_features:TERMINAL_AVG_AMOUNT_7D",
    "fraud_features:CUSTOMER_TX_COUNT_1D",
    "fraud_features:CUSTOMER_TX_COUNT_7D",
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

def load_training_data(input_path):
    """
    Load the combined training feature dataset from a CSV file.

    Args:
        input_path (str): Path to the training feature CSV file.

    Returns:
        pandas.DataFrame: Loaded training dataset.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        pandas.errors.EmptyDataError: If the input file is empty.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Training input file not found: {input_path}"
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
        df (pandas.DataFrame): Training dataset containing
            TRANSACTION_ID and TX_DATETIME columns.

    Returns:
        pandas.DataFrame: DataFrame containing TRANSACTION_ID
            and event_timestamp for Feast retrieval.

    Raises:
        KeyError: If required columns are missing from the
            training dataset.
    """
    required_columns = ["TRANSACTION_ID", "TX_DATETIME"]

    missing_columns = [
        column for column in required_columns
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
    Retrieve historical feature values from Feast using the
    supplied entity DataFrame.

    Args:
        store (feast.FeatureStore): Initialized Feast FeatureStore.
        entity_df (pandas.DataFrame): Entity DataFrame containing
            TRANSACTION_ID and event_timestamp.
        feature_refs (list): List of Feast feature references.

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
# Final dataset creation
# ---------------------------------------------------------

def create_training_feature_dataset(
    historical_features,
    training_df,
    feature_columns,
):
    """
    Create the final training dataset using features retrieved
    from Feast and the original fraud target.

    Args:
        historical_features (pandas.DataFrame): Features retrieved
            from Feast.
        training_df (pandas.DataFrame): Original training dataset
            containing the TX_FRAUD target.
        feature_columns (list): Names of the model features.

    Returns:
        pandas.DataFrame: Final training dataset containing
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

    if "TX_FRAUD" not in training_df.columns:
        raise KeyError(
            "TX_FRAUD column is missing from training data."
        )

    feature_dataset = historical_features[
        ["TRANSACTION_ID", "event_timestamp"] + feature_columns
    ].copy()

    target_dataset = training_df[
        ["TRANSACTION_ID", "TX_FRAUD"]
    ].copy()

    output_df = feature_dataset.merge(
        target_dataset,
        on="TRANSACTION_ID",
        how="inner",
        validate="one_to_one",
    )

    if len(output_df) != len(training_df):
        raise ValueError(
            "Transaction ID count mismatch after merging "
            "Feast features with TX_FRAUD."
        )

    return output_df


# ---------------------------------------------------------
# Feature validation
# ---------------------------------------------------------

def validate_retrieved_features(
    training_df,
    historical_features,
    feature_columns,
):
    """
    Compare Feast-retrieved feature values with the original
    training feature values.

    Args:
        training_df (pandas.DataFrame): Original training dataset.
        historical_features (pandas.DataFrame): Features retrieved
            from Feast.
        feature_columns (list): Names of the model features.

    Returns:
        None: Prints the maximum absolute difference for each
            feature.

    Raises:
        ValueError: If a retrieved feature differs from the
            original feature values.
    """
    original_features = training_df[
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
    Check the Feast-retrieved feature columns for missing values.

    Args:
        historical_features (pandas.DataFrame): Features retrieved
            from Feast.
        feature_columns (list): Names of the model features.

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
# Save output
# ---------------------------------------------------------

def save_training_features(output_df, output_path):
    """
    Save the final Feast-based training feature dataset to CSV.

    Args:
        output_df (pandas.DataFrame): Final training feature dataset.
        output_path (str): Destination path for the output CSV file.

    Returns:
        None: Writes the dataset to the specified output path.

    Raises:
        OSError: If the output directory or file cannot be created.
    """
    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    output_df.to_csv(output_path, index=False)

    print(f"\nTraining Feast feature dataset saved to:")
    print(output_path)


# ---------------------------------------------------------
# Main workflow
# ---------------------------------------------------------

def main():
    """
    Execute the complete Feast historical feature retrieval
    workflow for the training dataset.

    The workflow loads the training data, creates the Feast
    entity DataFrame, retrieves historical features, validates
    the retrieved values and missing values, creates the final
    training dataset, and saves the result.

    Args:
        None.

    Returns:
        None: Completes the training feature retrieval workflow.

    Raises:
        FileNotFoundError: If required input or Feast files are missing.
        KeyError: If required dataset columns are missing.
        ValueError: If feature validation or dataset matching fails.
    """

    # -----------------------------------------------------
    # Load training data
    # -----------------------------------------------------

    training_df = load_training_data(TRAIN_INPUT_PATH)

    print("Training data shape:", training_df.shape)

    print("\nTraining data columns:")
    print(training_df.columns.tolist())

    # -----------------------------------------------------
    # Create Feast entity DataFrame
    # -----------------------------------------------------

    entity_df = create_entity_dataframe(training_df)

    print("\nEntity DataFrame shape:", entity_df.shape)

    print("\nFirst 5 entity rows:")
    print(entity_df.head())

    # -----------------------------------------------------
    # Connect to Feast
    # -----------------------------------------------------

    store = create_feature_store(FEAST_REPO_PATH)

    print("\nFeast FeatureStore created successfully.")

    # -----------------------------------------------------
    # Display feature configuration
    # -----------------------------------------------------

    print("\nNumber of Feast features:", len(FEATURE_REFS))

    # -----------------------------------------------------
    # Retrieve historical features
    # -----------------------------------------------------

    historical_features = retrieve_historical_features(
        store=store,
        entity_df=entity_df,
        feature_refs=FEATURE_REFS,
    )

    print("\nHistorical features retrieved successfully.")

    print("Retrieved shape:", historical_features.shape)

    print("\nFirst 5 retrieved rows:")
    print(historical_features.head())

    print("\nRetrieved columns:")
    print(historical_features.columns.tolist())

    # -----------------------------------------------------
    # Validate retrieved features
    # -----------------------------------------------------

    validate_retrieved_features(
        training_df=training_df,
        historical_features=historical_features,
        feature_columns=FEATURE_COLUMNS,
    )

    # -----------------------------------------------------
    # Validate missing values
    # -----------------------------------------------------

    validate_missing_values(
        historical_features=historical_features,
        feature_columns=FEATURE_COLUMNS,
    )

    # -----------------------------------------------------
    # Create final training feature dataset
    # -----------------------------------------------------

    output_df = create_training_feature_dataset(
        historical_features=historical_features,
        training_df=training_df,
        feature_columns=FEATURE_COLUMNS,
    )

    print("\nFinal output shape:", output_df.shape)

    print("\nFinal output columns:")
    print(output_df.columns.tolist())

    # -----------------------------------------------------
    # Save final dataset
    # -----------------------------------------------------

    save_training_features(
        output_df=output_df,
        output_path=OUTPUT_PATH,
    )

    print("\nFeast training feature retrieval completed successfully.")


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
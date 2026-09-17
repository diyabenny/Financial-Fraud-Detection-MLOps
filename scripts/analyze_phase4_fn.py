import pandas as pd


# ============================================================
# PATH
# ============================================================

PREDICTION_PATH = (
    "reports/phase4/phase4_transaction_predictions.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4 - FALSE NEGATIVE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load transaction-level predictions
    # --------------------------------------------------------

    df = pd.read_csv(
        PREDICTION_PATH
    )

    print(
        "\nPrediction table shape:",
        df.shape
    )

    print(
        "Candidates:",
        df["candidate"].unique().tolist()
    )

    # --------------------------------------------------------
    # False Negatives
    #
    # Actual fraud = 1
    # Predicted legitimate = 0
    # --------------------------------------------------------

    fn = df[
        (df["TX_FRAUD"] == 1)
        & (df["predicted_class"] == 0)
    ].copy()

    print(
        "\nTotal False Negatives by candidate:"
    )

    print(
        fn.groupby("candidate")
        .size()
        .to_string()
    )

    # --------------------------------------------------------
    # FN descriptive summary
    # --------------------------------------------------------

    numeric_columns = [
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
        "predicted_probability",
    ]

    # The prediction file currently contains only the
    # transaction identifiers, labels, predictions and
    # scenario information. Therefore, feature-level
    # analysis will be added after verifying availability.
    #
    # For now, show the columns actually available.

    print(
        "\nAvailable columns:"
    )

    print(
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # FN probability summary
    # --------------------------------------------------------

    print(
        "\nFalse Negative probability summary:"
    )

    print(
        fn.groupby("candidate")[
            "predicted_probability"
        ]
        .describe()
        .round(6)
        .to_string()
    )

    # --------------------------------------------------------
    # FN scenario distribution
    #
    # Scenario analysis is descriptive only.
    # --------------------------------------------------------

    print(
        "\nFalse Negative scenario distribution:"
    )

    scenario_summary = (
        fn.groupby(
            [
                "candidate",
                "TX_FRAUD_SCENARIO",
            ]
        )
        .size()
        .reset_index(
            name="FN_count"
        )
    )

    print(
        scenario_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save FN transactions
    # --------------------------------------------------------

    output_path = (
        "reports/phase4/phase4_false_negatives.csv"
    )

    fn.to_csv(
        output_path,
        index=False
    )

    print(
        "\nFalse Negative transactions saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()
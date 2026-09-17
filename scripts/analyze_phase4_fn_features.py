import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PREDICTIONS_FILE = Path(
    "reports/phase4/phase4_transaction_predictions.csv"
)

EVALUATION_FILE = Path(
    "data/phase4/april_25_30_evaluation_features.csv"
)

OUTPUT_FILE = Path(
    "reports/phase4/phase4_fn_feature_analysis.csv"
)


# ============================================================
# Features to analyze
# ============================================================

FEATURES = [
    "TX_AMOUNT",
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
# Load data
# ============================================================

print("\nLoading prediction data...")
predictions = pd.read_csv(PREDICTIONS_FILE)

print("Loading frozen evaluation features...")
evaluation = pd.read_csv(EVALUATION_FILE)


# ============================================================
# Basic checks
# ============================================================

print("\n--- BASIC CHECKS ---")

print("Prediction shape:", predictions.shape)
print("Evaluation shape:", evaluation.shape)

print(
    "Unique prediction transaction IDs:",
    predictions["TRANSACTION_ID"].nunique()
)

print(
    "Unique evaluation transaction IDs:",
    evaluation["TRANSACTION_ID"].nunique()
)

if evaluation["TRANSACTION_ID"].duplicated().any():
    raise ValueError(
        "Evaluation dataset contains duplicate TRANSACTION_ID values."
    )

print("Evaluation transaction IDs are unique.")


# ============================================================
# Identify false negatives
# ============================================================

fn = predictions[
    (predictions["TX_FRAUD"] == 1)
    & (predictions["predicted_class"] == 0)
].copy()

print("\n--- FALSE NEGATIVE COUNTS ---")

print(
    fn.groupby("candidate")
    .size()
    .sort_index()
)


# ============================================================
# Merge FN predictions with original evaluation features
# ============================================================

print("\nJoining FN predictions with evaluation features...")

merge_columns = [
    "TRANSACTION_ID"
] + FEATURES

fn_features = fn.merge(
    evaluation[merge_columns],
    on="TRANSACTION_ID",
    how="left",
    validate="many_to_one"
)


# ============================================================
# Verify merge
# ============================================================

missing_features = fn_features[FEATURES].isna().all(axis=1).sum()

print("FN rows:", len(fn_features))
print("Rows missing feature data:", missing_features)

if missing_features > 0:
    raise ValueError(
        "Some FN transactions could not be matched with evaluation features."
    )

print("All FN transactions successfully matched.")


# ============================================================
# Feature-level analysis
# ============================================================

print("\n" + "=" * 70)
print("FALSE NEGATIVE FEATURE ANALYSIS")
print("=" * 70)


for candidate in sorted(fn_features["candidate"].unique()):

    candidate_fn = fn_features[
        fn_features["candidate"] == candidate
    ].copy()

    print("\n" + "-" * 70)
    print(f"CANDIDATE: {candidate}")
    print("-" * 70)

    print("FN count:", len(candidate_fn))

    print("\nProbability summary:")
    print(
        candidate_fn["predicted_probability"]
        .describe()
        .round(6)
        .to_string()
    )

    print("\nFeature summary:")

    summary = candidate_fn[FEATURES].describe().T

    summary = summary[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]

    print(summary.round(4).to_string())

    # --------------------------------------------------------
    # Scenario distribution
    # --------------------------------------------------------

    if "TX_FRAUD_SCENARIO" in candidate_fn.columns:

        print("\nScenario distribution:")

        scenario_counts = (
            candidate_fn["TX_FRAUD_SCENARIO"]
            .value_counts()
            .sort_index()
        )

        print(scenario_counts.to_string())

        scenario_percent = (
            candidate_fn["TX_FRAUD_SCENARIO"]
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
        )

        print("\nScenario percentage:")

        print(scenario_percent.to_string())


# ============================================================
# Compare FN feature distributions with overall evaluation set
# ============================================================

print("\n" + "=" * 70)
print("FN VS OVERALL EVALUATION FEATURE MEANS")
print("=" * 70)

overall_means = evaluation[FEATURES].mean()

comparison_rows = []

for candidate in sorted(fn_features["candidate"].unique()):

    candidate_fn = fn_features[
        fn_features["candidate"] == candidate
    ]

    fn_means = candidate_fn[FEATURES].mean()

    for feature in FEATURES:

        comparison_rows.append(
            {
                "candidate": candidate,
                "feature": feature,
                "overall_evaluation_mean": overall_means[feature],
                "fn_mean": fn_means[feature],
                "difference": (
                    fn_means[feature]
                    - overall_means[feature]
                ),
            }
        )


comparison = pd.DataFrame(comparison_rows)

print(
    comparison.round(4).to_string(index=False)
)


# ============================================================
# Save detailed FN feature-level dataset
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

fn_features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    f"Saved detailed FN feature analysis to:\n{OUTPUT_FILE}"
)

print("\nAnalysis completed successfully.")
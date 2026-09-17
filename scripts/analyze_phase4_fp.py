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
    "reports/phase4/phase4_fp_feature_analysis.csv"
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
# Identify false positives
# ============================================================

# False Positive:
# Actual transaction is legitimate (TX_FRAUD = 0)
# but model predicts fraud (predicted_class = 1)

fp = predictions[
    (predictions["TX_FRAUD"] == 0)
    & (predictions["predicted_class"] == 1)
].copy()

print("\n--- FALSE POSITIVE COUNTS ---")

print(
    fp.groupby("candidate")
    .size()
    .sort_index()
)


# ============================================================
# Merge FP predictions with evaluation features
# ============================================================

print("\nJoining FP predictions with evaluation features...")

merge_columns = [
    "TRANSACTION_ID"
] + FEATURES

fp_features = fp.merge(
    evaluation[merge_columns],
    on="TRANSACTION_ID",
    how="left",
    validate="many_to_one"
)


# ============================================================
# Verify merge
# ============================================================

missing_features = fp_features[FEATURES].isna().all(axis=1).sum()

print("FP rows:", len(fp_features))
print("Rows missing feature data:", missing_features)

if missing_features > 0:
    raise ValueError(
        "Some FP transactions could not be matched with evaluation features."
    )

print("All FP transactions successfully matched.")


# ============================================================
# Feature-level FP analysis
# ============================================================

print("\n" + "=" * 70)
print("FALSE POSITIVE FEATURE ANALYSIS")
print("=" * 70)


for candidate in sorted(fp_features["candidate"].unique()):

    candidate_fp = fp_features[
        fp_features["candidate"] == candidate
    ].copy()

    print("\n" + "-" * 70)
    print(f"CANDIDATE: {candidate}")
    print("-" * 70)

    print("FP count:", len(candidate_fp))

    print("\nProbability summary:")

    print(
        candidate_fp["predicted_probability"]
        .describe()
        .round(6)
        .to_string()
    )

    print("\nFeature summary:")

    summary = candidate_fp[FEATURES].describe().T

    summary = summary[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]

    print(summary.round(4).to_string())


# ============================================================
# Compare FP feature distributions with overall
# legitimate transactions
# ============================================================

print("\n" + "=" * 70)
print("FP VS OVERALL LEGITIMATE TRANSACTION FEATURE MEANS")
print("=" * 70)


legitimate_evaluation = evaluation[
    evaluation["TX_FRAUD"] == 0
].copy()

overall_legitimate_means = legitimate_evaluation[
    FEATURES
].mean()


comparison_rows = []


for candidate in sorted(fp_features["candidate"].unique()):

    candidate_fp = fp_features[
        fp_features["candidate"] == candidate
    ]

    fp_means = candidate_fp[FEATURES].mean()

    for feature in FEATURES:

        comparison_rows.append(
            {
                "candidate": candidate,
                "feature": feature,
                "overall_legitimate_mean": (
                    overall_legitimate_means[feature]
                ),
                "fp_mean": fp_means[feature],
                "difference": (
                    fp_means[feature]
                    - overall_legitimate_means[feature]
                ),
            }
        )


comparison = pd.DataFrame(comparison_rows)

print(
    comparison.round(4).to_string(index=False)
)


# ============================================================
# Save detailed FP feature-level dataset
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

fp_features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    f"Saved detailed FP feature analysis to:\n{OUTPUT_FILE}"
)

print("\nAnalysis completed successfully.")
from pathlib import Path
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path(
    "reports/phase4/phase4_transaction_predictions.csv"
)

OUTPUT_FILE = Path(
    "reports/phase4/phase4_scenario_analysis.csv"
)


# ============================================================
# Load prediction data
# ============================================================

print("Loading transaction-level predictions...")

df = pd.read_csv(INPUT_FILE)

print("Prediction shape:", df.shape)


# ============================================================
# Basic validation
# ============================================================

required_columns = [
    "candidate",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
    "predicted_class",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# Generate scenario analysis
# ============================================================

rows = []

candidates = sorted(df["candidate"].unique())

print("\nCandidates:")
for candidate in candidates:
    print(f"  - {candidate}")


for candidate in candidates:

    candidate_df = df[
        df["candidate"] == candidate
    ]

    for scenario in [1, 2, 3]:

        scenario_df = candidate_df[
            (candidate_df["TX_FRAUD"] == 1)
            & (
                candidate_df["TX_FRAUD_SCENARIO"]
                == scenario
            )
        ]

        total_fraud = len(scenario_df)

        tp = int(
            (
                scenario_df["predicted_class"] == 1
            ).sum()
        )

        fn = int(
            (
                scenario_df["predicted_class"] == 0
            ).sum()
        )

        if total_fraud == 0:
            detection_rate = 0.0
            fn_rate = 0.0
        else:
            detection_rate = tp / total_fraud
            fn_rate = fn / total_fraud

        rows.append(
            {
                "candidate": candidate,
                "TX_FRAUD_SCENARIO": scenario,
                "total_fraud": total_fraud,
                "TP": tp,
                "FN": fn,
                "scenario_detection_rate": detection_rate,
                "scenario_FN_rate": fn_rate,
            }
        )


scenario_analysis = pd.DataFrame(rows)


# ============================================================
# Validation
# ============================================================

print("\n" + "=" * 70)
print("SCENARIO ANALYSIS")
print("=" * 70)

print(
    scenario_analysis.to_string(index=False)
)


expected_scenarios = {
    1: 43,
    2: 324,
    3: 159,
}


for candidate in candidates:

    candidate_analysis = scenario_analysis[
        scenario_analysis["candidate"] == candidate
    ]

    if len(candidate_analysis) != 3:
        raise ValueError(
            f"{candidate} does not have exactly "
            f"3 scenario rows."
        )

    for scenario, expected_total in expected_scenarios.items():

        row = candidate_analysis[
            candidate_analysis["TX_FRAUD_SCENARIO"]
            == scenario
        ].iloc[0]

        if row["total_fraud"] != expected_total:
            raise ValueError(
                f"{candidate} Scenario {scenario}: "
                f"expected {expected_total} frauds, "
                f"found {row['total_fraud']}"
            )

        if row["TP"] + row["FN"] != row["total_fraud"]:
            raise ValueError(
                f"{candidate} Scenario {scenario}: "
                "TP + FN does not equal total fraud."
            )


# ============================================================
# Save
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

scenario_analysis.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    f"Scenario analysis saved to:\n{OUTPUT_FILE}"
)

print(
    "\nOutput shape:",
    scenario_analysis.shape
)

print("\nAll scenario checks passed.")
print("Scenario analysis completed successfully.")
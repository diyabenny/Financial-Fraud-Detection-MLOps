import pandas as pd
from pathlib import Path


# ============================================================
# Output path
# ============================================================

OUTPUT_FILE = Path(
    "reports/phase4/phase4_candidate_evidence.csv"
)


# ============================================================
# Verified Phase 4 candidate evidence
# ============================================================

evidence = [
    {
        "candidate": "LR_Original_C_0.01",
        "threshold": 0.10,
        "PR_AUC": 0.2720652421,
        "ROC_AUC": 0.6783075401,
        "precision": 0.7743902439,
        "recall": 0.2414448669,
        "F1": 0.3681159420,
        "FPR": 0.0006463898,
        "TP": 127,
        "FP": 37,
        "TN": 57204,
        "FN": 399,

        "Scenario_1_Total": 43,
        "Scenario_1_TP": 42,
        "Scenario_1_FN": 1,
        "Scenario_1_Detection_Rate": 0.9767441860,

        "Scenario_2_Total": 324,
        "Scenario_2_TP": 1,
        "Scenario_2_FN": 323,
        "Scenario_2_Detection_Rate": 0.0030864198,

        "Scenario_3_Total": 159,
        "Scenario_3_TP": 84,
        "Scenario_3_FN": 75,
        "Scenario_3_Detection_Rate": 0.5283018868,
    },

    {
        "candidate": "RF_Balanced_100_None_1",
        "threshold": 0.65,
        "PR_AUC": 0.2536834538,
        "ROC_AUC": 0.6254614520,
        "precision": 0.9919354839,
        "recall": 0.2338403042,
        "F1": 0.3784615385,
        "FPR": 0.0000174700,
        "TP": 123,
        "FP": 1,
        "TN": 57240,
        "FN": 403,

        "Scenario_1_Total": 43,
        "Scenario_1_TP": 43,
        "Scenario_1_FN": 0,
        "Scenario_1_Detection_Rate": 1.0000000000,

        "Scenario_2_Total": 324,
        "Scenario_2_TP": 1,
        "Scenario_2_FN": 323,
        "Scenario_2_Detection_Rate": 0.0030864198,

        "Scenario_3_Total": 159,
        "Scenario_3_TP": 79,
        "Scenario_3_FN": 80,
        "Scenario_3_Detection_Rate": 0.4968553459,
    },

    {
        "candidate": "RF_Balanced_300_None_1",
        "threshold": 0.65,
        "PR_AUC": 0.2574410304,
        "ROC_AUC": 0.6370331816,
        "precision": 1.0000000000,
        "recall": 0.2338403042,
        "F1": 0.3790446841,
        "FPR": 0.0000000000,
        "TP": 123,
        "FP": 0,
        "TN": 57241,
        "FN": 403,

        "Scenario_1_Total": 43,
        "Scenario_1_TP": 43,
        "Scenario_1_FN": 0,
        "Scenario_1_Detection_Rate": 1.0000000000,

        "Scenario_2_Total": 324,
        "Scenario_2_TP": 1,
        "Scenario_2_FN": 323,
        "Scenario_2_Detection_Rate": 0.0030864198,

        "Scenario_3_Total": 159,
        "Scenario_3_TP": 79,
        "Scenario_3_FN": 80,
        "Scenario_3_Detection_Rate": 0.4968553459,
    },

    {
        "candidate": "LR_Balanced_C_10",
        "threshold": 0.95,
        "PR_AUC": 0.2593350753,
        "ROC_AUC": 0.6794600284,
        "precision": 0.9770114943,
        "recall": 0.1615969582,
        "F1": 0.2773246329,
        "FPR": 0.0000349400,
        "TP": 85,
        "FP": 2,
        "TN": 57239,
        "FN": 441,

        "Scenario_1_Total": 43,
        "Scenario_1_TP": 10,
        "Scenario_1_FN": 33,
        "Scenario_1_Detection_Rate": 0.2325581395,

        "Scenario_2_Total": 324,
        "Scenario_2_TP": 0,
        "Scenario_2_FN": 324,
        "Scenario_2_Detection_Rate": 0.0000000000,

        "Scenario_3_Total": 159,
        "Scenario_3_TP": 75,
        "Scenario_3_FN": 84,
        "Scenario_3_Detection_Rate": 0.4716981132,
    },
]


# ============================================================
# Create DataFrame
# ============================================================

df = pd.DataFrame(evidence)


# ============================================================
# Basic verification
# ============================================================

print("\n" + "=" * 80)
print("UPDATED PHASE 4 CANDIDATE EVIDENCE TABLE")
print("=" * 80)

print(df.to_string(index=False))


# ============================================================
# Overall confusion-matrix checks
# ============================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX CHECKS")
print("=" * 80)

for _, row in df.iterrows():

    actual_fraud = row["TP"] + row["FN"]
    actual_legitimate = row["TN"] + row["FP"]

    print(
        f"{row['candidate']}: "
        f"TP+FN={actual_fraud}, "
        f"TN+FP={actual_legitimate}"
    )

    if actual_fraud != 526:
        raise ValueError(
            f"Fraud count mismatch for {row['candidate']}"
        )

    if actual_legitimate != 57241:
        raise ValueError(
            f"Legitimate count mismatch for {row['candidate']}"
        )


print("\nAll confusion-matrix checks passed.")


# ============================================================
# Scenario checks
# ============================================================

print("\n" + "=" * 80)
print("SCENARIO CHECKS")
print("=" * 80)

for _, row in df.iterrows():

    for scenario in [1, 2, 3]:

        total = row[f"Scenario_{scenario}_Total"]
        tp = row[f"Scenario_{scenario}_TP"]
        fn = row[f"Scenario_{scenario}_FN"]
        rate = row[f"Scenario_{scenario}_Detection_Rate"]

        # TP + FN must equal total fraud in scenario
        if tp + fn != total:
            raise ValueError(
                f"Scenario {scenario} mismatch "
                f"for {row['candidate']}"
            )

        # Detection rate must equal TP / total
        calculated_rate = tp / total

        if abs(rate - calculated_rate) > 1e-9:
            raise ValueError(
                f"Scenario {scenario} detection-rate mismatch "
                f"for {row['candidate']}"
            )

        print(
            f"{row['candidate']} | "
            f"Scenario {scenario}: "
            f"Total={total}, "
            f"TP={tp}, "
            f"FN={fn}, "
            f"Detection Rate={rate:.6f}"
        )


print("\nAll scenario checks passed.")


# ============================================================
# Verify scenario totals
# ============================================================

scenario_totals = [
    df["Scenario_1_Total"].iloc[0],
    df["Scenario_2_Total"].iloc[0],
    df["Scenario_3_Total"].iloc[0],
]

if sum(scenario_totals) != 526:
    raise ValueError(
        "Scenario totals do not equal 526."
    )

print(
    "\nScenario totals verified:",
    scenario_totals,
    "=>",
    sum(scenario_totals)
)


# ============================================================
# Save updated evidence table
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(
    f"Updated candidate evidence table saved to:\n{OUTPUT_FILE}"
)

print("Output shape:", df.shape)

print("\nUpdated candidate evidence table created successfully.")
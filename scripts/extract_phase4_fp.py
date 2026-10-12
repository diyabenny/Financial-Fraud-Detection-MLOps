import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PREDICTIONS_FILE = Path(
    "reports/phase4/phase4_transaction_predictions.csv"
)

OUTPUT_FILE = Path(
    "reports/phase4/phase4_false_positives.csv"
)


# ============================================================
# Load transaction-level predictions
# ============================================================

print("\nLoading transaction-level predictions...")

predictions = pd.read_csv(PREDICTIONS_FILE)

print("Prediction shape:", predictions.shape)


# ============================================================
# Extract False Positives
# ============================================================

# False Positive:
# Actual transaction = Legitimate (TX_FRAUD = 0)
# Model prediction = Fraud (predicted_class = 1)

fp = predictions[
    (predictions["TX_FRAUD"] == 0)
    & (predictions["predicted_class"] == 1)
].copy()


# ============================================================
# Display FP counts by candidate
# ============================================================

print("\n--- FALSE POSITIVE COUNTS ---")

fp_counts = (
    fp.groupby("candidate")
    .size()
    .sort_index()
)

print(fp_counts)


# ============================================================
# Verify expected total
# ============================================================

print("\nTotal False Positive records:", len(fp))

expected_fp = 40

if len(fp) != expected_fp:
    raise ValueError(
        f"Expected {expected_fp} FP records, "
        f"but found {len(fp)}."
    )

print("FP count verification passed.")


# ============================================================
# Save False Positives
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

fp.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Final verification
# ============================================================

print("\nSaved False Positive file to:")
print(OUTPUT_FILE)

print("\nOutput shape:", fp.shape)

print("\nColumns:")
print(fp.columns.tolist())

print("\nFalse Positive extraction completed successfully.")
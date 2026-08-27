# ============================================================
# CLASS-WEIGHTED LOGISTIC REGRESSION EXPERIMENT
# ============================================================

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/raw/Financial_fraud_dataset.csv")

print("Dataset shape:")
print(df.shape)


# ============================================================
# 2. PREPARE DATETIME
# ============================================================

df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

print("\nDate range:")
print("Start:", df["TX_DATETIME"].min())
print("End:", df["TX_DATETIME"].max())


# ============================================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

train_days = list(range(0, 25))
test_days = list(range(25, 30)) + [49]

train_df = df[df["TX_TIME_DAYS"].isin(train_days)].copy()
test_df = df[df["TX_TIME_DAYS"].isin(test_days)].copy()

print("\n========== TRAIN / TEST SPLIT ==========")

print("Training rows:", len(train_df))
print("Test rows:", len(test_df))

print("\nTraining class distribution:")
print(train_df["TX_FRAUD"].value_counts())

print("\nTest class distribution:")
print(test_df["TX_FRAUD"].value_counts())


# ============================================================
# 4. CREATE TIME-BASED FEATURES
# ============================================================

train_df["TX_HOUR"] = train_df["TX_DATETIME"].dt.hour
train_df["TX_DAY_OF_WEEK"] = train_df["TX_DATETIME"].dt.dayofweek

test_df["TX_HOUR"] = test_df["TX_DATETIME"].dt.hour
test_df["TX_DAY_OF_WEEK"] = test_df["TX_DATETIME"].dt.dayofweek


# ============================================================
# 5. SELECT BASELINE FEATURES
# ============================================================

baseline_features = [
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_HOUR",
    "TX_DAY_OF_WEEK"
]

target = "TX_FRAUD"

print("\n========== BASELINE FEATURES ==========")
print(baseline_features)


# ============================================================
# 6. CREATE X AND y
# ============================================================

X_train = train_df[baseline_features].copy()
y_train = train_df[target].copy()

X_test = test_df[baseline_features].copy()
y_test = test_df[target].copy()


# ============================================================
# 7. DATA QUALITY CHECK
# ============================================================

print("\n========== DATA QUALITY CHECK ==========")

print("\nMissing values in training:")
print(X_train.isnull().sum())

print("\nMissing values in test:")
print(X_test.isnull().sum())

print("\nDuplicate rows in training:", X_train.duplicated().sum())
print("Duplicate rows in test:", X_test.duplicated().sum())

print("\nTraining data types:")
print(X_train.dtypes)


# ============================================================
# 8. CREATE CLASS-WEIGHTED MODEL
# ============================================================

weighted_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\n========== CLASS-WEIGHTED MODEL TRAINING ==========")

weighted_model.fit(X_train, y_train)

print("Model: Logistic Regression")
print("Class weighting: balanced")
print("Training completed successfully.")


# ============================================================
# 10. TEST PREDICTIONS
# ============================================================

print("\n========== TEST PREDICTIONS ==========")

y_pred_weighted = weighted_model.predict(X_test)

y_prob_weighted = weighted_model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully.")
print("Number of predictions:", len(y_pred_weighted))
print("Number of probabilities:", len(y_prob_weighted))


# ============================================================
# 11. PREDICTED CLASS DISTRIBUTION
# ============================================================

print("\n========== PREDICTED CLASS DISTRIBUTION ==========")

print(
    "Predicted legitimate:",
    (y_pred_weighted == 0).sum()
)

print(
    "Predicted fraud:",
    (y_pred_weighted == 1).sum()
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

tn_w, fp_w, fn_w, tp_w = confusion_matrix(
    y_test,
    y_pred_weighted
).ravel()

print("True Negatives (TN):", tn_w)
print("False Positives (FP):", fp_w)
print("False Negatives (FN):", fn_w)
print("True Positives (TP):", tp_w)


# ============================================================
# 13. CALCULATE KPIs
# ============================================================

precision_w = precision_score(
    y_test,
    y_pred_weighted,
    zero_division=0
)

recall_w = recall_score(
    y_test,
    y_pred_weighted,
    zero_division=0
)

f1_w = f1_score(
    y_test,
    y_pred_weighted,
    zero_division=0
)

pr_auc_w = average_precision_score(
    y_test,
    y_prob_weighted
)

roc_auc_w = roc_auc_score(
    y_test,
    y_prob_weighted
)

accuracy_w = accuracy_score(
    y_test,
    y_pred_weighted
)

fpr_w = fp_w / (fp_w + tn_w)


# ============================================================
# 14. RESULTS
# ============================================================

print("\n========== CLASS-WEIGHTED KPI RESULTS ==========")

print("Recall:", recall_w)
print("PR-AUC:", pr_auc_w)
print("Precision:", precision_w)
print("F1-Score:", f1_w)
print("FPR:", fpr_w)
print("ROC-AUC:", roc_auc_w)
print("Accuracy:", accuracy_w)


# ============================================================
# 15. RESULTS IN PERCENTAGE
# ============================================================

print("\n========== CLASS-WEIGHTED KPI RESULTS (%) ==========")

print("Recall:", recall_w * 100, "%")
print("PR-AUC:", pr_auc_w * 100, "%")
print("Precision:", precision_w * 100, "%")
print("F1-Score:", f1_w * 100, "%")
print("FPR:", fpr_w * 100, "%")
print("ROC-AUC:", roc_auc_w * 100, "%")
print("Accuracy:", accuracy_w * 100, "%")


# ============================================================
# 16. ORIGINAL BASELINE RESULTS
# ============================================================

recall = 0.15749525616698293
pr_auc = 0.23881790787794868
precision = 0.9764705882352941
f1 = 0.27124183006535946
fpr = 3.486750348675035e-05
roc_auc = 0.6625475375735395
accuracy = 0.9922953340128181


# ============================================================
# 17. COMPARISON WITH ORIGINAL BASELINE
# ============================================================

print("\n========== BASELINE COMPARISON ==========")

print("                         Original       Class-Weighted")
print("Recall:                ", recall, "      ", recall_w)
print("PR-AUC:                ", pr_auc, "      ", pr_auc_w)
print("Precision:             ", precision, "      ", precision_w)
print("F1-Score:              ", f1, "      ", f1_w)
print("FPR:                   ", fpr, "      ", fpr_w)
print("ROC-AUC:               ", roc_auc, "      ", roc_auc_w)
print("Accuracy:              ", accuracy, "      ", accuracy_w)


# ============================================================
# 18. EXPERIMENT FINDINGS
# ============================================================

print("\n========== CLASS-WEIGHT EXPERIMENT FINDINGS ==========")

print("""
Class weighting substantially increased fraud detection recall
from 15.75% to 95.26%.

However, this improvement resulted in a major increase in
false-positive rate from 0.00349% to 93.11%.

Precision also decreased from 97.65% to 0.93%, while F1-Score
decreased from 27.12% to 1.84%.

PR-AUC decreased from 23.88% to 16.83%.

Therefore, class weighting alone is not considered a suitable
improvement to the official baseline.

The original Logistic Regression model is retained as the
official baseline reference.

This experiment demonstrates the trade-off between increasing
fraud detection recall and controlling false fraud alerts.

Future experiments should investigate approaches that improve
fraud detection while maintaining acceptable precision and FPR.
""")


# ============================================================
# 19. FINAL STATUS
# ============================================================

print("\n========== FINAL STATUS ==========")

print("Class-weighted experiment completed successfully.")
print("Official baseline remains: Logistic Regression")
print("Class weighting: Experiment only")
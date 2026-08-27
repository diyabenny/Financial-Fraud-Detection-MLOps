# ============================================================
# FINANCIAL FRAUD DETECTION - BASELINE MODEL
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
# 7. DATA QUALITY CHECKS
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

print("\nTraining feature ranges:")
print(X_train.describe().T)

print("\nInvalid TX_AMOUNT <= 0:")
print((X_train["TX_AMOUNT"] <= 0).sum())

print("\nInvalid TX_TIME_SECONDS < 0:")
print((X_train["TX_TIME_SECONDS"] < 0).sum())

print("\nInvalid TX_HOUR:")
print(
    (
        (X_train["TX_HOUR"] < 0)
        | (X_train["TX_HOUR"] > 23)
    ).sum()
)

print("\nInvalid TX_DAY_OF_WEEK:")
print(
    (
        (X_train["TX_DAY_OF_WEEK"] < 0)
        | (X_train["TX_DAY_OF_WEEK"] > 6)
    ).sum()
)


# ============================================================
# 8. CREATE BASELINE MODEL
# ============================================================

baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


# ============================================================
# 9. TRAIN BASELINE MODEL
# ============================================================

print("\n========== BASELINE MODEL TRAINING ==========")

baseline_model.fit(X_train, y_train)

print("Model: Logistic Regression")
print("Training completed successfully.")


# ============================================================
# 10. GENERATE TEST PREDICTIONS
# ============================================================

print("\n========== TEST PREDICTIONS ==========")

y_pred = baseline_model.predict(X_test)

y_prob = baseline_model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully.")
print("Number of predictions:", len(y_pred))


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()

print("True Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("True Positives (TP):", tp)


# ============================================================
# 12. CALCULATE BASELINE KPIs
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

fpr = fp / (fp + tn)


# ============================================================
# 13. BASELINE RESULTS
# ============================================================

print("\n========== BASELINE KPI RESULTS ==========")

print("Recall:", recall)
print("PR-AUC:", pr_auc)
print("Precision:", precision)
print("F1-Score:", f1)
print("FPR:", fpr)
print("ROC-AUC:", roc_auc)
print("Accuracy:", accuracy)


# ============================================================
# 14. BASELINE RESULTS (%)
# ============================================================

print("\n========== BASELINE KPI RESULTS (%) ==========")

print("Recall:", recall * 100, "%")
print("PR-AUC:", pr_auc * 100, "%")
print("Precision:", precision * 100, "%")
print("F1-Score:", f1 * 100, "%")
print("FPR:", fpr * 100, "%")
print("ROC-AUC:", roc_auc * 100, "%")
print("Accuracy:", accuracy * 100, "%")


# ============================================================
# 15. FINAL BASELINE SUMMARY
# ============================================================

print("\n========== BASELINE SUMMARY ==========")

print("Primary Business KPI - Recall:", recall)
print("Primary Model KPI - PR-AUC:", pr_auc)
print("Supporting KPI - Precision:", precision)
print("Supporting KPI - FPR:", fpr)
print("Supporting KPI - F1-Score:", f1)
print("Supporting KPI - ROC-AUC:", roc_auc)
print("Reference KPI - Accuracy:", accuracy)

print("\nBaseline model evaluation completed.")


# ============================================================
# 16. BASELINE FINDINGS
# ============================================================

"""
BASELINE FINDINGS

The original Logistic Regression baseline achieved:

Recall:     15.75%
PR-AUC:     23.88%
Precision:  97.65%
F1-Score:   27.12%
FPR:        0.00349%
ROC-AUC:    66.25%
Accuracy:   99.23%

The baseline achieves very high precision and a very low
false-positive rate. However, recall is relatively low,
meaning that a large proportion of actual fraudulent
transactions are missed.

A class-weighted Logistic Regression experiment was also
evaluated to investigate the effect of class imbalance.

The class-weighted model achieved:

Recall:     95.26%
PR-AUC:     16.83%
Precision:  0.93%
F1-Score:   1.84%
FPR:        93.11%
ROC-AUC:    65.77%
Accuracy:   7.69%

The class-weighted model substantially increased recall,
but this came with a very large increase in false-positive
rate and a major reduction in precision.

Therefore, class weighting alone does not provide a suitable
solution for the current baseline because it generates an
impractically high number of false fraud alerts.

The original Logistic Regression model is retained as the
official baseline reference model.

The baseline provides a reference point for evaluating future
improvements in fraud detection performance.

Future model development can investigate feature engineering,
classification threshold optimization, class-imbalance
handling, and more advanced machine learning models.

All future models should be evaluated using the same KPI
framework and evaluation procedure to allow fair comparison
with the baseline.
"""
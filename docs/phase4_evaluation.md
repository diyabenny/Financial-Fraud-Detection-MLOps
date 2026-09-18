\# Phase 4 — Model Evaluation and Error Analysis



\## 1. Overview

Phase 4 evaluates fraud detection model candidates using the feature layer prepared during Phase 3. The evaluation focuses on candidate model performance, precision, recall, F1, PR-AUC, ROC-AUC, false-positive and false-negative behavior, fraud-scenario-level detection, and controlled Random Forest experimentation. Champion selection is a team decision and remains pending at this stage.



\## 2. Evaluation Setup



\### Training Dataset

\- Dataset: `data/phase3/train\_feast\_features.csv`

\- Period: April 1–24

\- Rows: 230,295

\- Fraud transactions: 1,176



\### Evaluation Dataset

\- Dataset: `data/phase4/april\_25\_30\_evaluation\_features.csv`

\- Period: April 25–30

\- Rows: 57,767

\- Fraud transactions: 526



The same April 25–30 evaluation dataset was used for all four candidates.



\### Model Features

\- `TX\_AMOUNT`

\- `AMOUNT\_LOG`

\- `TX\_HOUR`

\- `TX\_DAY\_OF\_WEEK`

\- `TX\_IS\_WEEKEND`

\- `TERMINAL\_TX\_COUNT\_1D`

\- `TERMINAL\_TX\_COUNT\_7D`

\- `TERMINAL\_AVG\_AMOUNT\_7D`

\- `CUSTOMER\_TX\_COUNT\_1D`

\- `CUSTOMER\_TX\_COUNT\_7D`



\### Data Isolation

May 1–19 data was not used for model selection or tuning. The April 25–30 period was retained as the evaluation period for the Phase 4 candidate comparison.



\## 3. Candidate Models



\### Logistic Regression — Original

\- Model: Logistic Regression

\- Class weighting: None

\- `C`: 0.01

\- Threshold: 0.10



\### Random Forest — 100 Trees

\- Model: Random Forest

\- `n\_estimators`: 100

\- Class weighting: `balanced`

\- `max\_depth`: `None`

\- `min\_samples\_leaf`: 1

\- Threshold: 0.65

\- Random state: 42



\### Random Forest — 300 Trees

\- Model: Random Forest

\- `n\_estimators`: 300

\- Class weighting: `balanced`

\- `max\_depth`: `None`

\- `min\_samples\_leaf`: 1

\- Threshold: 0.65

\- Random state: 42



\### Logistic Regression — Balanced

\- Model: Logistic Regression

\- Class weighting: `balanced`

\- `C`: 10

\- Threshold: 0.95



\## 4. Four-Candidate Evaluation Results

All four candidates were evaluated on the same April 25–30 evaluation dataset.



| Candidate | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | TP | FP | TN | FN |

|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| LR Original C=0.01 | 0.10 | 0.2721 | 0.6783 | 0.7744 | 0.2414 | 0.3681 | 0.000646 | 127 | 37 | 57204 | 399 |

| RF Balanced 100 | 0.65 | 0.2537 | 0.6255 | 0.9919 | 0.2338 | 0.3785 | 0.000017 | 123 | 1 | 57240 | 403 |

| RF Balanced 300 | 0.65 | 0.2574 | 0.6370 | 1.0000 | 0.2338 | 0.3790 | 0.000000 | 123 | 0 | 57241 | 403 |

| LR Balanced C=10 | 0.95 | 0.2593 | 0.6795 | 0.9770 | 0.1616 | 0.2773 | 0.000035 | 85 | 2 | 57239 | 441 |



These results are evaluation evidence for team review. They do not designate a Champion or establish a ranking of the candidates.



\## 5. Controlled RF-100 → RF-300 Experiment

A controlled Random Forest experiment was performed by changing only the number of trees:



`RF-100 → RF-300`



\### Controlled Change

`n\_estimators: 100 → 300`



\### Variables Kept Constant

\- Class weighting: `balanced`

\- `max\_depth`: `None`

\- `min\_samples\_leaf`: 1

\- Random state: 42

\- Threshold: 0.65

\- Same 10 model features

\- Training period: April 1–24

\- Evaluation period: April 25–30



May 1–19 data remained excluded from model selection and tuning.



\### RF-300 Results

\- PR-AUC: 0.2574

\- ROC-AUC: 0.6370

\- Precision: 1.0000

\- Recall: 0.2338

\- F1: 0.3790

\- FPR: 0.0000

\- TP: 123

\- FP: 0

\- TN: 57,241

\- FN: 403



\### Observed Difference from RF-100

RF-100:

\- PR-AUC: 0.2537

\- ROC-AUC: 0.6255

\- Precision: 0.9919

\- Recall: 0.2338

\- F1: 0.3785

\- FP: 1

\- FN: 403



RF-300:

\- PR-AUC: 0.2574

\- ROC-AUC: 0.6370

\- Precision: 1.0000

\- Recall: 0.2338

\- F1: 0.3790

\- FP: 0

\- FN: 403



The controlled experiment showed a small change in ranking metrics and precision, while recall and the number of false negatives remained unchanged. These observations are recorded for team review and do not by themselves determine Champion selection.



\## 6. Transaction-Level Error Analysis



\### RF-300

On the April 25–30 evaluation dataset:

\- True Positives: 123

\- False Positives: 0

\- True Negatives: 57,241

\- False Negatives: 403



At the fixed threshold of 0.65, RF-300 produced no false-positive transactions while 403 fraudulent transactions were classified as negative.



The false-negative analysis is descriptive and does not establish causal reasons for individual model errors.



\### False Positives

RF-300 produced 0 false positives. Therefore, there are no RF-300 transaction rows to add to the false-positive dataset.



The existing `phase4\_false\_positives.csv` remains unchanged because it contains actual false-positive transactions from candidates that produced false positives. No artificial RF-300 row is added to represent the zero count.



\### False Negatives

The false-negative analysis was updated to include all four candidates.



For RF-300:

\- False Negatives: 403



\## 7. Fraud Scenario Analysis

The 526 fraudulent transactions in the April 25–30 evaluation period were distributed as follows:



| Scenario | Fraud Transactions | Percentage |

|---|---:|---:|

| Scenario 1 | 43 | 8.17% |

| Scenario 2 | 324 | 61.60% |

| Scenario 3 | 159 | 30.23% |

| \*\*Total\*\* | \*\*526\*\* | \*\*100%\*\* |



\### RF-300 Scenario Results



| Scenario | Total Fraud | TP | FN | Detection Rate |

|---|---:|---:|---:|---:|

| Scenario 1 | 43 | 43 | 0 | 100.00% |

| Scenario 2 | 324 | 1 | 323 | 0.31% |

| Scenario 3 | 159 | 79 | 80 | 49.69% |



\### Scenario 2 Observation

Scenario 2 represents 324 of the 526 fraud transactions in the evaluation dataset. RF-300 detected 1 of the 324 Scenario 2 fraud transactions and missed 323.



Of the 403 total RF-300 false negatives, 323 were Scenario 2 transactions:



`323 / 403 = 80.15%`



This is an important evaluation finding and requires further investigation. The current analysis does not establish the underlying cause of the Scenario 2 detection behavior.



\## 8. Feature-Level Error Analysis

Feature-level analysis was performed for false-negative and false-positive transactions to identify observed patterns in model errors.



The analysis is descriptive and does not establish that a particular feature caused a transaction to be classified incorrectly.



Generated analysis files:

\- `reports/phase4/phase4\_fn\_feature\_analysis.csv`

\- `reports/phase4/phase4\_fp\_feature\_analysis.csv`



Transaction-level predictions:

\- `reports/phase4/phase4\_transaction\_predictions.csv`



\## 9. Phase 4 Evidence Files

The Phase 4 evaluation package contains:

\- `reports/phase4/phase4\_candidate\_evidence.csv` — candidate configurations, metrics, confusion-matrix values and scenario evidence.

\- `reports/phase4/phase4\_transaction\_predictions.csv` — transaction-level predictions and probabilities.

\- `reports/phase4/phase4\_false\_negatives.csv` — false-negative transactions.

\- `reports/phase4/phase4\_false\_positives.csv` — false-positive transactions.

\- `reports/phase4/phase4\_fn\_feature\_analysis.csv` — feature-level false-negative analysis.

\- `reports/phase4/phase4\_fp\_feature\_analysis.csv` — feature-level false-positive analysis.

\- `reports/phase4/phase4\_scenario\_analysis.csv` — scenario-level detection results for all four candidates.



\## 10. Reproducibility and Validation

The four-candidate evaluation was reproduced using:



`scripts/evaluate\_phase4\_candidates.py`



Candidate evidence was generated and validated using:



`scripts/create\_phase4\_candidate\_evidence.py`



Scenario analysis was generated using:



`scripts/generate\_phase4\_scenario\_analysis.py`



The scenario-analysis generator is present in the repository at:



`scripts/generate\_phase4\_scenario\_analysis.py`



Error analysis was updated to include RF-300 using:



`scripts/error\_analysis\_phase4.py`



Validation checks confirmed:

\- All four candidates were evaluated.

\- Evaluation dataset shape and fraud count were consistent.

\- Confusion-matrix counts were internally consistent.

\- Scenario totals matched the expected fraud count: S1 = 43, S2 = 324, S3 = 159, total = 526.

\- RF-300 false-positive count was 0.

\- May data was not used for model selection or tuning.

\- The updated candidate evidence and scenario analysis contain RF-300 results.



\## 11. Limitations

1\. Evaluation was performed on the fixed April 25–30 period.

2\. May 1–19 data was excluded from model selection and tuning.

3\. Scenario analysis is descriptive.

4\. Feature-level error analysis does not establish causation.

5\. The RF-100 → RF-300 experiment changed only `n\_estimators`.

6\. The RF experiment retained the fixed threshold of 0.65.

7\. Scenario 2 detection remains an unresolved investigation point.

8\. No additional model or threshold experiment is included after the RF-300 controlled evaluation.

9\. Champion selection has not been finalized.



\## 12. Phase 4 Status and Handoff

The Phase 4 evaluation responsibility is considered complete for the current candidate-evaluation stage.



The completed package contains:

\- Four-candidate evaluation

\- RF-100 → RF-300 controlled experiment

\- Transaction-level predictions

\- False-negative analysis

\- False-positive analysis

\- Feature-level error analysis

\- Scenario analysis

\- Candidate evidence

\- Reproducibility and validation checks



The evidence is ready for team review and handoff.



Further model investigation, including investigation of Scenario 2/3 behavior and any additional controlled experiments, should be performed as a subsequent team activity if justified by the evidence.



\### Champion Status

\*\*Champion selection: Pending team decision.\*\*



This document does not designate a Champion or rank the candidates.



\## 13. Phase 4 Handoff Boundary

Diya's current Phase 4 evaluation responsibility ends after:

1\. Final repository verification

2\. Verification of the Phase 4 documentation and evidence

3\. Commit and push of the intended Phase 4 changes

4\. Handoff of the evaluation package to the team



Further investigation or experimentation after this handoff is a subsequent team activity and is not part of the completed Phase 4 evaluation package documented here.


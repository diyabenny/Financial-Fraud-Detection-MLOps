# Phase 4 — Model Evaluation, Error Analysis, and Pre-Champion Handoff

## 1. Objective

Phase 4 focused on evaluating shortlisted fraud-detection candidates using a controlled temporal evaluation process, selecting candidate-specific operating thresholds, performing error and scenario analysis, and consolidating reproducibility evidence for team-level Champion discussion.

The Phase 4 workflow covered:

- Candidate evaluation on a common April 25–30 evaluation period
- Candidate-specific threshold selection using earlier April data
- Controlled comparison of selected Logistic Regression and Random Forest candidates
- A controlled RF-300 experiment
- A controlled XGBoost experiment
- False-negative and false-positive analysis
- Scenario-level analysis
- Feature-level error analysis
- MLflow and reproducibility verification
- Feast historical and online retrieval verification
- Evidence consolidation for pre-Champion handoff

SMOTENC and other imbalance approaches were explored in earlier Phase 4 experimentation but were not introduced as additional controlled candidates in the final Phase 4 candidate comparison.

Phase 4 does **not** declare a Champion. Champion selection and subsequent final validation are outside the scope of this document.

---

## 2. Round 1 Candidate Tuning and Carry-Forward

### 2.1 Temporal Tuning Protocol

Round 1 tuning used a temporal split:

- **Tuning training period:** April 1–18, 2018
- **Tuning validation period:** April 19–24, 2018
- **Threshold during Round 1 tuning:** 0.5

The purpose was to compare candidate configurations using a time-ordered validation period while preserving temporal separation between training and validation data.

### 2.2 Round 1 Configuration Space

A total of **32 configurations** were evaluated.

#### Logistic Regression

Eight Logistic Regression configurations were evaluated:

- Original class weighting:
  - C = 0.01
  - C = 0.1
  - C = 1
  - C = 10
- Balanced class weighting:
  - C = 0.01
  - C = 0.1
  - C = 1
  - C = 10

The Logistic Regression implementation used:

- `StandardScaler`
- `LogisticRegression`
- `random_state=42`
- `max_iter=1000`
- Configurable `C`
- `class_weight=None` or `class_weight="balanced"`

#### Random Forest

Twenty-four balanced Random Forest configurations were evaluated using:

- `n_estimators`: 100, 300
- `max_depth`: None, 10, 20
- `min_samples_leaf`: 1, 5
- `class_weight="balanced"`
- `random_state=42`
- `n_jobs=-1`

### 2.3 Carry-Forward Candidates

The following configurations were carried forward from Round 1 for Phase 4 candidate evaluation:

1. Logistic Regression — Original — C=0.01
2. Random Forest — Balanced — 100 estimators
3. Logistic Regression — Balanced — C=10

These three configurations formed the Round 1 carry-forward set. RF-300 and controlled XGBoost were subsequently added as controlled Phase 4 candidates.

---

## 3. Evaluation Dataset and Temporal Protocol

### 3.1 Training Dataset

The candidate models were trained using:

`data/phase3/train_feast_features.csv`

Training period:

- **April 1–24, 2018**
- Rows: **230,295**
- Legitimate transactions: **229,119**
- Fraudulent transactions: **1,176**

### 3.2 Common Evaluation Dataset

All controlled candidate evaluations used:

`data/phase4/april_25_30_evaluation_features.csv`

Evaluation period:

- **April 25–30, 2018**
- Rows: **57,767**
- Legitimate transactions: **57,241**
- Fraudulent transactions: **526**

The evaluation period occurs after the candidate training period.

However, April 25–30 is not described as a completely untouched holdout because it had been accessed earlier for preliminary Phase 4 screening. It was not used for final candidate fitting or threshold selection.

### 3.3 Approved Model Features

The approved feature set contained ten model inputs:

```text
TX_AMOUNT
AMOUNT_LOG
TX_HOUR
TX_DAY_OF_WEEK
TX_IS_WEEKEND
TERMINAL_TX_COUNT_1D
TERMINAL_TX_COUNT_7D
TERMINAL_AVG_AMOUNT_7D
CUSTOMER_TX_COUNT_1D
CUSTOMER_TX_COUNT_7D
```

The following were not used as model inputs:

- Transaction identifiers
- Raw timestamps
- `TX_FRAUD_SCENARIO`

`TX_FRAUD_SCENARIO` was excluded from model inputs to avoid leakage and was retained for downstream analysis where available.

---

## 4. Original Candidate Threshold Selection

Candidate-specific thresholds were selected using an earlier temporal portion of the April training period rather than the common April 25–30 evaluation period.

### 4.1 Threshold Selection Protocol

The threshold-selection process used:

- **Fitting period:** April 1–12, 2018
- **Threshold-selection period:** April 13–18, 2018
- Threshold search range: **0.01 to 0.95**
- Selection criterion: **maximum F1**
- Overlap between fitting and threshold-selection periods: **0**

Dataset sizes:

| Period | Rows | Legitimate | Fraud |
|---|---:|---:|---:|
| April 1–12 | 114,978 | 114,610 | 368 |
| April 13–18 | 57,525 | 57,176 | 349 |

The three Round 1 carry-forward candidates were retrained using the complete April 1–24 training period before evaluation on April 25–30. The subsequently added RF-300 controlled candidate reused the RF-100 threshold of 0.65, while controlled XGBoost used its separately selected threshold of 0.86.
### 4.2 Selected Thresholds

| Candidate | Selected Threshold |
|---|---:|
| LR Original C=0.01 | 0.10 |
| RF Balanced 100 | 0.65 |
| LR Balanced C=10 | 0.95 |

---

## 5. Candidate Configuration Summary

| Candidate | Model Configuration | Threshold |
|---|---|---:|
| LR Original C=0.01 | StandardScaler → LogisticRegression, C=0.01, class_weight=None, random_state=42, max_iter=1000 | 0.10 |
| RF Balanced 100 | RandomForest, n_estimators=100, max_depth=None, min_samples_leaf=1, class_weight=balanced, random_state=42, n_jobs=-1 | 0.65 |
| RF Balanced 300 | RandomForest, n_estimators=300, max_depth=None, min_samples_leaf=1, class_weight=balanced, random_state=42, n_jobs=-1 | 0.65 |
| LR Balanced C=10 | StandardScaler → LogisticRegression, C=10, class_weight=balanced, random_state=42, max_iter=1000 | 0.95 |
| Controlled XGBoost | StandardScaler → XGBoost, n_estimators=300, max_depth=6, learning_rate=0.05, objective=binary:logistic, random_state=42, n_jobs=-1, eval_metric=logloss | 0.86 |

---

## 6. Candidate Aggregate Evaluation

All aggregate candidate metrics below were calculated on the common **April 25–30, 2018** evaluation dataset.

### 6.1 Aggregate Metrics

| Candidate | Threshold | PR-AUC | ROC-AUC | Recall | Precision | F1 | Accuracy | FPR (fraction) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LR Original C=0.01 | 0.10 | 0.272065 | 0.678308 | 0.241445 | 0.774390 | 0.368116 | 0.992452 | 0.000646 |
| RF Balanced 100 | 0.65 | 0.253683 | 0.625461 | 0.233840 | 0.991935 | 0.378462 | 0.993006 | 0.000017 |
| RF Balanced 300 | 0.65 | 0.257441 | 0.637033 | 0.233840 | 1.000000 | 0.379045 | 0.993024 | 0.000000 |
| LR Balanced C=10 | 0.95 | 0.259335 | 0.679460 | 0.161597 | 0.977011 | 0.277325 | 0.992331 | 0.000035 |
| Controlled XGBoost | 0.86 | 0.141099 | 0.628685 | 0.233840 | 0.480469 | 0.314578 | 0.990721 | 0.002324 |

### 6.2 Confusion-Matrix Counts

| Candidate | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| LR Original C=0.01 | 127 | 37 | 57,204 | 399 |
| RF Balanced 100 | 123 | 1 | 57,240 | 403 |
| RF Balanced 300 | 123 | 0 | 57,241 | 403 |
| LR Balanced C=10 | 85 | 2 | 57,239 | 441 |
| Controlled XGBoost | 123 | 133 | 57,108 | 403 |

The controlled XGBoost results are included in the aggregate candidate evidence for team comparison. The results apply only to the fixed XGBoost configuration tested in this experiment.

---

## 7. RF-300 Controlled Candidate

RF-300 was introduced as a controlled experiment based on the RF-100 configuration.

### 7.1 Controlled Change

Only the number of estimators was changed:

```text
RF-100 → RF-300
```

The following were held constant:

- Model family: Random Forest
- `class_weight="balanced"`
- `max_depth=None`
- `min_samples_leaf=1`
- `random_state=42`
- `n_jobs=-1`
- Approved feature set
- Training period: April 1–24
- Threshold: 0.65
- Evaluation period: April 25–30

No broad Random Forest hyperparameter search was introduced as part of this experiment.

### 7.2 RF-300 Evaluation

RF-300 produced:

- PR-AUC: **0.257441**
- ROC-AUC: **0.637033**
- Recall: **0.233840**
- Precision: **1.000000**
- F1: **0.379045**
- Accuracy: **0.993024**
- FPR: **0**
- TP: **123**
- FP: **0**
- TN: **57,241**
- FN: **403**

The RF-300 experiment was treated as a controlled candidate comparison and not as a Champion decision.

---

## 8. Controlled XGBoost Experiment

### 8.1 Objective

A fixed XGBoost configuration was evaluated as an additional controlled Phase 4 candidate.

No XGBoost hyperparameter tuning was performed as part of this experiment.

### 8.2 Configuration

```text
n_estimators=300
max_depth=6
learning_rate=0.05
objective=binary:logistic
random_state=42
n_jobs=-1
eval_metric=logloss
```

The model was implemented using a `StandardScaler` inside a scikit-learn pipeline.

The approved ten-feature set was unchanged.

No SMOTE, SMOTENC, ADASYN, or other additional resampling method was introduced for this controlled XGBoost experiment.

### 8.3 Temporal Procedure

#### Stage 1: Initial fitting

Period:

- April 1–12, 2018
- Rows: 114,978
- Legitimate: 114,610
- Fraud: 368
- `scale_pos_weight`: 311.4402173913044

#### Threshold Selection

Period:

- April 13–18, 2018
- Rows: 57,525
- Legitimate: 57,176
- Fraud: 349
- Selection criterion: maximum F1
- Selected threshold: **0.86**
- Overlap with Stage 1: **0**

#### Stage 2 Retraining for Final Evaluation

Period:

- April 1–24, 2018
- Rows: 230,295
- Legitimate: 229,119
- Fraud: 1,176
- `scale_pos_weight`: 194.82908163265307
- Overlap with April 25–30 evaluation period: **0**

#### Final Evaluation

Period:

- April 25–30, 2018
- Rows: 57,767
- Legitimate: 57,241
- Fraud: 526
- Threshold: **0.86**

### 8.4 Controlled XGBoost Results

| Metric | Result |
|---|---:|
| PR-AUC | 0.141099 |
| ROC-AUC | 0.628685 |
| Precision | 0.480469 |
| Recall | 0.233840 |
| F1 | 0.314578 |
| Accuracy | 0.990721 |
| FPR | 0.002324 |
| TP | 123 |
| FP | 133 |
| TN | 57,108 |
| FN | 403 |

### 8.5 XGBoost Evidence Boundary

The controlled XGBoost evidence available for Phase 4 consists of:

- Aggregate evaluation metrics
- Confusion-matrix results
- MLflow experiment/model evidence
- Provenance information
- Threshold results

A separate transaction-level prediction artifact with scenario-level XGBoost analysis was not available.

Therefore, XGBoost is not retrofitted into the transaction-level error-analysis or scenario-analysis artifacts.

### 8.6 Historical XGBoost Screening Evidence

Earlier XGBoost screening runs could not be confidently reconciled with the documented controlled configurations because the saved configuration information and original metadata/artifact structure were incomplete or inconsistent.

Historical X01/X02/X03 results are therefore retained as historical/unverified screening evidence and are not treated as controlled candidate evidence.

The historical XGBoost screening was not rerun.

### 8.7 MLflow Serialization Correction

During model logging, the MLflow/skops serialization step required explicit trusted XGBoost types. The logging configuration was corrected in commit `843bb87`; the model configuration and experiment results were not changed, and the experiment was not rerun.

---

## 9. False-Negative Analysis

False-negative analysis was performed to identify patterns among fraudulent transactions that were not detected by the evaluated candidates.

The analysis was supported by:

- Transaction-level prediction evidence
- False-negative records
- Feature-level false-negative analysis

The analysis was used to understand observed model errors and identify areas for further investigation.

It was not used to declare a Champion.

---

## 10. False-Positive and Feature-Level Analysis

False-positive analysis was performed for candidates with false-positive predictions on the common April 25–30 evaluation period.

Supporting artifacts include:

- False-positive records
- Feature-level false-positive analysis

The analysis provides candidate-specific error evidence and supports interpretation of aggregate evaluation results.

---

## 11. Scenario-Level Analysis

Scenario-level analysis was performed using the fraud-scenario labels available in the evaluation data.

### 11.1 Fraud Scenario Distribution

| Scenario | Fraud Count | Percentage |
|---|---:|---:|
| S1 | 43 | 8.17% |
| S2 | 324 | 61.60% |
| S3 | 159 | 30.23% |
| **Total** | **526** | **100%** |

### 11.2 RF-300 Scenario Detection

For RF-300:

| Scenario | Detected | Total Fraud | Detection Rate |
|---|---:|---:|---:|
| S1 | 43 | 43 | 100.00% |
| S2 | 1 | 324 | 0.31% |
| S3 | 79 | 159 | 49.69% |

RF-300 produced:

- TP: 123
- FN: 403

Of the 403 RF-300 false negatives:

- **323** were Scenario 2
- **80.15%** of RF-300 false negatives were therefore associated with Scenario 2

The RF-300 scenario evidence therefore identifies Scenario 2 as an observed detection limitation for RF-300.

### 11.3 Historical Terminal-Fraud Diagnostic

A historical terminal-fraud diagnostic had previously been investigated as a possible explanation for the Scenario 2 pattern. However, the underlying diagnostic calculation and its denominator could not be located in the current repository evidence. Therefore, previously reported historical terminal-fraud percentages and mean counts are not treated as verified Phase 4 evidence and are not reproduced here.

`TERMINAL_PREVIOUS_FRAUD_COUNT` was not added to the production feature set.

### 11.4 Velocity Diagnostic

A previously discussed velocity diagnostic was not traceable to a verified Phase 4 repository artifact during the final verification pass. Therefore exact velocity feature names, previously reported velocity metrics, and velocity-based improvement claims are not included.

No velocity feature is added to the approved Phase 4 feature set.

---

## 12. Candidate Evidence Consolidation

The Phase 4 candidate evidence package consolidates the available evaluation and error-analysis outputs.

### Candidate Evidence Coverage

The candidate evidence package contains five candidate records:

1. `LR_Original_C_0.01`
2. `RF_Balanced_100_None_1`
3. `RF_Balanced_300_None_1`
4. `LR_Balanced_C_10`
5. `Controlled_XGBoost`

Supporting coverage includes:

| Evidence Artifact | Candidate Coverage |
|---|---:|
| Transaction predictions | 4 candidates |
| False negatives | 4 candidates |
| FN feature analysis | 4 candidates |
| False positives | 3 candidates |
| FP feature analysis | 3 candidates |
| Scenario analysis | 4 candidates |
| Candidate evidence | 5 candidates |

The controlled XGBoost candidate remains represented in the aggregate candidate evidence, while its transaction-level and scenario-level evidence is explicitly bounded by the artifacts available.

---

## 13. MLflow and Reproducibility Verification

### 13.1 Controlled XGBoost MLflow Evidence

The controlled XGBoost experiment was logged to:

```text
Experiment:
financial_fraud_detection_phase4_controlled_xgboost

Experiment ID:
973447774010422392

Run ID:
3bfe6002d0fe49e68597ba0ad89b818c

Model ID:
m-ce7a1fe747c142c3aa514bc5ef01f766

Model Name:
controlled_xgboost_model

Model Status:
READY
```

The logged model was loaded successfully and inference was verified on five known April 25–30 rows.

### 13.2 Experiment Provenance

The controlled XGBoost experiment source commit was:

```text
0a35d56426c4f546a08ed15205f2e08b098feaea
```

Short form:

```text
0a35d56
```

The later MLflow serialization fix was recorded in:

```text
843bb87
```

The fix affected model logging configuration only. The model configuration and experiment results were not changed, and the experiment was not rerun.

---

## 14. Feast Sequencing and Data Usage

Feast retrieval was verified for the approved feature set.

### 14.1 Historical Retrieval

Historical Feast retrieval was verified for the training and validation datasets.

The retrieved features matched the Phase 3 feature outputs with:

```text
Maximum absolute difference = 0
```

This verified consistency between the Feast historical retrieval and the corresponding Phase 3 feature outputs.

### 14.2 Online Retrieval

Online Feast retrieval was verified using known April transaction data.

The behavioral features respected the temporal requirement:

```text
previous_time < current_time
```

This verified that the behavioral feature retrieval did not use the current transaction itself when constructing historical context.

### 14.3 May Data Usage Boundary

May 1–19 had previously been accessed for technical Feast validation, but was excluded from:

- Model selection
- Feature selection
- Threshold selection
- Hyperparameter selection
- Champion selection

The May dataset was therefore not used as the final Phase 4 evaluation dataset.

---

## 15. Evidence Package

The following Phase 4 evidence artifacts were verified:

```text
reports/phase4/phase4_candidate_evidence.csv
reports/phase4/phase4_transaction_predictions.csv
reports/phase4/phase4_false_negatives.csv
reports/phase4/phase4_fn_feature_analysis.csv
reports/phase4/phase4_false_positives.csv
reports/phase4/phase4_fp_feature_analysis.csv
reports/phase4/phase4_scenario_analysis.csv
reports/phase4/controlled_xgboost/provenance.json
reports/phase4/controlled_xgboost/threshold_results.csv
```

Supporting scripts verified include:

```text
scripts/analyze_phase4_fn.py
scripts/analyze_phase4_fn_features.py
scripts/analyze_phase4_fp.py
scripts/analyze_phase4_scenarios.py
scripts/create_phase4_candidate_evidence.py
scripts/error_analysis_phase4.py
scripts/evaluate_phase4_candidates.py
scripts/generate_phase4_scenario_analysis.py
scripts/run_controlled_xgboost_phase4.py
scripts/screen_xgboost_phase4.py
scripts/threshold_analysis_phase4.py
scripts/train_phase4.py
scripts/tune_phase4.py
```

---

## 16. Champion Selection Boundary

Phase 4 is a **pre-Champion evaluation and evidence-consolidation stage**.

This document does not:

- Declare a Champion
- Identify a best-performing model
- Rank candidates
- Freeze a Champion
- Perform final Champion training
- Perform final May 1–19 Champion evaluation
- Make production-readiness claims

The candidate evidence is intended for team review and subsequent Champion discussion.

The subsequent workflow is:

```text
Phase 4 Candidate Evaluation
        ↓
Team Champion Selection
        ↓
Champion Freeze
        ↓
Final Champion Training / Governance
        ↓
May 1–19 Final Champion Evaluation
        ↓
Final Validation / Governance
        ↓
Production / Deployment / Monitoring
```

Only activities that are actually implemented and verified should be documented in the subsequent phases.

---

## 17. Phase 4 Handoff

Phase 4 hands off the following evidence for team review:

- Candidate configuration details
- Candidate-specific thresholds
- Aggregate evaluation metrics
- Controlled RF-300 experiment
- Controlled XGBoost experiment
- False-negative analysis
- False-positive analysis
- Feature-level error analysis
- Scenario-level analysis
- MLflow reproducibility evidence
- Feast retrieval verification
- Candidate evidence artifacts
- Explicit evidence boundaries and unresolved historical diagnostics

The handoff does not include a Champion decision.

---

## 18. Documentation Status

Phase 4 documentation is considered complete for team review after:

1. The documented candidate and error-analysis evidence has been verified.
2. The controlled XGBoost evidence boundary has been documented.
3. Historical diagnostics that could not be reproduced or traced have been excluded from verified evidence.
4. Feast sequencing and May-data usage boundaries have been documented.
5. The evidence package has been consolidated.
6. Documentation formatting and Git diff checks have been completed.

No additional Phase 4 experiment, XGBoost rerun, Scenario 2 investigation, Feast work, or restructuring is required for this documentation handoff.

---

## 19. Phase 4 Scope Boundary

Phase 4 covers:

```text
Candidate Evaluation
        +
Threshold Methodology
        +
Controlled Experiments
        +
Error Analysis
        +
Scenario Analysis
        +
Reproducibility Verification
        +
Evidence Consolidation
        +
Pre-Champion Handoff
```

The following remain outside Phase 4:

```text
Champion Selection
Champion Freeze
Final Champion Training
May 1–19 Final Champion Evaluation
Final Production Validation
Deployment
Production Automation
Production Monitoring
```

These activities should be documented only when they are subsequently performed and verified.


---

## 20. Post-Phase 4 Champion Decision and Freeze

Following the Phase 4 candidate evaluation and team review, the team unanimously selected the following candidate as the project Champion:

```text
Candidate: RF_Balanced_300_None_1

Model:
Random Forest

Configuration:
n_estimators=300
max_depth=None
min_samples_leaf=1
class_weight=balanced
random_state=42
n_jobs=-1

Operating threshold:
0.65
```

The decision considered the verified Phase 4 candidate evidence, including fraud detection effectiveness, false-positive burden, PR-AUC, F1, ROC-AUC, confusion-matrix counts, and scenario-level evidence. Evidence completeness and reproducibility were applied as the agreed eligibility/traceability condition.
The decision did not involve a new experiment, retraining, threshold change, feature change, or additional XGBoost experiment.

### 20.1 Frozen Evaluation Evidence

The verified Phase 4 evaluation evidence for RF Balanced 300 on the April 25–30 evaluation period is:

| Metric    | RF Balanced 300 |
| --------- | --------------: |
| PR-AUC    |        0.257441 |
| ROC-AUC   |        0.637033 |
| Recall    |        0.233840 |
| Precision |        1.000000 |
| F1        |        0.379045 |
| FPR       |               0 |
| TP        |             123 |
| FP        |               0 |
| TN        |          57,241 |
| FN        |             403 |

Scenario evidence:

| Scenario | Detected | Total Fraud | Detection Rate |
| -------- | -------: | ----------: | -------------: |
| S1       |       43 |          43 |        100.00% |
| S2       |        1 |         324 |          0.31% |
| S3       |       79 |         159 |         49.69% |

### 20.2 Champion Freeze Boundary

The Champion Freeze establishes the selected model configuration and operating threshold for subsequent implementation.

It does not by itself constitute:

* Final Champion retraining
* May 1–19 final Champion evaluation
* Final production validation
* Deployment
* Production automation
* Production monitoring

Those activities remain pending until they are actually implemented and verified.

The existing Phase 4 candidate evidence remains the source of record for the documented candidate evaluation.

---

## 21. Isolated Step 18 Candidate Rerun — 10 October 2026

### 21.1 Purpose and Scope

An isolated rerun of the Step 18 candidate evaluation was performed using the existing RF300 Step 17 candidate artifact and the approved April 25–30 evaluation dataset.

The isolated evaluator was:

```text
scripts\evaluate_candidate_step18_isolated_20261010.py
```

The original evaluator, `scripts\evaluate_candidate_step18.py`, was preserved. The isolated copy was configured to write its evidence to a separate output directory. No retraining or Champion promotion was performed.

The evaluation used the existing ten-feature configuration and operating threshold of `0.65`.

### 21.2 Evaluation Dataset

```text
data\phase4\april_25_30_evaluation_features.csv
```

| Dataset property | Value |
|---|---:|
| Evaluation rows | 57,767 |
| Fraud cases | 526 |
| Legitimate transactions | 57,241 |
| Operating threshold | 0.65 |

### 21.3 Candidate Metrics

Candidate identity: `RF300_STEP17_CANDIDATE`

| Metric | Isolated candidate | Historical RF300 Champion benchmark |
|---|---:|---:|
| PR-AUC | 0.2571243664 | 0.2574410304 |
| ROC-AUC | 0.6330922031 | 0.6370331816 |
| Precision | 1.000000 | 1.000000 |
| Recall | 0.2319391635 | 0.2338403042 |
| F1 | 0.3765432099 | 0.3790446841 |
| False-positive rate | 0 | 0 |
| True positives | 122 | 123 |
| False positives | 0 | 0 |
| True negatives | 57,241 | 57,241 |
| False negatives | 404 | 403 |

The candidate was slightly below the historical RF300 Champion benchmark on PR-AUC, ROC-AUC, recall, and F1. At threshold `0.65`, it detected one fewer fraud case and produced one additional false negative. Precision and false-positive rate were unchanged.

These results document the isolated candidate evaluation only. **No promotion decision was made**, and the frozen Champion remains unchanged.

### 21.4 Evidence Files

The isolated evaluator reported writing the following files:

```text
reports\phase4\step18_candidate_evaluation_rerun_20261010\candidate_evaluation_metrics.txt
reports\phase4\step18_candidate_evaluation_rerun_20261010\candidate_predictions.csv
```

The saved metrics file was inspected and matched the reported candidate metrics and historical benchmark.

The predictions CSV was confirmed to exist. Its header is:

```text
TRANSACTION_ID,TX_FRAUD,candidate_probability,candidate_prediction
```

The first five lines were inspected. A full-row validation of the predictions CSV is not claimed here.

The following manifest was checked for before editing:

```text
reports\phase4\step18_candidate_evaluation_rerun_20261010\evaluation_manifest.txt
```

**Status: NOT FOUND.** No claim is made that this manifest exists or was created.

### 21.5 Protected-File Integrity Verification

After the isolated evaluation, all five protected-file SHA-256 checksums were rechecked against their recorded baselines. All five matched.

| Protected file | Verification result |
|---|---|
| `data\phase4\april_25_30_evaluation_features.csv` | PASS |
| `artifacts\rf300_champion\model.skops` | PASS |
| `artifacts\candidates\rf300_candidate\model.skops` | PASS |
| `reports\phase4\step18_candidate_evaluation\candidate_evaluation_metrics.txt` | PASS |
| `reports\phase4\step18_candidate_evaluation\candidate_predictions.csv` | PASS |

Recorded SHA-256 values:

```text
Evaluation dataset:
22b485eafab1e343df2e9cc691a35af68fab778e5c0bfa8a2b306fccfc48b5c3

Frozen Champion artifact:
a28e5e10dbe7478f1f446b8fd55fc9457984f0f3b719371bb0934e63fa58ed83

Candidate artifact:
ad2e9d7e598558fe660137ce4320a52e83e10b2b228c831d4a308e4e4fccac32

Original Step 18 metrics:
f7e31cc176612d91ca2620df257dea5dcfbd456376786eca43b4178447e75a02

Original Step 18 predictions:
b7c01c308a089d922eed548e159aac207a5560a79e33d7e3436f713cdc31b541
```

This verification establishes that the five protected files matched their recorded reference hashes. The original Step 18 evidence and frozen Champion were preserved.

### 21.6 MLflow Provenance Limitation

The isolated metrics file records the following Step 17 MLflow run ID:

```text
ca4b5285656c4906a01502115774347d
```

However, the direct provenance link between that MLflow run and the candidate artifact has not been independently established.

**Candidate-to-MLflow provenance: NEEDS VERIFICATION.**

The recorded run ID is reported for traceability only and is not presented as proof that the candidate artifact was produced by that run.

### 21.7 Decision and Change Boundary

The isolated rerun was an evaluation-only activity. No retraining or candidate promotion was performed. The frozen Champion, candidate artifact, evaluation dataset, original evaluator, and original Step 18 evidence were preserved.

This section records the rerun evidence and does not reopen the Champion decision or authorize changes to the model, features, preprocessing, threshold, MLflow tracking store, or deployment configuration.

The documentation change does not authorize a commit or push.

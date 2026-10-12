# Financial Fraud Detection MLOps Platform

An end-to-end MLOps project for developing, evaluating, and integrating a machine-learning workflow for financial transaction fraud detection.

## Project Overview

The project treats fraud detection as an imbalanced binary-classification problem:

- `TX_FRAUD = 0`: Legitimate transaction
- `TX_FRAUD = 1`: Fraudulent transaction

Because fraudulent transactions are rare, evaluation emphasizes recall, precision, F1-score, PR-AUC, ROC-AUC, false-positive rate, and confusion-matrix counts rather than accuracy alone.

## Technology Stack

The project's declared dependencies include:

- **Programming and data processing:** Python, Pandas, NumPy, PyArrow
- **Machine learning:** Scikit-learn, XGBoost, imbalanced-learn
- **Feature store:** Feast
- **Experiment tracking:** MLflow
- **Streaming:** Kafka producer and consumer using `confluent-kafka==2.16.0` (declared in `requirements.txt`)
- **API:** FastAPI, Uvicorn, Pydantic
- **Model serialization:** skops
- **Cloud and data versioning:** AWS S3, boto3, DVC
- **Containerization:** Docker

A declared dependency does not by itself prove that its associated service or integration is configured or production-ready.

## Frozen Champion Model

The recorded Champion is `RF_Balanced_300_None_1`, a Random Forest classifier with the following configuration:

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `class_weight` | `balanced` |
| `max_depth` | `None` |
| `min_samples_leaf` | 1 |
| `random_state` | 42 |
| `n_jobs` | -1 |
| Decision threshold | 0.65 |

The Champion and its approved feature contract are frozen for this closeout. Candidate-evaluation work is separate and does not constitute Champion promotion.

## Approved Feature Contract

The approved ten features are:

1. `TX_AMOUNT`
2. `AMOUNT_LOG`
3. `TX_HOUR`
4. `TX_DAY_OF_WEEK`
5. `TX_IS_WEEKEND`
6. `TERMINAL_TX_COUNT_1D`
7. `TERMINAL_TX_COUNT_7D`
8. `TERMINAL_AVG_AMOUNT_7D`
9. `CUSTOMER_TX_COUNT_1D`
10. `CUSTOMER_TX_COUNT_7D`

Transaction identifiers, raw timestamps, and scenario labels are not model features. Historical feature construction follows the rule that a contributing event must precede the current transaction (`previous_time < current_time`) to reduce temporal leakage.

## Repository Structure

| Path | Purpose |
|---|---|
| `src/` | Application and prediction-related code |
| `ingestion/` | Data-ingestion components |
| `local-kafka/` | Local Kafka-related resources |
| `feast_repo/` | Feast feature-store configuration and code |
| `scripts/` | Training, evaluation, and supporting scripts |
| `tests/` | Automated tests |
| `config/` | Project configuration |
| `data/` | Local data and database files |
| `reports/` | Evaluation and verification evidence |
| `docs/` | Requirements, feasibility, KPI, Feast, and evaluation documentation |
| `Dockerfile` | Container build configuration |
| `requirements.txt` | Declared Python dependencies |

## Environment Setup

From the repository root on Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Installing the Python requirements is only the initial setup. Running the full workflow may also require the appropriate model artifact, environment configuration, AWS credentials and S3 access, Feast registry and store data, and a configured Kafka broker, topics, and producer/consumer settings.

Keep credentials and secrets out of source control. Do not commit `.env` contents, local databases, or model artifacts unintentionally.

## Verification Evidence

The project retains the following evidence locations:

- **Step 24 controlled end-to-end verification:** `reports/phase4/step24_e2e_verification/step24_live_e2e_evidence.txt`
- **Step 25 architecture review and targeted tests:** `reports/phase4/step25_architecture_review/step25_architecture_review_evidence.txt`
- **Step 18 candidate evaluation rerun:** `reports/phase4/step18_candidate_evaluation_rerun_20261010/`

The Step 25 evidence records **27/27 targeted tests passing in the documented environment**. This result applies to that recorded test run; it should not be interpreted as proof that every production integration has been tested.

The Step 24 evidence records a controlled Kafka-to-inference-to-prediction-storage flow, including a database read-back check. It does not establish that the entire live S3-to-Kafka pipeline is verified.

Additional project documentation:

- `docs/business_requirements.md`
- `docs/feasibility_assessment.md`
- `docs/kpi_definition.md`
- `docs/phase3_feast.md`
- `docs/phase4_evaluation.md`

## Current Status and Limitations

The project contains model-development, feature-store, inference, storage, API, and verification work. The following limitations remain open and should be considered when evaluating deployment readiness:

- **Live S3-to-Kafka integration:** The complete live path has not been verified by the retained end-to-end evidence.
- **Kafka payload discrepancy:** Five-field and seven-field payload descriptions have appeared in project records. The discrepancy remains unresolved; the available evidence does not justify claiming that either historical description never existed.
- **Evaluation manifest:** The expected manifest for the Step 18 candidate rerun was not located during the recorded review. The rerun directory exists, but directory existence alone does not establish complete evaluation provenance.
- **Candidate-to-MLflow provenance:** The relationship between the candidate evaluation and its MLflow run requires further verification.
- **Deployment integration:** The Docker configuration does not by itself guarantee that the frozen Champion artifact will be available at runtime. Artifact delivery and complete live deployment integration remain unresolved.
- **CI/CD:** CI/CD was not implemented under the recorded project scope.
- **Monitoring and retraining:** Existing drift-related evidence is limited in scope and does not establish a complete automated drift-to-retraining-and-promotion system.

These limitations do not invalidate the recorded tests or existing model-evaluation evidence. They define the work that remains before broader production-readiness claims can be made.

## Future Improvements

Potential improvements for a separate, controlled development phase include:

1. Resolve and test the Kafka payload contract across the producer, consumer, and API.
2. Improve candidate-evaluation manifests and MLflow provenance.
3. Verify the full live S3-to-Kafka-to-inference-to-storage workflow.
4. Make model-artifact delivery and container execution reproducible.
5. Add CI/CD if the project scope is expanded.
6. Extend drift monitoring, alerting, and controlled retraining with explicit validation and approval gates.

These are proposed improvements, not claims of implemented functionality.

## Closeout Principles

The current Champion, approved feature contract, and original verification evidence are preserved. Unverified integrations remain explicitly identified, and candidate-evaluation results are kept separate from Champion status. Further improvements should be made in a controlled phase with their own tests and evidence.

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import skops.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_phase4 import MODEL_FEATURES, TARGET, train_model
from mlflow_utils import setup_mlflow

TRAINING_DATASET = PROJECT_ROOT / "data" / "phase3" / "train_combined_features.csv"
CANDIDATE_DIR = PROJECT_ROOT / "artifacts" / "candidates"
CANDIDATE_ARTIFACT = CANDIDATE_DIR / "rf300_candidate" / "model.skops"
CHAMPION_ARTIFACT = PROJECT_ROOT / "artifacts" / "rf300_champion" / "model.skops"

THRESHOLD = 0.65
CHAMPION_IDENTITY = "RF_Balanced_300_None_1"
CANDIDATE_IDENTITY = "RF300_STEP17_CANDIDATE"

EXPECTED_FEATURES = [
    "TX_AMOUNT",
    "AMOUNT_LOG",
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
    "TX_IS_WEEKEND",
    "TERMINAL_TX_COUNT_1D",
    "TERMINAL_TX_COUNT_7D",
    "TERMINAL_AVG_AMOUNT_7D",
    "CUSTOMER_TX_COUNT_1D",
    "CUSTOMER_TX_COUNT_7D",
]

RF_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
}


def validate_contract():
    if TRAINING_DATASET != PROJECT_ROOT / "data" / "phase3" / "train_combined_features.csv":
        raise RuntimeError("Step 17 training dataset path is not the approved fixed dataset.")
    candidate_root = CANDIDATE_DIR.resolve()
    candidate_path = CANDIDATE_ARTIFACT.resolve()
    champion_path = CHAMPION_ARTIFACT.resolve()
    if candidate_root not in candidate_path.parents:
        raise RuntimeError("SAFETY CHECK FAILED: candidate artifact is outside candidate directory.")
    if candidate_path == champion_path:
        raise RuntimeError("SAFETY CHECK FAILED: candidate artifact equals Champion artifact.")
    if CANDIDATE_IDENTITY == CHAMPION_IDENTITY:
        raise RuntimeError("SAFETY CHECK FAILED: candidate identity equals Champion identity.")
    if EXPECTED_FEATURES != MODEL_FEATURES:
        raise RuntimeError("SAFETY CHECK FAILED: approved 10-feature set does not match train_phase4.py.")
    if THRESHOLD != 0.65:
        raise RuntimeError("SAFETY CHECK FAILED: Step 17 threshold is not 0.65.")


def main():
    validate_contract()

    if not TRAINING_DATASET.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAINING_DATASET}")

    train_df = pd.read_csv(TRAINING_DATASET)
    if train_df.shape[0] != 230295:
        raise RuntimeError(f"SAFETY CHECK FAILED: expected 230295 training rows, found {train_df.shape[0]}.")
    if int(train_df[TARGET].sum()) != 1176:
        raise RuntimeError(f"SAFETY CHECK FAILED: expected 1176 fraud transactions, found {int(train_df[TARGET].sum())}.")
    missing_features = [f for f in MODEL_FEATURES if f not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing approved features: {missing_features}")
    if TARGET not in train_df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET]

    setup_mlflow()
    with mlflow.start_run(run_name=CANDIDATE_IDENTITY) as run:
        mlflow.log_params({
            "candidate_type": "RF300_equivalent",
            "model_type": "random_forest",
            "imbalance_strategy": "balanced",
            "n_estimators": 300,
            "class_weight": "balanced",
            "max_depth": "None",
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1,
            "threshold": THRESHOLD,
            "n_features": len(MODEL_FEATURES),
            "training_dataset": str(TRAINING_DATASET.relative_to(PROJECT_ROOT)),
            "temporal_boundary": "April days 0-23",
            "feature_identity": ",".join(MODEL_FEATURES),
            "candidate_identity": CANDIDATE_IDENTITY,
            "champion_identity": CHAMPION_IDENTITY,
        })

        model = train_model(
            X_train,
            y_train,
            model_type="random_forest",
            imbalance_strategy="balanced",
            model_parameters=RF_PARAMETERS,
        )

        CANDIDATE_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(model, CANDIDATE_ARTIFACT)
        mlflow.sklearn.log_model(model, artifact_path="candidate_model")

        print("=" * 70)
        print("STEP 17 - CANDIDATE TRAINING")
        print("=" * 70)
        print("Candidate MLflow run ID:", run.info.run_id)
        print("Training dataset:", TRAINING_DATASET)
        print("Training shape:", train_df.shape)
        print("Fraud count:", int(y_train.sum()))
        print("Candidate artifact:", CANDIDATE_ARTIFACT)
        print("Champion artifact protected:", CHAMPION_ARTIFACT)
        print("Threshold:", THRESHOLD)
        print("Candidate identity:", CANDIDATE_IDENTITY)
        print("Champion identity:", CHAMPION_IDENTITY)
        print("Candidate only: no promotion performed.")


if __name__ == "__main__":
    main()

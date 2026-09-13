import mlflow


EXPERIMENT_NAME = "financial_fraud_detection_phase4"


def setup_mlflow() -> None:
    """Create or select the Phase 4 MLflow experiment."""
    mlflow.set_experiment(EXPERIMENT_NAME)
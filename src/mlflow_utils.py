import mlflow


EXPERIMENT_NAME = "financial_fraud_detection_phase4"


def setup_mlflow() -> None:
    """Create or select the Phase 4 MLflow experiment.

    Returns:
        None.

    Side Effects:
        Creates the experiment if it does not already exist and
        sets it as the active MLflow experiment.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

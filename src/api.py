"""
FastAPI application for financial fraud prediction.

This API exposes the approved Step 13 inference workflow without
modifying the existing feature, Feast, model, or prediction-storage logic.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from feast_repo.fraud_features.scripts.may_online_processor import (
    load_april_history,
)
from feast_repo.fraud_features.scripts.online_prediction import (
    load_champion_model,
    process_new_transaction,
)
from src.prediction_storage import initialize_database


logger = logging.getLogger(__name__)


class TransactionRequest(BaseModel):
    """Request schema for one fraud-detection transaction."""

    model_config = ConfigDict(extra="forbid")

    TRANSACTION_ID: int = Field(..., description="Unique transaction ID")
    TX_DATETIME: datetime = Field(..., description="Transaction timestamp")
    CUSTOMER_ID: int = Field(..., description="Customer ID")
    TERMINAL_ID: int = Field(..., description="Terminal ID")
    TX_AMOUNT: float = Field(..., description="Transaction amount")


class PredictionResponse(BaseModel):
    """Response schema for the fraud prediction result."""

    probability: float
    prediction: int
    stored_prediction: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize required inference state when the API starts."""

    global history
    global model
    global startup_error

    try:
        initialize_database()
        history = load_april_history()
        model = load_champion_model()
        startup_error = None

        logger.info("Fraud detection API startup completed successfully.")

    except Exception:
        history = None
        model = None
        startup_error = True

        logger.exception("Fraud detection API startup failed.")

    yield


app = FastAPI(
    title="Financial Fraud Detection API",
    description="API for fraud prediction using the frozen RF-300 Champion.",
    version="1.0.0",
    lifespan=lifespan,
)


history = None
model = None
startup_error = False


@app.get("/")
def root():
    """Return the basic API service status."""

    return {
        "status": "running",
        "service": "Financial Fraud Detection API",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    """Process one transaction through the approved Step 13 workflow."""

    if startup_error or history is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Inference service is not ready.",
        )

    transaction_data = transaction.model_dump()

    transaction_data["TX_DATETIME"] = pd.Timestamp(
        transaction_data["TX_DATETIME"]
    )

    try:
        (
            _feast_result,
            probability,
            prediction,
            stored_prediction,
        ) = process_new_transaction(
            transaction_data,
            history,
            model,
        )

        return PredictionResponse(
            probability=float(probability),
            prediction=int(prediction),
            stored_prediction=int(stored_prediction),
        )

    except Exception:
        logger.exception("Prediction processing failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from None

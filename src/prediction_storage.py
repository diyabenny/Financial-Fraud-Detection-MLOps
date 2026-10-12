import sqlite3
from pathlib import Path


DATABASE_PATH = Path("data") / "prediction_storage.db"


def initialize_database() -> None:
    """Create the prediction storage database and table.

    Returns:
        None.

    Side Effects:
        Creates the SQLite database file and predictions table
        if they do not already exist.
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                TRANSACTION_ID INTEGER PRIMARY KEY,
                prediction INTEGER NOT NULL
            )
            """
        )


def store_prediction(transaction_id: int, prediction: int) -> None:
    """Store a transaction prediction.

    Args:
        transaction_id: Unique transaction identifier.
        prediction: Model prediction, where 0 is legitimate and 1 is fraud.

    Returns:
        None.

    Side Effects:
        Inserts or updates the prediction in the SQLite database.
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO predictions (TRANSACTION_ID, prediction)
            VALUES (?, ?)
            """,
            (transaction_id, prediction),
        )


def get_prediction(transaction_id: int) -> int | None:
    """Retrieve a stored prediction by transaction ID.

    Args:
        transaction_id: Unique transaction identifier.

    Returns:
        The stored prediction, or None if the transaction is not found.
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        result = connection.execute(
            """
            SELECT prediction
            FROM predictions
            WHERE TRANSACTION_ID = ?
            """,
            (transaction_id,),
        ).fetchone()

    return result[0] if result else None

def get_prediction_summary() -> dict[str, int]:
    """Return a summary of stored fraud predictions.

    Returns:
        A dictionary containing the total number of stored predictions,
        the number of fraud predictions, and the number of normal
        predictions.

    Raises:
        ValueError: If the fraud and normal prediction counts do not
            add up to the total prediction count.
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        total_predictions = connection.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            """
        ).fetchone()[0]

        fraud_predictions = connection.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE prediction = 1
            """
        ).fetchone()[0]

        normal_predictions = connection.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE prediction = 0
            """
        ).fetchone()[0]

    if fraud_predictions + normal_predictions != total_predictions:
        raise ValueError(
            "Prediction summary counts do not match the total prediction count."
        )

    return {
        "total_predictions": total_predictions,
        "fraud_predictions": fraud_predictions,
        "normal_predictions": normal_predictions,
    }
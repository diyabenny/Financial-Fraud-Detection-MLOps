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
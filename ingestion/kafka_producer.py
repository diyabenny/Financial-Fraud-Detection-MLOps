
"""Step 8: S3-to-Kafka producer for one approved May transaction.

Default execution validates one S3 transaction without publishing.
First Kafka publication requires a separate reviewed checkpoint.
"""

import json
import math
import numbers
from datetime import datetime

from confluent_kafka import Producer

from ingestion.simulated_api import read_source_from_s3


BROKER = "localhost:9092"
TOPIC = "fraud-transactions"

PAYLOAD_FIELDS = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
)


def validate_payload(payload):
    """Validate a transaction against the approved Kafka payload contract.

    Args:
        payload: Dictionary containing exactly the five approved fields.

    Returns:
        True when all required fields and values are valid.

    Raises:
        ValueError: If fields are missing or extra, IDs have invalid types,
            the timestamp has an invalid format, or the amount is invalid.

    Side effects:
        None. The input payload is not modified.
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    if set(payload) != set(PAYLOAD_FIELDS):
        raise ValueError("Payload must contain exactly the five approved fields.")

    for field in ("TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{field} must be an integer.")

    timestamp = payload["TX_DATETIME"]
    if not isinstance(timestamp, str):
        raise ValueError("TX_DATETIME must be a string.")

    try:
        parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError(
            "TX_DATETIME must use timezone-naive YYYY-MM-DD HH:MM:SS."
        ) from exc

    if parsed.strftime("%Y-%m-%d %H:%M:%S") != timestamp:
        raise ValueError("TX_DATETIME must use the exact approved format.")

    amount = payload["TX_AMOUNT"]
    if (
        isinstance(amount, bool)
        or not isinstance(amount, numbers.Real)
        or not math.isfinite(amount)
        or amount < 0
    ):
        raise ValueError("TX_AMOUNT must be a finite, nonnegative number.")

    return True


def build_payload(source_row):
    """Build the approved five-field payload from one S3 source record.

    Args:
        source_row: Dictionary representing one source CSV transaction.

    Returns:
        A validated dictionary containing only the five approved fields,
        with numeric values converted to JSON-compatible Python types.

    Raises:
        ValueError: If the source row is invalid, a required field is
            missing, or a projected field fails payload validation.

    Side effects:
        None. A new payload dictionary is created.
    """
    if not isinstance(source_row, dict):
        raise ValueError("Source row must be a dictionary.")

    payload = {}

    for field in PAYLOAD_FIELDS:
        if field not in source_row:
            raise ValueError(f"Missing source field: {field}")
        payload[field] = source_row[field]

    validate_payload(payload)

    # Convert NumPy scalar types to standard Python JSON-compatible types.
    for field in ("TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"):
        payload[field] = int(payload[field])

    payload["TX_AMOUNT"] = float(payload["TX_AMOUNT"])

    return payload


def select_one_valid_transaction(dataframe):
    """Select the first transaction that satisfies the Kafka contract.

    Args:
        dataframe: Pandas DataFrame loaded from the approved May S3 source.

    Returns:
        The first valid five-field transaction payload.

    Raises:
        ValueError: If the DataFrame is empty or contains no valid rows.

    Side effects:
        Prints the number of rejected rows and validation errors
        encountered while selecting a transaction.
    """
    if dataframe.empty:
        raise ValueError("The S3 source contains no transactions.")

    rejection_count = 0

    for row in dataframe.to_dict(orient="records"):
        try:
            payload = build_payload(row)
            print(f"Valid transaction selected after {rejection_count} rejection(s).")
            return payload
        except ValueError as exc:
            rejection_count += 1
            print(f"Skipping invalid source row: {exc}")

    raise ValueError("No valid transaction found in the S3 source.")


def publish_transaction(payload):
    """Publish one validated transaction to the approved Kafka topic.

    Args:
        payload: Dictionary containing the approved five-field transaction.

    Returns:
        None after the producer confirms successful Kafka delivery.

    Raises:
        ValueError: If the payload violates the approved contract.
        RuntimeError: If Kafka delivery cannot be confirmed.
        KafkaException: If the Kafka producer raises a client error.

    Side effects:
        Creates a Kafka producer, sends one message, waits for its
        delivery callback, and prints confirmed delivery metadata.
    """
    validate_payload(payload)

    message = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")

    key = str(payload["TRANSACTION_ID"]).encode("utf-8")

    producer = Producer({"bootstrap.servers": BROKER})
    delivery_result = {"error": None, "delivered": False}

    def delivery_callback(error, message_metadata):
        if error is not None:
            delivery_result["error"] = str(error)
        else:
            delivery_result["delivered"] = True
            print(
                f"Delivered to {message_metadata.topic()}, "
                f"partition {message_metadata.partition()}, "
                f"offset {message_metadata.offset()}"
            )

    producer.produce(
        topic=TOPIC,
        key=key,
        value=message,
        on_delivery=delivery_callback,
    )

    remaining = producer.flush(timeout=30)

    if remaining or delivery_result["error"] or not delivery_result["delivered"]:
        raise RuntimeError(
            f"Kafka delivery not confirmed: {delivery_result}"
        )


def main():
    """Perform the approved S3-to-Kafka producer dry run.

    Args:
        None.

    Returns:
        None.

    Raises:
        Exceptions from S3 ingestion, transaction selection, validation,
        or JSON serialization if the dry run cannot be completed.

    Side effects:
        Reads the approved May dataset from S3 and prints the selected
        payload, Kafka key, and target topic. Does not publish to Kafka.
    """
    dataframe = read_source_from_s3()
    payload = select_one_valid_transaction(dataframe)

    # Verify JSON serialization without sending anything to Kafka.
    serialized = json.dumps(payload, allow_nan=False)
    print("DRY RUN ONLY - no Kafka message published.")
    print("Validated payload:", serialized)
    print("Kafka key:", str(payload["TRANSACTION_ID"]))
    print("Target topic:", TOPIC)


if __name__ == "__main__":
    main()
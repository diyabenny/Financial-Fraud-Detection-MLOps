
"""Step 9: Kafka consumer with validation and manual offset commits.

Consumes existing transactions only. Does not publish or run inference.
"""

import argparse
import json
import logging

from confluent_kafka import Consumer, KafkaException

from ingestion.kafka_producer import BROKER, TOPIC, validate_payload


GROUP_ID = "fraud-inference-group"
LOGGER = logging.getLogger(__name__)


def decode_transaction(message):
    """Decode and validate one Kafka message and its transaction key.

    Args:
        message: A Kafka message with UTF-8 JSON value and UTF-8 key.

    Returns:
        The validated five-field transaction dictionary.

    Raises:
        ValueError: If the key, JSON, payload, or key/payload match is invalid.

    Side effects:
        None.
    """
    raw_value = message.value()
    raw_key = message.key()

    if raw_value is None or raw_key is None:
        raise ValueError("Kafka message must have both key and value.")

    try:
        def reject_nonstandard_constant(value):
            raise ValueError(f"Invalid JSON numeric constant: {value}")

        def reject_duplicate_fields(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON field: {key}")
                result[key] = value
            return result

        payload = json.loads(
            raw_value.decode("utf-8"),
            parse_constant=reject_nonstandard_constant,
            object_pairs_hook=reject_duplicate_fields,
        )
        kafka_key = raw_key.decode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Kafka message must contain valid UTF-8 JSON.") from exc

    validate_payload(payload)

    expected_key = str(payload["TRANSACTION_ID"])
    if kafka_key != expected_key:
        raise ValueError(
            f"Kafka key {kafka_key!r} does not match "
            f"TRANSACTION_ID {expected_key!r}."
        )

    return payload


def process_transaction(payload):
    """Perform the approved Step 9 transport-processing demonstration.

    Args:
        payload: A transaction already validated against the Kafka contract.

    Returns:
        None after the transaction is logged successfully.

    Raises:
        Exceptions from logging or processing if the operation fails.

    Side effects:
        Logs the validated transaction. Does not run model inference.
    """
    LOGGER.info(
        "Validated transaction %s, customer %s, amount %s",
        payload["TRANSACTION_ID"],
        payload["CUSTOMER_ID"],
        payload["TX_AMOUNT"],
    )


def handle_message(consumer, message):
    """Process one Kafka message and commit its offset after success.

    Args:
        consumer: The Kafka consumer that received the message.
        message: The Kafka message to process.

    Returns:
        The validated transaction after a successful synchronous commit.

    Raises:
        ValueError: If the message violates the approved contract.
        KafkaException: If the synchronous offset commit fails.
        Other exceptions raised during processing.

    Side effects:
        Processes the transaction and synchronously commits its offset
        only after processing succeeds.
    """
    payload = decode_transaction(message)
    process_transaction(payload)

    # The explicit message commits offset + 1 for this partition.
    consumer.commit(message=message, asynchronous=False)

    LOGGER.info(
        "Processed and committed topic=%s partition=%s offset=%s",
        message.topic(),
        message.partition(),
        message.offset(),
    )
    return payload


def consume_once(timeout_seconds=15):
    """Consume at most one existing message and close the consumer.

    Args:
        timeout_seconds: Maximum time to wait for a Kafka message.

    Returns:
        The validated transaction, or None if no message arrives.

    Raises:
        KafkaException: If Kafka reports an error.
        ValueError: If the received message fails validation.
        Other exceptions raised by processing or offset commit.

    Side effects:
        Joins the approved consumer group, processes at most one message,
        and commits only after successful processing. Always closes
        the consumer. Does not publish messages.
    """
    consumer = Consumer({
            "bootstrap.servers": "127.0.0.1:9092",
            "group.id": GROUP_ID,
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
            "auto.offset.reset": "earliest",
        }
    )

    try:
        consumer.subscribe([TOPIC])
        LOGGER.info(
            "Waiting for one existing message on %s, group %s",
            TOPIC,
            GROUP_ID,
        )

        message = consumer.poll(timeout_seconds)

        if message is None:
            LOGGER.warning("No message received within the timeout.")
            return None

        if message.error():
            raise KafkaException(message.error())

        try:
            return handle_message(consumer, message)
        except ValueError as exc:
            LOGGER.error(
                "Rejected invalid transaction at topic=%s "
                "partition=%s offset=%s: %s. "
                "Consumer run stopped; offset not committed.",
                message.topic(),
                message.partition(),
                message.offset(),
                exc,
            )
            raise


    finally:
        consumer.close()


def main():
    """Run the consumer only when explicitly requested.

    Args:
        None. Reads command-line arguments.

    Returns:
        None.

    Raises:
        Exceptions from Kafka, validation, processing, or offset commit.

    Side effects:
        Consumes at most one existing message only with --consume-once.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consume-once",
        action="store_true",
        help="Consume and process at most one existing Kafka message.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.consume_once:
        print("SAFE DEFAULT: no Kafka message consumed.")
        print("Use --consume-once only after live-test authorization.")
        return

    payload = consume_once()
    if payload is None:
        print("No transaction consumed; no offset committed.")
    else:
        print(
            "SUCCESS: processed and committed transaction",
            payload["TRANSACTION_ID"],
        )


if __name__ == "__main__":
    main()
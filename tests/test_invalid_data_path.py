"""Step 10: Offline tests for invalid-message rejection and logging."""

import json
import unittest
from unittest.mock import Mock, patch

from ingestion.kafka_consumer import consume_once, handle_message


HISTORY = Mock()
MODEL = Mock()

VALID_PAYLOAD = {
    "TRANSACTION_ID": 288062,
    "TX_DATETIME": "2018-05-01 00:01:21",
    "CUSTOMER_ID": 3546,
    "TERMINAL_ID": 2944,
    "TX_AMOUNT": 18.71,
}


def make_message(payload=None, key=b"288062", raw_value=None):
    """Create a fake Kafka message without broker access."""
    message = Mock()
    message.key.return_value = key
    message.value.return_value = (
        raw_value
        if raw_value is not None
        else json.dumps(
            VALID_PAYLOAD if payload is None else payload
        ).encode("utf-8")
    )
    message.topic.return_value = "fraud-transactions"
    message.partition.return_value = 0
    message.offset.return_value = 0
    message.error.return_value = None
    return message


class TestInvalidDataPath(unittest.TestCase):

    def test_missing_field_rejected_without_processing_or_commit(self):
        payload = VALID_PAYLOAD.copy()
        del payload["TX_AMOUNT"]
        consumer = Mock()

        with patch("ingestion.kafka_consumer.process_transaction") as process:
            with self.assertRaises(ValueError):
                handle_message(consumer, make_message(payload=payload), HISTORY, MODEL)
            process.assert_not_called()

        consumer.commit.assert_not_called()

    def test_extra_field_rejected_without_commit(self):
        payload = VALID_PAYLOAD.copy()
        payload["IS_FRAUD"] = 0
        consumer = Mock()

        with self.assertRaises(ValueError):
            handle_message(consumer, make_message(payload=payload), HISTORY, MODEL)

        consumer.commit.assert_not_called()

    def test_invalid_amount_rejected_without_commit(self):
        payload = VALID_PAYLOAD.copy()
        payload["TX_AMOUNT"] = -10
        consumer = Mock()

        with self.assertRaises(ValueError):
            handle_message(consumer, make_message(payload=payload), HISTORY, MODEL)

        consumer.commit.assert_not_called()

    def test_wrong_key_rejected_without_commit(self):
        consumer = Mock()

        with self.assertRaises(ValueError):
            handle_message(consumer, make_message(key=b"999999"), HISTORY, MODEL)

        consumer.commit.assert_not_called()

    def test_malformed_json_rejected_without_commit(self):
        consumer = Mock()

        with self.assertRaises(ValueError):
            handle_message(
                consumer,
                make_message(raw_value=b'{"TRANSACTION_ID":'),
                HISTORY,
                MODEL,
            )

        consumer.commit.assert_not_called()

    def test_invalid_message_logged_and_consumer_closed(self):
        payload = VALID_PAYLOAD.copy()
        del payload["TX_AMOUNT"]
        message = make_message(payload=payload)

        with patch("ingestion.kafka_consumer.load_april_history", return_value=HISTORY), \
             patch("ingestion.kafka_consumer.load_champion_model", return_value=MODEL), \
             patch("ingestion.kafka_consumer.Consumer") as consumer_class:

            consumer = consumer_class.return_value
            consumer.poll.return_value = message

            with patch("ingestion.kafka_consumer.process_transaction") as process:
                with self.assertLogs("ingestion.kafka_consumer", level="ERROR") as logs:
                    with self.assertRaises(ValueError):
                        consume_once(timeout_seconds=0)
                process.assert_not_called()

            consumer.commit.assert_not_called()
            consumer.poll.assert_called_once_with(0)
            consumer.close.assert_called_once()

        output = "\n".join(logs.output)
        self.assertIn("Rejected invalid transaction", output)
        self.assertIn("fraud-transactions", output)
        self.assertIn("partition=0", output)
        self.assertIn("offset=0", output)
        self.assertIn("offset not committed", output)

    def test_valid_message_still_commits_synchronously(self):
        consumer = Mock()
        message = make_message()

        with patch("ingestion.kafka_consumer.process_transaction") as process:
            result = handle_message(consumer, message, HISTORY, MODEL)
            process.assert_called_once_with(VALID_PAYLOAD, HISTORY, MODEL)

        self.assertEqual(result, VALID_PAYLOAD)
        consumer.commit.assert_called_once_with(
            message=message,
            asynchronous=False,
        )

    def test_kafka_message_error_stops_without_processing_or_commit(self):
        message = make_message()
        message.error.return_value = "Simulated Kafka error"

        with patch("ingestion.kafka_consumer.load_april_history", return_value=HISTORY), \
             patch("ingestion.kafka_consumer.load_champion_model", return_value=MODEL), \
             patch("ingestion.kafka_consumer.Consumer") as consumer_class:

            consumer = consumer_class.return_value
            consumer.poll.return_value = message

            with patch("ingestion.kafka_consumer.process_transaction") as process:
                with self.assertRaises(Exception):
                    consume_once(timeout_seconds=0)
                process.assert_not_called()

            consumer.commit.assert_not_called()
            consumer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
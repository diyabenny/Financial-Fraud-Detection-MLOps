"""Offline tests for the Kafka consumer."""

import json
import unittest
from unittest.mock import Mock, patch

from ingestion.kafka_consumer import (
    decode_transaction,
    handle_message,
)


VALID_PAYLOAD = {
    "TRANSACTION_ID": 288062,
    "TX_DATETIME": "2018-05-01 00:01:21",
    "CUSTOMER_ID": 3546,
    "TERMINAL_ID": 2944,
    "TX_AMOUNT": 18.71,
}

HISTORY = Mock()
MODEL = Mock()


def make_message(payload=None, key=b"288062", raw_value=None):
    """Create a mocked Kafka message without connecting to a broker."""
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
    return message


class TestKafkaConsumer(unittest.TestCase):
    def test_valid_message(self):
        self.assertEqual(
            decode_transaction(make_message()),
            VALID_PAYLOAD,
        )

    def test_wrong_kafka_key(self):
        with self.assertRaises(ValueError):
            decode_transaction(make_message(key=b"999999"))

    def test_missing_field(self):
        payload = VALID_PAYLOAD.copy()
        del payload["TX_AMOUNT"]

        with self.assertRaises(ValueError):
            decode_transaction(make_message(payload=payload))

    def test_extra_field(self):
        payload = VALID_PAYLOAD.copy()
        payload["IS_FRAUD"] = 0

        with self.assertRaises(ValueError):
            decode_transaction(make_message(payload=payload))

    def test_boolean_id(self):
        payload = VALID_PAYLOAD.copy()
        payload["TRANSACTION_ID"] = True

        with self.assertRaises(ValueError):
            decode_transaction(make_message(payload=payload))

    def test_invalid_timestamp(self):
        payload = VALID_PAYLOAD.copy()
        payload["TX_DATETIME"] = "2018-05-01T00:01:21Z"

        with self.assertRaises(ValueError):
            decode_transaction(make_message(payload=payload))

    def test_negative_amount(self):
        payload = VALID_PAYLOAD.copy()
        payload["TX_AMOUNT"] = -1

        with self.assertRaises(ValueError):
            decode_transaction(make_message(payload=payload))

    def test_nonfinite_amount(self):
        with self.assertRaises(ValueError):
            decode_transaction(
                make_message(
                    raw_value=json.dumps(
                        {
                            **VALID_PAYLOAD,
                            "TX_AMOUNT": float("nan"),
                        }
                    ).encode("utf-8")
                )
            )

    def test_duplicate_json_field(self):
        raw = (
            b'{"TRANSACTION_ID":288062,"TRANSACTION_ID":288062,'
            b'"TX_DATETIME":"2018-05-01 00:01:21",'
            b'"CUSTOMER_ID":3546,"TERMINAL_ID":2944,'
            b'"TX_AMOUNT":18.71}'
        )

        with self.assertRaises(ValueError):
            decode_transaction(make_message(raw_value=raw))

    @patch("ingestion.kafka_consumer.process_transaction")
    def test_commit_after_success(self, mock_process):
        consumer = Mock()
        message = make_message()

        result = handle_message(
            consumer,
            message,
            HISTORY,
            MODEL,
        )

        self.assertEqual(result, VALID_PAYLOAD)

        mock_process.assert_called_once_with(
            VALID_PAYLOAD,
            HISTORY,
            MODEL,
        )

        consumer.commit.assert_called_once_with(
            message=message,
            asynchronous=False,
        )

    @patch("ingestion.kafka_consumer.process_transaction")
    def test_no_commit_after_processing_failure(self, mock_process):
        mock_process.side_effect = RuntimeError(
            "Processing failed"
        )
        consumer = Mock()

        with self.assertRaises(RuntimeError):
            handle_message(
                consumer,
                make_message(),
                HISTORY,
                MODEL,
            )

        consumer.commit.assert_not_called()

    def test_no_commit_after_validation_failure(self):
        consumer = Mock()

        with self.assertRaises(ValueError):
            handle_message(
                consumer,
                make_message(key=b"wrong"),
                HISTORY,
                MODEL,
            )

        consumer.commit.assert_not_called()

    @patch("ingestion.kafka_consumer.process_transaction")
    def test_commit_failure_propagates(self, mock_process):
        consumer = Mock()
        consumer.commit.side_effect = RuntimeError(
            "Commit failed"
        )

        with self.assertRaises(RuntimeError):
            handle_message(
                consumer,
                make_message(),
                HISTORY,
                MODEL,
            )

        mock_process.assert_called_once_with(
            VALID_PAYLOAD,
            HISTORY,
            MODEL,
        )


if __name__ == "__main__":
    unittest.main()
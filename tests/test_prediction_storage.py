import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import prediction_storage


class TestPredictionStorage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True
        )
        self.database_path = Path(self.temp_dir.name) / "prediction_storage.db"

        self.path_patcher = patch.object(
            prediction_storage,
            "DATABASE_PATH",
            self.database_path,
        )
        self.path_patcher.start()

        prediction_storage.initialize_database()

    def tearDown(self):
        self.path_patcher.stop()
        self.temp_dir.cleanup()

    def test_initialize_database_creates_predictions_table(self):
        self.assertIsNone(
            prediction_storage.get_prediction(123456)
        )

    def test_store_and_get_prediction(self):
        prediction_storage.store_prediction(123456, 1)

        self.assertEqual(
            prediction_storage.get_prediction(123456),
            1,
        )

    def test_get_prediction_returns_none_for_unknown_transaction(self):
        self.assertIsNone(
            prediction_storage.get_prediction(999999)
        )

    def test_store_prediction_replaces_existing_prediction(self):
        prediction_storage.store_prediction(123456, 0)
        prediction_storage.store_prediction(123456, 1)

        self.assertEqual(
            prediction_storage.get_prediction(123456),
            1,
        )

    def test_prediction_summary(self):
        prediction_storage.store_prediction(100001, 1)
        prediction_storage.store_prediction(100002, 0)
        prediction_storage.store_prediction(100003, 0)

        summary = prediction_storage.get_prediction_summary()

        self.assertEqual(
            summary,
            {
                "total_predictions": 3,
                "fraud_predictions": 1,
                "normal_predictions": 2,
            },
        )

    def test_prediction_summary_invariant(self):
        prediction_storage.store_prediction(100001, 1)
        prediction_storage.store_prediction(100002, 0)

        summary = prediction_storage.get_prediction_summary()

        self.assertEqual(
            summary["fraud_predictions"] + summary["normal_predictions"],
            summary["total_predictions"],
        )


if __name__ == "__main__":
    unittest.main()

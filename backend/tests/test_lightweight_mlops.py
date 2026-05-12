import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_engine.data_quality import validate_ohlcv
from unified_engine.drift_monitor import assess_market_drift


class LightweightMLOpsTests(unittest.TestCase):
    def test_data_quality_passes_clean_ohlcv(self):
        history = pd.DataFrame({
            "Open": [100 + i for i in range(130)],
            "High": [101 + i for i in range(130)],
            "Low": [99 + i for i in range(130)],
            "Close": [100.5 + i for i in range(130)],
            "Volume": [1_000_000 + i for i in range(130)],
        })

        report = validate_ohlcv(history)

        self.assertEqual(report["status"], "passed")
        self.assertGreaterEqual(report["score"], 90)

    def test_data_quality_flags_missing_history(self):
        report = validate_ohlcv(pd.DataFrame())

        self.assertEqual(report["status"], "failed")
        self.assertIn("No market history", report["issues"][0])

    def test_drift_monitor_warms_up_with_short_history(self):
        history = pd.DataFrame({
            "Close": [100 + i for i in range(20)],
            "Volume": [1000 + i for i in range(20)],
        })

        report = assess_market_drift(history)

        self.assertEqual(report["status"], "warming_up")

    def test_event_bus_and_prediction_store_are_file_backed(self):
        from unified_engine import event_bus, prediction_store

        with tempfile.TemporaryDirectory() as tmp_dir:
            event_bus._EVENT_LOG = Path(tmp_dir) / "events.jsonl"
            prediction_store._STORE_FILE = Path(tmp_dir) / "predictions.jsonl"

            event = event_bus.publish_event("unit_test", ticker="AAPL", payload={"ok": True})
            prediction_store.record_served_prediction(
                ticker="AAPL",
                horizon_days=7,
                current_price=100.0,
                signal="BUY",
                trust_score=72,
                risk_label="Medium",
                model_version="test",
                prediction_ready=True,
                currency="USD",
            )

            self.assertEqual(event["ticker"], "AAPL")
            self.assertEqual(len(event_bus.list_events(ticker="AAPL")), 1)
            self.assertEqual(prediction_store.prediction_summary()["total"], 1)

    def test_scheduler_summary_is_file_backed(self):
        from scheduler import ModelTrainingScheduler

        with tempfile.TemporaryDirectory() as tmp_dir:
            scheduler = ModelTrainingScheduler()
            scheduler.summary_path = Path(tmp_dir) / "daily_training_summary.json"
            summary = {
                "timestamp": "2026-05-12T00:00:00",
                "total_trained": 1,
                "total_failed": 0,
                "total_skipped": 0,
                "results": {"AAPL": {"status": "success"}},
            }

            scheduler._write_summary(summary)
            status = scheduler.get_status()

            self.assertTrue(scheduler.summary_path.exists())
            self.assertEqual(status["summary_path"], str(scheduler.summary_path))
            self.assertIn("training_time_utc", status)


if __name__ == "__main__":
    unittest.main()

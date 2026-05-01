import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_engine import build_decision, signal_for_forecast_row


def make_history(days=90, start=100.0, drift=0.0005, vol=0.01):
    rng = np.random.default_rng(42)
    returns = rng.normal(drift, vol, size=days)
    close = start * np.cumprod(1 + returns)
    high = close * 1.01
    low = close * 0.99
    open_ = close / (1 + rng.normal(0, 0.002, size=days))
    volume = rng.integers(900_000, 1_200_000, size=days)
    index = pd.date_range("2025-01-01", periods=days, freq="B")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def base_indicators(current_price):
    return {
        "rsi": 45.0,
        "ema": current_price * 1.005,
        "macd": -0.05,
        "macd_signal": -0.03,
        "macd_histogram": -0.02,
        "sma20": current_price * 1.004,
        "sma50": current_price * 1.006,
        "atr": current_price * 0.012,
    }


class DecisionEngineTests(unittest.TestCase):
    def test_tiny_positive_move_with_bearish_sentiment_is_hold(self):
        hist = make_history()
        current = float(hist["Close"].iloc[-1])
        decision = build_decision(
            hist=hist,
            current_price=current,
            predictions=[current * 1.0008],
            indicators=base_indicators(current),
            sentiment={
                "score": -0.9,
                "buzz_articles": 12,
                "buzz_score": 1.0,
            },
            model_payload={
                "prediction": 0.0008,
                "lower_95": -0.018,
                "upper_95": 0.020,
                "confidence": 0.52,
                "direction_prob": 0.53,
            },
        )

        self.assertEqual(decision.signal, "HOLD")
        self.assertIn("Bearish news sentiment vetoed", " ".join(decision.reasons))
        self.assertLess(abs(decision.expected_move_pct), decision.min_required_move_pct)

    def test_aligned_high_confidence_forecast_can_buy(self):
        hist = make_history(drift=0.0015, vol=0.008)
        current = float(hist["Close"].iloc[-1])
        indicators = base_indicators(current)
        indicators.update(
            {
                "rsi": 54.0,
                "ema": current * 0.985,
                "macd": 0.15,
                "macd_signal": 0.05,
                "macd_histogram": 0.10,
                "sma20": current * 0.985,
                "sma50": current * 0.955,
            }
        )

        decision = build_decision(
            hist=hist,
            current_price=current,
            predictions=[current * 1.035],
            indicators=indicators,
            sentiment={
                "score": 0.65,
                "buzz_articles": 15,
                "buzz_score": 1.0,
            },
            model_payload={
                "prediction": 0.035,
                "lower_95": 0.012,
                "upper_95": 0.058,
                "confidence": 0.82,
                "direction_prob": 0.76,
            },
        )

        self.assertIn(decision.signal, {"BUY", "STRONG BUY"})
        self.assertGreaterEqual(decision.confidence, 0.38)
        self.assertGreater(decision.expected_move_pct, decision.min_required_move_pct)

    def test_forecast_rows_cannot_contradict_final_hold(self):
        hist = make_history()
        current = float(hist["Close"].iloc[-1])
        decision = build_decision(
            hist=hist,
            current_price=current,
            predictions=[current * 1.001],
            indicators=base_indicators(current),
            sentiment={"score": -0.8, "buzz_articles": 10, "buzz_score": 1.0},
            model_payload={
                "prediction": 0.001,
                "lower_95": -0.02,
                "upper_95": 0.022,
                "confidence": 0.45,
                "direction_prob": 0.52,
            },
        )

        self.assertEqual(decision.signal, "HOLD")
        self.assertEqual(
            signal_for_forecast_row(
                base_price=current,
                predicted_price=current * 1.001,
                final_decision=decision,
            ),
            "HOLD",
        )


if __name__ == "__main__":
    unittest.main()

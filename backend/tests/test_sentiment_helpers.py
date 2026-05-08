import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_api import _normalize_sentiment_fraction, _sentiment_label


class SentimentHelperTests(unittest.TestCase):
    def test_fraction_normalization_accepts_percent_or_fraction(self):
        self.assertAlmostEqual(_normalize_sentiment_fraction(0.65), 0.65)
        self.assertAlmostEqual(_normalize_sentiment_fraction(65), 0.65)
        self.assertEqual(_normalize_sentiment_fraction(-5), 0.0)
        self.assertEqual(_normalize_sentiment_fraction(500), 1.0)

    def test_sentiment_label_uses_trade_signal_contract(self):
        self.assertEqual(_sentiment_label(0.7)[0], "STRONG BUY")
        self.assertEqual(_sentiment_label(0.25)[0], "BUY")
        self.assertEqual(_sentiment_label(0.0)[0], "HOLD")
        self.assertEqual(_sentiment_label(-0.25)[0], "SELL")
        self.assertEqual(_sentiment_label(-0.7)[0], "STRONG SELL")


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_api import _article_relevance_score, _normalize_sentiment_fraction, _sentiment_label


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

    def test_news_relevance_prioritizes_searched_company(self):
        good = {
            "headline": "Oracle expands AI database cloud services",
            "summary": "Oracle revenue grows as enterprise demand improves.",
            "related": "ORCL",
        }
        weak = {
            "headline": "Nvidia pushes past major AI milestone",
            "summary": "The chipmaker also works with Oracle and other cloud vendors.",
            "related": "",
        }

        self.assertGreater(_article_relevance_score(good, "ORCL", "Oracle Corp"), 3.0)
        self.assertLess(_article_relevance_score(weak, "ORCL", "Oracle Corp"), 3.0)


if __name__ == "__main__":
    unittest.main()

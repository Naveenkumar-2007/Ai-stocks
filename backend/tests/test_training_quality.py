import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_engine.training import _find_validation_threshold, _macro_f1, _quality_gate


class TrainingQualityTests(unittest.TestCase):
    def test_validation_threshold_avoids_one_class_collapse(self):
        y_true = np.array([1] * 70 + [0] * 30)
        probs = np.array(
            [0.62] * 35 + [0.54] * 35 +
            [0.46] * 20 + [0.38] * 10
        )

        threshold, metrics = _find_validation_threshold(y_true, probs)
        preds = (probs >= threshold).astype(int)

        self.assertGreater(metrics["predicted_up_rate"], 0.05)
        self.assertGreater(metrics["predicted_down_rate"], 0.05)
        self.assertGreater(_macro_f1(y_true, preds), 0.35)

    def test_quality_gate_flags_class_imbalance(self):
        self.assertEqual(
            _quality_gate({
                "balanced_accuracy": 50.0,
                "macro_f1": 25.0,
                "auc": 0.62,
                "binom_pvalue": 0.001,
                "fold_std_accuracy": 12.0,
                "calibration_samples": 100,
            }),
            "class_imbalance_warning",
        )

    def test_quality_gate_flags_unstable_folds(self):
        self.assertEqual(
            _quality_gate({
                "balanced_accuracy": 58.0,
                "macro_f1": 55.0,
                "auc": 0.61,
                "binom_pvalue": 0.001,
                "fold_std_accuracy": 22.0,
                "calibration_samples": 120,
            }),
            "unstable_folds",
        )


if __name__ == "__main__":
    unittest.main()

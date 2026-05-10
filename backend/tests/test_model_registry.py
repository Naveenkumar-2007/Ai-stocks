import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_engine.model_registry import evaluate_promotion


class ModelRegistryPromotionTests(unittest.TestCase):
    def test_validated_model_promotes_to_production(self):
        promotion = evaluate_promotion({
            "balanced_accuracy": 61.0,
            "macro_f1": 58.0,
            "auc": 0.64,
            "binom_pvalue": 0.001,
            "fold_std_accuracy": 11.0,
            "calibration_samples": 160,
            "model_quality_gate": "validated",
        })

        self.assertEqual(promotion["stage"], "production")
        self.assertEqual(promotion["failed_checks"], [])
        self.assertGreater(promotion["quality_score"], 50)

    def test_one_class_model_is_quarantined(self):
        promotion = evaluate_promotion({
            "balanced_accuracy": 77.0,
            "macro_f1": 0.0,
            "auc": 0.59,
            "binom_pvalue": 0.001,
            "fold_std_accuracy": 14.0,
            "calibration_samples": 80,
            "model_quality_gate": "class_imbalance_warning",
        })

        self.assertEqual(promotion["stage"], "quarantined")
        self.assertIn("macro_f1", promotion["failed_checks"])
        self.assertIn("quality_gate", promotion["failed_checks"])

    def test_statistically_weak_model_is_quarantined(self):
        promotion = evaluate_promotion({
            "balanced_accuracy": 57.0,
            "macro_f1": 54.0,
            "auc": 0.61,
            "binom_pvalue": 0.22,
            "fold_std_accuracy": 12.0,
            "calibration_samples": 100,
            "model_quality_gate": "validated",
        })

        self.assertEqual(promotion["stage"], "quarantined")
        self.assertIn("p_value", promotion["failed_checks"])


if __name__ == "__main__":
    unittest.main()

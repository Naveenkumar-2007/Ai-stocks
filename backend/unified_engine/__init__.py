"""
Unified Prediction Engine v4.0
==============================
Single source of truth for stock prediction.
Fixes all train-inference mismatches, target leakage, and meta-learner overfitting.
"""

from unified_engine.inference import UnifiedPredictor
from unified_engine.training import UnifiedTrainer

__all__ = ["UnifiedPredictor", "UnifiedTrainer"]

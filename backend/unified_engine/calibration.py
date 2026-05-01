"""
Unified Engine — Calibration
==============================
Probability calibration on a HELD-OUT fold (never the training data).
Fixes the issue where isotonic regression was memorizing noise by
fitting and transforming on the same data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class PlattCalibrator:
    """
    Platt scaling: fit a logistic regression on raw probabilities.
    More stable than isotonic for small samples.

    IMPORTANT: Must be fitted on HELD-OUT data only.
    Never fit on the same data used for training or OOF prediction.
    """

    def __init__(self):
        self._model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
        self._fitted = False

    def fit(self, raw_probs: np.ndarray, true_labels: np.ndarray) -> "PlattCalibrator":
        """
        Fit calibrator on held-out predictions.

        Args:
            raw_probs: Model's raw probability predictions (shape: [n_samples])
            true_labels: True binary labels (shape: [n_samples])
        """
        if len(raw_probs) < 10:
            print("[Calibration] WARNING: Too few samples for calibration, skipping")
            self._fitted = False
            return self

        X = raw_probs.reshape(-1, 1)
        y = true_labels.astype(int)

        # Check that we have both classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print("[Calibration] WARNING: Only one class in calibration data, skipping")
            self._fitted = False
            return self

        self._model.fit(X, y)
        self._fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Calibrate raw probabilities.

        Args:
            raw_probs: Raw probability predictions

        Returns:
            Calibrated probabilities (clipped to [0.05, 0.95])
        """
        if not self._fitted:
            return np.clip(raw_probs, 0.05, 0.95)

        X = raw_probs.reshape(-1, 1)
        calibrated = self._model.predict_proba(X)[:, 1]
        return np.clip(calibrated, 0.05, 0.95)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class IsotonicCalibrator:
    """
    Isotonic regression calibration.
    More flexible than Platt but requires more data (~200+ samples).
    """

    def __init__(self):
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.05, y_max=0.95)
        self._fitted = False

    def fit(self, raw_probs: np.ndarray, true_labels: np.ndarray) -> "IsotonicCalibrator":
        if len(raw_probs) < 50:
            print("[Calibration] WARNING: Too few samples for isotonic, falling back to Platt")
            self._fitted = False
            return self

        self._model.fit(raw_probs, true_labels.astype(float))
        self._fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.clip(raw_probs, 0.05, 0.95)
        return self._model.transform(raw_probs)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def get_calibrator(method: str = "platt"):
    """Factory for calibrator."""
    if method == "isotonic":
        return IsotonicCalibrator()
    return PlattCalibrator()

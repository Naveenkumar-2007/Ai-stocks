"""
Unified Engine — Configuration
===============================
All hyperparameters in ONE place. No magic numbers scattered across files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class UnifiedConfig:
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    base_dir: Path = Path(__file__).resolve().parents[1]  # backend/
    model_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "unified_engine" / "models"
    )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    fetch_days: int = 1500  # ~6 years of daily bars
    min_train_rows: int = 504  # ~2 years minimum for training
    min_total_rows: int = 600  # after feature NaN drop

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------
    prediction_horizon: int = 5  # 5 trading days forward
    prediction_days_ui: int = 7  # max UI prediction days

    # -------------------------------------------------------------------------
    # Walk-Forward Validation
    # -------------------------------------------------------------------------
    wf_min_train_days: int = 504  # 2 years minimum training window
    wf_test_window: int = 63     # 3 months test per fold
    wf_step_size: int = 21       # 1 month step
    wf_purge_days: int = 10      # 2× prediction horizon — prevents leakage
    wf_embargo_days: int = 5     # extra gap after test window
    wf_min_folds: int = 3        # minimum folds required

    # -------------------------------------------------------------------------
    # XGBoost (Direction classifier)
    # -------------------------------------------------------------------------
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "min_child_weight": 10,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    })

    # -------------------------------------------------------------------------
    # LightGBM (Magnitude regressor)
    # -------------------------------------------------------------------------
    lgbm_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "min_child_samples": 30,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    })

    # -------------------------------------------------------------------------
    # Meta-Learner
    # -------------------------------------------------------------------------
    meta_learner_C: float = 0.1  # strong regularization
    meta_n_inner_splits: int = 3  # inner CV for meta-learner

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------
    calibration_method: str = "platt"  # "platt" or "isotonic"
    calibration_fold_pct: float = 0.15  # hold out 15% of OOF for calibration

    # -------------------------------------------------------------------------
    # Trading / Signal Generation
    # -------------------------------------------------------------------------
    entry_threshold: float = 0.55
    exit_threshold: float = 0.45
    strong_signal_confidence: float = 0.60

    # -------------------------------------------------------------------------
    # Risk
    # -------------------------------------------------------------------------
    vol_target: float = 0.15
    max_leverage: float = 1.5
    stop_loss_pct: float = 0.05
    trailing_stop_pct: float = 0.08

    # -------------------------------------------------------------------------
    # Feature selection
    # -------------------------------------------------------------------------
    max_features: int = 25  # top-K features by importance
    min_feature_importance: float = 0.005  # minimum importance to keep

    def ensure_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)


CONFIG = UnifiedConfig()
CONFIG.ensure_dirs()

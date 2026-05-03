"""
Unified Engine — Configuration (Production v5.0)
==================================================
ALL hyperparameters in ONE place.  No magic numbers scattered across files.
Tuned by senior ML engineering standards for financial time-series.
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
    fetch_days: int = 2000            # ~8 years — more data = more walk-forward folds
    min_train_rows: int = 504         # ~2 years minimum for training
    min_total_rows: int = 400         # after feature NaN drop (lowered for intl tickers)

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------
    prediction_horizon: int = 5       # 5 trading days forward
    prediction_days_ui: int = 7       # max UI prediction days

    # -------------------------------------------------------------------------
    # Target Construction (NEW: volatility-adjusted thresholds)
    # -------------------------------------------------------------------------
    target_threshold_vol_multiplier: float = 0.3  # move must exceed 0.3× daily vol
    target_use_threshold: bool = True              # enable threshold-based labeling

    # -------------------------------------------------------------------------
    # Walk-Forward Validation
    # -------------------------------------------------------------------------
    wf_min_train_days: int = 504      # 2 years minimum training window
    wf_test_window: int = 63          # 3 months test per fold
    wf_step_size: int = 21            # 1 month step
    wf_purge_days: int = 10           # 2× prediction horizon — prevents leakage
    wf_embargo_days: int = 5          # extra gap after test window
    wf_min_folds: int = 3             # minimum folds required

    # -------------------------------------------------------------------------
    # XGBoost (Direction classifier) — stronger regularization
    # -------------------------------------------------------------------------
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.015,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "reg_alpha": 1.5,
        "reg_lambda": 4.0,
        "min_child_weight": 12,
        "gamma": 0.05,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    })

    # -------------------------------------------------------------------------
    # LightGBM (Magnitude regressor)
    # -------------------------------------------------------------------------
    lgbm_params: Dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.015,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "reg_alpha": 1.5,
        "reg_lambda": 4.0,
        "min_child_samples": 25,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    })

    # -------------------------------------------------------------------------
    # LightGBM Quantile Regression (for real confidence intervals)
    # -------------------------------------------------------------------------
    quantile_alphas: list = field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90])
    lgbm_quantile_params: Dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.8,
        "reg_lambda": 2.0,
        "min_child_samples": 25,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    })

    # -------------------------------------------------------------------------
    # Meta-Learner
    # -------------------------------------------------------------------------
    meta_learner_C: float = 0.1       # strong regularization
    meta_n_inner_splits: int = 3      # inner CV for meta-learner

    # -------------------------------------------------------------------------
    # Calibration  — larger fold for stability
    # -------------------------------------------------------------------------
    calibration_method: str = "platt"  # "platt" is more stable for small samples
    calibration_fold_pct: float = 0.20 # 20% held-out for calibration (balance between stability and training data)

    # -------------------------------------------------------------------------
    # Sample Weighting
    # -------------------------------------------------------------------------
    use_sample_weights: bool = True
    sample_weight_halflife_days: int = 700  # exponential decay half-life

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
    max_features: int = 30            # top-K features by importance (was 25)
    min_feature_importance: float = 0.004  # slightly lower to include v5.1 features

    def ensure_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)


CONFIG = UnifiedConfig()
CONFIG.ensure_dirs()

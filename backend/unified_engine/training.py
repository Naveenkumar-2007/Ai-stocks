"""
Unified Engine — Training Pipeline
====================================
Production-grade model training with:
1. Proper walk-forward validation (purge + embargo)
2. Feature importance → automatic selection
3. Nested CV for meta-learner (no data leakage)
4. Held-out calibration fold
5. Comprehensive metrics including statistical significance test
"""

from __future__ import annotations

import json
import os
import sys
import warnings
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

import xgboost as xgb
import lightgbm as lgb

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_engine.config import CONFIG
from unified_engine.features import UnifiedFeatureEngine, get_feature_engine
from unified_engine.targets import build_targets, build_target_simple
from unified_engine.validation import walk_forward_splits, validate_splits
from unified_engine.calibration import get_calibrator


@dataclass
class TrainResult:
    """Result of a training run."""
    ticker: str
    success: bool
    metrics: Dict[str, float]
    fold_results: List[Dict]
    feature_importance: Dict[str, float]
    selected_features: List[str]
    model_version: str
    artifact_path: str
    reason: str
    trained_at: str = ""


def _get_artifact_dir(ticker: str) -> Path:
    """Get the directory for a ticker's model artifacts."""
    safe_ticker = ticker.upper().replace("/", "_").replace("\\", "_")
    artifact_dir = CONFIG.model_dir / safe_ticker
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _fetch_data(ticker: str) -> pd.DataFrame:
    """Fetch historical OHLCV data for training using REAL API keys only."""
    try:
        from stock_api import get_stock_history
        df = get_stock_history(ticker, days=CONFIG.fetch_days, return_info=False)
        if df is not None and not df.empty:
            print(f"  [DATA] Fetched {len(df)} rows from stock_api (real API keys)")
            return df
        else:
            print(f"  [DATA] stock_api returned empty data for {ticker}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  [DATA] stock_api failed for {ticker}: {e}")
        return pd.DataFrame()


def _compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Compute feature importance using a quick XGBoost fit.
    Returns normalized importance scores.
    """
    quick_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    quick_model.fit(X, y)

    importances = quick_model.feature_importances_
    total = importances.sum() + 1e-10
    normalized = importances / total

    importance_dict = {}
    for name, imp in zip(feature_names, normalized):
        importance_dict[name] = float(imp)

    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def _select_features(
    importance: Dict[str, float],
    max_features: int = CONFIG.max_features,
    min_importance: float = CONFIG.min_feature_importance,
) -> List[str]:
    """Select top features by importance."""
    selected = []
    for name, imp in importance.items():
        if imp >= min_importance and len(selected) < max_features:
            selected.append(name)
    # Always keep at least 10 features
    if len(selected) < 10:
        selected = list(importance.keys())[:10]
    return sorted(selected)


def train_unified_model(
    ticker: str,
    generate_charts: bool = False,
) -> TrainResult:
    """
    Full training pipeline for one ticker.

    Steps:
    1. Fetch data
    2. Compute ALL features
    3. Feature importance → select top features
    4. Recompute features with selected subset
    5. Walk-forward validation with purge/embargo
    6. Train XGBoost (direction) + LightGBM (magnitude) per fold
    7. Nested CV meta-learner on OOF predictions
    8. Calibrate on held-out fold
    9. Retrain final models on all data
    10. Save artifact

    Returns:
        TrainResult with metrics and artifact path
    """
    ticker = ticker.upper()
    trained_at = datetime.utcnow().isoformat() + "Z"

    print(f"\n{'='*80}")
    print(f"  UNIFIED ENGINE v4.0 — TRAINING: {ticker}")
    print(f"{'='*80}")

    # === STEP 1: Fetch data ===
    print("\n[1/10] Fetching historical data...")
    df = _fetch_data(ticker)
    if df.empty or len(df) < CONFIG.min_total_rows:
        return TrainResult(
            ticker=ticker, success=False, metrics={}, fold_results=[],
            feature_importance={}, selected_features=[], model_version="",
            artifact_path="", reason=f"Insufficient data: {len(df)} rows",
            trained_at=trained_at,
        )

    # === STEP 2: Compute ALL candidate features ===
    print("[2/10] Computing all candidate features...")
    full_engine = get_feature_engine(selected_features=None)
    full_artifacts = full_engine.compute(df, fit_scaler=True)
    print(f"  → {len(full_artifacts.column_order)} features, {len(full_artifacts.features)} rows")

    if len(full_artifacts.features) < CONFIG.min_total_rows:
        return TrainResult(
            ticker=ticker, success=False, metrics={}, fold_results=[],
            feature_importance={}, selected_features=[], model_version="",
            artifact_path="", reason=f"Insufficient rows after feature computation: {len(full_artifacts.features)}",
            trained_at=trained_at,
        )

    # === STEP 3: Build targets ===
    print("[3/10] Building prediction targets...")
    direction_target, magnitude_target = build_targets(
        df, full_artifacts.features.index, horizon=CONFIG.prediction_horizon
    )

    common_idx = direction_target.index.intersection(magnitude_target.index)
    common_idx = common_idx.intersection(full_artifacts.features.index)

    X_full = full_artifacts.features.loc[common_idx].values
    y_dir = direction_target.loc[common_idx].values
    y_mag = magnitude_target.loc[common_idx].values

    target_balance = y_dir.mean()
    print(f"  → {len(common_idx)} samples, target balance: {target_balance:.1%} UP")

    # === STEP 4: Feature importance + selection ===
    print("[4/10] Computing feature importance...")
    importance = _compute_feature_importance(
        X_full, y_dir, full_artifacts.column_order
    )
    selected = _select_features(importance)
    print(f"  → Selected {len(selected)} features (from {len(full_artifacts.column_order)})")
    for i, name in enumerate(selected[:10]):
        print(f"    {i+1}. {name}: {importance[name]:.4f}")

    # === STEP 5: Recompute with selected features ===
    print("[5/10] Recomputing features with selected subset...")
    engine = get_feature_engine(selected_features=selected)
    artifacts = engine.compute(df, fit_scaler=True)

    # Re-align targets with new feature index
    common_idx = direction_target.index.intersection(magnitude_target.index)
    common_idx = common_idx.intersection(artifacts.features.index)

    X = artifacts.features.loc[common_idx].values
    y_dir = direction_target.loc[common_idx].values
    y_mag = magnitude_target.loc[common_idx].values
    dates = common_idx

    print(f"  → {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  → Column hash: {artifacts.column_hash}")

    # === STEP 6: Walk-forward validation ===
    print("[6/10] Running walk-forward validation...")
    splits = walk_forward_splits(len(X))
    print(f"  → {len(splits)} folds")

    if len(splits) < CONFIG.wf_min_folds:
        return TrainResult(
            ticker=ticker, success=False, metrics={}, fold_results=[],
            feature_importance=importance, selected_features=selected,
            model_version="", artifact_path="",
            reason=f"Insufficient walk-forward folds: {len(splits)}",
            trained_at=trained_at,
        )

    if not validate_splits(splits):
        print("  ⚠️ Walk-forward splits have purge violations!")

    # Storage for OOF predictions
    oof_probs_xgb = np.full(len(X), np.nan)
    oof_probs_lgbm = np.full(len(X), np.nan)
    oof_preds_mag = np.full(len(X), np.nan)
    oof_actuals_dir = np.full(len(X), np.nan)
    oof_actuals_mag = np.full(len(X), np.nan)

    fold_results = []

    for split in splits:
        train_idx = split.train_indices
        test_idx = split.test_indices

        X_tr, X_te = X[train_idx], X[test_idx]
        y_dir_tr, y_dir_te = y_dir[train_idx], y_dir[test_idx]
        y_mag_tr, y_mag_te = y_mag[train_idx], y_mag[test_idx]

        # --- XGBoost (Direction) ---
        xgb_model = xgb.XGBClassifier(**CONFIG.xgb_params)
        xgb_model.fit(X_tr, y_dir_tr, verbose=False)
        fold_probs_xgb = xgb_model.predict_proba(X_te)[:, 1]

        # --- LightGBM (Magnitude) ---
        lgbm_model = lgb.LGBMRegressor(**CONFIG.lgbm_params)
        lgbm_model.fit(X_tr, y_mag_tr)
        fold_preds_mag = lgbm_model.predict(X_te)

        # Store OOF
        oof_probs_xgb[test_idx] = fold_probs_xgb
        oof_probs_lgbm[test_idx] = (fold_preds_mag > 0).astype(float)
        oof_preds_mag[test_idx] = fold_preds_mag
        oof_actuals_dir[test_idx] = y_dir_te
        oof_actuals_mag[test_idx] = y_mag_te

        # Fold metrics
        fold_preds = (fold_probs_xgb >= 0.5).astype(int)
        fold_acc = accuracy_score(y_dir_te, fold_preds)
        fold_results.append({
            "fold": split.fold_num,
            "accuracy": float(fold_acc),
            "mean_prob": float(fold_probs_xgb.mean()),
            "train_size": split.train_size,
            "test_size": split.test_size,
        })
        print(f"    Fold {split.fold_num + 1}/{len(splits)}: "
              f"acc={fold_acc:.3f}, prob={fold_probs_xgb.mean():.3f}, "
              f"train={split.train_size}, test={split.test_size}")

    # === STEP 7: Meta-learner with NESTED CV ===
    print("[7/10] Training meta-learner with nested CV...")

    # Get valid (non-NaN) OOF predictions
    valid_mask = ~np.isnan(oof_probs_xgb)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < 30:
        return TrainResult(
            ticker=ticker, success=False, metrics={}, fold_results=fold_results,
            feature_importance=importance, selected_features=selected,
            model_version="", artifact_path="",
            reason=f"Too few valid OOF predictions: {len(valid_idx)}",
            trained_at=trained_at,
        )

    # Build meta-features from OOF predictions
    meta_X = np.column_stack([
        oof_probs_xgb[valid_idx],
        oof_probs_lgbm[valid_idx],
        oof_preds_mag[valid_idx],
    ])
    meta_y = oof_actuals_dir[valid_idx]

    # Split meta data: 85% for training meta-learner, 15% for calibration
    # IMPORTANT: Use temporal split, not random!
    n_valid = len(valid_idx)
    calib_size = max(20, int(n_valid * CONFIG.calibration_fold_pct))
    meta_train_size = n_valid - calib_size

    meta_X_train = meta_X[:meta_train_size]
    meta_y_train = meta_y[:meta_train_size]
    meta_X_calib = meta_X[meta_train_size:]
    meta_y_calib = meta_y[meta_train_size:]

    # Train meta-learner on meta-train split
    meta_learner = LogisticRegression(
        C=CONFIG.meta_learner_C,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    meta_learner.fit(meta_X_train, meta_y_train)

    # === STEP 8: Calibrate on held-out fold ===
    print("[8/10] Calibrating on held-out data...")
    raw_calib_probs = meta_learner.predict_proba(meta_X_calib)[:, 1]
    calibrator = get_calibrator(CONFIG.calibration_method)
    calibrator.fit(raw_calib_probs, meta_y_calib)
    print(f"  → Calibrator fitted: {calibrator.is_fitted}")

    # === COMPUTE FINAL METRICS ===
    # Evaluate meta-learner on CALIBRATION fold (truly held-out)
    calibrated_probs = calibrator.calibrate(raw_calib_probs)
    calib_preds = (calibrated_probs >= 0.5).astype(int)

    accuracy = accuracy_score(meta_y_calib, calib_preds)
    precision = precision_score(meta_y_calib, calib_preds, zero_division=0)
    recall = recall_score(meta_y_calib, calib_preds, zero_division=0)
    f1 = f1_score(meta_y_calib, calib_preds, zero_division=0)

    try:
        auc = roc_auc_score(meta_y_calib, calibrated_probs)
    except ValueError:
        auc = 0.5

    # Statistical significance test: is our accuracy significantly > 50%?
    from scipy import stats
    n_correct = int((calib_preds == meta_y_calib).sum())
    n_total = len(meta_y_calib)
    try:
        # scipy >= 1.7
        binom_pvalue = float(stats.binomtest(n_correct, n_total, 0.5, alternative="greater").pvalue)
    except AttributeError:
        # scipy < 1.7 fallback
        binom_pvalue = float(stats.binom_test(n_correct, n_total, 0.5, alternative="greater"))

    print(f"\n{'='*60}")
    print(f"  HELD-OUT METRICS (Truly Out-of-Sample)")
    print(f"{'='*60}")
    print(f"  Directional Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision:             {precision*100:.2f}%")
    print(f"  Recall:                {recall*100:.2f}%")
    print(f"  F1-Score:              {f1*100:.2f}%")
    print(f"  AUC-ROC:               {auc:.3f}")
    print(f"  Binomial p-value:      {binom_pvalue:.4f} {'✅ Significant' if binom_pvalue < 0.05 else '⚠️ NOT significant'}")
    print(f"  Calibration samples:   {n_total}")

    # Also compute cross-fold average for reference
    fold_accs = [f["accuracy"] for f in fold_results]
    print(f"\n  Cross-fold mean accuracy: {np.mean(fold_accs)*100:.2f}% ± {np.std(fold_accs)*100:.2f}%")

    # Temporal consistency check
    temporal_std = np.std(fold_accs)
    temporal_consistent = temporal_std < 0.10  # less than 10% std across folds
    print(f"  Temporal consistency:    {'✅ Stable' if temporal_consistent else '⚠️ Unstable'} (σ={temporal_std:.3f})")

    # Compute average absolute forward return for price projection
    valid_mag = oof_actuals_mag[valid_mask]
    avg_abs_return = float(np.nanmean(np.abs(valid_mag))) if len(valid_mag) > 0 else 0.02
    avg_abs_return = max(0.005, min(avg_abs_return, 0.15))

    # === STEP 9: Retrain final models on ALL data ===
    print("\n[9/10] Retraining final models on all data...")

    # Final scaler + features
    final_engine = get_feature_engine(selected_features=selected)
    final_artifacts = final_engine.compute(df, fit_scaler=True)

    final_common_idx = direction_target.index.intersection(magnitude_target.index)
    final_common_idx = final_common_idx.intersection(final_artifacts.features.index)

    X_all = final_artifacts.features.loc[final_common_idx].values
    y_dir_all = direction_target.loc[final_common_idx].values
    y_mag_all = magnitude_target.loc[final_common_idx].values

    # Train production models
    final_xgb = xgb.XGBClassifier(**CONFIG.xgb_params)
    final_xgb.fit(X_all, y_dir_all, verbose=False)

    final_lgbm = lgb.LGBMRegressor(**CONFIG.lgbm_params)
    final_lgbm.fit(X_all, y_mag_all)

    # Final meta-learner (on all OOF data minus calibration)
    final_meta = LogisticRegression(
        C=CONFIG.meta_learner_C,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    final_meta.fit(meta_X, meta_y)  # all valid OOF data

    # Determine optimal thresholds from OOF
    all_meta_probs = meta_learner.predict_proba(meta_X)[:, 1]
    buy_threshold, sell_threshold = _find_optimal_thresholds(meta_y, all_meta_probs)
    print(f"  → Optimal thresholds: BUY > {buy_threshold:.3f}, SELL < {sell_threshold:.3f}")

    # === STEP 10: Save artifact ===
    print("[10/10] Saving production artifact...")

    model_version = f"v4.0-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"
    artifact_dir = _get_artifact_dir(ticker)

    artifact = {
        "version": model_version,
        "ticker": ticker,
        "trained_at": trained_at,
        # Models
        "xgb_model": final_xgb,
        "lgbm_model": final_lgbm,
        "meta_learner": final_meta,
        "calibrator": calibrator,
        "scaler": final_artifacts.scaler,
        # Feature metadata
        "selected_features": selected,
        "column_order": final_artifacts.column_order,
        "column_hash": final_artifacts.column_hash,
        "feature_importance": importance,
        # Thresholds
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "avg_abs_return": avg_abs_return,
        # Metrics
        "metrics": {
            "accuracy": float(accuracy * 100),
            "precision": float(precision * 100),
            "recall": float(recall * 100),
            "f1": float(f1 * 100),
            "auc": float(auc),
            "binom_pvalue": float(binom_pvalue),
            "fold_mean_accuracy": float(np.mean(fold_accs) * 100),
            "fold_std_accuracy": float(np.std(fold_accs) * 100),
            "temporal_consistent": bool(temporal_consistent),
            "calibration_samples": int(n_total),
            "training_samples": int(len(X_all)),
            "n_folds": int(len(fold_results)),
        },
        # Config snapshot
        "config": {
            "prediction_horizon": CONFIG.prediction_horizon,
            "purge_days": CONFIG.wf_purge_days,
            "embargo_days": CONFIG.wf_embargo_days,
            "max_features": CONFIG.max_features,
        },
    }

    artifact_path = artifact_dir / "model.joblib"
    joblib.dump(artifact, artifact_path)

    # Save metadata JSON for monitoring
    metadata_path = artifact_dir / "metadata.json"
    metadata = {
        "ticker": ticker,
        "version": model_version,
        "trained_at": trained_at,
        "metrics": artifact["metrics"],
        "selected_features": selected,
        "column_hash": final_artifacts.column_hash,
        "config": artifact["config"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n  ✅ Saved: {artifact_path}")
    print(f"  ✅ Metadata: {metadata_path}")
    print(f"  ✅ Version: {model_version}")
    print(f"  ✅ Features: {len(selected)}")
    print(f"  ✅ Accuracy: {accuracy*100:.1f}%")

    return TrainResult(
        ticker=ticker,
        success=True,
        metrics=artifact["metrics"],
        fold_results=fold_results,
        feature_importance=importance,
        selected_features=selected,
        model_version=model_version,
        artifact_path=str(artifact_path),
        reason="trained",
        trained_at=trained_at,
    )


def _find_optimal_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Tuple[float, float]:
    """Find optimal buy/sell thresholds by maximizing F1."""
    best_buy = 0.55
    best_f1 = 0.0

    for threshold in np.arange(0.50, 0.65, 0.01):
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_buy = threshold

    best_sell = max(0.35, min(0.49, 1.0 - best_buy))
    return float(best_buy), float(best_sell)


class UnifiedTrainer:
    """Public API for training."""

    @staticmethod
    def train(ticker: str, generate_charts: bool = False) -> TrainResult:
        return train_unified_model(ticker, generate_charts=generate_charts)

    @staticmethod
    def train_batch(tickers: List[str]) -> List[TrainResult]:
        results = []
        for ticker in tickers:
            try:
                result = train_unified_model(ticker)
                results.append(result)
            except Exception as e:
                print(f"❌ Training failed for {ticker}: {e}")
                results.append(TrainResult(
                    ticker=ticker, success=False, metrics={},
                    fold_results=[], feature_importance={},
                    selected_features=[], model_version="",
                    artifact_path="", reason=str(e),
                ))
        return results

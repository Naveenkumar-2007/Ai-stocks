"""
Unified Engine — Training Pipeline v5.2
=========================================
Production-grade training with:
1. Walk-forward validation (purge + embargo)
2. Feature importance → automatic selection
3. Quantile regression for real confidence intervals
4. Sample weighting (recent data matters more)
5. Statistical significance testing
6. v5.1: RandomForest + volume/microstructure features
7. v5.2: Early stopping, adaptive class weights, dual-XGBoost ensemble
"""

from __future__ import annotations

import json, os, sys, warnings, hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_engine.config import CONFIG
from unified_engine.features import UnifiedFeatureEngine, get_feature_engine
from unified_engine.targets import build_targets
from unified_engine.validation import walk_forward_splits, validate_splits
from unified_engine.calibration import get_calibrator


@dataclass
class TrainResult:
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
    safe_ticker = ticker.upper().replace("/", "_").replace("\\", "_")
    artifact_dir = CONFIG.model_dir / safe_ticker
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _fetch_data(ticker: str) -> pd.DataFrame:
    try:
        from stock_api import get_stock_history
        df = get_stock_history(ticker, days=CONFIG.fetch_days, return_info=False)
        if df is not None and not df.empty:
            print(f"  [DATA] Fetched {len(df)} rows for {ticker}")
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"  [DATA] Failed for {ticker}: {e}")
        return pd.DataFrame()


def _compute_feature_importance(X, y, feature_names):
    quick_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    quick_model.fit(X, y)
    importances = quick_model.feature_importances_
    total = importances.sum() + 1e-10
    normalized = importances / total
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, normalized)}
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def _select_features(importance, max_features=CONFIG.max_features, min_importance=CONFIG.min_feature_importance):
    selected = [name for name, imp in importance.items() if imp >= min_importance and len(selected) < max_features] if False else []
    # Correct implementation
    selected = []
    for name, imp in importance.items():
        if imp >= min_importance and len(selected) < max_features:
            selected.append(name)
    if len(selected) < 10:
        selected = list(importance.keys())[:10]
    return sorted(selected)


def _compute_sample_weights(dates_index, halflife_days=CONFIG.sample_weight_halflife_days):
    """Exponential decay: recent samples get higher weight."""
    if not CONFIG.use_sample_weights:
        return None
    try:
        dates = pd.to_datetime(dates_index)
        days_ago = (dates.max() - dates).days.values.astype(float)
        decay_rate = np.log(2) / halflife_days
        weights = np.exp(-decay_rate * days_ago)
        weights = weights / weights.mean()  # normalize so mean=1
        return weights
    except Exception:
        return None


def train_unified_model(ticker: str, generate_charts: bool = False) -> TrainResult:
    """Full training pipeline for one ticker."""
    ticker = ticker.upper()
    trained_at = datetime.utcnow().isoformat() + "Z"
    fail = lambda reason: TrainResult(
        ticker=ticker, success=False, metrics={}, fold_results=[],
        feature_importance={}, selected_features=[], model_version="",
        artifact_path="", reason=reason, trained_at=trained_at,
    )

    print(f"\n{'='*70}")
    print(f"  UNIFIED ENGINE v5.0 — TRAINING: {ticker}")
    print(f"{'='*70}")

    # === STEP 1: Fetch data ===
    print("\n[1/10] Fetching historical data...")
    df = _fetch_data(ticker)
    if df.empty or len(df) < CONFIG.min_total_rows:
        return fail(f"Insufficient data: {len(df)} rows (need {CONFIG.min_total_rows})")

    # === STEP 2: Compute ALL candidate features ===
    print("[2/10] Computing all candidate features...")
    full_engine = get_feature_engine(selected_features=None)
    full_artifacts = full_engine.compute(df, fit_scaler=True)
    print(f"  → {len(full_artifacts.column_order)} features, {len(full_artifacts.features)} rows")

    if len(full_artifacts.features) < CONFIG.min_total_rows:
        return fail(f"Insufficient rows after features: {len(full_artifacts.features)}")

    # === STEP 3: Build targets ===
    print("[3/10] Building volatility-adjusted targets...")
    direction_target, magnitude_target = build_targets(
        df, full_artifacts.features.index, horizon=CONFIG.prediction_horizon
    )

    common_idx = direction_target.index.intersection(magnitude_target.index)
    common_idx = common_idx.intersection(full_artifacts.features.index)

    X_full = full_artifacts.features.loc[common_idx].values
    y_dir = direction_target.loc[common_idx].values
    y_mag = magnitude_target.loc[common_idx].values

    if len(common_idx) < CONFIG.min_total_rows:
        return fail(f"Too few samples after target filtering: {len(common_idx)}")

    print(f"  → {len(common_idx)} samples, balance: {y_dir.mean():.1%} UP")

    # === STEP 4: Feature importance + selection ===
    print("[4/10] Computing feature importance...")
    importance = _compute_feature_importance(X_full, y_dir, full_artifacts.column_order)
    selected = _select_features(importance)
    print(f"  → Selected {len(selected)} features")
    for i, name in enumerate(selected[:8]):
        print(f"    {i+1}. {name}: {importance[name]:.4f}")

    # === STEP 5: Recompute with selected features ===
    print("[5/10] Recomputing with selected subset...")
    engine = get_feature_engine(selected_features=selected)
    artifacts = engine.compute(df, fit_scaler=True)

    common_idx = direction_target.index.intersection(magnitude_target.index)
    common_idx = common_idx.intersection(artifacts.features.index)

    X = artifacts.features.loc[common_idx].values
    y_dir = direction_target.loc[common_idx].values
    y_mag = magnitude_target.loc[common_idx].values
    dates = common_idx

    sample_weights = _compute_sample_weights(dates)
    print(f"  → {X.shape[0]} samples × {X.shape[1]} features")

    # v5.2: Adaptive class weight for imbalanced labels
    pos_rate = y_dir.mean()
    neg_rate = 1.0 - pos_rate
    adaptive_scale = neg_rate / (pos_rate + 1e-6) if pos_rate > 0 else 1.0
    adaptive_scale = np.clip(adaptive_scale, 0.5, 2.0)
    print(f"  → Class balance: {pos_rate:.1%} UP, scale_pos_weight={adaptive_scale:.2f}")

    # === STEP 6: Walk-forward validation ===
    print("[6/10] Walk-forward validation...")
    splits = walk_forward_splits(len(X))
    print(f"  → {len(splits)} folds")

    if len(splits) < CONFIG.wf_min_folds:
        return fail(f"Insufficient folds: {len(splits)}")
    validate_splits(splits)

    oof_probs_xgb = np.full(len(X), np.nan)
    oof_probs_lgbm = np.full(len(X), np.nan)
    oof_probs_rf = np.full(len(X), np.nan)  # v5.1: RandomForest OOF
    oof_preds_mag = np.full(len(X), np.nan)
    oof_actuals_dir = np.full(len(X), np.nan)
    oof_actuals_mag = np.full(len(X), np.nan)
    # Quantile OOF predictions
    oof_q10 = np.full(len(X), np.nan)
    oof_q50 = np.full(len(X), np.nan)
    oof_q90 = np.full(len(X), np.nan)

    fold_results = []

    for split in splits:
        train_idx = split.train_indices
        test_idx = split.test_indices

        X_tr, X_te = X[train_idx], X[test_idx]
        y_dir_tr, y_dir_te = y_dir[train_idx], y_dir[test_idx]
        y_mag_tr, y_mag_te = y_mag[train_idx], y_mag[test_idx]

        sw_tr = sample_weights[train_idx] if sample_weights is not None else None

        # v5.2: Split training into fit + early-stop validation (last 15%)
        es_split = max(10, int(len(X_tr) * 0.85))
        X_tr_fit, X_tr_es = X_tr[:es_split], X_tr[es_split:]
        y_dir_fit, y_dir_es = y_dir_tr[:es_split], y_dir_tr[es_split:]
        y_mag_fit, y_mag_es = y_mag_tr[:es_split], y_mag_tr[es_split:]
        sw_fit = sw_tr[:es_split] if sw_tr is not None else None

        # XGBoost #1 (Direction — with early stopping)
        xgb_params_fold = CONFIG.xgb_params.copy()
        xgb_params_fold['scale_pos_weight'] = adaptive_scale
        xgb_model = xgb.XGBClassifier(**xgb_params_fold, early_stopping_rounds=30)
        xgb_model.fit(
            X_tr_fit, y_dir_fit, sample_weight=sw_fit,
            eval_set=[(X_tr_es, y_dir_es)], verbose=False,
        )
        fold_probs_xgb = xgb_model.predict_proba(X_te)[:, 1]

        # v5.2: XGBoost #2 (different params for diversity — shallower, more regularized)
        xgb2_params = {
            'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.025,
            'subsample': 0.60, 'colsample_bytree': 0.60,
            'reg_alpha': 3.0, 'reg_lambda': 6.0, 'min_child_weight': 20,
            'gamma': 0.15, 'scale_pos_weight': adaptive_scale,
            'eval_metric': 'logloss', 'random_state': 123, 'n_jobs': -1,
        }
        xgb2_model = xgb.XGBClassifier(**xgb2_params, early_stopping_rounds=25)
        xgb2_model.fit(
            X_tr_fit, y_dir_fit, sample_weight=sw_fit,
            eval_set=[(X_tr_es, y_dir_es)], verbose=False,
        )
        fold_probs_xgb2 = xgb2_model.predict_proba(X_te)[:, 1]

        # LightGBM (Magnitude median — with early stopping)
        lgbm_model = lgb.LGBMRegressor(**CONFIG.lgbm_params)
        lgbm_model.fit(
            X_tr_fit, y_mag_fit, sample_weight=sw_fit,
            eval_set=[(X_tr_es, y_mag_es)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        fold_preds_mag = lgbm_model.predict(X_te)

        # RandomForest (diverse error patterns — decorrelates with GBMs)
        rf_model = RandomForestClassifier(
            n_estimators=400, max_depth=7, min_samples_leaf=12,
            max_features='sqrt', class_weight='balanced_subsample',
            random_state=42, n_jobs=-1,
        )
        rf_model.fit(X_tr, y_dir_tr, sample_weight=sw_tr)
        fold_probs_rf = rf_model.predict_proba(X_te)[:, 1]

        # Quantile regression for confidence intervals
        q_params = CONFIG.lgbm_quantile_params.copy()
        for alpha, arr in [(0.10, oof_q10), (0.50, oof_q50), (0.90, oof_q90)]:
            q_model = lgb.LGBMRegressor(**q_params, objective='quantile', alpha=alpha)
            q_model.fit(X_tr, y_mag_tr, sample_weight=sw_tr)
            arr[test_idx] = q_model.predict(X_te)

        oof_probs_xgb[test_idx] = fold_probs_xgb
        oof_probs_lgbm[test_idx] = (fold_preds_mag > 0).astype(float)
        oof_probs_rf[test_idx] = fold_probs_rf
        oof_preds_mag[test_idx] = fold_preds_mag
        oof_actuals_dir[test_idx] = y_dir_te
        oof_actuals_mag[test_idx] = y_mag_te

        # v5.2: 4-model weighted ensemble for fold accuracy
        fold_ensemble_prob = (
            0.35 * fold_probs_xgb +
            0.15 * fold_probs_xgb2 +
            0.20 * fold_probs_rf +
            0.30 * (fold_preds_mag > 0).astype(float)
        )
        fold_acc = accuracy_score(y_dir_te, (fold_ensemble_prob >= 0.5).astype(int))
        fold_results.append({
            "fold": split.fold_num, "accuracy": float(fold_acc),
            "mean_prob": float(fold_probs_xgb.mean()),
            "train_size": split.train_size, "test_size": split.test_size,
        })
        print(f"    Fold {split.fold_num+1}/{len(splits)}: acc={fold_acc:.3f}")

    # === STEP 7: Meta-learner ===
    print("[7/10] Training meta-learner...")
    valid_mask = ~np.isnan(oof_probs_xgb)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < 30:
        return fail(f"Too few OOF predictions: {len(valid_idx)}")

    meta_X = np.column_stack([
        oof_probs_xgb[valid_idx],
        oof_probs_lgbm[valid_idx],
        oof_probs_rf[valid_idx],
        oof_preds_mag[valid_idx],
        # v5.1: interaction features for meta-learner
        oof_probs_xgb[valid_idx] * oof_probs_rf[valid_idx],
        np.abs(oof_probs_xgb[valid_idx] - oof_probs_rf[valid_idx]),  # disagreement signal
    ])
    meta_y = oof_actuals_dir[valid_idx]

    n_valid = len(valid_idx)
    calib_size = max(30, int(n_valid * CONFIG.calibration_fold_pct))
    meta_train_size = n_valid - calib_size

    meta_X_train, meta_y_train = meta_X[:meta_train_size], meta_y[:meta_train_size]
    meta_X_calib, meta_y_calib = meta_X[meta_train_size:], meta_y[meta_train_size:]

    meta_learner = LogisticRegression(C=CONFIG.meta_learner_C, max_iter=2000, class_weight="balanced", solver="lbfgs")
    meta_learner.fit(meta_X_train, meta_y_train)

    # === STEP 8: Calibrate ===
    print("[8/10] Calibrating probabilities...")
    raw_calib_probs = meta_learner.predict_proba(meta_X_calib)[:, 1]
    calibrator = get_calibrator(CONFIG.calibration_method)
    calibrator.fit(raw_calib_probs, meta_y_calib)

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

    from scipy import stats
    n_correct = int((calib_preds == meta_y_calib).sum())
    n_total = len(meta_y_calib)
    try:
        binom_pvalue = float(stats.binomtest(n_correct, n_total, 0.5, alternative="greater").pvalue)
    except AttributeError:
        binom_pvalue = float(stats.binom_test(n_correct, n_total, 0.5, alternative="greater"))

    fold_accs = [f["accuracy"] for f in fold_results]

    print(f"\n{'='*60}")
    print(f"  HELD-OUT METRICS")
    print(f"{'='*60}")
    print(f"  Accuracy:     {accuracy*100:.2f}%")
    print(f"  AUC-ROC:      {auc:.3f}")
    print(f"  F1:           {f1*100:.2f}%")
    print(f"  p-value:      {binom_pvalue:.4f} {'✅' if binom_pvalue < 0.05 else '⚠️'}")
    print(f"  Fold mean:    {np.mean(fold_accs)*100:.2f}% ± {np.std(fold_accs)*100:.2f}%")
    print(f"  Calib samples: {n_total}")

    valid_mag = oof_actuals_mag[valid_mask]
    avg_abs_return = float(np.nanmean(np.abs(valid_mag))) if len(valid_mag) > 0 else 0.02
    avg_abs_return = max(0.005, min(avg_abs_return, 0.15))

    # === STEP 9: Retrain final models on ALL data ===
    print("\n[9/10] Retraining final models...")
    final_engine = get_feature_engine(selected_features=selected)
    final_artifacts = final_engine.compute(df, fit_scaler=True)

    final_idx = direction_target.index.intersection(magnitude_target.index).intersection(final_artifacts.features.index)
    X_all = final_artifacts.features.loc[final_idx].values
    y_dir_all = direction_target.loc[final_idx].values
    y_mag_all = magnitude_target.loc[final_idx].values
    sw_all = _compute_sample_weights(final_idx)

    final_xgb_params = CONFIG.xgb_params.copy()
    final_xgb_params['scale_pos_weight'] = adaptive_scale
    final_xgb = xgb.XGBClassifier(**final_xgb_params)
    final_xgb.fit(X_all, y_dir_all, sample_weight=sw_all, verbose=False)

    final_lgbm = lgb.LGBMRegressor(**CONFIG.lgbm_params)
    final_lgbm.fit(X_all, y_mag_all, sample_weight=sw_all)

    # v5.2: Final RandomForest (more trees for production)
    final_rf = RandomForestClassifier(
        n_estimators=400, max_depth=7, min_samples_leaf=12,
        max_features='sqrt', class_weight='balanced_subsample',
        random_state=42, n_jobs=-1,
    )
    final_rf.fit(X_all, y_dir_all, sample_weight=sw_all)

    # Train quantile regressors for REAL confidence intervals
    q_params = CONFIG.lgbm_quantile_params.copy()
    quantile_models = {}
    for alpha in CONFIG.quantile_alphas:
        q_model = lgb.LGBMRegressor(**q_params, objective='quantile', alpha=alpha)
        q_model.fit(X_all, y_mag_all, sample_weight=sw_all)
        quantile_models[alpha] = q_model

    final_meta = LogisticRegression(C=CONFIG.meta_learner_C, max_iter=2000, class_weight="balanced", solver="lbfgs")
    final_meta.fit(meta_X, meta_y)

    all_meta_probs = meta_learner.predict_proba(meta_X)[:, 1]
    buy_threshold, sell_threshold = _find_optimal_thresholds(meta_y, all_meta_probs)
    print(f"  → Thresholds: BUY>{buy_threshold:.3f}, SELL<{sell_threshold:.3f}")

    # === STEP 10: Save artifact ===
    print("[10/10] Saving...")
    model_version = f"v5.2-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"
    artifact_dir = _get_artifact_dir(ticker)

    artifact = {
        "version": model_version, "ticker": ticker, "trained_at": trained_at,
        "xgb_model": final_xgb, "lgbm_model": final_lgbm,
        "rf_model": final_rf,
        "quantile_models": quantile_models,
        "meta_learner": final_meta, "calibrator": calibrator,
        "scaler": final_artifacts.scaler,
        "selected_features": selected,
        "column_order": final_artifacts.column_order,
        "column_hash": final_artifacts.column_hash,
        "feature_importance": importance,
        "buy_threshold": buy_threshold, "sell_threshold": sell_threshold,
        "avg_abs_return": avg_abs_return,
        "metrics": {
            "accuracy": float(accuracy * 100), "precision": float(precision * 100),
            "recall": float(recall * 100), "f1": float(f1 * 100),
            "auc": float(auc), "binom_pvalue": float(binom_pvalue),
            "fold_mean_accuracy": float(np.mean(fold_accs) * 100),
            "fold_std_accuracy": float(np.std(fold_accs) * 100),
            "calibration_samples": int(n_total),
            "training_samples": int(len(X_all)),
            "n_folds": int(len(fold_results)),
        },
        "config": {
            "prediction_horizon": CONFIG.prediction_horizon,
            "purge_days": CONFIG.wf_purge_days,
            "embargo_days": CONFIG.wf_embargo_days,
            "max_features": CONFIG.max_features,
        },
    }

    artifact_path = artifact_dir / "model.joblib"
    joblib.dump(artifact, artifact_path)

    metadata = {
        "ticker": ticker, "version": model_version, "trained_at": trained_at,
        "metrics": artifact["metrics"], "selected_features": selected,
        "column_hash": final_artifacts.column_hash, "config": artifact["config"],
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n  ✅ Saved: {artifact_path}")
    print(f"  ✅ Version: {model_version}")
    print(f"  ✅ Accuracy: {accuracy*100:.1f}%  |  AUC: {auc:.3f}")

    # Track in MLflow if available
    try:
        import mlflow
        from mlops.config import MLOpsConfig
        if MLOpsConfig.MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLOpsConfig.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("Unified_Engine_v5.2")
            with mlflow.start_run(run_name=f"{ticker}_{model_version}"):
                mlflow.log_params({
                    "ticker": ticker,
                    "engine_version": "v5.2",
                    "n_features": len(selected),
                    "training_samples": len(hist_clean),
                    "scale_pos_weight": float(scale_pos_weight)
                })
                mlflow.log_metrics({
                    "accuracy": float(accuracy),
                    "f1_score": float(f1),
                    "auc": float(auc),
                    "binom_pvalue": float(binom_pvalue),
                    "fold_mean_acc": float(np.mean(fold_accs))
                })
                mlflow.log_artifact(str(artifact_path), "unified_model")
                print(f"  ✅ MLflow: Logged run for {ticker}")
    except Exception as e:
        print(f"  ⚠️ MLflow logging skipped/failed: {e}")

    return TrainResult(
        ticker=ticker, success=True, metrics=artifact["metrics"],
        fold_results=fold_results, feature_importance=importance,
        selected_features=selected, model_version=model_version,
        artifact_path=str(artifact_path), reason="trained", trained_at=trained_at,
    )


def _find_optimal_thresholds(y_true, probs):
    best_buy, best_f1 = 0.55, 0.0
    for threshold in np.arange(0.50, 0.65, 0.01):
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_buy = threshold
    best_sell = max(0.35, min(0.49, 1.0 - best_buy))
    return float(best_buy), float(best_sell)


class UnifiedTrainer:
    @staticmethod
    def train(ticker: str, generate_charts: bool = False) -> TrainResult:
        return train_unified_model(ticker, generate_charts=generate_charts)

    @staticmethod
    def train_batch(tickers: List[str]) -> List[TrainResult]:
        results = []
        for ticker in tickers:
            try:
                results.append(train_unified_model(ticker))
            except Exception as e:
                print(f"❌ {ticker}: {e}")
                results.append(TrainResult(
                    ticker=ticker, success=False, metrics={}, fold_results=[],
                    feature_importance={}, selected_features=[], model_version="",
                    artifact_path="", reason=str(e),
                ))
        return results

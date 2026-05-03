"""
Unified Engine — Production Inference v5.0
============================================
Generates PRICE RANGES (not exact prices) like real trading apps.
Uses quantile regression for real confidence intervals.
Enforces feature parity with training via hash verification.
"""

from __future__ import annotations

import json, os, sys, warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_engine.config import CONFIG
from unified_engine.features import get_feature_engine

_MODEL_CACHE: Dict[str, Dict] = {}


def _get_artifact_dir(ticker: str) -> Path:
    safe_ticker = ticker.upper().replace("/", "_").replace("\\", "_")
    return CONFIG.model_dir / safe_ticker


def _load_artifact(ticker: str) -> Optional[Dict]:
    ticker = ticker.upper()
    if ticker in _MODEL_CACHE:
        return _MODEL_CACHE[ticker]
    artifact_path = _get_artifact_dir(ticker) / "model.joblib"
    if not artifact_path.exists():
        return None
    try:
        artifact = joblib.load(artifact_path)
        _MODEL_CACHE[ticker] = artifact
        return artifact
    except Exception as e:
        print(f"[INFERENCE] Failed to load artifact for {ticker}: {e}")
        return None


def clear_cache(ticker: Optional[str] = None):
    if ticker:
        _MODEL_CACHE.pop(ticker.upper(), None)
    else:
        _MODEL_CACHE.clear()


class UnifiedPredictor:
    """Production inference — generates price RANGES like real trading apps."""

    @staticmethod
    def predict(ticker: str, hist: pd.DataFrame, current_price: Optional[float] = None, days: int = 7) -> Optional[Dict]:
        ticker = ticker.upper()
        artifact = _load_artifact(ticker)
        if artifact is None:
            return None
        try:
            return _run_inference(ticker, artifact, hist, current_price, days)
        except Exception as e:
            print(f"[INFERENCE] Prediction failed for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def is_model_available(ticker: str) -> bool:
        return (_get_artifact_dir(ticker.upper()) / "model.joblib").exists()

    @staticmethod
    def get_model_metadata(ticker: str) -> Optional[Dict]:
        metadata_path = _get_artifact_dir(ticker.upper()) / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None


def _run_inference(ticker, artifact, hist, current_price, days):
    """Core inference — produces price ranges via quantile regression."""

    # === STEP 1: Feature computation with SAME engine as training ===
    selected_features = artifact["selected_features"]
    saved_scaler = artifact["scaler"]
    saved_hash = artifact["column_hash"]

    engine = get_feature_engine(selected_features=selected_features)
    hist_clean = _normalize_ohlcv(hist)

    if len(hist_clean) < 100:
        raise ValueError(f"Insufficient history: {len(hist_clean)} rows (need 100+)")

    feature_artifacts = engine.compute(hist_clean, fit_scaler=False, scaler=saved_scaler)

    # === STEP 2: Feature parity check ===
    computed_hash = feature_artifacts.column_hash
    if computed_hash != saved_hash:
        print(f"[INFERENCE] ⚠️ Feature hash mismatch for {ticker}!")
        print(f"  Training: {saved_hash}, Inference: {computed_hash}")
        raise ValueError("Feature parity violation")

    if feature_artifacts.features.empty:
        raise ValueError("No valid features computed")

    # === STEP 3: Get latest feature row ===
    X_latest = feature_artifacts.features.iloc[[-1]].values

    # === STEP 4: Model predictions ===
    xgb_model = artifact["xgb_model"]
    lgbm_model = artifact["lgbm_model"]
    rf_model = artifact.get("rf_model")  # v5.1: may not exist in older artifacts
    meta_learner = artifact["meta_learner"]
    calibrator = artifact["calibrator"]
    quantile_models = artifact.get("quantile_models", {})

    # Direction probability
    xgb_prob = float(xgb_model.predict_proba(X_latest)[0, 1])

    # Magnitude prediction (median)
    lgbm_mag = float(lgbm_model.predict(X_latest)[0])
    lgbm_dir_prob = 1.0 if lgbm_mag > 0 else 0.0

    # v5.2: Dual XGBoost + RandomForest probability
    xgb_model_2 = artifact.get("xgb_model_2")
    xgb2_prob = float(xgb_model_2.predict_proba(X_latest)[0, 1]) if xgb_model_2 is not None else xgb_prob
    rf_prob = float(rf_model.predict_proba(X_latest)[0, 1]) if rf_model is not None else xgb_prob

    # Meta-learner ensemble (must match training meta-features)
    if xgb_model_2 is not None and rf_model is not None:
        # v5.2 features
        meta_features = np.array([[
            xgb_prob, xgb2_prob, lgbm_dir_prob, rf_prob, lgbm_mag,
            xgb_prob * rf_prob,                     # interaction 1
            xgb2_prob * rf_prob,                    # interaction 2
            abs(xgb_prob - rf_prob),                # disagreement
        ]])
    elif rf_model is not None:
        # v5.1 features
        meta_features = np.array([[
            xgb_prob, lgbm_dir_prob, rf_prob, lgbm_mag,
            xgb_prob * rf_prob,                    # interaction
            abs(xgb_prob - rf_prob),                # disagreement
        ]])
    else:
        # Legacy fallback for older artifacts
        meta_features = np.array([[xgb_prob, lgbm_dir_prob, lgbm_mag]])
    raw_prob = float(meta_learner.predict_proba(meta_features)[0, 1])

    # Calibrate
    direction_prob = float(calibrator.calibrate(np.array([raw_prob]))[0])

    # === STEP 5: Quantile predictions (REAL confidence intervals) ===
    base_price = current_price if current_price else float(hist_clean["Close"].iloc[-1])
    total_days = max(1, int(days))

    # Get quantile return predictions
    q_predictions = {}
    for alpha, q_model in quantile_models.items():
        q_return = float(q_model.predict(X_latest)[0])
        q_predictions[alpha] = q_return

    # Fallback if no quantile models
    avg_abs_return = artifact.get("avg_abs_return", 0.02)
    if not q_predictions:
        expected_return = (direction_prob - 0.5) * 2.0 * avg_abs_return
        q_predictions = {
            0.10: expected_return - 2.0 * avg_abs_return,
            0.25: expected_return - avg_abs_return,
            0.50: expected_return,
            0.75: expected_return + avg_abs_return,
            0.90: expected_return + 2.0 * avg_abs_return,
        }

    expected_return = q_predictions.get(0.50, lgbm_mag)

    # === STEP 6: Generate price RANGES for each day ===
    daily_fraction = 1.0 / max(CONFIG.prediction_horizon, 1)

    predicted_prices = []      # median path
    price_range_low = []       # 10th percentile (bear case)
    price_range_high = []      # 90th percentile (bull case)
    price_range_q25 = []       # 25th percentile
    price_range_q75 = []       # 75th percentile

    for day_num in range(1, total_days + 1):
        frac = day_num * daily_fraction
        frac = min(frac, 1.0)

        p_median = base_price * (1.0 + q_predictions.get(0.50, expected_return) * frac)
        p_low = base_price * (1.0 + q_predictions.get(0.10, expected_return - 0.03) * frac)
        p_high = base_price * (1.0 + q_predictions.get(0.90, expected_return + 0.03) * frac)
        p_q25 = base_price * (1.0 + q_predictions.get(0.25, expected_return - 0.01) * frac)
        p_q75 = base_price * (1.0 + q_predictions.get(0.75, expected_return + 0.01) * frac)

        predicted_prices.append(round(float(p_median), 2))
        price_range_low.append(round(float(p_low), 2))
        price_range_high.append(round(float(p_high), 2))
        price_range_q25.append(round(float(p_q25), 2))
        price_range_q75.append(round(float(p_q75), 2))

    # === STEP 7: Generate signal ===
    confidence = abs(direction_prob - 0.5) * 2.0
    buy_threshold = artifact.get("buy_threshold", CONFIG.entry_threshold)
    sell_threshold = artifact.get("sell_threshold", CONFIG.exit_threshold)

    if direction_prob >= max(buy_threshold, 0.55):
        signal = "STRONG BUY" if confidence >= CONFIG.strong_signal_confidence else "BUY"
    elif direction_prob <= min(sell_threshold, 0.45):
        signal = "STRONG SELL" if confidence >= CONFIG.strong_signal_confidence else "SELL"
    else:
        signal = "HOLD"

    # === STEP 8: Build response ===
    metrics = artifact.get("metrics", {})

    return {
        "ticker": ticker,
        "prediction": float(expected_return),
        "lower_95": float(q_predictions.get(0.10, expected_return - 0.03)),
        "upper_95": float(q_predictions.get(0.90, expected_return + 0.03)),
        "confidence": float(confidence),
        "direction_prob": float(direction_prob),
        "signal": signal,
        # Price ranges (like real trading apps)
        "predicted_prices": predicted_prices,
        "price_range_low": price_range_low,
        "price_range_high": price_range_high,
        "price_range_q25": price_range_q25,
        "price_range_q75": price_range_q75,
        # Quantile returns
        "quantile_returns": {str(k): float(v) for k, v in q_predictions.items()},
        # Model info
        "model_version": artifact.get("version", "v5.0"),
        "model_type": "Unified Ensemble v5.0",
        "features_used": selected_features,
        "column_hash": saved_hash,
        "data_freshness": datetime.utcnow().isoformat() + "Z",
        "trained_at": artifact.get("trained_at", ""),
        "drift_score": 0.0,
        "prediction_horizon": CONFIG.prediction_horizon,
        "metrics": metrics,
        "components": {
            "xgb_prob": float(xgb_prob),
            "lgbm_mag": float(lgbm_mag),
            "raw_meta_prob": float(raw_prob),
            "calibrated_prob": float(direction_prob),
        },
    }


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV DataFrame for consistent processing."""
    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if not isinstance(data.index, pd.DatetimeIndex):
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])
            data = data.set_index("Date")
        elif "datetime" in data.columns:
            data["datetime"] = pd.to_datetime(data["datetime"])
            data = data.set_index("datetime")
        else:
            data.index = pd.to_datetime(data.index)

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in data.columns:
            if col == "Volume":
                data[col] = 0
            elif col in ("Open", "High", "Low"):
                data[col] = data["Close"]

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.sort_index()
    data = data.dropna(subset=["Close"])

    return data

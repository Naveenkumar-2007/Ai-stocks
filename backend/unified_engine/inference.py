"""
Unified Engine — Production Inference
=======================================
Single entry point for all predictions.
Enforces feature parity with training via hash verification.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_engine.config import CONFIG
from unified_engine.features import UnifiedFeatureEngine, get_feature_engine


# In-memory cache for loaded models
_MODEL_CACHE: Dict[str, Dict] = {}


def _get_artifact_dir(ticker: str) -> Path:
    safe_ticker = ticker.upper().replace("/", "_").replace("\\", "_")
    return CONFIG.model_dir / safe_ticker


def _load_artifact(ticker: str) -> Optional[Dict]:
    """Load trained model artifact from disk, with in-memory caching."""
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
    """Clear model cache. If ticker is None, clear all."""
    if ticker:
        _MODEL_CACHE.pop(ticker.upper(), None)
    else:
        _MODEL_CACHE.clear()


class UnifiedPredictor:
    """
    Production inference service.
    Uses the SAME feature engine as training to guarantee parity.
    """

    @staticmethod
    def predict(
        ticker: str,
        hist: pd.DataFrame,
        current_price: Optional[float] = None,
        days: int = 7,
    ) -> Optional[Dict]:
        """
        Generate predictions for a stock.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            hist: Historical OHLCV DataFrame (at least 200 rows)
            current_price: Current price (if None, uses last close)
            days: Number of days to predict ahead

        Returns:
            Dict with prediction data, or None if model not available
        """
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
        """Check if a trained model exists for this ticker."""
        artifact_path = _get_artifact_dir(ticker.upper()) / "model.joblib"
        return artifact_path.exists()

    @staticmethod
    def get_model_metadata(ticker: str) -> Optional[Dict]:
        """Get model metadata without loading the full artifact."""
        metadata_path = _get_artifact_dir(ticker.upper()) / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None


def _run_inference(
    ticker: str,
    artifact: Dict,
    hist: pd.DataFrame,
    current_price: Optional[float],
    days: int,
) -> Dict:
    """Core inference logic."""

    # === STEP 1: Feature computation with SAME engine as training ===
    selected_features = artifact["selected_features"]
    saved_scaler = artifact["scaler"]
    saved_hash = artifact["column_hash"]

    engine = get_feature_engine(selected_features=selected_features)

    # Normalize input DataFrame
    hist_clean = _normalize_ohlcv(hist)
    if len(hist_clean) < 200:
        raise ValueError(f"Insufficient history: {len(hist_clean)} rows (need 200+)")

    # Compute features using the SAME engine, with the SAVED scaler
    feature_artifacts = engine.compute(
        hist_clean,
        fit_scaler=False,
        scaler=saved_scaler,
    )

    # === STEP 2: Feature parity check ===
    computed_hash = feature_artifacts.column_hash
    if computed_hash != saved_hash:
        print(f"[INFERENCE] ⚠️ Feature hash mismatch for {ticker}!")
        print(f"  Training hash: {saved_hash}")
        print(f"  Inference hash: {computed_hash}")
        print(f"  Training columns: {artifact['column_order']}")
        print(f"  Inference columns: {feature_artifacts.column_order}")
        raise ValueError("Feature parity violation: train/inference features don't match")

    if feature_artifacts.features.empty:
        raise ValueError("No valid features computed from input data")

    # === STEP 3: Get latest feature row ===
    X_latest = feature_artifacts.features.iloc[[-1]].values

    # === STEP 4: Model predictions ===
    xgb_model = artifact["xgb_model"]
    lgbm_model = artifact["lgbm_model"]
    meta_learner = artifact["meta_learner"]
    calibrator = artifact["calibrator"]

    # XGBoost direction probability
    xgb_prob = float(xgb_model.predict_proba(X_latest)[0, 1])

    # LightGBM magnitude prediction
    lgbm_mag = float(lgbm_model.predict(X_latest)[0])
    lgbm_dir_prob = 1.0 if lgbm_mag > 0 else 0.0

    # Meta-learner ensemble
    meta_features = np.array([[xgb_prob, lgbm_dir_prob, lgbm_mag]])
    raw_prob = float(meta_learner.predict_proba(meta_features)[0, 1])

    # Calibrate
    direction_prob = float(calibrator.calibrate(np.array([raw_prob]))[0])

    # === STEP 5: Generate signals ===
    confidence = abs(direction_prob - 0.5) * 2.0
    buy_threshold = artifact.get("buy_threshold", CONFIG.entry_threshold)
    sell_threshold = artifact.get("sell_threshold", CONFIG.exit_threshold)

    if direction_prob >= max(buy_threshold, 0.55):
        signal = "STRONG BUY" if confidence >= CONFIG.strong_signal_confidence else "BUY"
    elif direction_prob <= min(sell_threshold, 0.45):
        signal = "STRONG SELL" if confidence >= CONFIG.strong_signal_confidence else "SELL"
    else:
        signal = "HOLD"

    # === STEP 6: Price projections ===
    avg_abs_return = artifact.get("avg_abs_return", 0.02)
    expected_return = (direction_prob - 0.5) * 2.0 * avg_abs_return

    base_price = current_price if current_price else float(hist_clean["Close"].iloc[-1])
    total_days = max(1, int(days))
    daily_return = (1.0 + expected_return) ** (1.0 / max(CONFIG.prediction_horizon, total_days)) - 1.0

    predicted_prices = []
    price = base_price
    for _ in range(total_days):
        price *= (1.0 + daily_return)
        predicted_prices.append(round(float(price), 2))

    # === STEP 7: Confidence intervals ===
    uncertainty = max(0.005, avg_abs_return * (1.0 - min(confidence, 0.95)))

    # === STEP 8: Build response ===
    metrics = artifact.get("metrics", {})

    return {
        # Core prediction
        "ticker": ticker,
        "prediction": float(expected_return),
        "lower_95": float(expected_return - 1.96 * uncertainty),
        "upper_95": float(expected_return + 1.96 * uncertainty),
        "confidence": float(confidence),
        "direction_prob": float(direction_prob),
        "signal": signal,
        # Model info
        "model_version": artifact.get("version", "v4.0"),
        "model_type": "Unified Ensemble v4.0",
        "features_used": selected_features,
        "column_hash": saved_hash,
        # Timestamps
        "data_freshness": datetime.utcnow().isoformat() + "Z",
        "trained_at": artifact.get("trained_at", ""),
        # Drift
        "drift_score": 0.0,
        # Prices
        "predicted_prices": predicted_prices,
        "prediction_horizon": CONFIG.prediction_horizon,
        # Metrics from training
        "metrics": metrics,
        # Components (for debugging)
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

    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])
            data = data.set_index("Date")
        elif "datetime" in data.columns:
            data["datetime"] = pd.to_datetime(data["datetime"])
            data = data.set_index("datetime")
        else:
            data.index = pd.to_datetime(data.index)

    # Remove timezone
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Ensure required columns
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in data.columns:
            if col == "Volume":
                data[col] = 0
            elif col in ("Open", "High", "Low"):
                data[col] = data["Close"]

    # Convert to numeric
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.sort_index()
    data = data.dropna(subset=["Close"])

    return data

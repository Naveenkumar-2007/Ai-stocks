"""
Unified Engine — Feature Engineering (SINGLE SOURCE OF TRUTH)
=============================================================
This is the ONLY place features are computed.
Both training AND inference MUST use this module.

Design principles:
1. ALL features are stationary (ratios, returns, oscillators) — no raw prices
2. Column order is FROZEN and deterministic (sorted alphabetically)
3. A feature hash is stored with the model to detect mismatches at inference
4. Feature importance drives selection — unused features are pruned
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# =============================================================================
# CANONICAL FEATURE LIST (sorted alphabetically for deterministic ordering)
# =============================================================================

# These are ALL candidate features. After training, feature importance will
# select a subset. But the computation order never changes.
_ALL_CANDIDATE_FEATURES = sorted([
    # --- Returns ---
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return_1d",
    # --- Return lags (autoregressive) ---
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    # --- Moving average ratios (price relative to MA) ---
    "sma_5_ratio",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    # --- MACD ratios ---
    "macd_ratio",
    "macd_signal_ratio",
    "macd_hist_ratio",
    # --- Oscillators ---
    "rsi_14",
    "stoch_k",
    "stoch_d",
    # --- Volatility ---
    "volatility_10d",
    "volatility_20d",
    "bb_width_ratio",
    "bb_position",
    "atr_ratio",
    # --- Volume ---
    "volume_ratio_5",
    "volume_ratio_20",
    "volume_zscore_20",
    "obv_normalized",
    # --- Price position ---
    "daily_range_position",
    "overnight_gap_pct",
    # --- Momentum ---
    "momentum_roc_5",
    "momentum_roc_10",
    "momentum_roc_20",
    # --- Trend strength ---
    "adx_proxy",
    "dist_to_20d_high_ratio",
    "dist_to_20d_low_ratio",
    # --- Higher moments ---
    "return_skew_20",
    "return_kurt_20",
])


@dataclass
class FeatureArtifacts:
    """Container for feature computation results."""
    features: pd.DataFrame       # computed features (scaled)
    features_raw: pd.DataFrame   # computed features (unscaled, for drift detection)
    scaler: RobustScaler
    column_order: List[str]      # frozen column order
    column_hash: str             # SHA256 of column names for parity check


class UnifiedFeatureEngine:
    """
    Single source of truth for feature engineering.
    Used identically by training AND inference.
    """

    def __init__(self, selected_features: Optional[List[str]] = None):
        """
        Args:
            selected_features: If provided, only compute these features.
                               Used after training has determined feature importance.
                               If None, compute ALL candidate features.
        """
        if selected_features is not None:
            self._feature_list = sorted(selected_features)
        else:
            self._feature_list = _ALL_CANDIDATE_FEATURES

    @property
    def feature_columns(self) -> List[str]:
        return list(self._feature_list)

    @staticmethod
    def compute_column_hash(columns: List[str]) -> str:
        """Deterministic hash of column names for train/inference parity check."""
        return hashlib.sha256("|".join(sorted(columns)).encode()).hexdigest()[:16]

    def compute(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True,
        scaler: Optional[RobustScaler] = None,
    ) -> FeatureArtifacts:
        """
        Compute features from OHLCV DataFrame.

        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex
            fit_scaler: If True, fit a new scaler. If False, use provided scaler.
            scaler: Pre-fitted scaler (required if fit_scaler=False)

        Returns:
            FeatureArtifacts with scaled features, scaler, and column metadata
        """
        feat = self._compute_raw_features(df)

        # Select only the features we want
        available = [c for c in self._feature_list if c in feat.columns]
        feat = feat[available].copy()

        # Drop rows with NaN (from rolling windows)
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

        if feat.empty:
            raise ValueError("No valid feature rows after NaN removal")

        # Winsorize extreme values (1st and 99th percentile)
        for col in feat.columns:
            lower = feat[col].quantile(0.01)
            upper = feat[col].quantile(0.99)
            feat[col] = feat[col].clip(lower=lower, upper=upper)

        # Store raw features (unscaled) for drift detection
        features_raw = feat.copy()

        # Scale
        if fit_scaler:
            scaler = RobustScaler()
            scaled_values = scaler.fit_transform(feat.values)
        else:
            if scaler is None:
                raise ValueError("Must provide scaler when fit_scaler=False")
            scaled_values = scaler.transform(feat.values)

        features_scaled = pd.DataFrame(
            scaled_values,
            index=feat.index,
            columns=feat.columns
        )

        column_order = list(features_scaled.columns)
        column_hash = self.compute_column_hash(column_order)

        return FeatureArtifacts(
            features=features_scaled,
            features_raw=features_raw,
            scaler=scaler,
            column_order=column_order,
            column_hash=column_hash,
        )

    def _compute_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ALL candidate features from OHLCV data."""
        data = df.copy()

        # Ensure numeric
        close = pd.to_numeric(data["Close"], errors="coerce")
        high = pd.to_numeric(data["High"], errors="coerce")
        low = pd.to_numeric(data["Low"], errors="coerce")
        open_price = pd.to_numeric(data.get("Open", close), errors="coerce")
        volume = pd.to_numeric(data.get("Volume", pd.Series(0, index=data.index)), errors="coerce")

        feat = pd.DataFrame(index=data.index)

        # === RETURNS (stationary by construction) ===
        feat["return_1d"] = close.pct_change(1)
        feat["return_2d"] = close.pct_change(2)
        feat["return_5d"] = close.pct_change(5)
        feat["return_10d"] = close.pct_change(10)
        feat["return_20d"] = close.pct_change(20)
        feat["log_return_1d"] = np.log(close / close.shift(1))

        # === RETURN LAGS ===
        feat["return_lag_1"] = feat["return_1d"].shift(1)
        feat["return_lag_2"] = feat["return_1d"].shift(2)
        feat["return_lag_3"] = feat["return_1d"].shift(3)
        feat["return_lag_5"] = feat["return_1d"].shift(5)

        # === MOVING AVERAGE RATIOS (price / MA — always near 1.0, stationary) ===
        for window in [5, 10, 20, 50]:
            sma = close.rolling(window=window, min_periods=window).mean()
            feat[f"sma_{window}_ratio"] = close / sma

        for span in [12, 26]:
            ema = close.ewm(span=span, adjust=False).mean()
            feat[f"ema_{span}_ratio"] = close / ema

        # === MACD RATIOS (normalized by price) ===
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        feat["macd_ratio"] = macd_line / close
        feat["macd_signal_ratio"] = macd_signal / close
        feat["macd_hist_ratio"] = macd_hist / close

        # === RSI ===
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
        rs = gain / (loss + 1e-10)
        feat["rsi_14"] = 100 - (100 / (1 + rs))

        # === STOCHASTIC ===
        lowest_14 = low.rolling(14, min_periods=14).min()
        highest_14 = high.rolling(14, min_periods=14).max()
        feat["stoch_k"] = (close - lowest_14) / (highest_14 - lowest_14 + 1e-10) * 100
        feat["stoch_d"] = feat["stoch_k"].rolling(3, min_periods=3).mean()

        # === VOLATILITY ===
        feat["volatility_10d"] = feat["return_1d"].rolling(10, min_periods=10).std() * np.sqrt(252)
        feat["volatility_20d"] = feat["return_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)

        # === BOLLINGER BANDS ===
        bb_middle = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        feat["bb_width_ratio"] = (bb_upper - bb_lower) / (close + 1e-10)
        feat["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # === ATR RATIO ===
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        feat["atr_ratio"] = atr / (close + 1e-10)

        # === VOLUME ===
        vol_sma_5 = volume.rolling(5, min_periods=5).mean()
        vol_sma_20 = volume.rolling(20, min_periods=5).mean()
        feat["volume_ratio_5"] = volume / (vol_sma_5 + 1e-10)
        feat["volume_ratio_20"] = volume / (vol_sma_20 + 1e-10)

        vol_mean_20 = volume.rolling(20, min_periods=5).mean()
        vol_std_20 = volume.rolling(20, min_periods=5).std()
        feat["volume_zscore_20"] = (volume - vol_mean_20) / (vol_std_20 + 1e-10)

        # OBV normalized
        direction = np.sign(close.diff().fillna(0))
        vol_avg = volume.rolling(20, min_periods=5).mean()
        feat["obv_normalized"] = (volume * direction) / (vol_avg + 1e-10)

        # === PRICE POSITION ===
        feat["daily_range_position"] = (close - low) / (high - low + 1e-10)
        feat["overnight_gap_pct"] = (open_price - close.shift(1)) / (close.shift(1) + 1e-10)

        # === MOMENTUM (Rate of Change) ===
        feat["momentum_roc_5"] = close / close.shift(5) - 1
        feat["momentum_roc_10"] = close / close.shift(10) - 1
        feat["momentum_roc_20"] = close / close.shift(20) - 1

        # === TREND STRENGTH ===
        high_range = high.rolling(10, min_periods=10).max() - low.rolling(10, min_periods=10).min()
        feat["adx_proxy"] = (close - close.shift(10)).abs() / (high_range + 1e-10)

        high_20 = high.rolling(20, min_periods=20).max()
        low_20 = low.rolling(20, min_periods=20).min()
        feat["dist_to_20d_high_ratio"] = (high_20 - close) / (close + 1e-10)
        feat["dist_to_20d_low_ratio"] = (close - low_20) / (close + 1e-10)

        # === HIGHER MOMENTS ===
        feat["return_skew_20"] = feat["return_1d"].rolling(20, min_periods=20).skew()
        feat["return_kurt_20"] = feat["return_1d"].rolling(20, min_periods=20).kurt()

        return feat


def get_feature_engine(selected_features: Optional[List[str]] = None) -> UnifiedFeatureEngine:
    """Factory function for feature engine."""
    return UnifiedFeatureEngine(selected_features=selected_features)

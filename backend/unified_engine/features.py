"""
Unified Engine — Feature Engineering (SINGLE SOURCE OF TRUTH)  v5.0
=====================================================================
This is the ONLY place features are computed.
Both training AND inference MUST use this module.

Design principles:
1. ALL features are stationary (ratios, returns, oscillators) — no raw prices
2. Column order is FROZEN and deterministic (sorted alphabetically)
3. A feature hash is stored with the model to detect mismatches at inference
4. Feature importance drives selection — unused features are pruned
5. Includes mean-reversion, momentum-acceleration, and calendar signals
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# =============================================================================
# CANONICAL FEATURE LIST  (sorted alphabetically for deterministic ordering)
# =============================================================================

_ALL_CANDIDATE_FEATURES = sorted([
    # --- Returns ---
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return_1d",
    # --- Return lags (autoregressive memory) ---
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
    "rsi_7",                       # v5.1: faster RSI for short-term signals
    "stoch_k",
    "stoch_d",
    "williams_r",                  # v5.1: Williams %R oscillator
    # --- Volatility ---
    "volatility_10d",
    "volatility_20d",
    "bb_width_ratio",
    "bb_position",
    "atr_ratio",
    "volatility_ratio",
    "parkinson_vol",               # v5.1: intraday range-based volatility
    # --- Volume ---
    "volume_ratio_5",
    "volume_ratio_20",
    "volume_zscore_20",
    "obv_normalized",
    "volume_breakout",
    "volume_price_confirm",        # v5.1: do price and volume agree?
    "force_index_norm",            # v5.1: force index (price × volume)
    "chaikin_mf_proxy",            # v5.1: money flow proxy
    # --- Price position ---
    "daily_range_position",
    "overnight_gap_pct",
    # --- Candle microstructure (v5.1) ---
    "candle_body_ratio",           # v5.1: |close-open| / range
    "upper_shadow_ratio",          # v5.1: rejection wicks
    "lower_shadow_ratio",          # v5.1: absorption wicks
    # --- Momentum ---
    "momentum_roc_5",
    "momentum_roc_10",
    "momentum_roc_20",
    "price_acceleration",
    "momentum_consistency",        # v5.1: % of last N days that were positive
    # --- Trend strength ---
    "adx_proxy",
    "dist_to_20d_high_ratio",
    "dist_to_20d_low_ratio",
    "trend_strength_5_20",         # v5.1: SMA5 vs SMA20 cross-over signal
    "trend_strength_10_50",        # v5.1: SMA10 vs SMA50
    # --- Higher moments ---
    "return_skew_20",
    "return_kurt_20",
    # --- Serial correlation (v5.1) ---
    "return_autocorr_5",           # v5.1: return predictability
    "return_autocorr_20",          # v5.1: longer-term serial dependence
    # --- Mean-reversion signals ---
    "zscore_20d",
    "zscore_50d",
    "mean_reversion_5d",
    # --- Calendar / seasonality ---
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    # --- Realized-vs-Implied volatility proxy ---
    "vol_regime",
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

        # NEW: Volatility ratio (short / long — detects vol regime changes)
        vol_5 = feat["return_1d"].rolling(5, min_periods=5).std()
        vol_20 = feat["return_1d"].rolling(20, min_periods=20).std()
        feat["volatility_ratio"] = vol_5 / (vol_20 + 1e-10)

        # NEW: Vol regime (is current vol above or below median?)
        vol_60 = feat["return_1d"].rolling(60, min_periods=30).std()
        vol_median = vol_60.rolling(252, min_periods=60).median()
        feat["vol_regime"] = (vol_60 - vol_median) / (vol_median + 1e-10)

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

        # Volume breakout (binary: volume > 2σ above 20d mean)
        feat["volume_breakout"] = (feat["volume_zscore_20"] > 2.0).astype(float)

        # v5.1: Volume-price confirmation (do price and volume agree?)
        price_up = (close.diff() > 0).astype(float)
        volume_up = (volume.diff() > 0).astype(float)
        feat["volume_price_confirm"] = (price_up == volume_up).astype(float).rolling(5, min_periods=3).mean()

        # v5.1: Force Index normalized (close change × volume, relative to avg)
        force_raw = close.diff() * volume
        force_avg = force_raw.rolling(13, min_periods=5).mean()
        force_norm = force_raw.rolling(20, min_periods=5).std()
        feat["force_index_norm"] = force_avg / (force_norm + 1e-10)

        # v5.1: Chaikin Money Flow proxy (accumulation/distribution)
        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
        mf_volume = mf_multiplier * volume
        feat["chaikin_mf_proxy"] = mf_volume.rolling(20, min_periods=10).sum() / (volume.rolling(20, min_periods=10).sum() + 1e-10)

        # === PRICE POSITION ===
        feat["daily_range_position"] = (close - low) / (high - low + 1e-10)
        feat["overnight_gap_pct"] = (open_price - close.shift(1)) / (close.shift(1) + 1e-10)

        # === CANDLE MICROSTRUCTURE (v5.1) ===
        body = (close - open_price).abs()
        full_range = high - low + 1e-10
        feat["candle_body_ratio"] = body / full_range
        feat["upper_shadow_ratio"] = (high - pd.concat([close, open_price], axis=1).max(axis=1)) / full_range
        feat["lower_shadow_ratio"] = (pd.concat([close, open_price], axis=1).min(axis=1) - low) / full_range

        # === MOMENTUM (Rate of Change) ===
        feat["momentum_roc_5"] = close / close.shift(5) - 1
        feat["momentum_roc_10"] = close / close.shift(10) - 1
        feat["momentum_roc_20"] = close / close.shift(20) - 1

        # Price acceleration (2nd derivative — rate of trend change)
        roc_5 = close / close.shift(5) - 1
        roc_5_prev = roc_5.shift(5)
        feat["price_acceleration"] = roc_5 - roc_5_prev

        # v5.1: Momentum consistency (% of last 10 days that were positive)
        feat["momentum_consistency"] = price_up.rolling(10, min_periods=5).mean()

        # === TREND STRENGTH ===
        high_range = high.rolling(10, min_periods=10).max() - low.rolling(10, min_periods=10).min()
        feat["adx_proxy"] = (close - close.shift(10)).abs() / (high_range + 1e-10)

        high_20 = high.rolling(20, min_periods=20).max()
        low_20 = low.rolling(20, min_periods=20).min()
        feat["dist_to_20d_high_ratio"] = (high_20 - close) / (close + 1e-10)
        feat["dist_to_20d_low_ratio"] = (close - low_20) / (close + 1e-10)

        # v5.1: Cross-MA trend strength (SMA alignment)
        sma_5 = close.rolling(5, min_periods=5).mean()
        sma_20_trend = close.rolling(20, min_periods=20).mean()
        sma_10 = close.rolling(10, min_periods=10).mean()
        sma_50_trend = close.rolling(50, min_periods=50).mean()
        feat["trend_strength_5_20"] = (sma_5 - sma_20_trend) / (close + 1e-10)
        feat["trend_strength_10_50"] = (sma_10 - sma_50_trend) / (close + 1e-10)

        # === HIGHER MOMENTS ===
        feat["return_skew_20"] = feat["return_1d"].rolling(20, min_periods=20).skew()
        feat["return_kurt_20"] = feat["return_1d"].rolling(20, min_periods=20).kurt()

        # === SERIAL CORRELATION (v5.1) ===
        feat["return_autocorr_5"] = feat["return_1d"].rolling(20, min_periods=10).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 5 else 0, raw=False
        )
        feat["return_autocorr_20"] = feat["return_1d"].rolling(60, min_periods=20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 10 else 0, raw=False
        )

        # === MEAN-REVERSION SIGNALS ===
        sma_20 = close.rolling(20, min_periods=20).mean()
        std_20 = close.rolling(20, min_periods=20).std()
        feat["zscore_20d"] = (close - sma_20) / (std_20 + 1e-10)

        sma_50 = close.rolling(50, min_periods=50).mean()
        std_50 = close.rolling(50, min_periods=50).std()
        feat["zscore_50d"] = (close - sma_50) / (std_50 + 1e-10)

        # Mean-reversion tendency: negative correlation between past and future returns
        feat["mean_reversion_5d"] = -feat["return_5d"]  # sign-flip of recent return

        # === WILLIAMS %R (v5.1) ===
        highest_14_wr = high.rolling(14, min_periods=14).max()
        lowest_14_wr = low.rolling(14, min_periods=14).min()
        feat["williams_r"] = (highest_14_wr - close) / (highest_14_wr - lowest_14_wr + 1e-10) * -100

        # === RSI-7 (v5.1: faster RSI) ===
        delta7 = close.diff()
        gain7 = delta7.where(delta7 > 0, 0.0).rolling(7, min_periods=7).mean()
        loss7 = (-delta7.where(delta7 < 0, 0.0)).rolling(7, min_periods=7).mean()
        rs7 = gain7 / (loss7 + 1e-10)
        feat["rsi_7"] = 100 - (100 / (1 + rs7))

        # === PARKINSON VOLATILITY (v5.1: uses high-low range, more efficient) ===
        log_hl = np.log(high / (low + 1e-10))
        feat["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) * (log_hl ** 2).rolling(20, min_periods=10).mean()
        ) * np.sqrt(252)

        # === CALENDAR / SEASONALITY ===
        if isinstance(data.index, pd.DatetimeIndex):
            dow = data.index.dayofweek  # 0=Monday, 4=Friday
            month = data.index.month
            feat["day_of_week_sin"] = np.sin(2 * np.pi * dow / 5)
            feat["day_of_week_cos"] = np.cos(2 * np.pi * dow / 5)
            feat["month_sin"] = np.sin(2 * np.pi * month / 12)
            feat["month_cos"] = np.cos(2 * np.pi * month / 12)
        else:
            feat["day_of_week_sin"] = 0.0
            feat["day_of_week_cos"] = 1.0
            feat["month_sin"] = 0.0
            feat["month_cos"] = 1.0

        return feat


def get_feature_engine(selected_features: Optional[List[str]] = None) -> UnifiedFeatureEngine:
    """Factory function for feature engine."""
    return UnifiedFeatureEngine(selected_features=selected_features)

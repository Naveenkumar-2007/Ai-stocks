from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import RobustScaler

FEATURE_COLUMNS = [
    "rsi_14",
    "macd_diff",
    "bb_width_price",
    "volume_zscore_20",
    "return_5d",
    "atr_14",
    "stoch_k",
    "overnight_gap_pct",
    "return_1d",
    "volume_sma_ratio",
]


@dataclass
class FeatureArtifacts:
    features: pd.DataFrame
    scaler: RobustScaler


class FeatureEngineer:
    """Single source of truth for training/inference features (Feast-compatible schema)."""

    def compute_features(self, df: pd.DataFrame) -> FeatureArtifacts:
        work = df.copy().sort_index()

        close = pd.to_numeric(work["Close"], errors="coerce")
        high = pd.to_numeric(work["High"], errors="coerce")
        low = pd.to_numeric(work["Low"], errors="coerce")
        volume = pd.to_numeric(work["Volume"], errors="coerce")
        open_price = pd.to_numeric(work.get("Open", work["Close"]), errors="coerce")

        feat = pd.DataFrame(index=work.index)
        feat["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        macd = ta.trend.MACD(close=close)
        feat["macd_diff"] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(close=close, window=20)
        feat["bb_width_price"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close.replace(0, np.nan)

        vol_mean = volume.rolling(20, min_periods=5).mean()
        vol_std = volume.rolling(20, min_periods=5).std().replace(0, np.nan)
        feat["volume_zscore_20"] = (volume - vol_mean) / vol_std

        feat["return_5d"] = close.pct_change(5)

        feat["atr_14"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

        feat["stoch_k"] = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14).stoch()

        feat["overnight_gap_pct"] = (open_price - close.shift(1)) / close.shift(1).replace(0, np.nan)

        feat["return_1d"] = close.pct_change(1)

        feat["volume_sma_ratio"] = volume / volume.rolling(20, min_periods=5).mean().replace(0, np.nan)

        # Winsorize 1st and 99th percentiles per feature.
        for col in FEATURE_COLUMNS:
            lower = feat[col].quantile(0.01)
            upper = feat[col].quantile(0.99)
            feat[col] = feat[col].clip(lower=lower, upper=upper)

        feat = feat.dropna().copy()
        scaler = RobustScaler()
        if not feat.empty:
            feat[FEATURE_COLUMNS] = scaler.fit_transform(feat[FEATURE_COLUMNS])

        return FeatureArtifacts(features=feat[FEATURE_COLUMNS], scaler=scaler)

    def build_targets(self, df: pd.DataFrame, features: pd.DataFrame, horizon_days: int = 5) -> Tuple[pd.Series, pd.Series]:
        close = pd.to_numeric(df["Close"], errors="coerce")
        aligned_close = close.reindex(features.index)

        future_ret_1d = aligned_close.shift(-1) / aligned_close - 1.0
        direction = (future_ret_1d > 0).astype(int)

        future_ret_5d = aligned_close.shift(-horizon_days) / aligned_close - 1.0

        valid = direction.notna() & future_ret_5d.notna()
        return direction[valid], future_ret_5d[valid]

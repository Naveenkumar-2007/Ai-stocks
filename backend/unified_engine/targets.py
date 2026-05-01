"""
Unified Engine — Target Construction
=====================================
Consistent target labels for training.
Both direction (classification) and magnitude (regression) targets
use the SAME prediction horizon everywhere.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from unified_engine.config import CONFIG


def build_targets(
    df: pd.DataFrame,
    features_index: pd.Index,
    horizon: int = CONFIG.prediction_horizon,
) -> Tuple[pd.Series, pd.Series]:
    """
    Build consistent direction and magnitude targets.

    IMPORTANT: Both targets use the SAME horizon (5 days by default).
    This fixes the bug where XGBoost was trained on 1-day direction
    but LSTM on 5-day magnitude.

    Args:
        df: Raw OHLCV DataFrame with DatetimeIndex
        features_index: Index of the feature DataFrame (for alignment)
        horizon: Number of trading days to look ahead

    Returns:
        direction: Series of 0/1 (did price go up in `horizon` days?)
        magnitude: Series of float (% return over `horizon` days)
    """
    close = pd.to_numeric(df["Close"], errors="coerce")

    # Align close prices to feature index
    aligned_close = close.reindex(features_index)

    # Forward returns over the prediction horizon
    future_close = aligned_close.shift(-horizon)
    forward_return = (future_close - aligned_close) / (aligned_close + 1e-10)

    # Direction: binary classification target
    direction = (forward_return > 0).astype(int)

    # Magnitude: regression target (raw % return)
    magnitude = forward_return

    # Only keep rows where we have valid future data
    valid_mask = forward_return.notna()
    direction = direction[valid_mask]
    magnitude = magnitude[valid_mask]

    return direction, magnitude


def build_target_simple(
    close_prices: pd.Series,
    horizon: int = CONFIG.prediction_horizon,
) -> pd.Series:
    """
    Simple binary target for walk-forward splits.
    Returns 1 if price went up over horizon, else 0.
    """
    future_return = close_prices.shift(-horizon) / close_prices - 1
    target = (future_return > 0).astype(float)
    target[future_return.isna()] = np.nan
    return target

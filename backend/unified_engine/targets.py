"""
Unified Engine — Target Construction  v5.0
=============================================
Volatility-adjusted target labels for training.

Key improvement: instead of labeling every +0.001% as "UP" (noise),
we only label moves that exceed a volatility-adjusted threshold.
This gives the model a clearer signal to learn from.
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
    Build volatility-adjusted direction and magnitude targets.

    The direction target uses a volatility-adjusted dead-zone:
    - forward_return > +threshold  →  1 (UP)
    - forward_return < -threshold  →  0 (DOWN)
    - |forward_return| <= threshold →  dropped from training (ambiguous)

    This dramatically improves signal quality by removing noisy near-zero moves.

    Args:
        df: Raw OHLCV DataFrame with DatetimeIndex
        features_index: Index of the feature DataFrame (for alignment)
        horizon: Number of trading days to look ahead

    Returns:
        direction: Series of 0/1 (filtered — no ambiguous moves)
        magnitude: Series of float (% return over `horizon` days)
    """
    close = pd.to_numeric(df["Close"], errors="coerce")

    # Align close prices to feature index
    aligned_close = close.reindex(features_index)

    # Forward returns over the prediction horizon
    future_close = aligned_close.shift(-horizon)
    forward_return = (future_close - aligned_close) / (aligned_close + 1e-10)

    # Only keep rows where we have valid future data
    valid_mask = forward_return.notna()
    forward_return = forward_return[valid_mask]

    # Magnitude target: raw % return (always available)
    magnitude = forward_return.copy()

    if CONFIG.target_use_threshold:
        # --- Volatility-adjusted threshold ---
        # Compute rolling daily volatility, then scale by sqrt(horizon)
        daily_returns = aligned_close.pct_change()
        rolling_vol = daily_returns.rolling(20, min_periods=10).std()
        # Scale to horizon-level volatility
        horizon_vol = rolling_vol * np.sqrt(horizon)
        # Align to forward_return index
        horizon_vol = horizon_vol.reindex(forward_return.index)

        # Threshold: must exceed this to be labeled
        threshold = horizon_vol * CONFIG.target_threshold_vol_multiplier
        # Floor at 0.5% to avoid near-zero thresholds for low-vol stocks
        threshold = threshold.clip(lower=0.005)

        # Label: UP if exceeds +threshold, DOWN if below -threshold
        direction = pd.Series(np.nan, index=forward_return.index)
        direction[forward_return > threshold] = 1.0
        direction[forward_return < -threshold] = 0.0
        # Rows with |return| <= threshold are NaN → will be dropped from training

        # Drop ambiguous rows
        clean_mask = direction.notna()
        direction = direction[clean_mask].astype(int)
        magnitude = magnitude[clean_mask]

        pct_kept = len(direction) / max(len(forward_return), 1) * 100
        pct_up = direction.mean() * 100 if len(direction) > 0 else 50
        print(f"  [TARGET] Threshold filtering: kept {pct_kept:.0f}% of samples "
              f"(dropped {100-pct_kept:.0f}% ambiguous moves), "
              f"balance: {pct_up:.1f}% UP")
    else:
        # Fallback: simple binary (not recommended)
        direction = (forward_return > 0).astype(int)

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

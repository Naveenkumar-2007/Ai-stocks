"""
Unified Engine - Drift Monitor
==============================
Fast statistical drift checks for live market data.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < bins * 2 or len(actual) < bins * 2:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    expected_pct = np.maximum(expected_counts / max(1, expected_counts.sum()), 1e-6)
    actual_pct = np.maximum(actual_counts / max(1, actual_counts.sum()), 1e-6)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def assess_market_drift(history: pd.DataFrame, *, baseline_window: int = 180, recent_window: int = 30) -> Dict:
    """Compare recent return/volume behavior to the prior baseline."""
    if history is None or history.empty or len(history) < baseline_window + recent_window:
        return {
            "status": "warming_up",
            "score": 0.0,
            "reason": "Not enough history for drift comparison.",
        }

    close = pd.to_numeric(history["Close"], errors="coerce")
    volume = pd.to_numeric(history.get("Volume"), errors="coerce").replace(0, np.nan)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    log_volume = np.log(volume).replace([np.inf, -np.inf], np.nan)

    baseline_returns = returns.iloc[-baseline_window - recent_window:-recent_window].dropna().to_numpy()
    recent_returns = returns.iloc[-recent_window:].dropna().to_numpy()
    baseline_volume = log_volume.iloc[-baseline_window - recent_window:-recent_window].dropna().to_numpy()
    recent_volume = log_volume.iloc[-recent_window:].dropna().to_numpy()

    return_psi = _psi(baseline_returns, recent_returns)
    volume_psi = _psi(baseline_volume, recent_volume)
    baseline_vol = float(np.nanstd(baseline_returns)) if len(baseline_returns) else 0.0
    recent_vol = float(np.nanstd(recent_returns)) if len(recent_returns) else 0.0
    vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1.0

    score = max(return_psi, volume_psi, abs(vol_ratio - 1.0) * 0.25)
    if score >= 0.35:
        status = "high_drift"
    elif score >= 0.18:
        status = "watch"
    else:
        status = "stable"

    return {
        "status": status,
        "score": round(float(score), 4),
        "return_psi": round(float(return_psi), 4),
        "volume_psi": round(float(volume_psi), 4),
        "volatility_ratio": round(float(vol_ratio), 4),
        "recent_window": recent_window,
        "baseline_window": baseline_window,
    }

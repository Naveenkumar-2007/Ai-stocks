"""
Unified Engine - Data Quality Checks
====================================
Validates OHLCV input before forecasts are trusted.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def validate_ohlcv(history: pd.DataFrame, *, min_rows: int = 120) -> Dict:
    """Return user-safe quality status for a market history frame."""
    issues: List[str] = []
    warnings: List[str] = []

    if history is None or history.empty:
        return {
            "status": "failed",
            "score": 0,
            "rows": 0,
            "issues": ["No market history returned by the data provider."],
            "warnings": [],
        }

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in history.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")

    row_count = int(len(history))
    if row_count < min_rows:
        issues.append(f"Only {row_count} rows available; at least {min_rows} are preferred.")

    close = pd.to_numeric(history.get("Close"), errors="coerce")
    high = pd.to_numeric(history.get("High"), errors="coerce")
    low = pd.to_numeric(history.get("Low"), errors="coerce")
    volume = pd.to_numeric(history.get("Volume"), errors="coerce")

    null_close_rate = float(close.isna().mean()) if len(close) else 1.0
    if null_close_rate > 0.05:
        issues.append(f"Close price has {null_close_rate:.1%} missing values.")

    invalid_price_rate = float(((close <= 0) | (high <= 0) | (low <= 0)).mean()) if len(close) else 1.0
    if invalid_price_rate > 0.02:
        issues.append(f"{invalid_price_rate:.1%} of rows contain invalid prices.")

    broken_ohlc_rate = float(((high < low) | (high < close) | (low > close)).mean()) if len(close) else 0.0
    if broken_ohlc_rate > 0.02:
        issues.append(f"{broken_ohlc_rate:.1%} of rows fail OHLC consistency checks.")

    zero_volume_rate = float((volume.fillna(0) <= 0).mean()) if len(volume) else 1.0
    if zero_volume_rate > 0.20:
        warnings.append(f"{zero_volume_rate:.1%} of rows have zero or missing volume.")

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    extreme_return_rate = float((returns.abs() > 0.35).mean()) if len(returns) else 0.0
    if extreme_return_rate > 0.03:
        warnings.append(f"{extreme_return_rate:.1%} of returns are extreme; verify splits or data spikes.")

    penalty = (len(issues) * 30) + (len(warnings) * 10)
    score = max(0, min(100, 100 - penalty))
    status = "passed" if not issues else "warning" if score >= 50 else "failed"

    return {
        "status": status,
        "score": int(score),
        "rows": row_count,
        "issues": issues,
        "warnings": warnings,
        "null_close_rate": round(null_close_rate, 4),
        "zero_volume_rate": round(zero_volume_rate, 4),
        "extreme_return_rate": round(extreme_return_rate, 4),
    }

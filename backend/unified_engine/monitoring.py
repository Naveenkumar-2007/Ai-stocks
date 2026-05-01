"""
Unified Engine — Prediction Monitoring
========================================
Tracks predictions vs actuals to detect model degradation.
This is the feedback loop that was completely missing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from unified_engine.config import CONFIG


_PREDICTION_LOG_DIR = CONFIG.model_dir / "_prediction_logs"
_PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_prediction(
    ticker: str,
    direction_prob: float,
    signal: str,
    predicted_return: float,
    current_price: float,
    model_version: str,
) -> str:
    """
    Log a prediction for future comparison against actuals.

    Returns:
        prediction_id: Unique ID for this prediction
    """
    ticker = ticker.upper()
    timestamp = datetime.utcnow()
    prediction_id = f"{ticker}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    log_entry = {
        "prediction_id": prediction_id,
        "ticker": ticker,
        "timestamp": timestamp.isoformat() + "Z",
        "direction_prob": float(direction_prob),
        "signal": signal,
        "predicted_return": float(predicted_return),
        "current_price": float(current_price),
        "model_version": model_version,
        "actual_price": None,       # filled in later
        "actual_return": None,       # filled in later
        "correct": None,             # filled in later
        "evaluated_at": None,        # filled in later
    }

    log_path = _PREDICTION_LOG_DIR / f"{ticker}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return prediction_id


def evaluate_predictions(
    ticker: str,
    current_prices: Dict[str, float],
) -> Dict:
    """
    Evaluate past predictions against actual prices.
    Call this periodically (e.g., daily) to track model accuracy.

    Args:
        ticker: Stock symbol
        current_prices: Dict of {date_str: price} for lookback

    Returns:
        Summary of evaluated predictions
    """
    ticker = ticker.upper()
    log_path = _PREDICTION_LOG_DIR / f"{ticker}.jsonl"

    if not log_path.exists():
        return {"evaluated": 0, "correct": 0, "accuracy": None}

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    evaluated = 0
    correct = 0
    updated_entries = []

    for entry in entries:
        if entry.get("actual_return") is not None:
            # Already evaluated
            evaluated += 1
            if entry.get("correct"):
                correct += 1
            updated_entries.append(entry)
            continue

        # Check if enough time has passed (5 trading days ≈ 7 calendar days)
        pred_time = datetime.fromisoformat(entry["timestamp"].replace("Z", ""))
        days_elapsed = (datetime.utcnow() - pred_time).days

        if days_elapsed < 7:
            updated_entries.append(entry)
            continue

        # Look up actual price
        pred_price = entry["current_price"]
        # Use the most recent available price as "actual"
        if current_prices:
            actual_price = list(current_prices.values())[-1]
            actual_return = (actual_price - pred_price) / pred_price

            predicted_up = entry["direction_prob"] > 0.5
            actual_up = actual_return > 0

            entry["actual_price"] = actual_price
            entry["actual_return"] = float(actual_return)
            entry["correct"] = bool(predicted_up == actual_up)
            entry["evaluated_at"] = datetime.utcnow().isoformat() + "Z"

            evaluated += 1
            if entry["correct"]:
                correct += 1

        updated_entries.append(entry)

    # Write back updated entries
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")

    accuracy = correct / evaluated if evaluated > 0 else None

    return {
        "ticker": ticker,
        "total_predictions": len(entries),
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_pct": f"{accuracy*100:.1f}%" if accuracy else "N/A",
    }


def get_prediction_history(ticker: str, limit: int = 50) -> List[Dict]:
    """Get recent prediction history for a ticker."""
    ticker = ticker.upper()
    log_path = _PREDICTION_LOG_DIR / f"{ticker}.jsonl"

    if not log_path.exists():
        return []

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return entries[-limit:]


def get_model_health(ticker: str) -> Dict:
    """
    Get overall model health for a ticker.
    Returns recent accuracy, drift score, and retraining recommendation.
    """
    history = get_prediction_history(ticker, limit=20)

    if not history:
        return {
            "ticker": ticker,
            "status": "no_data",
            "recent_accuracy": None,
            "needs_retraining": True,
            "reason": "No prediction history",
        }

    evaluated = [h for h in history if h.get("actual_return") is not None]
    if len(evaluated) < 5:
        return {
            "ticker": ticker,
            "status": "insufficient_evaluations",
            "recent_accuracy": None,
            "needs_retraining": False,
            "reason": f"Only {len(evaluated)} evaluated predictions",
        }

    recent_correct = sum(1 for h in evaluated if h.get("correct"))
    recent_accuracy = recent_correct / len(evaluated)

    needs_retraining = recent_accuracy < 0.50  # Below random baseline

    return {
        "ticker": ticker,
        "status": "healthy" if not needs_retraining else "degraded",
        "recent_accuracy": float(recent_accuracy),
        "recent_accuracy_pct": f"{recent_accuracy*100:.1f}%",
        "evaluated_count": len(evaluated),
        "needs_retraining": needs_retraining,
        "reason": "Below random baseline" if needs_retraining else "Within acceptable range",
    }

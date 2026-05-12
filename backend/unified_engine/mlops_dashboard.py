"""
Unified Engine - MLOps Dashboard Assembly
=========================================
Combines registry, events, prediction logs, and drift into one API payload.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from unified_engine.event_bus import event_counts, list_events
from unified_engine.model_registry import list_model_records
from unified_engine.prediction_store import list_served_predictions, prediction_summary


def build_mlops_dashboard(ticker: Optional[str] = None) -> Dict[str, Any]:
    ticker_key = ticker.upper() if ticker else None
    records = list_model_records()
    if ticker_key:
        records = [record for record in records if record.get("ticker") == ticker_key]

    latest_by_ticker: Dict[str, Dict[str, Any]] = {}
    for record in records:
        symbol = record.get("ticker")
        if symbol and symbol not in latest_by_ticker:
            latest_by_ticker[symbol] = record

    predictions = list_served_predictions(ticker=ticker_key, limit=50)
    events = list_events(ticker=ticker_key, limit=100)

    return {
        "success": True,
        "scope": ticker_key or "all",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "models": {
            "count": len(latest_by_ticker),
            "latest": list(latest_by_ticker.values())[:50],
        },
        "inference": prediction_summary(limit=1000) if not ticker_key else {
            "recent_count": len(predictions),
            "recent": predictions[-10:],
        },
        "events": {
            "counts": event_counts(limit=1000) if not ticker_key else {},
            "recent": events[-25:],
        },
    }

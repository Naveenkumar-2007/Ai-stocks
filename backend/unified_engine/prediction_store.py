"""
Unified Engine - Prediction Store
=================================
Tracks every served forecast as an inference audit record.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from unified_engine.config import CONFIG


_STORE_DIR = CONFIG.base_dir / "monitoring" / "predictions"
_STORE_DIR.mkdir(parents=True, exist_ok=True)
_STORE_FILE = _STORE_DIR / "served_predictions.jsonl"
_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def record_served_prediction(
    *,
    ticker: str,
    horizon_days: int,
    current_price: float,
    signal: str,
    trust_score: Optional[float],
    risk_label: str,
    model_version: str,
    prediction_ready: bool,
    currency: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = {
        "id": str(uuid.uuid4()),
        "ticker": ticker.upper(),
        "horizon_days": int(horizon_days),
        "current_price": float(current_price),
        "signal": signal,
        "trust_score": trust_score,
        "risk_label": risk_label,
        "model_version": model_version,
        "prediction_ready": bool(prediction_ready),
        "currency": currency,
        "extra": extra or {},
        "created_at": _utc_now(),
    }
    with _LOCK:
        with open(_STORE_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    return record


def list_served_predictions(*, ticker: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    if not _STORE_FILE.exists():
        return []

    ticker_key = ticker.upper() if ticker else None
    rows: List[Dict[str, Any]] = []
    with open(_STORE_FILE, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ticker_key and record.get("ticker") != ticker_key:
                continue
            rows.append(record)
    return rows[-max(1, min(int(limit or 100), 1000)):]


def prediction_summary(limit: int = 1000) -> Dict[str, Any]:
    rows = list_served_predictions(limit=limit)
    by_signal: Dict[str, int] = {}
    by_ticker: Dict[str, int] = {}
    ready_count = 0
    for row in rows:
        by_signal[row.get("signal", "UNKNOWN")] = by_signal.get(row.get("signal", "UNKNOWN"), 0) + 1
        by_ticker[row["ticker"]] = by_ticker.get(row["ticker"], 0) + 1
        if row.get("prediction_ready"):
            ready_count += 1
    return {
        "total": len(rows),
        "ready_rate": round(ready_count / len(rows), 4) if rows else None,
        "by_signal": by_signal,
        "top_tickers": sorted(by_ticker.items(), key=lambda item: item[1], reverse=True)[:10],
    }

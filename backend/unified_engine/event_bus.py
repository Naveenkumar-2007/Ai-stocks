"""
Unified Engine - Lightweight Event Bus
======================================
Kafka-style append-only events without an external broker.

This is intentionally file-backed so it works on Hugging Face Spaces and local
Windows development. It gives the project production-like observability without
requiring Redis, Kafka, or Kubernetes.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from unified_engine.config import CONFIG


_EVENT_DIR = CONFIG.base_dir / "monitoring" / "events"
_EVENT_DIR.mkdir(parents=True, exist_ok=True)
_EVENT_LOG = _EVENT_DIR / "mlops_events.jsonl"
_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def publish_event(
    event_type: str,
    *,
    ticker: Optional[str] = None,
    source: str = "backend",
    payload: Optional[Dict[str, Any]] = None,
    severity: str = "info",
) -> Dict[str, Any]:
    """Append one structured MLOps event and return it."""
    event = {
        "id": str(uuid.uuid4()),
        "event_type": event_type,
        "ticker": ticker.upper() if ticker else None,
        "source": source,
        "severity": severity,
        "payload": payload or {},
        "created_at": _utc_now(),
    }
    with _LOCK:
        with open(_EVENT_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")
    return event


def list_events(
    *,
    ticker: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Return recent events, newest last."""
    if not _EVENT_LOG.exists():
        return []

    ticker_key = ticker.upper() if ticker else None
    rows: List[Dict[str, Any]] = []
    with open(_EVENT_LOG, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ticker_key and event.get("ticker") != ticker_key:
                continue
            if event_type and event.get("event_type") != event_type:
                continue
            rows.append(event)
    return rows[-max(1, min(int(limit or 100), 1000)):]


def event_counts(limit: int = 1000) -> Dict[str, int]:
    """Aggregate recent event counts by type."""
    counts: Dict[str, int] = {}
    for event in list_events(limit=limit):
        key = event.get("event_type") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts

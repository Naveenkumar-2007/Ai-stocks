"""
Unified Engine model registry and promotion gates.

This is intentionally lightweight and local. MLflow/DagsHub can still receive
runs, but the application needs a fast registry it can trust at inference time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from unified_engine.config import CONFIG


REGISTRY_PATH = CONFIG.base_dir / "mlops" / "unified_model_registry.json"


PROMOTION_THRESHOLDS = {
    "balanced_accuracy": 53.0,
    "macro_f1": 45.0,
    "auc": 0.52,
    "max_pvalue": 0.05,
    "max_fold_std": 18.0,
    "min_calibration_samples": 50,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if value != value:
            return default
        return value
    except Exception:
        return default


def _read_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {
            "registry_version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "models": {},
        }
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "registry_version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "models": {},
        }


def _write_registry(payload: Dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_promotion(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return lifecycle stage, pass/fail checks, and a single quality score."""
    metrics = metrics or {}
    checks = {
        "balanced_accuracy": _safe_float(metrics.get("balanced_accuracy")) >= PROMOTION_THRESHOLDS["balanced_accuracy"],
        "macro_f1": _safe_float(metrics.get("macro_f1")) >= PROMOTION_THRESHOLDS["macro_f1"],
        "auc": _safe_float(metrics.get("auc"), 0.5) >= PROMOTION_THRESHOLDS["auc"],
        "p_value": _safe_float(metrics.get("binom_pvalue"), 1.0) < PROMOTION_THRESHOLDS["max_pvalue"],
        "fold_stability": _safe_float(metrics.get("fold_std_accuracy"), 100.0) <= PROMOTION_THRESHOLDS["max_fold_std"],
        "calibration_samples": _safe_float(metrics.get("calibration_samples")) >= PROMOTION_THRESHOLDS["min_calibration_samples"],
    }

    quality_gate = str(metrics.get("model_quality_gate", "") or "").lower()
    gate_ok = quality_gate in {"validated", ""}
    checks["quality_gate"] = gate_ok

    failed = [name for name, passed in checks.items() if not passed]
    if not failed:
        stage = "production"
    elif "calibration_samples" in failed:
        stage = "candidate"
    else:
        stage = "quarantined"

    balanced = _safe_float(metrics.get("balanced_accuracy"))
    macro = _safe_float(metrics.get("macro_f1"))
    auc_score = _safe_float(metrics.get("auc"), 0.5) * 100.0
    fold_std = _safe_float(metrics.get("fold_std_accuracy"), 100.0)
    stability = max(0.0, 100.0 - fold_std * 3.0)
    p_penalty = 12.0 if not checks["p_value"] else 0.0
    gate_penalty = 15.0 if not checks["quality_gate"] else 0.0
    quality_score = max(0.0, (0.35 * balanced) + (0.30 * macro) + (0.20 * auc_score) + (0.15 * stability) - p_penalty - gate_penalty)

    return {
        "stage": stage,
        "quality_score": round(float(quality_score), 4),
        "checks": checks,
        "failed_checks": failed,
        "thresholds": PROMOTION_THRESHOLDS,
    }


def register_unified_model(
    ticker: str,
    model_version: str,
    artifact_path: str,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register a trained Unified Engine artifact and apply promotion gates."""
    ticker_key = ticker.upper()
    registry = _read_registry()
    models = registry.setdefault("models", {})
    history = models.setdefault(ticker_key, [])

    promotion = evaluate_promotion(metrics)
    record = {
        "ticker": ticker_key,
        "version": model_version,
        "artifact_path": artifact_path,
        "registered_at": datetime.utcnow().isoformat() + "Z",
        "stage": promotion["stage"],
        "quality_score": promotion["quality_score"],
        "promotion": promotion,
        "metrics": metrics or {},
        "metadata": metadata or {},
    }
    history.append(record)
    history.sort(key=lambda item: item.get("registered_at", ""))
    models[ticker_key] = history[-5:]
    _write_registry(registry)
    return record


def get_model_record(ticker: str, stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
    ticker_key = ticker.upper()
    records = _read_registry().get("models", {}).get(ticker_key, [])
    if stage:
        records = [record for record in records if record.get("stage") == stage]
    if not records:
        return None
    return sorted(records, key=lambda item: item.get("registered_at", ""))[-1]


def get_latest_or_production_record(ticker: str) -> Optional[Dict[str, Any]]:
    return get_model_record(ticker, stage="production") or get_model_record(ticker)


def list_model_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for ticker_records in _read_registry().get("models", {}).values():
        records.extend(ticker_records)
    return sorted(records, key=lambda item: item.get("registered_at", ""), reverse=True)

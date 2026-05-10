# app/tools/prediction_tools.py
"""
Agentic prediction tools backed by the production Unified Engine.

The chatbot should not invent trading answers. These helpers read the same
model artifacts, quality gates, and live market data used by the website.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if app_dir in sys.path:
    sys.path.remove(app_dir)
sys.path.insert(0, app_dir)
if backend_dir in sys.path:
    sys.path.remove(backend_dir)
sys.path.insert(1, backend_dir)

from models.schemas import PredictionResult

logger = logging.getLogger(__name__)


def _load_trained_stocks() -> set[str]:
    try:
        path = os.path.join(backend_dir, "mlops", "stocks.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return {str(t).upper() for t in json.load(f)}
    except Exception:
        pass
    return set()


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


def _quality_label(stage: str, gate: str) -> str:
    if stage == "production" and gate == "validated":
        return "production validated"
    if gate:
        return gate.replace("_", " ")
    if stage:
        return stage
    return "unregistered"


class PredictionTools:
    def __init__(self):
        self.trained_stocks = _load_trained_stocks()
        logger.info("PredictionTools initialized with %s configured tickers", len(self.trained_stocks))

    def refresh(self) -> None:
        self.trained_stocks = _load_trained_stocks()

    def _predict_sync(self, symbol: str, days: int = 14) -> Optional[Dict[str, Any]]:
        symbol = symbol.upper()
        try:
            from stock_api import get_stock_history
            from unified_engine.inference import UnifiedPredictor

            hist = get_stock_history(symbol, days=900, return_info=False)
            if hist is None or hist.empty or len(hist) < 120:
                return {
                    "symbol": symbol,
                    "status": "no_live_history",
                    "message": "Live market history is not sufficient yet.",
                }

            current_price = float(hist["Close"].iloc[-1])
            payload = UnifiedPredictor.predict(symbol, hist, current_price=current_price, days=days)
            metadata = UnifiedPredictor.get_model_metadata(symbol) or {}

            if not payload or not payload.get("predicted_prices"):
                return {
                    "symbol": symbol,
                    "status": "training_or_missing",
                    "current_price": current_price,
                    "model_available": UnifiedPredictor.is_model_available(symbol),
                    "metadata": metadata,
                    "message": "A trained forecast artifact is not ready yet. The website can queue training and show live market analysis while it prepares.",
                }

            metrics = payload.get("metrics") or {}
            promotion = payload.get("promotion") or metadata.get("promotion") or {}
            stage = payload.get("lifecycle_stage") or metadata.get("lifecycle_stage") or "unregistered"
            quality_gate = str(metrics.get("model_quality_gate") or "").lower()
            low = payload.get("price_range_low") or []
            high = payload.get("price_range_high") or []
            predicted = payload.get("predicted_prices") or []
            expected_move = _safe_float(payload.get("prediction")) * 100.0

            if expected_move > 1.0 and stage != "quarantined":
                direction = "BUY bias"
            elif expected_move < -1.0 and stage != "quarantined":
                direction = "SELL bias"
            else:
                direction = "HOLD / wait"

            return {
                "symbol": symbol,
                "status": "ready",
                "current_price": current_price,
                "signal": payload.get("signal", "HOLD"),
                "direction": direction,
                "confidence": _safe_float(payload.get("confidence")),
                "direction_prob": _safe_float(payload.get("direction_prob"), 0.5),
                "expected_move_pct": expected_move,
                "range_low": low[-1] if low else None,
                "range_high": high[-1] if high else None,
                "target_mid": predicted[-1] if predicted else None,
                "model_version": payload.get("model_version"),
                "trained_at": payload.get("trained_at"),
                "lifecycle_stage": stage,
                "quality_gate": quality_gate,
                "quality_label": _quality_label(stage, quality_gate),
                "promotion": promotion,
                "metrics": metrics,
            }
        except Exception as exc:
            logger.exception("Unified prediction failed for %s", symbol)
            return {"symbol": symbol, "status": "error", "message": str(exc)}

    async def get_prediction_payload(self, symbol: str, days: int = 14) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._predict_sync, symbol, days)

    async def get_latest_prediction(self, symbol: str) -> Optional[PredictionResult]:
        payload = await self.get_prediction_payload(symbol, days=14)
        if not payload:
            return None

        if payload.get("status") != "ready":
            return PredictionResult(
                symbol=symbol.upper(),
                predicted_direction="Model training or validation is not ready",
                confidence=0.0,
                target_price=None,
                key_factors=[payload.get("message", "Model is not ready.")],
                metrics={},
            )

        metrics = payload.get("metrics") or {}
        factors = [
            f"Lifecycle: {payload.get('quality_label')}",
            f"Model version: {payload.get('model_version')}",
            f"Forecast range: {payload.get('range_low')} to {payload.get('range_high')}",
            f"Expected move: {payload.get('expected_move_pct', 0):+.2f}%",
            f"Balanced accuracy: {_safe_float(metrics.get('balanced_accuracy')):.1f}%",
            f"Macro-F1: {_safe_float(metrics.get('macro_f1')):.1f}%",
            f"AUC: {_safe_float(metrics.get('auc'), 0.5):.3f}",
        ]
        failed = ((payload.get("promotion") or {}).get("failed_checks") or [])
        if failed:
            factors.append("Failed promotion checks: " + ", ".join(failed))

        return PredictionResult(
            symbol=symbol.upper(),
            predicted_direction=str(payload.get("direction") or payload.get("signal") or "HOLD"),
            confidence=round(_safe_float(payload.get("confidence")), 3),
            target_price=payload.get("target_mid"),
            key_factors=factors,
            metrics={k: _safe_float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        )

    async def rank_watchlist(self, symbols: List[str], days: int = 14) -> Dict[str, Any]:
        cleaned = []
        for symbol in symbols:
            symbol = str(symbol).upper().strip()
            if symbol and symbol not in cleaned:
                cleaned.append(symbol)
        cleaned = cleaned[:10]

        payloads = await asyncio.gather(
            *[self.get_prediction_payload(symbol, days=days) for symbol in cleaned],
            return_exceptions=True,
        )

        rows = []
        for symbol, payload in zip(cleaned, payloads):
            if isinstance(payload, Exception) or not payload:
                rows.append({"symbol": symbol, "status": "error", "score": 0, "why": "Prediction tool failed."})
                continue
            if payload.get("status") != "ready":
                rows.append({
                    "symbol": symbol,
                    "status": payload.get("status"),
                    "score": 0,
                    "why": payload.get("message", "Model is not ready."),
                })
                continue

            metrics = payload.get("metrics") or {}
            promotion = payload.get("promotion") or {}
            stage = payload.get("lifecycle_stage", "unregistered")
            quality_score = _safe_float(promotion.get("quality_score"), 0.0)
            confidence_score = _safe_float(payload.get("confidence")) * 100.0
            edge_score = min(100.0, abs(_safe_float(payload.get("expected_move_pct"))) * 12.0)
            stage_penalty = 25.0 if stage == "quarantined" else 8.0 if stage == "candidate" else 0.0
            score = max(0.0, (0.50 * quality_score) + (0.25 * confidence_score) + (0.25 * edge_score) - stage_penalty)

            rows.append({
                "symbol": symbol,
                "status": "ready",
                "score": round(score, 1),
                "signal": payload.get("signal"),
                "direction": payload.get("direction"),
                "current_price": payload.get("current_price"),
                "forecast_range": [payload.get("range_low"), payload.get("range_high")],
                "expected_move_pct": round(_safe_float(payload.get("expected_move_pct")), 2),
                "trust": round(_safe_float(metrics.get("balanced_accuracy")), 1),
                "macro_f1": round(_safe_float(metrics.get("macro_f1")), 1),
                "auc": round(_safe_float(metrics.get("auc"), 0.5), 3),
                "stage": stage,
                "quality_gate": payload.get("quality_gate"),
                "failed_checks": promotion.get("failed_checks", []),
                "why": self._row_reason(payload),
            })

        rows.sort(key=lambda item: item.get("score", 0), reverse=True)
        return {"symbols": cleaned, "ranked": rows, "count": len(rows)}

    def _row_reason(self, payload: Dict[str, Any]) -> str:
        failed = ((payload.get("promotion") or {}).get("failed_checks") or [])
        if failed:
            return "Use caution: failed " + ", ".join(failed) + "."
        move = _safe_float(payload.get("expected_move_pct"))
        stage = payload.get("lifecycle_stage")
        if stage == "production":
            return f"Production model with {move:+.2f}% expected move and validated quality gates."
        return f"{stage} model with {move:+.2f}% expected move."

    async def get_trained_stocks_list(self) -> list:
        self.refresh()
        return sorted(self.trained_stocks)


prediction_tools = PredictionTools()

"""
Evaluate the production signal path for the top 10 portfolio tickers.

This script is intentionally backend-only. It checks the same trained Unified
Engine artifacts used by the API, computes live/current signals, and writes a
JSON report that compares current signal behavior with saved model metadata.

Usage:
    python evaluate_top10_models.py
    python evaluate_top10_models.py --retrain
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

TOP_10_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _compute_basic_indicators(hist: pd.DataFrame, current_price: float) -> Dict[str, float]:
    close = pd.to_numeric(hist["Close"], errors="coerce")
    high = pd.to_numeric(hist.get("High", close), errors="coerce")
    low = pd.to_numeric(hist.get("Low", close), errors="coerce")

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()

    return {
        "rsi": _safe_float(rsi.iloc[-1], 50.0),
        "ema": _safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1], current_price),
        "macd": _safe_float(macd.iloc[-1]),
        "macd_signal": _safe_float(macd_signal.iloc[-1]),
        "macd_histogram": _safe_float((macd - macd_signal).iloc[-1]),
        "sma20": _safe_float(close.rolling(20).mean().iloc[-1], current_price),
        "sma50": _safe_float(close.rolling(50).mean().iloc[-1], current_price),
        "atr": _safe_float(atr.iloc[-1], current_price * 0.015),
    }


def _load_metadata(ticker: str) -> Dict[str, Any]:
    from unified_engine.inference import _get_artifact_dir

    metadata_path = _get_artifact_dir(ticker) / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def evaluate_ticker(ticker: str, retrain: bool = False) -> Dict[str, Any]:
    from decision_engine import build_decision
    from stock_api import get_sentiment_analysis, get_stock_history
    from unified_engine.inference import UnifiedPredictor, clear_cache

    if retrain:
        from unified_engine.training import train_unified_model

        train_result = train_unified_model(ticker)
        clear_cache(ticker)
        if not train_result.success:
            return {
                "ticker": ticker,
                "status": "train_failed",
                "reason": train_result.reason,
            }

    hist = get_stock_history(ticker, days=900, return_info=False)
    if hist is None or hist.empty or len(hist) < 120:
        return {"ticker": ticker, "status": "no_data", "rows": 0 if hist is None else len(hist)}

    current_price = float(hist["Close"].iloc[-1])
    payload = UnifiedPredictor.predict(ticker, hist, current_price=current_price, days=14)
    metadata = _load_metadata(ticker)

    if not payload or not payload.get("predicted_prices"):
        return {
            "ticker": ticker,
            "status": "no_model_or_prediction",
            "model_available": UnifiedPredictor.is_model_available(ticker),
            "metadata": metadata,
        }

    sentiment = get_sentiment_analysis(ticker)
    indicators = _compute_basic_indicators(hist, current_price)
    decision = build_decision(
        hist=hist,
        current_price=current_price,
        predictions=[float(p) for p in payload["predicted_prices"]],
        indicators=indicators,
        sentiment=sentiment,
        model_payload=payload,
    )

    metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
    signal_metrics = {
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
        "macro_f1": metrics.get("macro_f1"),
        "negative_f1": metrics.get("negative_f1"),
        "auc": metrics.get("auc"),
        "mcc": metrics.get("mcc"),
        "brier_score": metrics.get("brier_score"),
        "binom_pvalue": metrics.get("binom_pvalue"),
        "fold_mean_accuracy": metrics.get("fold_mean_accuracy"),
        "fold_mean_train_accuracy": metrics.get("fold_mean_train_accuracy"),
        "fold_std_accuracy": metrics.get("fold_std_accuracy"),
        "mean_generalization_gap": metrics.get("mean_generalization_gap"),
        "overfit_risk": metrics.get("overfit_risk"),
        "underfit_risk": metrics.get("underfit_risk"),
        "training_samples": metrics.get("training_samples"),
        "calibration_samples": metrics.get("calibration_samples"),
        "n_folds": metrics.get("n_folds"),
        "actual_up_rate": metrics.get("actual_up_rate"),
        "predicted_up_rate": metrics.get("predicted_up_rate"),
        "validation_threshold": metrics.get("validation_threshold"),
        "model_quality_gate": metrics.get("model_quality_gate"),
    }
    return {
        "ticker": ticker,
        "status": "ok",
        "signal": decision.signal,
        "confidence": decision.confidence,
        "score": decision.score,
        "expected_move_pct": decision.expected_move_pct,
        "min_required_move_pct": decision.min_required_move_pct,
        "direction_prob": _safe_float(payload.get("direction_prob"), 0.5),
        "prediction": _safe_float(payload.get("prediction"), 0.0),
        "lower_95": _safe_float(payload.get("lower_95"), 0.0),
        "upper_95": _safe_float(payload.get("upper_95"), 0.0),
        "model_version": payload.get("model_version"),
        "lifecycle_stage": payload.get("lifecycle_stage"),
        "promotion": payload.get("promotion"),
        "trained_at": payload.get("trained_at") or metadata.get("trained_at"),
        "metrics": signal_metrics,
        "metadata_accuracy": metrics.get("accuracy"),
        "metadata_auc": metrics.get("auc"),
        "metadata_f1": metrics.get("f1"),
        "reasons": decision.reasons,
        "components": decision.components,
        "sentiment": {
            "score": sentiment.get("score", 0) if isinstance(sentiment, dict) else 0,
            "confidence": sentiment.get("sentiment_confidence", 0) if isinstance(sentiment, dict) else 0,
            "buzz_articles": sentiment.get("buzz_articles", 0) if isinstance(sentiment, dict) else 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Retrain each ticker before evaluating.")
    parser.add_argument("--tickers", nargs="*", default=TOP_10_TICKERS, help="Override ticker list.")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []
    for ticker in args.tickers:
        print(f"Evaluating {ticker}...")
        try:
            results.append(evaluate_ticker(ticker, retrain=args.retrain))
        except Exception as exc:
            results.append({"ticker": ticker, "status": "error", "error": str(exc)})

    ok = [row for row in results if row.get("status") == "ok"]
    signal_counts: Dict[str, int] = {}
    for row in ok:
        signal_counts[row["signal"]] = signal_counts.get(row["signal"], 0) + 1

    def avg_metric(name: str) -> float:
        values = [
            _safe_float((row.get("metrics") or {}).get(name), None)
            for row in ok
            if (row.get("metrics") or {}).get(name) is not None
        ]
        values = [value for value in values if value is not None]
        return round(sum(values) / max(len(values), 1), 4)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "retrained": bool(args.retrain),
        "tickers": args.tickers,
        "summary": {
            "evaluated": len(results),
            "ok": len(ok),
            "signal_counts": signal_counts,
            "avg_confidence": round(sum(row["confidence"] for row in ok) / max(len(ok), 1), 4),
            "avg_accuracy": avg_metric("accuracy"),
            "avg_balanced_accuracy": avg_metric("balanced_accuracy"),
            "avg_f1": avg_metric("f1"),
            "avg_macro_f1": avg_metric("macro_f1"),
            "avg_auc": avg_metric("auc"),
            "supported_prediction_horizons": [1, 7, 14],
        },
        "results": results,
    }

    out_dir = Path("monitoring") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "top10_signal_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(f"Saved report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

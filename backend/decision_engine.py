"""
Institutional-style decision fusion for stock recommendations.

This module is deliberately independent from model training.  It consumes the
latest model forecast, technical state, sentiment state, and realized risk, then
returns the single recommendation contract used by the API and frontend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


SIGNALS = ("STRONG SELL", "SELL", "HOLD", "BUY", "STRONG BUY")


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _signal_from_score(score: float, confidence: float, expected_move_pct: float, min_move_pct: float) -> str:
    """Map a risk-adjusted score to a trade label with a hard no-trade band."""
    if confidence < 0.38 or abs(expected_move_pct) < min_move_pct:
        return "HOLD"
    if score >= 0.72 and confidence >= 0.70:
        return "STRONG BUY"
    if score >= 0.34:
        return "BUY"
    if score <= -0.72 and confidence >= 0.70:
        return "STRONG SELL"
    if score <= -0.34:
        return "SELL"
    return "HOLD"


@dataclass
class DecisionResult:
    signal: str
    stance: str
    confidence: float
    score: float
    expected_move_pct: float
    min_required_move_pct: float
    risk_adjusted_quality: float
    uncertainty: str
    reasons: List[str]
    components: Dict[str, float]
    thresholds: Dict[str, float]
    risk: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["confidence_percent"] = round(self.confidence * 100.0, 1)
        return payload


def compute_market_state(hist: pd.DataFrame, current_price: float, indicators: Dict[str, Any]) -> Dict[str, float]:
    close = pd.to_numeric(hist.get("Close"), errors="coerce").dropna()
    volume = pd.to_numeric(hist.get("Volume"), errors="coerce").fillna(0.0)
    if close.empty:
        return {
            "daily_vol_pct": 1.0,
            "atr_pct": 1.0,
            "trend_strength": 0.0,
            "momentum_5d_pct": 0.0,
            "momentum_20d_pct": 0.0,
            "volume_ratio": 1.0,
        }

    returns = close.pct_change().dropna()
    daily_vol_pct = float(returns.tail(20).std() * 100.0) if len(returns) >= 5 else 1.0
    daily_vol_pct = max(0.15, min(daily_vol_pct, 12.0))

    atr = _safe_float(indicators.get("atr"), 0.0)
    atr_pct = (atr / current_price) * 100.0 if current_price > 0 and atr > 0 else daily_vol_pct
    atr_pct = max(0.15, min(float(atr_pct), 15.0))

    sma20 = _safe_float(indicators.get("sma20"), current_price)
    sma50 = _safe_float(indicators.get("sma50"), sma20)
    trend_strength = 0.0
    if current_price > 0:
        trend_strength += _clamp(((current_price / max(sma20, 1e-9)) - 1.0) / 0.04)
        trend_strength += _clamp(((sma20 / max(sma50, 1e-9)) - 1.0) / 0.035)
        trend_strength /= 2.0

    momentum_5d_pct = 0.0
    momentum_20d_pct = 0.0
    if len(close) > 5 and close.iloc[-6] > 0:
        momentum_5d_pct = float((close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0)
    if len(close) > 20 and close.iloc[-21] > 0:
        momentum_20d_pct = float((close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0)

    avg_volume = float(volume.tail(20).mean()) if len(volume) else 0.0
    latest_volume = float(volume.iloc[-1]) if len(volume) else 0.0
    volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0

    return {
        "daily_vol_pct": round(daily_vol_pct, 4),
        "atr_pct": round(atr_pct, 4),
        "trend_strength": round(_clamp(trend_strength), 4),
        "momentum_5d_pct": round(momentum_5d_pct, 4),
        "momentum_20d_pct": round(momentum_20d_pct, 4),
        "volume_ratio": round(float(max(0.0, min(volume_ratio, 5.0))), 4),
    }


def score_technical(indicators: Dict[str, Any], current_price: float, market: Dict[str, float]) -> float:
    rsi = _safe_float(indicators.get("rsi"), 50.0)
    macd = _safe_float(indicators.get("macd"), 0.0)
    macd_signal = _safe_float(indicators.get("macd_signal"), 0.0)
    macd_hist = _safe_float(indicators.get("macd_histogram"), macd - macd_signal)
    ema20 = _safe_float(indicators.get("ema"), current_price)
    sma20 = _safe_float(indicators.get("sma20"), current_price)
    sma50 = _safe_float(indicators.get("sma50"), sma20)

    rsi_score = 0.0
    if rsi < 25:
        rsi_score = 0.55
    elif rsi < 35:
        rsi_score = 0.30
    elif rsi > 75:
        rsi_score = -0.55
    elif rsi > 65:
        rsi_score = -0.30
    else:
        rsi_score = _clamp((50.0 - rsi) / 50.0) * 0.15

    macd_score = _clamp(macd_hist / max(abs(current_price) * 0.01, 1e-9))
    ema_score = _clamp(((current_price / max(ema20, 1e-9)) - 1.0) / 0.03)
    sma_score = _clamp(((sma20 / max(sma50, 1e-9)) - 1.0) / 0.04)

    return _clamp(
        0.30 * rsi_score
        + 0.25 * macd_score
        + 0.25 * ema_score
        + 0.20 * sma_score
        + 0.20 * market["trend_strength"]
    )


def score_sentiment(sentiment: Optional[Dict[str, Any]]) -> tuple[float, float]:
    if not sentiment:
        return 0.0, 0.25

    raw_score = _clamp(_safe_float(sentiment.get("score"), 0.0))
    buzz_articles = _safe_float(sentiment.get("buzz_articles"), 0.0)
    buzz_score = _safe_float(sentiment.get("buzz_score"), 0.0)

    article_quality = min(1.0, buzz_articles / 8.0) if buzz_articles > 0 else 0.35
    provider_quality = min(1.0, max(float(buzz_score), article_quality))
    quality = max(0.25, min(1.0, provider_quality))
    return _clamp(raw_score * quality), quality


def build_decision(
    *,
    hist: pd.DataFrame,
    current_price: float,
    predictions: List[float],
    indicators: Dict[str, Any],
    sentiment: Optional[Dict[str, Any]],
    model_payload: Optional[Dict[str, Any]] = None,
) -> DecisionResult:
    """Create the final backend-owned decision."""
    if not current_price or current_price <= 0:
        raise ValueError("current_price must be positive")

    first_prediction = float(predictions[0]) if predictions else current_price
    last_prediction = float(predictions[-1]) if predictions else first_prediction
    one_day_move_pct = ((first_prediction - current_price) / current_price) * 100.0
    horizon_move_pct = ((last_prediction - current_price) / current_price) * 100.0

    market = compute_market_state(hist, current_price, indicators)

    model_expected_return = _safe_float((model_payload or {}).get("prediction"), horizon_move_pct / 100.0)
    model_expected_pct = model_expected_return * 100.0
    expected_move_pct = model_expected_pct if abs(model_expected_pct) >= abs(one_day_move_pct) else one_day_move_pct

    direction_prob = _safe_float((model_payload or {}).get("direction_prob"), 0.5 + _clamp(expected_move_pct / 10.0) / 2.0)
    model_conf = _safe_float((model_payload or {}).get("confidence"), abs(direction_prob - 0.5) * 2.0)
    ci_low = _safe_float((model_payload or {}).get("lower_95"), model_expected_return - 0.02) * 100.0
    ci_high = _safe_float((model_payload or {}).get("upper_95"), model_expected_return + 0.02) * 100.0
    ci_width_pct = max(0.01, ci_high - ci_low)

    ml_score = _clamp((direction_prob - 0.5) * 2.0)
    if np.sign(ml_score) != np.sign(expected_move_pct) and abs(expected_move_pct) > 0.05:
        ml_score *= 0.45

    technical_score = score_technical(indicators, current_price, market)
    sentiment_score, sentiment_quality = score_sentiment(sentiment)
    momentum_score = _clamp((0.65 * market["momentum_5d_pct"] + 0.35 * market["momentum_20d_pct"]) / 8.0)
    volume_confirmation = _clamp((market["volume_ratio"] - 1.0) / 1.5)
    volume_score = volume_confirmation * _clamp(np.sign(expected_move_pct or ml_score))

    volatility_penalty = min(0.35, max(0.0, (market["daily_vol_pct"] - 3.0) / 20.0))
    raw_score = (
        0.35 * ml_score
        + 0.25 * technical_score
        + 0.20 * sentiment_score
        + 0.10 * momentum_score
        + 0.05 * volume_score
    )
    final_score = _clamp(raw_score - volatility_penalty * np.sign(raw_score))

    min_required_move_pct = max(0.35, market["atr_pct"] * 0.35, market["daily_vol_pct"] * 0.45)
    move_quality = min(1.0, abs(expected_move_pct) / max(min_required_move_pct * 2.0, 1e-9))
    interval_quality = max(0.0, min(1.0, 1.0 - (ci_width_pct / max(abs(expected_move_pct) * 4.0 + 1.0, 1.0))))
    agreement = 1.0 - (np.std([ml_score, technical_score, sentiment_score, momentum_score]) / 1.2)
    agreement = max(0.0, min(1.0, float(agreement)))
    confidence = max(0.0, min(1.0, 0.40 * model_conf + 0.25 * move_quality + 0.20 * agreement + 0.15 * interval_quality))

    reasons: List[str] = []
    if abs(expected_move_pct) < min_required_move_pct:
        reasons.append(
            f"Expected move {expected_move_pct:+.2f}% is below the volatility-adjusted trade threshold "
            f"{min_required_move_pct:.2f}%."
        )
    if sentiment_score <= -0.45 and expected_move_pct > 0 and abs(expected_move_pct) < min_required_move_pct * 1.8:
        reasons.append("Bearish news sentiment vetoed a weak bullish forecast.")
        final_score = min(final_score, 0.15)
        confidence = min(confidence, 0.55)
    if sentiment_score >= 0.45 and expected_move_pct < 0 and abs(expected_move_pct) < min_required_move_pct * 1.8:
        reasons.append("Bullish news sentiment vetoed a weak bearish forecast.")
        final_score = max(final_score, -0.15)
        confidence = min(confidence, 0.55)
    if technical_score * ml_score < -0.15:
        reasons.append("Technical indicators disagree with the ML direction.")
        confidence = min(confidence, 0.62)
    if ci_low <= 0 <= ci_high:
        reasons.append("Prediction interval crosses zero, so directional edge is uncertain.")
        confidence = min(confidence, 0.60)
    if market["daily_vol_pct"] > 5.0:
        reasons.append("High realized volatility reduced signal quality.")

    signal = _signal_from_score(final_score, confidence, expected_move_pct, min_required_move_pct)
    if signal == "HOLD" and not reasons:
        reasons.append("Composite score is inside the no-trade band.")

    if signal.endswith("BUY"):
        stance = "Bullish" if signal == "BUY" else "Strong Bullish"
    elif signal.endswith("SELL"):
        stance = "Bearish" if signal == "SELL" else "Strong Bearish"
    elif abs(final_score) < 0.18:
        stance = "Uncertain Market"
    elif final_score > 0:
        stance = "Weak Bullish"
    else:
        stance = "Weak Bearish"

    risk_adjusted_quality = max(0.0, min(1.0, confidence * (1.0 - volatility_penalty)))
    uncertainty = "high" if confidence < 0.45 else "medium" if confidence < 0.68 else "low"

    return DecisionResult(
        signal=signal,
        stance=stance,
        confidence=round(float(confidence), 4),
        score=round(float(final_score), 4),
        expected_move_pct=round(float(expected_move_pct), 4),
        min_required_move_pct=round(float(min_required_move_pct), 4),
        risk_adjusted_quality=round(float(risk_adjusted_quality), 4),
        uncertainty=uncertainty,
        reasons=reasons[:4],
        components={
            "ml": round(float(ml_score), 4),
            "technical": round(float(technical_score), 4),
            "sentiment": round(float(sentiment_score), 4),
            "sentiment_quality": round(float(sentiment_quality), 4),
            "momentum": round(float(momentum_score), 4),
            "volume": round(float(volume_score), 4),
            "volatility_penalty": round(float(volatility_penalty), 4),
        },
        thresholds={
            "buy_score": 0.34,
            "strong_buy_score": 0.72,
            "sell_score": -0.34,
            "strong_sell_score": -0.72,
            "min_confidence": 0.38,
            "min_required_move_pct": round(float(min_required_move_pct), 4),
        },
        risk=market,
    )


def signal_for_forecast_row(
    *,
    base_price: float,
    predicted_price: float,
    final_decision: DecisionResult,
) -> str:
    """Create horizon-row labels without contradicting the final decision."""
    if base_price <= 0:
        return "HOLD"
    change_pct = ((predicted_price - base_price) / base_price) * 100.0
    min_move = final_decision.min_required_move_pct
    confidence = final_decision.confidence
    score = final_decision.score

    if abs(change_pct) < min_move or confidence < 0.38:
        return "HOLD"
    if change_pct > 0 and score > 0:
        if change_pct >= min_move * 2.5 and confidence >= 0.70:
            return "STRONG BUY"
        return "BUY"
    if change_pct < 0 and score < 0:
        if abs(change_pct) >= min_move * 2.5 and confidence >= 0.70:
            return "STRONG SELL"
        return "SELL"
    return "HOLD"

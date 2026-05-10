from flask import Flask, request, render_template, jsonify, send_from_directory, g
from flask_cors import CORS
import os
import json
import base64
from datetime import datetime, timedelta
import warnings
import logging
import threading
import uuid
from logging.handlers import RotatingFileHandler
from functools import wraps
from werkzeug.exceptions import HTTPException

# Suppress verbose HTTP retry warnings from urllib3 when DagsHub is slow
logging.getLogger("urllib3").setLevel(logging.ERROR)

warnings.filterwarnings('ignore')

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth, credentials
except ImportError:  # Firebase admin is optional; admin endpoints will be disabled if missing
    firebase_admin = None
    firebase_auth = None
    credentials = None

from dotenv import load_dotenv

load_dotenv()


def _offline_chatbot_reply(message: str):
    """Return a useful assistant response when the dedicated chatbot service is not running."""
    text = (message or '').strip()
    lower = text.lower()

    if any(word in lower for word in ('hi', 'hello', 'hey', 'how it work', 'how its work', 'how does it work')):
        reply = (
            "Hey, welcome to Datavision. Here is how it works:\n\n"
            "1. Enter a stock symbol like AAPL, NVDA, RELIANCE.NS, or 7203.T.\n"
            "2. The app loads live market data, technical indicators, sentiment, and news.\n"
            "3. If a ticker-specific ML model is ready, it shows the forecast range, signal, trust score, risk, and trading chart.\n"
            "4. If the model is not ready yet, the page shows live analysis while the stock-specific forecast is prepared.\n"
            "5. Use Opportunity Radar to compare multiple stocks and rank the best setup by risk-adjusted AI score.\n\n"
            "Ask me for a stock prediction by typing something like: predict AAPL."
        )
    elif any(word in lower for word in ('predict', 'prediction', 'forecast', 'target', 'outlook')):
        reply = (
            "I can help with predictions. Please send the ticker you want, for example AAPL, NVDA, TSLA, "
            "RELIANCE.NS, or 7203.T. The main prediction engine will then show live price, AI range, signal, trust, and risk."
        )
    else:
        reply = (
            "I can help explain this ML stock platform, compare watchlist stocks, or guide you to a prediction. "
            "Try asking: predict AAPL, explain trust score, or how does Opportunity Radar work?"
        )

    return jsonify({
        'reply': reply,
        'stock': None,
        'data': {'mode': 'fallback_assistant', 'reason': 'chatbot_service_unavailable'},
        'chat_id': str(uuid.uuid4())
    })


def _proxy_chatbot(subpath: str):
    """Forward requests to the FastAPI chatbot running on port 8001."""
    import requests as _requests
    url = f'http://127.0.0.1:8001{subpath}'
    try:
        # Build headers — keep content-type, drop hop-by-hop headers
        fwd_headers = {}
        for k, v in request.headers:
            kl = k.lower()
            if kl in ('host', 'transfer-encoding', 'connection'):
                continue
            fwd_headers[k] = v

        resp = _requests.request(
            method=request.method,
            url=url,
            headers=fwd_headers,
            data=request.get_data(),
            params=request.args,
            timeout=120,
        )
        excluded_headers = {'content-encoding', 'content-length', 'transfer-encoding', 'connection'}
        headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded_headers}
        return (resp.content, resp.status_code, headers)
    except _requests.exceptions.ConnectionError:
        if subpath == '/chat' and request.method == 'POST':
            payload = request.get_json(silent=True) or {}
            return _offline_chatbot_reply(payload.get('message', ''))
        if subpath == '/chats' and request.method == 'GET':
            return jsonify({'chats': []})
        if subpath == '/feedback' and request.method == 'POST':
            return jsonify({'message': 'Feedback received locally. Dedicated chat learning service is offline.', 'rating': (request.get_json(silent=True) or {}).get('rating')})
        return jsonify({'error': 'Chatbot service is starting up or unavailable', 'detail': 'The chatbot server at port 8001 is not responding. It may still be initializing.'}), 503
    except Exception as e:
        return jsonify({'error': 'Chatbot service unavailable', 'detail': str(e)}), 503

# Heavy imports (pandas, numpy, ta, stock_api) will be lazy-loaded when needed
# for standard production configuration

# Model will be loaded lazily on first use
# Check multiple possible model paths; prefer native .keras and keep legacy .h5 fallback.
MODEL_PATHS = [
    'artifacts/stock_lstm_model.keras',
    'artifacts/AAPL_lstm_model.keras',
    'artifacts/stock_lstm_model.h5',
    'artifacts/AAPL_lstm_model.h5',
]
SCALER_PATH = 'artifacts/scaler.pkl'
# Per-ticker model cache: { 'AAPL': model, 'TSLA': model, ... }
_model_cache = {}
_scaler_cache = {}
_ultimate_prediction_cache = {}
# Per-ticker model metadata cache: { 'AAPL': {'version': '1', 'run_id': 'xyz', 'source': 'mlflow'} }
_model_metadata = {}
_training_in_progress = set()
_training_state_lock = threading.Lock()
_ultimate_training_in_progress = set()
_ultimate_training_state_lock = threading.Lock()
_ticker_training_jobs = {}
_ticker_training_jobs_lock = threading.Lock()

SUFFIX_EXCHANGE_COUNTRY = {
    '.NS': ('NSE', 'India'),
    '.BO': ('BSE', 'India'),
    '.L': ('LSE', 'United Kingdom'),
    '.HK': ('HKEX', 'Hong Kong'),
    '.TO': ('TSX', 'Canada'),
    '.AX': ('ASX', 'Australia'),
    '.SI': ('SGX', 'Singapore'),
    '.SS': ('SSE', 'China'),
    '.SZ': ('SZSE', 'China'),
    '.F': ('FWB', 'Germany'),
    '.SW': ('SIX', 'Switzerland'),
    '.PA': ('EURONEXT', 'France'),
    '.MC': ('BME', 'Spain'),
    '.MI': ('Borsa Italiana', 'Italy')
}

PREDICTION_HORIZONS = (1, 7, 14)
NON_TRAINABLE_SUFFIXES = ('.PVT',)


def _is_trainable_market_ticker(ticker: str) -> bool:
    ticker_key = (ticker or '').strip().upper()
    if not ticker_key:
        return False
    return not ticker_key.endswith(NON_TRAINABLE_SUFFIXES)


def _unsupported_training_message(ticker: str) -> str:
    ticker_key = (ticker or '').strip().upper() or 'This symbol'
    return (
        f'{ticker_key} cannot be trained as an AI forecast model yet. '
        'Datavision needs a public exchange-listed ticker with enough daily OHLCV history. '
        'Search and choose the listed market symbol, for example AAPL, RELIANCE.NS, TCS.NS, or 7203.T.'
    )


def _training_failure_status(reason) -> tuple[str, str, str]:
    """Classify training failures into user-facing pipeline states."""
    raw_reason = str(reason or 'unknown training issue')
    normalized = raw_reason.lower()
    insufficient_history_terms = (
        'no data', 'insufficient', 'not enough', 'empty', 'unsupported',
        'private', 'delisted', 'no price', 'no history', 'failed download',
        'possibly delisted', 'not found', 'symbol may be delisted'
    )
    if any(term in normalized for term in insufficient_history_terms):
        return (
            'untrainable',
            'not_enough_market_history',
            'This stock does not have enough reliable market history for a validated AI forecast. Try the exchange-listed ticker from search suggestions.'
        )
    return (
        'failed',
        'retry_available',
        'Training stopped before validation completed. You can retry and Datavision will publish the forecast only after validation succeeds.'
    )


def _clamp_number(value, low=0.0, high=1.0):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low


def _build_market_regime(hist, indicators, current_price):
    """User-facing market regime label derived from trend, volatility, volume, and range state."""
    try:
        close = hist['Close'].dropna()
        returns = close.pct_change().dropna()
        vol_pct = float(returns.tail(20).std() * 100.0) if len(returns) >= 5 else 0.0
        sma20 = float(indicators.get('sma20') or current_price)
        sma50 = float(indicators.get('sma50') or sma20)
        ema20 = float(indicators.get('ema') or sma20)
        rsi = float(indicators.get('rsi') or 50.0)
        adx = float(indicators.get('adx') or 0.0)
        atr = float(indicators.get('atr') or 0.0)
        atr_pct = (atr / current_price) * 100.0 if current_price and atr > 0 else vol_pct
        momentum_20d = ((close.iloc[-1] / close.iloc[-21]) - 1.0) * 100.0 if len(close) > 21 and close.iloc[-21] else 0.0
        volume = hist['Volume'].dropna() if 'Volume' in hist else []
        avg_volume = float(volume.tail(20).mean()) if len(volume) >= 5 else 0.0
        volume_ratio = float(volume.iloc[-1] / avg_volume) if avg_volume > 0 and len(volume) else 1.0
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        bb_width_pct = ((float(bb_upper) - float(bb_lower)) / current_price) * 100.0 if bb_upper and bb_lower and current_price else None
        range_pct = ((close.tail(20).max() - close.tail(20).min()) / current_price) * 100.0 if len(close) >= 20 and current_price else vol_pct
        trend_strength = _clamp_number((adx or 0.0) / 45.0, 0.0, 1.0)

        if range_pct <= 5.0 and atr_pct <= 2.2 and (bb_width_pct is None or bb_width_pct <= 8.0):
            label = 'Volatility Squeeze'
            tone = 'neutral'
        elif vol_pct >= 4.5 or atr_pct >= 5.5:
            label = 'High Volatility'
            tone = 'warning'
        elif current_price > ema20 > sma50 and momentum_20d >= 6.0 and adx >= 25:
            label = 'Momentum Leader'
            tone = 'bullish'
        elif current_price > ema20 and ema20 >= sma50 and momentum_20d > 2.0 and adx >= 18:
            label = 'Bull Trend'
            tone = 'bullish'
        elif current_price < ema20 < sma50 and momentum_20d < -2.0 and volume_ratio >= 1.1:
            label = 'Distribution'
            tone = 'bearish'
        elif current_price < ema20 < sma50 and momentum_20d < -2.0:
            label = 'Bear Trend'
            tone = 'bearish'
        elif abs(momentum_20d) <= 2.0 and adx < 18:
            label = 'Consolidation'
            tone = 'neutral'
        elif current_price > ema20 and momentum_20d > 0:
            label = 'Recovery'
            tone = 'bullish'
        else:
            label = 'Sideways'
            tone = 'neutral'

        return {
            'label': label,
            'tone': tone,
            'daily_vol_pct': round(float(vol_pct), 2),
            'atr_pct': round(float(atr_pct), 2),
            'momentum_20d_pct': round(float(momentum_20d), 2),
            'adx': round(float(adx), 2),
            'trend_strength': round(float(trend_strength), 4),
            'volume_ratio': round(float(volume_ratio), 2),
            'bb_width_pct': None if bb_width_pct is None else round(float(bb_width_pct), 2),
        }
    except Exception:
        return {
            'label': 'Unknown Regime',
            'tone': 'neutral',
            'daily_vol_pct': None,
            'atr_pct': None,
            'momentum_20d_pct': None,
            'adx': None,
            'trend_strength': 0.0,
            'volume_ratio': None,
            'bb_width_pct': None,
        }


def _build_risk_profile(hist, indicators, current_price, market_regime):
    """Risk score based on ATR, realized volatility, and recent drawdown."""
    try:
        close = hist['Close'].dropna()
        returns = close.pct_change().dropna()
        atr = float(indicators.get('atr') or 0.0)
        atr_pct = (atr / current_price) * 100.0 if current_price and atr > 0 else 0.0
        std_20d_pct = float(returns.tail(20).std() * 100.0) if len(returns) >= 5 else 0.0
        window = close.tail(90)
        running_max = window.cummax()
        drawdowns = (window / running_max - 1.0) * 100.0 if len(window) else []
        max_drawdown_pct = abs(float(drawdowns.min())) if len(drawdowns) else 0.0

        atr_component = _clamp_number(atr_pct / 7.0, 0.0, 1.0)
        std_component = _clamp_number(std_20d_pct / 5.0, 0.0, 1.0)
        drawdown_component = _clamp_number(max_drawdown_pct / 35.0, 0.0, 1.0)
        score = (0.4 * atr_component + 0.3 * std_component + 0.3 * drawdown_component) * 100.0
        if market_regime.get('tone') == 'warning':
            score += 10.0
        score = _clamp_number(score, 0.0, 100.0)

        if score >= 72:
            label = 'Speculative'
        elif score >= 55:
            label = 'High'
        elif score >= 32:
            label = 'Medium'
        else:
            label = 'Low'

        return {
            'label': label,
            'score': round(float(score), 1),
            'atr_pct': round(float(atr_pct), 2),
            'std_20d_pct': round(float(std_20d_pct), 2),
            'max_drawdown_pct': round(float(max_drawdown_pct), 2),
        }
    except Exception:
        return {
            'label': 'Medium',
            'score': 45.0,
            'atr_pct': None,
            'std_20d_pct': None,
            'max_drawdown_pct': None,
        }


def _build_backtest_metrics(history_rows):
    """Convert evaluated prediction history into trading-quality metrics."""
    evaluated = []
    for item in history_rows or []:
        if item.get('actual_price') is None or item.get('current_price') is None:
            continue
        try:
            base_price = float(item.get('current_price') or 0.0)
            actual_price = float(item.get('actual_price') or 0.0)
            predicted_return = float(item.get('predicted_return') or 0.0)
        except (TypeError, ValueError):
            continue
        if base_price <= 0:
            continue
        actual_return = (actual_price / base_price) - 1.0
        signed_return = actual_return if predicted_return >= 0 else -actual_return
        evaluated.append({
            'signed_return': signed_return,
            'correct': bool(item.get('correct')),
        })

    if len(evaluated) < 5:
        return {
            'status': 'warming_up',
            'evaluated_trades': len(evaluated),
            'win_rate': None,
            'sharpe': None,
            'max_drawdown_pct': None,
            'cagr_pct': None,
            'profit_factor': None,
        }

    returns = [item['signed_return'] for item in evaluated]
    wins = [ret for ret in returns if ret > 0]
    losses = [ret for ret in returns if ret < 0]
    mean_return = sum(returns) / len(returns)
    variance = sum((ret - mean_return) ** 2 for ret in returns) / max(1, len(returns) - 1)
    std_return = variance ** 0.5
    sharpe = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else None
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for ret in returns:
        equity *= (1.0 + ret)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, (equity / peak) - 1.0)
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(wins) > 0 else None
    cagr = (equity ** (252 / len(returns)) - 1.0) if equity > 0 else None

    return {
        'status': 'ready',
        'evaluated_trades': len(evaluated),
        'win_rate': round((len(wins) / len(returns)) * 100.0, 1),
        'sharpe': None if sharpe is None else round(float(sharpe), 2),
        'max_drawdown_pct': round(abs(float(max_drawdown)) * 100.0, 2),
        'cagr_pct': None if cagr is None else round(float(cagr) * 100.0, 1),
        'profit_factor': None if profit_factor is None else round(float(profit_factor), 2),
    }


def _build_ai_explanation(recommendation, market_regime, risk_profile, model_trust, sentiment_data):
    """Short trader-facing explanation for the current signal."""
    recommendation = recommendation or {}
    signal = str(recommendation.get('signal') or 'HOLD').upper()
    expected_move = recommendation.get('expected_move_pct')
    confidence = recommendation.get('confidence_percent')
    sentiment_score = (sentiment_data or {}).get('score')
    reasons = []
    if signal == 'HOLD':
        if expected_move is not None:
            reasons.append(
                f"HOLD because the forecast edge is {float(expected_move):+.2f}% and current risk/validation checks do not justify a directional trade."
            )
        else:
            reasons.append("HOLD because the decision engine does not see enough validated edge for a BUY or SELL.")
        if market_regime.get('label'):
            reasons.append(f"Regime is {market_regime.get('label')} with {market_regime.get('momentum_20d_pct', 'n/a')}% 20-day momentum.")
        if risk_profile.get('label'):
            reasons.append(f"Risk is {risk_profile.get('label')} from ATR, realized volatility, and max drawdown.")
        if sentiment_score is not None:
            tone = 'bullish' if float(sentiment_score) > 0.2 else 'bearish' if float(sentiment_score) < -0.2 else 'neutral'
            reasons.append(f"News sentiment is {tone} at {float(sentiment_score):.2f}, so it is treated as context instead of a trade signal.")
        return reasons[:5]
    if expected_move is not None:
        risk_multiplier = 1.0 - _clamp_number((risk_profile or {}).get('score', 45.0) / 100.0, 0.0, 0.85)
        adjusted_move = float(expected_move) * risk_multiplier
        direction = 'positive' if adjusted_move >= 0 else 'negative'
        reasons.append(f"{signal} because the model sees a {direction} {abs(adjusted_move):.2f}% risk-adjusted forecast edge.")
    else:
        reasons.append(f"{signal} because the decision engine is balancing forecast edge against current risk.")
    if market_regime.get('label'):
        reasons.append(f"Regime is {market_regime.get('label')} with {market_regime.get('momentum_20d_pct', 'n/a')}% 20-day momentum.")
    if confidence is not None:
        reasons.append(f"Forecast confidence is {float(confidence):.0f}% after calibration and MLOps checks.")
    if risk_profile.get('label'):
        reasons.append(f"Risk is {risk_profile.get('label')} from ATR, realized volatility, and max drawdown.")
    if sentiment_score is not None:
        tone = 'supportive' if float(sentiment_score) >= 0 else 'negative'
        reasons.append(f"News sentiment is {tone} at {float(sentiment_score):.2f}.")
    if model_trust.get('score', 0) < 55:
        reasons.append("Treat this as a smaller-size setup until more evaluated predictions improve trust.")
    return reasons[:5]


def _build_user_trust_payload(recommendation, v2_payload, model_health, market_regime, risk_profile):
    """Convert ML/MLOps signals into a trader-friendly trust score."""
    recommendation = recommendation or {}
    v2_payload = v2_payload or {}
    model_health = model_health or {}
    components = recommendation.get('components') or {}

    confidence = _clamp_number(recommendation.get('confidence', v2_payload.get('confidence', 0.0)))
    model_quality = _clamp_number(components.get('model_quality', 0.55), 0.0, 1.0)
    drift_score = _clamp_number(v2_payload.get('drift_score', 0.0), 0.0, 1.0)
    health_accuracy = model_health.get('recent_accuracy')
    if health_accuracy is None:
        health_component = 0.55
    else:
        health_component = _clamp_number(health_accuracy, 0.0, 1.0)

    risk_score = _clamp_number((risk_profile or {}).get('score', 45.0) / 100.0, 0.0, 1.0)
    regime_penalty = 0.10 if market_regime.get('tone') == 'warning' else 0.0
    risk_penalty = risk_score * 0.18
    trust = (
        0.34 * confidence
        + 0.28 * model_quality
        + 0.22 * health_component
        + 0.16 * (1.0 - drift_score)
        - regime_penalty
        - risk_penalty
    )
    trust_score = int(round(_clamp_number(trust, 0.0, 1.0) * 100))

    if trust_score >= 75:
        label = 'High Trust'
    elif trust_score >= 55:
        label = 'Moderate Trust'
    elif trust_score >= 35:
        label = 'Low Trust'
    else:
        label = 'Training Trust'

    reasons = []
    if health_accuracy is None:
        evaluated_count = int(model_health.get('evaluated_count') or 0)
        min_evaluations = int(model_health.get('min_evaluations') or 5)
        reasons.append(f'Reliability monitoring is warming up: {evaluated_count}/{min_evaluations} completed forecast evaluations')
    if drift_score >= 0.35:
        reasons.append('Feature drift is elevated')
    if market_regime.get('tone') == 'warning':
        reasons.append('High volatility regime')
    if (risk_profile or {}).get('label') in ('High', 'Speculative'):
        reasons.append(f"{risk_profile.get('label')} risk from volatility and drawdown")
    if confidence < 0.45:
        reasons.append('Model confidence is limited')
    if not reasons:
        reasons.append('Confidence, validation, drift, and recent accuracy are acceptable')

    return {
        'score': trust_score,
        'label': label,
        'confidence_component': round(confidence, 4),
        'model_quality_component': round(model_quality, 4),
        'recent_accuracy_component': None if health_accuracy is None else round(float(health_accuracy), 4),
        'drift_score': round(drift_score, 4),
        'risk_component': round(float(risk_score), 4),
        'reasons': reasons[:3],
    }


def _normalize_prediction_days(raw_days) -> int:
    """Keep forecasts on the production-supported horizons: 1, 7, or 14 days."""
    try:
        requested = int(raw_days)
    except (TypeError, ValueError):
        return 7
    if requested <= 1:
        return 1
    if requested <= 7:
        return 7
    return 14


def _parse_env_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(',') if item.strip()]


def _update_ticker_training_job(ticker: str, *, state: str, stage: str, progress: int, message: str = '') -> None:
    ticker_key = (ticker or '').strip().upper()
    if not ticker_key:
        return
    payload = {
        'ticker': ticker_key,
        'state': state,
        'stage': stage,
        'progress': max(0, min(100, int(progress))),
        'message': message,
        'updated_at': datetime.utcnow().isoformat() + 'Z'
    }
    with _ticker_training_jobs_lock:
        existing = _ticker_training_jobs.get(ticker_key, {})
        payload['started_at'] = existing.get('started_at') or datetime.utcnow().isoformat() + 'Z'
        _ticker_training_jobs[ticker_key] = payload


def _get_ticker_training_job(ticker: str) -> dict:
    ticker_key = (ticker or '').strip().upper()
    with _ticker_training_jobs_lock:
        return dict(_ticker_training_jobs.get(ticker_key, {}))


def _start_ticker_model_training(ticker: str, *, force: bool = False) -> tuple[bool, dict]:
    """Start a fresh ticker-specific training job and expose progress through model-status."""
    ticker_key = (ticker or '').strip().upper()
    if not ticker_key:
        return False, {'error': 'Ticker is required'}
    if not _is_trainable_market_ticker(ticker_key):
        _update_ticker_training_job(
            ticker_key,
            state='untrainable',
            stage='unsupported_symbol',
            progress=0,
            message=_unsupported_training_message(ticker_key)
        )
        return False, _build_model_status(ticker_key)

    with _ultimate_training_state_lock:
        if ticker_key in _ultimate_training_in_progress and not force:
            return False, _build_model_status(ticker_key)
        if ticker_key in _ultimate_training_in_progress and force:
            return False, _build_model_status(ticker_key)
        _ultimate_training_in_progress.add(ticker_key)

    _ultimate_prediction_cache.pop(ticker_key, None)
    try:
        from unified_engine.inference import clear_cache as clear_unified_cache
        clear_unified_cache(ticker_key)
    except Exception:
        pass
    _update_ticker_training_job(
        ticker_key,
        state='queued',
        stage='queued',
        progress=8,
        message='Training request accepted. Preparing a fresh stock-specific model.'
    )

    def background_train():
        try:
            print(f"Background retry training started for {ticker_key}", flush=True)
            _update_ticker_training_job(
                ticker_key,
                state='training',
                stage='market_data_validation',
                progress=18,
                message='Fetching and validating market history for this ticker.'
            )

            from unified_engine.training import train_unified_model
            _update_ticker_training_job(
                ticker_key,
                state='training',
                stage='unified_model_training',
                progress=38,
                message='Training the unified ensemble and forecast range model.'
            )
            train_result = train_unified_model(ticker_key)
            if train_result and train_result.success:
                _publish_training_metrics(ticker_key, train_result.metrics)
                _update_ticker_training_job(
                    ticker_key,
                    state='training',
                    stage='validation_observability',
                    progress=82,
                    message='Validating predictions and publishing monitoring artifacts.'
                )
                _run_v2_observability_training(ticker_key)
                _update_ticker_training_job(
                    ticker_key,
                    state='ready',
                    stage='custom_model_ready',
                    progress=100,
                    message='Model is ready. Forecasts now use trained ticker-specific ML artifacts and live market features.'
                )
                print(f"Background retry training completed for {ticker_key}", flush=True)
            else:
                reason = train_result.reason if train_result else 'unknown training failure'
                failure_state, failure_stage, failure_message = _training_failure_status(reason)
                _update_ticker_training_job(
                    ticker_key,
                    state=failure_state,
                    stage=failure_stage,
                    progress=0,
                    message=failure_message
                )
                print(f"Background retry training failed for {ticker_key}: {reason}", flush=True)
        except Exception as train_err:
            failure_state, failure_stage, failure_message = _training_failure_status(train_err)
            _update_ticker_training_job(
                ticker_key,
                state=failure_state,
                stage=failure_stage,
                progress=0,
                message=failure_message
            )
            print(f"Background retry training failed for {ticker_key}: {train_err}", flush=True)
        finally:
            with _ultimate_training_state_lock:
                _ultimate_training_in_progress.discard(ticker_key)

    threading.Thread(target=background_train, daemon=True).start()
    return True, _build_model_status(ticker_key)


def _ticker_has_trained_model(ticker: str) -> bool:
    ticker_key = (ticker or '').strip().upper()
    if not ticker_key:
        return False
    try:
        from unified_engine.inference import UnifiedPredictor
        if UnifiedPredictor.is_model_available(ticker_key):
            return True
    except Exception:
        pass
    try:
        ultimate_path = os.path.join(
            'mlops',
            'ultimate_engine',
            ticker_key.replace('/', '_').replace('\\', '_'),
            'model.joblib'
        )
        if os.path.exists(ultimate_path):
            return True
    except Exception:
        pass
    try:
        from mlops.registry import ModelRegistry
        best = ModelRegistry().get_best_model(ticker_key)
        if best and os.path.exists(best.get('model_path', '')):
            return True
    except Exception:
        pass
    return False


def _build_model_status(ticker: str, *, prediction_ready: bool | None = None) -> dict:
    ticker_key = (ticker or '').strip().upper()
    trained_model_ready = _ticker_has_trained_model(ticker_key)
    job = _get_ticker_training_job(ticker_key)
    in_memory_training = ticker_key in _training_in_progress or ticker_key in _ultimate_training_in_progress

    started_at = job.get('started_at') if job else None
    elapsed_seconds = 0
    if started_at:
        try:
            elapsed_seconds = max(0, int((datetime.utcnow() - datetime.fromisoformat(started_at.replace('Z', ''))).total_seconds()))
        except Exception:
            elapsed_seconds = 0

    if trained_model_ready or prediction_ready:
        state = 'ready'
        stage = 'custom_model_ready'
        progress = 100
        message = 'Model is ready. Forecasts now use trained ticker-specific ML artifacts and live market features.'
    elif job:
        state = job.get('state', 'training')
        stage = job.get('stage', 'training')
        progress = int(job.get('progress', 25))
        message = job.get('message') or 'Please wait a little while. We are training this ticker model and will refresh the forecast automatically when it is ready.'
    elif in_memory_training:
        state = 'training'
        stage = 'background_training'
        progress = 35
        message = 'Please wait a little while. The dedicated stock model is training in the background and this screen will update automatically.'
    else:
        state = 'preliminary'
        stage = 'not_trained'
        progress = 0
        message = 'No trained model exists for this stock yet. Showing live market analysis now while the custom model is prepared.'

    terminal_without_training = state in ('failed', 'untrainable', 'unsupported')
    eta_seconds = 0 if (trained_model_ready or prediction_ready or terminal_without_training) else max(45, min(360, int(240 * (1 - progress / 100.0)) - elapsed_seconds))
    eta_label = 'Ready now' if eta_seconds == 0 else f"about {max(1, round(eta_seconds / 60))} min"

    return {
        'ticker': ticker_key,
        'state': state,
        'stage': stage,
        'progress': progress,
        'model_ready': bool(trained_model_ready or prediction_ready),
        'is_training': state in ('queued', 'training'),
        'analysis_mode': 'custom_model' if (trained_model_ready or prediction_ready) else 'preliminary',
        'message': message,
        'estimated_seconds_remaining': eta_seconds,
        'estimated_completion_label': eta_label,
        'started_at': started_at,
        'elapsed_seconds': elapsed_seconds,
        'updated_at': job.get('updated_at') if job else datetime.utcnow().isoformat() + 'Z'
    }


def _metric_as_percent(value) -> float:
    try:
        numeric = float(value or 0.0)
        return numeric * 100.0 if 0.0 <= numeric <= 1.0 else numeric
    except Exception:
        return 0.0


def _publish_training_metrics(ticker: str, metrics: dict | None) -> None:
    """Push the latest training metrics into Prometheus/Grafana gauges."""
    metrics = metrics or {}
    try:
        from mlops_v2.monitoring import set_accuracy_20d, set_drift_score, set_sharpe_ratio, set_simulated_pnl

        accuracy = metrics.get('accuracy', metrics.get('directional_accuracy', metrics.get('xgb_accuracy', 0.0)))
        set_accuracy_20d(ticker, _metric_as_percent(accuracy))

        if 'drift_score' in metrics:
            set_drift_score(ticker, float(metrics.get('drift_score') or 0.0))
        if 'simulated_pnl' in metrics:
            set_simulated_pnl(ticker, float(metrics.get('simulated_pnl') or 0.0))
        if 'sharpe_ratio' in metrics:
            set_sharpe_ratio(ticker, float(metrics.get('sharpe_ratio') or 0.0))
    except Exception as monitoring_err:
        print(f"Failed to update monitoring gauges for {ticker}: {monitoring_err}")


def _run_v2_observability_training(ticker: str) -> None:
    """
    Train the v2 observability path for unseen tickers.
    This writes DVC-friendly raw-data manifests, logs MLflow/DagsHub runs,
    saves v2 model metadata, and refreshes Grafana gauges.
    """
    try:
        from mlops_v2.training import TrainerV2

        result = TrainerV2().train_if_needed(ticker=ticker, force=True)
        metrics = dict(result.metrics or {})
        metrics['drift_score'] = result.drift_score
        _publish_training_metrics(ticker, metrics)
        if result.trained:
            print(f"✅ V2 observability training completed for {ticker}; run_id={result.run_id or 'local'}")
        else:
            print(f"⚠️ V2 observability training skipped for {ticker}: {result.reason}")
    except Exception as obs_err:
        print(f"⚠️ V2 observability training failed for {ticker}: {obs_err}")


def configure_logging(flask_app: Flask) -> None:
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    os.makedirs('logs', exist_ok=True)
    handler = RotatingFileHandler('logs/app.log', maxBytes=1_048_576, backupCount=3)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
    flask_app.logger.addHandler(handler)
    flask_app.logger.setLevel(log_level)
    logging.getLogger('werkzeug').setLevel(log_level)


def initialize_firebase_admin() -> bool:
    if firebase_admin is None or credentials is None:
        return False

    if firebase_admin._apps:  # type: ignore[attr-defined]
        return True

    # Preferred for container platforms: full JSON in env var
    service_account_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
    if service_account_json:
        try:
            cred_dict = json.loads(service_account_json)
            firebase_admin.initialize_app(credentials.Certificate(cred_dict))
            return True
        except Exception:
            pass

    # Alternate format: base64-encoded JSON in env var
    service_account_b64 = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_B64')
    if service_account_b64:
        try:
            decoded = base64.b64decode(service_account_b64).decode('utf-8')
            cred_dict = json.loads(decoded)
            firebase_admin.initialize_app(credentials.Certificate(cred_dict))
            return True
        except Exception:
            pass

    # Local/dev fallback from filesystem path
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
    if service_account_path and os.path.exists(service_account_path):
        firebase_admin.initialize_app(credentials.Certificate(service_account_path))
        return True

    return False


print("=" * 60)
print("Stock Predictor App - Using Twelve Data API")
print("=" * 60)

# Start automatic model training scheduler
try:
    from scheduler import scheduler
    scheduler.start()
    startup_catchup = os.getenv('ENABLE_STARTUP_CATCHUP', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
    catchup_text = "with startup catch-up" if startup_catchup else "startup catch-up disabled"
    print(f"Automatic model training started (runs daily; {catchup_text})")
except Exception as e:
    print(f"Could not start training scheduler: {e}")

# Configure Flask to serve React build (optimized)
os.makedirs('cache', exist_ok=True)
application = Flask(__name__, static_folder='build', static_url_path='/static_bypass')
app = application

# --- Database & Admin Setup ---
from database import init_db, db_session
init_db()
from admin_api import admin_bp
app.register_blueprint(admin_bp)

@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

# Optional startup seed for first-run local demos. Disabled by default in production.
try:
    from models import ActiveTicker
    seed_on_startup = os.getenv('SEED_INITIAL_STOCKS_ON_STARTUP', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
    if seed_on_startup and db_session.query(ActiveTicker).count() == 0:
        initial_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        for s in initial_stocks:
            db_session.add(ActiveTicker(ticker=s, is_active=True))
        db_session.commit()
        print(f"Seeded {len(initial_stocks)} initial stocks into database.")
except Exception as e:
    print(f"Could not seed database: {e}")
    db_session.rollback()

print("SQLAlchemy Database & Admin API initialized.")

configure_logging(app)

# Allow CORS from all origins for API endpoints
# Allow CORS from all origins for local development to prevent connectivity issues
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/chatbot/*": {"origins": "*"}}, supports_credentials=True)

firebase_ready = initialize_firebase_admin()
if firebase_ready:
    app.logger.info('Firebase Admin SDK initialised successfully.')
else:
    app.logger.warning('Firebase Admin SDK not configured. Admin-only endpoints are disabled.')

ADMIN_EMAILS = {email.lower() for email in _parse_env_list(os.getenv('ADMIN_EMAILS'), [])}


def _user_has_admin_privileges(claims: dict) -> bool:
    if not claims:
        return False

    if claims.get('admin') is True:
        return True

    email = (claims.get('email') or '').lower()
    return bool(email and email in ADMIN_EMAILS)


def firebase_auth_required(admin_only: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if firebase_auth is None:
                app.logger.warning('Attempted to access a secured endpoint without Firebase Admin configured.')
                return jsonify({'success': False, 'error': 'Auth service unavailable'}), 503

            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return jsonify({'success': False, 'error': 'Missing authentication token'}), 401

            token = auth_header.split(' ', 1)[1].strip()
            try:
                decoded = firebase_auth.verify_id_token(token, check_revoked=True)
            except Exception as exc:  # pylint: disable=broad-except
                app.logger.info('Invalid or expired token')
                return jsonify({'success': False, 'error': 'Invalid or expired token'}), 401

            if admin_only and not _user_has_admin_privileges(decoded):
                return jsonify({'success': False, 'error': 'Insufficient privileges'}), 403

            g.firebase_user = decoded
            
            # Enterprise: Auto-register user in DB if they don't exist
            try:
                from models import User
                email = decoded.get('email')
                uid = decoded.get('uid')
                if uid:
                    user = db_session.query(User).filter_by(firebase_uid=uid).first()
                    if not user:
                        role = "admin" if email and email.lower() in ADMIN_EMAILS else "user"
                        user = User(firebase_uid=uid, email=email, role=role)
                        db_session.add(user)
                        db_session.commit()
            except Exception as db_err:
                app.logger.error(f"Failed to auto-register user: {db_err}")
                db_session.rollback()

            return func(*args, **kwargs)

        return wrapper

    return decorator


@app.route('/api/auth/sync', methods=['POST'])
@firebase_auth_required(admin_only=False)
def sync_user():
    """Endpoint to explicitly sync/register user in the SQL database."""
    return jsonify({"success": True, "message": "User synchronized successfully."})


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    app.logger.warning('HTTP error: %s', exc.description)
    response = exc.get_response()
    response.data = jsonify({'success': False, 'error': exc.description}).data
    response.content_type = 'application/json'
    return response


@app.errorhandler(Exception)
def handle_unexpected_exception(exc: Exception):  # pylint: disable=broad-except
    app.logger.exception('Unhandled server error: %s', exc)
    return jsonify({'success': False, 'error': 'An unexpected server error occurred.'}), 500

def load_lstm_model(ticker):
    """Load the pre-trained LSTM model for a specific ticker (per-ticker cache)"""
    global _model_cache, _model_metadata
    ticker = (ticker or '').strip().upper()
    if not ticker:
        return None
    
    # Registering every searched symbol into the scheduled training list can
    # quickly fill MLflow with old one-off stocks. Keep that opt-in.
    try:
        from mlops.config import MLOpsConfig
        auto_add_to_schedule = os.getenv('AUTO_ADD_SEARCHED_TICKERS_TO_TRAINING', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
        if auto_add_to_schedule:
            MLOpsConfig.add_stock(ticker)
        # Also sync to ActiveTicker DB for admin dashboard visibility
        try:
            from models import ActiveTicker
            existing = db_session.query(ActiveTicker).filter_by(ticker=ticker).first()
            if not existing:
                db_session.add(ActiveTicker(ticker=ticker, is_active=True))
                db_session.commit()
        except Exception:
            db_session.rollback()
    except Exception as e:
        print(f"Failed to register ticker {ticker}: {e}")
    
    # Return cached model if already loaded for this ticker
    if ticker in _model_cache:
        return _model_cache[ticker]
    
    loaded_model = None

    def trigger_background_training():
        if not _is_trainable_market_ticker(ticker):
            _update_ticker_training_job(
                ticker,
                state='untrainable',
                stage='unsupported_symbol',
                progress=0,
                message=_unsupported_training_message(ticker)
            )
            return
        with _training_state_lock:
            should_start_training = ticker not in _training_in_progress
            if should_start_training:
                _training_in_progress.add(ticker)
                _update_ticker_training_job(
                    ticker,
                    state='queued',
                    stage='queued',
                    progress=8,
                    message='Training request accepted. Market data remains available while the model is prepared.'
                )

        if not should_start_training:
            print(f"⏳ Training already in progress for {ticker}. Skipping duplicate trigger.")
            return

        print(f"🚀 No model for {ticker}. Triggering background training...")

        def background_train():
            try:
                # Train Unified Engine v4.0 FIRST (primary prediction path)
                try:
                    from unified_engine.training import UnifiedTrainer
                    _update_ticker_training_job(
                        ticker,
                        state='training',
                        stage='unified_engine_training',
                        progress=25,
                        message='Training unified ensemble with walk-forward validation and calibration.'
                    )
                    print(f"Background Unified Engine v4.0 training started for {ticker}")
                    ue_result = UnifiedTrainer.train(ticker)
                    if ue_result.success:
                        _update_ticker_training_job(
                            ticker,
                            state='training',
                            stage='unified_engine_complete',
                            progress=55,
                            message='Unified ensemble trained. Preparing sequence model and monitoring artifacts.'
                        )
                        print(f"Unified Engine v4.0 completed for {ticker}: "
                              f"accuracy={ue_result.metrics.get('accuracy', 0):.1f}%")
                    else:
                        print(f"Unified Engine v4.0 skipped for {ticker}: {ue_result.reason}")
                except Exception as unified_err:
                    print(f"Unified Engine v4.0 error for {ticker}: {unified_err}")

                # Train V1 (powers multi-day LSTM charting & prediction endpoints)
                try:
                    from mlops.training_pipeline import MLOpsTrainingPipeline
                    pipeline = MLOpsTrainingPipeline()
                    _update_ticker_training_job(
                        ticker,
                        state='training',
                        stage='lstm_training',
                        progress=65,
                        message='Training ticker-specific LSTM forecast model.'
                    )
                    print(f"🚀 Background V1 training started for {ticker}")
                    pipeline.train_model(ticker=ticker, epochs=20, days=730)
                    print(f"✅ Background V1 training completed for {ticker}")
                except Exception as v1_err:
                    print(f"⚠️ Background V1 training failed for {ticker}: {v1_err}")

                # Train V2 (powers inference metrics and drift detection)
                _update_ticker_training_job(
                    ticker,
                    state='training',
                    stage='calibration_monitoring',
                    progress=82,
                    message='Building confidence intervals, drift checks, and model metadata.'
                )
                from mlops_v2.training import TrainerV2
                trainer = TrainerV2()
                result = trainer.train_if_needed(ticker=ticker, force=True)
                metrics = dict(result.metrics or {})
                metrics['drift_score'] = result.drift_score
                _publish_training_metrics(ticker, metrics)
                _update_ticker_training_job(
                    ticker,
                    state='ready',
                    stage='custom_model_ready',
                    progress=100,
                    message='Model is ready. Forecasts now use trained ticker-specific ML artifacts and live market features.'
                )
                if result.trained:
                    print(f"✅ Background v2 training completed for {ticker}")
                else:
                    print(f"⚠️ Background v2 training skipped for {ticker}: {result.reason}")
            except Exception as e:
                print(f"❌ Background training failed for {ticker}: {e}")
                failure_state, failure_stage, failure_message = _training_failure_status(e)
                _update_ticker_training_job(
                    ticker,
                    state=failure_state,
                    stage=failure_stage,
                    progress=0,
                    message=failure_message
                )
            finally:
                with _training_state_lock:
                    _training_in_progress.discard(ticker)

        def delayed_background_train():
            import time
            print(f"⏳ Waiting 10 seconds before starting background training for {ticker} to allow API response...")
            time.sleep(10)
            background_train()

        threading.Thread(target=delayed_background_train, daemon=True).start()
    
    try:
        # FIRST: Try loading from MLOps registry (best versioned model for this ticker)
        try:
            from mlops.registry import ModelRegistry
            registry = ModelRegistry()
            best = registry.get_best_model(ticker)
            if best and os.path.exists(best.get('model_path', '')):
                print(f"Loading model from registry for {ticker}: {best.get('version_id', best.get('version', 'unknown'))}")
                from tensorflow.keras.models import load_model as keras_load
                loaded_model = keras_load(best['model_path'])
                _model_cache[ticker] = loaded_model
                
                # Cache MLOps metadata for health check
                _model_metadata[ticker] = {
                    'version': best.get('version', 'unknown'),
                    'run_id': best.get('run_id', 'unknown'),
                    'source': best.get('source', 'local'),
                    'loaded_at': datetime.now().isoformat()
                }
                
                print(f"✅ Registry model loaded for {ticker} (v{best['version']}) from {best.get('source', 'local')}")
                
                # Also cache the scaler
                if best.get('scaler_type') == 'unified_library':
                    _scaler_cache[ticker] = registry.get_scaler_from_library(ticker)
                    print(f"✅ Registry scaler (Unified) cached for {ticker}")
                else:
                    reg_scaler_path = best.get('scaler_path', '')
                    if reg_scaler_path and os.path.exists(reg_scaler_path):
                        import pickle
                        with open(reg_scaler_path, 'rb') as f:
                            _scaler_cache[ticker] = pickle.load(f)
                        print(f"✅ Registry scaler (Path) cached for {ticker}")
                
                _model_cache[ticker] = loaded_model
                return loaded_model
            else:
                trigger_background_training()
                
        except Exception as e:
            print(f"Registry lookup failed for {ticker}: {e}")

        if loaded_model is None:
            # Never use global fallback artifacts for arbitrary tickers.
            trigger_background_training()
            print(f"⚠️ No trained model available yet for {ticker}")
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
    
    return loaded_model

def load_saved_scaler(ticker):
    """Load the scaler saved during training for a specific ticker"""
    global _scaler_cache
    
    # Return cached scaler
    if ticker in _scaler_cache:
        return _scaler_cache[ticker]
    
    try:
        # 1. Try MLOps Unified Library first (Zero-Clutter system)
        try:
            from mlops.registry import ModelRegistry
            registry = ModelRegistry()
            scaler = registry.get_scaler_from_library(ticker)
            if scaler:
                _scaler_cache[ticker] = scaler
                return scaler
        except Exception:
            pass

        # 2. Try ticker-specific scaler artifact
        ticker_scaler_path = f'artifacts/{ticker}_scaler.pkl'
        if os.path.exists(ticker_scaler_path):
            import joblib
            scaler = joblib.load(ticker_scaler_path)
            _scaler_cache[ticker] = scaler
            return scaler
            
        # Intentionally no global scaler fallback for arbitrary tickers.
    except Exception as e:
        print(f"Could not load scaler for {ticker}: {e}")
    return None

# Health check endpoint (minimal dependencies)
@app.route('/health')
@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    # Also check if chatbot server is reachable
    chatbot_status = 'unknown'
    try:
        import requests as _requests
        r = _requests.get('http://127.0.0.1:8001/', timeout=3)
        chatbot_status = 'healthy' if r.status_code == 200 else f'error ({r.status_code})'
    except Exception as e:
        chatbot_status = f'unreachable ({type(e).__name__})'
    return jsonify({
        'status': 'healthy',
        'chatbot': chatbot_status,
        'version': '1.0.1-safety-gate',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/mlops/status')
def mlops_status():
    """Get status of active ML models and their MLOps metadata"""
    return jsonify({
        'success': True,
        'active_models': _model_metadata,
        'mlflow_tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/mlops/registry')
def mlops_unified_registry():
    """Return latest Unified Engine registry records and promotion status."""
    try:
        from unified_engine.model_registry import list_model_records

        latest = {}
        for record in list_model_records():
            ticker = record.get('ticker')
            if ticker and ticker not in latest:
                latest[ticker] = record
        return jsonify({
            'success': True,
            'count': len(latest),
            'models': list(latest.values()),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/mlops/registry/<ticker>')
def mlops_unified_registry_ticker(ticker):
    """Return the production or latest registry record for one ticker."""
    try:
        from unified_engine.model_registry import get_latest_or_production_record

        record = get_latest_or_production_record(ticker)
        if not record:
            return jsonify({
                'success': False,
                'ticker': ticker.upper(),
                'error': 'No registered Unified Engine model found for this ticker.'
            }), 404
        return jsonify({
            'success': True,
            'ticker': ticker.upper(),
            'model': record,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as exc:
        return jsonify({'success': False, 'ticker': ticker.upper(), 'error': str(exc)}), 500


@app.route('/api/model-status/<ticker>')
def ticker_model_status(ticker):
    """Return per-ticker model readiness and background training progress."""
    try:
        status = _build_model_status(ticker)
        return jsonify({
            'success': True,
            'ticker': status['ticker'],
            'model_status': status,
        })
    except Exception as exc:
        return jsonify({
            'success': False,
            'ticker': ticker.upper(),
            'error': str(exc)
        }), 500


@app.route('/api/model-status/<ticker>/retry', methods=['POST'])
def retry_ticker_model_training(ticker):
    """Clear a failed ticker job and start a fresh stock-specific training run."""
    try:
        started, status = _start_ticker_model_training(ticker, force=True)
        return jsonify({
            'success': True,
            'started': started,
            'ticker': status.get('ticker', ticker.upper()),
            'model_status': status,
            'message': 'Fresh ticker training started.' if started else 'Ticker training is already running.'
        })
    except Exception as exc:
        return jsonify({
            'success': False,
            'ticker': ticker.upper(),
            'error': str(exc)
        }), 500


# Cache management endpoints
@app.route('/api/cache/stats')
@firebase_auth_required(admin_only=True)
def cache_stats():
    """Get cache statistics (admin only)"""
    from cache_manager import get_cache
    cache_mgr = get_cache()
    stats = cache_mgr.get_stats()
    return jsonify({
        'success': True,
        'cache_stats': stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/cache/clear', methods=['POST'])
@firebase_auth_required(admin_only=True)
def clear_cache():
    """Clear all cache (admin only)"""
    from cache_manager import get_cache
    cache_mgr = get_cache()
    removed = cache_mgr.clear_all()
    return jsonify({
        'success': True,
        'message': f'Cleared {removed} cache files',
        'files_removed': removed
    })

@app.route('/api/cache/cleanup', methods=['POST'])
def cleanup_expired_cache():
    """Clean up expired cache files (public endpoint, runs automatically)"""
    from cache_manager import get_cache
    cache_mgr = get_cache()
    removed = cache_mgr.clear_expired()
    return jsonify({
        'success': True,
        'message': f'Cleaned up {removed} expired cache files',
        'files_removed': removed
    })

# Training health endpoint — keeps Hugging Face Spaces awake and shows training diagnostics
@app.route('/api/health/training')
def training_health_check():
    """Health check with training status — useful for external uptime monitors"""
    try:
        from scheduler import scheduler
        status = scheduler.get_status()
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'scheduler_running': status.get('is_running', False),
            'last_training': status.get('last_training'),
            'next_training': status.get('next_training'),
        })
    except Exception as e:
        return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat(), 'scheduler_error': str(e)})


@app.route('/metrics')
def prometheus_metrics():
    """Expose Prometheus metrics for monitoring stack."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # type: ignore
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    except Exception:
        return "prometheus_client not installed", 501

# Model Training & Status Endpoints

@app.route('/api/models/training-status')
def get_training_status():
    """Get model training scheduler status"""
    try:
        from scheduler import scheduler
        status = scheduler.get_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/train', methods=['POST'])
def trigger_manual_training():
    """Manually trigger model training"""
    try:
        from scheduler import scheduler
        
        # Run training in background
        import threading
        def train_async():
            scheduler.train_models_job()
        
        thread = threading.Thread(target=train_async, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Model training started in background'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



# EXPLICIT React SPA Routes - Handle each React page route explicitly
# This ensures page refreshes work correctly for all client-side routes

@app.route('/prediction')
@app.route('/prediction/')
def prediction_page():
    """Serve React app for prediction page"""
    response = send_from_directory(app.static_folder, 'index.html')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

@app.route('/about')
@app.route('/about/')
def about_page():
    """Serve React app for about page"""
    response = send_from_directory(app.static_folder, 'index.html')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

@app.route('/profile')
@app.route('/profile/')
def profile_page():
    """Serve React app for profile/account page"""
    response = send_from_directory(app.static_folder, 'index.html')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

@app.route('/login')
@app.route('/login/')
def login_page():
    """Serve React app for login page"""
    response = send_from_directory(app.static_folder, 'index.html')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

@app.route('/register')
@app.route('/register/')
def register_page():
    """Serve React app for register page"""
    response = send_from_directory(app.static_folder, 'index.html')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

@app.route('/forgot-password')
@app.route('/forgot-password/')
def forgot_password_page():
    """Serve React app for forgot password page"""
    response = send_from_directory(app.static_folder, 'index.html')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

@app.route('/change-password')
@app.route('/change-password/')
def change_password_page():
    """Serve React app for change password page"""
    response = send_from_directory(app.static_folder, 'index.html')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

@app.route('/dashboard')
@app.route('/dashboard/')
def dashboard_page():
    """Serve React app for dashboard page"""
    response = send_from_directory(app.static_folder, 'index.html')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

@app.route('/portfolio')
@app.route('/portfolio/')
def portfolio_page():
    """Serve React app for portfolio page"""
    response = send_from_directory(app.static_folder, 'index.html')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

@app.route('/watchlist')
@app.route('/watchlist/')
def watchlist_page():
    """Serve React app for watchlist page"""
    response = send_from_directory(app.static_folder, 'index.html')
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response


# API Routes
@app.route('/api/search')
def search_tickers():
    """Search tickers across global exchanges."""
    # Lazy import stock_api
    from stock_api import search_symbols
    
    query = request.args.get('q', default='', type=str).strip()
    limit = request.args.get('limit', default=5, type=int)
    limit = max(1, min(limit, 10))

    if not query:
        return jsonify({'success': True, 'results': []})

    try:
        results = search_symbols(query, limit)
        return jsonify({'success': True, 'results': results})
    except Exception as exc:
        print(f"Error in search endpoint: {exc}")
        return jsonify({'success': False, 'results': [], 'error': 'Search failed'}), 500

@app.route('/api/stock/<ticker>')
def get_stock_data(ticker):
    """Get comprehensive stock data with prediction and profit/loss analysis"""
    sentiment_data = {}
    # Lazy import heavy dependencies
    import pandas as pd
    import numpy as np
    import ta
    from stock_api import (
        get_stock_history, 
        get_intraday_data,
        get_company_profile,
        get_company_metrics,
        get_quote_data,
        search_symbols
    )
    
    days = _normalize_prediction_days(request.args.get('days', default=7))
    
    try:
        requested_ticker = ticker.upper()
        
        # Optional: Try to identify user for logging even if auth is not strictly required
        auth_header = request.headers.get('Authorization', '')
        user_id = None
        if auth_header.startswith('Bearer '):
            try:
                token = auth_header.split(' ', 1)[1].strip()
                decoded = firebase_auth.verify_id_token(token)
                uid = decoded.get('uid')
                from models import User
                user = db_session.query(User).filter_by(firebase_uid=uid).first()
                if user:
                    user_id = user.id
            except Exception:
                pass

        resolved_ticker = requested_ticker
        resolved_exchange = None
        resolved_country = None
        data_provider = None
        provider_message = None

        candidates = [{'symbol': requested_ticker, 'exchange': None, 'country': None}]
        seen_candidates = {(requested_ticker, None)}

        try:
            search_results = search_symbols(requested_ticker, limit=5)
        except Exception as search_exc:
            print(f"Symbol search fallback warning: {search_exc}")
            search_results = []

        for match in search_results:
            symbol = (match.get('symbol') or '').upper()
            exchange = (match.get('exchange') or '').upper() or None
            country = match.get('country') or None
            if not symbol:
                continue
            key = (symbol, exchange)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidates.append({
                'symbol': symbol,
                'exchange': exchange,
                'country': country
            })

        hist = pd.DataFrame()
        history_info = {}

        for candidate in candidates:
            candidate_hist, info = get_stock_history(
                candidate['symbol'],
                days=900,
                exchange=candidate.get('exchange'),
                country=candidate.get('country'),
                return_info=True
            )

            if not candidate_hist.empty and len(candidate_hist) >= 2:
                hist = candidate_hist
                history_info = info or {}
                resolved_ticker = info.get('symbol', candidate['symbol']).upper()
                resolved_exchange = (candidate.get('exchange') or '').upper() or resolved_exchange
                resolved_country = (candidate.get('country') or '').upper() or resolved_country
                data_provider = info.get('source')
                provider_message = info.get('provider_message')
                break

            if not provider_message and info.get('provider_message'):
                provider_message = info['provider_message']

        if resolved_exchange is None:
            for suffix, (exchange_name, country_name) in SUFFIX_EXCHANGE_COUNTRY.items():
                if resolved_ticker.endswith(suffix):
                    resolved_exchange = exchange_name
                    if resolved_country is None:
                        resolved_country = country_name
                    break
        elif resolved_country is None:
            for _, (exchange_name, country_name) in SUFFIX_EXCHANGE_COUNTRY.items():
                if resolved_exchange == exchange_name:
                    resolved_country = country_name
                    break

        # Validate data
        if hist.empty or len(hist) < 2:
            suggestion_candidates = [c.get('symbol') for c in candidates[1:] if c.get('symbol')]
            suggestions = [sym for sym in suggestion_candidates if sym != requested_ticker][:3]

            detail_parts = []
            # Don't expose internal API provider messages to users
            if suggestions:
                detail_parts.append(f"Try: {', '.join(suggestions)}")

            error_text = f'No data found for {requested_ticker}.'
            if detail_parts:
                error_text = f"{error_text} {' '.join(detail_parts)}"

            return jsonify({
                'success': False,
                'error': error_text.strip()
            }), 404

        # Handle multi-index columns if needed
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        # Ensure we have Close column
        if 'Close' not in hist.columns:
            return jsonify({
                'success': False,
                'error': f'Invalid data structure for {resolved_ticker}'
            }), 500

        metadata_symbols = []
        for candidate_symbol in [resolved_ticker, requested_ticker, history_info.get('symbol')]:
            if candidate_symbol and candidate_symbol.upper() not in metadata_symbols:
                metadata_symbols.append(candidate_symbol.upper())
        if not metadata_symbols and requested_ticker:
            metadata_symbols.append(requested_ticker.upper())

        print(
            "Resolved ticker: requested={}, used={}, exchange={}, provider={}".format(
                requested_ticker,
                resolved_ticker,
                resolved_exchange or 'N/A',
                data_provider or 'primary'
            )
        )
        # Persist user-demanded symbol so future scheduled runs include it.
        try:
            if not _is_trainable_market_ticker(resolved_ticker):
                raise ValueError(f"{resolved_ticker} is not a trainable public market ticker")
            from mlops.config import MLOpsConfig
            MLOpsConfig.add_stock(resolved_ticker)
            # Also sync to ActiveTicker DB for admin dashboard visibility
            try:
                from models import ActiveTicker
                existing = db_session.query(ActiveTicker).filter_by(ticker=resolved_ticker).first()
                if not existing:
                    db_session.add(ActiveTicker(ticker=resolved_ticker, is_active=True))
                    db_session.commit()
            except Exception:
                db_session.rollback()
        except Exception as add_exc:
            print(f"Ticker persistence warning for {resolved_ticker}: {add_exc}")

        # Enterprise Analytics: Log this prediction search
        try:
            from models import PredictionLog
            log_entry = PredictionLog(user_id=user_id, ticker=resolved_ticker)
            db_session.add(log_entry)
            db_session.commit()
        except Exception as log_err:
            print(f"Failed to log prediction to db: {log_err}")
            db_session.rollback()

        if provider_message:
            print(f"Provider note: {provider_message}")

        # Current price
        current_price = float(hist['Close'].iloc[-1])
        previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price

        # Calculate technical indicators
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
        hist['RSI'] = ta.momentum.RSIIndicator(hist['Close']).rsi()

        macd_indicator = ta.trend.MACD(hist['Close'])
        hist['MACD'] = macd_indicator.macd()
        hist['MACD_signal'] = macd_indicator.macd_signal()
        hist['MACD_histogram'] = macd_indicator.macd_diff()
        hist['ATR'] = ta.volatility.AverageTrueRange(
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            window=14
        ).average_true_range()
        hist['ADX'] = ta.trend.ADXIndicator(
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            window=14
        ).adx()
        bb_indicator = ta.volatility.BollingerBands(hist['Close'], window=20, window_dev=2)
        hist['BB_upper'] = bb_indicator.bollinger_hband()
        hist['BB_middle'] = bb_indicator.bollinger_mavg()
        hist['BB_lower'] = bb_indicator.bollinger_lband()
        hist['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=hist['Close'],
            volume=hist['Volume']
        ).on_balance_volume()

        hist = hist.dropna()
        if hist.empty or len(hist) < 5:
            return jsonify({
                'success': False,
                'error': f'Insufficient data for technical analysis'
            }), 500

        # Latest indicators helper
        latest_row = hist.iloc[-1]

        def safe_float(value, decimals=2):
            """Convert value to float, return None only for indicators (not OHLC)"""
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            try:
                return float(round(float(value), decimals))
            except (ValueError, TypeError):
                return None

        indicators = {
            'rsi': safe_float(latest_row.get('RSI')),
            'ema': safe_float(latest_row.get('EMA_20')),
            'macd': safe_float(latest_row.get('MACD'), 3),
            'macd_signal': safe_float(latest_row.get('MACD_signal'), 3),
            'macd_histogram': safe_float(latest_row.get('MACD_histogram'), 3),
            'sma20': safe_float(latest_row.get('SMA_20')),
            'sma50': safe_float(latest_row.get('SMA_50')),
            'atr': safe_float(latest_row.get('ATR')),
            'adx': safe_float(latest_row.get('ADX')),
            'bb_upper': safe_float(latest_row.get('BB_upper')),
            'bb_middle': safe_float(latest_row.get('BB_middle')),
            'bb_lower': safe_float(latest_row.get('BB_lower')),
            'obv': safe_float(latest_row.get('OBV'), 0)
        }

        # Predictions — pass the resolved ticker so the right model is loaded
        predictions = predict_multi_day_lstm(hist, current_price, days, ticker=resolved_ticker)

        # Company fundamentals
        company_profile = None
        profile_symbol_used = None
        for symbol_option in metadata_symbols:
            profile_candidate = get_company_profile(symbol_option)
            company_profile = profile_candidate
            profile_symbol_used = symbol_option
            if profile_candidate and (
                profile_candidate.get('market_cap') or
                profile_candidate.get('name', '').upper() != symbol_option.upper()
            ):
                break

        if company_profile is None:
            company_profile = {
                'name': resolved_ticker,
                'market_cap': None,
                'industry': 'N/A',
                'logo': '',
                'country': resolved_country or 'N/A',
                'currency': None,
                'exchange': resolved_exchange or ''
            }

        company_metrics = None
        metrics_symbol_used = None
        for symbol_option in metadata_symbols:
            metrics_candidate = get_company_metrics(symbol_option)
            company_metrics = metrics_candidate
            metrics_symbol_used = symbol_option
            if metrics_candidate and any(
                isinstance(metrics_candidate.get(key), (int, float)) and not pd.isna(metrics_candidate.get(key))
                for key in ('pe_ratio', 'eps')
            ):
                break

        if company_metrics is None:
            company_metrics = {'pe_ratio': None, 'eps': None}

        quote_data = None
        quote_symbol_used = None
        for symbol_option in metadata_symbols:
            quote_candidate = get_quote_data(symbol_option)
            if quote_candidate:
                quote_data = quote_candidate
                quote_symbol_used = symbol_option
                break

        company_name = company_profile.get('name', resolved_ticker)
        market_cap = company_profile.get('market_cap')
        if not isinstance(market_cap, (int, float)) or pd.isna(market_cap) or market_cap <= 0:
            market_cap = None
        currency = (company_profile.get('currency') or '').upper() if isinstance(company_profile, dict) else ''
        if not currency:
            country_key = (resolved_country or company_profile.get('country') or '').upper() if isinstance(company_profile, dict) else (resolved_country or '').upper()
            exchange_key = (resolved_exchange or company_profile.get('exchange') or '').upper() if isinstance(company_profile, dict) else (resolved_exchange or '').upper()
            if resolved_ticker.endswith(('.NS', '.BO')) or 'INDIA' in country_key or 'NSE' in exchange_key or 'BSE' in exchange_key:
                currency = 'INR'
            elif resolved_ticker.endswith('.L') or 'LONDON' in exchange_key or 'UNITED KINGDOM' in country_key:
                currency = 'GBP'
            elif resolved_ticker.endswith('.TO') or 'TORONTO' in exchange_key or 'CANADA' in country_key:
                currency = 'CAD'
            elif resolved_ticker.endswith('.AX') or 'AUSTRALIA' in country_key:
                currency = 'AUD'
            elif resolved_ticker.endswith('.T') or 'TOKYO' in exchange_key or 'JAPAN' in country_key:
                currency = 'JPY'
            elif resolved_ticker.endswith('.HK') or 'HONG KONG' in country_key:
                currency = 'HKD'
            else:
                currency = 'USD'
        pe_ratio = company_metrics.get('pe_ratio') if isinstance(company_metrics, dict) else None
        volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0

        if quote_data:
            quote_current = quote_data.get('current')
            quote_previous = quote_data.get('previous_close')
            if isinstance(quote_current, (int, float)) and quote_current > 0:
                current_price = float(quote_current)
            if isinstance(quote_previous, (int, float)) and quote_previous > 0:
                previous_close = float(quote_previous)

        day_high = None
        day_low = None
        if quote_data:
            quote_high = quote_data.get('high')
            quote_low = quote_data.get('low')
            if isinstance(quote_high, (int, float)) and quote_high > 0:
                day_high = float(quote_high)
            if isinstance(quote_low, (int, float)) and quote_low > 0:
                day_low = float(quote_low)
        if day_high is None and 'High' in hist.columns:
            day_high = float(hist['High'].iloc[-1])
        if day_low is None and 'Low' in hist.columns:
            day_low = float(hist['Low'].iloc[-1])

        day_change = current_price - previous_close
        day_change_percent = (day_change / previous_close) * 100 if previous_close else 0

        tomorrow_prediction = predictions[0] if predictions else current_price
        profit_loss = tomorrow_prediction - current_price
        profit_loss_percent = (profit_loss / current_price) * 100 if current_price else 0

        # 2. Sentiment Component: Fetch from news analysis
        sentiment_data = {}
        try:
            from stock_api import get_sentiment_analysis
            raw_sentiment = get_sentiment_analysis(resolved_ticker)
            if raw_sentiment:
                sentiment_data = raw_sentiment
                # Update Grafana Viral Ticker gauge
                try:
                    from mlops_v2.monitoring import set_viral_ticker
                    buzz_score = raw_sentiment.get('buzz_score') or raw_sentiment.get('buzz_articles') or 0
                    set_viral_ticker(resolved_ticker, float(buzz_score))
                except Exception as monitoring_err:
                    print(f"Failed to update viral ticker gauge: {monitoring_err}")
        except Exception as e:
            print(f"Sentiment fetch failed: {e}")

        # 🧠 HYBRID BRAIN LOGIC: Weighted Signal Analysis
        # 1. AI Component (60%): Based on forecasted price change
        # Normalize change to -1..1 range (±5% change = ±1.0)
        ai_score = max(-1.0, min(1.0, profit_loss_percent / 5.0))
        
        # 2. Sentiment Component (20%): Uses fetched data
        sentiment_score = sentiment_data.get('score', 0)  # Already -1..1
        
        # 3. Technical Component (20%): Indicators (RSI, Trend)
        tech_score = 0
        latest_rsi = indicators.get('rsi')
        if latest_rsi:
            if latest_rsi < 30: tech_score += 0.5    # Oversold (Bullish)
            elif latest_rsi > 70: tech_score -= 0.5  # Overbought (Bearish)
        
        # Trend check relative to SMA 20
        sma20 = indicators.get('sma20')
        if sma20 and current_price:
            if current_price > sma20: tech_score += 0.3
            else: tech_score -= 0.3
            
        # Final Weighted Aggregation
        final_brain_score = (ai_score * 0.6) + (sentiment_score * 0.2) + (tech_score * 0.2)
        
        # Signal Mapping
        if final_brain_score >= 0.7:
            ai_signal = 'STRONG BUY'
        elif final_brain_score >= 0.2:
            ai_signal = 'BUY'
        elif final_brain_score <= -0.7:
            ai_signal = 'STRONG SELL'
        elif final_brain_score <= -0.2:
            ai_signal = 'SELL'
        else:
            ai_signal = 'HOLD'

        fallback_used = bool(data_provider and data_provider.lower() != 'twelvedata')

        # Historical data
        chart_slice = hist.tail(30)
        historical_data = {
            'dates': [date.strftime('%Y-%m-%d') for date in chart_slice.index],
            'prices': [safe_float(price) for price in chart_slice['Close'].tolist()]
        }

        # Use the last historical date as the base for predictions (avoids weekend gaps)
        last_hist_date = hist.index[-1] if not hist.empty else datetime.now()
        
        future_predictions = []
        prev_price = current_price
        
        # Helper to generate trading day dates (skipping weekends)
        def get_trading_date(start_date, day_offset):
            current = start_date
            count = 0
            while count < day_offset:
                current += timedelta(days=1)
                # Skip Saturday (5) and Sunday (6)
                if current.weekday() < 5:
                    count += 1
            return current

        for index, price in enumerate(predictions):
            exp_change = price - prev_price
            exp_change_pct = (exp_change / prev_price) * 100 if prev_price else 0
            
            # Use index + 1 as offset from last historical date
            pred_date = get_trading_date(last_hist_date, index + 1)
            
            future_predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'price': safe_float(price),
                'expected_change': safe_float(exp_change),
                'expected_change_pct': safe_float(exp_change_pct, 3)
            })
            prev_price = price
        
        # --- Predicted Volume (based on recent volume trend) ---
        recent_vol = hist['Volume'].tail(20) if 'Volume' in hist.columns else pd.Series([0])
        avg_volume = float(recent_vol.mean())
        vol_trend = float(recent_vol.tail(5).mean()) / avg_volume if avg_volume > 0 else 1.0
        predicted_volumes = []
        for i in range(days):
            # Volume tends to revert to mean with some trend
            pred_vol = int(avg_volume * (0.7 + 0.3 * vol_trend))
            predicted_volumes.append(pred_vol)

        recent_ohlcv = hist.tail(60)
        candlestick_data = []
        volume_data = []
        
        for index, row in recent_ohlcv.iterrows():
            date_str = index.strftime('%Y-%m-%d')
            
            # Get raw values from DataFrame - ensure they're never None
            try:
                open_val = float(row['Open']) if pd.notna(row['Open']) else float(row['Close'])
                high_val = float(row['High']) if pd.notna(row['High']) else open_val
                low_val = float(row['Low']) if pd.notna(row['Low']) else open_val
                close_val = float(row['Close']) if pd.notna(row['Close']) else open_val
                volume_val = int(float(row['Volume'])) if 'Volume' in row and pd.notna(row['Volume']) else 0
            except (ValueError, TypeError, KeyError) as e:
                print(f"⚠️ Error parsing OHLCV data for {date_str}: {e}")
                continue  # Skip this candle if data is corrupted
            
            # Round to 2 decimal places
            candle = {
                'date': date_str,
                'open': round(open_val, 2),
                'high': round(high_val, 2),
                'low': round(low_val, 2),
                'close': round(close_val, 2),
                'volume': volume_val
            }
            candlestick_data.append(candle)
            volume_data.append({
                'date': date_str,
                'volume': volume_val
            })
        
        # Debug: Log first candle to verify structure
        if candlestick_data:
            print(f"📊 Sample candle data: {candlestick_data[0]}")
            print(f"📊 Total candles: {len(candlestick_data)}")

        ma_data = {'sma20': [], 'sma50': []}
        for index, row in recent_ohlcv.iterrows():
            date_str = index.strftime('%Y-%m-%d')
            if not pd.isna(row['SMA_20']):
                ma_data['sma20'].append({'date': date_str, 'value': safe_float(row['SMA_20'])})
            if not pd.isna(row['SMA_50']):
                ma_data['sma50'].append({'date': date_str, 'value': safe_float(row['SMA_50'])})

        def calc_period_change(window):
            if len(hist) <= window:
                return None
            start = hist['Close'].iloc[-window - 1]
            end = hist['Close'].iloc[-1]
            if pd.isna(start) or start == 0:
                return None
            return safe_float(((end - start) / start) * 100)

        performance = {
            '1W': calc_period_change(5),
            '1M': calc_period_change(21),
            '3M': calc_period_change(63),
            '1Y': calc_period_change(252)
        }

        perf_slice = hist['Close'].tail(90)
        performance_chart = {
            'dates': [date.strftime('%Y-%m-%d') for date in perf_slice.index],
            'prices': [safe_float(price) for price in perf_slice.tolist()]
        }

        indicator_slice = hist.tail(40)
        indicator_trends = {
            'rsi': {
                'dates': [date.strftime('%Y-%m-%d') for date in indicator_slice.index],
                'values': [safe_float(val) if not pd.isna(val) else None for val in indicator_slice['RSI'].tolist()]
            },
            'ema': {
                'dates': [date.strftime('%Y-%m-%d') for date in indicator_slice.index],
                'values': [safe_float(val) if not pd.isna(val) else None for val in indicator_slice['EMA_20'].tolist()]
            },
            'macd': {
                'dates': [date.strftime('%Y-%m-%d') for date in indicator_slice.index],
                'values': [safe_float(val, 3) if not pd.isna(val) else None for val in indicator_slice['MACD'].tolist()],
                'histogram': [safe_float(val, 3) if not pd.isna(val) else None for val in indicator_slice['MACD_histogram'].tolist()]
            }
        }

        # --- Stock Sentiment (already fetched above for brain logic) ---
        pass
        
        # --- Model info (from registry) ---
        model_info_data = {}
        try:
            from mlops.registry import ModelRegistry
            registry = ModelRegistry()
            best = registry.get_best_model(resolved_ticker)
            if best:
                model_info_data = {
                    'version': f"v{best.get('version', '?')}",
                    'r2': safe_float(best.get('metrics', {}).get('r2'), 4),
                    'mape': safe_float(best.get('metrics', {}).get('mape'), 2),
                    'directional_accuracy': safe_float(best.get('metrics', {}).get('directional_accuracy'), 1),
                    'last_trained': best.get('metadata', {}).get('trained_at', 'unknown')
                }
        except Exception as e:
            print(f"Registry lookup failed: {e}")

        ultimate_payload = _ultimate_prediction_cache.get(resolved_ticker.upper())
        if ultimate_payload:
            ultimate_metrics = ultimate_payload.get('metrics', {})
            model_info_data = {
                'version': ultimate_payload.get('model_version', 'v36-production'),
                'model_type': 'Ultimate Regime Ensemble',
                'directional_accuracy': safe_float(ultimate_metrics.get('accuracy'), 1),
                'f1': safe_float(ultimate_metrics.get('f1'), 1),
                'auc': safe_float(ultimate_metrics.get('auc'), 4),
                'last_trained': ultimate_payload.get('trained_at', 'unknown')
            }
            ai_signal = ultimate_payload.get('signal', ai_signal)
        
        # --- Volume trend label ---
        volume_trend = 'normal'
        if avg_volume > 0:
            current_vol_ratio = volume / avg_volume if volume and avg_volume else 1.0
            if current_vol_ratio > 1.3:
                volume_trend = 'above_average'
            elif current_vol_ratio < 0.7:
                volume_trend = 'below_average'
        
        # --- Bollinger position ---
        bb_pos = 'middle'
        if indicators.get('sma20') and hist.get('Close') is not None:
            try:
                bb_std = hist['Close'].tail(20).std()
                bb_upper = indicators['sma20'] + 2 * bb_std
                bb_lower = indicators['sma20'] - 2 * bb_std
                if current_price > bb_upper:
                    bb_pos = 'above_upper'
                elif current_price < bb_lower:
                    bb_pos = 'below_lower'
            except Exception:
                pass
        
        # Add enriched indicator info
        indicators['volume_trend'] = volume_trend
        indicators['bollinger_position'] = bb_pos
        indicators['atr'] = safe_float(latest_row.get('ATR')) if 'ATR' in hist.columns else None
        
        # v2 inference contract (5-day forward return + confidence interval)
        v2_payload = ultimate_payload
        if v2_payload is None:
            try:
                from mlops_v2.inference import InferenceServiceV2
                v2_payload = InferenceServiceV2().predict(resolved_ticker)
                
                # Update Grafana Accuracy and PnL gauges
                try:
                    from mlops_v2.monitoring import set_accuracy_20d, set_simulated_pnl, set_sharpe_ratio
                    import json
                    from mlops_v2.registry import get_model_paths
                    metadata_path = get_model_paths(resolved_ticker).metadata
                    if metadata_path.exists():
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        metrics = metadata.get("metrics", {})
                        
                        accuracy = _metric_as_percent(metrics.get("xgb_accuracy", 0.0))
                        set_accuracy_20d(resolved_ticker, accuracy)
                        
                        pnl = float(metrics.get("simulated_pnl", 0.0))
                        set_simulated_pnl(resolved_ticker, pnl)
                        
                        sharpe = float(metrics.get("sharpe_ratio", 0.0))
                        set_sharpe_ratio(resolved_ticker, sharpe)
                        
                except Exception as monitoring_err:
                    print(f"Failed to update monitoring gauges: {monitoring_err}")
                    
            except Exception as _v2_exc:
            # Fallback to local computation if v2 models are unavailable.
                five_day_pred = predictions[min(4, len(predictions) - 1)] if predictions else current_price
                pred_return = ((five_day_pred - current_price) / current_price) if current_price else 0.0
                direction_prob = 0.5 + max(-0.2, min(0.2, pred_return * 5.0))
                uncertainty = abs(pred_return) * 0.5 + 0.01
                try:
                    from mlops_v2.feature_engineering import FEATURE_COLUMNS
                    features_used = FEATURE_COLUMNS
                except Exception:
                    features_used = []

                v2_payload = {
                    'ticker': resolved_ticker,
                    'prediction': float(pred_return),
                    'lower_95': float(pred_return - 1.96 * uncertainty),
                    'upper_95': float(pred_return + 1.96 * uncertainty),
                    'confidence': float(max(0.0, min(1.0, 1.0 - uncertainty))),
                    'direction_prob': float(max(0.0, min(1.0, direction_prob))),
                    'model_version': (model_info_data.get('version') or 'fallback') if isinstance(model_info_data, dict) else 'fallback',
                    'features_used': features_used,
                    'data_freshness': datetime.utcnow().isoformat() + 'Z',
                    'drift_score': 0.0,
                }

        # ------------------------------------------------------------------
        # Final recommendation: single source of truth for backend + frontend.
        # This replaces naive "predicted_price > current_price => BUY" behavior.
        # ------------------------------------------------------------------
        try:
            from decision_engine import build_decision, signal_for_forecast_row

            decision_predictions = predictions
            if not decision_predictions and isinstance(v2_payload, dict):
                decision_predictions = v2_payload.get('predicted_prices') or []

            final_decision = build_decision(
                hist=hist,
                current_price=current_price,
                predictions=[float(p) for p in decision_predictions],
                indicators=indicators,
                sentiment=sentiment_data,
                model_payload=v2_payload,
            )
            ai_signal = final_decision.signal

            # If the primary forecast list was empty but the inference payload
            # has calibrated prices, use them for cards/charts/tables.
            if not predictions and len(decision_predictions) > 0:
                predictions = [float(p) for p in decision_predictions[:days]]
                tomorrow_prediction = predictions[0]
                profit_loss = tomorrow_prediction - current_price
                profit_loss_percent = (profit_loss / current_price) * 100 if current_price else 0

            for row in future_predictions:
                row_price = row.get('price')
                if row_price is not None:
                    row['signal'] = signal_for_forecast_row(
                        base_price=current_price,
                        predicted_price=float(row_price),
                        final_decision=final_decision,
                    )
                    row['confidence'] = safe_float(final_decision.confidence, 4)
                    row['min_required_move_pct'] = safe_float(final_decision.min_required_move_pct, 3)
        except Exception as decision_err:
            print(f"Decision engine failed, falling back to HOLD-safe signal: {decision_err}")
            final_decision = None
            if abs(profit_loss_percent) < 0.35:
                ai_signal = 'HOLD'
            for row in future_predictions:
                row.setdefault('signal', 'HOLD')

        model_status = _build_model_status(
            resolved_ticker,
            prediction_ready=bool(len(predictions) > 0 and _ticker_has_trained_model(resolved_ticker))
        )
        trained_prediction_ready = bool(model_status.get('model_ready') and len(predictions) > 0)
        if not trained_prediction_ready:
            predictions = []
            future_predictions = []
            predicted_volumes = []
            tomorrow_prediction = current_price
            profit_loss = 0.0
            profit_loss_percent = 0.0
            ai_signal = 'TRAINING'
            final_decision = None
            v2_payload = {
                'prediction': 0.0,
                'lower_95': 0.0,
                'upper_95': 0.0,
                'confidence': 0.0,
                'direction_prob': 0.5,
                'model_version': 'training',
                'features_used': [],
                'data_freshness': datetime.utcnow().isoformat() + 'Z',
                'drift_score': 0.0,
            }

        cached_payload = _ultimate_prediction_cache.get(resolved_ticker.upper()) or {}
        price_range_low = cached_payload.get('price_range_low', []) if isinstance(cached_payload, dict) else []
        price_range_high = cached_payload.get('price_range_high', []) if isinstance(cached_payload, dict) else []
        price_range_q25 = cached_payload.get('price_range_q25', []) if isinstance(cached_payload, dict) else []
        price_range_q75 = cached_payload.get('price_range_q75', []) if isinstance(cached_payload, dict) else []

        def _build_fallback_price_ranges(price_path):
            """Create conservative forecast ranges when quantile models are unavailable."""
            if not price_path:
                return [], [], [], []

            closes = pd.to_numeric(hist.get('Close', pd.Series(dtype=float)), errors='coerce').dropna()
            daily_vol = float(closes.pct_change().tail(30).std()) if len(closes) > 5 else 0.0
            atr_value = indicators.get('atr') if isinstance(indicators, dict) else None
            atr_pct = float(atr_value) / float(current_price) if atr_value and current_price else 0.0
            daily_uncertainty = max(0.006, min(0.08, max(daily_vol, atr_pct * 0.60)))

            lows, highs, q25s, q75s = [], [], [], []
            for idx, center in enumerate(price_path[:days]):
                center = float(center)
                scale = (idx + 1) ** 0.5
                p10_90_width = center * daily_uncertainty * 1.28 * scale
                p25_75_width = center * daily_uncertainty * 0.67 * scale
                lows.append(round(max(0.01, center - p10_90_width), 2))
                highs.append(round(max(0.01, center + p10_90_width), 2))
                q25s.append(round(max(0.01, center - p25_75_width), 2))
                q75s.append(round(max(0.01, center + p25_75_width), 2))

            return lows, highs, q25s, q75s

        if predictions and (len(price_range_low) < len(predictions) or len(price_range_high) < len(predictions)):
            price_range_low, price_range_high, price_range_q25, price_range_q75 = _build_fallback_price_ranges(predictions)

        # Validate and normalize all range arrays. The UI should always receive
        # a range around the model midpoint instead of a naked exact forecast.
        for idx, center in enumerate(predictions[:days]):
            center = float(center)
            if idx >= len(price_range_low) or idx >= len(price_range_high):
                continue

            low = safe_float(price_range_low[idx])
            high = safe_float(price_range_high[idx])
            if low is None or high is None:
                continue

            low, high = sorted((float(low), float(high)))
            low = min(low, center)
            high = max(high, center)
            price_range_low[idx] = safe_float(low)
            price_range_high[idx] = safe_float(high)

            if idx < len(price_range_q25):
                q25 = safe_float(price_range_q25[idx])
                price_range_q25[idx] = safe_float(max(low, min(center, float(q25)))) if q25 is not None else None
            if idx < len(price_range_q75):
                q75 = safe_float(price_range_q75[idx])
                price_range_q75[idx] = safe_float(min(high, max(center, float(q75)))) if q75 is not None else None

        if predictions and not future_predictions:
            prev_price = current_price
            for index, price in enumerate(predictions[:days]):
                exp_change = price - prev_price
                exp_change_pct = (exp_change / prev_price) * 100 if prev_price else 0
                pred_date = get_trading_date(last_hist_date, index + 1)
                future_predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'price': safe_float(price),
                    'expected_change': safe_float(exp_change),
                    'expected_change_pct': safe_float(exp_change_pct, 3),
                })
                prev_price = price

        for index, row in enumerate(future_predictions):
            if index < len(price_range_low):
                row['range_low'] = safe_float(price_range_low[index])
            if index < len(price_range_high):
                row['range_high'] = safe_float(price_range_high[index])
            if index < len(price_range_q25):
                row['range_q25'] = safe_float(price_range_q25[index])
            if index < len(price_range_q75):
                row['range_q75'] = safe_float(price_range_q75[index])

        prediction_chart = {
            'dates': [row.get('date') for row in future_predictions],
            'median': [row.get('price') for row in future_predictions],
            'range_low': [row.get('range_low') for row in future_predictions],
            'range_high': [row.get('range_high') for row in future_predictions],
            'range_q25': [row.get('range_q25') for row in future_predictions],
            'range_q75': [row.get('range_q75') for row in future_predictions],
            'direction_prob': safe_float(v2_payload.get('direction_prob') if isinstance(v2_payload, dict) else 0.5, 4),
            'confidence': safe_float(v2_payload.get('confidence') if isinstance(v2_payload, dict) else 0.0, 4),
            'model_version': v2_payload.get('model_version') if isinstance(v2_payload, dict) else None,
        }

        recommendation_payload = final_decision.to_dict() if final_decision else {
            'signal': ai_signal,
            'stance': 'Training In Progress' if not trained_prediction_ready else 'Uncertain Market',
            'confidence': 0,
            'confidence_percent': 0,
            'reasons': [
                'Backend training is still running. Real prediction prices will be shown after the trained model is ready.'
                if not trained_prediction_ready
                else 'Decision engine unavailable; defaulted to conservative display.'
            ]
        }
        if 'expected_move_pct' not in recommendation_payload:
            recommendation_payload['expected_move_pct'] = safe_float(profit_loss_percent, 3)

        market_regime = _build_market_regime(hist, indicators, current_price)
        risk_profile = _build_risk_profile(hist, indicators, current_price, market_regime)
        model_health = {}
        backtest_ghost = []
        backtest_metrics = _build_backtest_metrics([])
        try:
            from unified_engine.monitoring import evaluate_predictions, get_model_health, get_prediction_history, log_prediction

            current_prices = {
                date.strftime('%Y-%m-%d'): float(price)
                for date, price in hist['Close'].tail(30).items()
                if not pd.isna(price)
            }
            evaluate_predictions(resolved_ticker, current_prices)
            model_health = get_model_health(resolved_ticker)

            history_rows = get_prediction_history(resolved_ticker, limit=60)
            backtest_metrics = _build_backtest_metrics(history_rows)
            for item in history_rows:
                if item.get('actual_price') is None or item.get('predicted_return') is None:
                    continue
                base_price = float(item.get('current_price') or 0)
                if base_price <= 0:
                    continue
                timestamp = item.get('evaluated_at') or item.get('timestamp')
                date_label = str(timestamp).split('T')[0] if timestamp else None
                if not date_label:
                    continue
                backtest_ghost.append({
                    'date': date_label,
                    'predicted': safe_float(base_price * (1.0 + float(item.get('predicted_return') or 0.0))),
                    'actual': safe_float(item.get('actual_price')),
                    'correct': bool(item.get('correct')),
                })

            if trained_prediction_ready and isinstance(v2_payload, dict):
                log_prediction(
                    resolved_ticker,
                    float(v2_payload.get('direction_prob', 0.5)),
                    recommendation_payload.get('signal', ai_signal),
                    float(v2_payload.get('prediction', 0.0)),
                    float(current_price),
                    str(v2_payload.get('model_version', 'unknown')),
                )
        except Exception as monitoring_err:
            print(f"Prediction reliability monitoring failed: {monitoring_err}")
            model_health = {
                'status': 'unavailable',
                'recent_accuracy': None,
                'needs_retraining': False,
                'reason': 'Prediction monitoring unavailable',
            }

        raw_direction_prob = float(v2_payload.get('direction_prob', 0.5)) if isinstance(v2_payload, dict) else 0.5
        empirical_accuracy = model_health.get('recent_accuracy')
        empirical_anchor = 0.55 if empirical_accuracy is None else _clamp_number(empirical_accuracy, 0.0, 1.0)
        empirical_direction = empirical_anchor if raw_direction_prob >= 0.5 else 1.0 - empirical_anchor
        calibrated_direction_prob = (0.65 * raw_direction_prob) + (0.35 * empirical_direction)
        calibrated_confidence = abs(calibrated_direction_prob - 0.5) * 2.0
        probability_calibration = {
            'method': 'empirical_accuracy_shrinkage',
            'raw_direction_prob': round(float(raw_direction_prob), 4),
            'calibrated_direction_prob': round(float(calibrated_direction_prob), 4),
            'calibrated_confidence': round(float(calibrated_confidence), 4),
            'sample_size': model_health.get('evaluated_predictions') or model_health.get('total_predictions'),
            'status': 'ready' if empirical_accuracy is not None else 'warming_up',
        }

        v2_trust_payload = dict(v2_payload) if isinstance(v2_payload, dict) else {}
        v2_trust_payload['confidence'] = min(
            float(v2_trust_payload.get('confidence', 0.0) or 0.0),
            max(0.18, calibrated_confidence),
        )
        model_trust = _build_user_trust_payload(
            recommendation_payload,
            v2_trust_payload,
            model_health,
            market_regime,
            risk_profile,
        )
        ai_explanation = _build_ai_explanation(
            recommendation_payload,
            market_regime,
            risk_profile,
            model_trust,
            sentiment_data,
        )
        prediction_horizon = {
            'days': days,
            'label': f"{days} Trading Day" if days == 1 else f"{days} Trading Days",
            'style': 'intraday' if days == 1 else 'swing' if days <= 10 else 'position',
        }
        trained_at = (v2_payload.get('trained_at') if isinstance(v2_payload, dict) else None) or model_info_data.get('last_trained')
        model_freshness = {
            'trained_at': trained_at or 'unknown',
            'data_freshness': v2_payload.get('data_freshness') if isinstance(v2_payload, dict) else None,
            'drift_score': safe_float(v2_payload.get('drift_score') if isinstance(v2_payload, dict) else 0.0, 4),
            'status': 'fresh' if trained_prediction_ready and not model_health.get('needs_retraining') else 'retrain_watch',
        }

        response = {
            'success': True,
            'ticker': resolved_ticker,
            'requested_ticker': requested_ticker,
            'resolved_exchange': resolved_exchange,
            'resolved_country': resolved_country,
            'used_fallback_source': fallback_used,
            'company_name': company_name,
            'currency': currency,
            'current_price': safe_float(current_price),
            'predicted_price': safe_float(tomorrow_prediction),
            'profit_loss': safe_float(profit_loss),
            'profit_loss_percent': safe_float(profit_loss_percent),
            'is_profit': bool(profit_loss > 0),
            'ai_signal': ai_signal,
            'recommendation': recommendation_payload,
            'market_regime': market_regime,
            'risk_profile': risk_profile,
            'model_trust': model_trust,
            'reliability': model_health,
            'backtest_metrics': backtest_metrics,
            'ai_explanation': ai_explanation,
            'prediction_horizon': prediction_horizon,
            'model_freshness': model_freshness,
            'probability_calibration': probability_calibration,
            'backtest_ghost': backtest_ghost[-30:],
            'day_change': safe_float(day_change),
            'day_change_percent': safe_float(day_change_percent),
            'day_high': safe_float(day_high),
            'day_low': safe_float(day_low),
            'volume': volume,
            'predicted_volume': predicted_volumes,
            'is_training': not trained_prediction_ready,
            'prediction_ready': trained_prediction_ready,
            'analysis_mode': model_status.get('analysis_mode'),
            'model_status': model_status,
            'market_cap': market_cap,
            'pe_ratio': safe_float(pe_ratio) if isinstance(pe_ratio, (int, float)) and not pd.isna(pe_ratio) else None,
            'days_predicted': days,
            'supported_prediction_horizons': list(PREDICTION_HORIZONS),
            'indicators': indicators,
            'sentiment': sentiment_data,
            'model_info': model_info_data,
            'historical_data': historical_data,
            'future_predictions': future_predictions,
            'prediction_chart': prediction_chart,
            'signal_diagnostics': recommendation_payload,
            'technical_chart': {
                'candles': candlestick_data,
                'volumes': volume_data,
                'moving_averages': ma_data
            },
            # Required inference contract fields for production API consumers.
            'prediction': v2_payload['prediction'],
            'lower_95': v2_payload['lower_95'],
            'upper_95': v2_payload['upper_95'],
            'confidence': v2_payload['confidence'],
            'direction_prob': v2_payload['direction_prob'],
            'model_version': v2_payload['model_version'],
            'features_used': v2_payload['features_used'],
            'data_freshness': v2_payload['data_freshness'],
            'drift_score': v2_payload['drift_score'],
            'indicator_trends': indicator_trends,
            'performance': performance,
            'performance_chart': performance_chart,
            # Price RANGES from quantile regression (like real trading apps)
            'price_range_low': price_range_low,
            'price_range_high': price_range_high,
            'price_range_q25': price_range_q25,
            'price_range_q75': price_range_q75,
            'quantile_returns': cached_payload.get('quantile_returns', {}) if isinstance(cached_payload, dict) else {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if app.debug and provider_message:
            response['provider_note'] = provider_message
        if app.debug and profile_symbol_used:
            response['profile_symbol_used'] = profile_symbol_used
        if app.debug and quote_symbol_used:
            response['quote_symbol_used'] = quote_symbol_used
        if app.debug and metrics_symbol_used:
            response['metrics_symbol_used'] = metrics_symbol_used

        return jsonify(response)

    except Exception as e:
        import traceback
        print(f"ERROR: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/news/<ticker>')
def get_news(ticker):
    """Get company news"""
    # Lazy import stock_api
    from stock_api import get_company_news
    
    try:
        ticker = ticker.upper()
        days = request.args.get('days', default=7, type=int)
        
        news = get_company_news(ticker, days=days)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'news': news
        })
        
    except Exception as e:
        print(f"ERROR fetching news: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/sentiment/<ticker>')
def get_sentiment(ticker):
    """Get sentiment analysis"""
    # Lazy import stock_api
    from stock_api import get_sentiment_analysis
    
    try:
        ticker = ticker.upper()
        sentiment = get_sentiment_analysis(ticker)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'sentiment': sentiment
        })
        
    except Exception as e:
        print(f"ERROR fetching sentiment: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/admin/system-health')
@firebase_auth_required(admin_only=True)
def admin_system_health():
    metrics = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'modelPath': MODEL_PATHS[0], # Assuming first path is primary
        'modelLoaded': bool(_model_cache), # Check if any model is loaded
        'modelFilePresent': any(os.path.exists(p) for p in MODEL_PATHS),
        'firebaseAdminConfigured': firebase_ready,
        'allowedOrigins': os.environ.get('ALLOWED_ORIGINS', '*'),
        'adminEmailsConfigured': bool(ADMIN_EMAILS)
    }

    return jsonify({'success': True, 'metrics': metrics})

# =============================================================================
# UNIFIED ENGINE v4.0 API ENDPOINTS
# =============================================================================

@app.route('/api/unified/train/<ticker>', methods=['POST'])
@firebase_auth_required(admin_only=True)
def unified_train_ticker(ticker):
    """Train Unified Engine v4.0 model for a specific ticker (admin only)."""
    try:
        from unified_engine.training import UnifiedTrainer
        result = UnifiedTrainer.train(ticker)
        return jsonify({
            'success': result.success,
            'ticker': result.ticker,
            'reason': result.reason,
            'metrics': result.metrics,
            'model_version': result.model_version,
            'selected_features': result.selected_features,
            'trained_at': result.trained_at,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/unified/health/<ticker>')
def unified_model_health(ticker):
    """Get Unified Engine model health and prediction tracking for a ticker."""
    try:
        from unified_engine.inference import UnifiedPredictor
        from unified_engine.monitoring import get_model_health, get_prediction_history

        metadata = UnifiedPredictor.get_model_metadata(ticker)
        health = get_model_health(ticker)
        history = get_prediction_history(ticker, limit=10)

        return jsonify({
            'success': True,
            'ticker': ticker.upper(),
            'model_available': UnifiedPredictor.is_model_available(ticker),
            'metadata': metadata,
            'health': health,
            'recent_predictions': history,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/unified/status')
def unified_engine_status():
    """Get overall Unified Engine v4.0 status."""
    try:
        from unified_engine.config import CONFIG
        from unified_engine.inference import UnifiedPredictor

        model_dir = CONFIG.model_dir
        trained_tickers = []
        if model_dir.exists():
            for d in model_dir.iterdir():
                if d.is_dir() and (d / 'model.joblib').exists():
                    trained_tickers.append(d.name)

        return jsonify({
            'success': True,
            'engine': 'Unified Engine v4.0',
            'trained_tickers': sorted(trained_tickers),
            'total_models': len(trained_tickers),
            'model_dir': str(model_dir),
            'config': {
                'prediction_horizon': CONFIG.prediction_horizon,
                'supported_prediction_horizons': list(PREDICTION_HORIZONS),
                'fetch_days': CONFIG.fetch_days,
                'purge_days': CONFIG.wf_purge_days,
                'embargo_days': CONFIG.wf_embargo_days,
                'max_features': CONFIG.max_features,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def predict_multi_day_lstm(hist, current_price, days, ticker):
    """
    Predict multiple days ahead — Unified Engine v5.0 is the ONLY path.
    Returns price RANGES (not exact prices) like real trading apps.
    """
    import numpy as np

    def trigger_background_training():
        ticker_key = ticker.upper()
        if not _is_trainable_market_ticker(ticker_key):
            _update_ticker_training_job(
                ticker_key,
                state='untrainable',
                stage='unsupported_symbol',
                progress=0,
                message=_unsupported_training_message(ticker_key)
            )
            return
        with _ultimate_training_state_lock:
            should_start = ticker_key not in _ultimate_training_in_progress
            if should_start:
                _ultimate_training_in_progress.add(ticker_key)

        if not should_start:
            print(f"Training already in progress for {ticker_key}.")
            return

        _update_ticker_training_job(
            ticker, state='queued', stage='queued', progress=8,
            message='Training request accepted. Market data remains available while the model is prepared.'
        )

        def background_train():
            try:
                print(f"Background v5.0 training started for {ticker_key}", flush=True)
                from unified_engine.training import train_unified_model
                train_result = train_unified_model(ticker_key)
                if train_result and train_result.success:
                    _publish_training_metrics(ticker_key, train_result.metrics)
                    _update_ticker_training_job(
                        ticker_key, state='training', stage='v2_observability_training',
                        progress=82, message='Publishing MLflow, DVC, and Grafana observability artifacts.'
                    )
                    _run_v2_observability_training(ticker_key)
                    _update_ticker_training_job(
                        ticker_key, state='ready', stage='custom_model_ready',
                        progress=100, message='Unified Engine v5.0 model and observability artifacts are ready.'
                    )
                    print(f"✅ Background v5.0 training completed for {ticker_key}", flush=True)
                else:
                    reason = train_result.reason if train_result else "unknown"
                    print(f"Background v5.0 training failed for {ticker_key}: {reason}", flush=True)
                    failure_state, failure_stage, failure_message = _training_failure_status(reason)
                    _update_ticker_training_job(
                        ticker_key, state=failure_state, stage=failure_stage,
                        progress=0, message=failure_message
                    )
            except Exception as train_err:
                print(f"Background v5.0 training failed for {ticker_key}: {train_err}", flush=True)
                failure_state, failure_stage, failure_message = _training_failure_status(train_err)
                _update_ticker_training_job(
                    ticker_key, state=failure_state, stage=failure_stage,
                    progress=0, message=failure_message
                )
            finally:
                with _ultimate_training_state_lock:
                    _ultimate_training_in_progress.discard(ticker_key)

        threading.Thread(target=background_train, daemon=True).start()

    try:
        _ultimate_prediction_cache.pop(ticker.upper(), None)

        # =================================================================
        # SINGLE PATH: Unified Engine v5.0
        # XGBoost + LightGBM + Quantile Regression + Meta-Learner
        # =================================================================
        try:
            from unified_engine.inference import UnifiedPredictor
            payload = UnifiedPredictor.predict(
                ticker, hist, current_price=current_price, days=days
            )
            if payload and payload.get('predicted_prices'):
                _ultimate_prediction_cache[ticker.upper()] = payload
                print(f"✅ v5.0 prediction for {ticker}: signal={payload.get('signal')}, "
                      f"prob={payload.get('direction_prob', 0):.3f}, "
                      f"confidence={payload.get('confidence', 0):.3f}")

                # Update Prometheus metrics
                try:
                    from mlops_v2.monitoring import inc_prediction
                    inc_prediction(ticker.upper())
                    metrics = payload.get('metrics', {})
                    if 'accuracy' in metrics:
                        from mlops_v2.monitoring import set_accuracy_20d
                        set_accuracy_20d(ticker.upper(), metrics['accuracy'])
                except Exception:
                    pass

                return payload['predicted_prices']
            else:
                print(f"No v5.0 model for {ticker}. Triggering background training...")
                trigger_background_training()
        except Exception as engine_err:
            print(f"Unified Engine v5.0 unavailable for {ticker}: {engine_err}")
            trigger_background_training()

        # No model available yet — return empty (frontend shows training state)
        return []

    except Exception as e:
        print(f"Prediction error for {ticker}: {e}")
        return []

def predict_with_technical_analysis(hist, current_price):
    """Fallback prediction using technical indicators"""
    try:
        last_row = hist.iloc[-1]
        recent_prices = hist['Close'].tail(5).values
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        prediction_change = 0
        
        rsi = last_row['RSI']
        if rsi < 30:
            prediction_change += 0.01
        elif rsi > 70:
            prediction_change -= 0.01
        
        if current_price > last_row['SMA_20'] and last_row['SMA_20'] > last_row['SMA_50']:
            prediction_change += 0.005
        elif current_price < last_row['SMA_20'] and last_row['SMA_20'] < last_row['SMA_50']:
            prediction_change -= 0.005
        
        if last_row['MACD'] > 0:
            prediction_change += 0.003
        else:
            prediction_change -= 0.003
        
        prediction_change += trend * 0.3
        predicted_price = current_price * (1 + prediction_change)
        return predicted_price
        
    except Exception as e:
        return current_price * 1.002

# Serve React App (with caching)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def serve(path):
    """Serve React frontend with proper caching and SPA routing support
    This function will:
    - Serve static files when they exist
    - Serve index.html for known SPA routes (so client-side routing works)
    - Return JSON 404 for API routes that aren't found
    """

    # Skip API routes - let API handlers respond (or return 404)
    if path.startswith('api/'):
        app.logger.debug(f"API path requested that was not matched: {path}")
        return jsonify({'error': 'API endpoint not found'}), 404

    # Proxy chatbot routes to FastAPI chatbot on port 8001
    if path.startswith('chatbot/'):
        return _proxy_chatbot(path[len('chatbot'):])

    # Serve static assets if they exist
    static_path = os.path.join(app.static_folder, path)
    if path != '' and os.path.exists(static_path):
        try:
            response = send_from_directory(app.static_folder, path)
            if path.startswith('static/'):
                response.cache_control.max_age = 31536000
                response.cache_control.public = True
            return response
        except Exception:
            pass # Fallthrough to serve index.html

    # Default: serve index.html for all other client-side routes
    try:
        response = send_from_directory(app.static_folder, 'index.html')
        response.cache_control.no_cache = True
        response.cache_control.no_store = True
        response.cache_control.must_revalidate = True
        return response
    except Exception as e:
        # If build folder is missing, provide a professional JSON error when hit directly
        if not os.path.exists(app.static_folder):
             return jsonify({'error': 'Static Assets Missing', 'message': 'Backend is configured to serve frontend from /build folder, but it was not found. For local development, use port 3000.'}), 404
        return jsonify({'error': 'Application error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Use environment variable PORT (defaulting to 8000 for local dev)
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

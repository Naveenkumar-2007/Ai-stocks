# app/tools/prediction_tools.py
"""
LSTM Prediction integration — fetches real trained model data from the MLOps registry
and runs actual LSTM inference to determine price direction.
"""
import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Add the backend parent directory to sys.path to import mlops
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from mlops.registry import ModelRegistry
from models.schemas import PredictionResult

logger = logging.getLogger(__name__)


def _load_trained_stocks():
    """Load the list of trained stocks from stocks.json"""
    try:
        path = os.path.join(backend_dir, "mlops", "stocks.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return set(json.load(f))
    except Exception:
        pass
    return set()



def _run_lstm_inference(symbol: str) -> Optional[Tuple[float, float]]:
    """
    Run actual LSTM inference for a symbol and return (current_price, predicted_price).
    Uses the same model loading and prediction logic as the main app.
    Returns None if inference cannot be performed.
    """
    try:
        # Import the prediction infrastructure from the main backend
        sys.path.insert(0, backend_dir)
        from app import load_lstm_model, load_saved_scaler, predict_multi_day_lstm
        from services.finnhub_service import finnhub_service

        import pandas as pd
        import ta as ta_lib
        import numpy as np

        # Fetch historical data using finnhub (faster and has timeout management)
        # Use asyncio.run if this is called from a thread, or simple await
        # Since this is run in a thread via to_thread, we need a sync wrapper or just use candles
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            data = loop.run_until_complete(finnhub_service.get_candles(symbol, resolution="D", count=100))
        finally:
            loop.close()

        if not data or data.get('s') != 'ok':
            logger.warning(f"Insufficient historical data for {symbol} inference from Finnhub")
            return None

        # Format data for prediction
        hist = pd.DataFrame({
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data['v']
        })
        
        if len(hist) < 60:
            logger.warning(f"Insufficient historical data points ({len(hist)}) for {symbol} inference")
            return None

        current_price = float(hist['Close'].iloc[-1])

        # Add technical indicators needed by predict_multi_day_lstm
        close = hist['Close']
        hist['SMA_20'] = close.rolling(window=20).mean()
        hist['SMA_50'] = close.rolling(window=50).mean()
        hist['RSI'] = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
        macd_indicator = ta_lib.trend.MACD(close)
        hist['MACD'] = macd_indicator.macd()
        hist = hist.dropna()

        if hist.empty:
            return None

        # Get 1-day prediction
        predictions = predict_multi_day_lstm(hist, current_price, days=1, ticker=symbol)
        if predictions and len(predictions) > 0:
            predicted_price = float(predictions[0])
            return (current_price, predicted_price)
        return None

    except ImportError as e:
        logger.warning(f"Cannot import prediction infrastructure: {e}")
        return None
    except Exception as e:
        logger.warning(f"LSTM inference failed for {symbol}: {e}")
        return None


class PredictionTools:
    def __init__(self):
        try:
            self.registry = ModelRegistry()
            self.trained_stocks = _load_trained_stocks()
            logger.info(f"🤖 PredictionTools initialized with {len(self.trained_stocks)} trained tickers")
        except Exception as e:
            logger.error(f"Failed to initialize ModelRegistry: {e}")
            self.registry = None
            self.trained_stocks = set()

    async def get_latest_prediction(self, symbol: str) -> Optional[PredictionResult]:
        """Fetch real model metrics and run LSTM inference to determine price direction"""
        if not self.registry:
            return None

        try:
            # Check if this stock has a trained model
            if symbol not in self.trained_stocks:
                logger.info(f"⚠️ No trained model for {symbol}")
                return None

            # Get latest model info from the registry
            model_info = self.registry.get_latest_model(symbol)
            if not model_info:
                logger.info(f"⚠️ No model found in registry for {symbol}")
                return None

            # Extract real metrics from the trained model
            metrics = model_info.get('metrics', {})
            val_loss = metrics.get('val_loss', None)
            mape = metrics.get('mape', None)
            rmse = metrics.get('rmse', None)
            r2 = metrics.get('r2', None)
            mae = metrics.get('mae', None)

            # Get model stats for additional context
            stats = self.registry.get_model_stats(symbol)
            total_versions = stats.get('total_versions', 0)
            best_val_loss = stats.get('best_val_loss', None)

            # ── Determine DIRECTION from actual LSTM inference ──
            direction = "➡️ Neutral"
            confidence = 0.5
            target_price = None

            # Try actual LSTM inference to determine real price direction
            inference_result = await asyncio.to_thread(_run_lstm_inference, symbol)

            if inference_result:
                current_price, predicted_price = inference_result
                target_price = round(predicted_price, 2)
                price_change_pct = ((predicted_price - current_price) / current_price) * 100

                # Direction from actual price forecast
                if price_change_pct > 2.0:
                    direction = "📈 Bullish (AI Forecast: Price Up)"
                elif price_change_pct > 0.5:
                    direction = "📈 Slightly Bullish"
                elif price_change_pct < -2.0:
                    direction = "📉 Bearish (AI Forecast: Price Down)"
                elif price_change_pct < -0.5:
                    direction = "📉 Slightly Bearish"
                else:
                    direction = "➡️ Neutral (Sideways)"

                # Confidence from model accuracy metrics
                if mape is not None:
                    confidence = min(0.95, max(0.3, 1.0 / (1.0 + mape / 100.0)))
                elif r2 is not None:
                    confidence = min(0.95, max(0.3, r2))
                elif val_loss is not None:
                    confidence = 0.85 if val_loss < 0.001 else (0.7 if val_loss < 0.01 else 0.5)
            else:
                # Fallback: no inference available, use metrics for confidence only
                # Label direction as "Model available" without making directional claims
                logger.info(f"⚠️ LSTM inference unavailable for {symbol}, using metrics-only assessment")
                if mape is not None:
                    if mape < 3.0:
                        direction = "🤖 High Accuracy Model (run prediction page for direction)"
                        confidence = min(0.95, (100 - mape) / 100)
                    elif mape < 5.0:
                        direction = "🤖 Good Accuracy Model"
                        confidence = (100 - mape) / 100
                    elif mape < 10.0:
                        direction = "🤖 Moderate Accuracy Model"
                        confidence = max(0.4, (100 - mape) / 100)
                    else:
                        direction = "⚠️ Low Accuracy Model"
                        confidence = max(0.2, (100 - mape) / 100)
                elif val_loss is not None:
                    if val_loss < 0.001:
                        direction = "🤖 High Accuracy Model (Low Val Loss)"
                        confidence = 0.85
                    elif val_loss < 0.01:
                        direction = "🤖 Good Accuracy Model"
                        confidence = 0.7
                    else:
                        direction = "🤖 Model Available"
                        confidence = 0.5

            # Build key factors from real metrics + prediction info
            key_factors = []
            if target_price is not None:
                key_factors.append(f"Predicted Price: ${target_price:.2f}")
            if mape is not None:
                key_factors.append(f"MAPE: {mape:.2f}%")
            if rmse is not None:
                key_factors.append(f"RMSE: {rmse:.4f}")
            if r2 is not None:
                key_factors.append(f"R²: {r2:.4f}")
            if mae is not None:
                key_factors.append(f"MAE: {mae:.4f}")
            if val_loss is not None:
                key_factors.append(f"Val Loss: {val_loss:.6f}")

            key_factors.append(f"Model Versions: {total_versions}")

            trained_on = model_info.get('timestamp', model_info.get('created_at', 'N/A'))
            key_factors.append(f"Last Trained: {trained_on}")

            return PredictionResult(
                symbol=symbol,
                predicted_direction=direction,
                confidence=round(confidence, 3),
                target_price=target_price,
                key_factors=key_factors,
                metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )

        except Exception as e:
            logger.error(f"❌ Error fetching LSTM prediction for {symbol}: {e}")
            return None

    async def get_trained_stocks_list(self) -> list:
        """Return list of all stocks with trained models"""
        return list(self.trained_stocks)


prediction_tools = PredictionTools()


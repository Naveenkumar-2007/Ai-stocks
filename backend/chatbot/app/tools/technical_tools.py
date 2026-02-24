# app/tools/technical_tools.py
"""
Advanced Technical Analysis Tools
Calculates RSI, SMA, MACD, Bollinger Bands, EMA, Stochastic, ADX, and VWAP.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from services.finnhub_service import finnhub_service
from models.schemas import TechnicalIndicator
import logging

logger = logging.getLogger(__name__)


class TechnicalTools:
    @staticmethod
    async def get_indicators(symbol: str) -> List[TechnicalIndicator]:
        """Calculate a comprehensive set of technical indicators for a symbol"""
        indicators = []
        try:
            data = await finnhub_service.get_candles(symbol, resolution="D", count=100)
            if not data or data.get('s') != 'ok':
                return []

            closes = pd.Series(data['c'], dtype=float)
            highs = pd.Series(data['h'], dtype=float)
            lows = pd.Series(data['l'], dtype=float)
            volumes = pd.Series(data.get('v', [0] * len(closes)), dtype=float)

            # ── 1. RSI (14) ───────────────────────────────────────
            delta = closes.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            if np.isnan(current_rsi):
                current_rsi = 50.0

            rsi_signal = "neutral"
            rsi_desc = "neutral momentum"
            if current_rsi > 70:
                rsi_signal = "overbought"
                rsi_desc = "overbought — potential pullback or reversal ahead"
            elif current_rsi > 60:
                rsi_signal = "buy"
                rsi_desc = "bullish momentum building"
            elif current_rsi < 30:
                rsi_signal = "oversold"
                rsi_desc = "oversold — potential bounce or reversal ahead"
            elif current_rsi < 40:
                rsi_signal = "sell"
                rsi_desc = "bearish momentum, watch for reversal"

            indicators.append(TechnicalIndicator(
                name="RSI(14)",
                value=float(round(current_rsi, 2)),
                signal=rsi_signal,
                description=f"RSI at {current_rsi:.1f} — {rsi_desc}."
            ))

            current_price = closes.iloc[-1]

            # ── 2. SMA (20) ──────────────────────────────────────
            sma20 = closes.rolling(window=20).mean().iloc[-1]
            if not np.isnan(sma20):
                sma_pct = ((current_price - sma20) / sma20) * 100
                ma_signal = "buy" if current_price > sma20 else "sell"
                indicators.append(TechnicalIndicator(
                    name="SMA(20)",
                    value=float(round(sma20, 2)),
                    signal=ma_signal,
                    description=f"Price is {abs(sma_pct):.1f}% {'above' if ma_signal == 'buy' else 'below'} the 20-day SMA — {'bullish trend' if ma_signal == 'buy' else 'bearish trend'}."
                ))

            # ── 3. EMA (12) ──────────────────────────────────────
            ema12 = closes.ewm(span=12, adjust=False).mean().iloc[-1]
            if not np.isnan(ema12):
                ema_signal = "buy" if current_price > ema12 else "sell"
                indicators.append(TechnicalIndicator(
                    name="EMA(12)",
                    value=float(round(ema12, 2)),
                    signal=ema_signal,
                    description=f"Price is {'above' if ema_signal == 'buy' else 'below'} the 12-day EMA — short-term {'bullish' if ema_signal == 'buy' else 'bearish'} signal."
                ))

            # ── 4. MACD (12, 26, 9) ──────────────────────────────
            ema_12 = closes.ewm(span=12, adjust=False).mean()
            ema_26 = closes.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line

            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_hist = macd_histogram.iloc[-1]
            prev_hist = macd_histogram.iloc[-2] if len(macd_histogram) > 1 else 0

            if not np.isnan(current_macd):
                if current_macd > current_signal and current_hist > 0:
                    macd_sig = "buy"
                    macd_desc = "bullish crossover — MACD above signal line with positive histogram"
                elif current_macd < current_signal and current_hist < 0:
                    macd_sig = "sell"
                    macd_desc = "bearish crossover — MACD below signal line with negative histogram"
                elif current_hist > 0 and current_hist < prev_hist:
                    macd_sig = "neutral"
                    macd_desc = "bullish momentum fading — histogram shrinking"
                elif current_hist < 0 and current_hist > prev_hist:
                    macd_sig = "neutral"
                    macd_desc = "bearish momentum fading — histogram recovering"
                else:
                    macd_sig = "neutral"
                    macd_desc = "no clear crossover signal"

                indicators.append(TechnicalIndicator(
                    name="MACD(12,26,9)",
                    value=float(round(current_macd, 4)),
                    signal=macd_sig,
                    description=f"MACD: {current_macd:.4f}, Signal: {current_signal:.4f}, Histogram: {current_hist:.4f} — {macd_desc}."
                ))

            # ── 5. Bollinger Bands (20, 2σ) ──────────────────────
            sma20_bb = closes.rolling(window=20).mean()
            std20 = closes.rolling(window=20).std()
            upper_band = (sma20_bb + 2 * std20).iloc[-1]
            lower_band = (sma20_bb - 2 * std20).iloc[-1]
            middle_band = sma20_bb.iloc[-1]

            if not np.isnan(upper_band):
                bb_width = ((upper_band - lower_band) / middle_band) * 100
                if current_price >= upper_band:
                    bb_signal = "overbought"
                    bb_desc = f"price at/above upper band (${upper_band:.2f}) — overbought, potential reversal"
                elif current_price <= lower_band:
                    bb_signal = "oversold"
                    bb_desc = f"price at/below lower band (${lower_band:.2f}) — oversold, potential bounce"
                elif current_price > middle_band:
                    bb_signal = "buy"
                    bb_desc = f"price between middle and upper band — bullish zone"
                else:
                    bb_signal = "sell"
                    bb_desc = f"price between lower and middle band — bearish zone"

                indicators.append(TechnicalIndicator(
                    name="Bollinger Bands(20,2)",
                    value=float(round(bb_width, 2)),
                    signal=bb_signal,
                    description=f"Upper: ${upper_band:.2f} | Mid: ${middle_band:.2f} | Lower: ${lower_band:.2f} — {bb_desc}. Band width: {bb_width:.1f}%."
                ))

            # ── 6. Stochastic Oscillator (14, 3) ─────────────────
            if len(highs) >= 14:
                low_14 = lows.rolling(window=14).min()
                high_14 = highs.rolling(window=14).max()
                denom = (high_14 - low_14).replace(0, np.nan)
                stoch_k = ((closes - low_14) / denom) * 100
                stoch_d = stoch_k.rolling(window=3).mean()

                k_val = stoch_k.iloc[-1]
                d_val = stoch_d.iloc[-1]

                if not np.isnan(k_val):
                    if k_val > 80:
                        stoch_sig = "overbought"
                        stoch_desc = "overbought territory — momentum may slow"
                    elif k_val < 20:
                        stoch_sig = "oversold"
                        stoch_desc = "oversold territory — potential bounce"
                    elif k_val > d_val:
                        stoch_sig = "buy"
                        stoch_desc = "%K crossed above %D — bullish momentum"
                    elif k_val < d_val:
                        stoch_sig = "sell"
                        stoch_desc = "%K crossed below %D — bearish momentum"
                    else:
                        stoch_sig = "neutral"
                        stoch_desc = "no clear signal"

                    indicators.append(TechnicalIndicator(
                        name="Stochastic(14,3)",
                        value=float(round(k_val, 2)),
                        signal=stoch_sig,
                        description=f"%K: {k_val:.1f}, %D: {d_val:.1f} — {stoch_desc}."
                    ))

            # ── 7. ADX (14) — Average Directional Index ──────────
            if len(closes) >= 28:
                tr1 = highs - lows
                tr2 = abs(highs - closes.shift(1))
                tr3 = abs(lows - closes.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr14 = tr.rolling(window=14).mean()

                plus_dm = highs.diff()
                minus_dm = -lows.diff()
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
                minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

                plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr14.replace(0, np.nan))
                minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr14.replace(0, np.nan))

                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
                adx = dx.rolling(window=14).mean()

                current_adx = adx.iloc[-1]
                current_plus = plus_di.iloc[-1]
                current_minus = minus_di.iloc[-1]

                if not np.isnan(current_adx):
                    if current_adx > 25 and current_plus > current_minus:
                        adx_sig = "buy"
                        adx_desc = f"strong uptrend (ADX {current_adx:.0f}, +DI > -DI)"
                    elif current_adx > 25 and current_minus > current_plus:
                        adx_sig = "sell"
                        adx_desc = f"strong downtrend (ADX {current_adx:.0f}, -DI > +DI)"
                    elif current_adx < 20:
                        adx_sig = "neutral"
                        adx_desc = f"weak/no trend (ADX {current_adx:.0f}) — market is range-bound"
                    else:
                        adx_sig = "neutral"
                        adx_desc = f"moderate trend strength (ADX {current_adx:.0f})"

                    indicators.append(TechnicalIndicator(
                        name="ADX(14)",
                        value=float(round(current_adx, 2)),
                        signal=adx_sig,
                        description=f"+DI: {current_plus:.1f}, -DI: {current_minus:.1f} — {adx_desc}."
                    ))

            # ── 8. Approximate VWAP ──────────────────────────────
            if volumes.sum() > 0 and len(volumes) > 0:
                typical_price = (highs + lows + closes) / 3
                cum_vol = volumes.cumsum()
                cum_tp_vol = (typical_price * volumes).cumsum()
                vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

                current_vwap = vwap.iloc[-1]
                if not np.isnan(current_vwap):
                    vwap_signal = "buy" if current_price > current_vwap else "sell"
                    vwap_pct = ((current_price - current_vwap) / current_vwap) * 100
                    indicators.append(TechnicalIndicator(
                        name="VWAP",
                        value=float(round(current_vwap, 2)),
                        signal=vwap_signal,
                        description=f"Price is {abs(vwap_pct):.1f}% {'above' if vwap_signal == 'buy' else 'below'} VWAP (${current_vwap:.2f}) — {'trading above' if vwap_signal == 'buy' else 'trading below'} institutional fair value."
                    ))

            # ── Summary Signal ────────────────────────────────────
            buy_count = sum(1 for i in indicators if i.signal == "buy")
            sell_count = sum(1 for i in indicators if i.signal in ("sell", "overbought"))
            total = len(indicators)

            if total > 0:
                if buy_count / total >= 0.6:
                    summary_signal = "buy"
                    summary_desc = f"Overall BULLISH — {buy_count}/{total} indicators are bullish"
                elif sell_count / total >= 0.6:
                    summary_signal = "sell"
                    summary_desc = f"Overall BEARISH — {sell_count}/{total} indicators are bearish"
                else:
                    summary_signal = "neutral"
                    summary_desc = f"MIXED signals — {buy_count} bullish, {sell_count} bearish out of {total} indicators"

                indicators.append(TechnicalIndicator(
                    name="Overall Technical Score",
                    value=float(round(buy_count / total * 100, 1)),
                    signal=summary_signal,
                    description=summary_desc
                ))

            return indicators
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return []

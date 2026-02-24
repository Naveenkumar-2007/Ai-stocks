# app/services/finnhub_service.py
"""
Finnhub Market Data Service
Provides live quotes, company news, candles, company profile,
basic financials, and analyst recommendation trends.
Supports automatic API key rotation on rate-limit errors (429).
"""
import aiohttp
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


class AsyncKeyRotator:
    """Manages multiple API keys with automatic rotation on rate-limit errors (async)."""
    def __init__(self, name: str, keys: list):
        self.name = name
        self.keys = [k.strip() for k in keys if k.strip()]
        if not self.keys:
            raise RuntimeError(f"No API keys configured for {name}")
        self._idx = 0
        logger.info(f"🔑 {name}: loaded {len(self.keys)} API key(s)")

    @property
    def key(self) -> str:
        return self.keys[self._idx % len(self.keys)]

    def rotate(self, failed_key: str = None) -> str:
        if failed_key and self.keys[self._idx % len(self.keys)] != failed_key:
            return self.key
        self._idx += 1
        new_key = self.keys[self._idx % len(self.keys)]
        logger.info(f"🔄 {self.name}: rotated to key #{(self._idx % len(self.keys)) + 1}/{len(self.keys)}")
        return new_key

    def __len__(self):
        return len(self.keys)


def _load_finnhub_keys() -> list:
    """Load Finnhub keys from environment (comma-separated or single)."""
    keys_str = os.getenv('FINNHUB_API_KEYS', '')
    if keys_str:
        return keys_str.split(',')
    single = os.getenv('FINNHUB_API_KEY', '')
    return [single] if single else []


# Initialize rotator at module level
_finnhub_keys = _load_finnhub_keys()
finnhub_rotator = AsyncKeyRotator('Finnhub-Chatbot', _finnhub_keys) if _finnhub_keys else None


class FinnhubService:
    def __init__(self):
        from config import settings
        self.base_url = settings.finnhub_base_url
        self.rotator = finnhub_rotator
        # Fallback to settings if rotator is not available
        if not self.rotator:
            self._static_key = settings.finnhub_api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
    @property
    def api_key(self) -> str:
        """Always return the currently active key."""
        if self.rotator:
            return self.rotator.key
        return self._static_key
    
    async def _get_session(self, timeout: float = 10.0) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            )
        return self.session

    async def _rotate_and_refresh_session(self, failed_key: str):
        """Rotate key after a rate-limit error."""
        if self.rotator:
            self.rotator.rotate(failed_key)

    async def _request_with_rotation(self, url: str, params: dict, timeout: float = 10.0) -> Any:
        """Make a request with automatic key rotation on rate limits."""
        attempts = len(self.rotator) if self.rotator else 1
        for attempt in range(attempts):
            current_key = self.api_key
            params_with_token = {**params, "token": current_key}
            
            # Use a fresh session for each request to ensure different timeouts apply correctly
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                try:
                    async with session.get(url, params=params_with_token) as response:
                        if response.status == 429:
                            logger.warning(f"⚠️ Finnhub rate limited (429), rotating key...")
                            await self._rotate_and_refresh_session(current_key)
                            continue
                        response.raise_for_status()
                        return await response.json()
                except aiohttp.ClientResponseError as e:
                    if e.status == 429 and attempt < attempts - 1:
                        await self._rotate_and_refresh_session(current_key)
                        continue
                    raise
                except asyncio.TimeoutError:
                    logger.warning(f"🕒 Finnhub request timed out ({timeout}s) for {url}")
                    if attempt < attempts - 1:
                        await self._rotate_and_refresh_session(current_key)
                        continue
                    return None
        return None
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    # ── Existing endpoints ──────────────────────────────────

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time quote from Finnhub"""
        url = f"{self.base_url}/quote"
        data = await self._request_with_rotation(url, {"symbol": symbol})
        if not data or data.get('c') == 0:
            raise ValueError(f"No price data for {symbol}")
        return data

    async def get_company_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch recent company news"""
        url = f"{self.base_url}/company-news"
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        try:
            return await self._request_with_rotation(url, {
                "symbol": symbol,
                "from": from_date,
                "to": to_date
            }, timeout=5.0) or []
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    async def get_candles(self, symbol: str, resolution: str = "D", count: int = 100) -> Dict[str, Any]:
        """Fetch historical candles"""
        url = f"{self.base_url}/stock/candle"
        end = int(datetime.now().timestamp())
        start = end - (count * 24 * 3600 if resolution == "D" else count * 3600)
        
        try:
            return await self._request_with_rotation(url, {
                "symbol": symbol,
                "resolution": resolution,
                "from": start,
                "to": end
            }) or {}
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return {}

    # ── NEW: Company Profile ────────────────────────────────

    async def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch company profile — name, sector, market cap, IPO date, etc."""
        url = f"{self.base_url}/stock/profile2"
        try:
            data = await self._request_with_rotation(url, {"symbol": symbol}, timeout=5.0)
            if data and data.get("name"):
                return data
            return None
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {e}")
            return None

    # ── NEW: Basic Financials (Fundamentals) ─────────────────

    async def get_basic_financials(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch fundamental financial metrics — P/E, EPS, beta, 52w range, dividend yield, etc."""
        url = f"{self.base_url}/stock/metric"
        try:
            data = await self._request_with_rotation(url, {
                "symbol": symbol,
                "metric": "all"
            }, timeout=5.0)
            if data and data.get("metric"):
                return data["metric"]
            return None
        except Exception as e:
            logger.error(f"Error fetching basic financials for {symbol}: {e}")
            return None

    # ── NEW: Analyst Recommendation Trends ───────────────────

    async def get_recommendation_trends(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch analyst recommendation trends — strongBuy, buy, hold, sell, strongSell."""
        url = f"{self.base_url}/stock/recommendation"
        try:
            data = await self._request_with_rotation(url, {"symbol": symbol}, timeout=5.0)
            if isinstance(data, list):
                return data[:6]  # Last 6 months
            return []
        except Exception as e:
            logger.error(f"Error fetching recommendations for {symbol}: {e}")
            return []

    # ── NEW: Earnings Calendar (recent) ──────────────────────

    async def get_earnings(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch recent earnings surprises for a symbol."""
        url = f"{self.base_url}/stock/earnings"
        try:
            data = await self._request_with_rotation(url, {"symbol": symbol}, timeout=5.0)
            if isinstance(data, list):
                return data[:4]  # Last 4 quarters
            return []
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return []

    # ── NEW: Peers / Similar Stocks ──────────────────────────

    async def get_peers(self, symbol: str) -> List[str]:
        """Fetch a list of peer/similar company tickers."""
        url = f"{self.base_url}/stock/peers"
        try:
            data = await self._request_with_rotation(url, {"symbol": symbol}, timeout=5.0)
            if isinstance(data, list):
                return [p for p in data if p != symbol][:8]
            return []
        except Exception as e:
            logger.error(f"Error fetching peers for {symbol}: {e}")
            return []


finnhub_service = FinnhubService()

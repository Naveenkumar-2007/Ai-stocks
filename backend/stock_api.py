"""
Stock Data API Module using Twelve Data and Alpha Vantage
Provides reliable stock data access from cloud hosting
WITH INTELLIGENT CACHING TO MINIMIZE API CALLS
"""
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
from cache_manager import get_cache

# Fix Windows console encoding for emoji characters in log messages
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Fallback gracefully if reconfigure not available

# Setup logging
logger = logging.getLogger(__name__)

# Initialize cache manager
cache = get_cache()

# Optional fallback provider
try:
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    yf = None

# Load environment variables
load_dotenv()


# â”€â”€ API Key Rotator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import threading

class KeyRotator:
    """Manages multiple API keys with automatic rotation on rate-limit errors."""
    def __init__(self, name: str, keys: list):
        self.name = name
        self.keys = [k.strip() for k in keys if k.strip()]
        if not self.keys:
            raise RuntimeError(f"No API keys configured for {name}. Update your .env file.")
        self._idx = 0
        self._lock = threading.Lock()
        logger.info(f"ðŸ”‘ {name}: loaded {len(self.keys)} API key(s)")

    @property
    def key(self) -> str:
        """Return the currently active key."""
        with self._lock:
            return self.keys[self._idx % len(self.keys)]

    def rotate(self, failed_key: str = None) -> str:
        """Move to next key and return it. Only rotates if the caller's key is still active."""
        with self._lock:
            if failed_key and self.keys[self._idx % len(self.keys)] != failed_key:
                return self.keys[self._idx % len(self.keys)]  # already rotated
            self._idx += 1
            new_key = self.keys[self._idx % len(self.keys)]
            logger.info(f"ðŸ”„ {self.name}: rotated to key #{(self._idx % len(self.keys)) + 1}/{len(self.keys)}")
            return new_key

    def __len__(self):
        return len(self.keys)


def _load_keys(env_plural: str, env_singular: str) -> list:
    """Load keys from comma-separated plural var, falling back to singular var."""
    keys_str = os.getenv(env_plural, '')
    if keys_str:
        return keys_str.split(',')
    single = os.getenv(env_singular, '')
    return [single] if single else []


# Initialize key rotators
twelve_data_rotator = KeyRotator('TwelveData', _load_keys('TWELVE_DATA_API_KEYS', 'TWELVE_DATA_API_KEY'))
finnhub_rotator     = KeyRotator('Finnhub',    _load_keys('FINNHUB_API_KEYS',     'FINNHUB_API_KEY'))
alpha_keys = _load_keys('ALPHA_VANTAGE_API_KEYS', 'ALPHA_VANTAGE_API_KEY')
alpha_vantage_rotator = KeyRotator('AlphaVantage', alpha_keys) if alpha_keys else None

# Backward-compatible variable names (used throughout the file)
TWELVE_DATA_API_KEY = twelve_data_rotator.key
BASE_URL = 'https://api.twelvedata.com'
FINNHUB_API_KEY = finnhub_rotator.key
FINNHUB_BASE_URL = 'https://finnhub.io/api/v1'
ALPHA_VANTAGE_API_KEY = alpha_vantage_rotator.key if alpha_vantage_rotator else os.getenv('ALPHA_VANTAGE_API_KEY', '')

YFINANCE_EXCHANGE_SUFFIXES = {
    'NSE': '.NS',
    'NFO': '.NS',
    'BSE': '.BO',
    'BFO': '.BO',
    'BOMBAY STOCK EXCHANGE': '.BO',
    'NATIONAL STOCK EXCHANGE OF INDIA': '.NS',
    'LSE': '.L',
    'XLON': '.L',
    'LONDON STOCK EXCHANGE': '.L',
    'HKEX': '.HK',
    'HKSE': '.HK',
    'HKEX - HONG KONG': '.HK',
    'TSX': '.TO',
    'TSXV': '.V',
    'ASX': '.AX',
    'SGX': '.SI',
    'SSE': '.SS',
    'SZSE': '.SZ',
    'JPX': '.T',
    'TSE': '.T',
    'KRX': '.KS',
    'KOSDAQ': '.KQ',
    'FWB': '.F',
    'SWB': '.SW',
    'SIX': '.SW',
    'EURONEXT': '.PA',
    'EPA': '.PA',
    'BME': '.MC',
    'Borsa Italiana': '.MI'
}

COUNTRY_SUFFIX_FALLBACKS = {
    'india': ['.NS', '.BO'],
    'canada': ['.TO', '.V'],
    'united kingdom': ['.L'],
    'australia': ['.AX'],
    'hong kong': ['.HK'],
    'japan': ['.T'],
    'china': ['.SS', '.SZ'],
    'germany': ['.DE', '.F'],
    'france': ['.PA'],
    'spain': ['.MC'],
    'italy': ['.MI'],
    'switzerland': ['.SW'],
    'singapore': ['.SI'],
    'south korea': ['.KS', '.KQ']
}

DEFAULT_SUFFIX_FALLBACKS = ['.NS', '.BO', '.L', '.HK', '.TO']


# â”€â”€ Dynamic key getters (always return the currently active key) â”€â”€
def _get_twelve_key():
    return twelve_data_rotator.key

def _get_finnhub_key():
    return finnhub_rotator.key

def _get_alpha_key():
    return alpha_vantage_rotator.key if alpha_vantage_rotator else ALPHA_VANTAGE_API_KEY


def _request_with_rotation(rotator, url, params, apikey_param='apikey', timeout=30):
    """
    Make a GET request with automatic key rotation on rate-limit errors.
    Tries each key in the rotator before giving up.
    Returns the requests.Response object on success.
    """
    attempts = len(rotator)
    last_error = None

    for attempt in range(attempts):
        current_key = rotator.key
        params[apikey_param] = current_key
        try:
            response = requests.get(url, params=params, timeout=timeout)

            # HTTP 429 = rate limited
            if response.status_code == 429:
                logger.warning(f"âš ï¸ {rotator.name}: rate limited (429), rotating key...")
                rotator.rotate(current_key)
                continue

            # Some APIs return 200 with rate-limit message in body
            if response.status_code == 200:
                try:
                    body = response.json()
                    msg = str(body.get('message', '') or body.get('error', '') or body.get('note', '')).lower()
                    if any(w in msg for w in ['rate limit', 'api limit', 'too many', 'exceeded', 'throttl']):
                        logger.warning(f"âš ï¸ {rotator.name}: API rate-limit in response body, rotating key...")
                        rotator.rotate(current_key)
                        continue
                except (ValueError, AttributeError):
                    pass

            return response

        except requests.exceptions.Timeout:
            last_error = f"Timeout on attempt {attempt + 1}"
            rotator.rotate(current_key)
            continue
        except Exception as e:
            last_error = str(e)
            break

    raise requests.exceptions.RequestException(
        f"{rotator.name}: all {attempts} keys exhausted. Last error: {last_error}"
    )


def _normalize_symbol(symbol: str) -> str:
    """Return a cleaned, uppercase base symbol without exchange suffixes."""
    if not symbol:
        return ''
    base = symbol.strip().upper()
    if ':' in base:
        base = base.split(':')[0]
    return base.replace(' ', '')


def _yfinance_variant_candidates(symbol: str, exchange: str | None, country: str | None) -> list[str]:
    """Generate likely Yahoo Finance ticker variants for a given symbol."""
    base = _normalize_symbol(symbol)
    variants: list[str] = []
    seen: set[str] = set()

    def push(candidate: str):
        normalized = candidate.upper()
        if normalized and normalized not in seen:
            variants.append(normalized)
            seen.add(normalized)

    push(symbol.upper())
    push(base)

    suffixes: list[str] = []
    if exchange:
        exchange_upper = exchange.upper()
        for key, suffix in YFINANCE_EXCHANGE_SUFFIXES.items():
            if exchange_upper == key or exchange_upper in key:
                if suffix not in suffixes:
                    suffixes.append(suffix)

    if country:
        country_suffixes = COUNTRY_SUFFIX_FALLBACKS.get(country.lower(), [])
        for suffix in country_suffixes:
            if suffix not in suffixes:
                suffixes.append(suffix)

    if country and country.lower() == 'india':
        for suffix in ('.NS', '.BO'):
            if suffix not in suffixes:
                suffixes.append(suffix)

    for suffix in DEFAULT_SUFFIX_FALLBACKS:
        if suffix not in suffixes:
            suffixes.append(suffix)

    suffixes.append('')  # Ensure bare symbol attempt at the end

    for suffix in suffixes:
        candidate = base if not suffix else f"{base}{suffix}"
        push(candidate)

    return variants


def _get_stock_history_yfinance(
    symbol: str,
    days: int = 60,
    exchange: str | None = None,
    country: str | None = None
) -> tuple[pd.DataFrame, str | None, str | None]:
    """Attempt to fetch historical data using Yahoo Finance as a fallback."""
    if yf is None:
        return pd.DataFrame(), None, 'yfinance package is not installed'

    variants = _yfinance_variant_candidates(symbol, exchange, country)
    start = datetime.now() - timedelta(days=max(days + 30, 365))
    end = datetime.now()
    errors: list[str] = []

    for variant in variants:
        try:
            ticker = yf.Ticker(variant)
            history = ticker.history(start=start, end=end, interval='1d', auto_adjust=False)
            if history.empty:
                errors.append(f'{variant}: empty response')
                continue

            required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
            if not required_cols.issubset(history.columns):
                errors.append(f'{variant}: missing expected columns')
                continue

            df = history[list(required_cols)].copy()
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            df.sort_index(inplace=True)

            for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[column] = pd.to_numeric(df[column], errors='coerce')

            df.dropna(subset=['Close'], inplace=True)
            if df.empty:
                errors.append(f'{variant}: all close values were NaN')
                continue

            df['Dividends'] = 0.0
            df['Stock Splits'] = 0.0
            df.index.name = 'datetime'

            print(f"Using yfinance fallback for {variant}. Rows fetched: {len(df)}")
            return df, variant, None

        except Exception as exc:  # pragma: no cover - network dependent
            errors.append(f'{variant}: {exc}')
            continue

    return pd.DataFrame(), None, '; '.join(errors) if errors else 'No fallback variants succeeded'


def search_symbols(query: str, limit: int = 5):
    """Search for symbols across global exchanges using Twelve Data."""
    try:
        if not query:
            return []

        url = f'{BASE_URL}/symbol_search'
        params = {
            'symbol': query,
            'outputsize': max(1, min(limit, 30)),
            'apikey': _get_twelve_key()
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json() or {}
        data = payload.get('data') or []

        results = []
        for entry in data[:limit]:
            results.append({
                'symbol': entry.get('symbol', '').upper(),
                'name': entry.get('instrument_name') or entry.get('name') or entry.get('symbol', ''),
                'exchange': entry.get('exchange', ''),
                'country': entry.get('country', ''),
                'currency': entry.get('currency', '')
            })

        return results

    except Exception as exc:
        print(f"Error searching symbols: {exc}")
        return []

def get_stock_history(
    ticker,
    days=60,
    interval='1day',
    exchange=None,
    country=None,
    return_info=False
):
    """
    Fetch historical stock data with automatic fallback providers when available.
    USES INTELLIGENT CACHING TO MINIMIZE API CALLS (1 hour cache TTL)

    Args:
        ticker (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
        days (int): Number of days of historical data (default: 60)
        interval (str): Time interval - '1min', '5min', '15min', '30min', '1h', '1day', '1week', '1month'
        exchange (str | None): Optional exchange code for improved fallback mapping
        country (str | None): Optional country name for improved fallback mapping
        return_info (bool): When True, returns tuple (DataFrame, metadata dict)

    Returns:
        pd.DataFrame or (pd.DataFrame, dict): Historical data with columns [Open, High, Low, Close, Volume].
        Returns empty DataFrame if all providers fail.
    """
    
    # Check cache first (1 hour TTL for historical data)
    cache_params = {
        'ticker': ticker,
        'days': days,
        'interval': interval,
        'exchange': exchange,
        'country': country
    }
    
    cached_data = cache.get('stock_history', cache_params, ttl_seconds=3600)  # 1 hour cache
    if cached_data:
        df = pd.DataFrame(cached_data['dataframe'])
        if not df.empty and 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        if return_info:
            return df, cached_data.get('info', {})
        return df

    info = {
        'symbol': ticker.upper() if isinstance(ticker, str) else ticker,
        'source': 'twelvedata',
        'provider_message': None
    }

    data_frame = pd.DataFrame()
    provider_error = None

    try:
        print(f"ðŸ“Š Fetching {ticker} data from Twelve Data API...")

        url = f'{BASE_URL}/time_series'
        params = {
            'symbol': ticker,
            'interval': interval,
            'outputsize': min(days, 5000),  # Max 5000 data points
            'format': 'JSON'
        }

        # Use key rotation for automatic failover on rate limits
        response = _request_with_rotation(twelve_data_rotator, url, params, apikey_param='apikey', timeout=30)
        response.raise_for_status()
        payload = response.json()

        if payload.get('status') == 'error':
            provider_error = payload.get('message', 'Unknown error')
            print(f"âŒ API Error: {provider_error}")
        elif 'values' not in payload:
            provider_error = 'No data returned from Twelve Data'
            print(f"âŒ No data returned for {ticker}")
        else:
            data_frame = pd.DataFrame(payload['values'])
            data_frame['datetime'] = pd.to_datetime(data_frame['datetime'])
            data_frame.set_index('datetime', inplace=True)
            data_frame.sort_index(inplace=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data_frame.columns:
                    data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce')

            data_frame.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)

            data_frame['Dividends'] = 0.0
            data_frame['Stock Splits'] = 0.0

            print(f"âœ… Successfully fetched {len(data_frame)} data points for {ticker}")
            if not data_frame.empty:
                print(
                    f"ðŸ“… Date range: {data_frame.index[0].strftime('%Y-%m-%d')} "
                    f"to {data_frame.index[-1].strftime('%Y-%m-%d')}"
                )
                print(f"ðŸ’µ Latest price: ${data_frame['Close'].iloc[-1]:.2f}")
                
                # Cache the successful result
                cache_entry = {
                    'dataframe': data_frame.reset_index().to_dict('records'),
                    'info': info
                }
                cache.set('stock_history', cache_params, cache_entry)
                print(f"ðŸ’¾ Cached data for {ticker}")

    except requests.exceptions.RequestException as exc:
        provider_error = f'Network error fetching {ticker}: {exc}'
        print(provider_error)
    except Exception as exc:  # pragma: no cover - network dependent
        provider_error = f'Error processing {ticker} data: {exc}'
        print(provider_error)

    # Attempt fallbacks when Twelve Data fails or returns insufficient data
    needs_fallback = data_frame.empty or len(data_frame) < 2
    allow_fallback = str(interval).lower() in {'1day', '1d', 'daily'}

    # FIRST FALLBACK: Try Finnhub API (with key rotation)
    if needs_fallback and allow_fallback and finnhub_rotator:
        logger.info(f"ðŸ”„ Trying Finnhub fallback for {ticker}...")
        print(f"Twelve Data unavailable for {ticker}, trying Finnhub...")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)
            
            # Finnhub candle endpoint â€” use key rotation
            finnhub_url = f"{FINNHUB_BASE_URL}/stock/candle"
            params = {
                'symbol': ticker.upper(),
                'resolution': 'D',  # Daily resolution
                'from': int(start_date.timestamp()),
                'to': int(end_date.timestamp()),
            }
            
            print(f"  Trying Finnhub with {ticker} (with key rotation)...")
            response = _request_with_rotation(finnhub_rotator, finnhub_url, params, apikey_param='token', timeout=15)
            finnhub_data = response.json()
            
            if finnhub_data.get('s') == 'ok' and finnhub_data.get('c'):
                # Finnhub returns arrays: c=close, h=high, l=low, o=open, v=volume, t=timestamp
                df_data = []
                for i in range(len(finnhub_data['c'])):
                    df_data.append({
                        'datetime': datetime.fromtimestamp(finnhub_data['t'][i]),
                        'Open': float(finnhub_data['o'][i]),
                        'High': float(finnhub_data['h'][i]),
                        'Low': float(finnhub_data['l'][i]),
                        'Close': float(finnhub_data['c'][i]),
                        'Volume': int(finnhub_data['v'][i])
                    })
                
                if df_data:
                    finnhub_df = pd.DataFrame(df_data)
                    finnhub_df.set_index('datetime', inplace=True)
                    finnhub_df.sort_index(inplace=True)
                    
                    # Limit to requested days
                    if days and len(finnhub_df) > days:
                        finnhub_df = finnhub_df.tail(days)
                    
                    # Add required columns
                    finnhub_df['Dividends'] = 0.0
                    finnhub_df['Stock Splits'] = 0.0
                    
                    data_frame = finnhub_df
                    info['source'] = 'finnhub'
                    needs_fallback = False
                    
                    logger.info(f"âœ… Finnhub: Successfully fetched {len(data_frame)} data points for {ticker}")
                    print(f"âœ… Finnhub: Successfully fetched {len(data_frame)} data points for {ticker}")
                    
                    # Cache the successful result
                    cache_entry = {
                        'dataframe': data_frame.reset_index().to_dict('records'),
                        'info': info
                    }
                    cache.set('stock_history', cache_params, cache_entry)
                    print(f"ðŸ’¾ Cached Finnhub data for {ticker}")
            else:
                error_msg = finnhub_data.get('error', 'No data available')
                print(f"  Finnhub: {error_msg}")
                
        except Exception as e:
            logger.error(f"âŒ Finnhub error for {ticker}: {str(e)}")
            print(f"  Finnhub error: {str(e)}")

    # SECOND FALLBACK: Try Alpha Vantage API (with key rotation)
    if needs_fallback and allow_fallback and alpha_vantage_rotator:
        logger.info(f"ðŸ”„ Trying Alpha Vantage fallback for {ticker}...")
        print(f"Finnhub unavailable for {ticker}, trying Alpha Vantage as last resort...")
        
        try:
            av_url = "https://www.alphavantage.co/query"
            av_params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'outputsize': 'compact'  # Last 100 data points
            }
            
            print(f"  Trying Alpha Vantage with {ticker} (with key rotation)...")
            response = _request_with_rotation(alpha_vantage_rotator, av_url, av_params, apikey_param='apikey', timeout=15)
            av_data = response.json()
            
            if 'Time Series (Daily)' in av_data:
                time_series = av_data['Time Series (Daily)']
                
                # Convert to DataFrame
                df_data = []
                for date_str, values in time_series.items():
                    df_data.append({
                        'datetime': datetime.strptime(date_str, '%Y-%m-%d'),
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': int(values['5. volume'])
                    })
                
                if df_data:
                    fallback_df = pd.DataFrame(df_data)
                    fallback_df.set_index('datetime', inplace=True)
                    fallback_df.sort_index(inplace=True)
                    
                    # Limit to requested days
                    if days and len(fallback_df) > days:
                        fallback_df = fallback_df.tail(days)
                    
                    # Rename columns to match expected format
                    fallback_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    
                    # Add required columns
                    fallback_df['Dividends'] = 0.0
                    fallback_df['Stock Splits'] = 0.0
                    
                    data_frame = fallback_df
                    info['source'] = 'alphavantage'
                    needs_fallback = False
                    
                    logger.info(f"âœ… Alpha Vantage: Successfully fetched {len(data_frame)} data points for {ticker}")
                    print(f"âœ… Alpha Vantage: Successfully fetched {len(data_frame)} data points for {ticker}")
                    
                    # Cache the successful result
                    cache_entry = {
                        'dataframe': data_frame.reset_index().to_dict('records'),
                        'info': info
                    }
                    cache.set('stock_history', cache_params, cache_entry)
                    print(f"ðŸ’¾ Cached Alpha Vantage data for {ticker}")
                    
            else:
                error_msg = av_data.get('Note') or av_data.get('Error Message')
                if error_msg:
                    print(f"  Alpha Vantage: {error_msg}")
                
        except Exception as e:
            logger.error(f"âŒ Alpha Vantage error for {ticker}: {str(e)}")
            print(f"  Alpha Vantage error: {str(e)}")
    
    # THIRD FALLBACK: Try Yahoo Finance (Strongest for international/NSE stocks)
    if needs_fallback and allow_fallback:
        logger.info(f"ðŸ”„ Trying Yahoo Finance fallback for {ticker}...")
        print(f"Previous providers failed for {ticker}, trying Yahoo Finance...")
        
        try:
            yf_df, yf_symbol, yf_error = _get_stock_history_yfinance(
                ticker, days=days, exchange=exchange, country=country
            )
            
            if not yf_df.empty:
                data_frame = yf_df
                info['source'] = 'yfinance'
                info['symbol'] = yf_symbol
                needs_fallback = False
                
                logger.info(f"âœ… Yahoo Finance: Successfully fetched {len(data_frame)} data points for {yf_symbol}")
                print(f"âœ… Yahoo Finance: Successfully fetched {len(data_frame)} data points for {yf_symbol}")
                
                # Cache the successful result
                cache_entry = {
                    'dataframe': data_frame.reset_index().to_dict('records'),
                    'info': info
                }
                cache.set('stock_history', cache_params, cache_entry)
                print(f"ðŸ’¾ Cached Yahoo Finance data for {ticker}")
            else:
                logger.warning(f"âŒ Yahoo Finance failed for {ticker}: {yf_error}")
                print(f"  Yahoo Finance error: {yf_error}")
        except Exception as e:
            logger.error(f"âŒ Yahoo Finance critical error for {ticker}: {str(e)}")
            print(f"  Yahoo Finance critical error: {str(e)}")

    if needs_fallback:
        # If both Twelve Data and Alpha Vantage failed, try to return stale cache data
        # Check cache again without TTL restriction
        cached_data_stale = cache.get('stock_history', cache_params, ttl_seconds=None)  # No TTL check
        if cached_data_stale and cached_data_stale.get('dataframe'):
            df = pd.DataFrame(cached_data_stale['dataframe'])
            if not df.empty and 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                data_frame = df
                info['source'] = 'cache-stale'
                info['provider_message'] = f"Using cached data for {ticker} (APIs temporarily unavailable)"
                print(f"âš ï¸ Using stale cached data for {ticker} - APIs exhausted")
                needs_fallback = False
        
        # If still no data, provide user-friendly error
        if needs_fallback:
            info['provider_message'] = f"Unable to retrieve data for {ticker}. Please verify the stock symbol is correct."
    
    if not needs_fallback and info['provider_message'] is None and provider_error:
        # Log provider error for debugging but don't expose to user
        logger.warning(f"Provider error (not shown to user): {provider_error}")
        info['provider_message'] = None  # Don't expose internal errors

    if return_info:
        return data_frame, info
    return data_frame



def fetch_stock_data_cached(ticker, days=60):
    """
    Fetch stock data with intelligent caching.
    Wrapper for get_stock_history to return data in the format expected by model_trainer.py
    
    Args:
        ticker (str): Stock symbol
        days (int): Number of days of historical data
        
    Returns:
        dict: Format expected by model_trainer.py:
        {
            'historical_data': {
                'dates': [...],
                'prices': [...]
            }
        }
    """
    try:
        df = get_stock_history(ticker, days=days)
        
        if df.empty:
            return None
            
        return {
            'historical_data': {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'prices': df['Close'].tolist()
            }
        }
    except Exception as e:
        logger.error(f"Error in fetch_stock_data_cached for {ticker}: {e}")
        return None


def get_intraday_data(ticker, interval='5min', outputsize=78):
    """
    Fetch intraday stock data - tries multiple intervals to get the best available data
    
    Args:
        ticker (str): Stock symbol
        interval (str): Time interval ('1min', '5min', '15min', '30min', '1h')
        outputsize (int): Number of data points (default: 78 = full day of 5min data)
    
    Returns:
        pd.DataFrame: Intraday stock data
    """
    # Try different intervals in order of preference
    intervals_to_try = [interval, '1min', '5min', '15min', '30min', '1h']
    
    for try_interval in intervals_to_try:
        try:
            print(f"Fetching intraday data for {ticker} with {try_interval} interval...")
            
            url = f'{BASE_URL}/time_series'
            params = {
                'symbol': ticker,
                'interval': try_interval,
                'outputsize': outputsize,
                'apikey': _get_twelve_key(),
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'values' not in data or not data['values']:
                print(f"No data for {try_interval}, trying next interval...")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.sort_index()
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Rename columns
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            # Filter to today's data only (if available)
            today = datetime.now().date()
            df_today = df[df.index.date == today]
            
            if not df_today.empty:
                print(f"Fetched {len(df_today)} intraday data points for today ({try_interval})")
                return df_today
            else:
                print(f"Fetched {len(df)} recent intraday data points ({try_interval})")
                return df
            
        except Exception as e:
            print(f"Error with {try_interval}: {e}")
            continue
    
    # If all intervals fail, return empty DataFrame
    print(f"No intraday data available for {ticker}")
    return pd.DataFrame()


def get_real_time_price(ticker):
    """
    Get real-time price quote
    
    Args:
        ticker (str): Stock symbol
    
    Returns:
        dict: Real-time price data
    """
    try:
        url = f'{BASE_URL}/quote'
        params = {
            'symbol': ticker,
            'apikey': _get_twelve_key()
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        return {
            'price': float(data.get('close', 0)),
            'open': float(data.get('open', 0)),
            'high': float(data.get('high', 0)),
            'low': float(data.get('low', 0)),
            'volume': int(data.get('volume', 0)),
            'timestamp': data.get('datetime', '')
        }
        
    except Exception as e:
        print(f" Error fetching real-time price: {e}")
        return None


def get_stock_fundamentals(ticker):
    """
    Get stock fundamental data (company info, market cap, P/E ratio)
    
    Args:
        ticker (str): Stock symbol
    
    Returns:
        dict: Fundamental data including market cap, P/E ratio, company name
    """
    try:
        print(f" Fetching fundamentals for {ticker}...")
        
        # Get statistics endpoint for fundamentals
        url = f'{BASE_URL}/statistics'
        params = {
            'symbol': ticker,
            'apikey': _get_twelve_key()
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        stats = response.json()
        
        # Get company profile/logo endpoint for company name
        profile_url = f'{BASE_URL}/profile'
        profile_params = {
            'symbol': ticker,
            'apikey': _get_twelve_key()
        }
        
        profile_response = requests.get(profile_url, params=profile_params, timeout=30)
        profile_data = {}
        if profile_response.status_code == 200:
            profile_data = profile_response.json()
        
        # Extract data
        company_name = profile_data.get('name', ticker)
        market_cap = stats.get('statistics', {}).get('valuations_metrics', {}).get('market_capitalization', None)
        pe_ratio = stats.get('statistics', {}).get('valuations_metrics', {}).get('trailing_pe', None)
        
        # Twelve Data appears to return market_cap in raw USD units (trillions for AAPL).
        if market_cap is not None:
            try:
                market_cap = float(market_cap)
            except (ValueError, TypeError):
                pass
        
        # Also try from quote endpoint as fallback
        if not market_cap or not pe_ratio:
            quote_url = f'{BASE_URL}/quote'
            quote_params = {
                'symbol': ticker,
                'apikey': _get_twelve_key()
            }
            quote_response = requests.get(quote_url, params=quote_params, timeout=30)
            if quote_response.status_code == 200:
                quote_data = quote_response.json()
                if not company_name or company_name == ticker:
                    company_name = quote_data.get('name', ticker)
                # Note: /quote might return different units, but we prioritize /statistics
        
        print(f"Fundamentals: {company_name}, Market Cap: {market_cap}, P/E: {pe_ratio}")
        
        return {
            'company_name': company_name,
            'market_cap': market_cap,
            'pe_ratio': pe_ratio
        }
        
    except Exception as e:
        print(f"Error fetching fundamentals: {e}")
        return {
            'company_name': ticker,
            'market_cap': None,
            'pe_ratio': None
        }


# Test function
if __name__ == "__main__":
    print("="*60)
    print("Testing Twelve Data API")
    print("="*60)
    
    # Test with popular stocks
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    for ticker in tickers:
        hist = get_stock_history(ticker, days=60)
        if not hist.empty:
            print(f"\n{ticker} - Last 5 days:")
            print(hist[['Open', 'High', 'Low', 'Close', 'Volume']].tail())
        else:
            print(f"\n {ticker} - Failed to fetch data")
        print("-"*60)


# ===== FINNHUB API FUNCTIONS =====

def get_company_news(ticker, days=7):
    """
    Fetch company news from Finnhub API - ONLY news specifically about the searched stock
    Uses company logo for ALL news articles (with fallback to default logo)
    USES INTELLIGENT CACHING (2 hour TTL to minimize API calls)
    
    Args:
        ticker (str): Stock symbol
        days (int): Number of days of news to fetch (default: 7)
    
    Returns:
        list: List of news articles with title, summary, url, source, image, and timestamp
    """
    
    # Check cache first (2 hour TTL for news)
    cache_params = {'ticker': ticker, 'days': days}
    cached_news = cache.get('company_news', cache_params, ttl_seconds=7200)  # 2 hour cache
    if cached_news:
        print(f"ðŸ’¾ Using cached news for {ticker}")
        return cached_news
    
    try:
        print(f"ðŸ“° Fetching news for {ticker} from Finnhub API only...")
        
        # Get company profile to get company name AND logo
        company_name = None
        company_logo = None
        try:
            profile_url = f'{FINNHUB_BASE_URL}/stock/profile2'
            profile_params = {'symbol': ticker, 'token': _get_finnhub_key()}
            profile_response = requests.get(profile_url, params=profile_params, timeout=3)
            if profile_response.status_code == 200:
                profile_data = profile_response.json()
                company_name = profile_data.get('name', '').lower()
                company_logo = profile_data.get('logo', '')
                
                # If no logo from profile, try alternative logo URL
                if not company_logo:
                    company_logo = f"https://static2.finnhub.io/file/publicdatany/finnhubimage/stock_logo/{ticker.upper()}.png"
                
                print(f"Company: {company_name}, Logo: {company_logo}")
        except Exception as e:
            print(f"Could not fetch company profile: {e}")
            # Use default logo URL even if profile fails
            company_logo = f"https://static2.finnhub.io/file/publicdatany/finnhubimage/stock_logo/{ticker.upper()}.png"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch news from Finnhub API ONLY (no Yahoo Finance)
        url = f'{FINNHUB_BASE_URL}/company-news'
        params = {
            'symbol': ticker,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'token': _get_finnhub_key()
        }
        
        print(f"Using Finnhub API: {url} (No Yahoo Finance)")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        news_data = response.json()
        
        if not news_data:
            print(f"No news found for {ticker}")
            return []
        
        # Filter and format news articles - Be more lenient with relevance filtering
        news_articles = []
        ticker_lower = ticker.lower()
        
        for article in news_data[:15]:  # Check first 15 articles
            headline = article.get('headline', '').lower()
            summary = article.get('summary', '').lower()
            
            # Check if article is relevant to the searched stock
            is_relevant = False
            
            # Check if ticker symbol is mentioned (case insensitive)
            if ticker_lower in headline or ticker_lower in summary:
                is_relevant = True
            
            # Check if company name is mentioned (if we have it)
            if company_name and len(company_name) > 3:  # Avoid short names causing false matches
                if company_name in headline or company_name in summary:
                    is_relevant = True
            
            # If no company name match, be more lenient - include articles from the same sector
            # Or just include all articles from the API response since Finnhub already filters by ticker
            if not is_relevant and len(news_articles) < 5:  # At least show 5 news items
                is_relevant = True  # Include all Finnhub company news since it's already filtered
            
            # Only add relevant articles
            if is_relevant:
                # Get timestamp and format date properly
                timestamp = article.get('datetime', 0)
                try:
                    # Convert Unix timestamp to readable date
                    if timestamp > 0:
                        date_obj = datetime.fromtimestamp(timestamp)
                        formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
                    else:
                        formatted_date = 'Unknown date'
                except Exception as e:
                    print(f"Error formatting date: {e}")
                    formatted_date = 'Unknown date'
                
                news_articles.append({
                    'headline': article.get('headline', 'No headline'),
                    'summary': article.get('summary', 'No summary available'),
                    'source': 'Finnhub',  # Always show Finnhub as source (no Yahoo)
                    'url': article.get('url', '#'),
                    'image': company_logo,  # ALWAYS use company logo for consistency
                    'datetime': formatted_date,  # Properly formatted date
                    'timestamp': timestamp,  # Keep original timestamp for sorting
                    'related': ticker.upper(),  # Show the ticker being searched
                    'company_name': company_name or ticker  # Include company name
                })
            
            # Limit to 30 most recent relevant articles to support "Load more"
            if len(news_articles) >= 30:
                break
        
        print(f"âœ… Fetched {len(news_articles)} relevant news articles for {ticker} from Finnhub only")
        print(f"All articles using company logo: {company_logo}")
        print(f"All data from Finnhub API - No Yahoo Finance used")
        
        # Cache the results (2 hour TTL)
        cache.set('company_news', cache_params, news_articles)
        print(f"ðŸ’¾ Cached news for {ticker}")
        
        return news_articles
        
    except Exception as e:
        print(f"Error fetching news from Finnhub API: {e}")
        return []


def get_sentiment_analysis(ticker):
    """
    Fetch sentiment analysis from Finnhub API
    USES INTELLIGENT CACHING (4 hour TTL - sentiment changes slowly)
    
    Args:
        ticker (str): Stock symbol
    
    Returns:
        dict: Sentiment data including overall sentiment, score, and breakdown
    """
    
    # Check cache first (4 hour TTL for sentiment)
    cache_params = {'ticker': ticker}
    cached_sentiment = cache.get('sentiment_analysis', cache_params, ttl_seconds=14400)  # 4 hour cache
    if cached_sentiment:
        print(f"ðŸ’¾ Using cached sentiment for {ticker}")
        return cached_sentiment
    
    try:
        print(f"ðŸŽ¯ Fetching sentiment analysis for {ticker} from Finnhub...")
        
        # Get news sentiment
        url = f'{FINNHUB_BASE_URL}/news-sentiment'
        params = {
            'symbol': ticker,
            'token': _get_finnhub_key()
        }
        
        response = requests.get(url, params=params, timeout=3)
        
        # If news-sentiment not available, analyze company news
        if response.status_code != 200 or not response.json():
            print("Using alternative sentiment calculation from news...")
            return calculate_sentiment_from_news(ticker)
        
        sentiment_data = response.json()
        
        # Extract sentiment metrics
        buzz = sentiment_data.get('buzz', {})
        sentiment = sentiment_data.get('sentiment', {})
        
        # Finnhub returns bullishPercent/bearishPercent as FRACTIONS (0.0 to 1.0)
        # e.g. bullishPercent=0.65 means 65% bullish
        # Score = bullish - bearish, already in -1 to +1 range
        bullish_frac = sentiment.get('bullishPercent', 0)
        bearish_frac = sentiment.get('bearishPercent', 0)
        
        # If Finnhub returns 0 for both, fall back to news-based analysis
        if bullish_frac == 0 and bearish_frac == 0:
            print(f"Finnhub returned zero sentiment for {ticker}, falling back to news...")
            return calculate_sentiment_from_news(ticker)
        
        overall_score = bullish_frac - bearish_frac  # Already -1..+1
        
        if overall_score > 0.6:
            sentiment_label = 'STRONG BUY'
            sentiment_class = 'positive'
        elif overall_score > 0.2:
            sentiment_label = 'BUY'
            sentiment_class = 'positive'
        elif overall_score > -0.2:
            sentiment_label = 'HOLD'
            sentiment_class = 'neutral'
        elif overall_score > -0.6:
            sentiment_label = 'SELL'
            sentiment_class = 'negative'
        else:
            sentiment_label = 'STRONG SELL'
            sentiment_class = 'negative'
        
        # Convert fractions to display percentages (0-100)
        bullish_display = round(bullish_frac * 100, 2)
        bearish_display = round(bearish_frac * 100, 2)
        
        result = {
            'sentiment': sentiment_label,
            'sentiment_class': sentiment_class,
            'score': round(overall_score, 2),
            'bullish_percent': bullish_display,
            'bearish_percent': bearish_display,
            'buzz_articles': buzz.get('articlesInLastWeek', 0),
            'buzz_score': buzz.get('buzz', 0)
        }
        
        print(f"âœ… Sentiment: {sentiment_label} (Score: {overall_score:.2f}, Bull: {bullish_display}%, Bear: {bearish_display}%)")
        
        # Cache the result (4 hour TTL)
        cache.set('sentiment_analysis', cache_params, result)
        print(f"ðŸ’¾ Cached sentiment for {ticker}")
        
        return result
        
    except Exception as e:
        print(f"Error fetching sentiment from Finnhub: {e}")
        return calculate_sentiment_from_news(ticker)


def calculate_sentiment_from_news(ticker):
    """
    Calculate sentiment from news headlines using per-headline scoring.
    Uses expanded keyword lists and weights headlines more than summaries.
    
    Args:
        ticker (str): Stock symbol
    
    Returns:
        dict: Calculated sentiment data with score in -1..+1 range
    """
    try:
        news = get_company_news(ticker, days=7)
        
        if not news:
            return {
                'sentiment': 'NEUTRAL',
                'sentiment_class': 'neutral',
                'score': 0,
                'bullish_percent': 50,
                'bearish_percent': 50,
                'buzz_articles': 0,
                'buzz_score': 0
            }
        
        # Expanded keyword lists for financial news sentiment
        positive_keywords = [
            'surge', 'surges', 'surging', 'gain', 'gains', 'profit', 'profits',
            'growth', 'beat', 'beats', 'success', 'bullish', 'rise', 'rises',
            'rising', 'strong', 'outperform', 'outperforms', 'expansion',
            'dividend', 'buyback', 'upgrade', 'upgraded', 'positive', 'momentum',
            'record', 'exceed', 'exceeds', 'optimistic', 'rally', 'rallies',
            'rallying', 'soar', 'soars', 'soaring', 'boost', 'boosts',
            'improve', 'improves', 'improved', 'innovation', 'breakthrough',
            'partnership', 'acquisition', 'launch', 'launches', 'revenue',
            'upside', 'outpace', 'winner', 'winning', 'accelerate', 'accelerating',
            'recover', 'recovery', 'rebound', 'rebounds', 'robust', 'confident',
            'upbeat', 'higher', 'expand', 'expands', 'milestone', 'top',
            'tops', 'topping', 'advance', 'advances', 'advancing'
        ]
        negative_keywords = [
            'fall', 'falls', 'falling', 'loss', 'losses', 'decline', 'declines',
            'declining', 'miss', 'misses', 'weak', 'weakness', 'bearish',
            'drop', 'drops', 'dropping', 'underperform', 'underperforms',
            'recession', 'debt', 'lawsuit', 'lawsuits', 'investigation',
            'negative', 'warning', 'warns', 'layoff', 'layoffs', 'downgrade',
            'downgraded', 'slump', 'slumps', 'pessimistic', 'crash', 'crashes',
            'plunge', 'plunges', 'plunging', 'sink', 'sinks', 'sinking',
            'tumble', 'tumbles', 'tumbling', 'cut', 'cuts', 'cutting',
            'risk', 'risks', 'risky', 'concern', 'concerns', 'worried',
            'fear', 'fears', 'sell-off', 'selloff', 'volatile', 'volatility',
            'struggle', 'struggles', 'struggling', 'disappointing', 'disappoint',
            'lower', 'shrink', 'shrinks', 'deficit', 'penalty', 'fine',
            'fraud', 'scandal', 'bankruptcy', 'default', 'collapse'
        ]
        
        # Per-headline scoring: each article gets a score from -1 to +1
        article_scores = []
        
        for article in news:
            headline_lower = article.get('headline', '').lower()
            summary_lower = article.get('summary', '').lower()
            
            # Count matches â€” headlines weighted 2x more than summaries
            headline_pos = sum(1 for kw in positive_keywords if kw in headline_lower)
            headline_neg = sum(1 for kw in negative_keywords if kw in headline_lower)
            summary_pos = sum(1 for kw in positive_keywords if kw in summary_lower)
            summary_neg = sum(1 for kw in negative_keywords if kw in summary_lower)
            
            pos_score = (headline_pos * 2) + summary_pos
            neg_score = (headline_neg * 2) + summary_neg
            total = pos_score + neg_score
            
            if total > 0:
                # Article score: -1..+1
                article_score = (pos_score - neg_score) / total
                article_scores.append(article_score)
            # Articles with no keyword matches are skipped (not neutral)
        
        if not article_scores:
            # No keyword matches at all â€” return neutral with article count
            return {
                'sentiment': 'NEUTRAL',
                'sentiment_class': 'neutral',
                'score': 0,
                'bullish_percent': 50,
                'bearish_percent': 50,
                'buzz_articles': len(news),
                'buzz_score': min(len(news) / 10, 1.0)
            }
        
        # Average article scores â€” this naturally produces -1..+1 range
        score = sum(article_scores) / len(article_scores)
        
        # Convert to percentages for display
        bullish_percent = round(((score + 1) / 2) * 100, 2)  # 0..100
        bearish_percent = round(100 - bullish_percent, 2)
        
        # Determine sentiment label (same thresholds as Finnhub path)
        if score > 0.6:
            sentiment_label = 'STRONG BUY'
            sentiment_class = 'positive'
        elif score > 0.2:
            sentiment_label = 'BUY'
            sentiment_class = 'positive'
        elif score > -0.2:
            sentiment_label = 'HOLD'
            sentiment_class = 'neutral'
        elif score > -0.6:
            sentiment_label = 'SELL'
            sentiment_class = 'negative'
        else:
            sentiment_label = 'STRONG SELL'
            sentiment_class = 'negative'
        
        print(f"ðŸ“Š News Sentiment ({len(article_scores)}/{len(news)} articles scored): {sentiment_label} (Score: {score:.2f})")
        
        result = {
            'sentiment': sentiment_label,
            'sentiment_class': sentiment_class,
            'score': round(score, 2),
            'bullish_percent': bullish_percent,
            'bearish_percent': bearish_percent,
            'buzz_articles': len(news),
            'buzz_score': min(len(news) / 10, 1.0)
        }
        
        # Cache the calculated result too
        cache_params = {'ticker': ticker}
        cache.set('sentiment_analysis', cache_params, result)
        
        return result
        
    except Exception as e:
        print(f"Error calculating sentiment: {e}")
        return {
            'sentiment': 'NEUTRAL',
            'sentiment_class': 'neutral',
            'score': 0,
            'bullish_percent': 50,
            'bearish_percent': 50,
            'buzz_articles': 0,
            'buzz_score': 0
        }


def get_quote_data(ticker):
    """
    Get real-time quote data from Finnhub
    
    Args:
        ticker (str): Stock symbol
    
    Returns:
        dict: Real-time quote with current price, change, percent change, high, low, open, previous close
    """
    try:
        print(f"Fetching real-time quote for {ticker} from Finnhub...")
        
        url = f'{FINNHUB_BASE_URL}/quote'
        params = {
            'symbol': ticker,
            'token': _get_finnhub_key()
        }
        
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        quote = response.json()
        
        current = quote.get('c', 0)  # Current price
        previous = quote.get('pc', current)  # Previous close
        change = current - previous
        change_percent = (change / previous * 100) if previous != 0 else 0
        
        result = {
            'current': float(current),
            'high': float(quote.get('h', current)),
            'low': float(quote.get('l', current)),
            'open': float(quote.get('o', current)),
            'previous_close': float(previous),
            'change': float(change),
            'change_percent': float(change_percent),
            'timestamp': quote.get('t', int(datetime.now().timestamp()))
        }
        
        print(f"Quote: ${current:.2f} ({change:+.2f}, {change_percent:+.2f}%)")
        return result
        
    except Exception as e:
        print(f"Error fetching quote from Finnhub: {e}")
        return None


def get_company_profile(ticker):
    """
    Get company profile from Finnhub
    
    Args:
        ticker (str): Stock symbol
    
    Returns:
        dict: Company profile with name, market cap, industry, etc.
    """
    try:
        print(f"Fetching company profile for {ticker} from Finnhub...")
        
        url = f'{FINNHUB_BASE_URL}/stock/profile2'
        params = {
            'symbol': ticker,
            'token': _get_finnhub_key()
        }
        
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        profile = response.json()
        
        # Finnhub returns marketCapitalization in MILLIONS of USD
        raw_mcap = profile.get('marketCapitalization', 0)
        market_cap = float(raw_mcap) * 1_000_000 if raw_mcap else 0
        print(f"Finnhub marketCapitalization raw={raw_mcap}, converted={market_cap}")
        
        # Fallback 1: If market cap is missing, try Twelve Data Statistics
        if market_cap <= 0:
            try:
                print(f"Market cap missing from Finnhub profile. Trying Twelve Data for {ticker}...")
                td_url = f"{BASE_URL}/statistics"
                td_params = {
                    'symbol': ticker,
                    'apikey': twelve_data_rotator.key
                }
                td_resp = requests.get(td_url, params=td_params, timeout=15)
                if td_resp.status_code == 200:
                    td_data = td_resp.json()
                    if 'statistics' in td_data:
                        val = td_data['statistics'].get('stock_price_statistics', {}).get('market_capitalization')
                        if val:
                            td_mcap = float(val)
                            # Twelve Data returns market cap in raw USD (e.g. 3.5T for AAPL)
                            # Sanity check: if value < 1000, it's likely in billions â€” multiply
                            if td_mcap < 1000:
                                td_mcap = td_mcap * 1_000_000_000  # Convert billions to raw
                                print(f"  Twelve Data value looks like billions, normalized: {td_mcap}")
                            market_cap = td_mcap
                            print(f"âœ… Found Market Cap in Twelve Data: {market_cap}")
            except Exception as e:
                print(f"Twelve Data fallback failed: {e}")

        # Fallback 2: Try Finnhub Metrics endpoint
        if market_cap <= 0:
            try:
                print(f"Market cap still missing. Trying Finnhub metrics for {ticker}...")
                m_url = f'{FINNHUB_BASE_URL}/stock/metric'
                m_params = {'symbol': ticker, 'metric': 'all', 'token': _get_finnhub_key()}
                m_resp = requests.get(m_url, params=m_params, timeout=15)
                if m_resp.status_code == 200:
                    m_data = m_resp.json().get('metric', {})
                    val = m_data.get('marketCapitalization')
                    if val:
                        market_cap = float(val) * 1_000_000
                        print(f"âœ… Found Market Cap in Finnhub Metrics: {market_cap}")
            except Exception as e:
                print(f"Finnhub metrics fallback failed: {e}")

        result = {
            'name': profile.get('name', ticker),
            'ticker': profile.get('ticker', ticker),
            'market_cap': market_cap,
            'industry': profile.get('finnhubIndustry', 'N/A'),
            'logo': profile.get('logo', ''),
            'country': profile.get('country', 'US'),
            'currency': profile.get('currency', 'USD'),
            'exchange': profile.get('exchange', 'NASDAQ')
        }
        
        print(f"Company: {result['name']} ({result['industry']})")
        return result
        
    except Exception as e:
        print(f"Error fetching company profile from Finnhub: {e}")
        return {
            'name': ticker,
            'ticker': ticker,
            'market_cap': 0,
            'industry': 'N/A',
            'logo': '',
            'country': 'US',
            'currency': 'USD',
            'exchange': 'NASDAQ'
        }


def get_company_metrics(ticker):
    """Fetch fundamental metrics (including P/E ratio) from Finnhub."""
    try:
        print(f"Fetching fundamental metrics for {ticker} from Finnhub...")

        url = f'{FINNHUB_BASE_URL}/stock/metric'
        params = {
            'symbol': ticker,
            'metric': 'all',
            'token': _get_finnhub_key()
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json() or {}
        metrics = payload.get('metric', {}) or {}

        pe_candidates = [
            metrics.get('peBasicExclExtraTTM'),
            metrics.get('peBasicInclExtraTTM'),
            metrics.get('peNormalizedAnnual'),
            metrics.get('trailingPE'),
            metrics.get('peTTM')
        ]

        pe_ratio = next(
            (float(val) for val in pe_candidates if isinstance(val, (int, float)) and not pd.isna(val)),
            None
        )

        eps_candidates = [
            metrics.get('epsBasicExclExtraTTM'),
            metrics.get('epsBasicInclExtraTTM'),
            metrics.get('epsNormalizedAnnual'),
            metrics.get('epsDilutedTTM')
        ]

        eps = next(
            (float(val) for val in eps_candidates if isinstance(val, (int, float)) and not pd.isna(val)),
            None
        )

        print("Metrics retrieved" if pe_ratio is not None else "PE ratio unavailable from metrics response")

        return {
            'pe_ratio': pe_ratio,
            'eps': eps
        }

    except Exception as exc:
        print(f"Error fetching metrics from Finnhub: {exc}")
        return {
            'pe_ratio': None,
            'eps': None
        }

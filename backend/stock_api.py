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
import re
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



import threading

class KeyRotator:
    """Manages multiple API keys with automatic rotation on rate-limit errors."""
    def __init__(self, name: str, keys: list):
        self.name = name
        self.keys = [k.strip() for k in keys if k.strip()]
        if not self.keys:
            logger.warning("%s: no API keys configured; provider calls will fail until keys are set", name)
        self._idx = 0
        self._lock = threading.Lock()
        logger.info(f"ðŸ”‘ {name}: loaded {len(self.keys)} API key(s)")

    @property
    def key(self) -> str:
        """Return the currently active key."""
        with self._lock:
            if not self.keys:
                raise RuntimeError(f"No API keys configured for {self.name}. Update your .env file.")
            return self.keys[self._idx % len(self.keys)]

    def rotate(self, failed_key: str = None) -> str:
        """Move to next key and return it. Only rotates if the caller's key is still active."""
        with self._lock:
            if not self.keys:
                raise RuntimeError(f"No API keys configured for {self.name}. Update your .env file.")
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
TWELVE_DATA_API_KEY = twelve_data_rotator.key if len(twelve_data_rotator) else ''
BASE_URL = 'https://api.twelvedata.com'
FINNHUB_API_KEY = finnhub_rotator.key if len(finnhub_rotator) else ''
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

TWELVE_DATA_SUFFIX_EXCHANGES = {
    '.NS': 'NSE',
    '.BO': 'BSE',
}


#  Dynamic key getters (always return the currently active key) 
def _get_twelve_key():
    return twelve_data_rotator.key if len(twelve_data_rotator) else ''

def _get_finnhub_key():
    return finnhub_rotator.key if len(finnhub_rotator) else ''

def _get_alpha_key():
    return alpha_vantage_rotator.key if alpha_vantage_rotator else ALPHA_VANTAGE_API_KEY


def _request_with_rotation(rotator, url, params, apikey_param='apikey', timeout=30):
    """
    Make a GET request with automatic key rotation on rate-limit errors.
    Tries each key in the rotator before giving up.
    Returns the requests.Response object on success.
    """
    attempts = len(rotator)
    if attempts == 0:
        raise requests.exceptions.RequestException(
            f"{rotator.name}: no API keys configured. Update your environment secrets."
        )
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
    base = base.replace(' ', '')
    for suffix in ('.NS', '.BO', '.BSE', '.NSE'):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


COMPANY_ALIAS_OVERRIDES = {
    'AAPL': ['apple'],
    'MSFT': ['microsoft'],
    'GOOGL': ['alphabet', 'google'],
    'GOOG': ['alphabet', 'google'],
    'AMZN': ['amazon'],
    'NVDA': ['nvidia'],
    'TSLA': ['tesla'],
    'META': ['meta platforms', 'facebook', 'instagram'],
    'ORCL': ['oracle'],
    'RELIANCE': ['reliance industries', 'reliance'],
    'TCS': ['tata consultancy', 'tcs'],
    'INFY': ['infosys'],
    'HDFCBANK': ['hdfc bank'],
    'ICICIBANK': ['icici bank'],
}


def _company_terms_for_ticker(ticker: str, company_name: str | None = None) -> list[str]:
    base = _normalize_symbol(ticker)
    terms = {base.lower()} if base else set()
    terms.update(COMPANY_ALIAS_OVERRIDES.get(base, []))
    if company_name:
        cleaned = re.sub(r'\b(inc|corp|corporation|ltd|limited|plc|class a|common stock|ordinary shares)\b', '', company_name.lower())
        cleaned = re.sub(r'[^a-z0-9\s&.-]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned) > 2:
            terms.add(cleaned)
            first_word = cleaned.split()[0]
            if len(first_word) >= 4:
                terms.add(first_word)
    return sorted(term for term in terms if len(term) >= 2)


def _article_relevance_score(article: dict, ticker: str, company_name: str | None = None) -> float:
    """Score whether a news article is really about the searched stock."""
    terms = _company_terms_for_ticker(ticker, company_name)
    if not terms:
        return 0.0

    headline = str(article.get('headline', '') or '').lower()
    summary = str(article.get('summary', '') or '').lower()
    related = str(article.get('related', '') or article.get('symbol', '') or '').upper()
    base = _normalize_symbol(ticker)

    score = 0.0
    if base and base in related.replace(',', ' ').split():
        score += 3.0
    for term in terms:
        term_pattern = rf'(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])'
        if re.search(term_pattern, headline):
            score += 3.0
        if re.search(term_pattern, summary[:280]):
            score += 1.0

    # If another mega-cap dominates the headline and the searched company is
    # only buried in the summary, keep it below first-page news.
    competing_terms = [
        term for symbol, aliases in COMPANY_ALIAS_OVERRIDES.items()
        if symbol != base
        for term in aliases[:2]
    ]
    if score < 3.0 and any(re.search(rf'(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])', headline) for term in competing_terms):
        score -= 1.0

    return max(0.0, score)


def _twelvedata_symbol_params(symbol: str, exchange: str | None = None) -> tuple[str, str | None]:
    """Translate Yahoo-style suffixes into Twelve Data symbol + exchange params."""
    cleaned = (symbol or '').strip().upper()
    exchange_hint = (exchange or '').strip().upper() or None

    for suffix, td_exchange in TWELVE_DATA_SUFFIX_EXCHANGES.items():
        if cleaned.endswith(suffix):
            return cleaned[:-len(suffix)], exchange_hint or td_exchange

    return cleaned, exchange_hint


def _alphavantage_symbol_candidates(
    symbol: str,
    exchange: str | None = None,
    country: str | None = None
) -> list[str]:
    """Generate Alpha Vantage symbols, including Indian BSE fallbacks for Yahoo-style tickers."""
    cleaned = (symbol or '').strip().upper().replace(' ', '')
    base = _normalize_symbol(cleaned)
    exchange_hint = (exchange or '').strip().upper()
    country_hint = (country or '').strip().lower()
    candidates: list[str] = []
    seen: set[str] = set()

    def push(candidate: str):
        normalized = candidate.strip().upper()
        if normalized and normalized not in seen:
            candidates.append(normalized)
            seen.add(normalized)

    is_india_hint = (
        country_hint == 'india'
        or exchange_hint in {'NSE', 'BSE', 'BOMBAY STOCK EXCHANGE', 'NATIONAL STOCK EXCHANGE OF INDIA'}
        or cleaned.endswith(('.NS', '.BO', '.NSE', '.BSE'))
    )

    if cleaned.endswith(('.NS', '.NSE', '.BO')):
        push(f'{base}.BSE')
        push(base)
    elif cleaned.endswith('.BSE'):
        push(cleaned)
        push(base)
    elif is_india_hint:
        push(f'{base}.BSE')
        push(base)
    else:
        push(cleaned)
        push(base)

    return candidates


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

        search_query, exchange_hint = _twelvedata_symbol_params(query)
        url = f'{BASE_URL}/symbol_search'
        params = {
            'symbol': search_query,
            'outputsize': max(1, min(limit, 30)),
            'apikey': _get_twelve_key()
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json() or {}
        data = payload.get('data') or []

        results = []
        if exchange_hint:
            data = sorted(
                data,
                key=lambda entry: 0 if str(entry.get('exchange', '')).upper() == exchange_hint else 1
            )

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

    td_symbol, td_exchange = _twelvedata_symbol_params(ticker, exchange)

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
            'symbol': td_symbol,
            'interval': interval,
            'outputsize': min(days, 5000),  # Max 5000 data points
            'format': 'JSON'
        }
        if td_exchange:
            params['exchange'] = td_exchange

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
            info['twelvedata_symbol'] = td_symbol
            info['exchange'] = td_exchange or exchange

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
            av_candidates = _alphavantage_symbol_candidates(ticker, exchange=exchange, country=country)
            outputsize = 'full' if days and days > 100 else 'compact'

            for av_symbol in av_candidates:
                av_params = {
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': av_symbol,
                    'outputsize': outputsize
                }

                print(f"  Trying Alpha Vantage with {av_symbol} (with key rotation)...")
                response = _request_with_rotation(alpha_vantage_rotator, av_url, av_params, apikey_param='apikey', timeout=15)
                av_data = response.json()

                if 'Time Series (Daily)' not in av_data:
                    error_msg = av_data.get('Note') or av_data.get('Information') or av_data.get('Error Message')
                    if error_msg:
                        print(f"  Alpha Vantage {av_symbol}: {error_msg}")
                    continue

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

                if not df_data:
                    continue

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
                info['alphavantage_symbol'] = av_symbol
                if av_symbol.endswith('.BSE'):
                    info['exchange'] = 'BSE'
                needs_fallback = False

                logger.info(f"âœ… Alpha Vantage: Successfully fetched {len(data_frame)} data points for {ticker} via {av_symbol}")
                print(f"âœ… Alpha Vantage: Successfully fetched {len(data_frame)} data points for {ticker} via {av_symbol}")

                # Cache the successful result
                cache_entry = {
                    'dataframe': data_frame.reset_index().to_dict('records'),
                    'info': info
                }
                cache.set('stock_history', cache_params, cache_entry)
                print(f"ðŸ’¾ Cached Alpha Vantage data for {ticker}")
                break
                
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
        
        # Filter and format news articles. Company-news providers can still
        # return broad sector stories, so do not show weakly related articles
        # as if they are ticker-specific news.
        ranked_articles = []
        ticker_lower = ticker.lower()
        
        for article in news_data[:60]:
            relevance = _article_relevance_score(article, ticker, company_name)
            if relevance < 1.0:
                continue

            timestamp = article.get('datetime', 0)
            try:
                if timestamp > 0:
                    date_obj = datetime.fromtimestamp(timestamp)
                    formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
                else:
                    formatted_date = 'Unknown date'
            except Exception as e:
                print(f"Error formatting date: {e}")
                formatted_date = 'Unknown date'

            ranked_articles.append((relevance, float(timestamp or 0), {
                'headline': article.get('headline', 'No headline'),
                'summary': article.get('summary', 'No summary available'),
                'source': 'Finnhub',
                'url': article.get('url', '#'),
                'image': company_logo,
                'datetime': formatted_date,
                'timestamp': timestamp,
                'related': ticker.upper(),
                'company_name': company_name or ticker,
                'relevance_score': round(float(relevance), 2),
            }))

            if len(ranked_articles) >= 30:
                break

        ranked_articles.sort(key=lambda item: (item[0], item[1]), reverse=True)
        news_articles = [item[2] for item in ranked_articles]
        
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


def _clamp_sentiment_score(value):
    try:
        return max(-1.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _normalize_sentiment_fraction(value):
    try:
        numeric = float(value or 0.0)
    except Exception:
        return 0.0
    if numeric > 1.0 and numeric <= 100.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))


def _sentiment_label(score):
    score = _clamp_sentiment_score(score)
    if score > 0.6:
        return 'STRONG BUY', 'positive'
    if score > 0.2:
        return 'BUY', 'positive'
    if score >= -0.2:
        return 'HOLD', 'neutral'
    if score >= -0.6:
        return 'SELL', 'negative'
    return 'STRONG SELL', 'negative'


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
        bullish_frac = _normalize_sentiment_fraction(sentiment.get('bullishPercent', 0))
        bearish_frac = _normalize_sentiment_fraction(sentiment.get('bearishPercent', 0))
        
        # If Finnhub returns 0 for both, fall back to news-based analysis
        if bullish_frac == 0 and bearish_frac == 0:
            print(f"Finnhub returned zero sentiment for {ticker}, falling back to news...")
            return calculate_sentiment_from_news(ticker)
        
        total_frac = bullish_frac + bearish_frac
        if total_frac > 1.05:
            bullish_frac = bullish_frac / total_frac
            bearish_frac = bearish_frac / total_frac

        overall_score = _clamp_sentiment_score(bullish_frac - bearish_frac)
        article_count = float(buzz.get('articlesInLastWeek', 0) or 0)
        buzz_score = float(buzz.get('buzz', 0) or 0)
        sentiment_confidence = max(
            0.10,
            min(1.0, max(min(article_count / 12.0, 1.0), min(buzz_score, 1.0)))
        )

        if sentiment_confidence < 0.35:
            overall_score *= 0.70

        if sentiment_confidence < 0.45 or article_count < 3:
            news_result = calculate_sentiment_from_news(ticker)
            news_conf = float(news_result.get('sentiment_confidence', 0.0) or 0.0)
            if news_conf > sentiment_confidence:
                return news_result

        sentiment_label, sentiment_class = _sentiment_label(overall_score)
        
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
            'buzz_score': buzz.get('buzz', 0),
            'sentiment_confidence': round(float(sentiment_confidence), 3),
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
                'buzz_score': 0,
                'sentiment_confidence': 0
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

        positive_phrases = {
            'raises price target': 1.3,
            'price target raised': 1.3,
            'maintains buy': 1.1,
            'beats estimates': 1.2,
            'record revenue': 1.2,
            'strong guidance': 1.2,
            'upgrade to buy': 1.4,
            'initiates buy': 1.2,
            'overweight rating': 1.1,
            'positive guidance': 1.2,
            'margin expansion': 1.0,
            'free cash flow growth': 1.1,
            'earnings surprise': 1.0,
        }
        negative_phrases = {
            'lowers price target': 1.3,
            'price target lowered': 1.3,
            'maintains sell': 1.1,
            'misses estimates': 1.2,
            'weak guidance': 1.2,
            'downgrade to sell': 1.4,
            'target decrease': 1.1,
            'initiates sell': 1.2,
            'underweight rating': 1.1,
            'negative guidance': 1.2,
            'margin pressure': 1.0,
            'cash burn': 1.1,
            'earnings miss': 1.1,
        }
        negation_terms = ('not', 'no', 'never', 'without', 'less')
        source_weights = {
            'reuters': 1.20,
            'bloomberg': 1.20,
            'wall street journal': 1.15,
            'cnbc': 1.05,
            'marketwatch': 1.00,
            'finnhub': 1.00,
            'seeking alpha': 0.90,
            'benzinga': 0.85,
        }
        
        # Per-headline scoring: each article gets a score from -1 to +1.
        # Weight = source_quality * exponential_time_decay * headline_importance.
        weighted_scores = []
        weights = []
        now_ts = datetime.now().timestamp()
        
        scored_articles = []

        def count_terms(text_value, terms):
            tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text_value)
            token_set = set(tokens)
            return sum(1 for term in terms if term in token_set)

        def negation_adjustment(text_value):
            # Damp keyword sentiment near "not/no/less" to avoid false positives
            # like "not strong" or "no growth".
            adjustment = 1.0
            words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text_value)
            for i, word in enumerate(words):
                if word in negation_terms:
                    window = words[i + 1:i + 4]
                    if any(w in positive_keywords or w in negative_keywords for w in window):
                        adjustment *= 0.65
            return max(0.35, adjustment)

        for article in news:
            headline_lower = article.get('headline', '').lower()
            summary_lower = article.get('summary', '').lower()
            source_lower = str(article.get('source', '')).lower()
            text = f"{headline_lower} {summary_lower}"
            
            # Count matches - headlines weighted more than summaries.
            headline_pos = count_terms(headline_lower, positive_keywords)
            headline_neg = count_terms(headline_lower, negative_keywords)
            summary_pos = count_terms(summary_lower, positive_keywords)
            summary_neg = count_terms(summary_lower, negative_keywords)
            
            pos_score = (headline_pos * 2) + summary_pos
            neg_score = (headline_neg * 2) + summary_neg
            pos_score += sum(weight for phrase, weight in positive_phrases.items() if phrase in text)
            neg_score += sum(weight for phrase, weight in negative_phrases.items() if phrase in text)
            neg_adj = negation_adjustment(text)
            pos_score *= neg_adj
            neg_score *= neg_adj

            # Financial nuance: "maintains neutral/market perform" is not strongly bearish
            # even when a target is trimmed. Damp both sides toward neutrality.
            if any(phrase in text for phrase in ['maintains neutral', 'market perform', 'equal weight']):
                pos_score *= 0.65
                neg_score *= 0.65

            total = pos_score + neg_score
            
            if total > 0:
                article_score = (pos_score - neg_score) / total
                article_score = max(-1.0, min(1.0, article_score))

                timestamp = article.get('timestamp')
                if not timestamp:
                    try:
                        timestamp = datetime.strptime(str(article.get('datetime')), '%Y-%m-%d %H:%M').timestamp()
                    except Exception:
                        timestamp = now_ts
                age_days = max(0.0, (now_ts - float(timestamp)) / 86400.0)
                time_decay = float(0.5 ** (age_days / 3.0))  # 3-day half-life
                source_weight = next(
                    (weight for name, weight in source_weights.items() if name in source_lower),
                    0.85
                )
                weight = max(0.10, min(1.50, source_weight * time_decay))
                weighted_scores.append(article_score * weight)
                weights.append(weight)
                scored_articles.append({
                    'headline': article.get('headline', ''),
                    'source': article.get('source', ''),
                    'score': round(float(article_score), 3),
                    'weight': round(float(weight), 3),
                })
            # Articles with no keyword matches are skipped (not neutral)
        
        if not weighted_scores or not weights:
            # No keyword matches at all â€” return neutral with article count
            return {
                'sentiment': 'NEUTRAL',
                'sentiment_class': 'neutral',
                'score': 0,
                'bullish_percent': 50,
                'bearish_percent': 50,
                'buzz_articles': len(news),
                'buzz_score': 0,
                'sentiment_confidence': 0
            }
        
        # Weighted average naturally stays in -1..+1 range.
        score = sum(weighted_scores) / max(sum(weights), 1e-9)
        coverage = len(weighted_scores) / max(len(news), 1)
        sentiment_confidence = max(0.10, min(1.0, coverage * min(sum(weights) / 4.0, 1.0)))

        # Low-coverage headline sentiment is useful context, not a trade driver.
        if sentiment_confidence < 0.35:
            score *= 0.65

        score = max(-1.0, min(1.0, score))
        
        # Convert to percentages for display
        bullish_percent = round(((score + 1) / 2) * 100, 2)  # 0..100
        bearish_percent = round(100 - bullish_percent, 2)
        
        # Determine sentiment label (same thresholds as Finnhub path)
        sentiment_label, sentiment_class = _sentiment_label(score)
        
        print(f"ðŸ“Š News Sentiment ({len(weighted_scores)}/{len(news)} articles scored): {sentiment_label} (Score: {score:.2f})")
        
        result = {
            'sentiment': sentiment_label,
            'sentiment_class': sentiment_class,
            'score': round(score, 2),
            'bullish_percent': bullish_percent,
            'bearish_percent': bearish_percent,
            'buzz_articles': len(news),
            'buzz_score': round(float(sentiment_confidence), 3),
            'sentiment_confidence': round(float(sentiment_confidence), 3),
            'scored_articles': len(weighted_scores),
            'top_sentiment_drivers': scored_articles[:5],
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

from flask import Flask, request, render_template, jsonify, send_from_directory, g
from flask_cors import CORS
import os
from datetime import datetime, timedelta
import warnings
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
from werkzeug.exceptions import HTTPException

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
        return jsonify({'error': 'Chatbot service is starting up or unavailable', 'detail': 'The chatbot server at port 8001 is not responding. It may still be initializing.'}), 503
    except Exception as e:
        return jsonify({'error': 'Chatbot service unavailable', 'detail': str(e)}), 503

# Heavy imports (pandas, numpy, ta, stock_api) will be lazy-loaded when needed
# for standard production configuration

# Model will be loaded lazily on first use
# Check multiple possible model paths (pipeline saves as ticker_lstm_model.h5,
# component ModelTrainer saves as stock_lstm_model.h5)
MODEL_PATHS = [
    'artifacts/stock_lstm_model.h5',
    'artifacts/AAPL_lstm_model.h5',
]
SCALER_PATH = 'artifacts/scaler.pkl'
# Per-ticker model cache: { 'AAPL': model, 'TSLA': model, ... }
_model_cache = {}
_scaler_cache = {}
# Per-ticker model metadata cache: { 'AAPL': {'version': '1', 'run_id': 'xyz', 'source': 'mlflow'} }
_model_metadata = {}

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


def _parse_env_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(',') if item.strip()]


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

    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
    if not service_account_path or not os.path.exists(service_account_path):
        return False

    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    return True


print("=" * 60)
print("Stock Predictor App - Using Twelve Data API")
print("=" * 60)

# Start automatic model training scheduler
try:
    from scheduler import scheduler
    scheduler.start()
    print("✅ Automatic model training started (runs every hour)")
except Exception as e:
    print(f"⚠️ Could not start training scheduler: {e}")

# Configure Flask to serve React build (optimized)
os.makedirs('cache', exist_ok=True)
application = Flask(__name__, static_folder='build', static_url_path='')
app = application

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
                app.logger.info('Invalid or expired token: %s', exc)
                return jsonify({'success': False, 'error': 'Invalid or expired token'}), 401

            if admin_only and not _user_has_admin_privileges(decoded):
                return jsonify({'success': False, 'error': 'Insufficient privileges'}), 403

            g.firebase_user = decoded
            return func(*args, **kwargs)

        return wrapper

    return decorator


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
    
    # NEW: Register interest in this ticker immediately (adds to stocks.json)
    try:
        from mlops.config import MLOpsConfig
        MLOpsConfig.add_stock(ticker)
    except Exception as e:
        print(f"⚠️ Failed to register ticker {ticker} in stocks.json: {e}")
    
    # Return cached model if already loaded for this ticker
    if ticker in _model_cache:
        return _model_cache[ticker]
    
    loaded_model = None
    
    try:
        # FIRST: Try loading from MLOps registry (best versioned model for this ticker)
        try:
            from mlops.registry import ModelRegistry
            registry = ModelRegistry()
            best = registry.get_best_model(ticker)
            if best and os.path.exists(best.get('model_path', '')):
                print(f"Loading model from registry for {ticker}: {best['version_id']}")
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
                # NEW: If model not found, trigger background training for this new ticker
                print(f"🚀 No model for {ticker}. Triggering background training...")
                import threading
                from mlops.training_pipeline import MLOpsTrainingPipeline
                
                def background_train():
                    try:
                        pipeline = MLOpsTrainingPipeline()
                        # Use lower epochs for background jobs to avoid excessive API wait
                        pipeline.train_model(ticker=ticker, epochs=15)
                        print(f"✅ Background training and registration completed for {ticker}")
                    except Exception as e:
                        print(f"❌ Background training failed for {ticker}: {e}")
                
                threading.Thread(target=background_train, daemon=True).start()
                
        except Exception as e:
            print(f"Registry lookup failed for {ticker}: {e}")
        
        # SECOND: Fall back to generic artifact file paths
        from tensorflow.keras.models import load_model as keras_load
        for model_path in MODEL_PATHS:
            if os.path.exists(model_path):
                print(f"Loading fallback model from {model_path} for {ticker}...")
                loaded_model = keras_load(model_path)
                print(f"✅ Fallback model loaded from {model_path}")
                _model_cache[ticker] = loaded_model
                break
        else:
            print(f"⚠️ No model found for {ticker}")
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
            
        # 3. Try generic fallback scaler artifact
        if os.path.exists(SCALER_PATH):
            import joblib
            scaler = joblib.load(SCALER_PATH)
            _scaler_cache[ticker] = scaler
            return scaler
    except Exception as e:
        print(f"Could not load scaler for {ticker}: {e}")
    return None

# Health check endpoint (minimal dependencies)
@app.route('/health')
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
    
    days = request.args.get('days', default=7, type=int)
    days = max(1, min(days, 30))  # Limit between 1-30 days
    
    try:
        requested_ticker = ticker.upper()
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
                days=180,
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
            'sma50': safe_float(latest_row.get('SMA_50'))
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
        pe_ratio = company_metrics.get('pe_ratio') if isinstance(company_metrics, dict) else None
        volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0

        if quote_data:
            quote_current = quote_data.get('current')
            quote_previous = quote_data.get('previous_close')
            if isinstance(quote_current, (int, float)) and quote_current > 0:
                current_price = float(quote_current)
            if isinstance(quote_previous, (int, float)) and quote_previous > 0:
                previous_close = float(quote_previous)

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
        
        response = {
            'success': True,
            'ticker': resolved_ticker,
            'requested_ticker': requested_ticker,
            'resolved_exchange': resolved_exchange,
            'resolved_country': resolved_country,
            'used_fallback_source': fallback_used,
            'company_name': company_name,
            'current_price': safe_float(current_price),
            'predicted_price': safe_float(tomorrow_prediction),
            'profit_loss': safe_float(profit_loss),
            'profit_loss_percent': safe_float(profit_loss_percent),
            'is_profit': bool(profit_loss > 0),
            'ai_signal': ai_signal,
            'day_change': safe_float(day_change),
            'day_change_percent': safe_float(day_change_percent),
            'volume': volume,
            'predicted_volume': predicted_volumes,
            'market_cap': market_cap,
            'pe_ratio': safe_float(pe_ratio) if isinstance(pe_ratio, (int, float)) and not pd.isna(pe_ratio) else None,
            'days_predicted': days,
            'indicators': indicators,
            'sentiment': sentiment_data,
            'model_info': model_info_data,
            'historical_data': historical_data,
            'future_predictions': future_predictions,
            'technical_chart': {
                'candles': candlestick_data,
                'volumes': volume_data,
                'moving_averages': ma_data
            },
            'indicator_trends': indicator_trends,
            'performance': performance,
            'performance_chart': performance_chart,
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

def predict_multi_day_lstm(hist, current_price, days, ticker):
    """Predict multiple days ahead using LSTM model for ANY ticker"""
    import numpy as np
    import ta as ta_lib
    
    predictions = []
    
    try:
        current_model = load_lstm_model(ticker)
        
        if current_model is None:
            for day in range(days):
                pred = predict_with_technical_analysis(hist, current_price)
                predictions.append(pred)
                current_price = pred
            return predictions
        
        # Determine the model's expected input feature count
        expected_features = current_model.input_shape[-1]  # last dim of (batch, seq, features)
        
        # Try to load the saved scaler for this specific ticker
        saved_scaler = load_saved_scaler(ticker)
        
        if saved_scaler is not None:
            # --- Robust multi-feature path (matches training pipeline) ---
            # Determine how many features the model and scaler expect
            n_features_model = current_model.input_shape[-1]
            try:
                n_features_scaler = saved_scaler.n_features_in_
            except AttributeError:
                n_features_scaler = n_features_model # Fallback for older sklearn
            
            # Load feature list for THIS ticker from training
            import json as json_mod
            feature_path = f'artifacts/{ticker}_features.json'
            # Fallback to general library in registry
            training_features = None
            try:
                from mlops.registry import ModelRegistry
                registry = ModelRegistry()
                training_features = registry.get_features_from_library(ticker)
            except Exception:
                pass
                
            if not training_features and os.path.exists(feature_path):
                try:
                    with open(feature_path, 'r') as f:
                        training_features = json_mod.load(f)
                except Exception:
                    training_features = None
            
            # Build feature DataFrame matching training (DataIngestion)
            import pandas as pd
            feature_df = pd.DataFrame()
            close = hist['Close'].squeeze()
            high = hist['High'] if 'High' in hist.columns else close
            low = hist['Low'] if 'Low' in hist.columns else close
            volume = hist['Volume'] if 'Volume' in hist.columns else 0
            
            # --- STATIONARY FEATURES (Matching DataIngestion) ---
            feature_df['returns'] = close.pct_change()
            feature_df['log_returns'] = np.log(close / close.shift(1))
            
            # Ratios
            feature_df['sma_5_ratio'] = close.rolling(window=5).mean() / close
            feature_df['sma_10_ratio'] = close.rolling(window=10).mean() / close
            feature_df['sma_20_ratio'] = close.rolling(window=20).mean() / close
            feature_df['sma_50_ratio'] = close.rolling(window=50).mean() / close
            feature_df['ema_12_ratio'] = close.ewm(span=12, adjust=False).mean() / close
            feature_df['ema_26_ratio'] = close.ewm(span=26, adjust=False).mean() / close
            
            # MACD Ratios
            # Note: We calculate MACD manually or via ta-lib, but always normalize by close
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            
            feature_df['macd_ratio'] = macd / close
            feature_df['macd_signal_ratio'] = macd_signal / close
            feature_df['macd_hist_ratio'] = (macd - macd_signal) / close
            
            # Momentum / Volatility
            feature_df['rsi'] = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
            bb = ta_lib.volatility.BollingerBands(close, window=20)
            feature_df['bb_middle_ratio'] = bb.bollinger_mavg() / close
            feature_df['bb_upper_ratio'] = bb.bollinger_hband() / close
            feature_df['bb_lower_ratio'] = bb.bollinger_lband() / close
            feature_df['bb_width_ratio'] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
            feature_df['atr_ratio'] = ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range() / close
            feature_df['volatility'] = feature_df['returns'].rolling(window=20).std()
            
            # Volume
            feature_df['volume_sma_20'] = volume.rolling(window=20).mean()
            feature_df['volume_ratio'] = volume / feature_df['volume_sma_20'].replace(0, np.nan)
            
            direction = np.sign(close.diff().fillna(0))
            volume_avg = volume.rolling(window=20).mean().replace(0, np.nan)
            feature_df['obv'] = (volume * direction) / volume_avg.fillna(volume.mean() + 1e-9)
            
            # Lags
            for lag in [1, 2, 3, 5]:
                feature_df[f'returns_lag_{lag}'] = feature_df['returns'].shift(lag)
            
            # Drop intermediate calculation columns if they aren't in canonical list
            if 'volume_sma_20' in feature_df.columns:
                feature_df = feature_df.drop(columns=['volume_sma_20'])
                
            feature_df = feature_df.dropna()
            
            # CRITICAL: Feature Alignment
            if training_features:
                feature_df = feature_df[[c for c in training_features if c in feature_df.columns]]
            
            if feature_df.shape[1] != n_features_model:
                print(f"⚠️ Feature mismatch for {ticker}: model expects {n_features_model}, got {feature_df.shape[1]}. Falling back...")
                raise ValueError("Feature mismatch")

            if len(feature_df) < 10:
                raise ValueError("Insufficient data for multi-feature prediction")
            
            # Scale using synchronized scaler
            scaled_data = saved_scaler.transform(feature_df.values)
            
            model_seq_len = current_model.input_shape[1]
            sequence_length = min(model_seq_len, len(scaled_data))
            
            if len(scaled_data) < model_seq_len:
                pad_count = model_seq_len - len(scaled_data)
                padding = np.tile(scaled_data[0:1], (pad_count, 1))
                scaled_data = np.vstack([padding, scaled_data])
                sequence_length = model_seq_len
            
            last_sequence = list(scaled_data[-sequence_length:])
            n_features = scaled_data.shape[1]
            
            predicting_price = current_price
            
            for day in range(days):
                input_seq = np.array(last_sequence[-sequence_length:]).reshape(1, sequence_length, n_features)
                predicted_scaled = current_model.predict(input_seq, verbose=0)[0][0]
                
                # --- Inference Safety Gate (For Returns) ---
                if predicted_scaled > 5.0 or predicted_scaled < -5.0:
                    print(f"⚠️ LSTM Sanity Check Failed for {ticker} (scaled return {predicted_scaled:.2f}). Falling back.")
                    raise ValueError(f"Extreme prediction: {predicted_scaled:.2f}")

                # Update sequence for next day (recursive prediction)
                next_row = list(last_sequence[-1])
                next_row[0] = predicted_scaled # Update 'returns' col
                last_sequence.append(next_row)
                
                # --- Convert predicted return back to price ---
                dummy_row = np.zeros((1, n_features))
                dummy_row[0, 0] = predicted_scaled
                predicted_return = saved_scaler.inverse_transform(dummy_row)[0][0]
                
                # Apply return: P_t+1 = P_t * (1 + return)
                predicting_price = predicting_price * (1 + predicted_return)
                
                # Final check in real price space
                if predicting_price > current_price * 2.0 or predicting_price < current_price * 0.5:
                    print(f"⚠️ Extreme price projection for {ticker}: ${predicting_price:.2f}. Falling back.")
                    raise ValueError("Unrealistic price jump")
                
                predictions.append(float(predicting_price))
            
            return predictions
        
        else:
            # --- Safe fallback to technical analysis if scaler is missing ---
            # Never use on-the-fly scaling for a pre-trained model as it produces garbage
            print(f"⚠️ No scaler found for {ticker} LSTM. Falling back to technical analysis for accuracy.")
            for day in range(days):
                pred = predict_with_technical_analysis(hist, current_price)
                predictions.append(pred)
                current_price = pred
            return predictions
        
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        
        # FALLBACK 2: Technical analysis
        for day in range(days):
            pred = predict_with_technical_analysis(hist, current_price)
            predictions.append(pred)
            current_price = pred
        return predictions

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
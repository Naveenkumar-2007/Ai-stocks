from flask import Flask, request, render_template, jsonify, send_from_directory, g
from flask_cors import CORS
import os
import json
import base64
from datetime import datetime, timedelta
import warnings
import logging
import threading
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
    print("Automatic model training started (runs daily with startup catch-up)")
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
    
    # NEW: Register interest in this ticker immediately (adds to stocks.json + DB)
    try:
        from mlops.config import MLOpsConfig
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
        with _training_state_lock:
            should_start_training = ticker not in _training_in_progress
            if should_start_training:
                _training_in_progress.add(ticker)

        if not should_start_training:
            print(f"⏳ Training already in progress for {ticker}. Skipping duplicate trigger.")
            return

        print(f"🚀 No model for {ticker}. Triggering background training...")

        def background_train():
            try:
                # Train Unified Engine v4.0 FIRST (primary prediction path)
                try:
                    from unified_engine.training import UnifiedTrainer
                    print(f"Background Unified Engine v4.0 training started for {ticker}")
                    ue_result = UnifiedTrainer.train(ticker)
                    if ue_result.success:
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
                    print(f"🚀 Background V1 training started for {ticker}")
                    pipeline.train_model(ticker=ticker, epochs=20, days=730)
                    print(f"✅ Background V1 training completed for {ticker}")
                except Exception as v1_err:
                    print(f"⚠️ Background V1 training failed for {ticker}: {v1_err}")

                # Train V2 (powers inference metrics and drift detection)
                from mlops_v2.training import TrainerV2
                trainer = TrainerV2()
                result = trainer.train_if_needed(ticker=ticker, force=True)
                if result.trained:
                    print(f"✅ Background v2 training completed for {ticker}")
                else:
                    print(f"⚠️ Background v2 training skipped for {ticker}: {result.reason}")
            except Exception as e:
                print(f"❌ Background training failed for {ticker}: {e}")
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
    
    days = request.args.get('days', default=7, type=int)
    days = max(1, min(days, 30))  # Limit between 1-30 days
    
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
                        
                        accuracy = float(metrics.get("xgb_accuracy", 0.0))
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
            'is_training': len(predictions) == 0,
            'prediction_ready': len(predictions) > 0,
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
                'fetch_days': CONFIG.fetch_days,
                'purge_days': CONFIG.wf_purge_days,
                'embargo_days': CONFIG.wf_embargo_days,
                'max_features': CONFIG.max_features,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def predict_multi_day_lstm(hist, current_price, days, ticker):
    """Predict multiple days ahead — Unified Engine v4.0 is the primary path."""
    import numpy as np
    import ta as ta_lib
    
    predictions = []

    def trigger_background_ultimate_training():
        ticker_key = ticker.upper()
        with _ultimate_training_state_lock:
            should_start_training = ticker_key not in _ultimate_training_in_progress
            if should_start_training:
                _ultimate_training_in_progress.add(ticker_key)

        if not should_start_training:
            print(f"v3.6 training already in progress for {ticker_key}.")
            return

        def background_train():
            try:
                print(f"Background v3.6 training started for {ticker_key}")
                from ultimate_stock_engine_v36 import train_ultimate_model
                train_result = train_ultimate_model(ticker_key, use_regime=True, generate_charts=True)
                if train_result:
                    print(f"Background v3.6 training completed for {ticker_key}")
                else:
                    print(f"Background v3.6 training returned no artifact for {ticker_key}")
            except Exception as train_err:
                print(f"Background v3.6 training failed for {ticker_key}: {train_err}")
            finally:
                with _ultimate_training_state_lock:
                    _ultimate_training_in_progress.discard(ticker_key)

        threading.Thread(target=background_train, daemon=True).start()
    
    try:
        _ultimate_prediction_cache.pop(ticker.upper(), None)

        # =====================================================================
        # PRIMARY PATH: Ultimate Engine v3.6 (PROVEN — real backtested results)
        # 4-model ensemble + regime detection + calibrated predictions
        # =====================================================================
        try:
            from ultimate_stock_engine_v36 import predict_ultimate_realtime
            ultimate_payload = predict_ultimate_realtime(ticker, hist, current_price=current_price, days=days)
            if ultimate_payload and ultimate_payload.get('predicted_prices'):
                _ultimate_prediction_cache[ticker.upper()] = ultimate_payload
                print(f"✅ v3.6 prediction for {ticker}: signal={ultimate_payload.get('signal')}, "
                      f"prob={ultimate_payload.get('direction_prob', 0):.3f}, "
                      f"confidence={ultimate_payload.get('confidence', 0):.3f}")
                
                # Update Prometheus metrics for Grafana
                try:
                    from mlops_v2.monitoring import set_accuracy_20d, set_simulated_pnl, set_sharpe_ratio, inc_prediction
                    metrics = ultimate_payload.get('metrics', {})
                    if 'accuracy' in metrics:
                        set_accuracy_20d(ticker.upper(), metrics['accuracy'])
                    if 'total_return' in metrics:
                        set_simulated_pnl(ticker.upper(), metrics['total_return'])
                    if 'sharpe_ratio' in metrics:
                        set_sharpe_ratio(ticker.upper(), metrics['sharpe_ratio'])
                    inc_prediction(ticker.upper())
                except Exception as metric_err:
                    print(f"⚠️ Prometheus metrics update failed for {ticker}: {metric_err}")
                    
                return ultimate_payload['predicted_prices']
            else:
                # No trained model exists. Train in the background so the API can
                # still return market data before frontend/proxy timeouts fire.
                print(f"No v3.6 model for {ticker}. Triggering background training...")
                trigger_background_ultimate_training()
        except Exception as ultimate_err:
            print(f"Ultimate Engine v3.6 unavailable for {ticker}: {ultimate_err}")

        current_model = load_lstm_model(ticker)
        
        if current_model is None:
            # User specifically requested NOT to give mathematical calculations as a fallback.
            # Instead, we return an empty list to indicate that the model is still training in the background.
            return []
        
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
            # Strict production behavior: scaler mismatch means model output is unsafe.
            print(f"⚠️ No scaler found for {ticker}. Returning training-in-progress state.")
            return []
        
    except Exception as e:
        print(f"LSTM prediction error: {e}")
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

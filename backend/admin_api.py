from flask import Blueprint, jsonify, request, make_response
from database import db_session
from models import User, ActiveTicker, ChatSession, ChatMessage, Watchlist, PredictionLog
from sqlalchemy import func
import datetime
import threading
import os
import shutil
from pathlib import Path
import json
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import time
import secrets


def _json_nocache(payload, status_code=200):
    """Return JSON response with explicit no-cache headers for live dashboards."""
    response = make_response(jsonify(payload), status_code)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


def _safe_remove(path: Path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def _parse_admin_emails():
    raw = os.getenv('ADMIN_EMAILS', '')
    return {item.strip().lower() for item in raw.split(',') if item.strip()}


def _load_chat_logs_from_storage(limit: int):
    chat_dir = Path(__file__).resolve().parent / 'chatbot' / 'app' / 'chat_storage'
    if not chat_dir.exists():
        return []

    entries = []
    for file_path in chat_dir.glob('*.json'):
        try:
            data = json.loads(file_path.read_text(encoding='utf-8'))
        except Exception:
            continue

        timestamp = data.get('updated_at') or data.get('created_at')
        messages = data.get('messages', [])
        for msg in messages:
            role = (msg.get('role') or '').lower()
            sender = 'user' if role == 'user' else 'ai'
            entries.append({
                'session_id': data.get('chat_id'),
                'sender': sender,
                'content': msg.get('content', ''),
                'timestamp': timestamp
            })

    entries.sort(key=lambda item: item.get('timestamp') or '', reverse=True)
    return entries[:limit]

# Generate an unguessable fallback password to prevent unauthorized access if the env variable is missing
DEFAULT_SECURE_PASSWORD = secrets.token_hex(32)

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

# System start time for uptime calculation
START_TIME = time.time()

@admin_bp.route('/stats', methods=['GET'])
def get_admin_stats():
    """Get high-level statistics for the admin dashboard."""
    user_count = db_session.query(User).count()
    active_models_count = db_session.query(ActiveTicker).filter(ActiveTicker.is_active == True).count()
    chat_sessions_count = db_session.query(ChatSession).count()
    
    # Calculate Uptime
    uptime_seconds = time.time() - START_TIME
    uptime_hours = round(uptime_seconds / 3600, 2)
    
    # CPU & RAM Usage
    if PSUTIL_AVAILABLE:
        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_usage_percent = ram.percent
    else:
        cpu_usage = 0.0
        ram_usage_percent = 0.0
    
    # All-Time Searched Stocks
    all_time_query = db_session.query(
        PredictionLog.ticker, 
        func.count(PredictionLog.ticker).label('search_count')
    ).group_by(PredictionLog.ticker).order_by(func.count(PredictionLog.ticker).desc()).limit(5).all()
    
    top_stocks_all_time = [{"ticker": row[0], "count": row[1]} for row in all_time_query]
    
    # Currently Trending (Last 24 Hours)
    yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    trending_query = db_session.query(
        PredictionLog.ticker, 
        func.count(PredictionLog.ticker).label('search_count')
    ).filter(PredictionLog.timestamp >= yesterday).group_by(PredictionLog.ticker).order_by(func.count(PredictionLog.ticker).desc()).limit(5).all()
    
    trending_stocks = [{"ticker": row[0], "count": row[1]} for row in trending_query]
    
    total_watchlist_items = db_session.query(Watchlist).count()

    return _json_nocache({
        "success": True,
        "metrics": {
            "total_users": user_count,
            "active_models": active_models_count,
            "total_chat_sessions": chat_sessions_count,
            "total_watchlist_items": total_watchlist_items,
            "api_health": "Optimal",
            "uptime_hours": uptime_hours,
            "cpu_usage": cpu_usage,
            "ram_usage_percent": ram_usage_percent,
            "top_searched_stocks_all_time": top_stocks_all_time,
            "trending_stocks": trending_stocks
        }
    })

@admin_bp.route('/verify', methods=['POST'])
def verify_admin():
    """Enterprise feature: Secure Master Password verification."""
    data = request.get_json(silent=True) or {}
    password = str(data.get('password', '')).strip()
    # Use environment variable for master password, fallback to secure random token
    master_password = str(os.getenv('ADMIN_MASTER_PASSWORD', DEFAULT_SECURE_PASSWORD)).strip().strip('"').strip("'")
    
    if password == master_password:
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Invalid master password"}), 401

@admin_bp.route('/users', methods=['GET'])
def get_users():
    """Enterprise feature: Get all registered users and their stats."""
    users = db_session.query(User).all()

    firebase_auth = None
    try:
        import firebase_admin
        if firebase_admin._apps:  # type: ignore[attr-defined]
            from firebase_admin import auth as firebase_auth_mod
            firebase_auth = firebase_auth_mod
    except Exception:
        firebase_auth = None

    result = []
    for u in users:
        watchlist_entries = db_session.query(Watchlist).filter_by(user_id=u.id).all()
        watchlist_tickers = sorted({w.ticker.upper() for w in watchlist_entries if w.ticker})
        watchlist_count = len(watchlist_tickers)
        chat_count = db_session.query(ChatSession).filter_by(user_id=u.id).count()

        top_search = db_session.query(
            PredictionLog.ticker,
            func.count(PredictionLog.id).label('search_count')
        ).filter(PredictionLog.user_id == u.id).group_by(PredictionLog.ticker).order_by(func.count(PredictionLog.id).desc()).first()

        latest_search = db_session.query(func.max(PredictionLog.timestamp)).filter(PredictionLog.user_id == u.id).scalar()

        display_name = None
        if firebase_auth and u.firebase_uid:
            try:
                firebase_user = firebase_auth.get_user(u.firebase_uid)
                display_name = firebase_user.display_name
            except Exception:
                display_name = None

        if not display_name:
            if u.email:
                display_name = u.email.split('@')[0]
            else:
                display_name = f"user-{u.id}"

        result.append({
            "id": u.id,
            "email": u.email,
            "name": display_name,
            "role": u.role,
            "tier": u.subscription_tier,
            "joined": u.created_at.isoformat() if u.created_at else None,
            "watchlists": watchlist_count,
            "watchlist_tickers": watchlist_tickers,
            "chats": chat_count,
            "top_searched_stock": top_search[0] if top_search else None,
            "top_searched_count": int(top_search[1]) if top_search else 0,
            "last_active_at": latest_search.isoformat() if latest_search else None
        })
    return _json_nocache({"success": True, "users": result})


@admin_bp.route('/firebase/import-users', methods=['POST'])
def import_firebase_users():
    """Sync Firebase Auth users into the SQL users table."""
    try:
        import firebase_admin
        if not firebase_admin._apps:  # type: ignore[attr-defined]
            return _json_nocache({"success": False, "error": "Firebase Admin SDK not configured."}, 503)
        from firebase_admin import auth as firebase_auth
    except Exception:
        return _json_nocache({"success": False, "error": "Firebase Admin SDK not available."}, 503)

    admin_emails = _parse_admin_emails()
    created = 0
    updated = 0
    scanned = 0

    try:
        for fb_user in firebase_auth.list_users().iterate_all():
            scanned += 1
            uid = fb_user.uid
            email = (fb_user.email or '').strip().lower() or None
            if not uid:
                continue

            user = db_session.query(User).filter_by(firebase_uid=uid).first()
            role = "admin" if email and email in admin_emails else "user"
            if user:
                changed = False
                if email and user.email != email:
                    user.email = email
                    changed = True
                if role and user.role != role:
                    user.role = role
                    changed = True
                if changed:
                    updated += 1
            else:
                user = User(firebase_uid=uid, email=email, role=role)
                db_session.add(user)
                created += 1

        db_session.commit()
    except Exception as exc:
        db_session.rollback()
        return _json_nocache({"success": False, "error": str(exc)}, 500)

    return _json_nocache({
        "success": True,
        "scanned": scanned,
        "created": created,
        "updated": updated
    })

@admin_bp.route('/users/<int:user_id>/tier', methods=['POST'])
def update_user_tier(user_id):
    """Enterprise feature: Upgrade/Downgrade user subscription."""
    data = request.json
    new_tier = data.get('tier')
    user = db_session.query(User).get(user_id)
    if user and new_tier:
        user.subscription_tier = new_tier
        db_session.commit()
        return jsonify({"success": True, "message": f"User upgraded to {new_tier}"})
    return jsonify({"success": False, "error": "User not found"}), 404

@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Enterprise feature: Completely remove a user."""
    user = db_session.query(User).get(user_id)
    if user:
        # Also clean up dependent records
        db_session.query(Watchlist).filter_by(user_id=user_id).delete()
        chat_sessions = db_session.query(ChatSession).filter_by(user_id=user_id).all()
        for session in chat_sessions:
            db_session.query(ChatMessage).filter_by(session_id=session.id).delete()
        db_session.query(ChatSession).filter_by(user_id=user_id).delete()
        
        db_session.delete(user)
        db_session.commit()
        return jsonify({"success": True, "message": "User and all associated data permanently deleted."})
    return jsonify({"success": False, "error": "User not found"}), 404

@admin_bp.route('/models', methods=['GET'])
def get_active_models():
    """Get all active models and their drift scores."""
    include_untrained = str(request.args.get('include_untrained', 'false')).strip().lower() in ('1', 'true', 'yes', 'on')
    query = db_session.query(ActiveTicker).filter(ActiveTicker.is_active == True)
    if not include_untrained:
        query = query.filter(ActiveTicker.last_trained_date.isnot(None))
    tickers = query.order_by(ActiveTicker.ticker.asc()).all()
    result = []
    for t in tickers:
        result.append({
            "ticker": t.ticker,
            "last_trained": t.last_trained_date.isoformat() if t.last_trained_date else "Never",
            "drift_score": t.current_drift_score,
            "status": "Healthy" if t.current_drift_score < 0.25 else "Decaying"
        })
    return _json_nocache({"success": True, "models": result})

@admin_bp.route('/force_retrain/<ticker>', methods=['POST'])
def force_retrain(ticker):
    """Admin override to force retrain a specific model immediately."""
    ticker = ticker.upper()
    
    def background_train():
        try:
            from unified_engine.training import train_unified_model
            print(f"[ADMIN] 🚀 Force retraining started for {ticker}")
            train_result = train_unified_model(ticker)
            
            import datetime
            db_ticker = db_session.query(ActiveTicker).filter_by(ticker=ticker).first()
            if db_ticker:
                db_ticker.last_trained_date = datetime.datetime.utcnow()
                db_ticker.current_drift_score = 0.0
            else:
                db_session.add(ActiveTicker(
                    ticker=ticker, is_active=True,
                    last_trained_date=datetime.datetime.utcnow(),
                    current_drift_score=0.0
                ))
            db_session.commit()
            if train_result.success:
                print(f"[ADMIN] ✅ Force retrain completed for {ticker}: {train_result.model_version}")
            else:
                print(f"[ADMIN] ⚠️ Force retrain failed for {ticker}: {train_result.reason}")
        except Exception as e:
            print(f"[ADMIN] Force retraining failed: {e}")
            
    threading.Thread(target=background_train, daemon=True).start()
    return jsonify({"success": True, "message": f"Force retrain triggered in background for {ticker}."})

@admin_bp.route('/train_all', methods=['POST'])
def train_all_stocks():
    """Trigger a full batch training of all stocks immediately."""
    def background_train_all():
        try:
            print("[ADMIN] 🚀 Force training ALL stocks started...")
            from scheduler import scheduler
            # By default the job skips already trained stocks for today, 
            # so we should temporarily override that or just let it run.
            scheduler.train_models_job()
        except Exception as e:
            print(f"[ADMIN] ❌ Train All failed: {e}")
            
    threading.Thread(target=background_train_all, daemon=True).start()
    return jsonify({"success": True, "message": "Global training triggered. Processing all stocks in the background."})

@admin_bp.route('/set_training_time', methods=['POST'])
def set_training_time():
    """Set the daily scheduled training time (UTC)."""
    data = request.json
    new_time = data.get('time')
    if not new_time:
        return jsonify({"success": False, "error": "No time provided"}), 400
        
    try:
        from scheduler import scheduler
        import schedule
        
        scheduler.training_time_utc = new_time
        schedule.clear()
        
        if scheduler.weekdays_only:
            schedule.every().monday.at(scheduler.training_time_utc).do(scheduler.train_models_job)
            schedule.every().tuesday.at(scheduler.training_time_utc).do(scheduler.train_models_job)
            schedule.every().wednesday.at(scheduler.training_time_utc).do(scheduler.train_models_job)
            schedule.every().thursday.at(scheduler.training_time_utc).do(scheduler.train_models_job)
            schedule.every().friday.at(scheduler.training_time_utc).do(scheduler.train_models_job)
        else:
            schedule.every().day.at(scheduler.training_time_utc).do(scheduler.train_models_job)
            
        return jsonify({"success": True, "message": f"Global training schedule updated to {new_time} UTC."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@admin_bp.route('/trigger_tuning', methods=['POST'])
def trigger_tuning():
    """Enterprise feature: Trigger heavy Optuna Hyperparameter Tuning."""
    def background_tuning():
        try:
            print("[ADMIN] 🚀 Heavy Optuna tuning started...")
            from mlops_v2.run_weekly_tuning import main
            main()
        except Exception as e:
            print(f"[ADMIN] ❌ Tuning failed: {e}")
            
    threading.Thread(target=background_tuning, daemon=True).start()
    return jsonify({"success": True, "message": "Massive hyperparameter tuning cluster job started."})
    
@admin_bp.route('/chat_logs', methods=['GET'])
def get_chat_logs():
    """View recent chatbot conversations for quality assurance."""
    limit = request.args.get('limit', 50, type=int)
    messages = db_session.query(ChatMessage).order_by(ChatMessage.timestamp.desc()).limit(limit).all()
    
    result = []
    for m in messages:
        result.append({
            "session_id": m.session_id,
            "sender": m.sender,
            "content": m.content,
            "timestamp": m.timestamp.isoformat()
        })

    if not result:
        fallback = _load_chat_logs_from_storage(limit)
        return _json_nocache({"success": True, "logs": fallback})

    return _json_nocache({"success": True, "logs": result})


@admin_bp.route('/integrations/verify', methods=['GET'])
def verify_integrations():
    """Verify external integrations needed for production observability and model ops."""
    grafana_user = os.getenv('GRAFANA_USER', '').strip()
    grafana_token = os.getenv('GRAFANA_TOKEN', '').strip()
    grafana_auth = (grafana_user, grafana_token) if grafana_user and grafana_token else None
    remote_write_url = os.getenv('GRAFANA_REMOTE_WRITE_URL', '').strip() or os.getenv('PROM_REMOTE_WRITE_URL', '').strip()

    report = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "dagshub": {
            "configured": bool(os.getenv('DAGSHUB_REPO_OWNER') and os.getenv('DAGSHUB_REPO_NAME') and os.getenv('DAGSHUB_TOKEN')),
            "owner": os.getenv('DAGSHUB_REPO_OWNER', ''),
            "repo": os.getenv('DAGSHUB_REPO_NAME', '')
        },
        "mlflow": {
            "tracking_uri": os.getenv('MLFLOW_TRACKING_URI', '').strip() or 'auto',
            "ok": False,
            "error": None
        },
        "grafana": {
            "url": os.getenv('GRAFANA_URL', 'http://localhost:3000'),
            "auth_configured": bool(grafana_auth),
            "remote_write_configured": bool(remote_write_url),
            "ok": False,
            "error": None
        },
        "prometheus": {
            "url": os.getenv('PROMETHEUS_URL', 'http://localhost:9090'),
            "ok": False,
            "error": None
        }
    }

    try:
        import mlflow
        from mlops.config import MLOpsConfig
        mlflow.set_tracking_uri(MLOpsConfig.MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
        report['mlflow']['tracking_uri'] = MLOpsConfig.MLFLOW_TRACKING_URI
        report['mlflow']['ok'] = True
    except Exception as exc:
        report['mlflow']['error'] = str(exc)

    try:
        import requests
        grafana_url = report['grafana']['url'].rstrip('/') + '/api/health'
        resp = requests.get(grafana_url, timeout=5, auth=grafana_auth)
        report['grafana']['ok'] = resp.ok
        if not resp.ok:
            report['grafana']['error'] = f"HTTP {resp.status_code}"
    except Exception as exc:
        report['grafana']['error'] = str(exc)

    try:
        import requests
        prom_url = report['prometheus']['url'].rstrip('/') + '/-/ready'
        resp = requests.get(prom_url, timeout=5)
        report['prometheus']['ok'] = resp.ok
        if not resp.ok:
            report['prometheus']['error'] = f"HTTP {resp.status_code}"
    except Exception as exc:
        report['prometheus']['error'] = str(exc)

    overall_ok = report['mlflow']['ok'] and report['grafana']['ok'] and report['prometheus']['ok']
    return _json_nocache({"success": overall_ok, "report": report})


@admin_bp.route('/reset_mlops_state', methods=['POST'])
def reset_mlops_state():
    """
    Fresh-start reset for production ML state.
    This clears local model registry/runs and optionally the MLflow experiment
    and registered model entries.
    """
    data = request.get_json(silent=True) or {}
    confirm = str(data.get('confirm', '')).strip()
    if confirm != 'RESET_MY_MLOPS_STATE':
        return _json_nocache({
            "success": False,
            "error": "Confirmation phrase mismatch. Use confirm=RESET_MY_MLOPS_STATE"
        }, 400)

    clear_prediction_logs = bool(data.get('clear_prediction_logs', True))
    clear_active_tickers = bool(data.get('clear_active_tickers', True))
    wipe_mlflow_experiment = bool(data.get('wipe_mlflow_experiment', False))
    wipe_all_mlflow_experiments = bool(data.get('wipe_all_mlflow_experiments', False))
    wipe_mlflow_registered_models = bool(data.get('wipe_mlflow_registered_models', False))
    wipe_ultimate_mlflow_experiment = bool(data.get('wipe_ultimate_mlflow_experiment', True))
    seed_stocks = data.get('seed_stocks', [])
    if not isinstance(seed_stocks, list):
        seed_stocks = []

    root_dir = Path(__file__).resolve().parent
    removed_paths = []
    for rel_path in [
        Path('mlruns'),
        Path('mlops/model_registry/model_store'),
        Path('mlops/model_registry/metadata.json'),
        Path('mlops/model_registry/global_features.json'),
        Path('mlops/model_registry/global_scalers.pkl'),
        Path('mlops/ultimate_engine'),
        Path('mlops_v2/models'),
        Path('unified_engine/models'),
        Path('unified_engine/training_results.json'),
        Path('mlops/metrics/all_stocks_metrics.json'),
        Path('mlops/metrics/summary.json')
    ]:
        abs_path = root_dir / rel_path
        if abs_path.exists():
            _safe_remove(abs_path)
            removed_paths.append(str(rel_path))

    stocks_file = root_dir / 'mlops' / 'stocks.json'
    try:
        from mlops.config import MLOpsConfig
        normalized = []
        for symbol in seed_stocks:
            sym = MLOpsConfig.normalize_ticker(str(symbol))
            if MLOpsConfig.is_valid_ticker(sym) and sym not in normalized:
                normalized.append(sym)
        stocks_file.write_text(json.dumps(normalized, indent=4), encoding='utf-8')
    except Exception as exc:
        return _json_nocache({"success": False, "error": f"Failed to rewrite stocks.json: {exc}"}, 500)

    db_summary = {
        "prediction_logs_deleted": 0,
        "active_tickers_deleted": 0
    }

    try:
        if clear_prediction_logs:
            db_summary['prediction_logs_deleted'] = db_session.query(PredictionLog).delete(synchronize_session=False)
        if clear_active_tickers:
            db_summary['active_tickers_deleted'] = db_session.query(ActiveTicker).delete(synchronize_session=False)
        db_session.commit()
    except Exception as exc:
        db_session.rollback()
        return _json_nocache({"success": False, "error": f"DB reset failed: {exc}"}, 500)

    mlflow_reset = {
        "attempted": False,
        "deleted_experiment": None,
        "deleted_experiments": [],
        "deleted_registered_models": [],
        "registered_model_delete_errors": [],
        "error": None
    }
    if wipe_mlflow_experiment or wipe_all_mlflow_experiments or wipe_mlflow_registered_models or wipe_ultimate_mlflow_experiment:
        mlflow_reset['attempted'] = True
        try:
            import mlflow
            from mlops.config import MLOpsConfig
            mlflow.set_tracking_uri(MLOpsConfig.MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()

            if wipe_mlflow_registered_models:
                model_names = [
                    getattr(model, 'name', '')
                    for model in client.search_registered_models()
                    if getattr(model, 'name', '').startswith('LSTM_')
                ]
                for model_name in model_names:
                    try:
                        client.delete_registered_model(model_name)
                        mlflow_reset['deleted_registered_models'].append(model_name)
                    except Exception as model_exc:
                        error_text = str(model_exc)
                        if 'RESOURCE_DOES_NOT_EXIST' in error_text:
                            continue
                        mlflow_reset['registered_model_delete_errors'].append({
                            "model": model_name,
                            "error": error_text
                        })

            if wipe_all_mlflow_experiments:
                for exp in client.search_experiments():
                    # Skip non-active stages or protected default experiment only if needed.
                    if getattr(exp, 'lifecycle_stage', 'active') != 'active':
                        continue
                    client.delete_experiment(exp.experiment_id)
                    mlflow_reset['deleted_experiments'].append(exp.name)
            else:
                experiment_names = []
                if wipe_mlflow_experiment:
                    experiment_names.append(os.getenv('MLFLOW_EXPERIMENT_NAME', 'Prediction_Lineage'))
                if wipe_ultimate_mlflow_experiment:
                    experiment_names.append('ultimate_engine_v36')

                for exp_name in dict.fromkeys(experiment_names):
                    exp = client.get_experiment_by_name(exp_name)
                    if exp and getattr(exp, 'lifecycle_stage', 'active') == 'active':
                        client.delete_experiment(exp.experiment_id)
                        mlflow_reset['deleted_experiments'].append(exp_name)
                        if mlflow_reset['deleted_experiment'] is None:
                            mlflow_reset['deleted_experiment'] = exp_name
        except Exception as exc:
            mlflow_reset['error'] = str(exc)

    try:
        import app as app_module
        for cache_name in ('_model_cache', '_scaler_cache', '_model_metadata', '_ultimate_prediction_cache'):
            cache = getattr(app_module, cache_name, None)
            if isinstance(cache, dict):
                cache.clear()
    except Exception:
        pass

    try:
        from unified_engine.inference import clear_cache as clear_unified_cache
        clear_unified_cache()
    except Exception:
        pass

    return _json_nocache({
        "success": True,
        "message": "MLOps state reset complete. Fresh training can start from the provided seed stocks.",
        "removed_paths": removed_paths,
        "seed_stocks": seed_stocks,
        "database": db_summary,
        "mlflow": mlflow_reset
    })

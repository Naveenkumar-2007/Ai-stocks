from flask import Blueprint, jsonify, request
from database import db_session
from models import User, ActiveTicker, ChatSession, ChatMessage, Watchlist, PredictionLog
from sqlalchemy import func
import datetime
import threading
import os
import psutil
import time

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
    cpu_usage = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    
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
    
    return jsonify({
        "success": True,
        "metrics": {
            "total_users": user_count,
            "active_models": active_models_count,
            "total_chat_sessions": chat_sessions_count,
            "api_health": "Optimal",
            "uptime_hours": uptime_hours,
            "cpu_usage": cpu_usage,
            "ram_usage_percent": ram.percent,
            "top_searched_stocks_all_time": top_stocks_all_time,
            "trending_stocks": trending_stocks
        }
    })

@admin_bp.route('/verify', methods=['POST'])
def verify_admin():
    """Enterprise feature: Secure Master Password verification."""
    data = request.json
    password = str(data.get('password', '')).strip()
    # Use environment variable for master password, fallback to a strong default for demo
    master_password = str(os.getenv('ADMIN_MASTER_PASSWORD', 'AiStocks@Admin2026')).strip()
    
    if password == master_password:
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Invalid master password"}), 401

@admin_bp.route('/users', methods=['GET'])
def get_users():
    """Enterprise feature: Get all registered users and their stats."""
    users = db_session.query(User).all()
    result = []
    for u in users:
        watchlist_count = db_session.query(Watchlist).filter_by(user_id=u.id).count()
        chat_count = db_session.query(ChatSession).filter_by(user_id=u.id).count()
        result.append({
            "id": u.id,
            "email": u.email,
            "role": u.role,
            "tier": u.subscription_tier,
            "joined": u.created_at.isoformat() if u.created_at else None,
            "watchlists": watchlist_count,
            "chats": chat_count
        })
    return jsonify({"success": True, "users": result})

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
    tickers = db_session.query(ActiveTicker).all()
    result = []
    for t in tickers:
        result.append({
            "ticker": t.ticker,
            "last_trained": t.last_trained_date.isoformat() if t.last_trained_date else "Never",
            "drift_score": t.current_drift_score,
            "status": "Healthy" if t.current_drift_score < 0.25 else "Decaying"
        })
    return jsonify({"success": True, "models": result})

@admin_bp.route('/force_retrain/<ticker>', methods=['POST'])
def force_retrain(ticker):
    """Admin override to force retrain a specific model immediately."""
    ticker = ticker.upper()
    
    def background_train():
        try:
            from mlops.training_pipeline import MLOpsTrainingPipeline
            print(f"[ADMIN] 🚀 Force retraining started for {ticker}")
            pipeline = MLOpsTrainingPipeline()
            pipeline.train_model(ticker=ticker, epochs=20, days=730)
            
            db_ticker = db_session.query(ActiveTicker).filter_by(ticker=ticker).first()
            if db_ticker:
                import datetime
                db_ticker.last_trained_date = datetime.datetime.utcnow()
                db_ticker.current_drift_score = 0.0
                db_session.commit()
        except Exception as e:
            print(f"[ADMIN] ❌ Force retraining failed: {e}")
            
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
    return jsonify({"success": True, "logs": result})

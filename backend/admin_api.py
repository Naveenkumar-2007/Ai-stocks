from flask import Blueprint, jsonify, request
from database import db_session
from models import User, ActiveTicker, ChatSession, ChatMessage
import threading

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

# NOTE: In production, wrap these endpoints with @firebase_auth_required(admin_only=True)
# We are skipping it here so you can test it easily without Firebase config.

@admin_bp.route('/stats', methods=['GET'])
def get_admin_stats():
    """Get high-level statistics for the admin dashboard."""
    user_count = db_session.query(User).count()
    active_models_count = db_session.query(ActiveTicker).filter(ActiveTicker.is_active == True).count()
    chat_sessions_count = db_session.query(ChatSession).count()
    
    return jsonify({
        "success": True,
        "metrics": {
            "total_users": user_count,
            "active_models": active_models_count,
            "total_chat_sessions": chat_sessions_count,
            "api_health": "Optimal"
        }
    })

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
            
            # Update DB timestamp
            db_ticker = db_session.query(ActiveTicker).filter_by(ticker=ticker).first()
            if db_ticker:
                import datetime
                db_ticker.last_trained_date = datetime.datetime.utcnow()
                db_ticker.current_drift_score = 0.0 # Reset drift
                db_session.commit()
                
            print(f"[ADMIN] ✅ Force retraining completed for {ticker}")
        except Exception as e:
            print(f"[ADMIN] ❌ Force retraining failed: {e}")
            
    threading.Thread(target=background_train, daemon=True).start()
    
    return jsonify({
        "success": True,
        "message": f"Force retrain triggered in background for {ticker}."
    })
    
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

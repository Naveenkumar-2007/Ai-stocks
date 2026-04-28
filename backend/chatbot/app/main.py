# app/main.py
"""
FastAPI entry point for the Stock Trading Chatbot.
Provides conversational /chat endpoint with persistent memory and learning support.
"""
import os
import sys
import json
import uuid
import uvicorn
import importlib.util
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from models.schemas import ChatRequest, ChatResponse, ChatSession, ChatMessage
from core.agent import agent
from core.learning import save_feedback, get_feedback_stats
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_storage")
os.makedirs(CHATS_DIR, exist_ok=True)

_db_session = None
_backend_models = None


def _load_backend_db():
    """Load backend database/session and models without clashing with chatbot models."""
    global _db_session, _backend_models
    if _db_session is not None and _backend_models is not None:
        return True

    backend_dir = Path(__file__).resolve().parents[2]
    database_path = backend_dir / "database.py"
    models_path = backend_dir / "models.py"

    if not database_path.exists() or not models_path.exists():
        return False

    try:
        db_spec = importlib.util.spec_from_file_location("backend_database", str(database_path))
        if not db_spec or not db_spec.loader:
            return False
        db_module = importlib.util.module_from_spec(db_spec)
        db_spec.loader.exec_module(db_module)

        models_spec = importlib.util.spec_from_file_location("backend_models", str(models_path))
        if not models_spec or not models_spec.loader:
            return False
        models_module = importlib.util.module_from_spec(models_spec)
        models_spec.loader.exec_module(models_module)

        _db_session = getattr(db_module, "db_session", None)
        _backend_models = models_module
        return _db_session is not None and _backend_models is not None
    except Exception as exc:
        logger.warning("Failed to load backend DB modules: %s", exc)
        _db_session = None
        _backend_models = None
        return False


def _persist_chat_to_db(chat_id: str, user_message: str, ai_message: str) -> None:
    if not _load_backend_db() or not _db_session or not _backend_models:
        return

    ChatSession = getattr(_backend_models, "ChatSession", None)
    ChatMessage = getattr(_backend_models, "ChatMessage", None)
    if not ChatSession or not ChatMessage:
        return

    try:
        session_row = _db_session.query(ChatSession).filter_by(session_id=chat_id).first()
        if not session_row:
            session_row = ChatSession(session_id=chat_id)
            _db_session.add(session_row)
            _db_session.commit()

        _db_session.add(ChatMessage(session_id=session_row.id, sender="user", content=user_message))
        _db_session.add(ChatMessage(session_id=session_row.id, sender="ai", content=ai_message))
        _db_session.commit()
    except Exception as exc:
        logger.warning("Failed to persist chat to DB: %s", exc)
        try:
            _db_session.rollback()
        except Exception:
            pass

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Production-grade AI Trading Chatbot with Hybrid RAG and Learning"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def validate_config():
    """Check required API keys at startup and log clear warnings if missing."""
    missing = []
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    finnhub_keys = os.getenv("FINNHUB_API_KEYS", "") or os.getenv("FINNHUB_API_KEY", "")
    if not finnhub_keys and not settings.finnhub_api_key:
        missing.append("FINNHUB_API_KEY or FINNHUB_API_KEYS")
    if missing:
        logger.warning("=" * 60)
        logger.warning("⚠️  MISSING API KEYS — chatbot will not work correctly!")
        logger.warning(f"   Missing: {', '.join(missing)}")
        logger.warning("   Create a .env file in the backend/ directory with:")
        logger.warning("   GROQ_API_KEY=your_groq_api_key")
        logger.warning("   FINNHUB_API_KEY=your_finnhub_api_key")
        logger.warning("=" * 60)
    else:
        logger.info("✅ All required API keys are configured")


# ── Chat Storage ──
def _save_chat(session: ChatSession):
    path = os.path.join(CHATS_DIR, f"{session.chat_id}.json")
    with open(path, 'w') as f:
        json.dump(session.dict(), f, indent=2, default=str)

def _load_chat(chat_id: str) -> ChatSession | None:
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return ChatSession(**json.load(f))
    return None

def _generate_title(message: str) -> str:
    title = message.strip()[:50]
    if len(message) > 50: title += "..."
    return title


# ── Feedback Model ──
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str  # "thumbs_up" or "thumbs_down"
    chat_id: Optional[str] = None


# ── Routes ──
@app.get("/")
async def root():
    return {"message": "🚀 AI Trading Chatbot running!", "version": settings.version}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with persistent memory"""
    try:
        logger.info(f"💬 User: {request.message}")
        chat_id = request.chat_id or str(uuid.uuid4())

        session = _load_chat(chat_id) if request.chat_id else None
        if not session:
            session = ChatSession(
                chat_id=chat_id,
                title=_generate_title(request.message),
                messages=[],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )

        history = request.history if request.history else session.messages[-10:]

        result = await agent.chat(request.message, history)

        session.messages.append(ChatMessage(role="user", content=request.message))
        session.messages.append(ChatMessage(role="assistant", content=result.reply))
        session.updated_at = datetime.now().isoformat()
        _save_chat(session)
        _persist_chat_to_db(chat_id, request.message, result.reply)

        logger.info(f"🤖 Bot: {result.reply[:80]}...")

        return ChatResponse(
            reply=result.reply,
            stock=result.stock,
            data=result.data,
            chat_id=chat_id
        )
    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats")
async def list_chats():
    """List all saved chat sessions"""
    chats = []
    for f in os.listdir(CHATS_DIR):
        if f.endswith('.json'):
            try:
                with open(os.path.join(CHATS_DIR, f), 'r') as fh:
                    data = json.load(fh)
                    chats.append({
                        "chat_id": data["chat_id"],
                        "title": data["title"],
                        "message_count": len(data.get("messages", [])),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                    })
            except Exception:
                continue
    chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {"chats": chats}

@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    session = _load_chat(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")
    return session.dict()

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)
        if _load_backend_db() and _db_session and _backend_models:
            try:
                ChatSession = getattr(_backend_models, "ChatSession", None)
                ChatMessage = getattr(_backend_models, "ChatMessage", None)
                if ChatSession and ChatMessage:
                    session_row = _db_session.query(ChatSession).filter_by(session_id=chat_id).first()
                    if session_row:
                        _db_session.query(ChatMessage).filter_by(session_id=session_row.id).delete()
                        _db_session.delete(session_row)
                        _db_session.commit()
            except Exception as exc:
                logger.warning("Failed to delete chat from DB: %s", exc)
                try:
                    _db_session.rollback()
                except Exception:
                    pass
        return {"message": "Chat deleted", "chat_id": chat_id}
    raise HTTPException(status_code=404, detail="Chat not found")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit thumbs up/down feedback — drives the learning system"""
    if request.rating not in ("thumbs_up", "thumbs_down"):
        raise HTTPException(status_code=400, detail="Rating must be 'thumbs_up' or 'thumbs_down'")
    save_feedback(request.question, request.answer, request.rating, request.chat_id)
    return {"message": "Feedback recorded", "rating": request.rating}

@app.get("/feedback/stats")
async def feedback_stats():
    """Get learning statistics"""
    return get_feedback_stats()

@app.get("/trained-stocks")
async def get_trained_stocks():
    try:
        stocks_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "mlops", "stocks.json"
        )
        if os.path.exists(stocks_path):
            with open(stocks_path, 'r') as f:
                tickers = json.load(f)
            return {
                "total": len(tickers),
                "tickers": sorted(tickers),
                "us_stocks": sorted([t for t in tickers if '.' not in t]),
                "indian_stocks": sorted([t for t in tickers if '.NS' in t or '.BO' in t]),
            }
        return {"total": 0, "tickers": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

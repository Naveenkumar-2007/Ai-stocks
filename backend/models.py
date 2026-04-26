import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String(128), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True)
    role = Column(String(50), default="user") # 'user' or 'admin'
    subscription_tier = Column(String(50), default="free")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    watchlists = relationship("Watchlist", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")

class Watchlist(Base):
    __tablename__ = 'watchlists'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ticker = Column(String(20), index=True, nullable=False)
    added_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="watchlists")

class ActiveTicker(Base):
    """Tracks which stocks the MLOps pipeline should train every night."""
    __tablename__ = 'active_tickers'
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), unique=True, index=True, nullable=False)
    last_trained_date = Column(DateTime, nullable=True)
    current_drift_score = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)

class ChatSession(Base):
    """Stores AI Chatbot conversations."""
    __tablename__ = 'chat_sessions'
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True) # Nullable for anonymous guests
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('chat_sessions.id'))
    sender = Column(String(50), nullable=False) # 'user' or 'ai'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")

class PredictionLog(Base):
    """Tracks every time a user searches/predicts a stock for global analytics."""
    __tablename__ = 'prediction_logs'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True) # Null if anonymous
    ticker = Column(String(20), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os

class Settings(BaseSettings):
    # API Keys
    finnhub_api_key: str = ""
    groq_api_key: str = ""
    
    # Application
    app_name: str = "StockRAG Bot"
    debug: bool = True
    version: str = "1.0.0"
    
    # LLM Configuration
    llm_model: str = "llama-3.3-70b-versatile" 
    llm_temperature: float = 0.1
    max_tokens: int = 4096
    
    # Vector DB
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vector_db_path: str = os.path.join(base_dir, "data", "vector_store")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Finnhub
    finnhub_base_url: str = "https://finnhub.io/api/v1"
    
    class Config:
        # Find .env in the parent 'backend' directory
        current_file_path = os.path.abspath(__file__)
        # current_file_path is app/config.py
        # parent of app is chatbot/
        # parent of chatbot is backend/
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))), ".env")
        case_sensitive = False
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

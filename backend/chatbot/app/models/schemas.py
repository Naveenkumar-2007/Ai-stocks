# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class StockPrice(BaseModel):
    symbol: str
    current_price: float
    change: float
    percent_change: float
    high: float
    low: float
    open: float
    previous_close: float
    timestamp: datetime


class TechnicalIndicator(BaseModel):
    name: str
    value: float
    signal: Literal["buy", "sell", "neutral", "overbought", "oversold"]
    description: str


class NewsItem(BaseModel):
    headline: str
    source: str
    url: str
    summary: str
    sentiment: str
    sentiment_score: float
    datetime: datetime


class PredictionResult(BaseModel):
    symbol: str
    predicted_direction: str
    confidence: float
    target_price: Optional[float] = None
    key_factors: List[str]
    metrics: Optional[Dict[str, float]] = None


# ── NEW: Company Profile ──
class CompanyProfile(BaseModel):
    name: str
    ticker: str
    country: str = ""
    currency: str = ""
    exchange: str = ""
    sector: str = ""           # "Technology"
    industry: str = ""         # "Consumer Electronics"
    market_cap: float = 0      # in millions
    ipo_date: str = ""
    logo_url: str = ""
    website: str = ""
    phone: str = ""


# ── NEW: Fundamental Financial Data ──
class FundamentalData(BaseModel):
    pe_ratio: Optional[float] = None           # Price-to-Earnings
    pb_ratio: Optional[float] = None           # Price-to-Book
    ps_ratio: Optional[float] = None           # Price-to-Sales
    eps_ttm: Optional[float] = None            # Earnings Per Share (trailing 12m)
    dividend_yield: Optional[float] = None     # Annual dividend yield %
    beta: Optional[float] = None               # Volatility vs market
    week_52_high: Optional[float] = None       # 52-week high
    week_52_low: Optional[float] = None        # 52-week low
    week_52_high_date: str = ""
    week_52_low_date: str = ""
    avg_volume_10d: Optional[float] = None     # 10-day avg volume
    revenue_per_share_ttm: Optional[float] = None
    roe: Optional[float] = None                # Return on Equity %
    current_ratio: Optional[float] = None      # Current assets / liabilities
    debt_to_equity: Optional[float] = None     # Total debt / equity


# ── NEW: Analyst Recommendation ──
class AnalystRecommendation(BaseModel):
    period: str                  # "2025-01"
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0

    @property
    def total(self) -> int:
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell

    @property
    def consensus(self) -> str:
        if self.total == 0:
            return "No Data"
        scores = {
            "Strong Buy": self.strong_buy,
            "Buy": self.buy,
            "Hold": self.hold,
            "Sell": self.sell,
            "Strong Sell": self.strong_sell,
        }
        return max(scores, key=scores.get)


# ── Analysis Context (UPDATED with new fields) ──
class AnalysisContext(BaseModel):
    stock_data: Optional[StockPrice] = None
    technical_indicators: List[TechnicalIndicator] = []
    news_items: List[NewsItem] = []
    prediction: Optional[PredictionResult] = None
    knowledge_context: List[str] = []
    company_profile: Optional[CompanyProfile] = None
    fundamentals: Optional[FundamentalData] = None
    recommendations: List[AnalystRecommendation] = []
    peers: List[str] = []


# --- Chat Models ---
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    chat_id: Optional[str] = None  # For persistent memory


class ChatResponse(BaseModel):
    reply: str
    stock: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    chat_id: Optional[str] = None


class ChatSession(BaseModel):
    chat_id: str
    title: str
    messages: List[ChatMessage] = []
    created_at: str
    updated_at: str

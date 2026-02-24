# app/core/agent.py
"""
The Agentic Brain — handles intent detection, symbol extraction,
topic filtering, and orchestration of the RAG pipeline.
Uses stocks.json for dynamic symbol recognition.
Supports 150+ company-to-ticker mappings including crypto.
"""
import re
import os
import json
import logging
from typing import Optional, Dict, Any, List
from core.rag_pipeline import rag_pipeline
from models.schemas import ChatResponse, ChatMessage

logger = logging.getLogger(__name__)

# ── Load trained tickers from stocks.json ─────────────────────────
def _load_trained_tickers():
    """Load all tickers from stocks.json that have trained LSTM models"""
    try:
        stocks_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "mlops", "stocks.json"
        )
        if os.path.exists(stocks_path):
            with open(stocks_path, 'r') as f:
                tickers = json.load(f)
            logger.info(f"📊 Loaded {len(tickers)} trained stock tickers from stocks.json")
            return set(tickers)
    except Exception as e:
        logger.error(f"Could not load stocks.json: {e}")
    return set()

TRAINED_TICKERS = _load_trained_tickers()

# ── Company Name → Ticker Mapping (150+) ────────────────────────
COMPANY_TICKERS = {
    # ── US Tech / Mega-caps ──
    "tesla": "TSLA", "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "nvidia": "NVDA", "meta": "META",
    "facebook": "META", "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
    "broadcom": "AVGO", "qualcomm": "QCOM", "texas instruments": "TXN",
    "micron": "MU", "applied materials": "AMAT", "lam research": "LRCX",
    "asml": "ASML", "tsmc": "TSM",

    # ── Cloud / SaaS / Software ──
    "salesforce": "CRM", "adobe": "ADBE", "oracle": "ORCL", "cisco": "CSCO",
    "ibm": "IBM", "snowflake": "SNOW", "palantir": "PLTR", "datadog": "DDOG",
    "crowdstrike": "CRWD", "servicenow": "NOW", "workday": "WDAY",
    "splunk": "SPLK", "twilio": "TWLO", "confluent": "CFLT",
    "mongodb": "MDB", "elastic": "ESTC", "hashicorp": "HCP",
    "cloudflare": "NET", "zscaler": "ZS", "okta": "OKTA",
    "hubspot": "HUBS", "atlassian": "TEAM", "docusign": "DOCU",

    # ── E-commerce / Digital ──
    "shopify": "SHOP", "paypal": "PYPL", "square": "SQ", "block": "SQ",
    "uber": "UBER", "airbnb": "ABNB", "doordash": "DASH", "lyft": "LYFT",
    "etsy": "ETSY", "mercadolibre": "MELI", "sea limited": "SE",
    "grab": "GRAB", "coupang": "CPNG", "ebay": "EBAY",

    # ── Social Media / Entertainment ──
    "snap": "SNAP", "snapchat": "SNAP", "pinterest": "PINS", "twitter": "X",
    "zoom": "ZM", "spotify": "SPOT", "roblox": "RBLX", "unity": "U",
    "disney": "DIS", "warner bros": "WBD", "paramount": "PARA",
    "roku": "ROKU", "match group": "MTCH",

    # ── Finance / Banking ──
    "jpmorgan": "JPM", "jpmorgan chase": "JPM", "goldman sachs": "GS",
    "goldman": "GS", "morgan stanley": "MS", "bank of america": "BAC",
    "wells fargo": "WFC", "citigroup": "C", "citi": "C",
    "charles schwab": "SCHW", "blackrock": "BLK",
    "visa": "V", "mastercard": "MA", "american express": "AXP", "amex": "AXP",
    "robinhood": "HOOD", "coinbase": "COIN", "sofi": "SOFI",

    # ── Healthcare / Pharma ──
    "johnson & johnson": "JNJ", "johnson": "JNJ",
    "pfizer": "PFE", "moderna": "MRNA", "unitedhealth": "UNH",
    "eli lilly": "LLY", "lilly": "LLY", "novo nordisk": "NVO",
    "abbvie": "ABBV", "merck": "MRK", "amgen": "AMGN",
    "gilead": "GILD", "regeneron": "REGN", "intuitive surgical": "ISRG",
    "danaher": "DHR", "thermo fisher": "TMO", "abbott": "ABT",

    # ── Consumer / Retail ──
    "coca cola": "KO", "coca-cola": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "nike": "NKE", "walmart": "WMT", "costco": "COST", "target": "TGT",
    "home depot": "HD", "lowe's": "LOW", "lowes": "LOW",
    "starbucks": "SBUX", "mcdonalds": "MCD", "mcdonald's": "MCD",
    "procter & gamble": "PG", "procter": "PG", "p&g": "PG",
    "colgate": "CL", "lululemon": "LULU",

    # ── Industrial / Aerospace / Defense ──
    "boeing": "BA", "lockheed martin": "LMT", "lockheed": "LMT",
    "raytheon": "RTX", "northrop grumman": "NOC", "general dynamics": "GD",
    "caterpillar": "CAT", "deere": "DE", "john deere": "DE",
    "honeywell": "HON", "3m": "MMM", "general electric": "GE",

    # ── Auto / EV ──
    "ford": "F", "general motors": "GM", "gm": "GM",
    "rivian": "RIVN", "lucid": "LCID", "lucid motors": "LCID",
    "nio": "NIO", "xpeng": "XPEV", "li auto": "LI",
    "ferrari": "RACE",

    # ── Energy ──
    "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX",
    "shell": "SHEL", "bp": "BP", "conocophillips": "COP",
    "enphase": "ENPH", "first solar": "FSLR", "nextera": "NEE",

    # ── Berkshire / Conglomerate ──
    "berkshire": "BRK.B", "berkshire hathaway": "BRK.B",

    # ── India ──
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY",
    "hdfc": "HDFCBANK.NS", "hdfc bank": "HDFCBANK.NS",
    "tata motors": "TATAMOTORS.NS", "wipro": "WIPRO.NS",
    "tata steel": "TATASTEEL.NS", "sbi": "SBIN.NS",
    "icici bank": "ICICIBANK.NS", "icici": "ICICIBANK.NS",
    "kotak": "KOTAKBANK.NS", "axis bank": "AXISBANK.NS",
    "bharti airtel": "BHARTIARTL.NS", "airtel": "BHARTIARTL.NS",
    "asian paints": "ASIANPAINT.NS", "titan": "TITAN.NS",
    "maruti": "MARUTI.NS", "maruti suzuki": "MARUTI.NS",
    "sun pharma": "SUNPHARMA.NS", "bajaj finance": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS", "hcl tech": "HCLTECH.NS",
    "larsen & toubro": "LT.NS", "l&t": "LT.NS",
    "power grid": "POWERGRID.NS", "ntpc": "NTPC.NS",
    "adani enterprises": "ADANIENT.NS", "adani": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS",

    # ── Crypto (as tickers) ──
    "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "solana": "SOL-USD",
    "cardano": "ADA-USD", "dogecoin": "DOGE-USD", "ripple": "XRP-USD",
    "polkadot": "DOT-USD", "avalanche": "AVAX-USD",
}

# ── Words that look like tickers but aren't ─────────────────────
TICKER_BLACKLIST = {
    'A', 'I', 'IT', 'FOR', 'THE', 'IS', 'AM', 'AN', 'AS', 'AT', 'BE', 'BY',
    'DO', 'GO', 'HE', 'IF', 'IN', 'ME', 'MY', 'NO', 'OF', 'OK', 'ON', 'OR',
    'SO', 'TO', 'UP', 'US', 'WE', 'AND', 'ARE', 'BUT', 'CAN', 'DID', 'GET',
    'GOT', 'HAS', 'HAD', 'HER', 'HIM', 'HIS', 'HOW', 'ITS', 'LET', 'MAY',
    'NEW', 'NOT', 'NOW', 'OLD', 'OUR', 'OUT', 'OWN', 'PUT', 'SAY', 'SHE',
    'TOO', 'USE', 'WAS', 'WAY', 'WHO', 'WHY', 'WON', 'YET', 'YOU', 'ALL',
    'BIG', 'DAY', 'END', 'FAR', 'FEW', 'HIT', 'HOT', 'LOW', 'MAN', 'RUN',
    'SET', 'TOP', 'TRY', 'TWO', 'WIN', 'YES', 'WHAT', 'WHEN', 'WILL',
    'WITH', 'THAT', 'THIS', 'THEY', 'FROM', 'HAVE', 'BEEN', 'WERE', 'SOME',
    'MUCH', 'VERY', 'JUST', 'LIKE', 'LONG', 'MAKE', 'MANY', 'MOST', 'ONLY',
    'OVER', 'SUCH', 'TAKE', 'THAN', 'THEM', 'WELL', 'ALSO', 'BACK', 'BEEN',
    'COME', 'EACH', 'FIND', 'GIVE', 'GOOD', 'HELP', 'HERE', 'HIGH', 'KEEP',
    'LAST', 'LOOK', 'MOVE', 'MUCH', 'MUST', 'NAME', 'NEED', 'NEXT', 'OPEN',
    'PART', 'PICK', 'PLAY', 'REAL', 'SAME', 'SHOW', 'SIDE', 'TELL', 'TURN',
    'WANT', 'WORK', 'YEAR', 'ABOUT', 'AFTER', 'COULD', 'EVERY', 'FIRST',
    'GREAT', 'NEVER', 'OTHER', 'RIGHT', 'SHALL', 'STILL', 'THEIR', 'THESE',
    'THINK', 'THREE', 'UNDER', 'WHICH', 'WOULD', 'SHOULD', 'COULD', 'STOCK',
    'BUY', 'SELL', 'HOLD', 'PRICE', 'MARKET', 'TRADE', 'BEST', 'GIVE',
    'RSI', 'SMA', 'EMA', 'IPO', 'ETF', 'SEC', 'FED',
    'GDP', 'CPI', 'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'TELL', 'DOES',
    'CHAT', 'BOT', 'HELLO', 'THANKS', 'THANK', 'PLEASE', 'SORRY',
    'AI', 'ML', 'LSTM', 'MODEL', 'TRAIN', 'DATA',
    'VS', 'COMPARE', 'BETWEEN', 'VERSUS',
}

# ── Stock/trading/finance keywords for topic detection ───────────
STOCK_KEYWORDS = [
    # Core trading
    'stock', 'share', 'market', 'trading', 'invest', 'portfolio', 'dividend',
    'bull', 'bear', 'ipo', 'etf', 'mutual fund', 'bond', 'crypto', 'bitcoin',
    'forex', 'option', 'future', 'hedge', 'short', 'long', 'earning',
    'revenue', 'profit', 'loss', 'p/e', 'eps', 'rsi', 'macd', 'sma', 'ema',

    # Technical analysis
    'moving average', 'bollinger', 'candlestick', 'chart', 'technical',
    'stochastic', 'adx', 'vwap', 'fibonacci', 'ichimoku', 'parabolic sar',
    'momentum', 'divergence', 'convergence', 'crossover', 'golden cross',
    'death cross', 'head and shoulders', 'double top', 'double bottom',
    'cup and handle', 'flag pattern', 'wedge', 'triangle pattern',

    # Fundamental analysis
    'fundamental', 'valuation', 'p/b', 'p/s', 'roe', 'roa', 'eps',
    'free cash flow', 'fcf', 'debt to equity', 'current ratio', 'quick ratio',
    'book value', 'intrinsic value', 'dcf', 'discounted cash flow',
    'gross margin', 'operating margin', 'net margin', 'revenue growth',
    'earnings growth', 'balance sheet', 'income statement', 'cash flow',

    # Market indices & exchanges
    'sector', 'index', 'dow', 'nasdaq', 's&p', 'nifty', 'sensex',
    'bse', 'nse', 'nyse', 'ftse', 'dax', 'hang seng', 'nikkei',
    'russell', 'vix', 'fear', 'greed',

    # Analysis
    'analysis', 'forecast', 'predict', 'outlook', 'target price',
    'overbought', 'oversold', 'resistance', 'support', 'breakout',
    'volume', 'volatility', 'risk', 'return', 'yield', 'cap',
    'large cap', 'mid cap', 'small cap', 'blue chip', 'penny stock',

    # Macro / Economic
    'inflation', 'interest rate', 'fed ', 'rbi ', 'gdp', 'recession',
    'economic', 'financial', 'wealth', 'retire', 'sip', 'dollar',
    'rupee', 'currency', 'exchange', 'cpi', 'ppi', 'unemployment',
    'yield curve', 'quantitative easing', 'tapering', 'hawkish', 'dovish',
    'fiscal', 'monetary', 'deficit', 'surplus', 'treasury',

    # Options & Derivatives
    'call option', 'put option', 'strike price', 'expiry', 'expiration',
    'premium', 'implied volatility', 'greeks', 'delta', 'gamma', 'theta',
    'vega', 'iron condor', 'straddle', 'strangle', 'covered call',
    'protective put', 'bull spread', 'bear spread', 'butterfly',

    # Broker / Trading mechanics
    'broker', 'demat', 'swing', 'intraday', 'scalp', 'position',
    'day trad', 'algo', 'quant', 'backtest', 'strategy', 'indicator',
    'signal', 'entry', 'exit', 'stop loss', 'take profit', 'trailing stop',
    'margin', 'leverage', 'derivative', 'arbitrage',

    # Crypto
    'ethereum', 'solana', 'defi', 'nft', 'blockchain', 'altcoin',
    'stablecoin', 'mining', 'staking', 'wallet', 'token',

    # Portfolio & Risk
    'sharpe ratio', 'sortino', 'alpha', 'beta', 'correlation',
    'diversif', 'rebalance', 'asset allocation', 'risk management',
    'money', 'wealth', 'trade', 'trader', 'invest', 'capital',
    'finance', 'banking', 'loan', 'interest', 'economy',
    'compound interest', 'dollar cost averaging', 'dca',
    'tax loss harvesting', '401k', 'ira', 'roth',
    'price', 'ticker', 'symbol', 'company', 'wall street',
    'allocation', 'portfolio',

    # AI / Models
    'trained', 'model', 'lstm', 'prediction', 'ai model',

    # Compare / vs
    'compare', 'vs', 'versus', 'better', 'which is better',
    'difference between',
]

# ── Bot self-reference keywords ──
BOT_KEYWORDS = [
    'you', 'your', 'yourself', 'who are you', 'what are you', 'how do you',
    'how are you', 'what can you', 'help me', 'can you', 'tell me about you',
    'introduce', 'capabilities', 'features', 'how you work', 'how you answer',
    'what do you do', 'are you ai', 'are you bot', 'thank', 'thanks',
    'hello', 'hi', 'hey', 'good morning', 'good evening', 'good afternoon',
    'howdy', 'sup', 'what\'s up', 'greet', 'bye', 'goodbye', 'see you',
]

# ── Strict off-topic keywords ──
OFF_TOPIC_KEYWORDS = [
    'weather', 'recipe', 'cook', 'food', 'movie', 'film', 'music', 'song',
    'game', 'football', 'cricket', 'soccer', 'basketball', 'tennis',
    'homework', 'essay', 'write code', 'python code', 'javascript code',
    'html', 'css code', 'travel', 'vacation', 'hotel', 'flight booking',
    'health', 'doctor', 'medicine', 'exercise', 'yoga', 'gym',
    'dating', 'relationship', 'love', 'fashion', 'clothing',
]

# ── Patterns that ask about trained models / bot capabilities ──
TRAINED_MODELS_PATTERNS = [
    'what stocks have trained',
    'which stocks have trained',
    'trained ai model',
    'trained model',
    'trained lstm',
    'what stocks are trained',
    'which stocks are trained',
    'list trained',
    'list of trained',
    'show trained',
    'show me trained',
    'what models do you have',
    'which models',
    'available models',
    'available stocks',
    'supported stocks',
    'what stocks do you support',
    'what stocks can you analyze',
    'what tickers',
    'which tickers',
]

# ── Finance education question patterns ──
FINANCE_EDUCATION_PATTERNS = [
    'what is', 'what are', 'explain', 'define', 'how does', 'how do',
    'meaning of', 'difference between', 'types of', 'advantages of',
    'disadvantages of', 'pros and cons', 'when to use', 'how to calculate',
    'formula for', 'why is', 'importance of', 'role of', 'impact of',
    'how to invest', 'how to trade', 'best strategy', 'tips for',
    'guide to', 'introduction to', 'basics of', 'beginner',
]


class StockAgent:
    def __init__(self):
        self.rag = rag_pipeline
        self.trained_tickers = TRAINED_TICKERS

    def _is_asking_about_trained_models(self, query: str) -> bool:
        """Detect if user is asking about which stocks have trained models"""
        query_lower = query.lower().strip()
        for pattern in TRAINED_MODELS_PATTERNS:
            if pattern in query_lower:
                return True
        return False

    def _get_trained_stocks_response(self) -> ChatResponse:
        """Generate a formatted response listing all trained stocks"""
        tickers = sorted(self.trained_tickers)

        # Group by category
        us_stocks = [t for t in tickers if '.' not in t and '-' not in t]
        indian_stocks = [t for t in tickers if '.NS' in t or '.BO' in t]

        response = f"📊 **We have LSTM AI models trained for {len(tickers)} stocks!**\n\n"

        response += f"🇺🇸 **US Stocks ({len(us_stocks)}):**\n"
        for i in range(0, len(us_stocks), 8):
            chunk = us_stocks[i:i+8]
            response += "  " + " • ".join(chunk) + "\n"

        if indian_stocks:
            response += f"\n🇮🇳 **Indian Stocks ({len(indian_stocks)}):**\n"
            for i in range(0, len(indian_stocks), 5):
                chunk = indian_stocks[i:i+5]
                response += "  " + " • ".join(chunk) + "\n"

        response += (
            f"\n✨ These models are **trained daily** with real-time data and provide "
            f"AI-powered direction predictions based on MAPE, RMSE, and R² metrics.\n\n"
            f"💡 Try asking: **\"Analyze TSLA\"** or **\"What is the outlook for Nvidia?\"** "
            f"to see real AI predictions!"
        )

        return ChatResponse(
            reply=response,
            stock=None,
            data={"total_trained": len(tickers), "us_count": len(us_stocks), "india_count": len(indian_stocks)}
        )

    def _is_finance_education_question(self, query: str) -> bool:
        """Detect if user is asking a general finance/trading education question"""
        query_lower = query.lower().strip()
        # Must contain an education pattern AND a finance keyword
        has_edu = any(p in query_lower for p in FINANCE_EDUCATION_PATTERNS)
        has_finance = any(k in query_lower for k in STOCK_KEYWORDS)
        return has_edu and has_finance

    def _is_stock_related(self, query: str) -> bool:
        """Determine if query is stock-related, a bot question, or off-topic"""
        query_lower = query.lower().strip()

        # 1. Always allow bot meta-questions
        for keyword in BOT_KEYWORDS:
            if keyword in query_lower:
                return True

        # 2. Finance education questions always allowed
        if self._is_finance_education_question(query):
            return True

        # 3. Short messages (1-3 words) — allow unless clearly off-topic
        if len(query.split()) <= 3:
            for kw in OFF_TOPIC_KEYWORDS:
                if kw in query_lower:
                    return False
            return True

        # 4. Check for explicit off-topic keywords
        for kw in OFF_TOPIC_KEYWORDS:
            if kw in query_lower:
                has_stock_kw = any(sk in query_lower for sk in STOCK_KEYWORDS)
                has_company = any(c in query_lower for c in COMPANY_TICKERS)
                if has_stock_kw or has_company:
                    return True
                return False

        # 5. Check for stock/trading keywords
        for keyword in STOCK_KEYWORDS:
            if keyword in query_lower:
                return True

        # 6. Check for company names
        for company in COMPANY_TICKERS:
            if company in query_lower:
                return True

        # 7. Check for $SYMBOL pattern
        if re.search(r'\$[A-Z]{1,5}\b', query):
            return True

        # 8. Check for tickers in query (only KNOWN tickers from stocks.json)
        tickers_found = re.findall(r'\b[A-Z]{2,5}\b', query)
        valid_tickers = [t for t in tickers_found if t not in TICKER_BLACKLIST]
        known_tickers = [t for t in valid_tickers if t in self.trained_tickers]
        if known_tickers:
            return True

        # 9. Default: allow short, reject long
        if len(query.split()) > 8:
            return False
        return True

    def _extract_symbol(self, query: str) -> Optional[str]:
        """Extract stock symbol from query using company names and ticker patterns"""
        query_lower = query.lower()

        # 1. Company names first (try longest matches first to avoid partial matches)
        sorted_companies = sorted(COMPANY_TICKERS.keys(), key=len, reverse=True)
        for company in sorted_companies:
            if company in query_lower:
                ticker = COMPANY_TICKERS[company]
                logger.info(f"✅ Matched company '{company}' → {ticker}")
                return ticker

        # 2. $SYMBOL pattern
        dollar_match = re.search(r'\$([A-Z]{1,5})\b', query)
        if dollar_match:
            return dollar_match.group(1)

        # 3. Uppercase ticker patterns — prefer trained tickers
        matches = re.findall(r'\b[A-Z]{2,5}\b', query)
        valid = [m for m in matches if m not in TICKER_BLACKLIST]
        known = [m for m in valid if m in self.trained_tickers]
        if known:
            logger.info(f"✅ Matched trained ticker: {known[0]}")
            return known[0]
        if valid:
            return valid[0]

        # 4. Tickers with suffixes (RELIANCE.NS)
        suffix_matches = re.findall(r'\b[A-Z]{2,15}\.[A-Z]{1,2}\b', query.upper())
        if suffix_matches and suffix_matches[0] in self.trained_tickers:
            return suffix_matches[0]

        # 5. Crypto patterns (BTC-USD)
        crypto_matches = re.findall(r'\b[A-Z]{2,5}-USD\b', query.upper())
        if crypto_matches:
            return crypto_matches[0]

        return None

    async def chat(self, message: str, history: List[ChatMessage] = None) -> ChatResponse:
        """Main chat interface — the brain of the chatbot"""
        if history is None:
            history = []

        # Step 0: Check if asking about trained models (BEFORE symbol extraction!)
        if self._is_asking_about_trained_models(message):
            logger.info("🧠 User is asking about trained models — returning stock list")
            return self._get_trained_stocks_response()

        # Step 1: Topic filtering
        if not self._is_stock_related(message):
            return ChatResponse(
                reply=(
                    "I appreciate your curiosity! 😊 However, I'm specifically designed to help with "
                    "**stocks, trading, and financial market analysis**.\n\n"
                    "Here's what I can help you with:\n"
                    "📊 **Stock Analysis** — \"Analyze Tesla\" or \"NVDA outlook\"\n"
                    "📈 **Technical Analysis** — \"What is MACD?\" or \"Explain Bollinger Bands\"\n"
                    "💼 **Fundamentals** — \"What is P/E ratio?\" or \"Explain ROE\"\n"
                    "🤖 **AI Predictions** — \"Predict AAPL\" or \"LSTM forecast for MSFT\"\n"
                    "💰 **Market Concepts** — \"What is a bull market?\" or \"Explain options\"\n"
                    "👨‍💼 **Analyst Views** — \"Analyst consensus on Nvidia\"\n"
                    "📚 **Education** — \"How to start investing?\" or \"Best trading strategies\"\n\n"
                    "Feel free to ask me anything about stocks, trading, and finance! 📈"
                ),
                stock=None,
                data=None
            )

        # Step 2: Extract symbol if present
        symbol = self._extract_symbol(message)
        if symbol:
            logger.info(f"🎯 Agent identified symbol: {symbol}")
            has_model = symbol in self.trained_tickers
            logger.info(f"   {'✅ Trained model available' if has_model else '⚠️ No trained model'}")

        # Step 3: Route to RAG pipeline
        result = await self.rag.process_query(message, symbol=symbol, history=history)
        return result


agent = StockAgent()

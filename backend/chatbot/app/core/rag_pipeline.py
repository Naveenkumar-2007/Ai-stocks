# app/core/rag_pipeline.py
"""
Hybrid RAG Pipeline — fetches real data (price, technicals, fundamentals,
analyst recommendations, news, LSTM predictions), enriches context,
and uses LLM for professional financial synthesis.
Includes learning from user feedback via few-shot examples.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import settings
from services.finnhub_service import finnhub_service
from tools.technical_tools import TechnicalTools
from tools.prediction_tools import prediction_tools
from core.learning import get_relevant_knowledge
from models.schemas import (
    AnalysisContext, ChatResponse, ChatMessage,
    StockPrice, TechnicalIndicator, NewsItem,
    CompanyProfile, FundamentalData, AnalystRecommendation
)
from models.prompts import SYSTEM_PROMPT, ANALYSIS_PROMPT, GENERAL_FINANCE_PROMPT

logger = logging.getLogger(__name__)


class HybridRAGPipeline:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens
        )

    async def process_query(
        self, query: str, symbol: Optional[str] = None,
        history: List[ChatMessage] = None
    ) -> ChatResponse:
        """Main entry point — gather data, build context, call LLM"""

        if history is None:
            history = []

        # If a specific symbol was identified, fetch real data
        context_data = {}
        if symbol:
            context = await self._gather_context(symbol)
            context_data = self._format_context(context, symbol)

            analysis_prompt = ANALYSIS_PROMPT.format(
                symbol=symbol,
                company_info=context_data.get("company_info", "Not available"),
                price_data=context_data.get("price_data", "Not available"),
                fundamentals=context_data.get("fundamentals", "Not available"),
                technicals=context_data.get("technicals", "Not available"),
                analyst_consensus=context_data.get("analyst_consensus", "Not available"),
                news=context_data.get("news", "No recent news found"),
                prediction=context_data.get("prediction", "No AI model data available"),
                peers=context_data.get("peers", "Not available"),
                query=query
            )
            user_content = analysis_prompt
        else:
            # For general finance questions, use the dedicated prompt
            # to ensure thorough, accurate educational answers
            if self._is_finance_question(query):
                user_content = GENERAL_FINANCE_PROMPT.format(query=query)
            else:
                user_content = query

        # Build system prompt with learned knowledge (few-shot examples)
        enhanced_system = self._build_enhanced_prompt(query)

        # Build message list with history
        messages = [SystemMessage(content=enhanced_system)]

        for msg in history[-10:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))

        messages.append(HumanMessage(content=user_content))

        # Call LLM
        try:
            response = await self.llm.ainvoke(messages)
            reply_text = response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            reply_text = "I'm having trouble connecting to my analysis engine right now. Please try again in a moment! 🔄"

        return ChatResponse(
            reply=reply_text,
            stock=symbol,
            data=context_data if context_data else None
        )
    def _is_finance_question(self, query: str) -> bool:
        """Detect if this is a general finance/trading education question"""
        query_lower = query.lower().strip()
        
        # Question patterns that indicate educational/general queries
        question_patterns = [
            'what is', 'what are', 'what does', 'what do', 'what was',
            'how does', 'how do', 'how is', 'how to', 'how can',
            'explain', 'define', 'meaning of', 'tell me about',
            'difference between', 'compare', 'vs', 'versus',
            'types of', 'advantages', 'disadvantages', 'pros and cons',
            'why is', 'why does', 'why do', 'why are',
            'when to', 'when should', 'when is',
            'best way to', 'best strategy', 'tips for', 'guide to',
            'introduction to', 'basics of', 'beginner',
            'how to calculate', 'formula for', 'formula of',
            'importance of', 'role of', 'impact of',
            'example of', 'examples of',
            'is it good', 'is it safe', 'is it worth',
            'should i', 'can i', 'do i need',
        ]
        
        # Finance topic keywords
        finance_keywords = [
            'stock', 'share', 'market', 'trading', 'invest', 'portfolio',
            'dividend', 'bond', 'mutual fund', 'etf', 'option', 'call', 'put',
            'p/e', 'pe ratio', 'eps', 'roe', 'roa', 'rsi', 'macd', 'sma',
            'ema', 'bollinger', 'fibonacci', 'candlestick', 'vwap', 'adx',
            'stochastic', 'moving average', 'golden cross', 'death cross',
            'bull', 'bear', 'inflation', 'interest rate', 'fed', 'gdp',
            'recession', 'yield', 'yield curve', 'treasury',
            'day trading', 'swing trading', 'scalping', 'momentum',
            'value investing', 'growth investing', 'dca', 'dollar cost',
            'compound interest', 'asset allocation', 'diversif',
            'risk', 'hedge', 'leverage', 'margin', 'short selling',
            'ipo', 'spac', 'earnings', 'revenue', 'profit', 'loss',
            'balance sheet', 'income statement', 'cash flow',
            'free cash flow', 'dcf', 'valuation', 'intrinsic value',
            'book value', 'market cap', 'beta', 'alpha', 'sharpe',
            'crypto', 'bitcoin', 'ethereum', 'blockchain', 'defi',
            'nft', 'stablecoin', 'staking', 'mining',
            '401k', '401(k)', 'ira', 'roth', 'retirement', 'sip',
            'tax loss', 'emergency fund', 'financial', 'finance',
            'support', 'resistance', 'breakout', 'volume',
            'volatility', 'overbought', 'oversold',
            'sector', 'index', 'nasdaq', 'dow', 's&p', 'nifty', 'sensex',
            'forex', 'currency', 'exchange rate',
            'iron condor', 'straddle', 'covered call', 'greeks',
            'delta', 'gamma', 'theta', 'vega', 'implied volatility',
            'technical analysis', 'fundamental analysis',
            'price', 'ticker', 'broker', 'demat',
            'capital gain', 'depreciation', 'amortization',
            'p/b', 'p/s', 'debt to equity', 'current ratio',
            'money', 'wealth', 'saving', 'budget',
            'chart pattern', 'head and shoulders', 'cup and handle',
            'stop loss', 'take profit', 'trailing stop',
            'quantitative', 'algorithmic', 'backtest',
            'penny stock', 'blue chip', 'large cap', 'small cap',
        ]
        
        has_question = any(p in query_lower for p in question_patterns)
        has_finance = any(k in query_lower for k in finance_keywords)
        
        # Also check if it's short and looks like a concept query
        # e.g. "P/E ratio", "RSI", "Bollinger Bands"
        concept_only = len(query.split()) <= 4 and has_finance
        
        return (has_question and has_finance) or concept_only

    def _build_enhanced_prompt(self, query: str) -> str:
        """Enhance system prompt with learned knowledge from user feedback"""
        base_prompt = SYSTEM_PROMPT

        # Get relevant learned examples
        examples = get_relevant_knowledge(query, max_examples=3)

        if examples:
            examples_text = "\n\nLEARNED FROM PAST INTERACTIONS (use these as reference for similar questions):\n"
            for i, ex in enumerate(examples, 1):
                examples_text += f"\nExample {i}:\n"
                examples_text += f"  User asked: \"{ex['question']}\"\n"
                examples_text += f"  Good answer: \"{ex['answer'][:300]}\"\n"

            return base_prompt + examples_text

        return base_prompt

    async def _gather_context(self, symbol: str) -> AnalysisContext:
        """Fetch ALL data sources in parallel — price, technicals, fundamentals,
        analyst recommendations, news, peers, and LSTM predictions."""

        # Launch all 8 data fetches in parallel with a strict 15s timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    finnhub_service.get_quote(symbol),
                    TechnicalTools.get_indicators(symbol),
                    finnhub_service.get_company_news(symbol),
                    prediction_tools.get_latest_prediction(symbol),
                    finnhub_service.get_company_profile(symbol),
                    finnhub_service.get_basic_financials(symbol),
                    finnhub_service.get_recommendation_trends(symbol),
                    finnhub_service.get_peers(symbol),
                    return_exceptions=True
                ),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"🕒 RAG data gathering timed out for {symbol} after 15s. Using partial data.")
            # If it times out, we return an empty context or whatever was gathered (though gather doesn't partial return on timeout)
            # Actually wait_for cancels the gather Task. So we get nothing.
            # To handle partials, we would need more complex logic, but 15s is generous.
            results = [None] * 8

        quote_raw = results[0] if not isinstance(results[0], Exception) else None
        tech_indicators = results[1] if not isinstance(results[1], Exception) else []
        news_raw = results[2] if not isinstance(results[2], Exception) else []
        prediction = results[3] if not isinstance(results[3], Exception) else None
        profile_raw = results[4] if not isinstance(results[4], Exception) else None
        financials_raw = results[5] if not isinstance(results[5], Exception) else None
        recs_raw = results[6] if not isinstance(results[6], Exception) else []
        peers_raw = results[7] if not isinstance(results[7], Exception) else []

        # ── Parse stock price ──
        stock_data = None
        if quote_raw and quote_raw.get('c', 0) > 0:
            stock_data = StockPrice(
                symbol=symbol,
                current_price=quote_raw['c'],
                change=quote_raw.get('d', 0) or 0,
                percent_change=quote_raw.get('dp', 0) or 0,
                high=quote_raw.get('h', 0) or 0,
                low=quote_raw.get('l', 0) or 0,
                open=quote_raw.get('o', 0) or 0,
                previous_close=quote_raw.get('pc', 0) or 0,
                timestamp=datetime.now()
            )

        # ── Parse news ──
        news_items = []
        if isinstance(news_raw, list):
            for n in news_raw[:5]:
                try:
                    news_items.append(NewsItem(
                        headline=n.get('headline', ''),
                        source=n.get('source', ''),
                        url=n.get('url', ''),
                        summary=n.get('summary', '')[:200],
                        sentiment="neutral",
                        sentiment_score=0.0,
                        datetime=datetime.fromtimestamp(n.get('datetime', 0))
                    ))
                except Exception:
                    continue

        # ── Parse company profile ──
        company_profile = None
        if profile_raw:
            try:
                company_profile = CompanyProfile(
                    name=profile_raw.get('name', ''),
                    ticker=profile_raw.get('ticker', symbol),
                    country=profile_raw.get('country', ''),
                    currency=profile_raw.get('currency', ''),
                    exchange=profile_raw.get('exchange', ''),
                    sector=profile_raw.get('finnhubIndustry', ''),
                    industry=profile_raw.get('finnhubIndustry', ''),
                    market_cap=profile_raw.get('marketCapitalization', 0),
                    ipo_date=profile_raw.get('ipo', ''),
                    logo_url=profile_raw.get('logo', ''),
                    website=profile_raw.get('weburl', ''),
                    phone=profile_raw.get('phone', ''),
                )
            except Exception as e:
                logger.error(f"Error parsing company profile: {e}")

        # ── Parse fundamental data ──
        fundamentals = None
        if financials_raw:
            try:
                fundamentals = FundamentalData(
                    pe_ratio=financials_raw.get('peNormalizedAnnual'),
                    pb_ratio=financials_raw.get('pbAnnual'),
                    ps_ratio=financials_raw.get('psAnnual'),
                    eps_ttm=financials_raw.get('epsTTM'),
                    dividend_yield=financials_raw.get('dividendYieldIndicatedAnnual'),
                    beta=financials_raw.get('beta'),
                    week_52_high=financials_raw.get('52WeekHigh'),
                    week_52_low=financials_raw.get('52WeekLow'),
                    week_52_high_date=financials_raw.get('52WeekHighDate', ''),
                    week_52_low_date=financials_raw.get('52WeekLowDate', ''),
                    avg_volume_10d=financials_raw.get('10DayAverageTradingVolume'),
                    revenue_per_share_ttm=financials_raw.get('revenuePerShareTTM'),
                    roe=financials_raw.get('roeTTM'),
                    current_ratio=financials_raw.get('currentRatioQuarterly'),
                    debt_to_equity=financials_raw.get('totalDebt/totalEquityQuarterly'),
                )
            except Exception as e:
                logger.error(f"Error parsing fundamental data: {e}")

        # ── Parse analyst recommendations ──
        recommendations = []
        if isinstance(recs_raw, list):
            for rec in recs_raw[:3]:
                try:
                    recommendations.append(AnalystRecommendation(
                        period=rec.get('period', ''),
                        strong_buy=rec.get('strongBuy', 0),
                        buy=rec.get('buy', 0),
                        hold=rec.get('hold', 0),
                        sell=rec.get('sell', 0),
                        strong_sell=rec.get('strongSell', 0),
                    ))
                except Exception:
                    continue

        return AnalysisContext(
            stock_data=stock_data,
            technical_indicators=tech_indicators if isinstance(tech_indicators, list) else [],
            news_items=news_items,
            prediction=prediction,
            company_profile=company_profile,
            fundamentals=fundamentals,
            recommendations=recommendations,
            peers=peers_raw if isinstance(peers_raw, list) else [],
        )

    def _format_context(self, context: AnalysisContext, symbol: str) -> Dict[str, Any]:
        """Format all context data into readable strings for the LLM prompt"""
        result = {}

        # ── Company Info ──
        if context.company_profile:
            cp = context.company_profile
            mkt_cap_str = f"${cp.market_cap / 1000:.1f}B" if cp.market_cap > 1000 else f"${cp.market_cap:.0f}M"
            result["company_info"] = (
                f"Name: {cp.name}\n"
                f"  Sector: {cp.sector} | Exchange: {cp.exchange}\n"
                f"  Country: {cp.country} | Currency: {cp.currency}\n"
                f"  Market Cap: {mkt_cap_str}\n"
                f"  IPO Date: {cp.ipo_date}\n"
                f"  Website: {cp.website}"
            )
        else:
            result["company_info"] = f"Company profile not available for {symbol}."

        # ── Price Data ──
        if context.stock_data:
            sd = context.stock_data
            direction = "▲" if sd.change >= 0 else "▼"
            result["price_data"] = (
                f"Current: ${sd.current_price:.2f} ({direction} {abs(sd.percent_change):.2f}%)\n"
                f"  Open: ${sd.open:.2f} | High: ${sd.high:.2f} | Low: ${sd.low:.2f}\n"
                f"  Previous Close: ${sd.previous_close:.2f}"
            )
        else:
            result["price_data"] = f"Live price data not available for {symbol}."

        # ── Fundamentals ──
        if context.fundamentals:
            fd = context.fundamentals
            lines = []
            if fd.pe_ratio is not None:
                pe_label = "Growth/Expensive" if fd.pe_ratio > 30 else ("Moderate" if fd.pe_ratio > 15 else "Value/Undervalued")
                lines.append(f"• P/E Ratio: {fd.pe_ratio:.2f} ({pe_label})")
            if fd.pb_ratio is not None:
                lines.append(f"• P/B Ratio: {fd.pb_ratio:.2f}")
            if fd.ps_ratio is not None:
                lines.append(f"• P/S Ratio: {fd.ps_ratio:.2f}")
            if fd.eps_ttm is not None:
                lines.append(f"• EPS (TTM): ${fd.eps_ttm:.2f}")
            if fd.roe is not None:
                roe_label = "Strong" if fd.roe > 15 else ("Moderate" if fd.roe > 8 else "Weak")
                lines.append(f"• ROE: {fd.roe:.1f}% ({roe_label} profitability)")
            if fd.dividend_yield is not None:
                lines.append(f"• Dividend Yield: {fd.dividend_yield:.2f}%")
            if fd.beta is not None:
                beta_label = "High Volatility" if fd.beta > 1.5 else ("Market Average" if fd.beta > 0.8 else "Defensive/Low Vol")
                lines.append(f"• Beta: {fd.beta:.2f} ({beta_label})")
            if fd.week_52_high is not None and fd.week_52_low is not None:
                lines.append(f"• 52-Week Range: ${fd.week_52_low:.2f} — ${fd.week_52_high:.2f}")
            if fd.current_ratio is not None:
                health = "Healthy" if fd.current_ratio > 1.5 else ("Adequate" if fd.current_ratio > 1.0 else "⚠️ Liquidity Concern")
                lines.append(f"• Current Ratio: {fd.current_ratio:.2f} ({health})")
            if fd.debt_to_equity is not None:
                debt_label = "Conservative" if fd.debt_to_equity < 0.5 else ("Moderate" if fd.debt_to_equity < 1.5 else "⚠️ Highly Leveraged")
                lines.append(f"• Debt/Equity: {fd.debt_to_equity:.2f} ({debt_label})")
            if fd.avg_volume_10d is not None:
                vol_str = f"{fd.avg_volume_10d:.1f}M" if fd.avg_volume_10d >= 1 else f"{fd.avg_volume_10d * 1000:.0f}K"
                lines.append(f"• Avg Volume (10D): {vol_str}")

            result["fundamentals"] = "\n".join(lines) if lines else "Fundamental data not available."
        else:
            result["fundamentals"] = f"Fundamental metrics not available for {symbol}."

        # ── Technicals ──
        if context.technical_indicators:
            tech_lines = []
            for ind in context.technical_indicators:
                tech_lines.append(f"• {ind.name}: {ind.value} — Signal: {ind.signal} ({ind.description})")
            result["technicals"] = "\n".join(tech_lines)
        else:
            result["technicals"] = "Technical indicator data not available."

        # ── Analyst Consensus ──
        if context.recommendations:
            rec_lines = []
            for rec in context.recommendations:
                total = rec.strong_buy + rec.buy + rec.hold + rec.sell + rec.strong_sell
                if total > 0:
                    bullish_pct = ((rec.strong_buy + rec.buy) / total) * 100
                    rec_lines.append(
                        f"• {rec.period}: Strong Buy: {rec.strong_buy} | Buy: {rec.buy} | "
                        f"Hold: {rec.hold} | Sell: {rec.sell} | Strong Sell: {rec.strong_sell} "
                        f"({bullish_pct:.0f}% bullish, {total} analysts)"
                    )
            result["analyst_consensus"] = "\n".join(rec_lines) if rec_lines else "No analyst data available."
        else:
            result["analyst_consensus"] = f"Analyst recommendation data not available for {symbol}."

        # ── News ──
        if context.news_items:
            news_lines = []
            for n in context.news_items:
                news_lines.append(f"• [{n.source}] {n.headline}")
                if n.summary:
                    news_lines.append(f"  Summary: {n.summary}")
            result["news"] = "\n".join(news_lines)
        else:
            result["news"] = "No recent news articles found."

        # ── AI Prediction ──
        if context.prediction:
            p = context.prediction
            result["prediction"] = (
                f"Direction: {p.predicted_direction}\n"
                f"  Confidence: {p.confidence * 100:.1f}%\n"
                f"  Key Factors: {', '.join(p.key_factors)}"
            )
        else:
            result["prediction"] = f"No trained LSTM model available for {symbol}."

        # ── Peers ──
        if context.peers:
            result["peers"] = ", ".join(context.peers)
        else:
            result["peers"] = "Peer data not available."

        return result


rag_pipeline = HybridRAGPipeline()

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { TrendingUp, TrendingDown, Search, DollarSign, BarChart3, Activity, Calendar, Image } from 'lucide-react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, ComposedChart, ReferenceLine, Cell
} from 'recharts';
import { useAuth } from '../contexts/AuthContext';

const API_BASE = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000');

// Helper function to track user stats
const trackUserStats = (userEmail, ticker) => {
  if (!userEmail) return;

  const statsKey = `userStats_${userEmail}`;
  const stats = JSON.parse(localStorage.getItem(statsKey)) || {
    predictions: 0,
    stocksTracked: [],
    correctPredictions: 0
  };

  // Increment predictions
  stats.predictions += 1;

  // Add stock to tracked list if not already there
  if (!stats.stocksTracked.includes(ticker)) {
    stats.stocksTracked.push(ticker);
  }

  // For demo purposes, randomly mark some predictions as correct
  if (Math.random() > 0.3) {
    stats.correctPredictions += 1;
  }

  localStorage.setItem(statsKey, JSON.stringify(stats));
};

// Helper function to get stock logo
const getStockLogo = (symbol) => {
  return `https://logo.clearbit.com/${symbol.toLowerCase()}.com`;
};

// Candlestick component for traditional stock charts
const CandlestickShape = ({ fill, x, y, width, height, low, high, open, close }) => {
  const isGrowing = close > open;
  const color = isGrowing ? '#10b981' : '#ef4444';
  const bodyHeight = Math.abs(close - open);
  const bodyY = isGrowing ? y + height - (close - low) : y + height - (open - low);

  return (
    <g>
      {/* High-Low wick */}
      <line
        x1={x + width / 2}
        y1={y + height - (high - low)}
        x2={x + width / 2}
        y2={y + height}
        stroke={color}
        strokeWidth={Math.max(1, width * 0.1)}
      />
      {/* Open-Close body */}
      <rect
        x={x + width * 0.2}
        y={bodyY}
        width={width * 0.6}
        height={Math.max(1, bodyHeight)}
        fill={color}
        stroke={color}
      />
    </g>
  );
};

// Custom Tooltip Component with better mobile display
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white dark:bg-dark-card p-2 sm:p-3 border border-gray-200 dark:border-dark-border rounded-lg shadow-xl text-xs sm:text-sm max-w-[200px]">
        <p className="font-semibold text-gray-900 dark:text-white mb-1 truncate">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="font-medium truncate" style={{ color: entry.color }}>
            {entry.name}: ${typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// Format time ago for news
const formatTimeAgo = (timestamp) => {
  const now = Date.now() / 1000; // Current time in seconds
  const diff = now - timestamp;

  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(timestamp * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

// Custom Candlestick Tooltip
const CandlestickTooltip = ({ active, payload }) => {
  if (active && payload && payload.length > 0) {
    const data = payload[0].payload;
    return (
      <div className="bg-white dark:bg-dark-card p-2 sm:p-3 border border-gray-200 dark:border-dark-border rounded-lg shadow-lg text-xs sm:text-sm">
        <p className="font-medium text-gray-600 dark:text-gray-400 mb-1">
          {new Date(data.date).toLocaleDateString()}
        </p>
        <p className="text-green-600 dark:text-green-400">Open: ${data.open?.toFixed(2)}</p>
        <p className="text-blue-600 dark:text-blue-400">High: ${data.high?.toFixed(2)}</p>
        <p className="text-orange-600 dark:text-orange-400">Low: ${data.low?.toFixed(2)}</p>
        <p className="text-red-600 dark:text-red-400">Close: ${data.close?.toFixed(2)}</p>
        <p className="text-gray-600 dark:text-gray-400">Vol: {formatVolume(data.volume)}</p>
      </div>
    );
  }
  return null;
};

// Sentiment Gauge Component
const SentimentGauge = ({ sentiment }) => {
  if (!sentiment) return null;

  const score = sentiment.score || 0;
  const bull = sentiment.bullish_percent || 0;
  const bear = sentiment.bearish_percent || 0;
  const buzzCount = sentiment.buzz_articles || 0;

  // Insufficient only if we literally have no data at all (both are zero AND no articles)
  const isInsufficient = (bull === 0 && bear === 0 && buzzCount === 0);

  const getColor = () => {
    if (isInsufficient) return '#94a3b8';
    if (score > 0.3) return '#10b981';
    if (score > 0) return '#6ee7b7';
    if (score < -0.3) return '#ef4444';
    if (score < 0) return '#fca5a5';
    return '#94a3b8';
  };

  const getSentimentText = () => {
    if (isInsufficient) return 'Insufficient Data';
    if (score > 0.6) return 'Strongly Bullish';
    if (score > 0.2) return 'Bullish';
    if (score > 0) return 'Slightly Bullish';
    if (score > -0.2) return 'Neutral';
    if (score > -0.6) return 'Bearish';
    return 'Strongly Bearish';
  };

  const percentage = isInsufficient ? 50 : ((score + 1) / 2) * 100;

  return (
    <div className="flex flex-col items-center justify-center py-3 sm:py-4">
      <div className="relative sentiment-gauge" style={{ width: '140px', height: '140px' }}>
        <svg className="transform -rotate-90 w-full h-full">
          <circle cx="70" cy="70" r="60" stroke="#e5e7eb" strokeWidth="14" fill="none" className="dark:stroke-gray-700" />
          <circle
            cx="70" cy="70" r="60"
            stroke={getColor()}
            strokeWidth="14"
            fill="none"
            strokeDasharray={`${percentage * 3.77} 377`}
            strokeLinecap="round"
            style={{ transition: 'stroke-dasharray 1s ease-in-out' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-2">
          <span className={`font-bold leading-tight ${isInsufficient ? 'text-sm sm:text-base' : 'text-xl sm:text-2xl'}`} style={{ color: getColor() }}>
            {getSentimentText()}
          </span>
          {!isInsufficient && (
            <span className="text-xs sm:text-sm text-gray-500 dark:text-gray-400 mt-1">
              Score: {score.toFixed(2)}
            </span>
          )}
        </div>
      </div>
      {!isInsufficient && bull > 0 && bear > 0 && (
        <div className="w-full mt-2 px-2 text-center">
          <span className="text-xs text-green-600 dark:text-green-400 font-medium">Bull {bull.toFixed(0)}%</span>
          <span className="text-xs text-gray-400 mx-2">|</span>
          <span className="text-xs text-red-500 dark:text-red-400 font-medium">Bear {bear.toFixed(0)}%</span>
        </div>
      )}
      <div className="w-full mt-3 px-2">
        <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
          <span>Bearish</span>
          <span>Neutral</span>
          <span>Bullish</span>
        </div>
        <div className="h-2 bg-gradient-to-r from-red-500 via-gray-400 to-green-500 rounded-full"></div>
      </div>
    </div>
  );
};

const formatCurrencyCompact = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return 'N/A';
  }

  const absValue = Math.abs(numeric);
  if (absValue >= 1e12) {
    return `$${(numeric / 1e12).toFixed(2)}T`;
  }
  if (absValue >= 1e9) {
    return `$${(numeric / 1e9).toFixed(2)}B`;
  }
  if (absValue >= 1e6) {
    return `$${(numeric / 1e6).toFixed(2)}M`;
  }
  if (absValue >= 1e3) {
    return `$${(numeric / 1e3).toFixed(2)}K`;
  }
  return `$${numeric.toFixed(2)}`;
};

const formatVolume = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return 'N/A';
  }

  if (numeric >= 1e9) {
    return `${(numeric / 1e9).toFixed(2)}B`;
  }
  if (numeric >= 1e6) {
    return `${(numeric / 1e6).toFixed(2)}M`;
  }
  if (numeric >= 1e3) {
    return `${(numeric / 1e3).toFixed(2)}K`;
  }
  return numeric.toLocaleString();
};

const parseDateLocal = (dateStr) => {
  if (!dateStr) return new Date();
  const [year, month, day] = dateStr.split('-').map(Number);
  if (isNaN(year) || isNaN(month) || isNaN(day)) return new Date(dateStr);
  return new Date(year, month - 1, day);
};

function Prediction() {
  const { currentUser } = useAuth();

  // Persistence Helpers
  const getSaved = (key, fallback) => {
    try {
      const saved = localStorage.getItem(key);
      return saved ? JSON.parse(saved) : fallback;
    } catch (e) {
      return fallback;
    }
  };

  const [ticker, setTicker] = useState(() => localStorage.getItem('prediction_ticker') || '');
  const [days, setDays] = useState(() => Number(localStorage.getItem('prediction_days')) || 7);
  const [stockData, setStockData] = useState(() => getSaved('prediction_stock_data', null));
  const [sentiment, setSentiment] = useState(() => getSaved('prediction_sentiment', null));
  const [news, setNews] = useState(() => getSaved('prediction_news', []));
  const [trainingStatus, setTrainingStatus] = useState(() => getSaved('prediction_training_status', null));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [performancePeriod, setPerformancePeriod] = useState('1M');
  const [hasSearched, setHasSearched] = useState(() => localStorage.getItem('prediction_has_searched') === 'true');
  const [visibleNewsCount, setVisibleNewsCount] = useState(5);

  // Save state on changes
  useEffect(() => {
    localStorage.setItem('prediction_ticker', ticker);
    localStorage.setItem('prediction_days', days.toString());
    localStorage.setItem('prediction_has_searched', hasSearched.toString());
    if (stockData) localStorage.setItem('prediction_stock_data', JSON.stringify(stockData));
    if (sentiment) localStorage.setItem('prediction_sentiment', JSON.stringify(sentiment));
    if (news) localStorage.setItem('prediction_news', JSON.stringify(news));
    if (trainingStatus) localStorage.setItem('prediction_training_status', JSON.stringify(trainingStatus));
  }, [ticker, days, stockData, sentiment, news, trainingStatus, hasSearched]);

  // Filter performance data based on selected period
  const getPerformanceData = () => {
    if (!stockData?.performance_chart?.dates || !stockData?.performance_chart?.prices) {
      return { dates: [], prices: [] };
    }

    const dates = stockData.performance_chart.dates;
    const prices = stockData.performance_chart.prices;
    const now = new Date();
    let startDate;

    switch (performancePeriod) {
      case '1W':
        startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case '1M':
        startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        break;
      case '3M':
        startDate = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);
        break;
      case '6M':
        startDate = new Date(now.getTime() - 180 * 24 * 60 * 60 * 1000);
        break;
      case '1Y':
        startDate = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000);
        break;
      case 'ALL':
      default:
        return { dates, prices };
    }

    const filteredData = dates.reduce((acc, date, index) => {
      const dateObj = new Date(date);
      if (dateObj >= startDate) {
        acc.dates.push(date);
        acc.prices.push(prices[index]);
      }
      return acc;
    }, { dates: [], prices: [] });

    return filteredData.dates.length > 0 ? filteredData : { dates, prices };
  };

  const hideSuggestionsTimeoutRef = useRef(null);
  const searchDebounceRef = useRef(null);

  // No auto-fetch — user must search a stock explicitly

  useEffect(() => {
    return () => {
      if (hideSuggestionsTimeoutRef.current) {
        clearTimeout(hideSuggestionsTimeoutRef.current);
        hideSuggestionsTimeoutRef.current = null;
      }
      if (searchDebounceRef.current) {
        clearTimeout(searchDebounceRef.current);
        searchDebounceRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const activeTicker = stockData?.ticker || ticker;
    const status = trainingStatus || stockData?.model_status;
    const displayReady = Boolean(
      status?.model_ready &&
      stockData?.prediction_ready &&
      Array.isArray(stockData?.future_predictions) &&
      stockData.future_predictions.length > 0
    );
    const shouldPoll = Boolean(
      activeTicker &&
      stockData &&
      status &&
      !displayReady &&
      status.state !== 'failed'
    );

    if (!shouldPoll) return undefined;

    const pollStatus = async () => {
      try {
        const { data } = await axios.get(`${API_BASE}/api/model-status/${activeTicker}`, { timeout: 10000 });
        const nextStatus = data?.model_status;
        if (!nextStatus) return;
        setTrainingStatus(nextStatus);

        if (nextStatus.model_ready) {
          await fetchStockData(activeTicker, { trackStats: false, silent: true });
        }
      } catch (err) {
        console.warn('Model status polling failed:', err);
      }
    };

    const interval = setInterval(pollStatus, 10000);
    return () => clearInterval(interval);
  }, [stockData, trainingStatus, ticker]);

  const handleTickerInput = (value) => {
    if (hideSuggestionsTimeoutRef.current) {
      clearTimeout(hideSuggestionsTimeoutRef.current);
      hideSuggestionsTimeoutRef.current = null;
    }
    const formatted = value.toUpperCase();
    setTicker(formatted);

    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current);
      searchDebounceRef.current = null;
    }

    const query = formatted.trim();
    if (!query) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    searchDebounceRef.current = setTimeout(async () => {
      try {
        const { data } = await axios.get(`${API_BASE}/api/search`, {
          params: { q: query, limit: 5 }
        });
        const results = Array.isArray(data?.results) ? data.results : [];
        setSuggestions(results);
        setShowSuggestions(results.length > 0);
      } catch (err) {
        console.error('Ticker lookup failed:', err);
        setSuggestions([]);
        setShowSuggestions(false);
      } finally {
        searchDebounceRef.current = null;
      }
    }, 250);
  };

  const handleSuggestionSelect = (symbol) => {
    setTicker(symbol);
    setShowSuggestions(false);
    setSuggestions([]);
    if (hideSuggestionsTimeoutRef.current) {
      clearTimeout(hideSuggestionsTimeoutRef.current);
      hideSuggestionsTimeoutRef.current = null;
    }
    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current);
      searchDebounceRef.current = null;
    }
    fetchStockData(symbol);
  };

  const handleInputFocus = () => {
    if (hideSuggestionsTimeoutRef.current) {
      clearTimeout(hideSuggestionsTimeoutRef.current);
      hideSuggestionsTimeoutRef.current = null;
    }
    if (suggestions.length > 0) {
      setShowSuggestions(true);
    }
  };

  const handleInputBlur = () => {
    if (hideSuggestionsTimeoutRef.current) {
      clearTimeout(hideSuggestionsTimeoutRef.current);
    }
    hideSuggestionsTimeoutRef.current = setTimeout(() => {
      setShowSuggestions(false);
      hideSuggestionsTimeoutRef.current = null;
    }, 150);
  };

  const fetchStockData = async (symbol, options = {}) => {
    const { trackStats = true, silent = false } = options;
    if (!silent) setLoading(true);
    setError(null);
    const headers = {};
    if (currentUser) {
      try {
        const token = await currentUser.getIdToken();
        headers.Authorization = `Bearer ${token}`;
      } catch (err) {
        console.error('Failed to get token for prediction call', err);
      }
    }

    try {
      const [stockRes, sentimentRes, newsRes] = await Promise.all([
        axios.get(`${API_BASE}/api/stock/${symbol}?days=${days}`, { headers, timeout: 120000 }),
        axios.get(`${API_BASE}/api/sentiment/${symbol}`, { timeout: 15000 }).catch(() => ({ data: { sentiment: null } })),
        axios.get(`${API_BASE}/api/news/${symbol}?days=7`, { timeout: 15000 }).catch(() => ({ data: { news: [] } }))
      ]);

      const payload = stockRes.data;

      // Debug: Log the candles data structure
      console.log('📊 Stock Data Received:', {
        ticker: payload.ticker,
        candlesCount: payload.technical_chart?.candles?.length,
        firstCandle: payload.technical_chart?.candles?.[0],
        lastCandle: payload.technical_chart?.candles?.[payload.technical_chart?.candles?.length - 1]
      });

      setStockData(payload);
      setTrainingStatus(payload?.model_status || null);
      if (
        payload?.ticker &&
        payload?.requested_ticker &&
        payload.ticker !== payload.requested_ticker
      ) {
        setTicker(payload.ticker);
      }
      // Use stockData.sentiment (embedded in stock API response) as primary source,
      // fall back to separate /api/sentiment call
      const embeddedSentiment = payload?.sentiment;
      const separateSentiment = sentimentRes?.data?.sentiment;
      setSentiment(
        (embeddedSentiment && Object.keys(embeddedSentiment).length > 0)
          ? embeddedSentiment
          : separateSentiment || null
      );
      setNews(newsRes.data.news || []);

      // Track user stats
      if (trackStats && currentUser?.email) {
        trackUserStats(currentUser.email, payload?.ticker || symbol);
      }
    } catch (err) {
      const backendMessage = err.response?.data?.error;
      const timeoutMessage = err.code === 'ECONNABORTED'
        ? 'The market-data request timed out. This can happen on a first-time stock while the backend fetches history and starts model training. Please try again in a few seconds.'
        : null;
      const offlineMessage = err.request && !err.response
        ? 'Backend API is not reachable. Start the backend on port 8000, then search again.'
        : null;
      setError(backendMessage || timeoutMessage || offlineMessage || 'Failed to fetch data');
      console.error(err);
    } finally {
      if (!silent) setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    const query = ticker.trim().toUpperCase();
    if (!query) {
      return;
    }

    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current);
      searchDebounceRef.current = null;
    }
    if (hideSuggestionsTimeoutRef.current) {
      clearTimeout(hideSuggestionsTimeoutRef.current);
      hideSuggestionsTimeoutRef.current = null;
    }

    setTicker(query);
    setHasSearched(true); // Mark that user has searched
    fetchStockData(query);
    setSuggestions([]);
    setShowSuggestions(false);
    setVisibleNewsCount(5); // Reset news count on new search
  };

  const handleLoadMoreNews = () => {
    setVisibleNewsCount(prev => prev + 5);
  };

  const computePredictionRows = () => {
    if (!stockData?.future_predictions || !Array.isArray(stockData.future_predictions)) {
      return [];
    }

    return stockData.future_predictions.map((entry, index) => {
      const price = Number(entry?.price ?? NaN);
      const baseline = Number(stockData.current_price ?? NaN);
      if (!Number.isFinite(price) || !Number.isFinite(baseline) || baseline <= 0) {
        return null;
      }

      const change = price - baseline;
      const changePercent = (change / baseline) * 100;

      const minMove = Number(entry?.min_required_move_pct ?? stockData?.recommendation?.min_required_move_pct ?? 0.35);
      const confidence = Number(entry?.confidence ?? stockData?.recommendation?.confidence ?? 0);
      let signal = entry?.signal || 'HOLD';

      // Conservative fallback for legacy API payloads that do not include
      // backend row signals. Never turn tiny moves into BUY/SELL in the UI.
      if (!entry?.signal) {
        const finalSignal = stockData?.recommendation?.signal || stockData?.ai_signal || 'HOLD';
        const finalScore = Number(stockData?.recommendation?.score ?? 0);
        if (confidence >= 0.38 && Math.abs(changePercent) >= minMove) {
          if (changePercent > 0 && finalScore > 0 && finalSignal.includes('BUY')) {
            signal = confidence >= 0.7 && changePercent >= minMove * 2.5 ? 'STRONG BUY' : 'BUY';
          } else if (changePercent < 0 && finalScore < 0 && finalSignal.includes('SELL')) {
            signal = confidence >= 0.7 && Math.abs(changePercent) >= minMove * 2.5 ? 'STRONG SELL' : 'SELL';
          }
        }
      }

      return {
        id: `${entry.date}-${index}`,
        dateLabel: new Date(entry.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
        price: price.toFixed(2),
        change: change.toFixed(2),
        changePercent: changePercent.toFixed(2),
        signal
      };
    }).filter(Boolean);
  };

  const predictionRows = computePredictionRows();
  const finalRecommendation = stockData?.recommendation || {};
  const headlineSignal = finalRecommendation.signal || stockData?.ai_signal || 'HOLD';
  const headlineStance = finalRecommendation.stance || headlineSignal;
  const signalIsBullish = headlineSignal.includes('BUY');
  const signalIsBearish = headlineSignal.includes('SELL');
  const activeModelStatus = trainingStatus || stockData?.model_status || {};
  const predictionIsReady = Boolean(
    activeModelStatus.model_ready &&
    stockData?.prediction_ready &&
    Array.isArray(stockData?.future_predictions) &&
    stockData.future_predictions.length > 0
  );
  const isPreliminaryMode = activeModelStatus.analysis_mode === 'preliminary' || !predictionIsReady;
  const trainingProgress = Number(activeModelStatus.progress ?? 0);
  const trainingStateLabel = activeModelStatus.model_ready
    ? 'Custom AI Ready'
    : activeModelStatus.state === 'training'
      ? 'Custom AI Training'
      : activeModelStatus.state === 'queued'
        ? 'Training Queued'
        : activeModelStatus.state === 'failed'
          ? 'Training Needs Retry'
          : 'Preliminary Analysis';

  // Empty state - show beautiful placeholder when no stock is selected (FIRST VISIT)
  if (!hasSearched && !loading && !stockData && !error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-cyan-50 dark:from-gray-900 dark:via-gray-900 dark:to-gray-800 py-12 sm:py-20">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Main Content */}
          <div className="text-center mb-12 relative z-10">
            <div className="mb-8">
              <div className="inline-flex items-center justify-center w-28 h-28 sm:w-32 sm:h-32 rounded-full bg-gradient-to-br from-cyan-400 to-blue-500 shadow-2xl mb-6 transform hover:scale-105 transition-transform duration-500">
                <TrendingUp className="w-12 h-12 sm:w-16 sm:h-16 text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-green-600 via-emerald-600 to-green-700 dark:from-green-400 dark:via-emerald-400 dark:to-green-500 bg-clip-text text-transparent leading-tight mb-2">
                AI Stock Predictions
              </h1>
            </div>
            <p className="text-2xl sm:text-4xl font-extrabold text-gray-800 dark:text-white mb-4 tracking-tight">
              Search Your Favorite Stock
            </p>
            <p className="text-base sm:text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto leading-relaxed">
              Get AI-powered predictions, technical analysis, sentiment insights, and real-time news for any stock
            </p>
          </div>

          {/* Search Form - Prominent */}
          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-2xl p-6 sm:p-10 border-2 border-white/50 dark:border-gray-700/50 mb-12 relative z-50">
            <form onSubmit={handleSearch} className="space-y-6">
              <div className="flex flex-col lg:flex-row gap-4">
                <div className="relative flex-grow">
                  <div className="absolute inset-y-0 left-0 pl-5 flex items-center pointer-events-none">
                    <Search className="h-6 w-6 text-gray-400" />
                  </div>
                  <input
                    type="text"
                    value={ticker}
                    onChange={(e) => handleTickerInput(e.target.value)}
                    onFocus={handleInputFocus}
                    onBlur={handleInputBlur}
                    placeholder="Enter stock (e.g., AAPL)"
                    className="w-full pl-14 pr-6 py-5 text-xl border-2 border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-gray-900/50 text-gray-900 dark:text-white rounded-2xl focus:ring-4 focus:ring-cyan-500/20 focus:border-cyan-500 transition-all shadow-inner placeholder:text-gray-400"
                    autoFocus
                  />
                  {showSuggestions && suggestions.length > 0 && (
                    <div className="absolute z-[100] mt-3 w-full bg-white dark:bg-gray-800 border-2 border-cyan-500/30 dark:border-cyan-500/40 rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.3)] dark:shadow-[0_20px_50px_rgba(0,0,0,0.6)] overflow-hidden max-h-80 overflow-y-auto ring-1 ring-black ring-opacity-5">
                      <ul className="divide-y divide-gray-100 dark:divide-gray-700">
                        {suggestions.map(({ symbol, name, exchange, country, currency }) => (
                          <li key={`${symbol}-${exchange || 'NA'}`}>
                            <button
                              type="button"
                              onMouseDown={(event) => event.preventDefault()}
                              onClick={() => handleSuggestionSelect(symbol)}
                              className="w-full px-6 py-4 flex items-start justify-between text-left hover:bg-cyan-50 dark:hover:bg-cyan-500/10 transition-all group"
                            >
                              <div className="flex flex-col">
                                <span className="font-bold text-lg text-gray-900 dark:text-white group-hover:text-cyan-600 transition-colors">{symbol}</span>
                                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{exchange || 'Market'}</span>
                              </div>
                              <div className="flex-1 ml-6 overflow-hidden">
                                <p className="text-base text-gray-600 dark:text-gray-400 truncate font-medium">{name}</p>
                                {(country || currency) && (
                                  <p className="text-sm text-gray-400 dark:text-gray-500 truncate">
                                    {[country, currency].filter(Boolean).join(' · ')}
                                  </p>
                                )}
                              </div>
                            </button>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                <div className="relative w-full lg:w-72">
                  <div className="absolute inset-y-0 left-0 pl-5 flex items-center pointer-events-none">
                    <Calendar className="h-6 w-6 text-gray-400" />
                  </div>
                  <select
                    value={days}
                    onChange={(e) => setDays(Number(e.target.value))}
                    className="w-full pl-14 pr-10 py-5 text-lg border-2 border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-gray-900/50 text-gray-900 dark:text-white rounded-2xl focus:ring-4 focus:ring-cyan-500/20 focus:border-cyan-500 shadow-inner appearance-none cursor-pointer transition-all"
                  >
                    <option value={1}>1 Day Forecast</option>
                    <option value={7}>7 Days Forecast</option>
                    <option value={14}>14 Days Forecast</option>
                    <option value={30}>30 Days Forecast</option>
                  </select>
                </div>
              </div>

              <button
                type="submit"
                disabled={!ticker.trim()}
                className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed text-white px-10 py-5 rounded-2xl flex items-center justify-center gap-3 transition-all transform hover:scale-[1.01] active:scale-95 text-xl font-bold font-display shadow-xl hover:shadow-cyan-500/30"
              >
                <Search className="w-6 h-6" />
                Predict
              </button>
            </form>
          </div>

          {/* Quick Access Stocks */}
          <div className="text-center relative z-10">
            <p className="text-sm font-bold text-gray-400 dark:text-gray-500 mb-6 uppercase tracking-[0.2em]">Popular Market Symbols</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-6">
              {['AAPL', 'TSLA', 'MSFT', 'GOOGL'].map((symbol) => (
                <button
                  key={symbol}
                  onClick={() => {
                    setTicker(symbol);
                    setHasSearched(true);
                    fetchStockData(symbol);
                  }}
                  className="bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm border-2 border-gray-100 dark:border-gray-700 hover:border-cyan-500/50 dark:hover:border-cyan-400/50 rounded-2xl p-6 transition-all transform hover:-translate-y-2 hover:shadow-2xl group relative overflow-hidden"
                >
                  <div className="absolute top-0 right-0 w-20 h-20 bg-cyan-500/5 rounded-full -mr-10 -mt-10 group-hover:scale-150 transition-transform duration-500" />
                  <p className="text-2xl sm:text-3xl font-black text-gray-800 dark:text-white group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors relative z-10">
                    {symbol}
                  </p>
                  <p className="text-xs font-bold text-gray-400 dark:text-gray-500 mt-2 relative z-10 uppercase tracking-widest">Analyze</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Loading state - show animated loading with stock symbol (AFTER SEARCH)
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-24 w-24 border-b-4 border-t-4 border-cyan-600 dark:border-cyan-500 mx-auto"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <TrendingUp className="w-10 h-10 text-cyan-600 dark:text-cyan-400 animate-pulse" />
            </div>
          </div>
          <h3 className="mt-6 text-2xl font-bold text-gray-900 dark:text-white animate-pulse">
            Predicting {ticker.toUpperCase()}
          </h3>
          <p className="mt-3 text-lg text-gray-600 dark:text-gray-400">
            Analyzing stock data and generating predictions...
          </p>
          <div className="mt-6 flex items-center justify-center gap-2">
            <div className="w-2 h-2 bg-cyan-600 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-cyan-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-cyan-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-transparent py-4 sm:py-6">
      {/* Search Header */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-4 sm:mb-6">
        <div className="bg-white dark:bg-dark-card rounded-xl sm:rounded-2xl shadow-lg p-4 sm:p-6 border border-gray-200 dark:border-dark-border">
          <form onSubmit={handleSearch} className="flex flex-col sm:flex-row flex-wrap gap-3 sm:gap-4 items-stretch sm:items-center justify-center">
            <div className="relative w-full sm:w-72">
              <input
                type="text"
                value={ticker}
                onChange={(e) => handleTickerInput(e.target.value)}
                onFocus={handleInputFocus}
                onBlur={handleInputBlur}
                placeholder="Enter ticker (e.g., AAPL)"
                className="w-full px-4 py-3 sm:px-6 sm:py-3 border-2 border-gray-300 dark:border-dark-border bg-white dark:bg-dark-elevated text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition text-base sm:text-lg touch-target"
              />
              {showSuggestions && suggestions.length > 0 && (
                <div className="absolute z-50 mt-2 w-full bg-white dark:bg-dark-card border border-cyan-500/30 dark:border-cyan-500/40 rounded-xl shadow-[0_15px_35px_rgba(0,0,0,0.2)] dark:shadow-[0_15px_35px_rgba(0,0,0,0.5)] overflow-hidden max-h-64 overflow-y-auto ring-1 ring-black ring-opacity-5">
                  <ul className="divide-y divide-gray-100 dark:divide-dark-border">
                    {suggestions.map(({ symbol, name, exchange, country, currency }) => (
                      <li key={`${symbol}-${exchange || 'NA'}`}>
                        <button
                          type="button"
                          onMouseDown={(event) => event.preventDefault()}
                          onClick={() => handleSuggestionSelect(symbol)}
                          className="w-full px-3 py-3 sm:px-4 sm:py-2 flex items-start justify-between text-left hover:bg-cyan-50 dark:hover:bg-cyan-500/10 transition-colors touch-target active-scale"
                        >
                          <span className="font-semibold text-gray-900 dark:text-white text-sm sm:text-base">{symbol}</span>
                          <div className="flex-1 ml-3 overflow-hidden">
                            <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 truncate">{name}</p>
                            {(exchange || country || currency) && (
                              <p className="text-xs text-gray-400 dark:text-gray-500 truncate">
                                {[exchange, country, currency].filter(Boolean).join(' · ')}
                              </p>
                            )}
                          </div>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="w-full sm:w-auto px-4 py-3 border-2 border-gray-300 dark:border-dark-border bg-white dark:bg-dark-elevated text-gray-900 dark:text-white rounded-lg focus:ring-2 focus:ring-cyan-500 text-base sm:text-lg touch-target"
            >
              <option value={1}>1 Day</option>
              <option value={7}>7 Days</option>
              <option value={14}>14 Days</option>
              <option value={30}>30 Days</option>
            </select>
            <button
              type="submit"
              className="w-full sm:w-auto bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white px-6 py-3 sm:px-8 sm:py-3 rounded-lg flex items-center justify-center gap-2 transition transform hover:scale-105 active:scale-95 text-base sm:text-lg font-semibold touch-target active-scale ripple shadow-lg hover:shadow-cyan-500/25"
            >
              <Search className="w-5 h-5" />
              Predict
            </button>
          </form>
        </div>
      </div>

      {error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-4 sm:mb-6">
          <div className="bg-red-50 dark:bg-red-500/10 border-l-4 border-red-500 text-red-700 dark:text-red-400 px-4 py-3 sm:px-6 sm:py-4 rounded-lg">
            <p className="font-bold text-sm sm:text-base">Error</p>
            <p className="text-xs sm:text-sm">{error}</p>
          </div>
        </div>
      )}

      {stockData && !predictionIsReady && (
        <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-10">
          <div className="min-h-[58vh] flex items-center justify-center">
            <div className="w-full rounded-2xl border border-blue-200 dark:border-blue-500/30 bg-white dark:bg-gray-900 shadow-xl p-6 sm:p-10 text-center">
              <div className="mx-auto w-14 h-14 rounded-2xl bg-blue-50 dark:bg-blue-500/10 border border-blue-200 dark:border-blue-500/30 flex items-center justify-center mb-5">
                <Activity className="w-7 h-7 text-blue-600 dark:text-blue-400 animate-pulse" />
              </div>
              <p className="text-xs font-black uppercase tracking-wide text-blue-700 dark:text-blue-400">
                Backend Training
              </p>
              <h2 className="mt-2 text-2xl sm:text-4xl font-black text-gray-900 dark:text-white">
                {stockData.ticker} model is getting ready
              </h2>
              <p className="mt-3 text-sm sm:text-base font-medium text-gray-600 dark:text-gray-400 max-w-xl mx-auto">
                Please wait about 2 minutes. The page is connected to backend training and will automatically show real predictions, charts, and signals when the model is ready.
              </p>

              <div className="mt-7 grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-2xl mx-auto">
                <div className="rounded-xl border border-cyan-200 dark:border-cyan-500/30 bg-cyan-50 dark:bg-cyan-500/10 p-5 text-left">
                  <p className="text-xs font-bold uppercase text-cyan-700 dark:text-cyan-400">Current Price</p>
                  <p className="mt-2 text-3xl font-black text-gray-900 dark:text-white">${stockData.current_price}</p>
                  <p className={`mt-1 text-sm font-bold ${stockData.day_change >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {stockData.day_change >= 0 ? '+' : ''}{stockData.day_change.toFixed(2)} ({stockData.day_change_percent.toFixed(2)}%)
                  </p>
                </div>
                <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 p-5 text-left">
                  <p className="text-xs font-bold uppercase text-gray-500 dark:text-gray-400">Status</p>
                  <p className="mt-2 text-lg font-black text-gray-900 dark:text-white">{trainingStateLabel}</p>
                  <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                    {activeModelStatus.message || 'Training is running in the background.'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </main>
      )}

      {stockData && predictionIsReady && (
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-8">
          {/* Stock Header */}
          <div className="bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl p-5 sm:p-7 mb-5 sm:mb-7 border border-gray-200 dark:border-gray-700 hover:shadow-2xl transition-shadow duration-300">
            <div className="flex items-start justify-between flex-wrap gap-3 sm:gap-4">
              <div className="flex items-start gap-3 sm:gap-4">
                <div className="relative w-12 h-12 sm:w-16 sm:h-16 rounded-xl bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-cyan-900/30 dark:to-blue-900/30 flex items-center justify-center overflow-hidden border-2 border-cyan-200 dark:border-cyan-600/30 flex-shrink-0 stock-logo-lg">
                  <img
                    src={`https://financialmodelingprep.com/image-stock/${stockData.ticker}.png`}
                    alt={stockData.ticker}
                    className="w-full h-full object-contain p-1 stock-logo"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.style.display = 'none';
                      e.target.parentElement.innerHTML = `<span class="text-2xl sm:text-3xl font-bold text-cyan-600 dark:text-cyan-400">${stockData.ticker.charAt(0)}</span>`;
                    }}
                  />
                </div>
                <div>
                  <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900 dark:text-white">{stockData.ticker}</h2>
                  <p className="text-gray-600 dark:text-gray-400 mt-1 text-sm sm:text-base lg:text-lg line-clamp-2">{stockData.company_name}</p>
                </div>
              </div>

              <div className={`flex items-center gap-2 px-4 py-2 sm:px-6 sm:py-3 rounded-xl font-bold text-sm sm:text-base lg:text-lg ${signalIsBullish
                ? 'bg-gradient-to-r from-green-100 to-green-200 dark:from-green-500/20 dark:to-green-600/20 text-green-700 dark:text-green-400 border-2 border-green-300 dark:border-green-500/30'
                : signalIsBearish
                  ? 'bg-gradient-to-r from-red-100 to-red-200 dark:from-red-500/20 dark:to-red-600/20 text-red-700 dark:text-red-400 border-2 border-red-300 dark:border-red-500/30'
                  : 'bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-500/20 dark:to-gray-600/20 text-gray-700 dark:text-gray-400 border-2 border-gray-300 dark:border-gray-500/30'
                }`}>
                {signalIsBullish ? <TrendingUp className="w-5 h-5 sm:w-6 sm:h-6" /> : signalIsBearish ? <TrendingDown className="w-5 h-5 sm:w-6 sm:h-6" /> : <Activity className="w-5 h-5 sm:w-6 sm:h-6" />}
                <span className="whitespace-nowrap">{headlineSignal}</span>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-3">
              <div className="lg:col-span-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-white/70 dark:bg-gray-900/50 p-3">
                <p className="text-xs font-bold uppercase tracking-wide text-gray-500 dark:text-gray-400">Decision State</p>
                <p className="mt-1 text-sm sm:text-base font-semibold text-gray-900 dark:text-white">{headlineStance}</p>
                {Array.isArray(finalRecommendation.reasons) && finalRecommendation.reasons.length > 0 && (
                  <p className="mt-1 text-xs sm:text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                    {finalRecommendation.reasons[0]}
                  </p>
                )}
              </div>
              <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white/70 dark:bg-gray-900/50 p-3">
                <p className="text-xs font-bold uppercase tracking-wide text-gray-500 dark:text-gray-400">Confidence</p>
                <p className="mt-1 text-xl sm:text-2xl font-black text-gray-900 dark:text-white">
                  {Number(finalRecommendation.confidence_percent ?? 0).toFixed(1)}%
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Min move: {Number(finalRecommendation.min_required_move_pct ?? 0).toFixed(2)}%
                </p>
              </div>
            </div>

            <div className={`mt-4 rounded-xl border p-4 ${activeModelStatus.model_ready
              ? 'border-green-200 dark:border-green-500/30 bg-green-50/80 dark:bg-green-500/10'
              : 'border-blue-200 dark:border-blue-500/30 bg-blue-50/80 dark:bg-blue-500/10'
              }`}>
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div>
                  <p className={`text-xs font-black uppercase tracking-wide ${activeModelStatus.model_ready ? 'text-green-700 dark:text-green-400' : 'text-blue-700 dark:text-blue-400'}`}>
                    {trainingStateLabel}
                  </p>
                  <p className="mt-1 text-sm font-semibold text-gray-900 dark:text-white">
                    {activeModelStatus.model_ready
                      ? 'Forecasts are using a trained ticker-specific model.'
                      : 'Showing real-time market analysis while the dedicated model trains in the background.'}
                  </p>
                  <p className="mt-1 text-xs sm:text-sm text-gray-600 dark:text-gray-400">
                    {activeModelStatus.message || 'The page will auto-refresh when the custom model is ready.'}
                  </p>
                </div>
                <div className="min-w-[180px]">
                  <div className="flex items-center justify-between text-xs font-bold text-gray-600 dark:text-gray-300 mb-1">
                    <span>{activeModelStatus.stage || 'model_status'}</span>
                    <span>{Math.max(0, Math.min(100, trainingProgress)).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-white dark:bg-gray-800 overflow-hidden border border-gray-200 dark:border-gray-700">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${activeModelStatus.model_ready ? 'bg-green-500' : 'bg-blue-500'}`}
                      style={{ width: `${Math.max(4, Math.min(100, trainingProgress))}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 lg:gap-6 mt-4 sm:mt-6">
              <div className="bg-gradient-to-br from-cyan-50 to-cyan-100 dark:from-cyan-500/10 dark:to-cyan-600/10 p-3 sm:p-4 rounded-xl border border-cyan-200 dark:border-cyan-500/20">
                <p className="text-xs sm:text-sm text-cyan-700 dark:text-cyan-400 font-medium flex items-center gap-1">
                  <DollarSign className="w-3 h-3 sm:w-4 sm:h-4" />
                  <span className="truncate">Current Price</span>
                </p>
                <p className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900 dark:text-white mt-1">${stockData.current_price}</p>
                <p className={`text-xs sm:text-sm mt-1 sm:mt-2 font-semibold ${stockData.day_change >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                  {stockData.day_change >= 0 ? '+' : ''}{stockData.day_change.toFixed(2)} ({stockData.day_change_percent.toFixed(2)}%)
                </p>
              </div>

              <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-500/10 dark:to-blue-600/10 p-3 sm:p-4 rounded-xl border border-blue-200 dark:border-blue-500/20">
                <p className="text-xs sm:text-sm text-blue-700 dark:text-blue-400 font-medium flex items-center gap-1">
                  <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4" />
                  <span className="truncate">Predicted Price</span>
                </p>
                <p className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900 dark:text-white mt-1">${stockData.predicted_price}</p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 sm:mt-2 truncate">Next day forecast</p>
              </div>

              <div className={`p-3 sm:p-4 rounded-xl border-2 ${stockData.is_profit ? 'bg-gradient-to-br from-green-50 to-green-100 dark:from-green-500/10 dark:to-green-600/10 border-green-300 dark:border-green-500/20' : 'bg-gradient-to-br from-red-50 to-rose-100 dark:from-red-500/10 dark:to-red-600/10 border-red-300 dark:border-red-500/20'}`}>
                <p className={`text-xs sm:text-sm font-medium flex items-center gap-1 ${stockData.is_profit ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}>
                  <Activity className="w-3 h-3 sm:w-4 sm:h-4" />
                  <span className="truncate">Expected Change</span>
                </p>
                <p className={`text-xl sm:text-2xl lg:text-3xl font-bold mt-1 ${stockData.is_profit ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}>
                  {stockData.profit_loss >= 0 ? '+' : ''}${stockData.profit_loss.toFixed(2)}
                </p>
                <p className={`text-xs sm:text-sm mt-1 sm:mt-2 font-semibold ${stockData.is_profit ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                  {stockData.profit_loss_percent >= 0 ? '+' : ''}{stockData.profit_loss_percent.toFixed(2)}%
                </p>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-500/10 dark:to-purple-600/10 p-3 sm:p-4 rounded-xl border border-purple-200 dark:border-purple-500/20">
                <p className="text-xs sm:text-sm text-purple-700 dark:text-purple-400 font-medium truncate">Volume</p>
                <p className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900 dark:text-white mt-1 truncate">{formatVolume(stockData.volume)}</p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 sm:mt-2 truncate">
                  Mkt Cap: {formatCurrencyCompact(stockData.market_cap)}
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 sm:gap-7">
            {/* Left Column - Charts */}
            <div className="lg:col-span-2 space-y-5 sm:space-y-7">
              {/* Professional Stock Price Prediction Chart */}
              <div className="bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl p-5 sm:p-7 border border-gray-200 dark:border-gray-700 hover:shadow-2xl transition-all duration-300 overflow-hidden">
                <div className="flex items-center justify-between mb-3 sm:mb-4 flex-wrap gap-2">
                  <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-600 dark:text-cyan-400" />
                    <span>Stock Price Prediction</span>
                  </h3>
                  <span className="text-xs sm:text-sm text-cyan-700 dark:text-cyan-300 bg-cyan-100 dark:bg-cyan-500/20 px-3 py-1.5 rounded-full font-semibold border border-cyan-300 dark:border-cyan-500/30">
                    {days} Days Forecast
                  </span>
                </div>
                <div className="w-full bg-gray-50 dark:bg-gray-900 rounded-lg p-2 sm:p-3 overflow-hidden" style={{ height: '320px', minHeight: '280px', maxHeight: '320px' }}>
                  <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
                    <ComposedChart data={[
                      ...stockData.technical_chart.candles.slice(-30).map((candle, idx, arr) => {
                        const isLast = idx === arr.length - 1;
                        const dateObj = parseDateLocal(candle.date);
                        return {
                          date: dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                          fullDate: dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
                          price: candle.close,
                          // For continuity: prediction line starts at the last historical point
                          predicted: isLast ? candle.close : null,
                          type: 'historical'
                        };
                      }),
                      ...stockData.future_predictions.map((pred) => {
                        const dateObj = parseDateLocal(pred.date);
                        return {
                          date: dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                          fullDate: dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
                          predicted: pred.price,
                          type: 'predicted'
                        };
                      })
                    ]} margin={{ top: 10, right: 15, left: 0, bottom: 60 }}>
                      <defs>
                        <linearGradient id="historicalGradientGreen" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#10b981" stopOpacity={0.6} />
                          <stop offset="100%" stopColor="#10b981" stopOpacity={0.1} />
                        </linearGradient>
                        <linearGradient id="historicalGradientRed" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#ef4444" stopOpacity={0.6} />
                          <stop offset="100%" stopColor="#ef4444" stopOpacity={0.1} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#374151"
                        opacity={0.2}
                        vertical={false}
                      />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 10, fill: '#9ca3af' }}
                        angle={-45}
                        textAnchor="end"
                        height={65}
                        stroke="#6b7280"
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: '#9ca3af' }}
                        domain={['auto', 'auto']}
                        width={60}
                        stroke="#6b7280"
                        tickFormatter={(val) => `$${val.toFixed(0)}`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'rgba(17, 24, 39, 0.95)',
                          border: '1px solid #374151',
                          borderRadius: '8px'
                        }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-gray-900 p-3 border border-gray-700 rounded-lg shadow-xl">
                                <p className="text-xs font-bold text-white mb-2">{data.fullDate}</p>
                                {data.price && (
                                  <p className="text-xs text-cyan-400">
                                    Price: <span className="font-bold">${data.price.toFixed(2)}</span>
                                  </p>
                                )}
                                {data.predicted && (
                                  <p className={`text-xs mt-1 ${stockData.is_profit ? 'text-green-400' : 'text-red-400'}`}>
                                    Predicted: <span className="font-bold">${data.predicted.toFixed(2)}</span>
                                  </p>
                                )}
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Legend
                        wrapperStyle={{
                          fontSize: '12px',
                          paddingTop: '12px'
                        }}
                        iconType="line"
                      />
                      <Area
                        type="monotone"
                        dataKey="price"
                        stroke={stockData.is_profit ? '#10b981' : '#ef4444'}
                        fill={stockData.is_profit ? 'url(#historicalGradientGreen)' : 'url(#historicalGradientRed)'}
                        strokeWidth={3}
                        name="Historical Price"
                        dot={false}
                        activeDot={{ r: 5, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke={stockData.is_profit ? '#059669' : '#dc2626'}
                        strokeWidth={3.5}
                        strokeDasharray="6 4"
                        dot={{ fill: stockData.is_profit ? '#10b981' : '#ef4444', r: 5, strokeWidth: 2, stroke: '#fff' }}
                        name="AI Prediction"
                      />
                      <ReferenceLine
                        y={stockData.current_price}
                        stroke="#6b7280"
                        strokeDasharray="3 3"
                        strokeWidth={1.5}
                        label={{ value: 'Current', fontSize: 11, fill: '#6b7280', position: 'right' }}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Technical Chart - Professional Candlestick */}
              <div className="bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl p-5 sm:p-7 border border-gray-200 dark:border-gray-700 hover:shadow-2xl transition-all duration-300 overflow-hidden">
                <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-600 dark:text-cyan-400" />
                  <span>Historical & Predicted Price</span>
                </h3>
                <div className="w-full bg-gray-50 dark:bg-gray-900 rounded-lg p-2 sm:p-3 overflow-hidden h-[300px] sm:h-[400px] lg:h-[450px]">
                  <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
                    <ComposedChart
                      data={[
                        ...stockData.technical_chart.candles.slice(-30).map((candle, idx, arr) => ({
                          ...candle,
                          // Connect prediction line to the last historical close
                          predicted: idx === arr.length - 1 ? candle.close : null
                        })),
                        ...stockData.future_predictions.map(pred => ({
                          date: pred.date,
                          predicted: pred.price,
                          isPrediction: true
                        }))
                      ]}
                      margin={{ top: 20, right: 30, left: 5, bottom: 70 }}
                    >
                      <defs>
                        <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#374151"
                        opacity={0.2}
                        vertical={false}
                      />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 11, fill: '#9ca3af' }}
                        tickFormatter={(val) => parseDateLocal(val).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                        angle={-45}
                        textAnchor="end"
                        height={75}
                        stroke="#6b7280"
                      />
                      <YAxis
                        yAxisId="price"
                        orientation="right"
                        domain={['auto', 'auto']}
                        tick={{ fontSize: 11, fill: '#9ca3af' }}
                        width={70}
                        stroke="#6b7280"
                        tickFormatter={(val) => `$${val.toFixed(2)}`}
                      />
                      <YAxis
                        yAxisId="volume"
                        orientation="left"
                        tick={{ fontSize: 10, fill: '#9ca3af' }}
                        width={65}
                        stroke="#6b7280"
                        tickFormatter={(val) => `${(val / 1000000).toFixed(1)}M`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'rgba(17, 24, 39, 0.95)',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                          backdropFilter: 'blur(10px)'
                        }}
                        content={({ active, payload, label }) => {
                          if (!active || !payload || payload.length === 0) return null;

                          // Get the full candle data from the chart data using the date
                          const chartData = stockData.technical_chart.candles.slice(-30);
                          const candleData = chartData.find(candle => candle.date === label);

                          console.log('🎯 Label:', label);
                          console.log('🎯 Candle Data Found:', candleData);

                          if (!candleData) {
                            console.error('❌ Could not find candle data for date:', label);
                            return null;
                          }

                          // Now we have the full OHLC data
                          const open = parseFloat(candleData.open);
                          const high = parseFloat(candleData.high);
                          const low = parseFloat(candleData.low);
                          const close = parseFloat(candleData.close);
                          const volume = parseInt(candleData.volume) || 0;

                          console.log('💰 OHLC Values:', { open, high, low, close, volume });

                          if (isNaN(open) || isNaN(close)) {
                            console.error('❌ Invalid OHLC values');
                            return null;
                          }

                          const isGreen = close >= open;
                          const change = open > 0 ? ((close - open) / open * 100) : 0;

                          return (
                            <div className="bg-gray-900 p-3 border border-gray-700 rounded-lg shadow-xl">
                              <p className="text-xs font-bold text-white mb-2">
                                {parseDateLocal(candleData.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                              </p>
                              <div className="space-y-1.5">
                                <div className="flex justify-between gap-6">
                                  <span className="text-xs text-gray-400">Open:</span>
                                  <span className="text-xs font-bold text-blue-400">${open.toFixed(2)}</span>
                                </div>
                                <div className="flex justify-between gap-6">
                                  <span className="text-xs text-gray-400">High:</span>
                                  <span className="text-xs font-bold text-green-400">${high.toFixed(2)}</span>
                                </div>
                                <div className="flex justify-between gap-6">
                                  <span className="text-xs text-gray-400">Low:</span>
                                  <span className="text-xs font-bold text-red-400">${low.toFixed(2)}</span>
                                </div>
                                <div className="flex justify-between gap-6">
                                  <span className="text-xs text-gray-400">Close:</span>
                                  <span className={`text-xs font-bold ${isGreen ? 'text-green-400' : 'text-red-400'}`}>
                                    ${close.toFixed(2)}
                                  </span>
                                </div>
                                <div className="flex justify-between gap-6 pt-1.5 border-t border-gray-700">
                                  <span className="text-xs text-gray-400">Volume:</span>
                                  <span className="text-xs font-bold text-purple-400">
                                    {volume > 0 ? formatVolume(volume) : 'N/A'}
                                  </span>
                                </div>
                                <div className="flex justify-between gap-6">
                                  <span className="text-xs text-gray-400">Change:</span>
                                  <span className={`text-xs font-bold ${isGreen ? 'text-green-400' : 'text-red-400'}`}>
                                    {change > 0 ? '+' : ''}{change.toFixed(2)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Legend
                        wrapperStyle={{
                          fontSize: '12px',
                          paddingTop: '12px',
                          color: '#9ca3af'
                        }}
                      />

                      {/* Volume bars in background */}
                      <Bar
                        yAxisId="volume"
                        dataKey="volume"
                        fill="url(#volumeGradient)"
                        name="Volume"
                        radius={[2, 2, 0, 0]}
                        opacity={0.5}
                      />

                      {/* Candlestick Chart - Simplified and Working */}
                      <Bar
                        yAxisId="price"
                        dataKey="close"
                        name="Price"
                        isAnimationActive={false}
                        shape={(props) => {
                          const { x, y, width, payload, height } = props;
                          if (!payload || !payload.open || !payload.close || !payload.high || !payload.low) {
                            console.warn('Missing OHLC data in shape:', payload);
                            return null;
                          }

                          // Get the Y-axis scale from the chart
                          const chartData = stockData.technical_chart.candles.slice(-30);
                          const priceHigh = Math.max(...chartData.map(d => d.high));
                          const priceLow = Math.min(...chartData.map(d => d.low));
                          const priceRange = priceHigh - priceLow;

                          // Calculate Y positions based on chart dimensions
                          const yScale = (price) => {
                            const ratio = (price - priceLow) / priceRange;
                            return y + height - (ratio * height);
                          };

                          const openY = yScale(payload.open);
                          const closeY = yScale(payload.close);
                          const highY = yScale(payload.high);
                          const lowY = yScale(payload.low);

                          const isGreen = payload.close >= payload.open;
                          const color = isGreen ? '#10b981' : '#ef4444';
                          const fillColor = isGreen ? '#10b981' : '#ef4444';

                          const wickX = x + width / 2;
                          const candleWidth = Math.max(width * 0.65, 3);
                          const candleX = x + (width - candleWidth) / 2;
                          const bodyTop = Math.min(openY, closeY);
                          const bodyHeight = Math.max(Math.abs(closeY - openY), 1);

                          return (
                            <g
                              key={`candle-${payload.date}`}
                              data-open={payload.open}
                              data-high={payload.high}
                              data-low={payload.low}
                              data-close={payload.close}
                              data-volume={payload.volume}
                            >
                              {/* High-Low Wick */}
                              <line
                                x1={wickX}
                                y1={highY}
                                x2={wickX}
                                y2={lowY}
                                stroke={color}
                                strokeWidth={1.5}
                              />
                              {/* Open-Close Body */}
                              <rect
                                x={candleX}
                                y={bodyTop}
                                width={candleWidth}
                                height={bodyHeight}
                                fill={isGreen ? fillColor : 'transparent'}
                                stroke={color}
                                strokeWidth={1.5}
                              />
                            </g>
                          );
                        }}
                      />

                      {/* Moving Average Line */}
                      {stockData.technical_chart.moving_averages.sma20.length > 0 && (
                        <Line
                          yAxisId="price"
                          type="monotone"
                          data={stockData.technical_chart.moving_averages.sma20.slice(-30)}
                          dataKey="value"
                          stroke="#f59e0b"
                          strokeWidth={2.5}
                          dot={false}
                          name="SMA 20"
                          strokeDasharray="5 3"
                        />
                      )}

                      {/* AI Prediction Line Overlay on Technical Chart */}
                      <Line
                        yAxisId="price"
                        type="monotone"
                        dataKey="predicted"
                        stroke={stockData.is_profit ? '#10b981' : '#ef4444'}
                        strokeWidth={3}
                        strokeDasharray="5 5"
                        dot={{ r: 4, fill: '#fff', strokeWidth: 2 }}
                        name="AI Prediction"
                        connectNulls={true}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Professional Performance Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-xl sm:rounded-2xl shadow-lg p-4 sm:p-6 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
                  <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white">Performance</h3>
                  <div className="flex gap-1 sm:gap-2 bg-gray-100 dark:bg-gray-700 p-1 rounded-lg">
                    {['1W', '1M', '3M', '6M', '1Y', 'ALL'].map((period) => (
                      <button
                        key={period}
                        onClick={() => setPerformancePeriod(period)}
                        className={`px-2 sm:px-3 py-1.5 text-xs sm:text-sm font-semibold rounded-md transition-all performance-button touch-target ${performancePeriod === period
                          ? 'bg-cyan-600 text-white shadow-lg transform scale-105'
                          : 'text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                          }`}
                      >
                        {period}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="flex gap-2 sm:gap-3 mb-4 sm:mb-6 flex-wrap">
                  {Object.entries(stockData.performance).map(([period, value]) => (
                    value !== null && (
                      <div key={period} className="text-center bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 px-3 sm:px-4 py-2.5 rounded-lg border border-gray-200 dark:border-gray-600 flex-1 min-w-[70px] shadow-sm">
                        <p className="text-xs text-gray-500 dark:text-gray-400 font-semibold truncate uppercase">{period}</p>
                        <p className={`text-sm sm:text-base font-bold mt-1 ${value >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                          {value >= 0 ? '+' : ''}{value.toFixed(2)}%
                        </p>
                      </div>
                    )
                  ))}
                </div>
                <div className="w-full bg-gray-50 dark:bg-gray-900 rounded-lg p-3" style={{ height: '300px', minHeight: '250px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={(() => {
                        const perfData = getPerformanceData();
                        return perfData.dates.map((date, i) => {
                          const dateObj = parseDateLocal(date);
                          return {
                            date: dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                            fullDate: dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
                            price: perfData.prices[i]
                          };
                        });
                      })()}
                      margin={{ top: 10, right: 15, left: 0, bottom: 50 }}
                    >
                      <defs>
                        {(() => {
                          const perfData = getPerformanceData();
                          const isProfit = perfData.prices.length > 1 && perfData.prices[perfData.prices.length - 1] >= perfData.prices[0];
                          return (
                            <>
                              <linearGradient id="performanceGradientGreen" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#10b981" stopOpacity={0.6} />
                                <stop offset="100%" stopColor="#10b981" stopOpacity={0.05} />
                              </linearGradient>
                              <linearGradient id="performanceGradientRed" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.6} />
                                <stop offset="100%" stopColor="#ef4444" stopOpacity={0.05} />
                              </linearGradient>
                            </>
                          );
                        })()}
                      </defs>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#374151"
                        opacity={0.2}
                        vertical={false}
                      />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 11, fill: '#9ca3af' }}
                        angle={-45}
                        textAnchor="end"
                        height={60}
                        stroke="#6b7280"
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: '#9ca3af' }}
                        width={55}
                        stroke="#6b7280"
                        tickFormatter={(val) => `$${val.toFixed(0)}`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'rgba(17, 24, 39, 0.95)',
                          border: '1px solid #374151',
                          borderRadius: '8px'
                        }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            const perfData = getPerformanceData();
                            const startPrice = perfData.prices[0];
                            const change = data.price - startPrice;
                            const changePercent = ((change / startPrice) * 100).toFixed(2);
                            return (
                              <div className="bg-gray-900 p-3 border border-gray-700 rounded-lg shadow-xl">
                                <p className="text-xs font-bold text-white mb-2">{data.fullDate}</p>
                                <p className="text-xs text-cyan-400">Price: <span className="font-bold">${data.price.toFixed(2)}</span></p>
                                <p className={`text-xs mt-1 ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  Change: <span className="font-bold">{change >= 0 ? '+' : ''}${change.toFixed(2)} ({changePercent}%)</span>
                                </p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="price"
                        stroke={(() => {
                          const perfData = getPerformanceData();
                          return perfData.prices.length > 1 && perfData.prices[perfData.prices.length - 1] >= perfData.prices[0] ? '#10b981' : '#ef4444';
                        })()}
                        fill={(() => {
                          const perfData = getPerformanceData();
                          return perfData.prices.length > 1 && perfData.prices[perfData.prices.length - 1] >= perfData.prices[0]
                            ? 'url(#performanceGradientGreen)'
                            : 'url(#performanceGradientRed)';
                        })()}
                        strokeWidth={3}
                        dot={false}
                        activeDot={{ r: 6, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Professional Forecast Table */}
              <div className="bg-white dark:bg-gray-800 rounded-xl sm:rounded-2xl shadow-lg p-4 sm:p-6 border border-gray-200 dark:border-gray-700">
                <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4">Forecast Signals</h3>
                
                {stockData.is_training ? (
                  <div className="bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-200 dark:border-blue-500/30 rounded-xl p-6 sm:p-8 text-center">
                    <Activity className="w-10 h-10 text-blue-600 dark:text-blue-400 mx-auto mb-4 animate-pulse" />
                    <h4 className="text-lg sm:text-xl font-bold text-gray-900 dark:text-white mb-2">{trainingStateLabel}</h4>
                    <p className="text-gray-600 dark:text-gray-400 font-medium max-w-lg mx-auto">
                      {isPreliminaryMode
                        ? `This is a first-time ${stockData.ticker} request. We are showing preliminary real-time market analysis now while a custom ticker model trains in the background.`
                        : `The custom ${stockData.ticker} model is almost ready. This section will update automatically.`}
                    </p>
                    <div className="max-w-md mx-auto mt-5">
                      <div className="flex justify-between text-xs font-bold text-gray-600 dark:text-gray-300 mb-1">
                        <span>{activeModelStatus.stage || 'training'}</span>
                        <span>{Math.max(0, Math.min(100, trainingProgress)).toFixed(0)}%</span>
                      </div>
                      <div className="h-3 bg-white dark:bg-gray-800 rounded-full overflow-hidden border border-blue-200 dark:border-blue-500/30">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full transition-all duration-700"
                          style={{ width: `${Math.max(4, Math.min(100, trainingProgress))}%` }}
                        />
                      </div>
                    </div>
                    <p className="mt-4 text-xs text-gray-500 dark:text-gray-400">
                      No aggressive BUY/SELL forecast is shown until the trained model passes readiness checks.
                    </p>
                  </div>
                ) : (
                  <div className="overflow-x-auto table-container bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <table className="min-w-full text-left text-xs sm:text-sm">
                      <thead>
                        <tr className="bg-gradient-to-r from-gray-100 to-gray-50 dark:from-gray-700 dark:to-gray-800 text-gray-700 dark:text-gray-300 uppercase text-xs font-bold border-b-2 border-gray-200 dark:border-gray-600">
                          <th className="px-3 sm:px-4 py-3 sm:py-4">Date</th>
                          <th className="px-3 sm:px-4 py-3 sm:py-4">Signal</th>
                          <th className="px-3 sm:px-4 py-3 sm:py-4">Predicted</th>
                          <th className="px-3 sm:px-4 py-3 sm:py-4">Δ Price</th>
                          <th className="px-3 sm:px-4 py-3 sm:py-4">Δ %</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white dark:bg-gray-800">
                        {predictionRows.length === 0 && (
                          <tr>
                            <td colSpan={5} className="px-3 sm:px-4 py-3 sm:py-4 text-center text-gray-500 dark:text-gray-400">
                              No forecast data available.
                            </td>
                          </tr>
                        )}
                        {predictionRows.map((row) => (
                          <tr key={row.id} className="border-t border-gray-200 dark:border-gray-700 hover:bg-gradient-to-r hover:from-gray-50 hover:to-transparent dark:hover:from-gray-700 dark:hover:to-transparent transition-all duration-200">
                            <td className="px-3 sm:px-4 py-3 sm:py-4 font-semibold text-gray-900 dark:text-gray-100 whitespace-nowrap">{row.dateLabel}</td>
                            <td className="px-3 sm:px-4 py-3 sm:py-4">
                              <span
                                className={`px-3 py-1.5 rounded-lg text-xs font-bold whitespace-nowrap shadow-sm ${row.signal.includes('BUY')
                                  ? 'bg-gradient-to-r from-green-500 to-green-600 text-white'
                                  : row.signal.includes('SELL')
                                    ? 'bg-gradient-to-r from-red-500 to-red-600 text-white'
                                    : 'bg-gradient-to-r from-gray-400 to-gray-500 text-white'
                                  }`}
                              >
                                {row.signal}
                              </span>
                            </td>
                            <td className="px-3 sm:px-4 py-3 sm:py-4 text-gray-900 dark:text-gray-100 font-bold whitespace-nowrap">${row.price}</td>
                            <td className={`px-3 sm:px-4 py-3 sm:py-4 font-bold whitespace-nowrap ${Number(row.change) >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                              {Number(row.change) >= 0 ? '+' : ''}${row.change}
                            </td>
                            <td className={`px-3 sm:px-4 py-3 sm:py-4 font-bold whitespace-nowrap ${Number(row.changePercent) >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                              {Number(row.changePercent) >= 0 ? '+' : ''}{row.changePercent}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - Professional Indicators & Stats */}
            <div className="space-y-6">
              {/* Technical Indicators */}
              <div className="bg-white dark:bg-gray-800 rounded-xl sm:rounded-2xl shadow-lg p-4 sm:p-6 border border-gray-200 dark:border-gray-700">
                <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4">Technical Indicators</h3>

                <div className="space-y-3 sm:space-y-4">
                  {stockData.indicators.rsi && (
                    <div className="bg-gradient-to-br from-cyan-50 to-cyan-100 dark:from-cyan-900/20 dark:to-cyan-800/20 p-3 sm:p-4 rounded-xl border border-cyan-300 dark:border-cyan-600/30 indicator-card shadow-sm">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs sm:text-sm font-bold text-gray-700 dark:text-gray-200">RSI (14)</span>
                        <span className="text-base sm:text-lg font-bold text-cyan-700 dark:text-cyan-400 indicator-value">{stockData.indicators.rsi}</span>
                      </div>
                      <div className="h-12 sm:h-16">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={stockData.indicator_trends.rsi.dates.slice(-20).map((date, i) => ({
                            value: stockData.indicator_trends.rsi.values.slice(-20)[i]
                          }))}>
                            <Line type="monotone" dataKey="value" stroke="#0891b2" strokeWidth={2} dot={false} />
                            <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1} />
                            <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" strokeWidth={1} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                        <span>Oversold (30)</span>
                        <span>Overbought (70)</span>
                      </div>
                    </div>
                  )}

                  {stockData.indicators.ema && (
                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-3 sm:p-4 rounded-xl border border-blue-300 dark:border-blue-600/30 indicator-card shadow-sm">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs sm:text-sm font-bold text-gray-700 dark:text-gray-200">EMA (20)</span>
                        <span className="text-base sm:text-lg font-bold text-blue-700 dark:text-blue-400 indicator-value">${stockData.indicators.ema}</span>
                      </div>
                      <div className="h-12 sm:h-16">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={stockData.indicator_trends.ema.dates.slice(-20).map((date, i) => ({
                            value: stockData.indicator_trends.ema.values.slice(-20)[i]
                          }))}>
                            <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}

                  {stockData.indicators.macd && (
                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-3 sm:p-4 rounded-xl border border-purple-300 dark:border-purple-600/30 indicator-card shadow-sm">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs sm:text-sm font-bold text-gray-700 dark:text-gray-200">MACD</span>
                        <span className="text-base sm:text-lg font-bold text-purple-700 dark:text-purple-400 indicator-value">{stockData.indicators.macd}</span>
                      </div>
                      <div className="h-12 sm:h-16">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart data={stockData.indicator_trends.macd.dates.slice(-20).map((date, i) => ({
                            value: stockData.indicator_trends.macd.values.slice(-20)[i],
                            histogram: stockData.indicator_trends.macd.histogram.slice(-20)[i]
                          }))}>
                            <Bar dataKey="histogram" fill={stockData.indicators.macd_histogram >= 0 ? '#10b981' : '#ef4444'} radius={[2, 2, 0, 0]} />
                            <Line type="monotone" dataKey="value" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mt-1">
                        <span>Signal: {stockData.indicators.macd_signal || 'N/A'}</span>
                        <span>Hist: {stockData.indicators.macd_histogram?.toFixed(2) || 'N/A'}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Stats */}
              <div className="bg-white dark:bg-gray-800 rounded-xl sm:rounded-2xl shadow-lg p-4 sm:p-6 border border-gray-200 dark:border-gray-700">
                <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4">Stats</h3>
                <div className="space-y-2 sm:space-y-3">
                  <div className="flex justify-between items-center p-3 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 rounded-lg border border-gray-200 dark:border-gray-600 shadow-sm">
                    <span className="text-xs sm:text-sm text-gray-700 dark:text-gray-300 font-semibold">Market Cap</span>
                    <span className="text-xs sm:text-sm font-bold text-gray-900 dark:text-white">
                      {formatCurrencyCompact(stockData.market_cap)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 rounded-lg border border-gray-200 dark:border-gray-600 shadow-sm">
                    <span className="text-xs sm:text-sm text-gray-700 dark:text-gray-300 font-semibold">Volume</span>
                    <span className="text-xs sm:text-sm font-bold text-gray-900 dark:text-white">{formatVolume(stockData.volume)}</span>
                  </div>
                  {stockData.pe_ratio && (
                    <div className="flex justify-between items-center p-3 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 rounded-lg border border-gray-200 dark:border-gray-600 shadow-sm">
                      <span className="text-xs sm:text-sm text-gray-700 dark:text-gray-300 font-semibold">P/E Ratio</span>
                      <span className="text-xs sm:text-sm font-bold text-gray-900 dark:text-white">{stockData.pe_ratio}</span>
                    </div>
                  )}
                  {stockData.day_high && (
                    <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-50 dark:bg-dark-elevated rounded-lg border border-gray-200 dark:border-dark-border">
                      <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 font-medium">Day High</span>
                      <span className="text-xs sm:text-sm font-bold text-green-600 dark:text-green-400">${stockData.day_high}</span>
                    </div>
                  )}
                  {stockData.day_low && (
                    <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-50 dark:bg-dark-elevated rounded-lg border border-gray-200 dark:border-dark-border">
                      <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 font-medium">Day Low</span>
                      <span className="text-xs sm:text-sm font-bold text-red-600 dark:text-red-400">${stockData.day_low}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Sentiment */}
              {(sentiment || stockData?.sentiment) && (
                <div className="bg-white dark:bg-gray-800 rounded-xl sm:rounded-2xl shadow-lg p-4 sm:p-6 border border-gray-200 dark:border-gray-700">
                  <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white mb-2">Stock Sentiment</h3>
                  <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Real-time market sentiment analysis
                  </p>
                  <SentimentGauge sentiment={sentiment || stockData.sentiment} />
                </div>
              )}

              {/* News */}
              {news.length > 0 && (
                <div className="bg-white dark:bg-gray-800 rounded-xl sm:rounded-2xl shadow-lg p-4 sm:p-6 border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center gap-2 mb-4 flex-wrap">
                    <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white">Latest News</h3>
                    <div className="flex items-center gap-2 px-2 py-1 bg-cyan-50 dark:bg-cyan-900/30 rounded-lg border border-cyan-200 dark:border-cyan-600/30">
                      <div className="relative w-5 h-5 sm:w-6 sm:h-6 rounded bg-gray-100 dark:bg-gray-700 flex items-center justify-center overflow-hidden stock-logo-sm">
                        <img
                          src={`https://financialmodelingprep.com/image-stock/${stockData.ticker}.png`}
                          alt={stockData.ticker}
                          className="w-full h-full object-contain stock-logo"
                          onError={(e) => {
                            e.target.onerror = null;
                            e.target.style.display = 'none';
                            e.target.parentElement.innerHTML = `<span class="text-xs font-bold text-cyan-600 dark:text-cyan-400">${stockData.ticker.charAt(0)}</span>`;
                          }}
                        />
                      </div>
                      <span className="text-xs sm:text-sm font-bold text-cyan-700 dark:text-cyan-400">{stockData.ticker}</span>
                    </div>
                  </div>
                  <div className="space-y-3 sm:space-y-4">
                    {news.slice(0, visibleNewsCount).map((article, index) => {
                      // Safely handle datetime conversion
                      let newsDate;
                      let isValidDate = false;

                      try {
                        // Check if datetime exists and is valid
                        if (article.datetime) {
                          newsDate = new Date(article.datetime * 1000);
                          isValidDate = !isNaN(newsDate.getTime());
                        }
                      } catch (e) {
                        console.warn('Invalid date for article:', article);
                      }

                      // Fallback to current date if invalid
                      if (!isValidDate) {
                        newsDate = new Date();
                      }

                      const now = new Date();
                      const diffMs = now - newsDate;
                      const diffMins = Math.floor(diffMs / 60000);
                      const diffHours = Math.floor(diffMs / 3600000);
                      const diffDays = Math.floor(diffMs / 86400000);

                      let timeAgo;
                      if (!isValidDate) {
                        timeAgo = 'Recently';
                      } else if (diffMins < 1) {
                        timeAgo = 'Just now';
                      } else if (diffMins < 60) {
                        timeAgo = `${diffMins}m ago`;
                      } else if (diffHours < 24) {
                        timeAgo = `${diffHours}h ago`;
                      } else if (diffDays < 7) {
                        timeAgo = `${diffDays}d ago`;
                      } else {
                        timeAgo = newsDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
                      }

                      return (
                        <a
                          key={`${article.id || index}-${article.datetime || index}`}
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="block p-3 rounded-lg bg-gray-50 dark:bg-gray-900 hover:bg-gray-100 dark:hover:bg-gray-700 transition-all border border-gray-200 dark:border-gray-600 hover:border-cyan-400 dark:hover:border-cyan-500 group active-scale"
                        >
                          <div className="flex flex-col xs:flex-row gap-3">
                            {article.image && (
                              <div className="flex-shrink-0">
                                <img
                                  src={article.image}
                                  alt={article.headline}
                                  className="w-full xs:w-20 xs:h-20 sm:w-24 sm:h-24 object-cover rounded-lg news-image"
                                  onError={(e) => {
                                    e.target.style.display = 'none';
                                  }}
                                />
                              </div>
                            )}
                            <div className="flex-1 min-w-0">
                              <p className="text-xs sm:text-sm font-semibold text-gray-900 dark:text-white line-clamp-2 group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors mb-2">
                                {article.headline}
                              </p>
                              {article.summary && (
                                <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2 mb-2">
                                  {article.summary}
                                </p>
                              )}
                              <div className="flex items-center gap-2 flex-wrap text-xs text-gray-500 dark:text-gray-400">
                                <span className="font-medium bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded">
                                  {article.source || 'Finnhub'}
                                </span>
                                <span>•</span>
                                <time className="flex items-center gap-1">
                                  <Calendar className="w-3 h-3" />
                                  {timeAgo}
                                </time>
                                {isValidDate && (
                                  <>
                                    <span>•</span>
                                    <span className="hidden sm:inline truncate">
                                      {newsDate.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                                    </span>
                                  </>
                                )}
                              </div>
                            </div>
                          </div>
                        </a>
                      );
                    })}
                  </div>
                  {news.length > visibleNewsCount && (
                    <div className="mt-4 text-center">
                      <button
                        onClick={handleLoadMoreNews}
                        className="text-sm text-cyan-600 dark:text-cyan-400 hover:underline font-medium px-4 py-2 rounded-lg hover:bg-cyan-50 dark:hover:bg-cyan-900/30 transition-colors"
                      >
                        Load more news ({news.length - visibleNewsCount} more articles)
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </main>
      )}
    </div>
  );
}

export default Prediction;

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { TrendingUp, TrendingDown, Search, DollarSign, BarChart3, Activity, Calendar, Image } from 'lucide-react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, ComposedChart, ReferenceLine, Cell, Scatter
} from 'recharts';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import CinematicBackground from '../components/CinematicBackground';

const API_BASE = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000');
const FORECAST_HORIZONS = [1, 7, 14];
const PREDICTION_STORAGE_KEYS = {
  ticker: 'prediction_ticker',
  hasSearched: 'prediction_has_searched',
  stockData: 'prediction_stock_data',
  sentiment: 'prediction_sentiment',
  news: 'prediction_news',
  trainingStatus: 'prediction_training_status',
  days: 'prediction_days',
  watchlist: 'prediction_watchlist',
  agentCommand: 'prediction_agent_command'
};

const normalizeForecastDays = (value) => {
  const numeric = Number(value);
  if (numeric <= 1) return 1;
  if (numeric <= 7) return 7;
  return 14;
};

const readStoredJson = (key, fallback) => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (err) {
    console.warn(`Failed to restore ${key} from localStorage`, err);
    return fallback;
  }
};

const getSignalRankValue = (signal = '') => {
  const upper = String(signal).toUpperCase();
  if (upper.includes('STRONG BUY')) return 2;
  if (upper.includes('BUY')) return 1;
  if (upper.includes('STRONG SELL')) return -2;
  if (upper.includes('SELL')) return -1;
  return 0;
};

const clampNumber = (value, min = 0, max = 100) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return min;
  return Math.max(min, Math.min(max, numeric));
};

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


const getCurrencySymbol = (currency) => {
  if (!currency) return '$';
  const map = {
    'USD': '$',
    'INR': '₹',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CAD': 'CA$',
    'AUD': 'A$',
    'HKD': 'HK$',
    'CNY': '¥',
    'SGD': 'S$'
  };
  return map[currency.toUpperCase()] || currency + ' ';
};

const getDisplayCurrencySymbol = (currency) => {
  if (!currency) return '$';
  const map = {
    USD: '$',
    INR: '\u20b9',
    EUR: '\u20ac',
    GBP: '\u00a3',
    JPY: '\u00a5',
    CAD: 'CA$',
    AUD: 'A$',
    HKD: 'HK$',
    CNY: '\u00a5',
    SGD: 'S$'
  };
  return map[String(currency).toUpperCase()] || `${String(currency).toUpperCase()} `;
};

const formatMoney = (value, currency = 'USD') => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 'N/A';
  return `${getDisplayCurrencySymbol(currency)}${numeric.toFixed(2)}`;
};

const formatEta = (seconds) => {
  const numeric = Number(seconds);
  if (!Number.isFinite(numeric) || numeric <= 0) return 'Ready now';
  if (numeric < 60) return 'under 1 min';
  return `about ${Math.max(1, Math.round(numeric / 60))} min`;
};

const formatCurrencyCompact = (value, curr = "$") => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return 'N/A';
  }

  const absValue = Math.abs(numeric);
  if (absValue >= 1e12) {
    return `${curr}${(numeric / 1e12).toFixed(2)}T`;
  }
  if (absValue >= 1e9) {
    return `${curr}${(numeric / 1e9).toFixed(2)}B`;
  }
  if (absValue >= 1e6) {
    return `${curr}${(numeric / 1e6).toFixed(2)}M`;
  }
  if (absValue >= 1e3) {
    return `${curr}${(numeric / 1e3).toFixed(2)}K`;
  }
  return `${curr}${numeric.toFixed(2)}`;
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

const TradingViewStyleChart = ({
  candles = [],
  sma20 = [],
  predictions = [],
  isProfit,
  currencySymbol = "$",
  recommendation = null,
  tradePlan = null,
  backtestGhost = []
}) => {
  const [hoverIndex, setHoverIndex] = useState(null);
  const [viewStart, setViewStart] = useState(null);
  const [viewCount, setViewCount] = useState(45);
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef({ startX: 0, startViewStart: 0 });

  const width = 1120;
  const height = 500;
  const pad = { top: 34, right: 34, bottom: 46, left: 16 };
  const volumeHeight = 78;
  const volumeGap = 18;
  const priceBottom = height - pad.bottom - volumeHeight - volumeGap;
  const volumeTop = priceBottom + volumeGap;
  const priceHeight = priceBottom - pad.top;
  const plotWidth = width - pad.left - pad.right;

  const allValid = candles.filter((c) =>
    Number.isFinite(Number(c.open)) && Number.isFinite(Number(c.high)) &&
    Number.isFinite(Number(c.low)) && Number.isFinite(Number(c.close))
  );

  // Initialize viewStart
  const effectiveViewCount = Math.min(viewCount, Math.max(allValid.length, 1));
  const effectiveStart = viewStart ?? Math.max(0, allValid.length - effectiveViewCount);
  const visibleCandles = allValid.slice(effectiveStart, effectiveStart + effectiveViewCount);

  if (allValid.length === 0) {
    return (
      <div className="h-[340px] flex items-center justify-center text-sm font-semibold text-gray-500 dark:text-gray-400">
        Chart data is not available.
      </div>
    );
  }

  const futurePoints = predictions.slice(0, 6).filter((p) => Number.isFinite(Number(p.price)));
  const ghostPoints = (backtestGhost || []).filter((p) =>
    p?.date &&
    Number.isFinite(Number(p.predicted)) &&
    Number.isFinite(Number(p.actual))
  );
  const fallbackSignal = recommendation?.signal || '';
  const getPointSignal = (point) => {
    const raw = String(point?.signal || fallbackSignal || '').toUpperCase();
    if (raw.includes('BUY')) return 'BUY';
    if (raw.includes('SELL')) return 'SELL';
    if (raw.includes('HOLD')) return 'HOLD';
    return '';
  };
  const getSignalColor = (signal) => {
    if (signal === 'BUY') return '#10b981';
    if (signal === 'SELL') return '#ef4444';
    return '#94a3b8';
  };
  const smaByDate = new Map((sma20 || []).map((p) => [p.date, Number(p.value)]));
  const priceValues = visibleCandles.flatMap((c) => [Number(c.open), Number(c.high), Number(c.low), Number(c.close)]);
  visibleCandles.forEach((c) => { const v = smaByDate.get(c.date); if (Number.isFinite(v)) priceValues.push(v); });
  if (tradePlan) {
    ['entryLow', 'entryHigh', 'stop', 'target1', 'target2'].forEach((key) => {
      const value = Number(tradePlan[key]);
      if (Number.isFinite(value)) priceValues.push(value);
    });
  }
  ghostPoints.forEach((p) => {
    const visible = visibleCandles.some((c) => c.date === p.date);
    if (visible) {
      priceValues.push(Number(p.predicted), Number(p.actual));
    }
  });
  // Only include future points in price range if we're at the end
  const atEnd = effectiveStart + effectiveViewCount >= allValid.length;
  if (atEnd) {
    futurePoints.forEach((p) => {
      priceValues.push(Number(p.price));
      if (Number.isFinite(Number(p.range_low))) priceValues.push(Number(p.range_low));
      if (Number.isFinite(Number(p.range_high))) priceValues.push(Number(p.range_high));
    });
  }

  const minPrice = Math.min(...priceValues);
  const maxPrice = Math.max(...priceValues);
  const pricePadding = Math.max((maxPrice - minPrice) * 0.12, maxPrice * 0.003, 0.5);
  const lowBound = minPrice - pricePadding;
  const highBound = maxPrice + pricePadding;
  const priceRange = highBound - lowBound || 1;
  const maxVolume = Math.max(...visibleCandles.map((c) => Number(c.volume) || 0), 1);
  const pointCount = visibleCandles.length + (atEnd ? futurePoints.length : 0);
  const step = plotWidth / Math.max(pointCount - 1, 1);
  const candleWidth = Math.max(4, Math.min(14, step * 0.6));

  const xForIndex = (i) => pad.left + i * step;
  const yForPrice = (p) => pad.top + ((highBound - Number(p)) / priceRange) * priceHeight;
  const yForVolume = (v) => { const r = Math.max(0, Math.min(1, (Number(v) || 0) / maxVolume)); return height - pad.bottom - r * volumeHeight; };
  const fmt = (v) => `${currencySymbol}${Number(v).toFixed(2)}`;
  const fmtDate = (d) => parseDateLocal(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  const fmtDateFull = (d) => parseDateLocal(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  const gridPrices = Array.from({ length: 6 }, (_, i) => lowBound + (priceRange * i) / 5).reverse();
  const hoveredCandle = Number.isInteger(hoverIndex) && hoverIndex < visibleCandles.length ? visibleCandles[hoverIndex] : null;
  const hoverX = hoveredCandle ? xForIndex(hoverIndex) : null;
  const hoverY = hoveredCandle ? yForPrice(hoveredCandle.close) : null;
  const hoverRising = hoveredCandle ? Number(hoveredCandle.close) >= Number(hoveredCandle.open) : false;
  const hoverColor = hoverRising ? '#00e87b' : '#ff4d4f';
  const lastCandle = visibleCandles[visibleCandles.length - 1];
  const lastClose = Number(lastCandle?.close);
  const lastCloseY = Number.isFinite(lastClose) ? yForPrice(lastClose) : null;
  const lastCloseColor = lastCandle && Number(lastCandle.close) >= Number(lastCandle.open) ? '#059669' : '#dc2626';

  // Display candle (hover or last)
  const displayCandle = hoveredCandle || visibleCandles[visibleCandles.length - 1];
  const displayRising = displayCandle ? Number(displayCandle.close) >= Number(displayCandle.open) : true;
  const displayColor = displayRising ? '#00e87b' : '#ff4d4f';
  const displayChange = displayCandle ? (Number(displayCandle.close) - Number(displayCandle.open)).toFixed(2) : '0.00';
  const displayChangePct = displayCandle && Number(displayCandle.open) ? ((Number(displayCandle.close) - Number(displayCandle.open)) / Number(displayCandle.open) * 100).toFixed(2) : '0.00';

  // Drag to pan
  const handlePointerDown = (e) => {
    if (e.button !== 0 && !e.touches) return;
    setIsDragging(true);
    const clientX = e.touches?.[0]?.clientX ?? e.clientX;
    dragRef.current = { startX: clientX, startViewStart: effectiveStart };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const handlePointerMove = (e) => {
    const svg = e.currentTarget;
    const clientX = e.touches?.[0]?.clientX ?? e.clientX;
    const rect = svg.getBoundingClientRect();

    if (isDragging) {
      const dx = clientX - dragRef.current.startX;
      const candlesPerPixel = effectiveViewCount / rect.width;
      const shift = Math.round(-dx * candlesPerPixel);
      const newStart = Math.max(0, Math.min(allValid.length - effectiveViewCount, dragRef.current.startViewStart + shift));
      setViewStart(newStart);
    }

    // Hover
    const viewX = ((clientX - rect.left) / rect.width) * width;
    const idx = Math.round((viewX - pad.left) / step);
    setHoverIndex(Math.max(0, Math.min(visibleCandles.length - 1, idx)));
  };
  const handlePointerUp = () => { setIsDragging(false); };

  // Zoom
  const handleZoomIn = () => {
    const nc = Math.max(20, viewCount - 10);
    setViewCount(nc);
    const mid = effectiveStart + Math.floor(effectiveViewCount / 2);
    setViewStart(Math.max(0, Math.min(allValid.length - nc, mid - Math.floor(nc / 2))));
  };
  const handleZoomOut = () => {
    const nc = Math.min(allValid.length, viewCount + 10);
    setViewCount(nc);
    const mid = effectiveStart + Math.floor(effectiveViewCount / 2);
    setViewStart(Math.max(0, Math.min(allValid.length - nc, mid - Math.floor(nc / 2))));
  };
  const handleResetView = () => { setViewCount(45); setViewStart(null); };

  // Wheel zoom
  const handleWheel = (e) => {
    e.preventDefault();
    if (e.deltaY < 0) handleZoomIn();
    else handleZoomOut();
  };

  // SMA path
  let hasSma = false;
  const smaPath = visibleCandles.map((c, i) => {
    const v = smaByDate.get(c.date);
    if (!Number.isFinite(v)) return null;
    const cmd = hasSma ? 'L' : 'M'; hasSma = true;
    return `${cmd} ${xForIndex(i).toFixed(1)} ${yForPrice(v).toFixed(1)}`;
  }).filter(Boolean).join(' ');

  // Prediction path
  const predPath = atEnd && futurePoints.length > 0
    ? [`M ${xForIndex(visibleCandles.length - 1).toFixed(1)} ${yForPrice(visibleCandles[visibleCandles.length - 1].close).toFixed(1)}`,
       ...futurePoints.map((p, i) => `L ${xForIndex(visibleCandles.length + i).toFixed(1)} ${yForPrice(p.price).toFixed(1)}`)
      ].join(' ')
    : '';
  const rangeBandPath = atEnd && futurePoints.some((p) => Number.isFinite(Number(p.range_low)) && Number.isFinite(Number(p.range_high)))
    ? [
        `M ${xForIndex(visibleCandles.length - 1).toFixed(1)} ${yForPrice(visibleCandles[visibleCandles.length - 1].close).toFixed(1)}`,
        ...futurePoints.map((p, i) => {
          const high = Number.isFinite(Number(p.range_high)) ? Number(p.range_high) : Number(p.price);
          return `L ${xForIndex(visibleCandles.length + i).toFixed(1)} ${yForPrice(high).toFixed(1)}`;
        }),
        ...[...futurePoints].reverse().map((p, reverseIndex) => {
          const originalIndex = futurePoints.length - 1 - reverseIndex;
          const low = Number.isFinite(Number(p.range_low)) ? Number(p.range_low) : Number(p.price);
          return `L ${xForIndex(visibleCandles.length + originalIndex).toFixed(1)} ${yForPrice(low).toFixed(1)}`;
        }),
        'Z'
      ].join(' ')
    : '';
  const rangeHighPath = rangeBandPath
    ? futurePoints.map((p, i) => {
        const high = Number.isFinite(Number(p.range_high)) ? Number(p.range_high) : Number(p.price);
        return `${i === 0 ? 'M' : 'L'} ${xForIndex(visibleCandles.length + i).toFixed(1)} ${yForPrice(high).toFixed(1)}`;
      }).join(' ')
    : '';
  const rangeLowPath = rangeBandPath
    ? futurePoints.map((p, i) => {
        const low = Number.isFinite(Number(p.range_low)) ? Number(p.range_low) : Number(p.price);
        return `${i === 0 ? 'M' : 'L'} ${xForIndex(visibleCandles.length + i).toFixed(1)} ${yForPrice(low).toFixed(1)}`;
      }).join(' ')
    : '';
  const ghostByDate = new Map(ghostPoints.map((p) => [p.date, p]));
  let hasGhostPrediction = false;
  const ghostPredictionPath = visibleCandles.map((c, i) => {
    const point = ghostByDate.get(c.date);
    if (!point) return null;
    const cmd = hasGhostPrediction ? 'L' : 'M';
    hasGhostPrediction = true;
    return `${cmd} ${xForIndex(i).toFixed(1)} ${yForPrice(point.predicted).toFixed(1)}`;
  }).filter(Boolean).join(' ');
  let hasGhostActual = false;
  const ghostActualPath = visibleCandles.map((c, i) => {
    const point = ghostByDate.get(c.date);
    if (!point) return null;
    const cmd = hasGhostActual ? 'L' : 'M';
    hasGhostActual = true;
    return `${cmd} ${xForIndex(i).toFixed(1)} ${yForPrice(point.actual).toFixed(1)}`;
  }).filter(Boolean).join(' ');
  const drawPlanLine = (value, color, label, dash = '6 5') => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < lowBound || numeric > highBound) return null;
    const y = yForPrice(numeric);
    return (
      <g key={`plan-${label}`}>
        <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke={color} strokeWidth="1.5" strokeDasharray={dash} opacity="0.9" />
        <rect x={pad.left + 8} y={y - 11} width="86" height="21" rx="5" fill={color} opacity="0.92" />
        <text x={pad.left + 14} y={y + 4} fill="#ffffff" fontSize="10" fontWeight="900" fontFamily="monospace">{label}</text>
        <text x={width - pad.right + 8} y={y + 4} fill={color} fontSize="11" fontWeight="800" fontFamily="monospace">{fmt(numeric)}</text>
      </g>
    );
  };

  // Scrollbar
  const scrollBarWidth = plotWidth;
  const thumbWidth = Math.max(30, (effectiveViewCount / allValid.length) * scrollBarWidth);
  const thumbX = pad.left + (effectiveStart / Math.max(1, allValid.length - effectiveViewCount)) * (scrollBarWidth - thumbWidth);

  return (
    <div className="w-full overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700/60 bg-white dark:bg-[#0b1220] shadow-xl dark:shadow-2xl">
      {/* OHLCV Header Bar */}
      <div className="flex items-center gap-3 sm:gap-5 px-3 sm:px-5 py-2 border-b border-slate-200 dark:border-slate-800/80 bg-slate-50/90 dark:bg-slate-950/40 text-xs sm:text-sm font-mono select-none flex-wrap">
        <span className="text-slate-700 dark:text-slate-300 font-bold text-sm">OHLCV</span>
        {displayCandle && (
          <>
            <span className="text-slate-700 dark:text-slate-400">{fmtDateFull(displayCandle.date)}</span>
            <span className="text-slate-600 dark:text-slate-400">O <span className="text-slate-950 dark:text-white font-semibold">{fmt(displayCandle.open)}</span></span>
            <span className="text-slate-600 dark:text-slate-400">H <span className="text-cyan-700 dark:text-cyan-300 font-semibold">{fmt(displayCandle.high)}</span></span>
            <span className="text-slate-600 dark:text-slate-400">L <span className="text-amber-700 dark:text-amber-300 font-semibold">{fmt(displayCandle.low)}</span></span>
            <span className="text-slate-600 dark:text-slate-400">C <span className="font-bold" style={{ color: displayColor }}>{fmt(displayCandle.close)}</span></span>
            <span style={{ color: displayColor }} className="font-bold">{displayRising ? '+' : ''}{displayChange} ({displayRising ? '+' : ''}{displayChangePct}%)</span>
            <span className="text-slate-600 dark:text-slate-500">Vol <span className="text-slate-900 dark:text-slate-300 font-semibold">{formatVolume(displayCandle.volume)}</span></span>
          </>
        )}
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-[380px] sm:h-[460px] lg:h-[520px] select-none"
        style={{ cursor: isDragging ? 'grabbing' : 'crosshair' }}
        role="img"
        aria-label="Interactive stock candlestick chart"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={() => { setHoverIndex(null); setIsDragging(false); }}
        onWheel={handleWheel}
      >
        <defs>
          <clipPath id="tvPriceClip"><rect x={pad.left} y={pad.top} width={plotWidth} height={priceHeight} /></clipPath>
          <clipPath id="tvVolClip"><rect x={pad.left} y={volumeTop} width={plotWidth} height={volumeHeight} /></clipPath>
          <linearGradient id="tvPredGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={isProfit ? '#10b981' : '#ef4444'} stopOpacity="0.8" />
            <stop offset="100%" stopColor={isProfit ? '#10b981' : '#ef4444'} stopOpacity="0.3" />
          </linearGradient>
          <linearGradient id="tvRangeBandGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={isProfit ? '#10b981' : '#ef4444'} stopOpacity="0.16" />
            <stop offset="100%" stopColor={isProfit ? '#10b981' : '#ef4444'} stopOpacity="0.08" />
          </linearGradient>
        </defs>

        {/* Background */}
        <rect x="0" y="0" width={width} height={height} fill="transparent" />
        <rect x={pad.left} y={pad.top} width={plotWidth} height={priceHeight + volumeGap + volumeHeight} className="fill-slate-50/50 dark:fill-[#0f172a]" opacity="0.65" rx="8" />

        {/* Grid */}
        {gridPrices.map((p) => {
          const y = yForPrice(p);
          return (
            <g key={`g-${p}`}>
              <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} className="stroke-slate-200 dark:stroke-slate-700" strokeDasharray="3 6" />
              <text x={width - pad.right + 8} y={y + 4} className="fill-slate-400 dark:fill-slate-500" fontSize="11" fontWeight="500" fontFamily="monospace">{fmt(p)}</text>
            </g>
          );
        })}
        {visibleCandles.map((c, i) => {
          if (i % Math.ceil(visibleCandles.length / 8) !== 0) return null;
          return <line key={`vg-${c.date}`} x1={xForIndex(i)} x2={xForIndex(i)} y1={pad.top} y2={height - pad.bottom} className="stroke-slate-200 dark:stroke-slate-700/50" />;
        })}
        <line x1={pad.left} x2={width - pad.right} y1={priceBottom} y2={priceBottom} className="stroke-slate-300 dark:stroke-slate-700" />

        {/* Candles */}
        <g clipPath="url(#tvPriceClip)">
          {tradePlan && Number.isFinite(Number(tradePlan.entryLow)) && Number.isFinite(Number(tradePlan.entryHigh)) && (
            <rect
              x={pad.left}
              y={Math.min(yForPrice(tradePlan.entryLow), yForPrice(tradePlan.entryHigh))}
              width={plotWidth}
              height={Math.max(2, Math.abs(yForPrice(tradePlan.entryLow) - yForPrice(tradePlan.entryHigh)))}
              fill="#22d3ee"
              opacity="0.12"
            />
          )}
          {ghostPredictionPath && <path d={ghostPredictionPath} fill="none" stroke="#a855f7" strokeWidth="2" strokeDasharray="5 5" strokeLinecap="round" strokeLinejoin="round" opacity="0.85" />}
          {ghostActualPath && <path d={ghostActualPath} fill="none" stroke="#64748b" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" opacity="0.7" />}
          {visibleCandles.map((c, i) => {
            const x = xForIndex(i);
            const o = Number(c.open), h = Number(c.high), l = Number(c.low), cl = Number(c.close);
            const up = cl >= o;
            const col = up ? '#00e87b' : '#ff4d4f';
            const yH = yForPrice(h), yL = yForPrice(l), yO = yForPrice(o), yC = yForPrice(cl);
            const bY = Math.min(yO, yC), bH = Math.max(Math.abs(yC - yO), 2);
            const isHovered = hoverIndex === i;
            return (
              <g key={c.date} opacity={isHovered ? 1 : 0.88}>
                <line x1={x} x2={x} y1={yH} y2={yL} stroke={col} strokeWidth={isHovered ? 2 : 1.3} strokeLinecap="round" />
                <rect x={x - candleWidth / 2} y={bY} width={candleWidth} height={bH} rx="1.5"
                  fill={up ? col : 'transparent'} stroke={col} strokeWidth={isHovered ? 2 : 1.5} />
              </g>
            );
          })}
          {smaPath && <path d={smaPath} fill="none" stroke="#f59e0b" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" opacity="0.8" />}
          {rangeBandPath && <path d={rangeBandPath} fill="url(#tvRangeBandGrad)" stroke="none" />}
          {rangeHighPath && <path d={rangeHighPath} fill="none" stroke={isProfit ? '#10b981' : '#ef4444'} strokeWidth="1.5" strokeDasharray="4 4" strokeLinecap="round" opacity="0.75" />}
          {rangeLowPath && <path d={rangeLowPath} fill="none" stroke={isProfit ? '#10b981' : '#ef4444'} strokeWidth="1.5" strokeDasharray="4 4" strokeLinecap="round" opacity="0.75" />}
          {predPath && <path d={predPath} fill="none" stroke="url(#tvPredGrad)" strokeWidth="2.5" strokeDasharray="6 5" strokeLinecap="round" strokeLinejoin="round" />}
        </g>

        {tradePlan && (
          <g>
            {drawPlanLine(tradePlan.stop, '#ef4444', 'STOP')}
            {drawPlanLine(tradePlan.entryLow, '#22d3ee', 'ENTRY', '3 4')}
            {drawPlanLine(tradePlan.target1, '#10b981', 'TARGET 1')}
            {drawPlanLine(tradePlan.target2, '#059669', 'TARGET 2', '2 3')}
          </g>
        )}

        {/* Forecast signal markers */}
        {atEnd && futurePoints.map((p, i) => {
          const signal = getPointSignal(p);
          if (!signal) return null;
          const x = xForIndex(visibleCandles.length + i);
          const y = yForPrice(p.price);
          const color = getSignalColor(signal);
          const isBuy = signal === 'BUY';
          const isSell = signal === 'SELL';
          const markerY = isBuy ? y + 24 : isSell ? y - 24 : y - 22;
          const points = isBuy
            ? `${x},${markerY - 8} ${x - 7},${markerY + 6} ${x + 7},${markerY + 6}`
            : isSell
              ? `${x},${markerY + 8} ${x - 7},${markerY - 6} ${x + 7},${markerY - 6}`
              : null;

          return (
            <g key={`signal-${p.date || i}`} className="pointer-events-none">
              {points ? (
                <polygon points={points} fill={color} stroke="#ffffff" strokeWidth="1.4" opacity="0.96" />
              ) : (
                <circle cx={x} cy={markerY} r="6" fill={color} stroke="#ffffff" strokeWidth="1.4" opacity="0.96" />
              )}
              <rect x={x - 14} y={markerY + (isBuy ? 9 : -22)} width="28" height="13" rx="3" fill={color} opacity="0.94" />
              <text
                x={x}
                y={markerY + (isBuy ? 19 : -12)}
                fill="#ffffff"
                fontSize="8"
                fontWeight="900"
                textAnchor="middle"
                fontFamily="monospace"
              >
                {signal}
              </text>
            </g>
          );
        })}

        {lastCloseY != null && (
          <g>
            <line x1={pad.left} x2={width - pad.right} y1={lastCloseY} y2={lastCloseY} stroke={lastCloseColor} strokeDasharray="5 5" strokeWidth="1.4" opacity="0.72" />
            <rect x={width - pad.right + 4} y={lastCloseY - 12} width={pad.right - 8} height="24" rx="5" fill={lastCloseColor} />
            <text x={width - pad.right + 10} y={lastCloseY + 4} fill="#ffffff" fontSize="11" fontWeight="800" fontFamily="monospace">{fmt(lastClose)}</text>
          </g>
        )}

        {/* Volume */}
        <g clipPath="url(#tvVolClip)">
          {visibleCandles.map((c, i) => {
            const x = xForIndex(i), vY = Math.max(volumeTop, yForVolume(c.volume));
            const up = Number(c.close) >= Number(c.open);
            return <rect key={`v-${c.date}`} x={x - candleWidth / 2} y={vY} width={candleWidth} height={height - pad.bottom - vY}
              fill={up ? '#14b8a6' : '#f87171'} opacity={hoverIndex === i ? 0.7 : 0.35} rx="1" />;
          })}
        </g>

        {/* Date labels */}
        {visibleCandles.map((c, i) => {
          if (i % Math.ceil(visibleCandles.length / 7) !== 0 && i !== visibleCandles.length - 1) return null;
          return <text key={`d-${c.date}`} x={xForIndex(i)} y={height - 28} className="fill-slate-500 dark:fill-slate-400" fontSize="11" textAnchor="middle" fontFamily="monospace">{fmtDate(c.date)}</text>;
        })}
        {atEnd && futurePoints.map((p, i) => (
          <text key={`fd-${p.date || i}`} x={xForIndex(visibleCandles.length + i)} y={height - 28} className="fill-cyan-700 dark:fill-cyan-300" fontSize="10" textAnchor="middle" fontWeight="700" fontFamily="monospace">
            {fmtDate(p.date)}
          </text>
        ))}

        {/* Crosshair */}
        {hoveredCandle && (
          <g>
            <line x1={hoverX} x2={hoverX} y1={pad.top} y2={height - pad.bottom} className="stroke-slate-400 dark:stroke-slate-500" strokeDasharray="2 3" />
            <line x1={pad.left} x2={width - pad.right} y1={hoverY} y2={hoverY} className="stroke-slate-400 dark:stroke-slate-500" strokeDasharray="2 3" />
            <circle cx={hoverX} cy={hoverY} r="5" fill={hoverColor} className="stroke-white dark:stroke-slate-900" strokeWidth="2" />
            {/* Price tag on Y axis */}
            <rect x={width - pad.right} y={hoverY - 11} width={pad.right - 2} height="22" rx="4" fill={hoverColor} />
            <text x={width - pad.right + 6} y={hoverY + 4} fill="#ffffff" fontSize="11" fontWeight="700" fontFamily="monospace">{fmt(hoveredCandle.close)}</text>
            {/* Date tag on X axis */}
            <rect x={hoverX - 40} y={height - pad.bottom + 2} width="80" height="20" rx="4" className="fill-white dark:fill-slate-800 stroke-slate-200 dark:stroke-slate-700" strokeWidth="1" />
            <text x={hoverX} y={height - pad.bottom + 15} className="fill-slate-800 dark:fill-slate-200" fontSize="10" textAnchor="middle" fontFamily="monospace">{fmtDate(hoveredCandle.date)}</text>
          </g>
        )}

        {/* Scrollbar track */}
        <rect x={pad.left} y={height - 12} width={scrollBarWidth} height="6" rx="3" className="fill-slate-100 dark:fill-slate-800" />
        <rect x={thumbX} y={height - 12} width={thumbWidth} height="6" rx="3" className="fill-slate-300 dark:fill-slate-600" />

        {/* Legend */}
        <g transform={`translate(${pad.left}, ${height - 22})`}>
          <rect width="8" height="8" fill="#00e87b" rx="2" />
          <text x="12" y="8" fill="#64748b" fontSize="10" fontWeight="600">Up</text>
          <rect x="38" width="8" height="8" fill="transparent" stroke="#ff4d4f" rx="2" />
          <text x="50" y="8" fill="#64748b" fontSize="10" fontWeight="600">Down</text>
          <line x1="88" x2="104" y1="4" y2="4" stroke="#f59e0b" strokeWidth="2" />
          <text x="108" y="8" fill="#64748b" fontSize="10" fontWeight="600">SMA 20</text>
          {predPath && (
            <>
              <line x1="166" x2="184" y1="4" y2="4" stroke={isProfit ? '#10b981' : '#ef4444'} strokeWidth="2.5" strokeDasharray="4 3" />
              <text x="188" y="8" fill="#64748b" fontSize="10" fontWeight="600">AI Forecast Range</text>
              <polygon points="296,-2 290,10 302,10" fill="#10b981" />
              <text x="308" y="8" fill="#64748b" fontSize="10" fontWeight="600">Signal Markers</text>
            </>
          )}
          {tradePlan && (
            <>
              <line x1="410" x2="428" y1="4" y2="4" stroke="#22d3ee" strokeWidth="2" strokeDasharray="3 4" />
              <text x="432" y="8" fill="#64748b" fontSize="10" fontWeight="600">Trade Zones</text>
            </>
          )}
          {ghostPredictionPath && (
            <>
              <line x1="516" x2="534" y1="4" y2="4" stroke="#a855f7" strokeWidth="2" strokeDasharray="5 4" />
              <text x="538" y="8" fill="#64748b" fontSize="10" fontWeight="600">Backtest Ghost</text>
            </>
          )}
        </g>
      </svg>

      {/* Controls Bar */}
      <div className="flex items-center justify-between px-3 sm:px-5 py-2 border-t border-slate-200 dark:border-slate-800/60 bg-slate-50/90 dark:bg-slate-950/40 text-xs gap-3 flex-wrap">
        <div className="flex items-center gap-1.5">
          <button onClick={handleZoomIn} className="px-2.5 py-1 bg-white hover:bg-slate-100 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-300 rounded-md transition-colors font-mono font-bold border border-slate-200 dark:border-slate-700">+</button>
          <button onClick={handleZoomOut} className="px-2.5 py-1 bg-white hover:bg-slate-100 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-300 rounded-md transition-colors font-mono font-bold border border-slate-200 dark:border-slate-700">-</button>
          <button onClick={handleResetView} className="px-2.5 py-1 bg-white hover:bg-slate-100 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-400 rounded-md transition-colors text-[10px] font-semibold border border-slate-200 dark:border-slate-700">RESET</button>
        </div>
        <span className="text-slate-600 dark:text-slate-500 font-mono">{visibleCandles.length}/{allValid.length} candles &middot; Drag to scroll &middot; Scroll to zoom</span>
        <div className="flex items-center gap-1.5">
          {[30, 45, 60].map((n) => (
            <button key={n} onClick={() => { setViewCount(n); setViewStart(Math.max(0, allValid.length - n)); }}
              className={`px-2 py-1 rounded-md font-mono text-[10px] font-bold transition-colors border ${viewCount === n ? 'bg-cyan-600 text-white border-cyan-600' : 'bg-white hover:bg-slate-100 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-400 border-slate-200 dark:border-slate-700'}`}>
              {n}D
            </button>
          ))}
          <button onClick={() => { setViewCount(allValid.length); setViewStart(0); }}
            className={`px-2 py-1 rounded-md font-mono text-[10px] font-bold transition-colors border ${viewCount >= allValid.length ? 'bg-cyan-600 text-white border-cyan-600' : 'bg-white hover:bg-slate-100 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-400 border-slate-200 dark:border-slate-700'}`}>
            ALL
          </button>
        </div>
      </div>
    </div>
  );
};






function Prediction() {
  const { currentUser } = useAuth();
  const { isDark } = useTheme();

  const [ticker, setTicker] = useState(() => localStorage.getItem(PREDICTION_STORAGE_KEYS.ticker) || '');
  const [days, setDays] = useState(() => normalizeForecastDays(localStorage.getItem(PREDICTION_STORAGE_KEYS.days) || 7));
  const [stockData, setStockData] = useState(() => readStoredJson(PREDICTION_STORAGE_KEYS.stockData, null));
  const currencySymbol = getDisplayCurrencySymbol(stockData?.currency);
  const [sentiment, setSentiment] = useState(() => readStoredJson(PREDICTION_STORAGE_KEYS.sentiment, null));
  const [news, setNews] = useState(() => readStoredJson(PREDICTION_STORAGE_KEYS.news, []));
  const [trainingStatus, setTrainingStatus] = useState(() => readStoredJson(PREDICTION_STORAGE_KEYS.trainingStatus, null));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [performancePeriod, setPerformancePeriod] = useState('1M');
  const [hasSearched, setHasSearched] = useState(() => localStorage.getItem(PREDICTION_STORAGE_KEYS.hasSearched) === 'true' || Boolean(localStorage.getItem(PREDICTION_STORAGE_KEYS.stockData)));
  const [visibleNewsCount, setVisibleNewsCount] = useState(3);
  const [watchlist, setWatchlist] = useState(() => readStoredJson(PREDICTION_STORAGE_KEYS.watchlist, ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL']));
  const [watchlistInput, setWatchlistInput] = useState('');
  const [watchlistSuggestions, setWatchlistSuggestions] = useState([]);
  const [showWatchlistSuggestions, setShowWatchlistSuggestions] = useState(false);
  const [watchlistRanks, setWatchlistRanks] = useState([]);
  const [watchlistLoading, setWatchlistLoading] = useState(false);
  const [watchlistError, setWatchlistError] = useState(null);
  const [showWatchlistDesk, setShowWatchlistDesk] = useState(false);
  const [agentMode, setAgentMode] = useState({
    active: false,
    symbol: '',
    step: 'idle',
    message: ''
  });
  const searchInputRef = useRef(null);
  const daysSelectRef = useRef(null);
  const predictButtonRef = useRef(null);
  const agentTimersRef = useRef([]);
  const chartTheme = {
    panel: isDark ? 'bg-[#0b1220] border-slate-700/70' : 'bg-white/95 border-gray-200',
    plot: isDark ? 'bg-slate-950/60 border-slate-800' : 'bg-slate-50 border-gray-200',
    title: isDark ? 'text-white' : 'text-gray-900',
    icon: isDark ? 'text-cyan-300' : 'text-cyan-600',
    badge: isDark ? 'text-cyan-200 bg-cyan-500/15 border-cyan-400/25' : 'text-cyan-700 bg-cyan-50 border-cyan-200',
    controls: isDark ? 'bg-slate-950/80 border-slate-800' : 'bg-gray-100 border-gray-200',
    controlActive: isDark ? 'bg-cyan-500 text-slate-950 shadow-lg' : 'bg-cyan-600 text-white shadow-lg',
    controlIdle: isDark ? 'text-slate-300 hover:bg-slate-800' : 'text-gray-600 hover:bg-gray-200',
    metric: isDark ? 'bg-slate-950/55 border-slate-800' : 'bg-gray-50 border-gray-200',
    metricLabel: isDark ? 'text-slate-500' : 'text-gray-500',
    grid: isDark ? '#1e293b' : '#e2e8f0',
    axis: isDark ? '#334155' : '#cbd5e1',
    tick: isDark ? '#94a3b8' : '#64748b',
    tooltipBg: isDark ? 'bg-slate-950/95 border-slate-700' : 'bg-white/95 border-gray-200',
    tooltipTitle: isDark ? 'text-slate-100' : 'text-gray-900',
    tooltipText: isDark ? 'text-slate-300' : 'text-gray-700',
    reference: isDark ? '#f8fafc' : '#475569',
    activeDotStroke: isDark ? '#020617' : '#ffffff',
    legend: isDark ? '#94a3b8' : '#64748b',
    positive: isDark ? 'text-emerald-300' : 'text-green-600',
    negative: isDark ? 'text-red-300' : 'text-red-600'
  };

  // Restore the most recent prediction when the user navigates away and back.
  useEffect(() => {
    localStorage.setItem(PREDICTION_STORAGE_KEYS.days, days.toString());
  }, [days]);

  useEffect(() => {
    localStorage.setItem(PREDICTION_STORAGE_KEYS.hasSearched, hasSearched ? 'true' : 'false');
  }, [hasSearched]);

  useEffect(() => {
    const activeTicker = stockData?.ticker || ticker;
    if (activeTicker) {
      localStorage.setItem(PREDICTION_STORAGE_KEYS.ticker, activeTicker);
    }
  }, [stockData?.ticker, ticker]);

  useEffect(() => {
    if (!stockData) return;
    localStorage.setItem(PREDICTION_STORAGE_KEYS.stockData, JSON.stringify(stockData));
  }, [stockData]);

  useEffect(() => {
    if (sentiment) {
      localStorage.setItem(PREDICTION_STORAGE_KEYS.sentiment, JSON.stringify(sentiment));
    } else {
      localStorage.removeItem(PREDICTION_STORAGE_KEYS.sentiment);
    }
  }, [sentiment]);

  useEffect(() => {
    localStorage.setItem(PREDICTION_STORAGE_KEYS.news, JSON.stringify(news || []));
  }, [news]);

  useEffect(() => {
    localStorage.setItem(PREDICTION_STORAGE_KEYS.watchlist, JSON.stringify(watchlist || []));
  }, [watchlist]);

  useEffect(() => {
    if (trainingStatus) {
      localStorage.setItem(PREDICTION_STORAGE_KEYS.trainingStatus, JSON.stringify(trainingStatus));
    } else {
      localStorage.removeItem(PREDICTION_STORAGE_KEYS.trainingStatus);
    }
  }, [trainingStatus]);

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
      agentTimersRef.current.forEach((timer) => clearTimeout(timer));
      agentTimersRef.current = [];
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
    setHasSearched(true);
    setVisibleNewsCount(3);
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
    const { trackStats = true, silent = false, overrideDays = days } = options;
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
        axios.get(`${API_BASE}/api/stock/${symbol}?days=${normalizeForecastDays(overrideDays)}`, { headers, timeout: 120000 }),
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
      setHasSearched(true);
      if (
        payload?.ticker &&
        payload?.requested_ticker &&
        payload.ticker !== payload.requested_ticker
      ) {
        setTicker(payload.ticker);
      } else {
        setTicker(payload?.ticker || symbol);
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
      setAgentMode((prev) => prev.active ? {
        ...prev,
        step: 'running',
        message: `${payload?.ticker || symbol} result is loaded. Review the chart, forecast range, trust score, risk, sentiment, and news.`
      } : prev);
      if (agentTimersRef.current) {
        const timer = setTimeout(() => {
          setAgentMode((prev) => prev.active ? { ...prev, active: false } : prev);
        }, 3500);
        agentTimersRef.current.push(timer);
      }

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
      setAgentMode((prev) => prev.active ? {
        ...prev,
        step: 'running',
        message: backendMessage || timeoutMessage || offlineMessage || 'The agent could not complete this prediction request.'
      } : prev);
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
    setVisibleNewsCount(3); // Reset news count on new search
  };

  useEffect(() => {
    const runAgentCommand = (rawCommand) => {
      let command = rawCommand;
      if (typeof rawCommand === 'string') {
        try {
          command = JSON.parse(rawCommand);
        } catch {
          command = null;
        }
      }
      const symbol = String(command?.symbol || '').trim().toUpperCase();
      const nextDays = normalizeForecastDays(command?.days || days);
      if (command?.type === 'watchlist') {
        const commandSymbols = [...new Set((command?.symbols || []).map((item) => String(item).trim().toUpperCase()).filter(Boolean))].slice(0, 8);
        agentTimersRef.current.forEach((timer) => clearTimeout(timer));
        agentTimersRef.current = [];
        setAgentMode({
          active: true,
          symbol: 'WATCHLIST',
          step: 'ticker',
          message: commandSymbols.length
            ? `Agent is loading ${commandSymbols.join(', ')} into Opportunity Radar.`
            : 'Agent is opening AI Watchlist Ranking - Opportunity Radar.'
        });
        if (commandSymbols.length) {
          setWatchlist(commandSymbols);
        }
        setDays(nextDays);
        const openTimer = setTimeout(() => {
          setShowWatchlistDesk(true);
          setAgentMode({
            active: true,
            symbol: 'WATCHLIST',
            step: 'predict',
            message: `Agent is clicking Rank Watchlist and loading ${nextDays}-day comparison charts.`
          });
          analyzeWatchlist(commandSymbols.length ? commandSymbols : undefined, nextDays);
        }, 800);
        const hideTimer = setTimeout(() => {
          setAgentMode((prev) => prev.active ? { ...prev, active: false } : prev);
        }, 5200);
        agentTimersRef.current.push(openTimer, hideTimer);
        return;
      }
      if (!symbol) return;

      agentTimersRef.current.forEach((timer) => clearTimeout(timer));
      agentTimersRef.current = [];
      setAgentMode({
        active: true,
        symbol,
        step: 'ticker',
        message: `Agent is entering ${symbol} in the live search box.`
      });

      const schedule = (delay, fn) => {
        const timer = setTimeout(fn, delay);
        agentTimersRef.current.push(timer);
      };

      schedule(450, () => {
        searchInputRef.current?.focus();
        setTicker('');
        setSuggestions([]);
        setShowSuggestions(false);
      });

      symbol.split('').forEach((letter, index) => {
        schedule(850 + index * 220, () => {
          setTicker(symbol.slice(0, index + 1));
          setAgentMode({
            active: true,
            symbol,
            step: 'ticker',
            message: `Agent is typing ${symbol.slice(0, index + 1)} in the live search box.`
          });
        });
      });

      const afterTyping = 1000 + symbol.length * 220;

      schedule(afterTyping + 700, () => {
        setAgentMode({
          active: true,
          symbol,
          step: 'days',
          message: `Agent selected ${nextDays === 1 ? '1 Day' : `${nextDays} Days`} forecast horizon.`
        });
        setDays(nextDays);
        daysSelectRef.current?.focus();
      });

      schedule(afterTyping + 1600, () => {
        setAgentMode({
          active: true,
          symbol,
          step: 'predict',
          message: 'Agent is clicking Predict and starting the ML workflow.'
        });
        predictButtonRef.current?.focus();
      });

      schedule(afterTyping + 2500, () => {
        setHasSearched(true);
        setVisibleNewsCount(3);
        predictButtonRef.current?.focus();
        fetchStockData(symbol, { overrideDays: nextDays });
      });

      schedule(afterTyping + 4900, () => {
        setAgentMode((prev) => ({
          ...prev,
          step: 'running',
          message: 'Agent is watching the prediction desk for live price, charts, trust score, and model status.'
        }));
      });
    };

    const savedCommand = sessionStorage.getItem(PREDICTION_STORAGE_KEYS.agentCommand);
    if (savedCommand) {
      sessionStorage.removeItem(PREDICTION_STORAGE_KEYS.agentCommand);
      runAgentCommand(savedCommand);
    } else {
      const params = new URLSearchParams(window.location.search);
      const symbol = params.get('agentStock');
      const action = params.get('agentAction');
      if (action === 'watchlist') {
        const symbols = (params.get('symbols') || '').split(',').map((item) => item.trim()).filter(Boolean);
        runAgentCommand({ type: 'watchlist', symbols, days: params.get('days') || days });
      } else if (symbol) {
        runAgentCommand({ type: 'predict', symbol, days });
      }
    }

    const handler = (event) => runAgentCommand(event.detail || {});
    window.addEventListener('datavision:agent-predict', handler);
    return () => window.removeEventListener('datavision:agent-predict', handler);
  }, []);

  const handleLoadMoreNews = () => {
    setVisibleNewsCount(prev => prev + 3);
  };

  const retryTickerTraining = async () => {
    const symbol = (stockData?.ticker || ticker || '').trim().toUpperCase();
    if (!symbol) return;
    setError(null);
    setTrainingStatus({
      ticker: symbol,
      state: 'queued',
      stage: 'queued',
      progress: 8,
      model_ready: false,
      analysis_mode: 'preliminary',
      message: 'Fresh training request accepted. Preparing a new stock-specific model.',
      estimated_completion_label: 'about 2 min'
    });
    try {
      const { data } = await axios.post(`${API_BASE}/api/model-status/${symbol}/retry`, {}, { timeout: 20000 });
      if (data?.model_status) {
        setTrainingStatus(data.model_status);
      }
      setTimeout(() => fetchStockData(symbol, { trackStats: false, silent: true }), 1200);
    } catch (err) {
      const message = err.response?.data?.error || 'Could not restart training. Please try again.';
      setError(message);
      console.error(err);
    }
  };

  const buildWatchlistRow = (payload) => {
    const recommendation = payload?.recommendation || {};
    const trust = Number(payload?.model_trust?.score ?? 0);
    const current = Number(payload?.current_price ?? 0);
    const futurePredictions = Array.isArray(payload?.future_predictions) ? payload.future_predictions : [];
    const finalPrediction = Number(futurePredictions?.[futurePredictions.length - 1]?.price ?? payload?.predicted_price ?? current);
    const lastRangeLow = Number(
      futurePredictions?.[futurePredictions.length - 1]?.range_low ??
      payload?.price_range_low?.[payload?.price_range_low?.length - 1] ??
      finalPrediction
    );
    const lastRangeHigh = Number(
      futurePredictions?.[futurePredictions.length - 1]?.range_high ??
      payload?.price_range_high?.[payload?.price_range_high?.length - 1] ??
      finalPrediction
    );
    const rawEdge = Number(recommendation.expected_move_pct);
    const edge = Number.isFinite(rawEdge)
      ? rawEdge
      : current > 0 && Number.isFinite(finalPrediction)
        ? ((finalPrediction - current) / current) * 100
        : 0;
    const rawSignal = recommendation.signal || payload?.ai_signal || 'HOLD';
    const signal = String(rawSignal).toUpperCase().includes('TRAINING') ? 'LIVE ANALYSIS' : rawSignal;
    const signalRank = getSignalRankValue(signal);
    const regime = payload?.market_regime?.label || payload?.analysis_mode || 'Tracking';
    const regimeTone = payload?.market_regime?.tone || 'neutral';
    const riskScore = clampNumber(payload?.risk_profile?.score ?? 45, 0, 100);
    const backendRisk = payload?.risk_profile?.label;
    const sentimentScore = Number((payload?.sentiment || {}).score ?? 0);
    const calibratedConfidence = Number(payload?.probability_calibration?.calibrated_confidence) * 100;
    const confidence = clampNumber(
      Number.isFinite(calibratedConfidence)
        ? Math.min(Number(recommendation.confidence_percent ?? 100), calibratedConfidence)
        : recommendation.confidence_percent ?? payload?.confidence * 100 ?? 0,
      0,
      100
    );
    const risk = Math.abs(edge) >= 4 && confidence < 60
      ? 'Speculative'
      : backendRisk || (riskScore >= 55 || regimeTone === 'warning' ? 'High' : riskScore >= 32 ? 'Medium' : 'Low');
    const modelStatus = payload?.model_status || {};
    const isReady = Boolean(payload?.prediction_ready);
    const statusState = String(modelStatus.state || '').toLowerCase();
    const progress = clampNumber(modelStatus.progress ?? (isReady ? 100 : 0), 0, 100);
    const eta = modelStatus.estimated_completion_label || formatEta(modelStatus.estimated_seconds_remaining);
    const isBuilding = !isReady && ['training', 'queued', 'initializing', 'starting'].some((state) => statusState.includes(state));
    const statusLabel = isReady
      ? 'Model Ready'
      : isBuilding
        ? 'AI Forecast Building'
        : 'Live Analysis';
    const statusMessage = isReady
      ? 'Trained ticker-specific model is ready.'
      : `${modelStatus.message || 'Live market analysis is available while the stock-specific model is prepared.'} ETA: ${eta}.`;
    const readinessPenalty = payload?.prediction_ready ? 0 : 18;
    const volatilityPenalty = riskScore / 100;
    const adjustedEdge = edge * (confidence / 100) * (1 - volatilityPenalty);
    const momentumRaw = Number(payload?.market_regime?.momentum_20d_pct ?? 0);
    const momentumScore = clampNumber(50 + momentumRaw * 3, 0, 100);
    const trendScore = clampNumber((payload?.market_regime?.trend_strength ?? 0) * 100, 0, 100);
    const sentimentComponent = clampNumber(50 + sentimentScore * 50, 0, 100);
    const expectedReturnScore = clampNumber(50 + adjustedEdge * 7, 0, 100);
    const signalComponent = clampNumber(50 + signalRank * 18, 0, 100);
    const opportunityScore = clampNumber(
      confidence * 0.25 +
      expectedReturnScore * 0.20 +
      momentumScore * 0.20 +
      sentimentComponent * 0.15 +
      trendScore * 0.10 +
      signalComponent * 0.10 -
      riskScore * 0.10 -
      readinessPenalty,
      0,
      100
    );

    return {
      ticker: payload?.ticker || payload?.requested_ticker,
      company: payload?.company_name || payload?.ticker,
      currency: payload?.currency || 'USD',
      signal,
      trust,
      edge,
      adjustedEdge,
      regime,
      regimeTone,
      risk,
      riskScore,
      confidence,
      sentimentScore,
      momentumScore,
      trendScore,
      price: current,
      predictedPrice: finalPrediction,
      rangeLow: Number.isFinite(lastRangeLow) ? lastRangeLow : finalPrediction,
      rangeHigh: Number.isFinite(lastRangeHigh) ? lastRangeHigh : finalPrediction,
      rangeWidthPct: current > 0 && Number.isFinite(lastRangeLow) && Number.isFinite(lastRangeHigh)
        ? (Math.abs(lastRangeHigh - lastRangeLow) / current) * 100
        : 0,
      returnPotential: edge,
      opportunityScore,
      horizon: payload?.prediction_horizon?.label || `${payload?.days_predicted || 7} Trading Days`,
      backtest: payload?.backtest_metrics,
      freshness: payload?.model_freshness,
      calibration: payload?.probability_calibration,
      ready: isReady,
      progress,
      eta,
      stage: modelStatus.stage || (isReady ? 'custom_model_ready' : 'live_analysis'),
      statusLabel,
      statusMessage,
      reason: Array.isArray(payload?.ai_explanation) && payload.ai_explanation.length > 0
        ? payload.ai_explanation[0]
        : Array.isArray(recommendation.reasons)
          ? recommendation.reasons[0]
          : payload?.model_status?.message
    };
  };

  const analyzeWatchlist = async (overrideSymbols, overrideDays = 7) => {
    const sourceSymbols = overrideSymbols && overrideSymbols.length ? overrideSymbols : watchlist;
    const symbols = [...new Set((sourceSymbols || []).map((item) => item.trim().toUpperCase()).filter(Boolean))].slice(0, 8);
    if (symbols.length === 0) return;
    setWatchlistLoading(true);
    setWatchlistError(null);

    const headers = {};
    if (currentUser) {
      try {
        headers.Authorization = `Bearer ${await currentUser.getIdToken()}`;
      } catch (err) {
        console.error('Failed to get token for watchlist ranking', err);
      }
    }

    try {
      const results = await Promise.all(symbols.map(async (symbol) => {
        try {
          const { data } = await axios.get(`${API_BASE}/api/stock/${symbol}?days=${normalizeForecastDays(overrideDays)}`, { headers, timeout: 120000 });
          return buildWatchlistRow(data);
        } catch (err) {
          return {
            ticker: symbol,
            company: symbol,
            currency: 'USD',
            signal: 'UNAVAILABLE',
            trust: 0,
            edge: 0,
            adjustedEdge: 0,
            regime: 'Unavailable',
            regimeTone: 'neutral',
            risk: 'High',
            riskScore: 100,
            confidence: 0,
            sentimentScore: 0,
            momentumScore: 0,
            trendScore: 0,
            price: 0,
            predictedPrice: 0,
            rangeLow: 0,
            rangeHigh: 0,
            opportunityScore: 0,
            horizon: '7 Trading Days',
            statusLabel: 'Unavailable',
            statusMessage: 'Prediction service unavailable.',
            ready: false,
            reason: err.response?.data?.error || 'Prediction service unavailable'
          };
        }
      }));
      setWatchlistRanks(results.sort((a, b) => b.opportunityScore - a.opportunityScore));
    } catch (err) {
      setWatchlistError('Unable to rank watchlist right now.');
    } finally {
      setWatchlistLoading(false);
    }
  };

  const handleWatchlistInputChange = async (value) => {
    const formatted = value.toUpperCase();
    setWatchlistInput(formatted);
    const query = formatted.trim();
    if (!query) {
      setWatchlistSuggestions([]);
      setShowWatchlistSuggestions(false);
      return;
    }
    try {
      const { data } = await axios.get(`${API_BASE}/api/search`, {
        params: { q: query, limit: 6 }
      });
      const results = Array.isArray(data?.results) ? data.results : [];
      setWatchlistSuggestions(results);
      setShowWatchlistSuggestions(results.length > 0);
    } catch (err) {
      setWatchlistSuggestions([]);
      setShowWatchlistSuggestions(false);
    }
  };

  const handleAddWatchlistSymbol = () => {
    const symbol = watchlistInput.trim().toUpperCase();
    if (!symbol) return;
    setWatchlist((prev) => [...new Set([...(prev || []), symbol])].slice(0, 8));
    setWatchlistInput('');
    setWatchlistSuggestions([]);
    setShowWatchlistSuggestions(false);
  };

  const handleRemoveWatchlistSymbol = (symbol) => {
    setWatchlist((prev) => (prev || []).filter((item) => item !== symbol));
    setWatchlistRanks((prev) => (prev || []).filter((item) => item.ticker !== symbol));
  };

  const handleDaysChange = (value) => {
    const nextDays = normalizeForecastDays(value);
    setDays(nextDays);
    if (stockData?.ticker) {
      fetchStockData(stockData.ticker, { trackStats: false, silent: true, overrideDays: nextDays });
    }
  };

  const computePredictionRows = () => {
    if (!stockData?.future_predictions || !Array.isArray(stockData.future_predictions)) {
      return [];
    }

    const rangeLow = stockData.price_range_low || [];
    const rangeHigh = stockData.price_range_high || [];
    const rangeQ25 = stockData.price_range_q25 || [];
    const rangeQ75 = stockData.price_range_q75 || [];

    return stockData.future_predictions.map((entry, index) => {
      const price = Number(entry?.price ?? NaN);
      const baseline = Number(stockData.current_price ?? NaN);
      if (!Number.isFinite(price) || !Number.isFinite(baseline) || baseline <= 0) {
        return null;
      }

      const minMove = Number(entry?.min_required_move_pct ?? stockData?.recommendation?.min_required_move_pct ?? 0.35);
      const confidence = Number(entry?.confidence ?? stockData?.recommendation?.confidence ?? 0);
      let signal = entry?.signal || 'HOLD';

      const lowNumber = rangeLow[index] != null ? Number(rangeLow[index]) : Number(entry?.range_low ?? NaN);
      const highNumber = rangeHigh[index] != null ? Number(rangeHigh[index]) : Number(entry?.range_high ?? NaN);
      const hasRange = Number.isFinite(lowNumber) && Number.isFinite(highNumber);
      const priceLowNumber = hasRange ? Math.min(lowNumber, highNumber, price) : NaN;
      const priceHighNumber = hasRange ? Math.max(lowNumber, highNumber, price) : NaN;
      const midpoint = hasRange ? (priceLowNumber + priceHighNumber) / 2 : price;
      const midpointChangePercent = ((midpoint - baseline) / baseline) * 100;

      if (!entry?.signal) {
        const finalSignal = stockData?.recommendation?.signal || stockData?.ai_signal || 'HOLD';
        const finalScore = Number(stockData?.recommendation?.score ?? 0);
        if (confidence >= 0.38 && Math.abs(midpointChangePercent) >= minMove) {
          if (midpointChangePercent > 0 && finalScore > 0 && finalSignal.includes('BUY')) {
            signal = confidence >= 0.7 && midpointChangePercent >= minMove * 2.5 ? 'STRONG BUY' : 'BUY';
          } else if (midpointChangePercent < 0 && finalScore < 0 && finalSignal.includes('SELL')) {
            signal = confidence >= 0.7 && Math.abs(midpointChangePercent) >= minMove * 2.5 ? 'STRONG SELL' : 'SELL';
          }
        }
      }

      // Price range from quantile regression
      const priceLow = hasRange ? priceLowNumber.toFixed(2) : null;
      const priceHigh = hasRange ? priceHighNumber.toFixed(2) : null;
      const priceQ25 = rangeQ25[index] != null ? Number(rangeQ25[index]).toFixed(2) : null;
      const priceQ75 = rangeQ75[index] != null ? Number(rangeQ75[index]).toFixed(2) : null;
      const changeLow = hasRange ? priceLowNumber - baseline : price - baseline;
      const changeHigh = hasRange ? priceHighNumber - baseline : price - baseline;
      const changePctLow = (changeLow / baseline) * 100;
      const changePctHigh = (changeHigh / baseline) * 100;

      return {
        id: `${entry.date}-${index}`,
        dateLabel: new Date(entry.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
        priceLow,
        priceHigh,
        priceQ25,
        priceQ75,
        changeLow: changeLow.toFixed(2),
        changeHigh: changeHigh.toFixed(2),
        changePctLow: changePctLow.toFixed(2),
        changePctHigh: changePctHigh.toFixed(2),
        signal
      };
    }).filter(Boolean);
  };

  const predictionRows = computePredictionRows();
  const selectedRangeIndex = stockData?.price_range_low?.length
    ? Math.min(days === 1 ? 0 : 4, stockData.price_range_low.length - 1)
    : -1;
  const selectedRangeLow = selectedRangeIndex >= 0 ? Number(stockData.price_range_low?.[selectedRangeIndex]) : NaN;
  const selectedRangeHigh = selectedRangeIndex >= 0 ? Number(stockData.price_range_high?.[selectedRangeIndex]) : NaN;
  const hasSelectedRange = Number.isFinite(selectedRangeLow) && Number.isFinite(selectedRangeHigh) && Number.isFinite(Number(stockData?.current_price));
  const selectedRangeChangeLow = hasSelectedRange ? Math.min(selectedRangeLow, selectedRangeHigh) - Number(stockData.current_price) : NaN;
  const selectedRangeChangeHigh = hasSelectedRange ? Math.max(selectedRangeLow, selectedRangeHigh) - Number(stockData.current_price) : NaN;
  const selectedRangePctLow = hasSelectedRange ? (selectedRangeChangeLow / Number(stockData.current_price)) * 100 : NaN;
  const selectedRangePctHigh = hasSelectedRange ? (selectedRangeChangeHigh / Number(stockData.current_price)) * 100 : NaN;
  const finalRecommendation = stockData?.recommendation || {};
  const headlineSignal = finalRecommendation.signal || stockData?.ai_signal || 'HOLD';
  const headlineStance = finalRecommendation.stance || headlineSignal;
  const signalIsBullish = headlineSignal.includes('BUY');
  const signalIsBearish = headlineSignal.includes('SELL');
  const selectedRangeMin = hasSelectedRange ? Math.min(selectedRangeLow, selectedRangeHigh) : NaN;
  const selectedRangeMax = hasSelectedRange ? Math.max(selectedRangeLow, selectedRangeHigh) : NaN;
  const currentForRange = Number(stockData?.current_price);
  const rangeCrossesCurrent = hasSelectedRange && selectedRangeMin <= currentForRange && selectedRangeMax >= currentForRange;
  const rangeBiasLabel = !hasSelectedRange
    ? 'Range pending'
    : rangeCrossesCurrent
      ? signalIsBullish
        ? 'Bullish bias, two-way range'
        : signalIsBearish
          ? 'Bearish bias, two-way range'
          : 'Two-way range'
      : selectedRangeMin > currentForRange
        ? 'Range fully above market'
        : selectedRangeMax < currentForRange
          ? 'Range fully below market'
          : 'Range mixed';
  const rangeRiskText = !hasSelectedRange
    ? 'Waiting for calibrated forecast interval.'
    : rangeCrossesCurrent
      ? 'Forecast interval crosses the current price, so risk exists on both sides.'
      : selectedRangeMin > currentForRange
        ? 'Forecast interval is entirely above the current price.'
        : 'Forecast interval is entirely below the current price.';
  const modelQualityScore = Number(finalRecommendation.components?.model_quality ?? NaN);
  const modelQualityLabel = Number.isFinite(modelQualityScore)
    ? modelQualityScore >= 0.8
      ? 'Strong validation'
      : modelQualityScore >= 0.6
        ? 'Usable validation'
        : 'Weak validation'
    : 'Validation tracked';
  const modelQualityText = Number.isFinite(modelQualityScore)
    ? `Model quality weight ${(modelQualityScore * 100).toFixed(0)}%, uncertainty ${finalRecommendation.uncertainty || 'n/a'}.`
    : `Uncertainty ${finalRecommendation.uncertainty || 'tracked'} from the decision engine.`;
  const rsiValue = Number(stockData?.indicators?.rsi);
  const rsiState = Number.isFinite(rsiValue)
    ? rsiValue >= 70
      ? 'Overbought'
      : rsiValue <= 30
        ? 'Oversold'
        : rsiValue >= 55
          ? 'Bullish momentum'
          : rsiValue <= 45
            ? 'Bearish momentum'
            : 'Neutral momentum'
    : 'Waiting for RSI';
  const emaValue = Number(stockData?.indicators?.ema);
  const currentPriceNumber = Number(stockData?.current_price);
  const emaState = Number.isFinite(emaValue) && Number.isFinite(currentPriceNumber)
    ? currentPriceNumber >= emaValue
      ? 'Price above EMA'
      : 'Price below EMA'
    : 'Waiting for EMA';
  const macdValue = Number(stockData?.indicators?.macd);
  const macdHistogram = Number(stockData?.indicators?.macd_histogram);
  const macdState = Number.isFinite(macdHistogram)
    ? macdHistogram > 0
      ? 'Positive momentum'
      : macdHistogram < 0
        ? 'Negative momentum'
        : 'Momentum flat'
    : Number.isFinite(macdValue)
      ? 'MACD available'
      : 'Waiting for MACD';
  const atrValue = Number(stockData?.indicators?.atr);
  const planBuffer = Number.isFinite(atrValue) && atrValue > 0
    ? atrValue * 0.25
    : Number.isFinite(currentPriceNumber)
      ? currentPriceNumber * 0.006
      : 0;
  const tradePlan = Number.isFinite(currentPriceNumber) && currentPriceNumber > 0 && headlineSignal !== 'HOLD'
    ? (() => {
        const bullish = headlineSignal.includes('BUY');
        const firstPrediction = Number(stockData?.future_predictions?.[0]?.price);
        const finalPrediction = Number(stockData?.future_predictions?.[stockData?.future_predictions?.length - 1]?.price);
        const rangeTarget = bullish ? selectedRangeMax : selectedRangeMin;
        const targetBase = Number.isFinite(rangeTarget)
          ? rangeTarget
          : Number.isFinite(finalPrediction)
            ? finalPrediction
            : firstPrediction;
        const stopDistance = Number.isFinite(atrValue) && atrValue > 0
          ? atrValue * 1.15
          : currentPriceNumber * 0.018;
        const target1 = Number.isFinite(targetBase)
          ? targetBase
          : bullish
            ? currentPriceNumber + stopDistance * 1.6
            : currentPriceNumber - stopDistance * 1.6;
        const target2 = bullish
          ? Math.max(target1, currentPriceNumber + Math.abs(target1 - currentPriceNumber) * 1.45)
          : Math.min(target1, currentPriceNumber - Math.abs(target1 - currentPriceNumber) * 1.45);

        return {
          direction: bullish ? 'Long' : 'Short',
          entryLow: bullish ? currentPriceNumber - planBuffer : currentPriceNumber,
          entryHigh: bullish ? currentPriceNumber : currentPriceNumber + planBuffer,
          stop: bullish ? currentPriceNumber - stopDistance : currentPriceNumber + stopDistance,
          target1,
          target2
        };
      })()
    : null;
  const backtestGhost = Array.isArray(stockData?.backtest_ghost)
    ? stockData.backtest_ghost
    : Array.isArray(stockData?.backtest_chart?.dates)
      ? stockData.backtest_chart.dates.map((date, index) => ({
          date,
          predicted: stockData.backtest_chart.predicted?.[index],
          actual: stockData.backtest_chart.actual?.[index]
        }))
      : [];
  const modelTrust = stockData?.model_trust || {};
  const trustScore = Number(modelTrust.score);
  const trustLabel = modelTrust.label || 'Trust pending';
  const marketRegime = stockData?.market_regime || {};
  const reliability = stockData?.reliability || {};
  const reliabilityStatus = String(reliability.status || 'tracking').replace(/_/g, ' ');
  const reliabilityEvaluated = Number(reliability.evaluated_count ?? 0);
  const reliabilityMinimum = Number(reliability.min_evaluations ?? 5);
  const reliabilityProgress = Math.max(0, Math.min(100, reliabilityMinimum > 0 ? (reliabilityEvaluated / reliabilityMinimum) * 100 : 0));
  const reliabilityWarming = ['monitoring_warming_up', 'insufficient_evaluations', 'no_data', 'tracking'].includes(String(reliability.status || 'tracking'));
  const riskProfile = stockData?.risk_profile || {};
  const backtestMetrics = stockData?.backtest_metrics || {};
  const predictionHorizon = stockData?.prediction_horizon || {};
  const aiExplanation = Array.isArray(stockData?.ai_explanation) ? stockData.ai_explanation : [];
  const probabilityCalibration = stockData?.probability_calibration || {};
  const activeModelStatus = trainingStatus || stockData?.model_status || {};
  const predictionIsReady = Boolean(
    activeModelStatus.model_ready &&
    stockData?.prediction_ready &&
    Array.isArray(stockData?.future_predictions) &&
    stockData.future_predictions.length > 0
  );
  const isPreliminaryMode = activeModelStatus.analysis_mode === 'preliminary' || !predictionIsReady;
  const trainingProgress = Number(activeModelStatus.progress ?? 0);
  const trainingBlocked = ['untrainable', 'unsupported'].includes(String(activeModelStatus.state || '').toLowerCase());
  const trainingStateLabel = predictionIsReady
    ? 'Prediction Ready'
    : activeModelStatus.model_ready
      ? 'Preparing Prediction'
      : trainingBlocked
        ? 'Forecast Unavailable'
      : activeModelStatus.state === 'training'
        ? 'AI Forecast Building'
        : activeModelStatus.state === 'queued'
          ? 'Queued for AI Forecast'
          : activeModelStatus.state === 'failed'
            ? 'Training Retry Available'
            : 'Starting AI Forecast';
  const trainingStatusMessage = predictionIsReady
    ? 'Real prediction data is ready.'
    : 'Please stay on this screen. The stock-specific AI forecast is being prepared and will refresh automatically when the range is ready.';
  const trainingEtaLabel = activeModelStatus.estimated_completion_label || formatEta(activeModelStatus.estimated_seconds_remaining);
  const liveTrendData = stockData?.technical_chart?.candles?.slice(-30).map((candle) => ({
    date: parseDateLocal(candle.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    price: Number(candle.close),
    volume: Number(candle.volume || 0),
  })) || [];
  const liveRangeData = stockData?.technical_chart?.candles?.slice(-30).map((candle) => ({
    date: parseDateLocal(candle.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    high: Number(candle.high),
    low: Number(candle.low),
    close: Number(candle.close),
  })) || [];

  // Empty state - show beautiful placeholder when no stock is selected (FIRST VISIT)
  if (!hasSearched && !loading && !stockData && !error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-cyan-50 dark:from-gray-900 dark:via-gray-900 dark:to-gray-800 py-8 sm:py-10">
        <div className="max-w-[1500px] mx-auto px-4 sm:px-6 lg:px-8">
          {/* Main Content */}
          <div className="text-center mb-8 relative z-10">
            <div className="mb-5">
              <div className="inline-flex items-center justify-center w-20 h-20 sm:w-24 sm:h-24 rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-500 shadow-2xl mb-4 transform hover:scale-105 transition-transform duration-500">
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
          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-2xl shadow-2xl p-5 sm:p-7 border-2 border-white/50 dark:border-gray-700/50 mb-6 relative z-50">
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
                    onChange={(e) => handleDaysChange(e.target.value)}
                    className="w-full pl-14 pr-10 py-5 text-lg border-2 border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-gray-900/50 text-gray-900 dark:text-white rounded-2xl focus:ring-4 focus:ring-cyan-500/20 focus:border-cyan-500 shadow-inner appearance-none cursor-pointer transition-all"
                  >
                    {FORECAST_HORIZONS.map((horizon) => (
                      <option key={horizon} value={horizon}>
                        {horizon === 1 ? '1 Day Forecast' : `${horizon} Days Forecast`}
                      </option>
                    ))}
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

          <div className="bg-white/85 dark:bg-[#0b1220]/90 backdrop-blur-xl rounded-2xl shadow-xl p-4 sm:p-5 border border-cyan-100 dark:border-slate-800 mb-8 relative z-20 text-left">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
              <div>
                <p className="text-xs font-black uppercase tracking-[0.18em] text-cyan-600 dark:text-cyan-400">AI Watchlist Ranking</p>
                <h3 className="text-xl sm:text-2xl font-black text-slate-950 dark:text-white mt-1">Opportunity Radar</h3>
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                  Rank a watchlist before predicting a single stock.
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-2 w-full lg:w-auto">
                <input
                  value={watchlistInput}
                  onChange={(e) => handleWatchlistInputChange(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleAddWatchlistSymbol();
                    }
                  }}
                  placeholder="Add ticker, e.g. GOOGL"
                  className="w-full sm:w-56 px-3 py-2 border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 text-slate-900 dark:text-white rounded-md text-sm focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                />
                <button
                  type="button"
                  onClick={handleAddWatchlistSymbol}
                  className="px-4 py-2 rounded-md border border-slate-300 dark:border-slate-700 text-sm font-bold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-900"
                >
                  Add
                </button>
                <button
                  type="button"
                  onClick={analyzeWatchlist}
                  disabled={watchlistLoading || watchlist.length === 0}
                  className="px-5 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-400 text-white text-sm font-black shadow-lg hover:shadow-cyan-500/20"
                >
                  {watchlistLoading ? 'Ranking...' : 'Rank Watchlist'}
                </button>
              </div>
            </div>

            <div className="flex gap-2 flex-wrap mt-4">
              {watchlist.map((symbol) => (
                <span key={symbol} className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-xs font-black text-slate-700 dark:text-slate-200">
                  {symbol}
                  <button type="button" onClick={() => handleRemoveWatchlistSymbol(symbol)} className="text-slate-400 hover:text-red-500">x</button>
                </span>
              ))}
            </div>

            {watchlistRanks.length > 0 && (
              <div className="mt-4 space-y-4">
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  {[
                    ['Top setup', watchlistRanks[0]?.ticker || 'N/A', `${watchlistRanks[0]?.opportunityScore?.toFixed?.(0) || 0}/100`],
                    ['Avg trust', `${(watchlistRanks.reduce((sum, row) => sum + Number(row.trust || 0), 0) / Math.max(1, watchlistRanks.length)).toFixed(0)}/100`, 'model quality'],
                    ['Ready ranges', `${watchlistRanks.filter((row) => row.ready).length}/${watchlistRanks.length}`, 'AI targets'],
                    ['High risk', `${watchlistRanks.filter((row) => row.risk === 'High' || row.risk === 'Speculative').length}`, 'watch carefully']
                  ].map(([label, value, sub]) => (
                    <div key={label} className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                      <p className="text-[10px] font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
                      <p className="mt-1 text-lg font-black text-slate-950 dark:text-white">{value}</p>
                      <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400">{sub}</p>
                    </div>
                  ))}
                </div>
                <div className="rounded-lg border border-slate-200 dark:border-slate-800 overflow-x-auto bg-white dark:bg-[#0b1220]">
                  <table className="min-w-full text-left text-sm">
                    <thead className="bg-slate-50 dark:bg-slate-950 text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <tr>
                        <th className="px-3 py-3">Rank</th>
                        <th className="px-3 py-3">Stock</th>
                        <th className="px-3 py-3">Live Price</th>
                        <th className="px-3 py-3">Predicted Range</th>
                        <th className="px-3 py-3">Signal</th>
                        <th className="px-3 py-3">Trust</th>
                        <th className="px-3 py-3">Adj Edge</th>
                        <th className="px-3 py-3">Regime</th>
                        <th className="px-3 py-3">Risk</th>
                        <th className="px-3 py-3">Score</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                      {watchlistRanks.map((row, index) => (
                        <tr key={row.ticker}>
                          <td className="px-3 py-3 font-black text-slate-500">#{index + 1}</td>
                          <td className="px-3 py-3">
                            <button
                              type="button"
                              onClick={() => {
                                setTicker(row.ticker);
                                setHasSearched(true);
                                fetchStockData(row.ticker);
                              }}
                              className="font-black text-slate-950 dark:text-white hover:text-cyan-600 dark:hover:text-cyan-400"
                            >
                              {row.ticker}
                            </button>
                            <p className="text-xs text-slate-500 truncate max-w-[170px]">{row.company}</p>
                          </td>
                          <td className="px-3 py-3 font-black text-slate-900 dark:text-white whitespace-nowrap">
                            {formatMoney(row.price, row.currency)}
                          </td>
                          <td className="px-3 py-3 whitespace-nowrap">
                            <p className="text-xs font-black text-slate-900 dark:text-white">
                              {row.ready
                                ? `${formatMoney(Math.min(row.rangeLow, row.rangeHigh), row.currency)} - ${formatMoney(Math.max(row.rangeLow, row.rangeHigh), row.currency)}`
                                : 'Preparing AI range'}
                            </p>
                            <p className="text-[10px] font-semibold text-slate-400">target {row.ready ? formatMoney(row.predictedPrice, row.currency) : row.eta}</p>
                          </td>
                          <td className="px-3 py-3">
                            <span className={`px-2.5 py-1 rounded-md text-xs font-black ${row.signal.includes('BUY')
                              ? 'bg-green-100 text-green-700 dark:bg-green-500/15 dark:text-green-300'
                              : row.signal.includes('SELL')
                                ? 'bg-red-100 text-red-700 dark:bg-red-500/15 dark:text-red-300'
                                : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'
                              }`}>
                              {row.signal}
                            </span>
                          </td>
                          <td className="px-3 py-3 font-black text-slate-900 dark:text-white whitespace-nowrap">{row.trust}/100</td>
                          <td className={`px-3 py-3 font-black ${row.adjustedEdge >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                            {row.adjustedEdge >= 0 ? '+' : ''}{row.adjustedEdge.toFixed(2)}%
                            <p className="text-[10px] font-semibold text-slate-400">raw {row.edge >= 0 ? '+' : ''}{row.edge.toFixed(2)}%</p>
                          </td>
                          <td className="px-3 py-3 text-xs font-bold text-slate-700 dark:text-slate-300">{row.regime}</td>
                          <td className={`px-3 py-3 text-xs font-black ${row.risk === 'High' || row.risk === 'Speculative'
                            ? 'text-red-600 dark:text-red-400'
                            : row.risk === 'Medium'
                              ? 'text-amber-600 dark:text-amber-400'
                              : 'text-green-600 dark:text-green-400'
                            }`}>
                            {row.risk}
                            <p className="text-[10px] font-semibold text-slate-400">{row.riskScore.toFixed(0)}/100</p>
                          </td>
                          <td className="px-3 py-3 font-black text-cyan-700 dark:text-cyan-300">{row.opportunityScore.toFixed(0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  <div className="h-52 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Opportunity Score</p>
                      <p className="text-xs text-slate-500">Best setups first</p>
                    </div>
                    <ResponsiveContainer width="100%" height="85%">
                      <BarChart data={watchlistRanks.slice(0, 5)} layout="vertical" margin={{ top: 8, right: 18, left: 12, bottom: 8 }}>
                        <CartesianGrid stroke={chartTheme.grid} horizontal={false} />
                        <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} />
                        <YAxis type="category" dataKey="ticker" tick={{ fontSize: 12, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} width={72} />
                        <Bar dataKey="opportunityScore" radius={[0, 6, 6, 0]} fill="#06b6d4" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="h-52 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Live Price vs AI Target</p>
                      <p className="text-xs text-slate-500">Forecast comparison</p>
                    </div>
                    <ResponsiveContainer width="100%" height="85%">
                      <ComposedChart data={watchlistRanks.slice(0, 5)} margin={{ top: 8, right: 18, left: 4, bottom: 8 }}>
                        <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                        <XAxis dataKey="ticker" tick={{ fontSize: 11, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={54} />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const row = payload[0].payload;
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                <p className={chartTheme.tooltipText}>Live: {formatMoney(row.price, row.currency)}</p>
                                <p className={chartTheme.tooltipText}>Target: {row.ready ? formatMoney(row.predictedPrice, row.currency) : 'Waiting for AI range'}</p>
                                <p className={chartTheme.tooltipText}>Status: {row.ready ? 'Forecast ready' : `Live view, ETA ${row.eta}`}</p>
                              </div>
                            );
                          }}
                        />
                        <Bar dataKey="price" fill="#38bdf8" opacity={0.28} radius={[4, 4, 0, 0]} />
                        <Line type="monotone" dataKey="predictedPrice" stroke="#10b981" strokeWidth={2.2} dot={{ r: 3 }} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="h-52 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Trust, Risk, Edge</p>
                      <p className="text-xs text-slate-500">Decision quality</p>
                    </div>
                    <ResponsiveContainer width="100%" height="85%">
                      <ComposedChart data={watchlistRanks.slice(0, 5)} margin={{ top: 8, right: 18, left: 4, bottom: 8 }}>
                        <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                        <XAxis dataKey="ticker" tick={{ fontSize: 11, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} />
                        <YAxis yAxisId="score" domain={[0, 100]} tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={42} />
                        <YAxis yAxisId="edge" orientation="right" tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={42} />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const row = payload[0].payload;
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                <p className={chartTheme.tooltipText}>Trust: {row.trust}/100</p>
                                <p className={chartTheme.tooltipText}>Risk: {row.riskScore.toFixed(0)}/100</p>
                                <p className={row.adjustedEdge >= 0 ? chartTheme.positive : chartTheme.negative}>Adj edge: {row.adjustedEdge >= 0 ? '+' : ''}{row.adjustedEdge.toFixed(2)}%</p>
                              </div>
                            );
                          }}
                        />
                        <Bar yAxisId="score" dataKey="riskScore" fill="#ef4444" opacity={0.22} radius={[4, 4, 0, 0]} />
                        <Line yAxisId="score" type="monotone" dataKey="trust" stroke="#38bdf8" strokeWidth={2.2} dot={{ r: 3 }} />
                        <Line yAxisId="edge" type="monotone" dataKey="adjustedEdge" stroke="#10b981" strokeWidth={2.2} dot={{ r: 3 }} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="h-52 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Forecast Range Width</p>
                      <p className="text-xs text-slate-500">Lower is tighter</p>
                    </div>
                    <ResponsiveContainer width="100%" height="85%">
                      <BarChart data={watchlistRanks.slice(0, 5)} margin={{ top: 8, right: 18, left: 4, bottom: 8 }}>
                        <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                        <XAxis dataKey="ticker" tick={{ fontSize: 11, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} />
                        <YAxis tickFormatter={(val) => `${Number(val).toFixed(0)}%`} tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={42} />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const row = payload[0].payload;
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                <p className={chartTheme.tooltipText}>Range width: {row.rangeWidthPct.toFixed(2)}%</p>
                                <p className={chartTheme.tooltipText}>Range: {row.ready ? `${formatMoney(Math.min(row.rangeLow, row.rangeHigh), row.currency)} - ${formatMoney(Math.max(row.rangeLow, row.rangeHigh), row.currency)}` : 'Preparing AI range'}</p>
                              </div>
                            );
                          }}
                        />
                        <Bar dataKey="rangeWidthPct" fill="#f59e0b" opacity={0.72} radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="h-52 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Return Potential</p>
                      <p className="text-xs text-slate-500">Raw vs adjusted</p>
                    </div>
                    <ResponsiveContainer width="100%" height="85%">
                      <ComposedChart data={watchlistRanks.slice(0, 5)} margin={{ top: 8, right: 18, left: 4, bottom: 8 }}>
                        <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                        <XAxis dataKey="ticker" tick={{ fontSize: 11, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} />
                        <YAxis tickFormatter={(val) => `${Number(val).toFixed(0)}%`} tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={42} />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const row = payload[0].payload;
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                <p className={row.edge >= 0 ? chartTheme.positive : chartTheme.negative}>Raw edge: {row.edge >= 0 ? '+' : ''}{row.edge.toFixed(2)}%</p>
                                <p className={row.adjustedEdge >= 0 ? chartTheme.positive : chartTheme.negative}>Adjusted edge: {row.adjustedEdge >= 0 ? '+' : ''}{row.adjustedEdge.toFixed(2)}%</p>
                              </div>
                            );
                          }}
                        />
                        <Bar dataKey="returnPotential" fill="#94a3b8" opacity={0.45} radius={[4, 4, 0, 0]} />
                        <Line type="monotone" dataKey="adjustedEdge" stroke="#10b981" strokeWidth={2.2} dot={{ r: 3 }} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="h-52 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Signal Mix</p>
                      <p className="text-xs text-slate-500">Desk exposure</p>
                    </div>
                    <div className="space-y-4">
                      {[
                        ['Buy', watchlistRanks.filter((row) => row.signal.includes('BUY')).length, 'bg-green-500'],
                        ['Hold', watchlistRanks.filter((row) => row.signal.includes('HOLD')).length, 'bg-slate-500'],
                        ['Sell', watchlistRanks.filter((row) => row.signal.includes('SELL')).length, 'bg-red-500'],
                        ['Speculative', watchlistRanks.filter((row) => row.risk === 'Speculative' || row.risk === 'High').length, 'bg-amber-500']
                      ].map(([label, count, colorClass]) => {
                        const pct = Math.round((Number(count) / Math.max(1, watchlistRanks.length)) * 100);
                        return (
                          <div key={label}>
                            <div className="flex items-center justify-between text-xs font-bold text-slate-600 dark:text-slate-300 mb-1">
                              <span>{label}</span>
                              <span>{count} stocks</span>
                            </div>
                            <div className="h-2.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                              <div className={`h-full rounded-full ${colorClass}`} style={{ width: `${pct}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Quick Access Stocks */}
          <div className="text-center relative z-10">
            <p className="text-sm font-bold text-gray-400 dark:text-gray-500 mb-6 uppercase tracking-[0.2em]">Popular Market Symbols</p>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 sm:gap-6">
              {['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA'].map((symbol) => (
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
                  <p className="text-xs font-bold text-gray-400 dark:text-gray-500 mt-2 relative z-10 uppercase tracking-widest">Predict</p>
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

  if (stockData && !predictionIsReady) {
    const statusState = String(activeModelStatus.state || '').toLowerCase();
    const statusStage = String(activeModelStatus.stage || '').toLowerCase();
    const trainingFailed = statusState === 'failed';
    const canRetryTraining = trainingFailed && !trainingBlocked;
    const waitingProgress = trainingFailed || trainingBlocked ? 0 : Math.max(6, Math.min(100, Number(trainingProgress) || 0));
    const waitingEta = trainingBlocked ? 'Choose a listed ticker' : trainingFailed ? 'Retry available' : (trainingEtaLabel && trainingEtaLabel !== 'Ready now' ? trainingEtaLabel : 'about 2 min');
    const waitingTitle = trainingBlocked
      ? `${stockData.ticker} cannot be trained yet`
      : trainingFailed
      ? `${stockData.ticker} training did not finish`
      : `${stockData.ticker} model is being prepared`;
    const waitingCopy = trainingBlocked
      ? (activeModelStatus.message || 'Datavision needs a public exchange-listed ticker with enough daily OHLCV history. Search and choose the listed market symbol to run AI forecasting.')
      : trainingFailed
      ? 'The first training attempt stopped before validation completed. Retry training and Datavision will only show prediction prices after the model passes validation.'
      : `Please wait ${waitingEta}. We are training and validating this stock model before showing forecast prices, ranges, trust score, and trading signals.`;
    const stageDisplay = trainingBlocked
      ? 'Symbol cannot be trained from available market data'
      : trainingFailed
      ? 'Training stopped before validation'
      : statusStage.includes('market_data')
      ? 'Collecting market history'
      : statusStage.includes('unified') || statusStage.includes('lstm')
      ? 'Training forecast model'
      : statusStage.includes('validation') || statusStage.includes('observability') || statusStage.includes('calibration')
      ? 'Validating and publishing result'
      : statusState === 'queued'
      ? 'Waiting to start model training'
      : 'Preparing stock-specific forecast';
    return (
      <div className={`min-h-screen flex items-center justify-center px-4 py-10 relative overflow-hidden ${isDark ? 'bg-slate-950 text-white' : 'bg-gradient-to-br from-slate-50 via-cyan-50 to-blue-50 text-slate-950'}`}>
        <div className="absolute inset-0 opacity-35 pointer-events-none">
          <CinematicBackground />
        </div>
        <main className="relative z-10 w-full max-w-5xl">
          <div className={`w-full rounded-3xl border p-6 sm:p-10 ${isDark ? 'border-cyan-400/25 bg-slate-950/90 shadow-2xl shadow-cyan-950/40' : 'border-cyan-200 bg-white/92 shadow-2xl shadow-cyan-100/70'}`}>
            <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6">
              <div>
                <p className={`text-xs font-black uppercase tracking-[0.22em] ${isDark ? 'text-cyan-300' : 'text-cyan-700'}`}>Stock-Specific AI Model Training</p>
                <h2 className="mt-3 text-3xl sm:text-5xl font-black tracking-tight">
                  {waitingTitle}
                </h2>
                <p className={`mt-3 text-base sm:text-lg max-w-2xl leading-relaxed ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>
                  {waitingCopy}
                </p>
              </div>
              <div className={`rounded-2xl border p-5 min-w-[220px] ${isDark ? 'border-cyan-400/25 bg-cyan-500/10' : 'border-cyan-200 bg-cyan-50'}`}>
                <p className={`text-xs font-black uppercase ${isDark ? 'text-cyan-300' : 'text-cyan-700'}`}>Live price only</p>
                <p className="mt-2 text-3xl font-black">{formatMoney(stockData.current_price, stockData.currency)}</p>
                <p className={`mt-1 text-sm font-bold ${stockData.day_change >= 0 ? 'text-green-300' : 'text-red-300'}`}>
                  {stockData.day_change >= 0 ? '+' : ''}{Number(stockData.day_change || 0).toFixed(2)} ({Number(stockData.day_change_percent || 0).toFixed(2)}%)
                </p>
              </div>
            </div>

            <div className={`mt-8 rounded-2xl border p-5 ${isDark ? 'border-slate-700 bg-slate-900/80' : 'border-slate-200 bg-slate-50/90'}`}>
              <div className="flex items-center justify-between gap-3 mb-3">
                <div>
                  <p className={`text-sm font-black ${isDark ? 'text-white' : 'text-slate-950'}`}>{trainingStateLabel}</p>
                  <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                    {trainingBlocked
                      ? 'This ticker is private, unsupported, or does not have enough reliable historical candles.'
                      : trainingFailed
                      ? 'Retry will start a fresh stock-specific training and validation job.'
                      : 'This page will automatically show the full prediction result when the trained model is ready.'}
                  </p>
                </div>
                <span className={`text-sm font-black ${trainingBlocked ? 'text-amber-400' : trainingFailed ? 'text-cyan-400' : isDark ? 'text-cyan-300' : 'text-cyan-700'}`}>{waitingEta}</span>
              </div>
              <div className={`h-3 rounded-full overflow-hidden border ${isDark ? 'bg-slate-800 border-slate-700' : 'bg-slate-200 border-slate-300'}`}>
                <div
                  className={`h-full rounded-full transition-all duration-700 ${trainingBlocked ? 'bg-amber-400' : trainingFailed ? 'bg-cyan-500' : 'bg-gradient-to-r from-cyan-400 to-blue-500'}`}
                  style={{ width: `${waitingProgress}%` }}
                />
              </div>
              <div className={`mt-2 flex items-center justify-between text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                <span>{stageDisplay}</span>
                <span>{Math.min(100, Math.max(0, Number(trainingProgress) || 0)).toFixed(0)}%</span>
              </div>
              {canRetryTraining && (
                <button
                  type="button"
                  onClick={retryTickerTraining}
                  className="mt-4 inline-flex items-center justify-center rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 text-sm font-black shadow-lg hover:shadow-cyan-500/25"
                >
                  Retry Training
                </button>
              )}
              {trainingBlocked && (
                <button
                  type="button"
                  onClick={() => {
                    setStockData(null);
                    setTrainingStatus(null);
                    setTicker('');
                    setHasSearched(false);
                  }}
                  className="mt-4 inline-flex items-center justify-center rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 text-sm font-black shadow-lg hover:shadow-cyan-500/25"
                >
                  Search Listed Ticker
                </button>
              )}
            </div>

            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                ['1', 'Fetch market history', 'Live OHLCV, volume, indicators, and fundamentals are collected.'],
                ['2', 'Train and validate', 'The ticker-specific model is trained and checked before predictions are exposed.'],
                ['3', 'Publish result', 'Forecast range, trust score, risk, charts, and signal appear automatically.']
              ].map(([step, title, body]) => (
                <div key={step} className={`rounded-2xl border p-4 ${isDark ? 'border-slate-700 bg-slate-900/70' : 'border-slate-200 bg-white/80'}`}>
                  <div className={`w-8 h-8 rounded-lg border flex items-center justify-center font-black ${isDark ? 'bg-cyan-500/15 border-cyan-400/30 text-cyan-300' : 'bg-cyan-50 border-cyan-200 text-cyan-700'}`}>{step}</div>
                  <p className={`mt-3 text-sm font-black ${isDark ? 'text-white' : 'text-slate-950'}`}>{title}</p>
                  <p className={`mt-1 text-xs leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>{body}</p>
                </div>
              ))}
            </div>

            <div className={`mt-6 rounded-2xl border p-4 ${isDark ? 'border-amber-400/25 bg-amber-500/10' : 'border-amber-200 bg-amber-50'}`}>
              <p className={`text-sm font-black ${isDark ? 'text-amber-200' : 'text-amber-800'}`}>Why results are hidden now</p>
              <p className={`mt-1 text-sm ${isDark ? 'text-amber-100/80' : 'text-amber-700'}`}>
                To avoid unrealistic predictions, Datavision shows AI target prices only after market data checks, training, and validation are complete.
              </p>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-[#070b12] py-0">
      {agentMode.active && (
        <div className="fixed inset-x-0 top-[86px] z-[70] pointer-events-none px-4">
          <div className="mx-auto max-w-[980px] rounded-2xl border border-cyan-300/70 dark:border-cyan-400/40 bg-white/95 dark:bg-slate-950/95 shadow-[0_18px_60px_rgba(6,182,212,0.24)] backdrop-blur-xl overflow-hidden">
            <div className="flex items-center gap-4 px-5 py-4">
              <div className="relative h-12 w-12 rounded-2xl bg-cyan-500/15 border border-cyan-300/60 flex items-center justify-center">
                <Activity className="h-6 w-6 text-cyan-600 dark:text-cyan-300" />
                <span className="absolute -right-1 -top-1 h-3 w-3 rounded-full bg-emerald-400 animate-pulse" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-black uppercase tracking-[0.24em] text-cyan-600 dark:text-cyan-300">Agentic Prediction Mode</p>
                <h3 className="text-lg font-black text-slate-950 dark:text-white">
                  Operating Prediction Desk for {agentMode.symbol}
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-300">{agentMode.message}</p>
              </div>
              <div className="hidden sm:flex items-center gap-2 text-xs font-black text-slate-500 dark:text-slate-400">
                {[
                  ['ticker', 'Enter ticker'],
                  ['days', 'Select days'],
                  ['predict', 'Click Predict'],
                  ['running', 'Read result']
                ].map(([step, label]) => (
                  <span
                    key={step}
                    className={`rounded-full px-3 py-1 border ${
                      agentMode.step === step
                        ? 'border-cyan-400 bg-cyan-500/15 text-cyan-700 dark:text-cyan-200'
                        : 'border-slate-200 dark:border-slate-700'
                    }`}
                  >
                    {label}
                  </span>
                ))}
              </div>
            </div>
            <div className="h-1 bg-slate-100 dark:bg-slate-800">
              <div
                className="h-full bg-cyan-500 transition-all duration-500"
                style={{
                  width: agentMode.step === 'ticker' ? '25%' : agentMode.step === 'days' ? '50%' : agentMode.step === 'predict' ? '75%' : '100%'
                }}
              />
            </div>
          </div>
        </div>
      )}
      {/* Search Header */}
      <div className="sticky top-0 z-40 border-b border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-[#0b1220]/95 backdrop-blur-xl shadow-sm">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-3">
          <form onSubmit={handleSearch} className="grid grid-cols-1 md:grid-cols-[minmax(280px,1fr)_150px_130px] gap-3 items-center">
            <div className="relative w-full">
              <input
                ref={searchInputRef}
                type="text"
                value={ticker}
                onChange={(e) => handleTickerInput(e.target.value)}
                onFocus={handleInputFocus}
                onBlur={handleInputBlur}
                placeholder="Enter ticker (e.g., AAPL)"
                className={`w-full px-4 py-2.5 border bg-white dark:bg-slate-950 text-slate-900 dark:text-white rounded-md focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition text-sm touch-target ${
                  agentMode.active && agentMode.step === 'ticker'
                    ? 'border-cyan-400 ring-4 ring-cyan-400/25 shadow-[0_0_0_4px_rgba(34,211,238,0.16)]'
                    : 'border-slate-300 dark:border-slate-700'
                }`}
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
              ref={daysSelectRef}
              value={days}
              onChange={(e) => handleDaysChange(e.target.value)}
              className={`w-full px-4 py-2.5 border bg-white dark:bg-slate-950 text-slate-900 dark:text-white rounded-md focus:ring-2 focus:ring-cyan-500 text-sm touch-target ${
                agentMode.active && agentMode.step === 'days'
                  ? 'border-cyan-400 ring-4 ring-cyan-400/25 shadow-[0_0_0_4px_rgba(34,211,238,0.16)]'
                  : 'border-slate-300 dark:border-slate-700'
              }`}
            >
              {FORECAST_HORIZONS.map((horizon) => (
                <option key={horizon} value={horizon}>
                  {horizon === 1 ? '1 Day' : `${horizon} Days`}
                </option>
              ))}
            </select>
            <button
              ref={predictButtonRef}
              type="submit"
              className={`w-full bg-cyan-600 hover:bg-cyan-500 text-white px-5 py-2.5 rounded-md flex items-center justify-center gap-2 transition active:scale-95 text-sm font-bold touch-target shadow-lg hover:shadow-cyan-500/25 ${
                agentMode.active && agentMode.step === 'predict'
                  ? 'ring-4 ring-cyan-400/35 scale-[1.02]'
                  : ''
              }`}
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

      {false && stockData && !predictionIsReady && (
        <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-5 sm:py-6 pb-10">
          <div className="rounded-xl border border-blue-200 dark:border-blue-500/30 bg-white dark:bg-[#0b1220] shadow-sm p-4 sm:p-5 mb-4">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
              <div>
                <p className="text-xs font-black uppercase tracking-wide text-blue-700 dark:text-blue-400">Live analysis available</p>
                <h2 className="mt-1 text-2xl sm:text-3xl font-black text-gray-900 dark:text-white">
                  {stockData.ticker} live market view
                </h2>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 max-w-3xl">
                  Review the live chart, trend, volatility, and range behavior now. AI forecast range appears automatically when the model is ready.
                </p>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 min-w-full lg:min-w-[520px]">
                {[
                  ['Live price', formatMoney(stockData.current_price, stockData.currency)],
                  ['AI range', predictionIsReady ? 'Ready' : trainingEtaLabel],
                  ['Trend', marketRegime.label || 'Tracking'],
                  ['Risk', riskProfile.label || 'Tracking']
                ].map(([label, value]) => (
                  <div key={label} className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <p className="text-[10px] font-black uppercase text-slate-500 dark:text-slate-400">{label}</p>
                    <p className="mt-1 text-sm sm:text-base font-black text-slate-900 dark:text-white">{value}</p>
                  </div>
                ))}
              </div>
            </div>
            <div className="mt-4">
              <div className="flex items-center justify-between text-xs font-bold text-gray-600 dark:text-gray-300 mb-1">
                <span>AI forecast preparation</span>
                <span>{trainingEtaLabel}</span>
              </div>
              <div className="h-2 rounded-full bg-slate-100 dark:bg-slate-900 overflow-hidden border border-slate-200 dark:border-slate-800">
                <div className="h-full rounded-full bg-blue-500 transition-all duration-700" style={{ width: `${Math.max(6, Math.min(100, trainingProgress))}%` }} />
              </div>
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">Live data is shown now; forecast range is kept hidden until it passes readiness checks.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-[1.35fr_0.9fr] gap-4">
            <div className="bg-white dark:bg-[#0b1220] rounded-xl shadow-sm p-3 border border-slate-200 dark:border-slate-800">
              <h3 className="text-sm font-black uppercase tracking-wide text-slate-900 dark:text-white mb-3">Live Price Workspace</h3>
              <TradingViewStyleChart
                candles={stockData.technical_chart?.candles || []}
                sma20={stockData.technical_chart?.moving_averages?.sma20 || []}
                predictions={[]}
                isProfit={stockData.day_change >= 0}
                currencySymbol={currencySymbol}
                recommendation={{ signal: 'LIVE' }}
                tradePlan={null}
                backtestGhost={[]}
              />
            </div>

            <div className="grid grid-cols-1 gap-4">
              <div className={`${chartTheme.panel} rounded-xl shadow-sm p-4 border`}>
                <div className="flex items-center justify-between mb-3">
                  <h3 className={`text-sm font-black uppercase tracking-wide ${chartTheme.title}`}>30D Price Trend</h3>
                  <span className={`text-xs ${stockData.day_change >= 0 ? chartTheme.positive : chartTheme.negative}`}>{stockData.day_change_percent?.toFixed?.(2) ?? stockData.day_change_percent}% today</span>
                </div>
                <div className={`h-60 ${chartTheme.plot} rounded-lg border p-2`}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={liveTrendData} margin={{ top: 12, right: 12, left: 0, bottom: 8 }}>
                      <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                      <XAxis dataKey="date" tick={{ fontSize: 10, fill: chartTheme.tick }} axisLine={false} tickLine={false} />
                      <YAxis tickFormatter={(val) => `${currencySymbol}${Number(val).toFixed(0)}`} tick={{ fontSize: 10, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={52} />
                      <Tooltip formatter={(value) => [formatMoney(value, stockData.currency), 'Price']} />
                      <Area type="monotone" dataKey="price" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.18} strokeWidth={2.4} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className={`${chartTheme.panel} rounded-xl shadow-sm p-4 border`}>
                <div className="flex items-center justify-between mb-3">
                  <h3 className={`text-sm font-black uppercase tracking-wide ${chartTheme.title}`}>Range Pressure</h3>
                  <span className="text-xs text-slate-500">High / low / close</span>
                </div>
                <div className={`h-60 ${chartTheme.plot} rounded-lg border p-2`}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={liveRangeData} margin={{ top: 12, right: 12, left: 0, bottom: 8 }}>
                      <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                      <XAxis dataKey="date" tick={{ fontSize: 10, fill: chartTheme.tick }} axisLine={false} tickLine={false} />
                      <YAxis tickFormatter={(val) => `${currencySymbol}${Number(val).toFixed(0)}`} tick={{ fontSize: 10, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={52} />
                      <Tooltip formatter={(value, name) => [formatMoney(value, stockData.currency), name]} />
                      <Line type="monotone" dataKey="high" stroke="#22c55e" strokeWidth={1.8} dot={false} />
                      <Line type="monotone" dataKey="low" stroke="#ef4444" strokeWidth={1.8} dot={false} />
                      <Line type="monotone" dataKey="close" stroke="#38bdf8" strokeWidth={2.4} dot={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </main>
      )}

      {stockData && (
        <main className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-5 sm:py-6 pb-10 flex flex-col">
          {/* Stock Header */}
          <div className="bg-white dark:bg-[#0b1220] rounded-xl shadow-sm p-4 sm:p-5 mb-4 border border-slate-200 dark:border-slate-800 order-first">
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

            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
              <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                <p className="text-xs font-black uppercase tracking-wide text-gray-700 dark:text-gray-300">Decision State</p>
                <p className="mt-1 text-sm sm:text-base font-semibold text-gray-900 dark:text-white">{headlineStance}</p>
                {Array.isArray(finalRecommendation.reasons) && finalRecommendation.reasons.length > 0 && (
                  <p className="mt-1 text-xs sm:text-sm font-medium text-gray-800 dark:text-gray-300 line-clamp-2">
                    {finalRecommendation.reasons[0]}
                  </p>
                )}
              </div>
              <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                <p className="text-xs font-black uppercase tracking-wide text-gray-700 dark:text-gray-300">Forecast Confidence</p>
                <p className="mt-1 text-xl sm:text-2xl font-black text-gray-900 dark:text-white">
                  {Number(finalRecommendation.confidence_percent ?? 0).toFixed(1)}%
                </p>
                <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                  Strength of this specific forecast
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                <p className="text-xs font-black uppercase tracking-wide text-gray-700 dark:text-gray-300">Trust Score</p>
                <p className={`mt-1 text-xl sm:text-2xl font-black ${trustScore >= 75
                  ? 'text-green-600 dark:text-green-400'
                  : trustScore >= 55
                    ? 'text-cyan-600 dark:text-cyan-400'
                    : trustScore >= 35
                      ? 'text-amber-600 dark:text-amber-400'
                      : 'text-gray-500 dark:text-gray-400'
                  }`}>
                  {Number.isFinite(trustScore) ? trustScore : 0}<span className="text-sm text-gray-400">/100</span>
                </p>
                <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                  Reliability after drift, accuracy, and regime
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                <p className="text-xs font-black uppercase tracking-wide text-gray-700 dark:text-gray-300">Market Regime</p>
                <p className={`mt-1 text-sm sm:text-base font-black ${marketRegime.tone === 'bullish'
                  ? 'text-green-600 dark:text-green-400'
                  : marketRegime.tone === 'bearish'
                    ? 'text-red-600 dark:text-red-400'
                    : marketRegime.tone === 'warning'
                      ? 'text-amber-600 dark:text-amber-400'
                      : 'text-gray-900 dark:text-white'
                  }`}>
                  {marketRegime.label || 'Unknown Regime'}
                </p>
                <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                  Vol {marketRegime.daily_vol_pct ?? 'n/a'}% - 20D {marketRegime.momentum_20d_pct ?? 'n/a'}%
                </p>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className={`rounded-xl border p-3 ${signalIsBullish
                ? 'border-green-300 dark:border-green-500/30 bg-green-50/95 dark:bg-green-500/10 shadow-sm'
                : signalIsBearish
                  ? 'border-red-300 dark:border-red-500/30 bg-red-50/95 dark:bg-red-500/10 shadow-sm'
                  : 'border-gray-300 dark:border-gray-700 bg-white/95 dark:bg-gray-900/70 shadow-sm'
                }`}>
                <p className="text-xs font-black uppercase tracking-wide text-gray-700 dark:text-gray-300">Signal Check</p>
                <p className="mt-1 text-sm font-black text-gray-900 dark:text-white">{headlineSignal}</p>
                <p className="mt-1 text-xs font-medium text-gray-800 dark:text-gray-300">
                  Fused from ML probability, validation quality, technical state, sentiment, volume, and volatility.
                </p>
              </div>

              <div className={`rounded-xl border p-3 ${rangeCrossesCurrent
                ? 'border-amber-300 dark:border-amber-500/30 bg-amber-50/95 dark:bg-amber-500/10 shadow-sm'
                : signalIsBullish
                  ? 'border-green-300 dark:border-green-500/30 bg-green-50/95 dark:bg-green-500/10 shadow-sm'
                  : signalIsBearish
                    ? 'border-red-300 dark:border-red-500/30 bg-red-50/95 dark:bg-red-500/10 shadow-sm'
                    : 'border-gray-300 dark:border-gray-700 bg-white/95 dark:bg-gray-900/70 shadow-sm'
                }`}>
                <p className="text-xs font-black uppercase tracking-wide text-gray-700 dark:text-gray-300">Range Risk</p>
                <p className="mt-1 text-sm font-black text-gray-900 dark:text-white">{rangeBiasLabel}</p>
                <p className="mt-1 text-xs font-medium text-gray-800 dark:text-gray-300">{rangeRiskText}</p>
              </div>

              <div className="rounded-xl border border-gray-300 dark:border-gray-700 bg-white/95 dark:bg-gray-900/70 p-3 shadow-sm">
                <p className="text-xs font-black uppercase tracking-wide text-gray-700 dark:text-gray-300">Model Quality</p>
                <p className="mt-1 text-sm font-black text-gray-900 dark:text-white">{modelQualityLabel}</p>
                <p className="mt-1 text-xs font-medium text-gray-800 dark:text-gray-300">{modelQualityText}</p>
              </div>
            </div>

            <div className={`mt-4 rounded-lg border p-3 ${activeModelStatus.model_ready
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
                      ? 'Model is ready. Forecasts are using trained ticker-specific ML artifacts.'
                      : 'Please wait a little while. Live market analysis is shown now while the dedicated model trains.'}
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

            <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 mt-4">
              <div className="bg-slate-50 dark:bg-slate-950/60 p-3 sm:p-4 rounded-lg border border-slate-200 dark:border-slate-800">
                <p className="text-xs sm:text-sm text-cyan-700 dark:text-cyan-400 font-medium flex items-center gap-1">
                  <DollarSign className="w-3 h-3 sm:w-4 sm:h-4" />
                  <span className="truncate">Current Price</span>
                </p>
                <p className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900 dark:text-white mt-1">${stockData.current_price}</p>
                <p className={`text-xs sm:text-sm mt-1 sm:mt-2 font-semibold ${stockData.day_change >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                  {stockData.day_change >= 0 ? '+' : ''}{stockData.day_change.toFixed(2)} ({stockData.day_change_percent.toFixed(2)}%)
                </p>
              </div>

              <div className="bg-slate-50 dark:bg-slate-950/60 p-3 sm:p-4 rounded-lg border border-slate-200 dark:border-slate-800">
                <p className="text-xs sm:text-sm text-blue-700 dark:text-blue-400 font-medium flex items-center gap-1">
                  <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4" />
                  <span className="truncate">Price Range ({days}D)</span>
                </p>
                {stockData.price_range_low?.length > 0 && stockData.price_range_high?.length > 0 ? (
                  <>
                    <p className="text-lg sm:text-xl lg:text-2xl font-bold text-gray-900 dark:text-white mt-1">
                      {currencySymbol}{Number(stockData.price_range_low[selectedRangeIndex]).toFixed(2)}
                      <span className="text-gray-400 mx-1">-</span>
                      {currencySymbol}{Number(stockData.price_range_high[selectedRangeIndex]).toFixed(2)}
                    </p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 truncate">10th - 90th percentile</p>
                  </>
                ) : (
                  <>
                    <p className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900 dark:text-white mt-1">Range unavailable</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 sm:mt-2 truncate">Waiting for calibrated range</p>
                  </>
                )}
              </div>

              <div className={`p-3 sm:p-4 rounded-lg border ${selectedRangeChangeHigh >= 0 ? 'bg-green-50 dark:bg-green-500/10 border-green-200 dark:border-green-500/20' : 'bg-red-50 dark:bg-red-500/10 border-red-200 dark:border-red-500/20'}`}>
                <p className={`text-xs sm:text-sm font-medium flex items-center gap-1 ${selectedRangeChangeHigh >= 0 ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}>
                  <Activity className="w-3 h-3 sm:w-4 sm:h-4" />
                  <span className="truncate">Expected Move Range</span>
                </p>
                {hasSelectedRange ? (
                  <>
                    <p className={`text-lg sm:text-xl lg:text-2xl font-bold mt-1 ${selectedRangeChangeHigh >= 0 ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}>
                      {selectedRangeChangeLow >= 0 ? '+' : ''}{currencySymbol}{selectedRangeChangeLow.toFixed(2)}
                      <span className="text-gray-400 mx-1">–</span>
                      {selectedRangeChangeHigh >= 0 ? '+' : ''}{currencySymbol}{selectedRangeChangeHigh.toFixed(2)}
                    </p>
                    <p className={`text-xs sm:text-sm mt-1 sm:mt-2 font-semibold ${selectedRangeChangeHigh >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                      {selectedRangePctLow >= 0 ? '+' : ''}{selectedRangePctLow.toFixed(2)}%
                      <span className="text-gray-400 mx-1">–</span>
                      {selectedRangePctHigh >= 0 ? '+' : ''}{selectedRangePctHigh.toFixed(2)}%
                    </p>
                  </>
                ) : (
                  <>
                    <p className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-500 dark:text-gray-400 mt-1">N/A</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 sm:mt-2 truncate">Range not ready</p>
                  </>
                )}
              </div>

              <div className="bg-slate-50 dark:bg-slate-950/60 p-3 sm:p-4 rounded-lg border border-slate-200 dark:border-slate-800">
                <p className="text-xs sm:text-sm text-purple-700 dark:text-purple-400 font-medium truncate">Volume</p>
                <p className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900 dark:text-white mt-1 truncate">{formatVolume(stockData.volume)}</p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 sm:mt-2 truncate">
                  Mkt Cap: {formatCurrencyCompact(stockData.market_cap, currencySymbol)}
                </p>
              </div>
            </div>
          </div>

          {/* Opportunity Radar */}
          <div className="bg-white dark:bg-[#0b1220] rounded-xl shadow-sm p-4 sm:p-5 mb-4 border border-slate-200 dark:border-slate-800">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-4">
              <div>
                <p className="text-xs font-black uppercase tracking-[0.18em] text-cyan-600 dark:text-cyan-400">AI Watchlist Ranking</p>
                <h3 className="text-xl sm:text-2xl font-black text-slate-950 dark:text-white mt-1">Opportunity Radar</h3>
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                  Ranks your watchlist by ML edge, trust, risk/reward, regime, sentiment, and forecast readiness.
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-2 w-full lg:w-auto">
                <div className="relative w-full sm:w-56">
                  <input
                    value={watchlistInput}
                    onChange={(e) => handleWatchlistInputChange(e.target.value)}
                    onFocus={() => setShowWatchlistSuggestions(watchlistSuggestions.length > 0)}
                    onBlur={() => setTimeout(() => setShowWatchlistSuggestions(false), 150)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        handleAddWatchlistSymbol();
                      }
                    }}
                    placeholder="Add ticker, e.g. GOOGL"
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 text-slate-900 dark:text-white rounded-md text-sm focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                  />
                  {showWatchlistSuggestions && watchlistSuggestions.length > 0 && (
                    <div className="absolute z-30 mt-2 w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 shadow-xl overflow-hidden">
                      {watchlistSuggestions.map(({ symbol, name, exchange }) => (
                        <button
                          key={`${symbol}-${exchange || ''}`}
                          type="button"
                          onMouseDown={(e) => e.preventDefault()}
                          onClick={() => {
                            setWatchlistInput(symbol.toUpperCase());
                            setWatchlist((prev) => [...new Set([...(prev || []), symbol.toUpperCase()])].slice(0, 8));
                            setWatchlistSuggestions([]);
                            setShowWatchlistSuggestions(false);
                          }}
                          className="w-full px-3 py-2 text-left hover:bg-cyan-50 dark:hover:bg-cyan-500/10"
                        >
                          <span className="text-sm font-black text-slate-900 dark:text-white">{symbol}</span>
                          <span className="ml-2 text-xs text-slate-500">{name || exchange || 'Stock'}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={handleAddWatchlistSymbol}
                  className="px-4 py-2 rounded-md border border-slate-300 dark:border-slate-700 text-sm font-bold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-900"
                >
                  Add
                </button>
                <button
                  type="button"
                  onClick={analyzeWatchlist}
                  disabled={watchlistLoading || watchlist.length === 0}
                  className="px-5 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-400 text-white text-sm font-black shadow-lg hover:shadow-cyan-500/20"
                >
                  {watchlistLoading ? 'Ranking...' : 'Rank Watchlist'}
                </button>
                <button
                  type="button"
                  onClick={() => setShowWatchlistDesk((prev) => !prev)}
                  className="px-4 py-2 rounded-md border border-slate-300 dark:border-slate-700 text-sm font-bold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-900"
                >
                  {showWatchlistDesk ? 'Hide Desk' : 'Open Desk'}
                </button>
              </div>
            </div>

            {showWatchlistDesk && (
              <>
            <div className="flex gap-2 flex-wrap mb-4">
              {watchlist.map((symbol) => (
                <span key={symbol} className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-xs font-black text-slate-700 dark:text-slate-200">
                  {symbol}
                  <button type="button" onClick={() => handleRemoveWatchlistSymbol(symbol)} className="text-slate-400 hover:text-red-500">x</button>
                </span>
              ))}
            </div>

            {watchlistError && (
              <div className="mb-4 rounded-md border border-red-200 dark:border-red-500/30 bg-red-50 dark:bg-red-500/10 px-3 py-2 text-sm font-semibold text-red-700 dark:text-red-300">
                {watchlistError}
              </div>
            )}

            {watchlistRanks.length > 0 ? (
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="xl:col-span-3 overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0b1220]">
                  <table className="min-w-full text-left text-sm">
                    <thead className="bg-slate-50 dark:bg-slate-950 text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <tr>
                        <th className="px-3 py-3">Rank</th>
                        <th className="px-3 py-3">Stock</th>
                        <th className="px-3 py-3">Live Price</th>
                        <th className="px-3 py-3">Predicted Range</th>
                        <th className="px-3 py-3">Signal</th>
                        <th className="px-3 py-3">Trust</th>
                        <th className="px-3 py-3">Adj Edge</th>
                        <th className="px-3 py-3">Regime</th>
                        <th className="px-3 py-3">Risk</th>
                        <th className="px-3 py-3">Status</th>
                        <th className="px-3 py-3">Why</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800 bg-white dark:bg-[#0b1220]">
                      {watchlistRanks.map((row, index) => (
                        <tr key={row.ticker} className="hover:bg-cyan-50/60 dark:hover:bg-cyan-500/5 transition">
                          <td className="px-3 py-3 font-black text-slate-500">#{index + 1}</td>
                          <td className="px-3 py-3">
                            <button
                              type="button"
                              onClick={() => {
                                setTicker(row.ticker);
                                setHasSearched(true);
                                fetchStockData(row.ticker);
                              }}
                              className="font-black text-slate-950 dark:text-white hover:text-cyan-600 dark:hover:text-cyan-400"
                            >
                              {row.ticker}
                            </button>
                            <p className="text-xs text-slate-500 truncate max-w-[170px]">{row.company}</p>
                          </td>
                          <td className="px-3 py-3 font-black text-slate-900 dark:text-white whitespace-nowrap">
                            {formatMoney(row.price, row.currency)}
                          </td>
                          <td className="px-3 py-3 whitespace-nowrap">
                            <p className="text-xs font-black text-slate-900 dark:text-white">
                              {row.ready
                                ? `${formatMoney(Math.min(row.rangeLow, row.rangeHigh), row.currency)} - ${formatMoney(Math.max(row.rangeLow, row.rangeHigh), row.currency)}`
                                : 'AI range building'}
                            </p>
                            <p className="text-[10px] font-semibold text-slate-400">
                              {row.ready ? `target ${formatMoney(row.predictedPrice, row.currency)} - ${row.horizon}` : `ETA ${row.eta}`}
                            </p>
                          </td>
                          <td className="px-3 py-3">
                            <span className={`px-2.5 py-1 rounded-md text-xs font-black ${row.signal.includes('BUY')
                              ? 'bg-green-100 text-green-700 dark:bg-green-500/15 dark:text-green-300'
                              : row.signal.includes('SELL')
                                ? 'bg-red-100 text-red-700 dark:bg-red-500/15 dark:text-red-300'
                                : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'
                              }`}>
                              {row.signal}
                            </span>
                          </td>
                          <td className="px-3 py-3 font-black text-slate-900 dark:text-white">{row.trust}/100</td>
                          <td className={`px-3 py-3 font-black ${row.adjustedEdge >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                            {row.adjustedEdge >= 0 ? '+' : ''}{row.adjustedEdge.toFixed(2)}%
                            <p className="text-[10px] font-semibold text-slate-400">raw {row.edge >= 0 ? '+' : ''}{row.edge.toFixed(2)}%</p>
                          </td>
                          <td className="px-3 py-3 text-xs font-bold text-slate-700 dark:text-slate-300">{row.regime}</td>
                          <td className={`px-3 py-3 text-xs font-black ${row.risk === 'High' || row.risk === 'Speculative'
                            ? 'text-red-600 dark:text-red-400'
                            : row.risk === 'Medium'
                              ? 'text-amber-600 dark:text-amber-400'
                              : 'text-green-600 dark:text-green-400'
                            }`}>
                            {row.risk}
                            <p className="text-[10px] font-semibold text-slate-400">{row.riskScore.toFixed(0)}/100</p>
                          </td>
                          <td className="px-3 py-3">
                            <span className={`inline-flex rounded-md px-2 py-1 text-[10px] font-black uppercase ${row.ready
                              ? 'bg-green-100 text-green-700 dark:bg-green-500/15 dark:text-green-300'
                              : 'bg-blue-100 text-blue-700 dark:bg-blue-500/15 dark:text-blue-300'
                              }`}>
                              {row.statusLabel}
                            </span>
                            {!row.ready && (
                              <div className="mt-1 h-1.5 w-28 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                                <div className="h-full rounded-full bg-blue-500" style={{ width: `${Math.max(6, Math.min(100, row.progress))}%` }} />
                              </div>
                            )}
                            {!row.ready && <p className="mt-1 max-w-[180px] text-[10px] font-semibold text-slate-500 dark:text-slate-400 line-clamp-2">{row.statusMessage}</p>}
                          </td>
                          <td className="px-3 py-3">
                            <p className="max-w-[220px] text-xs font-medium leading-snug text-slate-600 dark:text-slate-400 line-clamp-2">{row.reason}</p>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Opportunity Score</p>
                    <p className="text-xs text-slate-500">Top {Math.min(6, watchlistRanks.length)}</p>
                  </div>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={watchlistRanks.slice(0, 6)} layout="vertical" margin={{ top: 8, right: 18, left: 12, bottom: 8 }}>
                        <CartesianGrid stroke={chartTheme.grid} horizontal={false} />
                        <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} />
                        <YAxis type="category" dataKey="ticker" tick={{ fontSize: 12, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} width={52} />
                        <Tooltip
                          cursor={{ fill: isDark ? 'rgba(34,211,238,0.08)' : 'rgba(6,182,212,0.08)' }}
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const row = payload[0].payload;
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                <p className={chartTheme.tooltipText}>Score: {row.opportunityScore.toFixed(0)}/100</p>
                                <p className={chartTheme.tooltipText}>Trust: {row.trust}/100</p>
                                <p className={row.adjustedEdge >= 0 ? chartTheme.positive : chartTheme.negative}>Adj edge: {row.adjustedEdge >= 0 ? '+' : ''}{row.adjustedEdge.toFixed(2)}%</p>
                                <p className={chartTheme.tooltipText}>Risk: {row.risk} ({row.riskScore.toFixed(0)}/100)</p>
                              </div>
                            );
                          }}
                        />
                        <Bar dataKey="opportunityScore" radius={[0, 6, 6, 0]}>
                          {watchlistRanks.slice(0, 6).map((row) => (
                            <Cell key={row.ticker} fill={row.signal.includes('BUY') ? '#10b981' : row.signal.includes('SELL') ? '#ef4444' : '#64748b'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="xl:col-span-2 grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Live Price vs Predicted Target</p>
                      <p className="text-xs text-slate-500">Forecast range</p>
                    </div>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={watchlistRanks.slice(0, 6)} margin={{ top: 10, right: 18, left: 4, bottom: 8 }}>
                          <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                          <XAxis dataKey="ticker" tick={{ fontSize: 11, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={58} />
                          <Tooltip
                            content={({ active, payload }) => {
                              if (!active || !payload?.length) return null;
                              const row = payload[0].payload;
                              return (
                                <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                  <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                  <p className={chartTheme.tooltipText}>Live: {formatMoney(row.price, row.currency)}</p>
                                  <p className={chartTheme.tooltipText}>Target: {formatMoney(row.predictedPrice, row.currency)}</p>
                                  <p className={chartTheme.tooltipText}>Range: {formatMoney(Math.min(row.rangeLow, row.rangeHigh), row.currency)} - {formatMoney(Math.max(row.rangeLow, row.rangeHigh), row.currency)}</p>
                                </div>
                              );
                            }}
                          />
                          <Line type="monotone" dataKey="rangeLow" name="Low Range" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
                          <Line type="monotone" dataKey="rangeHigh" name="High Range" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
                          <Line type="monotone" dataKey="price" name="Live" stroke="#38bdf8" strokeWidth={2.4} dot={{ r: 3 }} />
                          <Line type="monotone" dataKey="predictedPrice" name="Predicted" stroke="#10b981" strokeWidth={2.4} dot={{ r: 3 }} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Trust, Risk, and Adjusted Edge</p>
                      <p className="text-xs text-slate-500">Comparison</p>
                    </div>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={watchlistRanks.slice(0, 6)} margin={{ top: 10, right: 18, left: 4, bottom: 8 }}>
                          <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                          <XAxis dataKey="ticker" tick={{ fontSize: 11, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} />
                          <YAxis yAxisId="score" domain={[0, 100]} tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={42} />
                          <YAxis yAxisId="edge" orientation="right" tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={42} />
                          <Tooltip
                            content={({ active, payload }) => {
                              if (!active || !payload?.length) return null;
                              const row = payload[0].payload;
                              return (
                                <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                  <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                  <p className={chartTheme.tooltipText}>Trust: {row.trust}/100</p>
                                  <p className={chartTheme.tooltipText}>Risk: {row.riskScore.toFixed(0)}/100</p>
                                  <p className={row.adjustedEdge >= 0 ? chartTheme.positive : chartTheme.negative}>Adjusted edge: {row.adjustedEdge >= 0 ? '+' : ''}{row.adjustedEdge.toFixed(2)}%</p>
                                </div>
                              );
                            }}
                          />
                          <Bar yAxisId="score" dataKey="riskScore" fill="#ef4444" opacity={0.28} radius={[4, 4, 0, 0]} />
                          <Line yAxisId="score" type="monotone" dataKey="trust" stroke="#38bdf8" strokeWidth={2.4} dot={{ r: 3 }} />
                          <Line yAxisId="edge" type="monotone" dataKey="adjustedEdge" stroke="#10b981" strokeWidth={2.4} dot={{ r: 3 }} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
                <div className="xl:col-span-3 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  <div className="h-64 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Forecast Range Width</p>
                      <p className="text-xs text-slate-500">Lower is tighter</p>
                    </div>
                    <ResponsiveContainer width="100%" height="85%">
                      <BarChart data={watchlistRanks.slice(0, 6)} margin={{ top: 8, right: 18, left: 4, bottom: 8 }}>
                        <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                        <XAxis dataKey="ticker" tick={{ fontSize: 11, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} />
                        <YAxis tickFormatter={(val) => `${Number(val).toFixed(0)}%`} tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={42} />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const row = payload[0].payload;
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                <p className={chartTheme.tooltipText}>Range width: {row.rangeWidthPct.toFixed(2)}%</p>
                                <p className={chartTheme.tooltipText}>Range: {row.ready ? `${formatMoney(Math.min(row.rangeLow, row.rangeHigh), row.currency)} - ${formatMoney(Math.max(row.rangeLow, row.rangeHigh), row.currency)}` : 'Preparing AI range'}</p>
                              </div>
                            );
                          }}
                        />
                        <Bar dataKey="rangeWidthPct" fill="#f59e0b" opacity={0.72} radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="h-64 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Return Potential</p>
                      <p className="text-xs text-slate-500">Raw vs adjusted</p>
                    </div>
                    <ResponsiveContainer width="100%" height="85%">
                      <ComposedChart data={watchlistRanks.slice(0, 6)} margin={{ top: 8, right: 18, left: 4, bottom: 8 }}>
                        <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                        <XAxis dataKey="ticker" tick={{ fontSize: 11, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} />
                        <YAxis tickFormatter={(val) => `${Number(val).toFixed(0)}%`} tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={42} />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const row = payload[0].payload;
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                <p className={row.edge >= 0 ? chartTheme.positive : chartTheme.negative}>Raw edge: {row.edge >= 0 ? '+' : ''}{row.edge.toFixed(2)}%</p>
                                <p className={row.adjustedEdge >= 0 ? chartTheme.positive : chartTheme.negative}>Adjusted edge: {row.adjustedEdge >= 0 ? '+' : ''}{row.adjustedEdge.toFixed(2)}%</p>
                              </div>
                            );
                          }}
                        />
                        <Bar dataKey="returnPotential" fill="#94a3b8" opacity={0.45} radius={[4, 4, 0, 0]} />
                        <Line type="monotone" dataKey="adjustedEdge" stroke="#10b981" strokeWidth={2.2} dot={{ r: 3 }} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="h-64 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Signal Mix</p>
                      <p className="text-xs text-slate-500">Desk exposure</p>
                    </div>
                    <div className="space-y-4">
                      {[
                        ['Buy', watchlistRanks.filter((row) => row.signal.includes('BUY')).length, 'bg-green-500'],
                        ['Hold', watchlistRanks.filter((row) => row.signal.includes('HOLD')).length, 'bg-slate-500'],
                        ['Sell', watchlistRanks.filter((row) => row.signal.includes('SELL')).length, 'bg-red-500'],
                        ['Speculative', watchlistRanks.filter((row) => row.risk === 'Speculative' || row.risk === 'High').length, 'bg-amber-500']
                      ].map(([label, count, colorClass]) => {
                        const pct = Math.round((Number(count) / Math.max(1, watchlistRanks.length)) * 100);
                        return (
                          <div key={label}>
                            <div className="flex items-center justify-between text-xs font-bold text-slate-600 dark:text-slate-300 mb-1">
                              <span>{label}</span>
                              <span>{count} stocks</span>
                            </div>
                            <div className="h-2.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                              <div className={`h-full rounded-full ${colorClass}`} style={{ width: `${pct}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
                <div className="xl:col-span-3 grid grid-cols-1 xl:grid-cols-2 gap-4">
                  <div className="h-72 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">Risk-Adjusted Quadrant</p>
                      <p className="text-xs text-slate-500">Trust vs edge</p>
                    </div>
                    <ResponsiveContainer width="100%" height="86%">
                      <ComposedChart data={watchlistRanks.slice(0, 8)} margin={{ top: 10, right: 24, left: 6, bottom: 12 }}>
                        <CartesianGrid stroke={chartTheme.grid} />
                        <XAxis
                          dataKey="adjustedEdge"
                          type="number"
                          domain={['dataMin - 1', 'dataMax + 1']}
                          tickFormatter={(val) => `${Number(val).toFixed(1)}%`}
                          tick={{ fontSize: 11, fill: chartTheme.tick }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          dataKey="trust"
                          type="number"
                          domain={[0, 100]}
                          tick={{ fontSize: 11, fill: chartTheme.tick }}
                          axisLine={false}
                          tickLine={false}
                          width={42}
                        />
                        <ReferenceLine x={0} stroke={chartTheme.axis} strokeDasharray="4 4" />
                        <ReferenceLine y={60} stroke={chartTheme.axis} strokeDasharray="4 4" />
                        <Tooltip
                          cursor={{ stroke: chartTheme.tick, strokeDasharray: '3 3' }}
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const row = payload[0].payload;
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                <p className={chartTheme.tooltipText}>Trust: {row.trust}/100</p>
                                <p className={row.adjustedEdge >= 0 ? chartTheme.positive : chartTheme.negative}>Adjusted edge: {row.adjustedEdge >= 0 ? '+' : ''}{row.adjustedEdge.toFixed(2)}%</p>
                                <p className={chartTheme.tooltipText}>Risk: {row.risk} ({row.riskScore.toFixed(0)}/100)</p>
                                <p className={chartTheme.tooltipText}>Status: {row.statusLabel}</p>
                              </div>
                            );
                          }}
                        />
                        <Scatter data={watchlistRanks.slice(0, 8)} dataKey="trust">
                          {watchlistRanks.slice(0, 8).map((row) => (
                            <Cell
                              key={row.ticker}
                              fill={row.ready ? (row.adjustedEdge >= 0 ? '#10b981' : '#ef4444') : '#3b82f6'}
                              stroke={row.risk === 'Speculative' || row.risk === 'High' ? '#f59e0b' : 'transparent'}
                              strokeWidth={2}
                            />
                          ))}
                        </Scatter>
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="h-72 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-slate-400">AI Forecast Readiness</p>
                      <p className="text-xs text-slate-500">Ready vs building</p>
                    </div>
                    <ResponsiveContainer width="100%" height="86%">
                      <ComposedChart data={watchlistRanks.slice(0, 8)} margin={{ top: 10, right: 24, left: 4, bottom: 12 }}>
                        <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                        <XAxis dataKey="ticker" tick={{ fontSize: 11, fill: chartTheme.tick, fontWeight: 800 }} axisLine={false} tickLine={false} />
                        <YAxis yAxisId="progress" domain={[0, 100]} tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={42} />
                        <YAxis yAxisId="score" orientation="right" domain={[0, 100]} tick={{ fontSize: 11, fill: chartTheme.tick }} axisLine={false} tickLine={false} width={42} />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const row = payload[0].payload;
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl text-xs`}>
                                <p className={`font-black ${chartTheme.tooltipTitle}`}>{row.ticker}</p>
                                <p className={chartTheme.tooltipText}>Readiness: {row.progress.toFixed(0)}%</p>
                                <p className={chartTheme.tooltipText}>Status: {row.statusLabel}</p>
                                <p className={chartTheme.tooltipText}>ETA: {row.ready ? 'Ready now' : row.eta}</p>
                                <p className={chartTheme.tooltipText}>Opportunity: {row.opportunityScore.toFixed(0)}/100</p>
                              </div>
                            );
                          }}
                        />
                        <Bar yAxisId="progress" dataKey="progress" radius={[4, 4, 0, 0]} opacity={0.78}>
                          {watchlistRanks.slice(0, 8).map((row) => (
                            <Cell key={row.ticker} fill={row.ready ? '#22c55e' : '#3b82f6'} />
                          ))}
                        </Bar>
                        <Line yAxisId="score" type="monotone" dataKey="opportunityScore" stroke="#06b6d4" strokeWidth={2.4} dot={{ r: 3 }} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-950/60 p-5 text-center">
                <p className="text-sm font-bold text-slate-700 dark:text-slate-300">Rank your watchlist to find today&apos;s strongest AI opportunity.</p>
                <p className="text-xs text-slate-500 mt-1">Start with up to 8 stocks for fast comparison.</p>
              </div>
            )}
              </>
            )}
          </div>

          <div className="grid grid-cols-1 gap-4 sm:gap-5">
            {/* Left Column - Charts */}
            <div className="space-y-4 sm:space-y-5">
              <div className="bg-white dark:bg-[#0b1220] rounded-xl shadow-sm p-2 sm:p-3 border border-slate-200 dark:border-slate-800 overflow-hidden">
                <h3 className="text-sm sm:text-base font-black text-gray-900 dark:text-white mb-3 flex items-center gap-2 uppercase tracking-wide">
                  <TrendingUp className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-600 dark:text-cyan-400" />
                  <span>Chart Workspace</span>
                </h3>
                <TradingViewStyleChart
                  candles={stockData.technical_chart?.candles || []}
                  sma20={stockData.technical_chart?.moving_averages?.sma20 || []}
                  predictions={stockData.future_predictions || []}
                  isProfit={stockData.is_profit}
                  currencySymbol={currencySymbol}
                  recommendation={stockData.recommendation}
                  tradePlan={tradePlan}
                  backtestGhost={backtestGhost}
                />
                {tradePlan && (
                  <div className="mt-3 grid grid-cols-2 lg:grid-cols-5 gap-2">
                    {[
                      ['Direction', tradePlan.direction],
                      ['Entry', `${currencySymbol}${Math.min(tradePlan.entryLow, tradePlan.entryHigh).toFixed(2)} - ${currencySymbol}${Math.max(tradePlan.entryLow, tradePlan.entryHigh).toFixed(2)}`],
                      ['Stop', `${currencySymbol}${tradePlan.stop.toFixed(2)}`],
                      ['Target 1', `${currencySymbol}${tradePlan.target1.toFixed(2)}`],
                      ['Target 2', `${currencySymbol}${tradePlan.target2.toFixed(2)}`]
                    ].map(([label, value]) => (
                      <div key={label} className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/60 p-2.5">
                        <p className="text-[10px] font-black uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</p>
                        <p className="text-xs sm:text-sm font-black text-gray-900 dark:text-white mt-1 truncate">{value}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-5">
              <div className={`${chartTheme.panel} rounded-xl shadow-sm p-3 sm:p-4 border overflow-hidden`}>
                <div className="flex items-center justify-between mb-3 sm:mb-4 flex-wrap gap-2">
                  <h3 className={`text-base sm:text-lg lg:text-xl font-bold ${chartTheme.title} flex items-center gap-2`}>
                    <BarChart3 className={`w-4 h-4 sm:w-5 sm:h-5 ${chartTheme.icon}`} />
                    <span>Forecast Range Overview</span>
                  </h3>
                  <span className={`text-xs sm:text-sm ${chartTheme.badge} px-3 py-1.5 rounded-md font-semibold border font-mono`}>
                    {days} Days Forecast
                  </span>
                </div>
                <div className={`w-full ${chartTheme.plot} rounded-lg p-1 sm:p-2 overflow-hidden border`} style={{ height: '340px', minHeight: '300px', maxHeight: '340px' }}>
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
                          rangeLow: pred.range_low,
                          rangeHigh: pred.range_high,
                          forecastRange: Number.isFinite(Number(pred.range_low)) && Number.isFinite(Number(pred.range_high))
                            ? [Number(pred.range_low), Number(pred.range_high)]
                            : null,
                          type: 'predicted'
                        };
                      })
                    ]} margin={{ top: 12, right: 68, left: 8, bottom: 28 }}>
                      <defs>
                        <linearGradient id="forecastRangeFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.28} />
                          <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.05} />
                        </linearGradient>
                        <linearGradient id="forecastPriceFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={stockData.is_profit ? '#10b981' : '#ef4444'} stopOpacity={0.18} />
                          <stop offset="100%" stopColor={stockData.is_profit ? '#10b981' : '#ef4444'} stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid
                        stroke={chartTheme.grid}
                        strokeDasharray="1 0"
                        opacity={0.75}
                        vertical
                      />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 11, fill: chartTheme.tick, fontFamily: 'monospace' }}
                        stroke={chartTheme.axis}
                        tickLine={false}
                        axisLine={{ stroke: chartTheme.axis }}
                        minTickGap={18}
                        height={36}
                      />
                      <YAxis
                        orientation="right"
                        tick={{ fontSize: 11, fill: chartTheme.tick, fontFamily: 'monospace' }}
                        domain={['auto', 'auto']}
                        width={64}
                        stroke={chartTheme.axis}
                        tickLine={false}
                        axisLine={{ stroke: chartTheme.axis }}
                        tickFormatter={(val) => `${currencySymbol}${val.toFixed(0)}`}
                      />
                      <Tooltip
                        cursor={{ stroke: chartTheme.tick, strokeWidth: 1, strokeDasharray: '3 3', opacity: 0.7 }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl backdrop-blur-sm font-mono`}>
                                <p className={`text-xs font-bold ${chartTheme.tooltipTitle} mb-2`}>{data.fullDate}</p>
                                {data.price && (
                                  <p className={`text-xs ${chartTheme.tooltipText}`}>
                                    Price: <span className="font-bold">{currencySymbol}{data.price.toFixed(2)}</span>
                                  </p>
                                )}
                                {data.predicted && (
                                  <p className={`text-xs mt-1 ${stockData.is_profit ? chartTheme.positive : chartTheme.negative}`}>
                                    Forecast: <span className="font-bold">{currencySymbol}{Number(data.predicted).toFixed(2)}</span>
                                  </p>
                                )}
                                {data.rangeLow != null && data.rangeHigh != null ? (
                                  <p className="text-xs mt-1 text-cyan-300">
                                    Forecast range: <span className="font-bold">
                                      {currencySymbol}{Number(data.rangeLow).toFixed(2)} – {currencySymbol}{Number(data.rangeHigh).toFixed(2)}
                                    </span>
                                  </p>
                                ) : null}
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: '12px', paddingTop: '8px', color: chartTheme.legend }}
                        iconType="line"
                      />
                      <Area
                        type="monotone"
                        dataKey="forecastRange"
                        stroke="#22d3ee"
                        fill="url(#forecastRangeFill)"
                        strokeWidth={1.5}
                        connectNulls={false}
                        dot={false}
                        name="Forecast Range"
                      />
                      <Area
                        type="monotone"
                        dataKey="price"
                        stroke={stockData.is_profit ? '#10b981' : '#ef4444'}
                        fill="url(#forecastPriceFill)"
                        strokeWidth={2.2}
                        name="Historical Price"
                        dot={false}
                        activeDot={{ r: 4, fill: '#22d3ee', stroke: chartTheme.activeDotStroke, strokeWidth: 2 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke={stockData.is_profit ? '#34d399' : '#f87171'}
                        strokeWidth={2.6}
                        strokeDasharray="6 4"
                        dot={{ fill: stockData.is_profit ? '#34d399' : '#f87171', r: 4, strokeWidth: 2, stroke: chartTheme.activeDotStroke }}
                        name="AI Forecast Midpoint"
                      />
                      <ReferenceLine
                        y={stockData.current_price}
                        stroke={chartTheme.reference}
                        strokeDasharray="3 3"
                        strokeWidth={1.2}
                        label={{ value: `${currencySymbol}${Number(stockData.current_price).toFixed(2)}`, fontSize: 11, fill: chartTheme.reference, position: 'right' }}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Professional Performance Chart */}
              <div className={`${chartTheme.panel} rounded-xl shadow-sm p-3 sm:p-4 border`}>
                <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
                  <h3 className={`text-base sm:text-lg lg:text-xl font-bold ${chartTheme.title}`}>Performance</h3>
                  <div className={`flex gap-1 sm:gap-2 ${chartTheme.controls} p-1 rounded-md border`}>
                    {['1W', '1M', '3M', '6M', '1Y', 'ALL'].map((period) => (
                      <button
                        key={period}
                        onClick={() => setPerformancePeriod(period)}
                        className={`px-2 sm:px-3 py-1.5 text-xs sm:text-sm font-semibold rounded-md transition-all performance-button touch-target ${performancePeriod === period
                          ? chartTheme.controlActive
                          : chartTheme.controlIdle
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
                      <div key={period} className={`text-center ${chartTheme.metric} px-3 sm:px-4 py-2.5 rounded-md border flex-1 min-w-[70px] shadow-sm`}>
                        <p className={`text-xs ${chartTheme.metricLabel} font-semibold truncate uppercase font-mono`}>{period}</p>
                        <p className={`text-sm sm:text-base font-bold mt-1 ${value >= 0 ? chartTheme.positive : chartTheme.negative}`}>
                          {value >= 0 ? '+' : ''}{value.toFixed(2)}%
                        </p>
                      </div>
                    )
                  ))}
                </div>
                <div className={`w-full ${chartTheme.plot} rounded-lg p-1 sm:p-2 border`} style={{ height: '320px', minHeight: '270px' }}>
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
                      margin={{ top: 12, right: 68, left: 8, bottom: 28 }}
                    >
                      <defs>
                        {(() => {
                          const perfData = getPerformanceData();
                          const isProfit = perfData.prices.length > 1 && perfData.prices[perfData.prices.length - 1] >= perfData.prices[0];
                          return (
                            <>
                              <linearGradient id="performanceGradientGreen" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#10b981" stopOpacity={0.22} />
                                <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
                              </linearGradient>
                              <linearGradient id="performanceGradientRed" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.22} />
                                <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
                              </linearGradient>
                            </>
                          );
                        })()}
                      </defs>
                      <CartesianGrid
                        stroke={chartTheme.grid}
                        strokeDasharray="1 0"
                        opacity={0.75}
                        vertical
                      />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 11, fill: chartTheme.tick, fontFamily: 'monospace' }}
                        stroke={chartTheme.axis}
                        tickLine={false}
                        axisLine={{ stroke: chartTheme.axis }}
                        minTickGap={18}
                        height={36}
                      />
                      <YAxis
                        orientation="right"
                        tick={{ fontSize: 11, fill: chartTheme.tick, fontFamily: 'monospace' }}
                        width={64}
                        stroke={chartTheme.axis}
                        tickLine={false}
                        axisLine={{ stroke: chartTheme.axis }}
                        tickFormatter={(val) => `${currencySymbol}${val.toFixed(0)}`}
                      />
                      <Tooltip
                        cursor={{ stroke: chartTheme.tick, strokeWidth: 1, strokeDasharray: '3 3', opacity: 0.7 }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            const perfData = getPerformanceData();
                            const startPrice = perfData.prices[0];
                            const change = data.price - startPrice;
                            const changePercent = ((change / startPrice) * 100).toFixed(2);
                            return (
                              <div className={`${chartTheme.tooltipBg} p-3 border rounded-md shadow-xl backdrop-blur-sm font-mono`}>
                                <p className={`text-xs font-bold ${chartTheme.tooltipTitle} mb-2`}>{data.fullDate}</p>
                                <p className={`text-xs ${chartTheme.tooltipText}`}>Price: <span className="font-bold">{currencySymbol}{data.price.toFixed(2)}</span></p>
                                <p className={`text-xs mt-1 ${change >= 0 ? chartTheme.positive : chartTheme.negative}`}>
                                  Change: <span className="font-bold">{change >= 0 ? '+' : ''}{currencySymbol}{change.toFixed(2)} ({changePercent}%)</span>
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
                        activeDot={{ r: 4, fill: '#22d3ee', stroke: chartTheme.activeDotStroke, strokeWidth: 2 }}
                      />
                      {(() => {
                        const perfData = getPerformanceData();
                        const lastPrice = perfData.prices[perfData.prices.length - 1];
                        return Number.isFinite(Number(lastPrice)) ? (
                          <ReferenceLine
                            y={Number(lastPrice)}
                            stroke={chartTheme.reference}
                            strokeDasharray="3 3"
                            strokeWidth={1.2}
                            label={{ value: `${currencySymbol}${Number(lastPrice).toFixed(2)}`, fontSize: 11, fill: chartTheme.reference, position: 'right' }}
                          />
                        ) : null;
                      })()}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              </div>

              {/* Professional Forecast Table */}
              <div className="bg-white dark:bg-[#0b1220] rounded-xl shadow-sm p-4 sm:p-5 border border-slate-200 dark:border-slate-800">
                <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4">Forecast Signals</h3>
                
                {stockData.is_training ? (
                  <div className="bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-200 dark:border-blue-500/30 rounded-xl p-6 sm:p-8 text-center">
                    <Activity className="w-10 h-10 text-blue-600 dark:text-blue-400 mx-auto mb-4 animate-pulse" />
                    <h4 className="text-lg sm:text-xl font-bold text-gray-900 dark:text-white mb-2">{trainingStateLabel}</h4>
                    <p className="text-gray-600 dark:text-gray-400 font-medium max-w-lg mx-auto">
                      {isPreliminaryMode
                        ? `This looks like a first-time ${stockData.ticker} request. Live market analysis is available now, and the trained ticker model will appear here automatically when ready.`
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
                      Strong BUY/SELL signals stay locked until the trained model passes readiness checks.
                    </p>
                  </div>
                ) : (
                  <div className="overflow-x-auto table-container bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <table className="min-w-full text-left text-xs sm:text-sm">
                      <thead>
                        <tr className="bg-gradient-to-r from-gray-100 to-gray-50 dark:from-gray-700 dark:to-gray-800 text-gray-700 dark:text-gray-300 uppercase text-xs font-bold border-b-2 border-gray-200 dark:border-gray-600">
                          <th className="px-3 sm:px-4 py-3 sm:py-4">Date</th>
                          <th className="px-3 sm:px-4 py-3 sm:py-4">Signal</th>
                          <th className="px-3 sm:px-4 py-3 sm:py-4">Price Range</th>
                          <th className="px-3 sm:px-4 py-3 sm:py-4">Move Range</th>
                          <th className="px-3 sm:px-4 py-3 sm:py-4">Move % Range</th>
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
                            <td className="px-3 sm:px-4 py-3 sm:py-4 whitespace-nowrap">
                              {row.priceLow && row.priceHigh ? (
                                <div className="flex flex-col">
                                  <span className="text-xs font-bold text-gray-900 dark:text-gray-100">
                                    {currencySymbol}{row.priceLow} - {currencySymbol}{row.priceHigh}
                                  </span>
                                  <span className="text-[10px] text-gray-500 dark:text-gray-400 mt-0.5">
                                    Bear to bull
                                  </span>
                                </div>
                              ) : (
                                <span className="text-gray-400 text-xs">-</span>
                              )}
                            </td>
                            <td className={`px-3 sm:px-4 py-3 sm:py-4 font-bold whitespace-nowrap ${Number(row.changeHigh) >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                              {Number(row.changeLow) >= 0 ? '+' : ''}{currencySymbol}{row.changeLow}
                              <span className="text-gray-400 mx-1">-</span>
                              {Number(row.changeHigh) >= 0 ? '+' : ''}{currencySymbol}{row.changeHigh}
                            </td>
                            <td className={`px-3 sm:px-4 py-3 sm:py-4 font-bold whitespace-nowrap ${Number(row.changePctHigh) >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                              {Number(row.changePctLow) >= 0 ? '+' : ''}{row.changePctLow}%
                              <span className="text-gray-400 mx-1">-</span>
                              {Number(row.changePctHigh) >= 0 ? '+' : ''}{row.changePctHigh}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>

            {/* Analytics Deck */}
            <div className="columns-1 xl:columns-3 gap-4 sm:gap-5">
              {/* AI Price Forecast — Quantile Confidence Intervals */}
              {stockData.price_range_low?.length > 0 && (
                <div className="break-inside-avoid mb-4 sm:mb-5 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 border border-gray-200 dark:border-gray-700">
                  <h3 className="text-base sm:text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-600 dark:text-cyan-400" />
                    AI Price Forecast
                  </h3>
                  {(() => {
                    const lastIdx = Math.min(4, stockData.price_range_low.length - 1);
                    const low = Number(stockData.price_range_low[lastIdx]);
                    const high = Number(stockData.price_range_high[lastIdx]);
                    const q25 = stockData.price_range_q25?.[lastIdx] != null ? Number(stockData.price_range_q25[lastIdx]) : null;
                    const q75 = stockData.price_range_q75?.[lastIdx] != null ? Number(stockData.price_range_q75[lastIdx]) : null;
                    const current = Number(stockData.current_price);
                    const range = high - low || 1;

                    return (
                      <div className="space-y-4">
                        {/* Visual range bar */}
                        <div className="relative pt-1">
                          <div className="flex justify-between text-[10px] text-gray-500 dark:text-gray-400 mb-1">
                            <span>Bear Case</span>
                            <span>Forecast Range</span>
                            <span>Bull Case</span>
                          </div>
                          <div className="h-3 rounded-full bg-gradient-to-r from-red-400 via-gray-300 to-green-400 dark:from-red-600 dark:via-gray-600 dark:to-green-600 relative overflow-hidden">
                            {/* Current price marker */}
                            <div
                              className="absolute top-0 h-full w-0.5 bg-white dark:bg-gray-900 z-10"
                              style={{ left: `${Math.max(0, Math.min(100, ((current - low) / range) * 100))}%` }}
                              title={`Current: ${currencySymbol}${current.toFixed(2)}`}
                            />
                          </div>
                          <div className="flex justify-between text-xs font-bold mt-1">
                            <span className="text-red-600 dark:text-red-400">{currencySymbol}{low.toFixed(2)}</span>
                            <span className="text-green-600 dark:text-green-400">{currencySymbol}{high.toFixed(2)}</span>
                          </div>
                        </div>

                        {/* Quantile breakdown */}
                        <div className="grid grid-cols-2 gap-2">
                          <div className="bg-red-50 dark:bg-red-900/20 p-2.5 rounded-lg border border-red-200 dark:border-red-700/30">
                            <p className="text-[10px] font-bold uppercase text-red-600 dark:text-red-400">10th Pctl</p>
                            <p className="text-sm font-black text-gray-900 dark:text-white">{currencySymbol}{low.toFixed(2)}</p>
                          </div>
                          {q25 != null && (
                            <div className="bg-orange-50 dark:bg-orange-900/20 p-2.5 rounded-lg border border-orange-200 dark:border-orange-700/30">
                              <p className="text-[10px] font-bold uppercase text-orange-600 dark:text-orange-400">25th Pctl</p>
                              <p className="text-sm font-black text-gray-900 dark:text-white">{currencySymbol}{q25.toFixed(2)}</p>
                            </div>
                          )}
                          {q75 != null && (
                            <div className="bg-teal-50 dark:bg-teal-900/20 p-2.5 rounded-lg border border-teal-200 dark:border-teal-700/30">
                              <p className="text-[10px] font-bold uppercase text-teal-600 dark:text-teal-400">75th Pctl</p>
                              <p className="text-sm font-black text-gray-900 dark:text-white">{currencySymbol}{q75.toFixed(2)}</p>
                            </div>
                          )}
                          <div className="bg-green-50 dark:bg-green-900/20 p-2.5 rounded-lg border border-green-200 dark:border-green-700/30">
                            <p className="text-[10px] font-bold uppercase text-green-600 dark:text-green-400">90th Pctl</p>
                            <p className="text-sm font-black text-gray-900 dark:text-white">{currencySymbol}{high.toFixed(2)}</p>
                          </div>
                        </div>

                        <p className="text-[10px] text-gray-500 dark:text-gray-400 text-center">
                          5-day forecast range via quantile regression
                        </p>
                      </div>
                    );
                  })()}
                </div>
              )}

              {/* AI Reliability */}
              <div className="break-inside-avoid mb-4 sm:mb-5 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 border border-gray-200 dark:border-gray-700">
                <h3 className="text-base sm:text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                  <Activity className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-600 dark:text-cyan-400" />
                  AI Reliability
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/60 p-3">
                    <p className="text-[10px] font-black uppercase text-gray-500 dark:text-gray-400">Trust Score</p>
                    <p className={`text-2xl font-black mt-1 ${trustScore >= 75
                      ? 'text-green-600 dark:text-green-400'
                      : trustScore >= 55
                        ? 'text-cyan-600 dark:text-cyan-400'
                        : trustScore >= 35
                          ? 'text-amber-600 dark:text-amber-400'
                          : 'text-gray-500 dark:text-gray-400'
                      }`}>
                      {Number.isFinite(trustScore) ? trustScore : 0}
                      <span className="text-sm text-gray-400">/100</span>
                    </p>
                    <p className="text-xs font-bold text-gray-700 dark:text-gray-300 mt-1">{trustLabel}</p>
                  </div>
                  <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/60 p-3">
                    <p className="text-[10px] font-black uppercase text-gray-500 dark:text-gray-400">Market Regime</p>
                    <p className={`text-sm font-black mt-2 ${marketRegime.tone === 'bullish'
                      ? 'text-green-600 dark:text-green-400'
                      : marketRegime.tone === 'bearish'
                        ? 'text-red-600 dark:text-red-400'
                        : marketRegime.tone === 'warning'
                          ? 'text-amber-600 dark:text-amber-400'
                          : 'text-gray-700 dark:text-gray-300'
                      }`}>
                      {marketRegime.label || 'Unknown Regime'}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      Vol {marketRegime.daily_vol_pct ?? 'n/a'}% - 20D {marketRegime.momentum_20d_pct ?? 'n/a'}%
                    </p>
                  </div>
                </div>
                <div className="mt-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/60 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-black uppercase text-gray-500 dark:text-gray-400">Reliability</p>
                    <span className={`text-[10px] font-black uppercase px-2 py-1 rounded-md ${reliability.needs_retraining
                      ? 'bg-red-100 text-red-700 dark:bg-red-500/15 dark:text-red-300'
                      : reliabilityWarming
                        ? 'bg-cyan-100 text-cyan-700 dark:bg-cyan-500/15 dark:text-cyan-300'
                        : 'bg-green-100 text-green-700 dark:bg-green-500/15 dark:text-green-300'
                      }`}>
                      {reliabilityWarming ? 'Monitoring warming up' : reliabilityStatus}
                    </span>
                  </div>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                    {reliabilityWarming
                      ? `Real-world reliability activates after completed forecasts mature. Progress: ${reliabilityEvaluated}/${reliabilityMinimum} evaluated predictions.`
                      : Array.isArray(modelTrust.reasons) && modelTrust.reasons.length > 0
                        ? modelTrust.reasons.join('. ')
                        : reliability.reason || 'Reliability tracking will improve as predictions are evaluated.'}
                  </p>
                  {reliabilityWarming && (
                    <div className="mt-3 h-2 rounded-full bg-gray-200 dark:bg-gray-800 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-700"
                        style={{ width: `${Math.max(8, reliabilityProgress)}%` }}
                      />
                    </div>
                  )}
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/60 p-3">
                    <p className="text-[10px] font-black uppercase text-gray-500 dark:text-gray-400">Risk Engine</p>
                    <p className={`text-sm font-black mt-1 ${riskProfile.label === 'High' || riskProfile.label === 'Speculative'
                      ? 'text-red-600 dark:text-red-400'
                      : riskProfile.label === 'Medium'
                        ? 'text-amber-600 dark:text-amber-400'
                        : 'text-green-600 dark:text-green-400'
                      }`}>
                      {riskProfile.label || 'Tracking'} <span className="text-xs text-gray-400">{riskProfile.score ?? 'n/a'}/100</span>
                    </p>
                    <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-1">ATR {riskProfile.atr_pct ?? 'n/a'}% - DD {riskProfile.max_drawdown_pct ?? 'n/a'}%</p>
                  </div>
                  <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/60 p-3">
                    <p className="text-[10px] font-black uppercase text-gray-500 dark:text-gray-400">Calibration</p>
                    <p className="text-sm font-black text-gray-900 dark:text-white mt-1">
                      {probabilityCalibration.calibrated_direction_prob != null
                        ? `${(Number(probabilityCalibration.calibrated_direction_prob) * 100).toFixed(0)}%`
                        : 'Warming'}
                    </p>
                    <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-1">
                      Win {backtestMetrics.win_rate ?? 'n/a'}% - Sharpe {backtestMetrics.sharpe ?? 'n/a'}
                    </p>
                  </div>
                </div>
                <div className="mt-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/60 p-3">
                  <p className="text-xs font-black uppercase text-gray-500 dark:text-gray-400">
                    {predictionHorizon.label || `${stockData.days_predicted || days} Trading Days`} Thesis
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                    {aiExplanation[0] || 'Explanation will appear after the decision engine evaluates forecast edge, regime, risk, and sentiment.'}
                  </p>
                </div>
              </div>

              {/* Technical Indicators */}
              <div className="break-inside-avoid mb-4 sm:mb-5 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between gap-3 mb-3 sm:mb-4">
                  <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white">Technical Indicators</h3>
                  <span className="text-[10px] font-black uppercase tracking-wide text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-900 px-2.5 py-1 rounded-md border border-gray-200 dark:border-gray-700">
                    Feature Layer
                  </span>
                </div>

                <div className="space-y-3 sm:space-y-4">
                  {stockData.indicators?.rsi && (
                    <div className="bg-white dark:bg-gray-900/70 p-3 sm:p-4 rounded-xl border border-gray-200 dark:border-gray-700 indicator-card shadow-sm">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <span className="text-xs sm:text-sm font-bold text-gray-700 dark:text-gray-200">RSI (14)</span>
                          <p className="text-[10px] sm:text-xs text-gray-500 dark:text-gray-400 font-semibold mt-0.5">{rsiState}</p>
                        </div>
                        <span className={`text-base sm:text-lg font-bold indicator-value ${rsiValue >= 70
                          ? 'text-red-600 dark:text-red-400'
                          : rsiValue <= 30
                            ? 'text-green-600 dark:text-green-400'
                            : 'text-cyan-700 dark:text-cyan-400'
                          }`}>
                          {stockData.indicators.rsi}
                        </span>
                      </div>
                      <div className="h-12 sm:h-16">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={(stockData.indicator_trends?.rsi?.dates || []).slice(-20).map((date, i) => ({
                            value: (stockData.indicator_trends?.rsi?.values || []).slice(-20)[i]
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

                  {stockData.indicators?.ema && (
                    <div className="bg-white dark:bg-gray-900/70 p-3 sm:p-4 rounded-xl border border-gray-200 dark:border-gray-700 indicator-card shadow-sm">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <span className="text-xs sm:text-sm font-bold text-gray-700 dark:text-gray-200">EMA (20)</span>
                          <p className={`text-[10px] sm:text-xs font-semibold mt-0.5 ${currentPriceNumber >= emaValue ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                            {emaState}
                          </p>
                        </div>
                        <span className="text-base sm:text-lg font-bold text-blue-700 dark:text-blue-400 indicator-value">{currencySymbol}{stockData.indicators.ema}</span>
                      </div>
                      <div className="h-12 sm:h-16">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={(stockData.indicator_trends?.ema?.dates || []).slice(-20).map((date, i) => ({
                            value: (stockData.indicator_trends?.ema?.values || []).slice(-20)[i]
                          }))}>
                            <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}

                  {stockData.indicators?.macd && (
                    <div className="bg-white dark:bg-gray-900/70 p-3 sm:p-4 rounded-xl border border-gray-200 dark:border-gray-700 indicator-card shadow-sm">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <span className="text-xs sm:text-sm font-bold text-gray-700 dark:text-gray-200">MACD</span>
                          <p className={`text-[10px] sm:text-xs font-semibold mt-0.5 ${macdHistogram >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                            {macdState}
                          </p>
                        </div>
                        <span className="text-base sm:text-lg font-bold text-purple-700 dark:text-purple-400 indicator-value">{stockData.indicators.macd}</span>
                      </div>
                      <div className="h-12 sm:h-16">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart data={(stockData.indicator_trends?.macd?.dates || []).slice(-20).map((date, i) => ({
                            value: (stockData.indicator_trends?.macd?.values || []).slice(-20)[i],
                            histogram: (stockData.indicator_trends?.macd?.histogram || []).slice(-20)[i]
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
              <div className="break-inside-avoid mb-4 sm:mb-5 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 border border-gray-200 dark:border-gray-700">
                <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4">Stats</h3>
                <div className="space-y-2 sm:space-y-3">
                  <div className="flex justify-between items-center p-3 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800 rounded-lg border border-gray-200 dark:border-gray-600 shadow-sm">
                    <span className="text-xs sm:text-sm text-gray-700 dark:text-gray-300 font-semibold">Market Cap</span>
                    <span className="text-xs sm:text-sm font-bold text-gray-900 dark:text-white">
                      {formatCurrencyCompact(stockData.market_cap, currencySymbol)}
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
                      <span className="text-xs sm:text-sm font-bold text-green-600 dark:text-green-400">{currencySymbol}{stockData.day_high}</span>
                    </div>
                  )}
                  {stockData.day_low && (
                    <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-50 dark:bg-dark-elevated rounded-lg border border-gray-200 dark:border-dark-border">
                      <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 font-medium">Day Low</span>
                      <span className="text-xs sm:text-sm font-bold text-red-600 dark:text-red-400">{currencySymbol}{stockData.day_low}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Sentiment */}
              {(sentiment || stockData?.sentiment) && (
                <div className="break-inside-avoid mb-4 sm:mb-5 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 border border-gray-200 dark:border-gray-700">
                  <h3 className="text-base sm:text-lg lg:text-xl font-bold text-gray-900 dark:text-white mb-2">Stock Sentiment</h3>
                  <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Real-time market sentiment analysis
                  </p>
                  <SentimentGauge sentiment={sentiment || stockData.sentiment} />
                </div>
              )}

              {/* News */}
              {news.length > 0 && (
                <div className="break-inside-avoid mb-4 sm:mb-5 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 border border-gray-200 dark:border-gray-700">
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

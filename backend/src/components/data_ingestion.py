import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ta
import os
from datetime import datetime, timedelta
import sys
import json

# Add parent directory to path to import stock_api
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from stock_api import get_stock_history


class DataIngestion:
    """
    Fetches real stock data from APIs and engineers 20+ technical features
    for production-grade LSTM and ensemble model training.
    """
    
    # Feature columns (excluding target). This is the canonical feature list.
    # CRITICAL: All features are now stationary (relative/ratios) to avoid prediction drift.
    FEATURE_COLUMNS = [
        'returns',          # Target is now returns (index 0)
        'log_returns',
        'sma_5_ratio', 'sma_10_ratio', 'sma_20_ratio', 'sma_50_ratio',
        'ema_12_ratio', 'ema_26_ratio',
        'macd_ratio', 'macd_signal_ratio', 'macd_hist_ratio',
        'rsi',              # RSI is naturally stationary [0, 100]
        'bb_middle_ratio', 'bb_upper_ratio', 'bb_lower_ratio', 'bb_width_ratio',
        'atr_ratio',
        'volatility',
        'volume_ratio', 'obv',
        'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5',
    ]
    
    def __init__(self, ticker, days=730):
        """
        Args:
            ticker: Stock symbol (Required)
            days: Number of calendar days of history (default 730 = ~2 years)
        """
        self.ticker = ticker
        self.days = days
        
    def fetch_data(self):
        """Fetch stock data using the multi-provider API (Twelve Data → Finnhub → Alpha Vantage → yfinance)"""
        print(f"📊 Fetching {self.ticker} data ({self.days} days)...")
        
        data = get_stock_history(self.ticker, days=min(self.days, 5000))
        
        if data.empty:
            raise ValueError(f"No data fetched for {self.ticker}")
        
        # Normalize column names — API may return lowercase
        col_map = {c: c.capitalize() for c in data.columns 
                   if c.lower() in ('close', 'open', 'high', 'low', 'volume')}
        data = data.rename(columns=col_map)
        
        print(f"✅ Fetched {len(data)} data points for {self.ticker}")
        return data
    
    def preprocess(self, data):
        """
        Engineer 20+ technical indicator features from OHLCV data.
        
        All features are computed using the `ta` library and converted to 
        relative ratios to ensure stationarity across price levels.
        """
        df = data.copy()
        
        if 'Close' not in df.columns:
            raise ValueError(f"'Close' column not found. Columns: {df.columns.tolist()}")
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze() if 'High' in df.columns else close
        low = df['Low'].squeeze() if 'Low' in df.columns else close
        volume = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series(0, index=df.index)
        
        # --- Price-based features (Naturally Stationary) ---
        df['returns'] = close.pct_change()
        df['log_returns'] = np.log(close / close.shift(1))
        
        # --- Relative Moving Averages (Indicators / Close) ---
        df['sma_5_ratio'] = close.rolling(window=5).mean() / close
        df['sma_10_ratio'] = close.rolling(window=10).mean() / close
        df['sma_20_ratio'] = close.rolling(window=20).mean() / close
        df['sma_50_ratio'] = close.rolling(window=50).mean() / close
        df['ema_12_ratio'] = close.ewm(span=12, adjust=False).mean() / close
        df['ema_26_ratio'] = close.ewm(span=26, adjust=False).mean() / close
        
        # --- MACD Ratios ---
        macd_indicator = ta.trend.MACD(close)
        df['macd_ratio'] = macd_indicator.macd() / close
        df['macd_signal_ratio'] = macd_indicator.macd_signal() / close
        df['macd_hist_ratio'] = macd_indicator.macd_diff() / close
        
        # --- RSI (Naturally Stationary) ---
        df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        
        # --- Bollinger Bands Ratios ---
        bb = ta.volatility.BollingerBands(close, window=20)
        df['bb_middle_ratio'] = bb.bollinger_mavg() / close
        df['bb_upper_ratio'] = bb.bollinger_hband() / close
        df['bb_lower_ratio'] = bb.bollinger_lband() / close
        df['bb_width_ratio'] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
        
        # --- ATR Ratio ---
        df['atr_ratio'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range() / close
        
        # --- Volatility ---
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # --- Volume features ---
        df['volume_sma_20'] = volume.rolling(window=20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20'].replace(0, np.nan)
        
        # Stationary OBV
        direction = np.sign(close.diff().fillna(0))
        volume_avg = volume.rolling(window=20).mean().replace(0, np.nan)
        df['obv'] = (volume * direction) / volume_avg.fillna(volume.mean() + 1e-9)
        
        # --- Lag features ---
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Drop NaN rows (from indicators needing warmup period)
        df = df.dropna()
        
        # Select only the canonical feature columns
        available_features = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        
        print(f"🔧 Engineered {len(available_features)} stationary features, {len(df)} samples")
        
        return df[available_features]
    
    def split_data(self, data, test_size=0.2):
        """Time-series split (no shuffle — preserves temporal order)"""
        train, test = train_test_split(data, test_size=test_size, shuffle=False)
        print(f"📊 Split: Train={len(train)} | Test={len(test)}")
        return train, test
    
    
    def initiate_data_ingestion(self):
        """Full pipeline: fetch → preprocess → split"""
        data = self.fetch_data()
        processed_data = self.preprocess(data)
        train_data, test_data = self.split_data(processed_data)
        return train_data, test_data
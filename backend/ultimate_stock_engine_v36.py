#!/usr/bin/env python3
"""
AI STOCK PREDICTOR — ULTIMATE ENGINE v3.6 FIXED
================================================
Fixed: Chart generation actuals parameter bug
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            accuracy_score, confusion_matrix, roc_auc_score)
import xgboost as xgb
import lightgbm as lgb
import os
import sys
import json
import joblib

# Chart libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add paths to import your existing modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from stock_api import get_stock_history
    HAS_CUSTOM_API = True
except:
    HAS_CUSTOM_API = False

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKERS = ['NVDA', 'AAPL', 'GOOG', 'MSFT', 'AMZN']
ULTIMATE_REGISTRY_DIR = os.path.join('mlops', 'ultimate_engine')
ULTIMATE_MODEL_CACHE = {}

# Walk-Forward Parameters
MIN_TRAIN_DAYS = 504
TEST_WINDOW = 63
STEP_SIZE = 21
PURGE_DAYS = 5

# Trading Parameters
PREDICTION_HORIZON = 5
ENTRY_THRESHOLD_BASE = 0.55
EXIT_THRESHOLD_BASE = 0.45
MIN_HOLD_DAYS = 5
MAX_HOLD_DAYS = 30
TRAILING_STOP_PCT = 0.08
STOP_LOSS_PCT = 0.05

# Transaction Costs
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.0005
TOTAL_COST_PER_TRADE = COMMISSION_PCT + SLIPPAGE_PCT

# Risk Management
VOL_TARGET = 0.15
MAX_LEVERAGE = 1.5

# Regime Detection
REGIME_WINDOW = 60
BULL_THRESHOLD = 0.10
BEAR_THRESHOLD = -0.05

# Model Parameters (heavily regularized)
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.02,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
    'min_child_weight': 10,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': 42
}

LGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.02,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
    'min_child_samples': 50,
    'verbose': -1,
    'random_state': 42
}


def _ultimate_paths(ticker):
    safe_ticker = ticker.upper().replace('/', '_').replace('\\', '_')
    model_dir = os.path.join(ULTIMATE_REGISTRY_DIR, safe_ticker)
    return {
        'dir': model_dir,
        'artifact': os.path.join(model_dir, 'model.joblib'),
        'metadata': os.path.join(model_dir, 'metadata.json')
    }


def _normalize_ohlcv_for_engine(df):
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if 'Date' not in data.columns:
        data = data.reset_index()
    if 'datetime' in data.columns:
        data = data.rename(columns={'datetime': 'Date'})
    if 'index' in data.columns and 'Date' not in data.columns:
        data = data.rename(columns={'index': 'Date'})
    data['Date'] = pd.to_datetime(data['Date'])
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in data.columns:
            if col == 'Volume':
                data[col] = 0
            else:
                data[col] = data['Close']
    return data.sort_values('Date').reset_index(drop=True)


def _select_feature_columns(data):
    exclude = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
               'target', 'regime', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
               'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
               'bb_middle', 'bb_upper', 'bb_lower', 'macd', 'macd_signal', 'macd_hist',
               'atr_14', 'volume_sma_5', 'volume_sma_20',
               'force_index', 'vpt', 'dollar_volume']
    return [c for c in data.columns if c not in exclude]


def _fit_final_models(X, y, regimes, use_regime=True):
    if use_regime:
        models = train_regime_aware_models(X, y, regimes)
        if models:
            return {'type': 'regime_aware', 'models': models}

    models = {
        'xgb': xgb.XGBClassifier(**XGB_PARAMS),
        'lgbm': lgb.LGBMClassifier(**LGB_PARAMS),
        'rf': RandomForestClassifier(**RF_PARAMS),
        'gb': GradientBoostingClassifier(**GB_PARAMS)
    }
    for model in models.values():
        model.fit(X, y)
    return {'type': 'ensemble', 'models': models}


def _predict_final_models(model_bundle, X, regimes):
    if model_bundle['type'] == 'regime_aware':
        return predict_regime_aware(model_bundle['models'], X, regimes)

    models = model_bundle['models']
    meta = np.zeros((len(X), len(models)))
    for i, model in enumerate(models.values()):
        meta[:, i] = model.predict_proba(X)[:, 1]
    return np.mean(meta, axis=1)


def _best_threshold(y_true, probs):
    thresholds = np.arange(0.45, 0.66, 0.01)
    best = {'threshold': 0.55, 'score': 0.0}
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best['score']:
            best = {'threshold': float(threshold), 'score': float(score)}
    sell_threshold = max(0.35, min(0.49, 1.0 - best['threshold']))
    return float(best['threshold']), float(sell_threshold)


def save_ultimate_model(ticker, artifact):
    """Save model locally AND to MLflow/DagsHub for cloud persistence."""
    ticker = ticker.upper()
    paths = _ultimate_paths(ticker)
    os.makedirs(paths['dir'], exist_ok=True)
    joblib.dump(artifact, paths['artifact'])
    metadata = {
        'ticker': ticker,
        'model_type': 'ultimate_regime_ensemble_v36',
        'version': artifact.get('version', 'v36'),
        'trained_at': artifact.get('trained_at'),
        'metrics': artifact.get('metrics', {}),
        'feature_count': len(artifact.get('feature_cols', [])),
        'prediction_horizon': PREDICTION_HORIZON,
        'artifact_path': paths['artifact']
    }
    with open(paths['metadata'], 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    ULTIMATE_MODEL_CACHE[ticker] = artifact

    # --- Log to MLflow / DagsHub (cloud persistence) ---
    try:
        import mlflow
        from mlops.config import MLOpsConfig
        mlflow.set_tracking_uri(MLOpsConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"ultimate_engine_v36")

        with mlflow.start_run(run_name=f"v36-{ticker}-{datetime.utcnow().strftime('%Y%m%d')}"):
            # Log metrics
            metrics = artifact.get('metrics', {})
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(key, float(val))

            # Log params
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("version", artifact.get('version', 'v36'))
            mlflow.log_param("feature_count", len(artifact.get('feature_cols', [])))
            mlflow.log_param("prediction_horizon", PREDICTION_HORIZON)

            # Log model artifact to DagsHub
            mlflow.log_artifact(paths['artifact'], artifact_path=f"models/{ticker}")
            mlflow.log_artifact(paths['metadata'], artifact_path=f"models/{ticker}")

            # Log charts if available
            charts = artifact.get('charts', {})
            for chart_name, chart_path in charts.items():
                if chart_path and os.path.exists(chart_path):
                    mlflow.log_artifact(chart_path, artifact_path=f"charts/{ticker}")

        print(f"  ☁️ Model logged to MLflow/DagsHub for {ticker}")
    except Exception as mlflow_err:
        print(f"  ⚠️ MLflow logging skipped for {ticker}: {mlflow_err}")

    return metadata


def load_ultimate_model(ticker):
    """Load model from cache → local disk → MLflow/DagsHub (cloud)."""
    ticker = ticker.upper()
    if ticker in ULTIMATE_MODEL_CACHE:
        return ULTIMATE_MODEL_CACHE[ticker]

    # Try local disk first
    paths = _ultimate_paths(ticker)
    if os.path.exists(paths['artifact']):
        artifact = joblib.load(paths['artifact'])
        ULTIMATE_MODEL_CACHE[ticker] = artifact
        return artifact

    # Try MLflow/DagsHub (for cloud deployment where local models don't exist)
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlops.config import MLOpsConfig
        mlflow.set_tracking_uri(MLOpsConfig.MLFLOW_TRACKING_URI)

        client = MlflowClient()
        experiment = client.get_experiment_by_name("ultimate_engine_v36")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.ticker = '{ticker}'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                run = runs[0]
                # Download model artifact from DagsHub
                artifact_uri = f"{run.info.artifact_uri}/models/{ticker}/model.joblib"
                local_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=artifact_uri,
                    dst_path=paths['dir']
                )
                if os.path.exists(local_path):
                    artifact = joblib.load(local_path)
                    ULTIMATE_MODEL_CACHE[ticker] = artifact
                    print(f"  ☁️ Loaded {ticker} model from MLflow/DagsHub")
                    return artifact
    except Exception as mlflow_err:
        print(f"  ⚠️ MLflow load failed for {ticker}: {mlflow_err}")

    return None

RF_PARAMS = {
    'n_estimators': 80,
    'max_depth': 4,
    'min_samples_split': 30,
    'min_samples_leaf': 15,
    'max_features': 'sqrt',
    'random_state': 42
}

GB_PARAMS = {
    'n_estimators': 80,
    'max_depth': 2,
    'learning_rate': 0.03,
    'subsample': 0.6,
    'min_samples_split': 30,
    'random_state': 42
}

# =============================================================================
# REGIME DETECTION
# =============================================================================

class RegimeDetector:
    def __init__(self, window=REGIME_WINDOW):
        self.window = window
        self.regime_history = []
        
    def detect(self, prices, dates):
        returns = pd.Series(prices).pct_change()
        rolling_ret = returns.rolling(window=self.window).mean() * 252
        rolling_vol = returns.rolling(window=self.window).std() * np.sqrt(252)
        rolling_trend = (pd.Series(prices) / pd.Series(prices).rolling(window=self.window).mean() - 1)
        
        regimes = []
        for i in range(len(prices)):
            if i < self.window:
                regimes.append(1)
                continue
                
            ret = rolling_ret.iloc[i]
            vol = rolling_vol.iloc[i]
            trend = rolling_trend.iloc[i]
            
            sharpe = ret / (vol + 1e-10)
            
            if ret > BULL_THRESHOLD and trend > 0.05:
                regimes.append(2)
            elif ret < BEAR_THRESHOLD and trend < -0.03:
                regimes.append(0)
            else:
                regimes.append(1)
                
        return np.array(regimes)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    data = df.copy()
    
    # Basic returns
    data['returns_1d'] = data['Close'].pct_change()
    data['returns_2d'] = data['Close'].pct_change(2)
    data['returns_5d'] = data['Close'].pct_change(5)
    data['returns_10d'] = data['Close'].pct_change(10)
    data['returns_20d'] = data['Close'].pct_change(20)
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Moving average ratios
    for window in [5, 10, 20, 50, 200]:
        sma = data['Close'].rolling(window=window).mean()
        data[f'sma_{window}'] = sma
        data[f'sma_{window}_ratio'] = data['Close'] / sma
        ema = data['Close'].ewm(span=window, adjust=False).mean()
        data[f'ema_{window}'] = ema
        data[f'ema_{window}_ratio'] = data['Close'] / ema
    
    # Volatility
    for window in [10, 20, 60]:
        data[f'volatility_{window}d'] = data['returns_1d'].rolling(window=window).std() * np.sqrt(252)
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    data['macd_ratio'] = data['macd'] / data['Close']
    
    # Bollinger Bands
    data['bb_middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_pct'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-10)
    data['bb_position'] = data['bb_pct']
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / (data['bb_middle'] + 1e-10)
    
    # ATR
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['atr_14'] = true_range.rolling(14).mean()
    data['atr_ratio'] = data['atr_14'] / data['Close']
    
    # Volume
    data['volume_sma_5'] = data['Volume'].rolling(window=5).mean()
    data['volume_sma_20'] = data['Volume'].rolling(window=20).mean()
    data['volume_ratio_5'] = data['Volume'] / data['volume_sma_5']
    data['volume_ratio_20'] = data['Volume'] / data['volume_sma_20']
    data['dollar_volume'] = data['Close'] * data['Volume']
    
    # Price position
    data['daily_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-10)
    
    # Overnight gap
    data['overnight_gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    
    # Momentum
    data['momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
    data['momentum_10d'] = data['Close'] / data['Close'].shift(10) - 1
    data['momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
    
    # Trend strength
    data['adx'] = np.abs(data['Close'] - data['Close'].shift(10)) / \
                  (data['High'].rolling(10).max() - data['Low'].rolling(10).min() + 1e-10)
    
    # Support/Resistance
    data['dist_to_support'] = (data['Close'] - data['Low'].rolling(20).min()) / data['Close']
    data['dist_to_resistance'] = (data['High'].rolling(20).max() - data['Close']) / data['Close']
    
    # Statistical moments
    for window in [10, 20]:
        data[f'return_skew_{window}'] = data['returns_1d'].rolling(window).skew()
        data[f'return_kurt_{window}'] = data['returns_1d'].rolling(window).kurt()
    
    # Force Index
    data['force_index'] = data['Close'].diff(1) * data['Volume']
    data['force_index_ema'] = data['force_index'].ewm(span=13, adjust=False).mean()
    
    # VPT
    data['vpt'] = (data['Close'].pct_change() * data['Volume']).cumsum()
    
    # TARGET
    future_return = data['Close'].shift(-PREDICTION_HORIZON) / data['Close'] - 1
    data['target'] = np.where(future_return.notna(), (future_return > 0).astype(int), np.nan)
    
    return data

# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_split(data, min_train_days, test_window, step_size, purge_days):
    n = len(data)
    splits = []
    start = min_train_days
    
    while start + test_window + purge_days <= n:
        train_end = start - purge_days
        if train_end < min_train_days:
            start += step_size
            continue
            
        train_idx = list(range(0, train_end))
        test_idx = list(range(start, min(start + test_window, n)))
        
        if len(test_idx) > 5:
            splits.append((train_idx, test_idx))
        
        start += step_size
    
    return splits

# =============================================================================
# MODEL TRAINING WITH REGIME AWARENESS
# =============================================================================

def train_regime_aware_models(X_train, y_train, regimes_train):
    unique_regimes = np.unique(regimes_train)
    regime_models = {}
    
    for regime in unique_regimes:
        mask = regimes_train == regime
        if mask.sum() < 100:
            continue
            
        X_regime = X_train[mask]
        y_regime = y_train[mask]
        
        print(f"   Training regime {regime} model ({mask.sum()} samples)...")
        
        models = {}
        models['xgb'] = xgb.XGBClassifier(**XGB_PARAMS)
        models['xgb'].fit(X_regime, y_regime)
        
        models['lgbm'] = lgb.LGBMClassifier(**LGB_PARAMS)
        models['lgbm'].fit(X_regime, y_regime)
        
        models['rf'] = RandomForestClassifier(**RF_PARAMS)
        models['rf'].fit(X_regime, y_regime)
        
        models['gb'] = GradientBoostingClassifier(**GB_PARAMS)
        models['gb'].fit(X_regime, y_regime)
        
        regime_models[regime] = models
    
    return regime_models

def predict_regime_aware(regime_models, X_test, regimes_test):
    predictions = np.zeros(len(X_test))
    
    for regime, models in regime_models.items():
        mask = regimes_test == regime
        if mask.sum() == 0:
            continue
            
        meta = np.zeros((mask.sum(), len(models)))
        for i, (name, model) in enumerate(models.items()):
            meta[:, i] = model.predict_proba(X_test[mask])[:, 1]
        
        predictions[mask] = np.mean(meta, axis=1)
    
    zero_mask = predictions == 0
    if zero_mask.sum() > 0:
        all_models = []
        for models in regime_models.values():
            all_models.extend(models.values())
        
        if all_models:
            meta = np.zeros((zero_mask.sum(), len(all_models)))
            for i, model in enumerate(all_models):
                meta[:, i] = model.predict_proba(X_test[zero_mask])[:, 1]
            predictions[zero_mask] = np.mean(meta, axis=1)
    
    return predictions

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class AdvancedBacktester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.entry_date = None
        self.holding_days = 0
        self.peak_price = 0
        self.trades = []
        self.equity_curve = []
        self.pred_history = []
        self.entry_threshold = ENTRY_THRESHOLD_BASE
        self.exit_threshold = EXIT_THRESHOLD_BASE
        
    def update_thresholds(self):
        if len(self.pred_history) >= 63:
            recent = self.pred_history[-63:]
            self.entry_threshold = np.percentile(recent, 65)
            self.exit_threshold = np.percentile(recent, 35)
            self.entry_threshold = max(min(self.entry_threshold, 0.58), 0.52)
            self.exit_threshold = max(min(self.exit_threshold, 0.48), 0.35)
    
    def calculate_position_size(self, volatility):
        if volatility <= 0 or np.isnan(volatility):
            return 0.5
        target = VOL_TARGET / volatility
        return min(target, MAX_LEVERAGE)
    
    def execute_trade(self, date, price, direction, reason, position_size=1.0):
        if direction == 1:
            self.position = 1
            self.entry_price = price * (1 + SLIPPAGE_PCT)
            self.entry_date = date
            self.holding_days = 0
            self.peak_price = self.entry_price
            self.position_size = position_size
            
        elif direction == -1:
            if self.position == 1:
                exit_price = price * (1 - SLIPPAGE_PCT)
                gross_return = (exit_price / self.entry_price) - 1
                sized_return = gross_return * self.position_size
                net_return = sized_return - TOTAL_COST_PER_TRADE
                
                self.trades.append({
                    'entry_date': self.entry_date,
                    'exit_date': date,
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'return': net_return,
                    'holding_days': self.holding_days,
                    'reason': reason,
                    'position_size': self.position_size
                })
                
                self.capital *= (1 + net_return)
            
            self.position = 0
            self.entry_price = 0
            self.holding_days = 0
            
        self.equity_curve.append({
            'date': date,
            'capital': self.capital,
            'position': self.position
        })
    
    def run(self, dates, prices, predictions, volatilities, regimes=None):
        for i in range(len(dates)):
            date = dates[i]
            price = prices[i]
            pred = predictions[i]
            vol = volatilities[i]
            
            self.pred_history.append(pred)
            self.update_thresholds()
            
            if regimes is not None:
                regime = regimes[i]
                if regime == 2:
                    self.entry_threshold = max(self.entry_threshold, 0.58)
                elif regime == 0:
                    self.entry_threshold = min(self.entry_threshold, 0.52)
            
            if self.position == 1:
                self.holding_days += 1
                self.peak_price = max(self.peak_price, price)
                
                current_return = (price / self.entry_price) - 1
                drawdown = (self.peak_price - price) / self.peak_price
                
                if current_return < -STOP_LOSS_PCT:
                    self.execute_trade(date, price, -1, 'stop_loss')
                    continue
                
                if drawdown > TRAILING_STOP_PCT and current_return > 0:
                    self.execute_trade(date, price, -1, 'trailing_stop')
                    continue
                
                if self.holding_days >= MAX_HOLD_DAYS:
                    self.execute_trade(date, price, -1, 'max_hold')
                    continue
                
                if pred < self.exit_threshold and self.holding_days >= MIN_HOLD_DAYS:
                    self.execute_trade(date, price, -1, 'signal_exit')
                    continue
            
            elif self.position == 0:
                if pred > self.entry_threshold:
                    position_size = self.calculate_position_size(vol)
                    self.execute_trade(date, price, 1, 'signal_entry', position_size)
            
            if len(self.equity_curve) == 0 or self.equity_curve[-1]['date'] != date:
                self.equity_curve.append({
                    'date': date,
                    'capital': self.capital,
                    'position': self.position
                })
        
        if self.position == 1:
            self.execute_trade(dates[-1], prices[-1], -1, 'end_of_data')
        
        return self.get_metrics()
    
    def get_metrics(self):
        if len(self.trades) == 0:
            return {
                'total_return': 0.0, 'annualized_return': 0.0, 'sharpe_ratio': 0.0,
                'win_rate': 0.0, 'max_drawdown': 0.0, 'num_trades': 0,
                'avg_return_per_trade': 0.0, 'profit_factor': 0.0
            }
        
        returns = [t['return'] for t in self.trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        total_return = (self.capital / self.initial_capital) - 1
        
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 1:
            days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
            years = max(days / 365.25, 0.01)
            ann_return = (1 + total_return) ** (1 / years) - 1
            
            equity_df['daily_return'] = equity_df['capital'].pct_change()
            daily_vol = equity_df['daily_return'].std()
            sharpe = (equity_df['daily_return'].mean() / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0
            
            equity_df['peak'] = equity_df['capital'].cummax()
            equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']
            max_dd = abs(equity_df['drawdown'].min())
        else:
            ann_return = sharpe = max_dd = 0
        
        win_rate = len(wins) / len(returns) if returns else 0
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        
        return {
            'total_return': total_return * 100,
            'annualized_return': ann_return * 100,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate * 100,
            'max_drawdown': max_dd * 100,
            'num_trades': len(self.trades),
            'avg_return_per_trade': np.mean(returns) * 100,
            'profit_factor': gross_profit / gross_loss
        }

# =============================================================================
# CHART GENERATION - FIXED VERSION
# =============================================================================

class ChartGenerator:
    
    @staticmethod
    def generate_all_charts(ticker, dates, prices, predictions, actuals, trades, 
                          metrics, regimes, fold_results, save_dir='charts'):
        """Generate complete chart suite - FIXED with actuals parameter"""
        os.makedirs(save_dir, exist_ok=True)
        
        charts = {}
        
        # 1. Main chart: Price + Trades + Predictions + Actuals
        charts['main'] = ChartGenerator.plot_price_trades(
            ticker, dates, prices, predictions, actuals, trades, regimes,
            os.path.join(save_dir, f'{ticker}_main_analysis.png')
        )
        
        # 2. Equity Curve
        charts['equity'] = ChartGenerator.plot_equity_curve(
            trades, metrics,
            os.path.join(save_dir, f'{ticker}_equity.png')
        )
        
        # 3. Temporal Consistency
        if fold_results:
            charts['temporal'] = ChartGenerator.plot_temporal_consistency(
                fold_results,
                os.path.join(save_dir, f'{ticker}_temporal.png')
            )
        
        # 4. Regime Performance
        if regimes is not None:
            charts['regime'] = ChartGenerator.plot_regime_performance(
                dates, prices, predictions, actuals, regimes,
                os.path.join(save_dir, f'{ticker}_regimes.png')
            )
        
        # 5. Prediction Distribution
        charts['distribution'] = ChartGenerator.plot_prediction_distribution(
            predictions, actuals,
            os.path.join(save_dir, f'{ticker}_distribution.png')
        )
        
        return charts
    
    @staticmethod
    def plot_price_trades(ticker, dates, prices, predictions, actuals, trades, regimes, save_path):
        """FIXED: Added actuals parameter"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 14), 
                                gridspec_kw={'height_ratios': [3, 1, 1]})
        
        dates = pd.to_datetime(dates)
        
        # Chart 1: Price + Trades + Regimes
        ax1 = axes[0]
        
        # Plot regime backgrounds
        if regimes is not None:
            for i in range(len(dates)-1):
                if regimes[i] == 2:
                    color = '#00ff8815'
                elif regimes[i] == 0:
                    color = '#ff444415'
                else:
                    color = '#ffff0015'
                ax1.axvspan(dates[i], dates[i+1], alpha=1, color=color)
        
        # Plot price
        ax1.plot(dates, prices, color='white', linewidth=1.2, label='Close Price')
        
        # Plot trades
        for trade in trades:
            entry_dt = pd.to_datetime(trade['entry_date'])
            exit_dt = pd.to_datetime(trade['exit_date'])
            
            entry_idx = np.argmin(np.abs(dates - entry_dt))
            exit_idx = np.argmin(np.abs(dates - exit_dt))
            
            color = '#00ff88' if trade['return'] > 0 else '#ff4444'
            ax1.scatter(dates[entry_idx], prices[entry_idx], color='#00ff88', 
                       marker='^', s=120, zorder=5, edgecolors='white', linewidth=1)
            ax1.scatter(dates[exit_idx], prices[exit_idx], color='#ff4444', 
                       marker='v', s=120, zorder=5, edgecolors='white', linewidth=1)
            
            ax1.plot([dates[entry_idx], dates[exit_idx]], 
                    [prices[entry_idx], prices[exit_idx]], 
                    color=color, alpha=0.6, linewidth=2.5)
        
        ax1.set_title(f'{ticker} - Price Action & Trades', color='white', 
                     fontweight='bold', fontsize=14)
        ax1.set_ylabel('Price ($)', color='white')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)
        
        # Chart 2: Prediction Probability
        ax2 = axes[1]
        ax2.fill_between(dates, predictions, 0.5, where=(predictions > 0.5), 
                        alpha=0.3, color='#00ff88', label='Bullish Signal')
        ax2.fill_between(dates, predictions, 0.5, where=(predictions <= 0.5), 
                        alpha=0.3, color='#ff4444', label='Bearish Signal')
        ax2.axhline(y=0.5, color='white', linestyle='--', alpha=0.5)
        ax2.axhline(y=ENTRY_THRESHOLD_BASE, color='#00ff88', linestyle=':', alpha=0.5)
        ax2.axhline(y=EXIT_THRESHOLD_BASE, color='#ff4444', linestyle=':', alpha=0.5)
        ax2.plot(dates, predictions, color='#00ccff', linewidth=1)
        ax2.set_ylabel('P(UP)', color='white')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.2)
        
        # Chart 3: Direction Accuracy - FIXED with actuals
        ax3 = axes[2]
        correct = (actuals == (predictions > 0.5).astype(int))
        colors = ['#00ff88' if c else '#ff4444' for c in correct]
        ax3.bar(dates, actuals * 2 - 1, color=colors, alpha=0.7, width=1)
        ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        ax3.set_ylabel('Actual\nDirection', color='white')
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_xlabel('Date', color='white')
        ax3.grid(True, alpha=0.2)
        
        # Style all axes
        for ax in axes:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        fig.patch.set_facecolor('#1a1a2e')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        
        return save_path
    
    @staticmethod
    def plot_equity_curve(trades, metrics, save_path):
        """Plot equity curve and drawdown"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        if len(trades) == 0:
            return None
        
        # Equity curve
        equity_dates = [pd.to_datetime(t['exit_date']) for t in trades]
        equity_values = [100000]
        for t in trades:
            equity_values.append(equity_values[-1] * (1 + t['return']))
        equity_values = equity_values[1:]
        
        ax1 = axes[0]
        ax1.plot(equity_dates, equity_values, color='#00ff88', linewidth=2.5)
        ax1.fill_between(equity_dates, equity_values, 100000, 
                        alpha=0.2, color='#00ff88')
        ax1.axhline(y=100000, color='white', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Portfolio Value ($)', color='white')
        ax1.set_title(f'Equity Curve | Return: {metrics["total_return"]:+.1f}% | Sharpe: {metrics["sharpe_ratio"]:.2f}', 
                     color='white', fontweight='bold')
        ax1.grid(True, alpha=0.2)
        
        # Drawdown
        equity_series = pd.Series(equity_values, index=equity_dates)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100
        
        ax2 = axes[1]
        ax2.fill_between(equity_dates, drawdown, 0, alpha=0.4, color='#ff4444')
        ax2.plot(equity_dates, drawdown, color='#ff4444', linewidth=1.5)
        ax2.set_ylabel('Drawdown (%)', color='white')
        ax2.set_xlabel('Date', color='white')
        ax2.set_title(f'Max Drawdown: {metrics["max_drawdown"]:.1f}%', color='white', fontweight='bold')
        ax2.grid(True, alpha=0.2)
        
        for ax in axes:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
        
        fig.patch.set_facecolor('#1a1a2e')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        
        return save_path
    
    @staticmethod
    def plot_temporal_consistency(fold_results, save_path):
        """Plot temporal consistency"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        fold_nums = range(1, len(fold_results) + 1)
        accuracies = [r['accuracy'] for r in fold_results]
        mean_probs = [r['mean_prob'] for r in fold_results]
        
        # Accuracy over time
        ax1 = axes[0]
        ax1.plot(fold_nums, accuracies, marker='o', color='#00ccff', 
                linewidth=2, markersize=6, label='Fold Accuracy')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        ax1.axhline(y=np.mean(accuracies), color='#00ff88', linestyle='--', 
                   alpha=0.7, label=f'Mean: {np.mean(accuracies):.1%}')
        ax1.fill_between(fold_nums, accuracies, 0.5, 
                        where=[a > 0.5 for a in accuracies], 
                        alpha=0.2, color='#00ff88')
        ax1.set_ylabel('Accuracy', color='white')
        ax1.set_title('Temporal Consistency: Accuracy Over Folds', color='white', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
        # Prediction drift
        ax2 = axes[1]
        ax2.plot(fold_nums, mean_probs, marker='s', color='#ffaa00', 
                linewidth=2, markersize=6)
        ax2.axhline(y=0.5, color='white', linestyle='--', alpha=0.5)
        ax2.axhline(y=np.mean(mean_probs), color='#ffaa00', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Mean Prediction', color='white')
        ax2.set_xlabel('Fold Number', color='white')
        ax2.set_title('Prediction Distribution Drift', color='white', fontweight='bold')
        ax2.grid(True, alpha=0.2)
        
        for ax in axes:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
        
        fig.patch.set_facecolor('#1a1a2e')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        
        return save_path
    
    @staticmethod
    def plot_regime_performance(dates, prices, predictions, actuals, regimes, save_path):
        """Plot performance by regime - FIXED with actuals"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        dates = pd.to_datetime(dates)
        
        regime_names = {0: 'Bear', 1: 'Neutral', 2: 'Bull'}
        regime_colors = {0: '#ff4444', 1: '#ffff00', 2: '#00ff88'}
        
        for regime in [0, 1, 2]:
            mask = regimes == regime
            if mask.sum() == 0:
                continue
            
            pred_dir = (predictions[mask] > 0.5).astype(int)
            acc = (pred_dir == actuals[mask]).mean()
            
            ax.scatter(dates[mask], prices[mask], c=regime_colors[regime], 
                      alpha=0.3, s=10, label=f'{regime_names[regime]} (Acc: {acc:.1%})')
        
        ax.plot(dates, prices, color='white', linewidth=0.5, alpha=0.5)
        ax.set_ylabel('Price ($)', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_title('Performance by Market Regime', color='white', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        
        return save_path
    
    @staticmethod
    def plot_prediction_distribution(predictions, actuals, save_path):
        """Plot prediction distribution by actual class"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Predictions when actual = 1 (Up)
        pred_up = predictions[actuals == 1]
        # Predictions when actual = 0 (Down)
        pred_down = predictions[actuals == 0]
        
        ax.hist(pred_up, bins=30, alpha=0.6, color='#00ff88', label=f'Actual UP (n={len(pred_up)})', density=True)
        ax.hist(pred_down, bins=30, alpha=0.6, color='#ff4444', label=f'Actual DOWN (n={len(pred_down)})', density=True)
        
        ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.5, label='Decision Threshold')
        ax.set_xlabel('Predicted Probability (P(UP))', color='white')
        ax.set_ylabel('Density', color='white')
        ax.set_title('Prediction Distribution by Actual Direction', color='white', fontweight='bold')
        ax.legend()
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        
        return save_path

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_ticker(ticker, use_regime=True, generate_charts=True):
    """Complete pipeline for one ticker."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {ticker} | Regime-Aware: {use_regime} | Charts: {generate_charts}")
    print(f"{'='*80}\n")
    
    # Fetch data
    print("📥 Fetching data...")
    if HAS_CUSTOM_API:
        try:
            df, info = get_stock_history(ticker, days=1500, return_info=True)
            print(f"✅ Source: {info.get('source', 'unknown')} | Rows: {len(df)}")
        except Exception as e:
            print(f"❌ Real API failed for {ticker}: {e}")
            return None
    else:
        print(f"❌ No real API configured")
        return None
    
    if len(df) < MIN_TRAIN_DAYS + 100:
        print(f"❌ Insufficient data")
        return None
    
    df = df.reset_index()
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'Date'})
    elif 'Date' not in df.columns:
        df.index.name = 'Date'
        df = df.reset_index()
    
    # Feature engineering
    print("🔧 Engineering features...")
    data = engineer_features(df)
    
    # Regime detection
    regime_detector = RegimeDetector(window=REGIME_WINDOW)
    regimes = regime_detector.detect(data['Close'].values, data['Date'].values)
    data['regime'] = regimes
    
    # Select deployable features. Keep this in one helper so training and inference stay aligned.
    feature_cols = _select_feature_columns(data)
    data_clean = data.dropna()
    
    if len(data_clean) < MIN_TRAIN_DAYS + 100:
        print(f"❌ Insufficient data after cleaning")
        return None
    
    print(f"✅ {len(feature_cols)} features, {len(data_clean)} samples")
    print(f"📊 Target: {data_clean['target'].mean()*100:.1f}% UP")
    print(f"🎯 Regimes: Bull={(regimes==2).sum()}, Neutral={(regimes==1).sum()}, Bear={(regimes==0).sum()}")
    
    # Prepare arrays
    X = data_clean[feature_cols].values
    y = data_clean['target'].values
    dates = data_clean['Date'].values
    prices = data_clean['Close'].values
    vols = data_clean['volatility_20d'].fillna(0.2).values
    regimes_clean = data_clean['regime'].values
    
    # Walk-forward splits
    splits = walk_forward_split(data_clean, MIN_TRAIN_DAYS, TEST_WINDOW, STEP_SIZE, PURGE_DAYS)
    print(f"📊 Walk-Forward Folds: {len(splits)}")
    
    if len(splits) < 3:
        print(f"❌ Insufficient folds")
        return None    
    # Storage
    oof_preds = np.zeros(len(data_clean))
    oof_probs = np.zeros(len(data_clean))
    oof_actuals = np.zeros(len(data_clean))
    oof_regimes = np.zeros(len(data_clean))
    
    meta_X_train = []
    meta_y_train = []
    fold_results = []
    
    print("\n🚀 Running Walk-Forward Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        if max(train_idx) >= min(test_idx) - PURGE_DAYS:
            continue
        
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        reg_tr, reg_te = regimes_clean[train_idx], regimes_clean[test_idx]
        
        # Sample weights
        class_weights = {0: 1.0, 1: 1.0}
        if y_tr.mean() > 0.55:
            class_weights[1] = 0.8
            class_weights[0] = 1.2
        elif y_tr.mean() < 0.45:
            class_weights[0] = 0.8
            class_weights[1] = 1.2
        
        sample_weights = np.array([class_weights[int(yi)] for yi in y_tr])
        
        # Scale
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        # Train models
        if use_regime:
            models = train_regime_aware_models(X_tr_s, y_tr, reg_tr)
            fold_probs = predict_regime_aware(models, X_te_s, reg_te)
        else:
            models = {
                'xgb': xgb.XGBClassifier(**XGB_PARAMS),
                'lgbm': lgb.LGBMClassifier(**LGB_PARAMS),
                'rf': RandomForestClassifier(**RF_PARAMS),
                'gb': GradientBoostingClassifier(**GB_PARAMS)
            }
            for name, model in models.items():
                model.fit(X_tr_s, y_tr, sample_weight=sample_weights)
            
            meta_te = np.zeros((len(X_te_s), len(models)))
            for i, (name, model) in enumerate(models.items()):
                meta_te[:, i] = model.predict_proba(X_te_s)[:, 1]
            fold_probs = np.mean(meta_te, axis=1)
        
        # Store
        oof_probs[test_idx] = fold_probs
        oof_preds[test_idx] = (fold_probs > 0.5).astype(int)
        oof_actuals[test_idx] = y_te
        oof_regimes[test_idx] = reg_te
        
        # Fold metrics
        fold_acc = accuracy_score(y_te, (fold_probs > 0.5).astype(int))
        fold_results.append({
            'accuracy': fold_acc,
            'mean_prob': fold_probs.mean(),
            'regime': pd.Series(reg_te).mode()[0] if len(reg_te) > 0 else 1
        })
        
        print(f"   Fold {fold+1}/{len(splits)}: acc={fold_acc:.3f}, prob={fold_probs.mean():.3f}")
    
    # Meta-learner
    print("\n🎯 Training Meta-Learner...")
    meta_features = np.column_stack([
        oof_probs,
        oof_regimes == 0,
        oof_regimes == 2
    ])
    
    valid_idx = oof_probs > 0
    meta_learner = LogisticRegression(C=0.5, max_iter=1000, class_weight='balanced')
    meta_learner.fit(meta_features[valid_idx], oof_actuals[valid_idx])
    
    # Final predictions
    final_probs = meta_learner.predict_proba(meta_features)[:, 1]
    
    # Calibrate
    calibrator = None
    try:
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(final_probs[valid_idx], oof_actuals[valid_idx])
        final_probs[valid_idx] = calibrator.transform(final_probs[valid_idx])
    except:
        pass
    
    # Smooth
    pred_series = pd.Series(final_probs)
    smoothed = pred_series.ewm(span=5, adjust=False).mean().values
    
    # Metrics
    y_true = oof_actuals[valid_idx]
    y_pred = oof_preds[valid_idx]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, final_probs[valid_idx]) if len(np.unique(y_true)) > 1 else 0.5
    
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION METRICS")
    print(f"{'='*80}")
    print(f"Directional Accuracy:   {accuracy*100:.2f}%")
    print(f"Precision:              {precision*100:.2f}%")
    print(f"Recall:                 {recall*100:.2f}%")
    print(f"F1-Score:               {f1*100:.2f}%")
    print(f"AUC-ROC:                {auc:.3f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Regime-specific performance
    print(f"\n📊 Regime-Specific Performance:")
    for regime_val, regime_name in [(0, 'Bear'), (1, 'Neutral'), (2, 'Bull')]:
        mask = oof_regimes[valid_idx] == regime_val
        if mask.sum() > 0:
            reg_acc = accuracy_score(y_true[mask], y_pred[mask])
            print(f"   {regime_name}: {reg_acc*100:.1f}% accuracy ({mask.sum()} samples)")
    
    # Backtest
    print(f"\n📈 Running Advanced Backtest...")
    bt = AdvancedBacktester(initial_capital=100000)
    
    bt_dates = pd.to_datetime(dates[valid_idx])
    bt_prices = prices[valid_idx]
    bt_preds = smoothed[valid_idx]
    bt_vols = vols[valid_idx]
    bt_regimes = oof_regimes[valid_idx]
    
    metrics = bt.run(bt_dates, bt_prices, bt_preds, bt_vols, bt_regimes)
    
    print(f"\n{'='*80}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"Total Return:           {metrics['total_return']:+.2f}%")
    print(f"Annualized Return:      {metrics['annualized_return']:+.2f}%")
    print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
    print(f"Win Rate:               {metrics['win_rate']:.1f}%")
    print(f"Max Drawdown:           {metrics['max_drawdown']:.2f}%")
    print(f"Number of Trades:       {metrics['num_trades']}")
    print(f"Avg Return per Trade:   {metrics['avg_return_per_trade']:+.3f}%")
    print(f"Profit Factor:          {metrics['profit_factor']:.2f}")
    
    # Generate charts - FIXED with actuals parameter
    charts = {}
    if generate_charts:
        print(f"\n📊 Generating Analysis Charts...")
        
        charts = ChartGenerator.generate_all_charts(
            ticker=ticker,
            dates=bt_dates,
            prices=bt_prices,
            predictions=bt_preds,
            actuals=y_true,  # FIXED: Passing actuals explicitly
            trades=bt.trades,
            metrics=metrics,
            regimes=bt_regimes,
            fold_results=fold_results
        )
        
        print(f"📈 Charts saved:")
        for name, path in charts.items():
            if path:
                print(f"   • {name}: {path}")
    
    print(f"\nSaving Ultimate production artifact for {ticker}...")
    final_scaler = RobustScaler()
    X_scaled_all = final_scaler.fit_transform(X)
    final_model_bundle = _fit_final_models(X_scaled_all, y, regimes_clean, use_regime=use_regime)
    buy_threshold, sell_threshold = _best_threshold(oof_actuals[valid_idx], final_probs[valid_idx])
    future_returns = (data_clean['Close'].shift(-PREDICTION_HORIZON) / data_clean['Close'] - 1).dropna()
    avg_abs_forward_return = float(np.nanmean(np.abs(future_returns))) if len(future_returns) else 0.02
    avg_abs_forward_return = max(0.005, min(avg_abs_forward_return, 0.15))

    artifact_metrics = {
        'accuracy': float(accuracy * 100),
        'precision': float(precision * 100),
        'recall': float(recall * 100),
        'f1': float(f1 * 100),
        'auc': float(auc),
    }
    for key, value in metrics.items():
        artifact_metrics[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value

    save_ultimate_model(ticker, {
        'version': 'v36-production',
        'ticker': ticker.upper(),
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'feature_cols': feature_cols,
        'scaler': final_scaler,
        'model_bundle': final_model_bundle,
        'meta_learner': meta_learner,
        'calibrator': calibrator,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'avg_abs_forward_return': avg_abs_forward_return,
        'metrics': artifact_metrics
    })
    print(f"Ultimate artifact saved for {ticker}: {len(feature_cols)} features")

    return {
        'ticker': ticker,
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'auc': auc,
        **metrics,
        'charts': charts,
        'ultimate_artifact_saved': True,
        'model_version': 'v36-production'
    }


def train_ultimate_model(ticker, use_regime=True, generate_charts=False):
    """Train and save one production Ultimate Engine model for app.py and scheduler use."""
    return process_ticker(ticker, use_regime=use_regime, generate_charts=generate_charts)


def predict_ultimate_realtime(ticker, hist, current_price=None, days=7):
    """Realtime inference contract used by Flask. Returns direction plus UI-compatible prices."""
    ticker = ticker.upper()
    artifact = load_ultimate_model(ticker)
    if artifact is None:
        return None

    df = _normalize_ohlcv_for_engine(hist)
    if len(df) < MIN_TRAIN_DAYS:
        return None

    data = engineer_features(df)
    regime_detector = RegimeDetector(window=REGIME_WINDOW)
    data['regime'] = regime_detector.detect(data['Close'].values, data['Date'].values)
    data_clean = data.dropna()
    feature_cols = artifact['feature_cols']
    if data_clean.empty or not all(col in data_clean.columns for col in feature_cols):
        return None

    latest = data_clean.iloc[-1:]
    X_latest = latest[feature_cols].values
    regimes_latest = latest['regime'].values
    X_scaled = artifact['scaler'].transform(X_latest)
    base_prob = float(_predict_final_models(artifact['model_bundle'], X_scaled, regimes_latest)[0])

    direction_prob = base_prob
    meta_learner = artifact.get('meta_learner')
    if meta_learner is not None:
        meta_features = np.array([[base_prob, regimes_latest[0] == 0, regimes_latest[0] == 2]])
        direction_prob = float(meta_learner.predict_proba(meta_features)[:, 1][0])

    calibrator = artifact.get('calibrator')
    if calibrator is not None:
        direction_prob = float(calibrator.transform([direction_prob])[0])

    direction_prob = max(0.05, min(0.95, direction_prob))
    confidence = abs(direction_prob - 0.5) * 2.0
    avg_abs_forward_return = float(artifact.get('avg_abs_forward_return', 0.02))
    expected_return = (direction_prob - 0.5) * 2.0 * avg_abs_forward_return

    buy_threshold = float(artifact.get('buy_threshold', ENTRY_THRESHOLD_BASE))
    sell_threshold = float(artifact.get('sell_threshold', EXIT_THRESHOLD_BASE))
    if direction_prob >= max(buy_threshold, 0.55):
        signal = 'BUY' if confidence < 0.55 else 'STRONG BUY'
    elif direction_prob <= min(sell_threshold, 0.45):
        signal = 'SELL' if confidence < 0.55 else 'STRONG SELL'
    else:
        signal = 'HOLD'

    base_price = float(current_price if current_price is not None else data_clean['Close'].iloc[-1])
    predicted_prices = []
    total_days = max(1, int(days))
    daily_return = (1.0 + expected_return) ** (1.0 / max(PREDICTION_HORIZON, total_days)) - 1.0
    price = base_price
    for _ in range(total_days):
        price *= (1.0 + daily_return)
        predicted_prices.append(float(price))

    uncertainty = max(0.005, avg_abs_forward_return * (1.0 - min(confidence, 0.95)))
    return {
        'ticker': ticker,
        'prediction': float(expected_return),
        'lower_95': float(expected_return - 1.96 * uncertainty),
        'upper_95': float(expected_return + 1.96 * uncertainty),
        'confidence': float(confidence),
        'direction_prob': float(direction_prob),
        'signal': signal,
        'model_version': artifact.get('version', 'v36-production'),
        'features_used': feature_cols,
        'data_freshness': datetime.utcnow().isoformat() + 'Z',
        'drift_score': 0.0,
        'prediction_horizon': PREDICTION_HORIZON,
        'predicted_prices': predicted_prices,
        'metrics': artifact.get('metrics', {}),
        'trained_at': artifact.get('trained_at')
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("AI STOCK PREDICTOR — ULTIMATE ENGINE v3.6 FIXED")
    print("Fixed: Chart generation with proper actuals parameter")
    print("="*80)
    
    all_results = []
    
    for ticker in TICKERS:
        try:
            result = process_ticker(ticker, use_regime=True, generate_charts=True)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY — ALL TICKERS")
    print("="*80)
    print(f"{'Ticker':<8} {'Acc%':>6} {'Prec%':>6} {'Rec%':>6} {'F1%':>6} {'AUC':>5} {'AnnRet%':>8} {'Sharpe':>6} {'Win%':>6} {'Trades':>6} {'MaxDD%':>7}")
    print("-"*80)
    
    for r in all_results:
        print(f"{r['ticker']:<8} {r['accuracy']:>6.1f} {r['precision']:>6.1f} {r['recall']:>6.1f} "
              f"{r['f1']:>6.1f} {r['auc']:>5.3f} {r['annualized_return']:>+8.1f} {r['sharpe_ratio']:>6.2f} "
              f"{r['win_rate']:>6.1f} {r['num_trades']:>6} {r['max_drawdown']:>7.1f}")
    
    print("="*80)
    print("✅ Complete analysis with charts generated in ./charts/ folder")

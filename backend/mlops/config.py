"""
MLOps Configuration - Centralized configuration management
"""

import os
import threading
import re
from pathlib import Path


class MLOpsConfig:
    """
    Configuration settings for MLOps system
    """
    
    # Directory paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MLOPS_DIR = os.path.join(BASE_DIR, 'mlops')
    REGISTRY_DIR = os.path.join(MLOPS_DIR, 'model_registry')
    LOGS_DIR = os.path.join(MLOPS_DIR, 'logs')
    CHECKPOINTS_DIR = os.path.join(MLOPS_DIR, 'checkpoints')
    ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
    
    # Training configuration
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    
    # Scheduler configuration
    TRAINING_INTERVAL_HOURS = 1
    STOCKS_FILE_LOCK = threading.Lock()
    _TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,19}$")

    @classmethod
    def normalize_ticker(cls, ticker: str) -> str:
        return (ticker or '').strip().upper()

    @classmethod
    def is_valid_ticker(cls, ticker: str) -> bool:
        normalized = cls.normalize_ticker(ticker)
        return bool(normalized and cls._TICKER_PATTERN.match(normalized))
    
    @classmethod
    def get_stocks(cls) -> list:
        """Dynamic loading of stocks from persistent file"""
        import json
        path = os.path.join(cls.MLOPS_DIR, 'stocks.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                raw = json.load(f)
            # Keep insertion order while normalizing and deduplicating.
            seen = set()
            out = []
            for s in raw:
                symbol = cls.normalize_ticker(str(s))
                if not symbol or symbol in seen:
                    continue
                if not cls.is_valid_ticker(symbol):
                    continue
                seen.add(symbol)
                out.append(symbol)
            return out
        return [] # Return empty if no registry found

    @classmethod
    def add_stock(cls, ticker: str):
        """Append a new stock to the persistent hourly training list"""
        import json
        normalized = cls.normalize_ticker(ticker)
        if not cls.is_valid_ticker(normalized):
            return

        with cls.STOCKS_FILE_LOCK:
            stocks = cls.get_stocks()
            if normalized not in stocks:
                stocks.append(normalized)
                path = os.path.join(cls.MLOPS_DIR, 'stocks.json')
                tmp_path = f"{path}.tmp"
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(stocks, f, indent=4)
                os.replace(tmp_path, path)
                print(f"📌 Added {normalized} to automated hourly training list.")

    @classmethod
    def add_stocks(cls, tickers: list[str]) -> int:
        """Bulk-add symbols safely; returns number of newly added symbols."""
        import json

        normalized = []
        for ticker in tickers or []:
            symbol = cls.normalize_ticker(str(ticker))
            if cls.is_valid_ticker(symbol):
                normalized.append(symbol)

        if not normalized:
            return 0

        with cls.STOCKS_FILE_LOCK:
            stocks = cls.get_stocks()
            seen = set(stocks)
            added = 0
            for symbol in normalized:
                if symbol not in seen:
                    stocks.append(symbol)
                    seen.add(symbol)
                    added += 1

            if added:
                path = os.path.join(cls.MLOPS_DIR, 'stocks.json')
                tmp_path = f"{path}.tmp"
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(stocks, f, indent=4)
                os.replace(tmp_path, path)

            return added
    
    # Model configuration
    LOOKBACK_PERIOD = 60  # Days
    LSTM_UNITS = [50, 50]
    DROPOUT_RATE = 0.2

    @classmethod
    def _resolve_mlflow_tracking_uri(cls) -> str:
        """
        Resolve MLflow tracking URI with priority:
        1) Explicit MLFLOW_TRACKING_URI
        2) DagsHub URI from repo owner/name
        3) Local file store fallback
        """
        explicit = (os.getenv('MLFLOW_TRACKING_URI') or '').strip()
        if explicit:
            return explicit

        dagshub_uri = (os.getenv('DAGSHUB_MLFLOW_URI') or '').strip()
        if dagshub_uri:
            return dagshub_uri

        owner = (os.getenv('DAGSHUB_REPO_OWNER') or '').strip()
        repo = (os.getenv('DAGSHUB_REPO_NAME') or '').strip()
        if owner and repo:
            return f"https://dagshub.com/{owner}/{repo}.mlflow"

        return Path(os.path.join(cls.BASE_DIR, 'mlruns')).resolve().as_uri()

    @classmethod
    def configure_mlflow_env(cls) -> None:
        """Configure MLflow auth env vars for DagsHub when token is provided."""
        token = (os.getenv('DAGSHUB_TOKEN') or '').strip()
        if not token:
            return

        owner = (os.getenv('DAGSHUB_REPO_OWNER') or '').strip()
        username = (os.getenv('DAGSHUB_USERNAME') or owner or '').strip()

        # Respect explicitly configured MLflow credentials if already set.
        if username and not os.getenv('MLFLOW_TRACKING_USERNAME'):
            os.environ['MLFLOW_TRACKING_USERNAME'] = username
        if not os.getenv('MLFLOW_TRACKING_PASSWORD'):
            os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    
    # MLflow configuration
    # DagsHub-ready with local fallback (resolved after class definition).
    MLFLOW_TRACKING_URI = ""
    MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Prediction_Lineage')
    
    # Logging configuration
    LOG_FORMAT = '[%(asctime)s] %(levelname)s: %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    @classmethod
    def ensure_directories(cls):
        """Create all required directories"""
        for dir_path in [
            cls.MLOPS_DIR,
            cls.REGISTRY_DIR,
            cls.LOGS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.ARTIFACTS_DIR
        ]:
            os.makedirs(dir_path, exist_ok=True)


# Initialize directories on import
MLOpsConfig.ensure_directories()
MLOpsConfig.configure_mlflow_env()
MLOpsConfig.MLFLOW_TRACKING_URI = MLOpsConfig._resolve_mlflow_tracking_uri()

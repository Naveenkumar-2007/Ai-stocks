"""
MLOps Configuration - Centralized configuration management
"""

import os


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
    
    @classmethod
    def get_stocks(cls) -> list:
        """Dynamic loading of stocks from persistent file"""
        import json
        path = os.path.join(cls.MLOPS_DIR, 'stocks.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return [] # Return empty if no registry found

    @classmethod
    def add_stock(cls, ticker: str):
        """Append a new stock to the persistent hourly training list"""
        import json
        stocks = cls.get_stocks()
        if ticker not in stocks:
            stocks.append(ticker)
            path = os.path.join(cls.MLOPS_DIR, 'stocks.json')
            with open(path, 'w') as f:
                json.dump(stocks, f, indent=4)
            print(f"📌 Added {ticker} to automated hourly training list.")
    
    # Model configuration
    LOOKBACK_PERIOD = 60  # Days
    LSTM_UNITS = [50, 50]
    DROPOUT_RATE = 0.2
    
    # MLflow configuration
    # Simple "mlruns" folder in the backend directory
    MLFLOW_TRACKING_URI = "file:mlruns"
    MLFLOW_EXPERIMENT_NAME = "Prediction_Lineage"
    
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

# Production MLOps v2 Implementation

This document describes the implemented production-grade architecture in this repository.

## Implemented Components

## 1) Data Ingestion and Validation

- Module: backend/mlops_v2/data_ingestion.py
- Behavior:
  - Primary fetch with retries and exponential backoff
  - Uses existing provider chain from backend/stock_api.py (Twelve Data + fallback providers)
  - Persists raw data as Parquet under data/raw/{ticker}/{date}.parquet
  - Writes per-ticker manifest for DVC-friendly tracking

- Validation: backend/mlops_v2/validation.py
  - Great Expectations checks:
    - Close not null and in [0.01, 1,000,000]
    - Volume not null and > 0
    - Row count between 50 and 500
    - No duplicate timestamps
    - Date range coverage >= 100 days
  - Training aborts on validation failure

## 2) Feature Engineering (10 Features Exactly)

- Module: backend/mlops_v2/feature_engineering.py
- Features:
  1. rsi_14
  2. macd_diff
  3. bb_width_price
  4. volume_zscore_20
  5. return_5d
  6. atr_14
  7. stoch_k
  8. overnight_gap_pct
  9. return_1d
  10. volume_sma_ratio
- Rules:
  - No lookahead usage in indicators
  - Winsorization at 1st/99th percentile
  - RobustScaler normalization

## 3) Drift Detection

- Module: backend/mlops_v2/drift.py
- Uses Evidently report with DataDriftPreset
- Produces daily HTML report under monitoring/reports/{date}/
- Drift score is ratio of drifted features
- Safe default: if drift check fails, train anyway

## 4) Model Training (Ensemble)

- Module: backend/mlops_v2/training.py
- XGBoost Classifier:
  - n_estimators=200
  - max_depth=5
  - learning_rate=0.05
  - TimeSeriesSplit (3 splits)
- LSTM Regressor:
  - LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16) -> Dense(1)
  - Huber loss
  - EarlyStopping(patience=5), max epochs=50
  - Windows runtime safety: uses a sequence-regressor fallback artifact when native TensorFlow recurrent training is unstable
- Ensemble:
  - prediction = 0.7 * direction + 0.3 * magnitude
  - uncertainty from model disagreement
  - 95% interval = prediction +/- 1.96 * uncertainty

- Retraining Policy:
  - Drift-gated retrain: train only when drift > 0.25 or 7 days since last train
  - Monthly full retrain supported
  - Weekly tuning supported with Optuna (50 trials, Sharpe objective)

## 5) Registry and MLflow

- Module: backend/mlops_v2/registry.py
- Logs both models in one run
- Logs metrics:
  - xgb_accuracy
  - lstm_val_loss
  - drift_score
  - data_points
- Registers models as:
  - {ticker}_xgb
  - {ticker}_lstm
- Auto-promotion logic to Production using champion/challenger threshold
- Canary helper policy module:
  - backend/mlops_v2/champion_challenger.py
  - 10% stable-hash traffic routing helper
  - 3-day minimum canary window helper before auto-promotion

## 6) Monitoring and Metrics

- Module: backend/mlops_v2/monitoring.py
- Exposed Prometheus metrics:
  - predictions_total{ticker="..."}
  - prediction_latency_seconds
  - model_accuracy_20d{ticker="..."}
  - drift_score{ticker="..."}

- Endpoint: backend/app.py -> GET /metrics

## 7) Inference API Contract

- Module: backend/mlops_v2/inference.py
- Existing endpoint backend/app.py GET /api/stock/{ticker} now also includes:
  - prediction
  - lower_95
  - upper_95
  - confidence
  - direction_prob
  - model_version
  - features_used
  - data_freshness
  - drift_score

This is additive to existing response fields, so current frontend compatibility is preserved.

## 8) Orchestration

- Airflow DAG: backend/airflow/dags/stock_daily_training.py
  - Runs weekdays at 18:00 UTC
  - Uses retries with exponential backoff
  - Max active runs = 1
  - Catchup = false
  - Parallelism controlled by LocalExecutor/global airflow config

- DVC stages added in backend/dvc.yaml:
  - train_v2_drift_gated
  - train_v2_monthly_full
  - train_v2_weekly_tuning

## 9) Feast Feature Store

- Feature repo created at backend/feature_repo/
  - feature_store.yaml
  - stock_features.py
- Offline source set to Parquet, online store set to SQLite

## 10) Optional Dependencies for v2

- File: backend/requirements.txt
- Includes:
  - apache-airflow
  - great-expectations
  - evidently
  - xgboost
  - optuna
  - feast
  - prometheus-client
  - redis

## Environment Variable Names Used

Use variable names only (do not commit secret values):

- TWELVE_DATA_API_KEY or TWELVE_DATA_API_KEYS
- FINNHUB_API_KEY or FINNHUB_API_KEYS
- ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEYS
- GROQ_API_KEY
- FIREBASE_SERVICE_ACCOUNT_PATH
- ADMIN_EMAILS
- LOG_LEVEL
- MLFLOW_TRACKING_URI (optional override)

Frontend variables:
- REACT_APP_API_URL
- REACT_APP_FIREBASE_API_KEY
- REACT_APP_FIREBASE_AUTH_DOMAIN
- REACT_APP_FIREBASE_PROJECT_ID
- REACT_APP_FIREBASE_STORAGE_BUCKET
- REACT_APP_FIREBASE_MESSAGING_SENDER_ID
- REACT_APP_FIREBASE_APP_ID
- REACT_APP_FIREBASE_MEASUREMENT_ID
- REACT_APP_ADMIN_EMAILS

## Run Commands

From backend directory:

- DVC daily drift-gated training:
  - dvc repro train_v2_drift_gated

- DVC monthly full retraining:
  - dvc repro train_v2_monthly_full

- Start Flask API:
  - python app.py

- Start MLflow UI (correct store):
  - run_mlflow_ui.bat

- Start Airflow DAG scheduler/webserver (after Airflow setup):
  - airflow scheduler
  - airflow webserver --port 8080

- Prometheus scrape endpoint:
  - http://localhost:8000/metrics

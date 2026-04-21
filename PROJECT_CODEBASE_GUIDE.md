# AI Insights Project Guide (Codebase-Accurate)

## 1. Project Overview

This project is a full-stack stock intelligence platform with 4 major runtime parts:

1. Flask backend API for stock data, prediction, cache, and admin endpoints.
2. FastAPI chatbot service for conversational market analysis and feedback learning.
3. React frontend for user authentication, prediction dashboard, and chatbot UI.
4. MLOps layer using DVC + MLflow + local model registry for multi-stock training and lineage.

## 2. Tech Stack Actually Used

### Backend
- Python 3.x
- Flask
- Flask-CORS
- TensorFlow (LSTM model training/inference)
- scikit-learn, pandas, numpy, ta
- MLflow
- DVC
- schedule (automated training scheduler)

### Chatbot (inside backend)
- FastAPI + Uvicorn
- langchain + langchain-community + langchain-groq
- sentence-transformers
- faiss-cpu (non-Windows install path)
- Finnhub service integration

### Frontend
- React 18
- react-router-dom
- recharts
- framer-motion
- Firebase JS SDK

### Deployment
- Docker multi-stage build
- Gunicorn serves Flask
- entrypoint script injects frontend runtime env vars

## 3. Repository Runtime Layout

- backend/: Flask API, scheduler, stock providers, MLOps training pipeline
- backend/chatbot/app/: FastAPI chatbot app and chat storage
- backend/mlops/: config, registry, training pipeline, stock universe, train-all script
- frontend/: React app
- Dockerfile + entrypoint.sh: containerized startup and env injection

## 4. External APIs and Keys Used

Important: only variable names are documented below (no secret values).

### Market Data APIs
1. TWELVE_DATA_API_KEY or TWELVE_DATA_API_KEYS
- Used by backend stock provider for primary market/time-series data.
- File usage: backend/stock_api.py

2. FINNHUB_API_KEY or FINNHUB_API_KEYS
- Used by backend stock provider fallback logic.
- Used heavily by chatbot market data service.
- File usage: backend/stock_api.py, backend/chatbot/app/services/finnhub_service.py, backend/chatbot/app/main.py

3. ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEYS
- Optional fallback market data provider in backend stock provider.
- File usage: backend/stock_api.py

### LLM API
4. GROQ_API_KEY
- Required for chatbot LLM responses.
- File usage: backend/chatbot/app/config.py, backend/chatbot/app/main.py, backend/chatbot/app/core/rag_pipeline.py

### Firebase/Auth
5. FIREBASE_SERVICE_ACCOUNT_PATH
- Used by Flask backend to initialize Firebase Admin SDK for protected admin endpoints.
- File usage: backend/app.py

6. Frontend Firebase variables
- REACT_APP_FIREBASE_API_KEY
- REACT_APP_FIREBASE_AUTH_DOMAIN
- REACT_APP_FIREBASE_PROJECT_ID
- REACT_APP_FIREBASE_STORAGE_BUCKET
- REACT_APP_FIREBASE_MESSAGING_SENDER_ID
- REACT_APP_FIREBASE_APP_ID
- REACT_APP_FIREBASE_MEASUREMENT_ID
- File usage: frontend/src/firebase/config.js

### App Config and Access Control
7. REACT_APP_API_URL
- Frontend base URL for backend API access.
- File usage: frontend/src/pages/Prediction.jsx

8. REACT_APP_ADMIN_EMAILS
- Frontend admin allowlist behavior.
- File usage: frontend/src/contexts/AuthContext.jsx

9. ADMIN_EMAILS
- Backend admin allowlist for protected endpoints.
- File usage: backend/app.py

10. LOG_LEVEL
- Backend logging level.
- File usage: backend/app.py

11. MLFLOW_TRACKING_URI
- Optional override for reporting in mlops status endpoint.
- Core training uses mlops config class URI.
- File usage: backend/app.py, backend/mlops/config.py

## 5. How the System Works (End-to-End)

## 5.1 User Request Flow (Frontend -> Backend)

1. User opens React app and authenticates via Firebase.
2. Prediction page calls backend endpoints for search, stock data, news, and sentiment.
3. Backend fetches market data using provider chain (Twelve Data -> fallback providers).
4. Backend computes indicators and attempts LSTM prediction for the resolved ticker.
5. If model/scaler unavailable or invalid, backend falls back to technical-analysis prediction.
6. Response returns price data, predictions, indicators, sentiment, and metadata.

## 5.2 New Ticker Auto-Add and Auto-Train

When a new ticker is requested:

1. Backend normalizes and adds ticker to mlops/stocks.json via MLOpsConfig.add_stock().
2. If no model exists for that ticker, backend triggers background training thread.
3. Duplicate concurrent training for same ticker is guarded.
4. Trained model is registered in local model registry and tracked in MLflow (if available).

## 5.3 MLOps Training Flow (DVC + MLflow)

Stage: train_all (backend/dvc.yaml)

1. DVC runs: python -m mlops.train_all_stocks with params from backend/params.yaml.
2. train_all_stocks loads tickers from mlops/stocks.json (supports max_stocks and start_from).
3. For each ticker, pipeline performs:
- data ingestion and preprocessing
- transformation to sequences
- LSTM training with callbacks/checkpoints
- evaluation metrics
- local model registry save
- MLflow run logging and optional model registration
4. Aggregated metrics are written to:
- backend/mlops/metrics/summary.json
- backend/mlops/metrics/all_stocks_metrics.json

## 5.4 Scheduler Flow

1. Scheduler reads tickers from stocks.json dynamically.
2. Runs daily at configured UTC schedule.
3. Trains in batches and skips symbols already trained that day.
4. Stores recent training summaries in memory and exposes training-status endpoint.

## 6. Main API Endpoints

### Backend Flask
- GET /health
- GET /api/mlops/status
- GET /api/health/training
- GET /api/models/training-status
- POST /api/models/train
- GET /api/search
- GET /api/stock/<ticker>
- GET /api/news/<ticker>
- GET /api/sentiment/<ticker>
- GET /api/admin/system-health (admin-protected)
- POST /api/cache/clear (admin-protected)
- GET /api/cache/stats (admin-protected)

### Chatbot FastAPI (proxied through backend /chatbot/* in deployment)
- GET /
- POST /chat
- GET /chats
- GET /chats/{chat_id}
- DELETE /chats/{chat_id}
- POST /feedback
- GET /feedback/stats
- GET /trained-stocks

## 7. Ports and Runtime

### Local development
- Flask backend: 8000 (default in app.py)
- React dev server: 3000
- Chatbot FastAPI: 8001
- MLflow UI: 5000 (recommended)

### Docker runtime
- Gunicorn/Flask: 7860
- Chatbot: 127.0.0.1:8001 (internal)
- Flask proxies chatbot routes

## 8. Correct Run Commands

From project root (Ai-insights-main):

### Backend only
- cd backend
- python app.py

### Frontend only
- cd frontend
- npm start

### Chatbot only (local standalone)
- cd backend/chatbot/app
- python -m uvicorn main:app --host 127.0.0.1 --port 8001

### DVC training
- cd backend
- dvc repro train_all

### MLflow UI (correct store)
- cd backend
- run_mlflow_ui.bat

## 9. Required Environment Setup

### Backend .env should include
- TWELVE_DATA_API_KEY or TWELVE_DATA_API_KEYS
- FINNHUB_API_KEY or FINNHUB_API_KEYS
- GROQ_API_KEY (required for chatbot)
- Optional: ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEYS
- Optional: FIREBASE_SERVICE_ACCOUNT_PATH
- Optional: ADMIN_EMAILS, LOG_LEVEL

### Frontend .env should include
- REACT_APP_API_URL
- REACT_APP_FIREBASE_API_KEY
- REACT_APP_FIREBASE_AUTH_DOMAIN
- REACT_APP_FIREBASE_PROJECT_ID
- REACT_APP_FIREBASE_STORAGE_BUCKET
- REACT_APP_FIREBASE_MESSAGING_SENDER_ID
- REACT_APP_FIREBASE_APP_ID
- REACT_APP_FIREBASE_MEASUREMENT_ID
- REACT_APP_ADMIN_EMAILS

## 10. Security and Operational Notes

1. Never commit real API keys in repository files.
2. Use only environment variables or secret managers for credentials.
3. Rotate any keys that were previously exposed.
4. Keep MLflow UI pointed to backend/mlruns to avoid empty-run confusion.
5. Keep DVC outputs and metrics paths non-overlapping (already configured).

## 11. Current MLOps Defaults

From backend/params.yaml:
- epochs: 20
- batch_size: 32
- days: 730
- max_stocks: 61
- start_from: null

This document reflects the current code behavior in your workspace.

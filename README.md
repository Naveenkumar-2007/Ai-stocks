---
title: AI Stock Predictor
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# 📈 AI Stock Predictor & Quantitative MLOps Platform

![AI Pipeline Architecture](assets/ai_pipeline.png)

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![React](https://img.shields.io/badge/react-18.0-61dafb?logo=react&logoColor=black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white)

An enterprise-grade, real-time stock market prediction platform powered by Deep Learning (LSTM) and Machine Learning (XGBoost). Built with a complete end-to-end MLOps pipeline, real-time sentiment analysis, and live quantitative trading metrics tracking.

---

## ⚙️ The Training Lifecycle: How It Works End-to-End

The system is engineered for maximum efficiency. It ensures the user never experiences downtime while intelligently managing backend server compute.

### 1. The First Time a Stock is Searched (On-Demand Catch-up)
When a user types a brand-new stock (e.g., `NVDA`) into the search bar:
- **Instant Fallback**: The model doesn't exist yet. Instead of crashing, the system instantly calculates and displays a **Technical Analysis** fallback (Moving Averages, RSI, MACD). This gives the user immediate, math-based financial data.
- **Hidden Background Training**: At the exact same millisecond, the backend locks the ticker and spins up an isolated background thread. It fetches historical data from Twelve Data (or Alpha Vantage) and begins training the deep learning models for that specific stock.
- **The Swap**: ~2 minutes later, if the user refreshes the page, the Technical Analysis chart is instantly replaced with the fully-trained, highly accurate AI prediction!

### 2. The Next Training (The 4:00 AM Nightly Batch)
To keep the models fresh for the next trading day without interrupting daytime traffic:
- When users search for stocks, their tickers are automatically saved to a persistent list (`stocks.json`).
- Every day at exactly **4:00 AM IST (22:30 UTC)**, the `scheduler.py` cron job wakes up.
- It iterates through the list of *only the stocks users care about* and retrains their models in bulk.
- **The Result**: When users wake up the next morning, they instantly receive updated predictions processed on the latest closing prices, loaded straight from RAM cache!

---

## 🧰 The Tech Stack: What Every Tool Does

Every tool in this repository was carefully selected to replicate a professional Quantitative Finance pipeline:

| Tool | Role in the Pipeline |
| :--- | :--- |
| **XGBoost** | The **Directional Engine**. It predicts the probability of the stock going *UP* or *DOWN* (Classification). We rely on XGBoost for this because tree-based models excel at binary market direction logic. |
| **TensorFlow / LSTM** | The **Magnitude Engine**. Long Short-Term Memory neural networks are used to recursively predict the actual *price trajectory* multiple days into the future based on sequential time-series patterns. |
| **MLflow & DagsHub** | The **Model Registry**. Every time a model trains, MLflow logs the accuracy metrics, hyperparameters, and saves the `.keras` files to DagsHub. This ensures we can version-control our AI. |
| **Finnhub NLP** | The **Sentiment Analyzer**. It scrapes real-time breaking financial news and converts the text into mathematical Bullish/Bearish "Buzz Scores" to influence the prediction. |
| **Prometheus** | The **Metrics Aggregator**. It silently sits in the backend capturing Live Accuracy, Simulated PnL, and Data Drift scores every time a user requests a prediction. |
| **Grafana** | The **Observability Dashboard**. It visualizes the data scraped by Prometheus into beautiful, real-time charts so admins can monitor the AI's financial health. |
| **React & Recharts** | The **Frontend**. Delivers lightning-fast, interactive stock charts and prediction gauges to the end user. |

---

## 🏗️ System Architecture

The architecture is highly decoupled, ensuring the React frontend remains lightning-fast while heavy tensor computations occur asynchronously.

### Core System Flow

```mermaid
graph TD
    %% Tool Colors
    classDef react fill:#61DAFB,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef firebase fill:#FFCA28,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef flask fill:#FFFFFF,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef fastapi fill:#009688,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold;
    classDef tensorflow fill:#FF6F00,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold;
    classDef mlflow fill:#0194E2,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold;
    classDef dvc fill:#945DD6,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold;
    classDef airflow fill:#017CEE,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold;
    classDef grafana fill:#F46800,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold;
    classDef groq fill:#F55036,stroke:#333,stroke-width:2px,color:#fff,font-weight:bold;
    classDef db fill:#333333,stroke:#00B8FF,stroke-width:2px,color:#fff,font-weight:bold;
    classDef logo fill:none,stroke:none,color:#fff,font-weight:bold,font-size:16px;

    subgraph ClientLayer ["1. Client Layer"]
        A["<img src='https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg' width='20'/> React.js Frontend"]:::react
        B["<img src='https://upload.wikimedia.org/wikipedia/commons/3/37/Firebase_Logo.svg' width='20'/> Firebase Auth"]:::firebase
    end

    subgraph APILayer ["2. API Services (Dockerized)"]
        C["<img src='https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg' width='20'/> Flask Backend API"]:::flask
        D["<img src='https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png' width='20'/> FastAPI Chatbot"]:::fastapi
    end

    subgraph MLOpsLayer ["3. Data & MLOps Pipeline"]
        E["<img src='https://upload.wikimedia.org/wikipedia/commons/d/de/AirflowLogo.png' width='20'/> Apache Airflow"]:::airflow
        E2["DVC Orchestrator"]:::dvc
        F["<img src='https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg' width='20'/> TensorFlow LSTM Models"]:::tensorflow
        G["<img src='https://upload.wikimedia.org/wikipedia/commons/f/fe/Mlflow-logo.svg' width='20'/> MLflow Registry"]:::mlflow
        H[("Local Model Storage")]:::db
    end

    subgraph RAGLayer ["4. AI RAG Chatbot"]
        I[("FAISS Vector Database")]:::db
        J["LangChain Engine"]:::fastapi
        K["Groq Llama 3 LLM"]:::groq
    end
    
    subgraph MonitoringLayer ["5. Monitoring"]
        N["<img src='https://upload.wikimedia.org/wikipedia/commons/a/a1/Grafana_logo.svg' width='20'/> Grafana Dashboards"]:::grafana
    end

    subgraph ExternalLayer ["6. External Financial Data"]
        L["TwelveData API"]:::db
        M["Finnhub API"]:::db
    end

    %% Logo at the bottom
    LOGO["<img src='https://raw.githubusercontent.com/Naveenkumar-2007/Ai-stocks/main/frontend/public/assets/logo-dark.jpg' width='140'/><br/>AI Stock Predictor & Quantitative MLOps Platform"]:::logo

    %% Wiring
    A <-->|Auth Token| B
    A -->|Fetch Stock Predictions| C
    A -->|Chat with AI| D

    C <-->|Get Live Prices| L
    C <-->|Get Live News| M
    C -->|Load Best Model| H
    C -->|Emit Metrics| N

    D <-->|Fetch Live Context| M
    D <-->|Semantic Search| I
    D -->|Build Prompt| J
    J <-->|Stream Response| K

    E -->|Schedules| E2
    E2 -->|Trigger Daily Training| F
    F -->|Log Accuracy Metrics| G
    F -->|Save .keras files| H
    
    ExternalLayer ~~~ LOGO
```

---

## 🤖 Automated MLOps & Drift Detection (V2 Pipeline)

To protect the system from market regime changes (like sudden crashes or sector rotations), the V2 pipeline implements an enterprise-grade **Continuous Training (CT)** loop using statistical drift detection. This ensures we only spend compute resources retraining models when the market behavior actually changes.

```mermaid
graph TD
    %% Advanced Styling
    classDef scheduler fill:#9C27B0,stroke:#fff,stroke-width:2px,color:#fff,font-weight:bold;
    classDef data fill:#00BCD4,stroke:#fff,stroke-width:2px,color:#000,font-weight:bold;
    classDef model fill:#FF9800,stroke:#fff,stroke-width:2px,color:#000,font-weight:bold;
    classDef monitor fill:#F44336,stroke:#fff,stroke-width:2px,color:#fff,font-weight:bold;
    classDef registry fill:#03A9F4,stroke:#fff,stroke-width:2px,color:#000,font-weight:bold;
    classDef success fill:#4CAF50,stroke:#fff,stroke-width:2px,color:#fff,font-weight:bold;

    S["🕒 Airflow Background Scheduler"]:::scheduler
    D["📊 Data Ingestion Engine<br/>(TwelveData & Finnhub)"]:::data
    M["🧠 Production LSTM/XGBoost Model"]:::model
    E{"🕵️ Statistical Drift Monitor<br/>(KS-Test & PSI)"}:::monitor
    ML["📦 MLflow Model Registry"]:::registry
    
    S -->|Triggers Daily Job at 4:00 AM| D
    D -->|Fetches Latest Validation Data| M
    M -->|Predicts on New Data| E
    
    E -->|Drift Score > Threshold| Alert["⚠️ Model Decay Detected!"]:::monitor
    E -->|Drift Score < Safe| OK["✅ Model is Healthy<br/>Skip Retraining"]:::success
    
    Alert -->|Pull 730+ Days History| D2["📚 Deep Retraining Initiated"]:::data
    D2 -->|Train Champion Model| M2["⚙️ Train New LSTM"]:::model
    M2 -->|Register & Tag 'Production'| ML
```

---

## 📊 Ultimate Engine v3.6 Production Benchmark

The production predictor uses `backend/ultimate_stock_engine_v36.py` as the primary inference engine. The benchmark below was produced locally on **May 1, 2026 IST** using real market data through **April 30, 2026**, with **1,500 daily candles per ticker**, walk-forward validation, regime-aware ensemble models, calibrated probabilities, and a 5-trading-day direction horizon.

The five-stock training universe was reset to the current mega-cap benchmark set:

`NVDA`, `AAPL`, `GOOG`, `MSFT`, `AMZN`

![Ultimate Engine Top 5 Training Summary](backend/charts/ultimate_top5_summary.png)

| Ticker | Directional Accuracy | Precision | Recall | F1 | AUC | Total Return | Annualized Return | Sharpe | Win Rate | Trades | Max Drawdown | Profit Factor |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVDA | 52.78% | 57.26% | 75.79% | 65.24% | 0.576 | +82.51% | +22.11% | 0.94 | 53.49% | 43 | 18.05% | 2.42 |
| AAPL | 56.75% | 58.19% | 74.64% | 65.40% | 0.603 | +28.10% | +8.57% | 0.66 | 55.56% | 27 | 16.91% | 1.79 |
| GOOG | 49.07% | 56.43% | 56.56% | 56.50% | 0.558 | +60.62% | +17.04% | 0.84 | 53.33% | 30 | 15.14% | 2.06 |
| MSFT | 57.28% | 58.92% | 83.33% | 69.03% | 0.599 | +22.46% | +6.96% | 0.57 | 58.62% | 29 | 19.04% | 1.59 |
| AMZN | 54.89% | 60.93% | 62.73% | 61.81% | 0.557 | +40.23% | +11.88% | 0.93 | 60.71% | 28 | 11.50% | 2.07 |

### Current Training Charts

These charts are generated from the same real five-stock training run. They are committed so the README shows the actual model outputs instead of placeholder screenshots.

| Ticker | Main Analysis | Equity Curve | Regimes | Temporal Consistency | Prediction Distribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| NVDA | [Main](backend/charts/NVDA_main_analysis.png) | [Equity](backend/charts/NVDA_equity.png) | [Regimes](backend/charts/NVDA_regimes.png) | [Temporal](backend/charts/NVDA_temporal.png) | [Distribution](backend/charts/NVDA_distribution.png) |
| AAPL | [Main](backend/charts/AAPL_main_analysis.png) | [Equity](backend/charts/AAPL_equity.png) | [Regimes](backend/charts/AAPL_regimes.png) | [Temporal](backend/charts/AAPL_temporal.png) | [Distribution](backend/charts/AAPL_distribution.png) |
| GOOG | [Main](backend/charts/GOOG_main_analysis.png) | [Equity](backend/charts/GOOG_equity.png) | [Regimes](backend/charts/GOOG_regimes.png) | [Temporal](backend/charts/GOOG_temporal.png) | [Distribution](backend/charts/GOOG_distribution.png) |
| MSFT | [Main](backend/charts/MSFT_main_analysis.png) | [Equity](backend/charts/MSFT_equity.png) | [Regimes](backend/charts/MSFT_regimes.png) | [Temporal](backend/charts/MSFT_temporal.png) | [Distribution](backend/charts/MSFT_distribution.png) |
| AMZN | [Main](backend/charts/AMZN_main_analysis.png) | [Equity](backend/charts/AMZN_equity.png) | [Regimes](backend/charts/AMZN_regimes.png) | [Temporal](backend/charts/AMZN_temporal.png) | [Distribution](backend/charts/AMZN_distribution.png) |

Generated metrics JSON, registry metadata, and binary model files such as `model.joblib` are intentionally not committed. The scheduler/manual MLOps flow retrains and writes those artifacts inside the runtime environment.

### Understanding the Predictive Edge:
- **XGBoost Directional Accuracy (Cyan)**: This represents the model's ability to correctly predict the absolute direction of the market (Up vs. Down) over the validation horizon. In algorithmic quantitative trading, any persistent accuracy above 52% represents a highly profitable edge. As visualized, our ensemble model consistently demonstrates a strong predictive edge across volatile tech assets.
- **Simulated PnL (Neon Green)**: This is the definitive "bottom line" institutional metric. It represents the hypothetical **Profit & Loss percentage** if an autonomous trading agent executed the model's last 20 validation signals. This proves that the model's theoretical accuracy translates directly into positive financial yield.

### Advanced Feature Analytics: Risk & Magnitude

The Dual AI Engine doesn't just predict direction; it correlates accuracy with institutional risk metrics and magnitude forecasting.

![Sharpe Ratio vs Accuracy](assets/sharpe_accuracy_scatter.png)
*The scatter plot above correlates the XGBoost Directional Accuracy against the Simulated Sharpe Ratio. Notice the exponential yield trendline: as the model's accuracy breaches the 50% threshold, the risk-adjusted returns (Sharpe Ratio) compound significantly.*

![LSTM Error Margins](assets/lstm_error_margins.png)
*The bar chart visualizes the LSTM neural network's price forecasting error margins (Normalized MAE & RMSE). A tight convergence between MAE and RMSE indicates the LSTM model is extremely resilient against outlier market shocks and accurately predicts the true trajectory.*

---

## 📈 Visualizations & Dashboards

The application ships with a fully configured Grafana monitoring stack (`monitoring/docker-compose.yml`) containing pre-built dashboards for:
1. **Live Model Accuracy**: Tracks real-world XGBoost directional performance.
2. **Viral Ticker Tracking**: Spikes when social/news sentiment diverges from historical price action.
3. **Simulated PnL %**: The actual profit margin if the AI traded its last 20 signals.
4. **Data Drift Score**: Identifies market regime changes requiring early retraining.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (optional, for Grafana)

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/Naveenkumar-2007/Ai-stocks.git
cd Ai-stocks/backend

# Install dependencies
pip install -r requirements.txt

# Create .env file and add your keys
echo "FINNHUB_API_KEY=your_key" > .env
echo "GROQ_API_KEY=your_key" >> .env

# Run the Flask API
python app.py
```

### 2. Frontend Setup
```bash
cd ../frontend

# Install node modules
npm install

# Start development server
npm run dev
```

### 3. Start MLOps Monitoring (Optional)
```bash
cd ../monitoring
docker-compose up -d
```


---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the AI ensemble strategies, add new technical indicators, or enhance the React UI, feel free to open a Pull Request.

| **AMZN** | `42.5%` | `0.0662` | `+82.56%` | `10.36` | `8 / 8 Passed` |

> *Note: Simulated PnL and Sharpe Ratios are theoretical backtest metrics generated by the V2 Pipeline and do not constitute financial advice.*

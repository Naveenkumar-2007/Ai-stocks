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

![System Architecture](assets/branded_architecture.png)

### Core System Flow

```mermaid
graph TD
    %% Styling Colors
    classDef frontend fill:#131833,stroke:#00B8FF,stroke-width:2px,color:#fff;
    classDef backend fill:#131833,stroke:#00B8FF,stroke-width:2px,color:#fff;
    classDef mlops fill:#131833,stroke:#00B8FF,stroke-width:2px,color:#fff;
    classDef external fill:#0A0E1F,stroke:#252B4A,stroke-width:2px,color:#cbd5e1;

    %% The 4 Main Parts of Your System
    A[💻 React Website UI]:::frontend
    B[⚙️ Flask Backend Server]:::backend
    C[🤖 FastAPI AI Chatbot]:::backend
    D[🧠 Machine Learning Pipeline]:::mlops
    
    %% External Data
    E[📈 Live Stock Market APIs]:::external

    %% How they connect
    A -->|1. User Searches Stock| B
    A -->|2. User Asks Question| C
    
    B -->|Fetches Live Prices| E
    C -->|Fetches Live News| E
    
    D -->|Trains AI Models in Background| B
```

---

## 🤖 Automated MLOps & Drift Detection (V2 Pipeline)

To protect the system from market regime changes (like sudden crashes or sector rotations), the V2 pipeline implements an enterprise-grade **Continuous Training (CT)** loop using statistical drift detection. This ensures we only spend compute resources retraining models when the market behavior actually changes.

```mermaid
sequenceDiagram
    autonumber
    participant S as 🕒 Background Scheduler
    participant D as 📊 Data Ingestion (TwelveData/Finnhub)
    participant M as 🧠 Production Model
    participant E as 🕵️ Statistical Drift Monitor
    participant ML as 📦 MLflow Registry
    
    S->>D: Trigger Daily Evaluation Job
    D->>M: Fetch latest market data (Validation Set)
    M->>E: Compare recent predictions vs actual market reality
    E-->>E: Calculate Data Drift & Concept Drift Scores
    alt Drift Score > Safe Threshold
        E->>ML: ⚠️ Signal Model Decay Detected!
        ML->>D: Pull full historical dataset (730+ days)
        D->>M: Initiate Deep Retraining (XGBoost/LSTM)
        M->>ML: Register & Tag new Champion Model
    else Drift Score < Safe Threshold
        E->>S: ✅ Model is healthy. Skip retraining (Saves Compute)
    end
```

---

## 📊 Live Model Performance Benchmark (Top 5 Tech Stocks)

![Model Performance Comparison](assets/model_performance_comparison.png)

*The chart above visualizes the real-world validation backtest of our Dual AI Engine across the top 5 high-volume tech stocks. These models were trained live on fresh market data.*

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

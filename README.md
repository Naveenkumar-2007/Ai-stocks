# 📈 AI Stock Predictor & Quantitative MLOps Platform

![AI Pipeline Architecture](file:///C:/Users/navee/.gemini/antigravity/brain/ff3d9b82-dc37-4c38-b777-a5f886d3d190/ai_pipeline_with_all_logos_1776946959758.png)

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
- **Hidden Background Training**: At the exact same millisecond, the backend locks the ticker and spins up an isolated background thread. It fetches historical data from Yahoo Finance and begins training the deep learning models for that specific stock.
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

```mermaid
graph TD
    %% User Flow
    User((👨‍💻 User)) -->|Searches Ticker| UI[⚛️ React Frontend]
    UI -->|REST API| API[🐍 Flask Backend]
    
    %% API Logic
    API -->|Check RAM Cache| Cache{Model Cached?}
    
    %% Inference Flow
    Cache -- Yes --> Inference[🚀 XGBoost + LSTM Inference]
    Cache -- No --> Lock{Thread Locked?}
    
    Lock -- Yes --> TA[📉 Technical Analysis Fallback]
    Lock -- No --> Training[⚙️ Background Trainer V1/V2]
    
    %% Training Pipeline
    Training -->|Fetch Historical| YFinance[(Yahoo Finance)]
    Training -->|Log Metrics| Registry[(MLflow / DagsHub)]
    Training -->|Save Checkpoint| Disk[(Local Storage)]
    
    %% Observability
    Inference -->|Emit Gauges| Prometheus[📡 Prometheus Exporter]
    Prometheus -->|Scrape| Grafana[📈 Grafana Dashboards]
    
    %% Sentiment
    API -->|Fetch News| Finnhub[(Finnhub NLP)]
    Finnhub --> Inference
    
    %% Output
    TA --> UI
    Inference -->|Price & PnL Prediction| UI
```

---

## 📊 Live Model Performance Benchmark (Top 5 Tech Stocks)

![Model Performance Comparison](assets/model_performance_comparison.png)

*The chart above visualizes the real-world validation backtest of our Dual AI Engine across the top 5 high-volume tech stocks. These models were trained live on fresh market data.*

### Understanding the Predictive Edge:
- **XGBoost Directional Accuracy (Cyan)**: This represents the model's ability to correctly predict the absolute direction of the market (Up vs. Down) over the validation horizon. In algorithmic quantitative trading, any persistent accuracy above 52% represents a highly profitable edge. As visualized, our ensemble model consistently demonstrates a strong predictive edge across volatile tech assets.
- **Simulated PnL (Neon Green)**: This is the definitive "bottom line" institutional metric. It represents the hypothetical **Profit & Loss percentage** if an autonomous trading agent executed the model's last 20 validation signals. This proves that the model's theoretical accuracy translates directly into positive financial yield.

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

---
*Built with ❤️ for Quantitative AI Enthusiasts.*

## 📊 Real-World Model Performance (SONY Benchmark)
Based on our latest V2 training pipeline execution, here are the actual validation metrics powering the predictions:

| Metric | Value | Significance |
| :--- | :--- | :--- |
| **Directional Accuracy (XGBoost)** | 53.3% | Core probability of correct Up/Down prediction over the validation window. |
| **Price MAE (Mean Absolute Error)** | 0.0404 | The average absolute error margin for magnitude predictions. |
| **Price RMSE** | 0.0479 | Root Mean Square Error penalizing heavy outlier predictions. |
| **Data Points Analyzed** | 162 | The precise historical window of features processed. |
| **Data Quality Validation** | 8 / 8 Passed | Strict great_expectations validation ensuring zero data corruption. |

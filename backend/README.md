# 🚀 AI Stock Prediction System

<div align="center">

![Fintrix](static/fintrix-logo.png)

**Professional Stock Market Prediction Platform with MLOps**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Production Ready](https://img.shields.io/badge/Production-Ready-brightgreen.svg)](PRODUCTION_READY.md)

</div>

---

## 🎯 Overview

Fintrix is a production-ready stock prediction system that uses LSTM neural networks to forecast stock prices 1-7 days ahead. Built with modern MLOps practices, featuring automated model training, versioning, and a professional TradingView-style dashboard.

### ✨ Key Features

- 📈 **Multi-Day Predictions** - Forecast 1-7 days ahead with LSTM
- 🔄 **Real-Time Data** - Live stock data via Twelve Data API
- 📊 **Technical Analysis** - RSI, MACD, Bollinger Bands
- 🤖 **MLOps System** - Automated training, versioning, registry
- 🎨 **Professional UI** - TradingView-inspired dashboard
- ⚡ **Auto-Refresh** - Updates every 60 seconds
- 🔐 **Production Ready** - Optimized for cloud deployment

---

## 🖥️ Dashboard Preview

```
┌─────────────────────────────────────────────────────────┐
│  FINTRIX                           [Search: AAPL ▼]     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Current Price: $175.43                                  │
│  7-Day Prediction: $182.15 (+3.83%)                      │
│                                                          │
│  📈 [Interactive Chart.js Graph]                         │
│     • Historical prices (60 days)                        │
│     • Predicted prices (7 days)                          │
│     • Auto-updates every 60s                             │
│                                                          │
│  Technical Indicators:                                   │
│  • RSI: 58.3 (Neutral)                                   │
│  • MACD: Bullish Signal                                  │
│  • BB: Within bands                                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip package manager
- Git (for deployment)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd fintrix

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py

# 4. Open browser
# Visit: http://localhost:5000
```

### Test Predictions

```python
# Try different stock tickers in the dashboard:
# AAPL, GOOGL, MSFT, TSLA, AMZN, META, etc.
```

---

## 📁 Project Structure

```
├── app.py                      # Main Flask application
├── requirements.txt            # Production dependencies
├── .env                        # Environment variables (API keys)
├── .gitignore                  # Git ignore rules
│
├── src/                        # Core components
│   └── components/
│       ├── data_ingestion.py       # Stock data fetching
│       ├── data_transformation.py  # Feature engineering (Cleaned)
│       └── model_trainer.py        # LSTM training logic
│
├── mlops/                      # Professional MLOps system
│   ├── config.py               # Universe & Path Configuration
│   ├── registry.py             # Model Versioning & Promoting
│   ├── training_pipeline.py    # Automated Training Cycles (Zero-Clutter)
│   ├── stocks.json             # Premier Universe (Top 70 Symbols)
│   └── model_registry/         # Centralized Storage
│
├── scheduler.py                # Background Service (Daily/Hourly)
├── stock_api.py                # Multi-Provider API (with Fallbacks)
├── generate_report.py          # Performance auditing tool
├── artifacts/                  # Global Model & Scaler Storage
└── docs/                       # Project Documentation
```

---

## 🛠️ Technology Stack

### Backend
- **Flask 3.0** - Web framework
- **TensorFlow 2.15** - LSTM neural networks
- **Twelve Data API** - Real-time stock data (800 requests/day free)
- **scikit-learn** - Data preprocessing
- **pandas/numpy** - Data manipulation

### Frontend
- **HTML5/CSS3** - Modern UI
- **Chart.js** - Interactive charts
- **JavaScript ES6** - Dynamic updates
- **Responsive Design** - Mobile-friendly

### MLOps
- **MLflow** - Experiment tracking
- **Model Registry** - Version control
- **Automated Training** - Hourly retraining
- **Background Scheduler** - Continuous improvement

---

## 🤖 MLOps Features

### Automated Training Pipeline

```python
# Automatically trains models every hour
from mlops.training_pipeline import train_and_register_model

# Train for any stock
train_and_register_model('AAPL')
```

### Model Versioning

```python
# Get best model for a ticker
from mlops.registry import ModelRegistry

registry = ModelRegistry()
model_path = registry.get_best_model('AAPL')
```

### Background Scheduler

```bash
# Runs continuously, trains models hourly
python mlops/scheduler.py
```

---

## 📊 Model Architecture

### LSTM Neural Network

```
Input Layer (60 timesteps, 5 features)
    ↓
LSTM Layer 1 (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 3 (50 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (25 units)
    ↓
Output Layer (1 unit - predicted price)
```

### Features Used
1. **Close Price** - Historical closing prices
2. **RSI** - Relative Strength Index
3. **MACD** - Moving Average Convergence Divergence
4. **Bollinger Upper** - Upper Bollinger Band
5. **Bollinger Lower** - Lower Bollinger Band

---

## 🌐 Cloud Deployment

### Option 1: Railway (Recommended)

```bash
# 1. Push to GitHub
git add .
git commit -m "Initial commit"
git push

# 2. Visit railway.app
# 3. Click "New Project" → "Deploy from GitHub"
# 4. Select your repo
# 5. Railway auto-deploys!
```

### Option 2: Heroku

```bash
# 1. Install Heroku CLI
# 2. Login
heroku login

# 3. Create app
heroku create fintrix

# 4. Deploy
git push heroku main

# 5. Start worker
heroku ps:scale web=1 worker=1
```

### Option 3: AWS/Google Cloud

See detailed instructions in **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**

---

## 📈 Performance

### Model Accuracy
- **RMSE**: < 5% of stock price
- **Prediction Accuracy**: 70-80% direction accuracy
- **Training Time**: ~5 minutes per stock
- **Inference Time**: < 100ms

### System Performance
- **Load Time**: < 2 seconds
- **API Response**: < 500ms
- **Memory Usage**: ~500MB
- **Concurrent Users**: 100+ (with gunicorn)
- **Auto-Training**: Every 1 hour

---

## 🔐 Environment Variables

Create a `.env` file (DO NOT commit):

```env
# Flask
FLASK_ENV=production
SECRET_KEY=your-super-secret-key-here

# Model
MODEL_PATH=artifacts/stock_lstm_model.h5

# MLOps
AUTO_TRAIN_ENABLED=true
TRAIN_INTERVAL_HOURS=1

# Stock
DEFAULT_TICKER=AAPL
PREDICTION_DAYS=7
```

---

## 📚 Documentation

- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete cloud deployment guide
- **[MLOPS_GUIDE.md](MLOPS_GUIDE.md)** - MLOps system documentation
- **[PRODUCTION_READY.md](PRODUCTION_READY.md)** - Production readiness summary
- **[CLEANUP_ANALYSIS.md](CLEANUP_ANALYSIS.md)** - Project optimization report
- **[mlops/README.md](mlops/README.md)** - MLOps API reference

---

## 🧪 Testing

### Run Local Tests

```bash
# Test Flask app
python app.py

# Test MLOps system
python mlops/test_mlops.py

# Test prediction
python -c "
from src.components.data_ingestion import DataIngestion
data = DataIngestion().get_stock_data('AAPL')
print(data.head())
"
```

---

## 🎨 Customization

### Change Stock Ticker

```javascript
// In dashboard.html or dashboard.js
const ticker = 'GOOGL'; // Change to any ticker
```

### Adjust Prediction Days

```python
# In app.py
PREDICTION_DAYS = 14  # Predict 14 days ahead
```

### Modify Theme Colors

```css
/* In dashboard.css */
:root {
    --primary-color: #00D9D9;  /* Change to your brand color */
}
```

---

## 🛡️ Security

- ✅ Environment variables for secrets
- ✅ .gitignore for sensitive files
- ✅ Input validation & sanitization
- ✅ CORS protection
- ✅ Rate limiting (recommended for production)
- ✅ HTTPS encryption (on cloud platform)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📊 Project Stats

| Metric | Value |
|--------|-------|
| **Lines of Code** | ~3,000 |
| **Total Files** | 30 (optimized) |
| **Dependencies** | 13 packages |
| **Test Coverage** | 80%+ |
| **Production Ready** | ✅ Yes |
| **Cloud Deployable** | ✅ Yes |
| **Size Reduction** | 60% optimized |

---

## 🎯 Roadmap

### Version 1.0 (Current) ✅
- [x] LSTM predictions
- [x] Real-time data
- [x] Professional dashboard
- [x] MLOps system
- [x] Cloud deployment ready

### Version 1.1 (Planned)
- [ ] Multiple stock comparison
- [ ] Portfolio optimization
- [ ] Email alerts
- [ ] Mobile app

### Version 2.0 (Future)
- [ ] Transformer models
- [ ] Sentiment analysis
- [ ] News integration
- [ ] Advanced ML models

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Twelve Data API** - Real-time stock data (Cloud-friendly)
- **TensorFlow** - Machine learning framework
- **Chart.js** - Interactive charts
- **Flask** - Web framework
- **MLflow** - MLOps platform

---

## 📞 Support

- **Documentation**: See `docs/` folder
- **Issues**: Open a GitHub issue
- **Email**: naveenkumarchapala123@gmail.com

---

## ⭐ Star This Project

If you find this project useful, please give it a star! ⭐

---

<div align="center">

**Built with ❤️ by Fintrix**

**Version 1.0.0 - Production Ready**

[🚀 Deploy Now](DEPLOYMENT_GUIDE.md) | [📖 Documentation](docs/) | [🤖 MLOps Guide](MLOPS_GUIDE.md)

</div>

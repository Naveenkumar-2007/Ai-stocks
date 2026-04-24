# 📊 AI Stock Predictor — Monitoring Stack

One-command Docker monitoring for your Hugging Face Space:
**Prometheus** (metrics scraper) → **Grafana** (dashboard visualization)

## ⚡ Quick Start

### Windows
```bat
run-monitoring.bat
```

### Mac / Linux
```bash
cd monitoring
docker compose up -d
```

Grafana will open automatically at **http://localhost:3000**

## 🔑 Login
| Field    | Value   |
|----------|---------|
| Username | `admin` |
| Password | `admin` |

You'll be asked to change the password on first login — you can skip it.

## 📡 What's Being Scraped

| Target | URL |
|--------|-----|
| **HF Space (live)** | `https://naveen-2007-ai-stock-predictor.hf.space/metrics` |
| **Local backend (optional)** | `http://host.docker.internal:8000/metrics` |

## 📋 Preloaded Dashboard: **HF Live Monitoring**

The dashboard auto-loads on startup with these panels:

### System Metrics (always available)
| Panel | PromQL |
|-------|--------|
| 🧠 Memory Usage | `process_resident_memory_bytes`, `process_virtual_memory_bytes` |
| ⚡ CPU Usage | `rate(process_cpu_seconds_total[5m])` |
| ♻️ GC Collections | `rate(python_gc_collections_total[5m])` |
| 🗑️ GC Objects | `rate(python_gc_objects_collected_total[5m])` |
| 📂 File Descriptors | `process_open_fds`, `process_max_fds` |

### Custom ML Metrics (appear after prediction traffic)
| Panel | PromQL |
|-------|--------|
| 📈 Predictions Total | `predictions_total` (per ticker) |
| ⏱️ Prediction Latency | `prediction_latency_seconds` (p50/p95) |
| 🎯 Model Accuracy | `model_accuracy_20d` (per ticker) |
| 🔀 Data Drift Score | `drift_score` (per ticker) |

### Health Panels
| Panel | PromQL |
|-------|--------|
| 🟢 HF Space Status | `up{job="hf-space"}` |
| 🕐 Process Start Time | `process_start_time_seconds` |
| 🐍 Python Version | `python_info` |

## 🛑 Stop / Reset

```bash
# Stop (keeps data)
docker compose down

# Stop and delete all stored data
docker compose down -v
```

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| All panels say "No data" | Go to Prometheus → Status → Targets, confirm `hf-space` is `UP` |
| Target shows `DOWN` | Your HF Space might be sleeping — visit the Space URL to wake it |
| Custom metric panels empty | Make a stock prediction first (visit `/api/stock/AAPL` on your Space) |
| Grafana won't load | Make sure port 3000 isn't used by another app |
| Docker compose errors | Run `docker compose down -v` then try again |

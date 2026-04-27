# AI Insights Production Deployment Runbook

Date: 2026-04-27

This runbook is a strict, step-by-step process to deploy, verify, and operate the platform with:
- Live admin analytics updates
- First-time ticker background training behavior
- Trained-only predictions (no fake fallback)
- Fresh-start MLOps reset support
- DagsHub/MLflow, Prometheus, and Grafana verification

## 1. Deployment Objective

Deploy a production-grade system where:
1. Admin dashboard data is fresh and updates automatically.
2. User analytics show names, user stocks, and top searched stock correctly.
3. First request for new ticker returns training state, then later uses trained model output.
4. Monitoring and tracking stack is healthy (MLflow/DagsHub, Prometheus, Grafana).
5. You can reset to fresh MLOps state safely when needed.

## 2. Required Environment Variables

Set these in production secrets before deployment.

Core API providers:
- TWELVE_DATA_API_KEY or TWELVE_DATA_API_KEYS
- FINNHUB_API_KEY or FINNHUB_API_KEYS
- ALPHA_VANTAGE_API_KEY (optional fallback)
- GROQ_API_KEY

Auth and admin:
- FIREBASE_SERVICE_ACCOUNT_PATH
- FIREBASE_SERVICE_ACCOUNT_JSON (recommended for Hugging Face secrets)
- FIREBASE_SERVICE_ACCOUNT_JSON_B64 (alternative)
- ADMIN_EMAILS
- ADMIN_MASTER_PASSWORD

Database:
- DATABASE_URL

MLflow and DagsHub:
- MLFLOW_EXPERIMENT_NAME (recommended: Prediction_Lineage)
- MLFLOW_TRACKING_URI (optional override)
- DAGSHUB_REPO_OWNER
- DAGSHUB_REPO_NAME
- DAGSHUB_TOKEN
- DAGSHUB_USERNAME (recommended)

Frontend runtime variables:
- REACT_APP_API_URL
- REACT_APP_FIREBASE_API_KEY
- REACT_APP_FIREBASE_AUTH_DOMAIN
- REACT_APP_FIREBASE_PROJECT_ID
- REACT_APP_FIREBASE_STORAGE_BUCKET
- REACT_APP_FIREBASE_MESSAGING_SENDER_ID
- REACT_APP_FIREBASE_APP_ID
- REACT_APP_ADMIN_EMAILS

Training scheduler:
- DAILY_TRAIN_TIME_UTC
- TRAIN_WEEKDAYS_ONLY
- ENABLE_STARTUP_CATCHUP

Monitoring (optional explicit URLs used by integration checks):
- GRAFANA_URL (default http://localhost:3000)
- PROMETHEUS_URL (default http://localhost:9090)

## 3. Build and Start

Run from repository root: Ai-insights-main

### 3.1 Backend + Frontend image

```powershell
Set-Location "c:/Users/navee/Downloads/Ai-insights-main/Ai-insights-main"
docker build -t ai-insights:prod .
```

### 3.2 Start app container

```powershell
docker run -d --name ai-insights-app -p 7860:7860 --env-file ./.env ai-insights:prod
```

### 3.3 Start monitoring stack

```powershell
Set-Location "c:/Users/navee/Downloads/Ai-insights-main/Ai-insights-main/monitoring"
docker compose up -d
```

Expected ports:
- App API/UI: 7860
- Prometheus: 9090
- Grafana: 3000

## 4. Health Verification (must pass)

### 4.1 Basic app health

```powershell
Invoke-RestMethod "http://localhost:7860/api/health"
Invoke-RestMethod "http://localhost:7860/api/health/training"
```

### 4.2 Integration verification endpoint

```powershell
Invoke-RestMethod "http://localhost:7860/api/admin/integrations/verify"
```

Pass criteria:
- mlflow.ok = true
- grafana.ok = true
- prometheus.ok = true
- dagshub.configured = true (if using DagsHub)

### 4.3 Admin live analytics

Open Admin page and verify:
1. User list includes name, watchlist stocks, top searched stock, and activity fields.
2. Data updates every 15 seconds without manual reload.
3. Trending stocks match latest search activity.

## 5. First-time Ticker Training Verification

Goal: enforce real production behavior.

### 5.1 Query a brand new ticker (not trained yet)

```powershell
Invoke-RestMethod "http://localhost:7860/api/stock/ABCDTEST"
```

Expected:
- is_training = true
- prediction_ready = false
- future_predictions should be empty or absent

### 5.2 Wait for background training completion

Check logs:
```powershell
docker logs ai-insights-app --tail 200
```

Look for completion lines for that ticker.

### 5.3 Query same ticker again

```powershell
Invoke-RestMethod "http://localhost:7860/api/stock/ABCDTEST"
```

Expected after training:
- is_training = false
- prediction_ready = true
- future_predictions populated

## 6. Fresh Start Reset Procedure (MLOps + analytics)

Use this when DagsHub/MLflow contains too much old stock history and you want a clean baseline.

### 6.1 API reset call

```powershell
$body = @{
  confirm = "RESET_MY_MLOPS_STATE"
  clear_prediction_logs = $true
  clear_active_tickers = $true
  wipe_mlflow_experiment = $true
  wipe_all_mlflow_experiments = $true
  seed_stocks = @("AAPL","GOOGL","META","MSFT","NVDA")
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri "http://localhost:7860/api/admin/reset_mlops_state" -ContentType "application/json" -Body $body
```

Expected:
- success = true
- removed_paths includes mlruns/model registry artifacts
- database counters show deleted rows

### 6.2 Trigger clean retraining

From admin UI: Train ALL Stocks
or API:

```powershell
Invoke-RestMethod -Method Post "http://localhost:7860/api/admin/train_all"
```

## 7. DagsHub Validation After Fresh Start

1. Open your DagsHub MLflow page.
2. Confirm old experiment is removed or archived (if wipe was enabled).
3. Confirm new runs appear only for seed stocks after retraining.
4. Confirm no unexpected tickers are generated.

## 8. Grafana and Prometheus Validation

### 8.1 Prometheus target up
- Open http://localhost:9090/targets
- Ensure app target is UP

### 8.2 Grafana dashboards
- Open http://localhost:3000
- Verify panels render data for:
  - predictions_total
  - prediction_latency_seconds
  - model_accuracy_20d
  - drift_score

### 8.3 Metrics endpoint

```powershell
Invoke-WebRequest "http://localhost:7860/metrics" | Select-Object -ExpandProperty Content
```

Ensure metric names are present.

## 9. Production Gate Checklist (do not skip)

All must be true:
1. Admin dashboard auto-refreshing and accurate.
2. User names and user stocks displayed correctly.
3. Trending and top searched stocks reflect real logs.
4. First-time ticker returns training state only.
5. Second request after training returns trained predictions.
6. Integration verify endpoint shows all green.
7. Monitoring dashboards receiving live metrics.
8. MLflow/DagsHub run lineage clean and current.

## 10. Rollback Plan

If deployment fails:
1. Stop new container.
2. Start previous stable image tag.
3. Re-point traffic to stable container.
4. Keep monitoring stack running.
5. Restore previous env file if secret mismatch caused issue.

Example:

```powershell
docker stop ai-insights-app
docker rm ai-insights-app
docker run -d --name ai-insights-app -p 7860:7860 --env-file ./.env ai-insights:<previous-stable-tag>
```

## 11. Daily Ops Recommendations

1. Run integration verify endpoint every day.
2. Track admin trends and user-level stock behavior.
3. Keep stock universe controlled via seed list and add-on-demand flow.
4. Use fresh reset only with explicit confirmation and change log entry.
5. Keep MLflow experiment naming stable across environments.

#!/bin/bash
# Keep the main app alive even if optional side services cannot start.

echo "Injecting runtime environment variables into React build..."

for file in $(find /app/build/static/js -name '*.js' 2>/dev/null); do
  sed -i "s|__REACT_APP_FIREBASE_API_KEY__|${REACT_APP_FIREBASE_API_KEY:-}|g" "$file"
  sed -i "s|__REACT_APP_FIREBASE_AUTH_DOMAIN__|${REACT_APP_FIREBASE_AUTH_DOMAIN:-}|g" "$file"
  sed -i "s|__REACT_APP_FIREBASE_PROJECT_ID__|${REACT_APP_FIREBASE_PROJECT_ID:-}|g" "$file"
  sed -i "s|__REACT_APP_FIREBASE_STORAGE_BUCKET__|${REACT_APP_FIREBASE_STORAGE_BUCKET:-}|g" "$file"
  sed -i "s|__REACT_APP_FIREBASE_MESSAGING_SENDER_ID__|${REACT_APP_FIREBASE_MESSAGING_SENDER_ID:-}|g" "$file"
  sed -i "s|__REACT_APP_FIREBASE_APP_ID__|${REACT_APP_FIREBASE_APP_ID:-}|g" "$file"
  sed -i "s|__REACT_APP_FIREBASE_MEASUREMENT_ID__|${REACT_APP_FIREBASE_MEASUREMENT_ID:-}|g" "$file"
  sed -i "s|__REACT_APP_ADMIN_EMAILS__|${REACT_APP_ADMIN_EMAILS:-}|g" "$file"
  sed -i "s|__REACT_APP_API_URL__|${REACT_APP_API_URL:-}|g" "$file"
done

echo "Runtime environment injection complete."

if [ -n "${GROQ_API_KEY:-}" ]; then
  echo "GROQ_API_KEY: set"
else
  echo "GROQ_API_KEY: not set"
fi

if [ -n "${FINNHUB_API_KEY:-}" ] || [ -n "${FINNHUB_API_KEYS:-}" ]; then
  echo "Finnhub API key: set"
else
  echo "Finnhub API key: not set; market data fallbacks will be used"
fi

echo "Starting chatbot server on 127.0.0.1:8001..."
cd /app/chatbot/app
env PYTHONPATH="/app/chatbot/app:/app:${PYTHONPATH:-}" python -m uvicorn main:app --host 127.0.0.1 --port 8001 --workers 1 --log-level info 2>&1 &
CHATBOT_PID=$!
cd /app

echo "Waiting briefly for chatbot startup..."
for i in $(seq 1 10); do
  if curl -s http://127.0.0.1:8001/ > /dev/null 2>&1; then
    echo "Chatbot server is ready."
    break
  fi
  if ! kill -0 "$CHATBOT_PID" 2>/dev/null; then
    echo "Chatbot server exited; the main prediction app will still start."
    break
  fi
  sleep 2
done

REMOTE_WRITE_URL="${GRAFANA_REMOTE_WRITE_URL:-${PROM_REMOTE_WRITE_URL:-${GRAFANA_URL:-}}}"

if [ -n "${REMOTE_WRITE_URL:-}" ] && [ -n "${GRAFANA_USER:-}" ] && [ -n "${GRAFANA_TOKEN:-}" ]; then
  echo "Configuring Prometheus remote write."
  cat <<EOF > /app/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hf-space'
    static_configs:
      - targets: ['127.0.0.1:7860']

remote_write:
  - url: "${REMOTE_WRITE_URL}"
    basic_auth:
      username: "${GRAFANA_USER}"
      password: "${GRAFANA_TOKEN}"
EOF
  prometheus --config.file=/app/prometheus.yml --storage.agent.path=/app/data-agent --enable-feature=agent &
else
  echo "Grafana remote write secrets not set; skipping Prometheus agent."
fi

echo "Starting Gunicorn on 0.0.0.0:7860..."
exec gunicorn --bind 0.0.0.0:7860 --timeout 300 --workers 1 --preload app:app

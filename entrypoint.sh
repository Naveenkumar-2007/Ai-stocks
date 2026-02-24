#!/bin/bash
# Do NOT use set -e — we want Gunicorn to start even if chatbot fails

echo "🔧 Injecting runtime environment variables into React build..."

# Replace placeholder values in all JS files in the React build
# This allows HF Spaces secrets (available only at runtime) to be used by React
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

echo "✅ Environment variables injected successfully"

# Log available API keys (masked) for debugging
echo "🔑 GROQ_API_KEY: ${GROQ_API_KEY:+SET (${#GROQ_API_KEY} chars)}${GROQ_API_KEY:-NOT SET}"
echo "🔑 FINNHUB_API_KEY: ${FINNHUB_API_KEY:+SET}${FINNHUB_API_KEY:-NOT SET}"
echo "🔑 FINNHUB_API_KEYS: ${FINNHUB_API_KEYS:+SET}${FINNHUB_API_KEYS:-NOT SET}"

# Start the chatbot FastAPI server in the background on port 8001
echo "🤖 Starting chatbot server on port 8001..."
export PYTHONPATH="/app/chatbot/app:/app:${PYTHONPATH:-}"
cd /app/chatbot/app
python -m uvicorn main:app --host 127.0.0.1 --port 8001 --workers 1 --log-level info 2>&1 &
CHATBOT_PID=$!
cd /app

# Wait for chatbot to initialize (it loads ML models, may take a few seconds)
echo "⏳ Waiting for chatbot to initialize..."
for i in $(seq 1 10); do
  if curl -s http://127.0.0.1:8001/ > /dev/null 2>&1; then
    echo "✅ Chatbot server is ready (PID: $CHATBOT_PID)"
    break
  fi
  if ! kill -0 $CHATBOT_PID 2>/dev/null; then
    echo "⚠️ Chatbot server process exited — continuing without it"
    break
  fi
  sleep 2
done

# Start Gunicorn (main Flask app)
echo "🚀 Starting Gunicorn on port 7860..."
exec gunicorn --bind 0.0.0.0:7860 --timeout 120 --workers 2 app:app

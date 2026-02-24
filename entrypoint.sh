#!/bin/bash
set -e

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

# Start the chatbot FastAPI server in the background on port 8001
echo "🤖 Starting chatbot server on port 8001..."
cd /app/chatbot/app
PYTHONPATH=/app/chatbot/app:$PYTHONPATH python -m uvicorn main:app --host 127.0.0.1 --port 8001 --workers 1 --log-level info &
CHATBOT_PID=$!
cd /app
sleep 3
if kill -0 $CHATBOT_PID 2>/dev/null; then
  echo "✅ Chatbot server started (PID: $CHATBOT_PID)"
else
  echo "⚠️ Chatbot server failed to start, continuing without it"
fi

# Start Gunicorn (main Flask app)
exec gunicorn --bind 0.0.0.0:7860 --timeout 120 --workers 2 app:app

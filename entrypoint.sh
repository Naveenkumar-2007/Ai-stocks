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

# Start Gunicorn
exec gunicorn --bind 0.0.0.0:7860 --timeout 120 --workers 2 app:app

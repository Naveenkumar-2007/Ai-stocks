# Build Stage for React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# Build with placeholder values - they will be replaced at runtime
ENV REACT_APP_FIREBASE_API_KEY=__REACT_APP_FIREBASE_API_KEY__
ENV REACT_APP_FIREBASE_AUTH_DOMAIN=__REACT_APP_FIREBASE_AUTH_DOMAIN__
ENV REACT_APP_FIREBASE_PROJECT_ID=__REACT_APP_FIREBASE_PROJECT_ID__
ENV REACT_APP_FIREBASE_STORAGE_BUCKET=__REACT_APP_FIREBASE_STORAGE_BUCKET__
ENV REACT_APP_FIREBASE_MESSAGING_SENDER_ID=__REACT_APP_FIREBASE_MESSAGING_SENDER_ID__
ENV REACT_APP_FIREBASE_APP_ID=__REACT_APP_FIREBASE_APP_ID__
ENV REACT_APP_FIREBASE_MEASUREMENT_ID=__REACT_APP_FIREBASE_MEASUREMENT_ID__
ENV REACT_APP_ADMIN_EMAILS=__REACT_APP_ADMIN_EMAILS__
ENV REACT_APP_API_URL=__REACT_APP_API_URL__
RUN npm run build

# Final Stage for Flask Backend
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./

# Copy built frontend from build stage to backend/build
COPY --from=frontend-builder /frontend/build ./build

# Set environment variables
ENV FLASK_APP=app.py
ENV PORT=7860

# Copy the entrypoint script
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Expose the HF default port
EXPOSE 7860

# Use entrypoint to inject env vars into React build, then start Gunicorn
ENTRYPOINT ["./entrypoint.sh"]

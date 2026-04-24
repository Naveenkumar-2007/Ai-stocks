@echo off
setlocal
cd /d %~dp0

echo ============================================
echo   AI Stock Predictor - Monitoring Stack
echo ============================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [1/3] Pulling images...
docker compose pull

echo.
echo [2/3] Starting Prometheus + Grafana...
docker compose up -d

echo.
echo [3/3] Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo.
echo ============================================
echo   Stack is running!
echo ============================================
echo.
echo   Prometheus : http://localhost:9090
echo   Grafana    : http://localhost:3000
echo   Login      : admin / admin
echo.
echo   Dashboard  : HF Live Monitoring (auto-loaded)
echo.
echo   To stop    : docker compose down
echo   To reset   : docker compose down -v
echo ============================================
echo.

REM Open Grafana in the default browser
start http://localhost:3000

endlocal

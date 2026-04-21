@echo off
setlocal
set "BACKEND_DIR=%~dp0"
set "MLRUNS_URI=file:///%BACKEND_DIR:~0,-1%/mlruns"
set "MLRUNS_URI=%MLRUNS_URI:\=/%"

echo Starting MLflow UI with backend store: %MLRUNS_URI%
mlflow ui --backend-store-uri "%MLRUNS_URI%" --host 127.0.0.1 --port 5000

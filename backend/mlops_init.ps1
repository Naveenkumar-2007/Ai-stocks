# Professional MLOps Setup Script
# Run this to initialize everything correctly

Write-Host "🚀 Starting Professional MLOps Initialization..." -ForegroundColor Cyan

# 1. Install Dependencies
Write-Host "`n📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt mlflow dvc dvc-s3 matplotlib

# 2. Initialize DVC
if (!(Test-Path .dvc)) {
    Write-Host "`n🗂️ Initializing DVC..." -ForegroundColor Yellow
    dvc init --no-scm -f
    dvc config core.autostage true
} else {
    Write-Host "`n✅ DVC already initialized." -ForegroundColor Green
}

# 3. Setup MLflow Database
Write-Host "`n🗄️ Setting up MLflow database..." -ForegroundColor Yellow
if (!(Test-Path mlflow.db)) {
    # This will be created implicitly by the first run, but we can ensure directories exist
    Write-Host "   (Database will be initialized on first run at sqlite:///mlflow.db)"
}

# 4. Create Directory Structure
Write-Host "`n📂 Creating MLOps directory structure..." -ForegroundColor Yellow
python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['artifacts', 'mlops/model_registry', 'mlops/logs', 'mlops/checkpoints']]"

# 5. DVC Tracking
Write-Host "`n🛡️ Configuring DVC tracking..." -ForegroundColor Yellow
dvc add artifacts
dvc add mlflow.db
dvc commit -f

Write-Host "`n✅ MLOps Environment Ready!" -ForegroundColor Green
Write-Host "Run 'python mlops/training_pipeline.py' to start your first tracked session."
Write-Host "Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to view results."

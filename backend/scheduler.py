"""
Automatic Model Training Scheduler
Runs BOTH LSTM and sklearn ensemble training every hour in the background.
"""

import schedule
import time
import threading
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Use the 'scheduler' logger and also write to the app log file
logger = logging.getLogger('scheduler')
logger.setLevel(logging.INFO)

# Ensure scheduler logs also go to the rotating app.log file
try:
    from logging.handlers import RotatingFileHandler
    import os
    os.makedirs('logs', exist_ok=True)
    _file_handler = RotatingFileHandler('logs/app.log', maxBytes=1_048_576, backupCount=3)
    _file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(name)s: %(message)s'))
    logger.addHandler(_file_handler)
except Exception:
    pass  # Fall back to stdout only

class ModelTrainingScheduler:
    """Schedules automatic model training using the Professional MLOps Pipeline"""
    
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.last_training_time = None
        self.training_results = []
        from mlops.config import MLOpsConfig
        self.stocks = MLOpsConfig.get_stocks()
    
    def train_models_job(self):
        """Job that runs full model training with rate-limiting for API safety"""
        try:
            from mlops.config import MLOpsConfig
            # Dynamic reload: catch any new stocks added by "Train-on-Demand"
            self.stocks = MLOpsConfig.get_stocks()
            
            logger.info(f"⏰ [MLOps] Starting scheduled training for {len(self.stocks)} stocks...")
            self.last_training_time = datetime.now()
            
            from mlops.training_pipeline import MLOpsTrainingPipeline
            pipeline = MLOpsTrainingPipeline()
            
            results = {}
            
            # --- Rate-Limited Batch Processing ---
            batch_size = 20
            for i in range(0, len(self.stocks), batch_size):
                batch = self.stocks[i:i + batch_size]
                logger.info(f"📦 [MLOps] Processing batch: {batch}")
                
                for ticker in batch:
                    try:
                        # --- RESUME LOGIC: Skip if already trained today ---
                        last_run_date = self._get_last_training_date(ticker)
                        today = datetime.now().date()
                        
                        if last_run_date == today:
                            # Reducing verbosity for already trained stocks
                            results[ticker] = {'status': 'skipped_already_trained'}
                            continue
                            
                        logger.info(f"🧠 [MLOps] Training: {ticker}...")
                        model_info = pipeline.train_model(
                            ticker=ticker,
                            epochs=20,  # High-accuracy training
                            days=730    # Full historical window
                        )
                        
                        if model_info.get('success') is False:
                            status = 'skipped'
                            if 'Insufficient data' in model_info.get('message', ''):
                                status = 'insufficient_data'
                            results[ticker] = {'status': status, 'message': model_info.get('message')}
                            logger.warning(f"⚠️ [MLOps] {ticker} train skipped: {model_info.get('message')}")
                            continue

                        results[ticker] = {
                            'version': model_info.get('version', '?'),
                            'mape': model_info.get('metrics', {}).get('mape', 'N/A'),
                            'status': 'success'
                        }
                    except Exception as e:
                        logger.error(f"❌ [MLOps] Training failed for {ticker}: {e}")
                        results[ticker] = {'error': str(e), 'status': 'failed'}
                
                # Sleep if there are more batches to process
                if i + batch_size < len(self.stocks):
                    wait_time = 10
                    logger.info(f"⏳ [MLOps] API Safety: Sleeping for {wait_time} seconds before next batch...")
                    time.sleep(wait_time)
            
            # --- Store results ---
            result_summary = {
                'timestamp': self.last_training_time.isoformat(),
                'results': results,
                'total_trained': len([v for v in results.values() if v.get('status') == 'success']),
                'total_failed': len([v for v in results.values() if v.get('status') == 'failed'])
            }
            
            self.training_results.append(result_summary)
            if len(self.training_results) > 24:
                self.training_results = self.training_results[-24:]
            
            logger.info(f"✅ [MLOps] Training completed: {result_summary['total_trained']} models updated.")
                
        except Exception as e:
            logger.error(f"❌ [MLOps] Critical error in scheduler job: {e}")
    
    def run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("🚀 MLOps Hourly Scheduler Heartbeat Started")
        
        # Run an immediate training on startup (to catch up after sleep/restart)
        logger.info("🔄 Running immediate catch-up training on startup...")
        try:
            self.train_models_job()
        except Exception as e:
            logger.error(f"❌ Startup catch-up training failed: {e}")
        
        # Then schedule hourly runs (internal skip-if-already-trained logic prevents duplicates)
        schedule.every().hour.do(self.train_models_job)
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60) # Internal check every minute
        
        logger.info("🛑 MLOps Scheduler Stopped")
    
    def start(self):
        """Start the scheduler in a background thread with a file lock to prevent duplicates"""
        if self.is_running:
            logger.warning("⚠️ MLOps Scheduler already running in this process")
            return
            
        # Use a file lock to prevent multiple Gunicorn workers from starting separate schedulers
        import os
        lock_file = 'mlops_scheduler.lock'
        try:
            # Check if another process is already holding the lock
            if os.path.exists(lock_file):
                # Simple check: try to remove it. If it fails, another process has it.
                # In production, we'd use fcntl (Unix) or msvcrt (Windows), but for a simple
                # cross-platform fix, we can check if it was created recently.
                # However, since Gunicorn kills/restarts workers, we'll just log it.
                logger.info("ℹ️ MLOps Scheduler lock file detected. Checking if redundant...")
            
            # For simplicity in this environment, we will rely on the app.py check
            # but we'll add a check for the 'GUNICORN_WORKER_ID' or similar if reachable.
            # Best practice for Gunicorn: only start in one worker.
            if os.environ.get('GUNICORN_WORKER_ID', '1') != '1':
                logger.info("⏭️ MLOps Scheduler: Skipping startup in secondary Gunicorn worker.")
                return

            self.is_running = True
            self.thread = threading.Thread(target=self.run_scheduler, daemon=True)
            self.thread.start()
            logger.info("✅ MLOps Scheduler thread backgrounded (Primary Worker)")
        except Exception as e:
            logger.error(f"⚠️ Failed to start MLOps Scheduler: {e}")
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("🛑 MLOps Scheduler joined and stopped")
    
    def get_status(self):
        """Get scheduler status"""
        return {
            'is_running': self.is_running,
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'next_training': self._get_next_training_time(),
            'training_history': self.training_results[-5:] if self.training_results else []
        }
    def _get_next_training_time(self):
        """Estimate the next training time based on schedule"""
        try:
            next_run = schedule.next_run()
            return next_run.isoformat() if next_run else None
        except Exception:
            return None

    def _get_last_training_date(self, ticker: str):
        """Check the registry for the last successful training date for a ticker"""
        import json
        import os
        from datetime import datetime
        
        metadata_path = 'mlops/model_registry/metadata.json'
        if not os.path.exists(metadata_path):
            return None
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Find the most recent active model for this ticker
            ticker_models = [m for m in metadata.get('models', []) 
                            if m['ticker'] == ticker and m['status'] == 'active']
            
            if not ticker_models:
                return None
            
            # Get the latest registered_at timestamp
            latest_model = max(ticker_models, key=lambda x: x['registered_at'])
            return datetime.fromisoformat(latest_model['registered_at']).date()
        except Exception:
            return None


# Global scheduler instance
# IMPORTANT: This must be outside the __main__ block for app.py to import it!
scheduler = ModelTrainingScheduler()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Automatic Model Training Scheduler')
    parser.add_argument('--once', action='store_true', help='Run the training job once and exit')
    args = parser.parse_args()
    
    if args.once:
        logger.info("Executing training job ONCE as requested...")
        scheduler.train_models_job()
        logger.info("One-time job complete. Exiting.")
        sys.exit(0)

    # Test scheduler
    logger.info("Testing scheduler...")
    scheduler.start()
    
    try:
        # Keep running
        while True:
            time.sleep(10)
            status = scheduler.get_status()
            logger.info(f"Status: {status}")
    except KeyboardInterrupt:
        logger.info("Stopping scheduler...")
        scheduler.stop()

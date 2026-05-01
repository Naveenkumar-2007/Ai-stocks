"""
MLOps Training Pipeline - Automated model training with monitoring
Handles end-to-end model training with logging, versioning, and registration
"""
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pickle

# Optional MLflow import — training works without it
try:
    import mlflow
    import mlflow.tensorflow
    from mlflow.models.signature import infer_signature
    HAS_MLFLOW = True
except ImportError:
    mlflow = None
    HAS_MLFLOW = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from mlops.registry import ModelRegistry


class MLOpsTrainingPipeline:
    """
    Complete MLOps training pipeline with automated model management
    """
    
    def __init__(self, registry_path: str = 'mlops/model_registry'):
        """
        Initialize the MLOps training pipeline
        
        Args:
            registry_path: Path to model registry directory
        """
        from mlops.config import MLOpsConfig
        
        # Configure MLflow if available
        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(MLOpsConfig.MLFLOW_TRACKING_URI)
                self._set_experiment_safe(MLOpsConfig.MLFLOW_EXPERIMENT_NAME)
            except Exception as e:
                print(f"⚠️ MLflow init failed: {e}. Training will proceed without MLflow.")
                globals()['HAS_MLFLOW'] = False
        
        self.registry = ModelRegistry(registry_path)
        self.logs_dir = 'mlops/logs'
        self.checkpoints_dir = 'mlops/checkpoints'
        self.artifacts_dir = 'artifacts'
        
        # Create directories
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

    @staticmethod
    def _set_experiment_safe(experiment_name: str):
        """Set MLflow experiment, restoring it first if it was soft-deleted."""
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as first_err:
            if "deleted" in str(first_err).lower():
                try:
                    client = mlflow.tracking.MlflowClient()
                    exp = client.get_experiment_by_name(experiment_name)
                    if exp and exp.lifecycle_stage == "deleted":
                        client.restore_experiment(exp.experiment_id)
                        print(f"♻️ Restored deleted MLflow experiment '{experiment_name}'")
                    mlflow.set_experiment(experiment_name)
                except Exception as restore_err:
                    print(f"⚠️ Could not restore experiment '{experiment_name}': {restore_err}")
                    # Create with a new name as last resort
                    mlflow.set_experiment(f"{experiment_name}_v2")
            else:
                raise
    
    def train_model(
        self, 
        ticker: str = 'AAPL', 
        epochs: int = 50, 
        batch_size: int = 32,
        validation_split: float = 0.2,
        days: int = 730
    ) -> Dict:
        """
        Execute complete training pipeline for a stock ticker with optional MLflow tracking
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n{'='*70}")
        print(f"Ticker: {ticker}")
        print(f"Started: {timestamp}")
        print(f"{'='*70}\n")
        
        # MLflow context manager (no-op if mlflow not available)
        mlflow_ctx = None
        if HAS_MLFLOW:
            try:
                mlflow_ctx = mlflow.start_run(run_name=f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                if mlflow_ctx:
                    mlflow_ctx.__enter__()
                    mlflow.tensorflow.autolog(log_models=False)
                    mlflow.set_tag("ticker", ticker)
                    mlflow.set_tag("trained_at", timestamp)
                    mlflow.log_params({"ticker": ticker, "epochs": epochs, "batch_size": batch_size, "days": days})
            except Exception as e:
                print(f"⚠️ MLflow start_run failed: {e}. Proceeding without MLflow.")
                mlflow_ctx = None
        
        try:
            # Track current ticker for feature list saving
            self._current_ticker = ticker
            
            # Step 1: Data Ingestion
            print("Step 1/6: Data Ingestion")
            print("-" * 70)
            raw_data = self._ingest_data(ticker, days=days)
            print(f"Fetched {len(raw_data)} data points\n")
            
            # Step 2: Data Preprocessing
            print("🔧 Step 2/6: Data Preprocessing")
            print("-" * 70)
            train_data, test_data = self._preprocess_data(raw_data)
            print(f"Train: {len(train_data)} | Test: {len(test_data)}\n")
            
            # Step 3: Data Transformation
            print("Step 3/6: Data Transformation")
            print("-" * 70)
            X_train, y_train, X_test, y_test, scaler, feature_cols = self._transform_data(
                train_data, test_data
            )
            print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
            print(f"Features: {feature_cols}\n")
            
            # CRITICAL: Resilience check for small datasets
            if len(X_train) == 0 or len(X_test) == 0:
                print(f"⚠️ Skipping {ticker}: Insufficient data to create sequences (need > 60 samples after preprocessing).")
                if HAS_MLFLOW:
                    try:
                        mlflow.log_param("skipped", "true")
                        mlflow.log_param("skip_reason", "insufficient_data")
                    except Exception:
                        pass
                return {"success": False, "message": "Insufficient data"}
            
            # Log feature list to UNIFIED LIBRARY
            self.registry._update_feature_library(ticker, feature_cols)
            
            # Step 4: Model Training
            print("Step 4/6: Model Training")
            print("-" * 70)
            model, history = self._train_model(
                ticker, X_train, y_train, X_test, y_test,
                epochs, batch_size
            )
            print(f"Training completed in {len(history.history['loss'])} epochs\n")
            
            # Step 5: Model Evaluation
            print("Step 5/6: Model Evaluation")
            print("-" * 70)
            metrics = self._evaluate_model(model, history, X_test, y_test, scaler, len(train_data))
            self._print_metrics(metrics)
            
            # Push metrics to Grafana/Prometheus
            try:
                from mlops_v2.monitoring import set_accuracy_20d
                set_accuracy_20d(ticker, float(metrics.get('directional_accuracy', 0)))
                print(f"📊 Pushed model accuracy to Grafana for {ticker}: {metrics.get('directional_accuracy', 0)}%")
            except Exception as e:
                print(f"⚠️ Failed to push metrics to Grafana: {e}")
            
            # Log metrics to MLflow if available
            if HAS_MLFLOW:
                try:
                    mlflow.log_metrics({
                        "price_rmse": metrics['rmse'],
                        "price_mae": metrics['mae'],
                        "r2_score": metrics['r2'],
                        "mape": metrics['mape'],
                        "directional_accuracy": metrics['directional_accuracy']
                    })
                except Exception:
                    pass
            
            # Skipping training plots to save storage space
            print()
            
            # Step 6: Save & Register (local registry — always works)
            model_info = self._save_and_register(
                ticker, model, scaler, metrics
            )

            try:
                if os.getenv('TRAIN_ULTIMATE_ENGINE', 'false').strip().lower() in ('1', 'true', 'yes', 'on'):
                    from ultimate_stock_engine_v36 import train_ultimate_model
                    print(f"Step 6b/6: Ultimate Engine v3.6 Training for {ticker}")
                    ultimate_result = train_ultimate_model(ticker, use_regime=True, generate_charts=True)
                    if ultimate_result:
                        model_info['ultimate_engine'] = {
                            'status': 'success',
                            'model_version': ultimate_result.get('model_version'),
                            'accuracy': ultimate_result.get('accuracy'),
                            'f1': ultimate_result.get('f1'),
                            'auc': ultimate_result.get('auc'),
                        }
            except Exception as ultimate_err:
                print(f"Ultimate Engine training skipped/failed for {ticker}: {ultimate_err}")
                model_info['ultimate_engine'] = {
                    'status': 'failed',
                    'error': str(ultimate_err)
                }
            
            # Optional: MLflow model registration
            if HAS_MLFLOW:
                try:
                    signature = infer_signature(X_train[:1], y_train[:1])
                    generic_scaler_path = os.path.join(self.artifacts_dir, 'scaler.pkl')
                    if os.path.exists(generic_scaler_path):
                        mlflow.log_artifact(generic_scaler_path, "preprocessing")
                    if os.path.exists("requirements.txt"):
                        mlflow.log_artifact("requirements.txt")

                    model_name = f"LSTM_{ticker}"
                    mlflow.tensorflow.log_model(model, "model", signature=signature)
                    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                    mv = mlflow.register_model(model_uri, model_name)
                    
                    # Champion vs Challenger logic
                    client = mlflow.tracking.MlflowClient()
                    is_prod_ready = True
                    try:
                        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
                        if prod_versions:
                            champion_run = client.get_run(prod_versions[0].run_id)
                            champion_mape = champion_run.data.metrics.get("mape", float('inf'))
                            if metrics['mape'] >= champion_mape:
                                is_prod_ready = False
                                client.transition_model_version_stage(model_name, mv.version, "Staging")
                    except Exception:
                        pass
                    
                    if is_prod_ready:
                        client.transition_model_version_stage(
                            name=model_name, version=mv.version,
                            stage="Production", archive_existing_versions=True
                        )
                    
                    print(f"MLflow: {model_name} v{mv.version} registered")
                except Exception as mlflow_err:
                    print(f"⚠️ MLflow registration skipped: {mlflow_err}")
            
            # Success summary
            print(f"{'='*70}")
            print(f"Ticker: {ticker}")
            print(f"Version: v{model_info['version']}")
            if HAS_MLFLOW:
                try:
                    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
                except Exception:
                    pass
            print(f"{'='*70}\n")
            
            return model_info
            
        except Exception as e:
            if HAS_MLFLOW:
                try:
                    mlflow.log_param("error", str(e))
                except Exception:
                    pass
            print(f"\n{'='*70}")
            print(f" TRAINING FAILED")
            print(f"{'='*70}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*70}\n")
            
            self._log_error(ticker, str(e))
            raise
        finally:
            if mlflow_ctx:
                try:
                    mlflow_ctx.__exit__(None, None, None)
                except Exception:
                    pass

    
    def _ingest_data(self, ticker: str, days: int = 730):
        """Ingest stock data from APIs with rich features"""
        data_ingestion = DataIngestion(ticker=ticker, days=days)
        raw_data = data_ingestion.fetch_data()
        
        if raw_data.empty:
            raise ValueError(f"No data fetched for {ticker}")
        
        return raw_data
    
    def _preprocess_data(self, raw_data):
        """Preprocess with rich feature engineering and split data"""
        data_ingestion = DataIngestion(ticker=self._current_ticker)
        processed_data = data_ingestion.preprocess(raw_data)
        
        # Feature columns are tracked via the training_pipeline for unified storage
        feature_cols = processed_data.columns.tolist()
        
        train_data, test_data = data_ingestion.split_data(processed_data)
        
        return train_data, test_data
    
    def _transform_data(self, train_data, test_data):
        """Transform data for LSTM model"""
        # Save feature column names for prediction use
        feature_cols = list(train_data.columns) if hasattr(train_data, 'columns') else []
        
        data_transformation = DataTransformation()
        X_train, y_train, X_test, y_test, scaler = data_transformation.transform(
            train_data, test_data
        )
        
        return X_train, y_train, X_test, y_test, scaler, feature_cols
    
    def _train_model(
        self, 
        ticker: str, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        epochs: int, 
        batch_size: int
    ):
        """Train LSTM model with callbacks"""
        # Create callbacks
        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        )
        
        tensorboard_log_dir = os.path.join(
            self.logs_dir,
            'tensorboard',
            f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=tensorboard_log_dir,
                histogram_freq=1
            )
        ]
        
        # Train model
        model_trainer = ModelTrainer()
        model, history = model_trainer.train_model(
            X_train, y_train, X_test, y_test,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return model, history
    
    def _evaluate_model(self, model, history, X_test, y_test, scaler, train_samples: int) -> Dict:
        """Evaluate model in REAL PRICE SPACE with production-grade metrics"""
        from sklearn.metrics import r2_score
        
        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history['val_loss'][-1])
        eval_result = model.evaluate(X_test, y_test, verbose=0)
        test_loss = float(eval_result[0]) if isinstance(eval_result, list) else float(eval_result)
        
        # Predict on scaled data
        predictions_scaled = model.predict(X_test, verbose=0).flatten()
        
        # --- Inverse-transform to real prices ---
        n_features = scaler.n_features_in_
        # Build dummy arrays with Close (col 0) filled and rest zeros
        pred_full = np.zeros((len(predictions_scaled), n_features))
        pred_full[:, 0] = predictions_scaled
        actual_full = np.zeros((len(y_test), n_features))
        actual_full[:, 0] = y_test
        
        predicted_prices = scaler.inverse_transform(pred_full)[:, 0]
        actual_prices = scaler.inverse_transform(actual_full)[:, 0]
        
        # --- Real price metrics ---
        price_mse = float(np.mean((predicted_prices - actual_prices) ** 2))
        price_rmse = float(np.sqrt(price_mse))
        price_mae = float(np.mean(np.abs(predicted_prices - actual_prices)))
        
        # R² score
        r2 = float(r2_score(actual_prices, predicted_prices))
        
        # MAPE (with division-by-zero guard)
        nonzero_mask = actual_prices != 0
        if nonzero_mask.sum() > 0:
            mape = float(np.mean(np.abs(
                (actual_prices[nonzero_mask] - predicted_prices[nonzero_mask]) / actual_prices[nonzero_mask]
            )) * 100)
        else:
            mape = float('inf')
        
        # Directional accuracy: did we predict up/down correctly?
        if len(actual_prices) > 1:
            actual_direction = np.diff(actual_prices) > 0
            predicted_direction = np.diff(predicted_prices) > 0
            directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100)
        else:
            directional_accuracy = 0.0
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'mse': price_mse,
            'rmse': price_rmse,
            'mae': price_mae,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'epochs_trained': len(history.history['loss']),
            'train_samples': train_samples,
            'test_samples': len(X_test)
        }
        
        # Print actual vs predicted sanity check (last 5 points)
        print("\n📊 Actual vs Predicted (last 5 test points):")
        for i in range(-min(5, len(actual_prices)), 0):
            actual = actual_prices[i]
            pred = predicted_prices[i]
            diff_pct = ((pred - actual) / actual) * 100 if actual != 0 else 0
            print(f"  ${actual:>8.2f}  →  ${pred:>8.2f}  ({diff_pct:+.2f}%)")
        print()
        
        return metrics
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics in formatted output"""
        print(f"Train Loss:           {metrics['train_loss']:.6f}")
        print(f"Validation Loss:      {metrics['val_loss']:.6f}")
        print(f"Test Loss:            {metrics['test_loss']:.6f}")
        print(f"─── Real Price Metrics ───")
        print(f"RMSE:                 ${metrics['rmse']:.2f}")
        print(f"MAE:                  ${metrics['mae']:.2f}")
        print(f"R²:                   {metrics['r2']:.4f}")
        print(f"MAPE:                 {metrics['mape']:.2f}%")
        print(f"Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
        print(f"Epochs:               {metrics['epochs_trained']}")
    
    def _save_and_register(self, ticker: str, model, scaler, metrics: Dict) -> Dict:
        """Save model artifacts and register in MLOps registry relying on Unified Storage"""
        # 1. Save model locally for registry ingestion
        temp_model_path = os.path.join(self.artifacts_dir, f'{ticker}_temp_model.keras')
        model.save(temp_model_path)
        
        # 2. Register in MLOps registry (Handles Unified Scaler Library & model_store)
        model_info = self.registry.register_model(
            ticker=ticker,
            model_path=temp_model_path,
            metrics=metrics,
            scaler_data=scaler, # Pass actual object for library consolidation
            metadata={
                'framework': 'TensorFlow/Keras',
                'model_type': 'LSTM',
                'trained_at': datetime.now().isoformat()
            }
        )
        
        # 3. Clean up temp file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        
        # 4. Save generic artifacts for app.py compatibility (Simple overwrite)
        model.save(os.path.join(self.artifacts_dir, 'stock_lstm_model.keras'))
        generic_scaler_path = os.path.join(self.artifacts_dir, 'scaler.pkl')
        with open(generic_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return model_info
    
    def _log_error(self, ticker: str, error: str):
        """Log training errors to file"""
        error_log = os.path.join(self.logs_dir, 'training_errors.log')
        
        with open(error_log, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {ticker}: {error}\n")
    
    def batch_train(self, tickers: list, **kwargs):
        """
        Train models for multiple tickers in batch
        
        Args:
            tickers: List of stock ticker symbols
            **kwargs: Additional arguments passed to train_model()
        
        Returns:
            Dictionary mapping tickers to their model info
        """
        results = {}
        failed = []
        
        print(f"\n{'#'*70}")
        print(f"BATCH TRAINING: {len(tickers)} stocks")
        print(f"{'#'*70}\n")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Training {ticker}...")
            
            try:
                model_info = self.train_model(ticker, **kwargs)
                results[ticker] = model_info
            except Exception as e:
                print(f" Failed: {ticker} - {str(e)}")
                failed.append(ticker)
        
        # Summary
        print(f"\n{'#'*70}")
        print(f"BATCH TRAINING SUMMARY")
        print(f"{'#'*70}")
        print(f" Successful: {len(results)}/{len(tickers)}")
        if failed:
            print(f" Failed: {', '.join(failed)}")
        print(f"{'#'*70}\n")
        
        return results


if __name__ == "__main__":
    import argparse
    from mlops.config import MLOpsConfig
    
    parser = argparse.ArgumentParser(description='Run Professional MLOps Training Pipeline')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    pipeline = MLOpsTrainingPipeline()
    pipeline.train_model(
        ticker=args.ticker,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

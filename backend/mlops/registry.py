"""
Model Registry - Version control and management for ML models
Tracks model versions, performance metrics, and deployment status
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any


class ModelRegistry:
    """
    Manages ML model versions, metadata, and lifecycle
    """
    
    def __init__(self, registry_path: str = 'mlops/model_registry'):
        """
        Initialize the model registry with MLflow integration
        
        Args:
            registry_path: Directory to store model registry data
        """
        import mlflow
        from mlops.config import MLOpsConfig
        
        self.registry_path = registry_path
        self.metadata_file = os.path.join(registry_path, 'metadata.json')
        self.scaler_library_file = os.path.join(registry_path, 'global_scalers.pkl')
        self.feature_library_file = os.path.join(registry_path, 'global_features.json')
        self.model_store_path = os.path.join(registry_path, 'model_store')
        
        # Configure MLflow
        mlflow.set_tracking_uri(MLOpsConfig.MLFLOW_TRACKING_URI)
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
        # Create registry & store directories
        os.makedirs(registry_path, exist_ok=True)
        os.makedirs(self.model_store_path, exist_ok=True)
        
        # Load or initialize metadata
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize or load existing metadata"""
        if os.path.exists(self.metadata_file):
            self._load_metadata()
        else:
            self.metadata = {
                'models': [],
                'registry_version': '1.0.0',
                'created_at': datetime.now().isoformat()
            }
            self._save_metadata()
    
    def _load_metadata(self):
        """Load metadata from JSON file"""
        try:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self._initialize_metadata()
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=4)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def register_model(
        self, 
        ticker: str, 
        model_path: str, 
        metrics: Dict, 
        scaler_data: Optional[Any] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Register a new trained model using the Unified Storage System
        """
        # Calculate version number
        existing_models = [m for m in self.metadata['models'] if m['ticker'] == ticker]
        version = len(existing_models) + 1
        
        # Create version-specific directory in CENTRAL STORE
        version_id = f"{ticker}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = os.path.join(self.model_store_path, version_id)
        os.makedirs(version_dir, exist_ok=True)
        
        # 1. Save model file
        model_filename = os.path.join(version_dir, 'model.h5')
        shutil.copy(model_path, model_filename)
        
        # 2. Add scaler to UNIFIED LIBRARY (Zero-Clutter)
        if scaler_data:
            self._update_scaler_library(ticker, scaler_data)
        
        # Create model information
        model_info = {
            'ticker': ticker,
            'version': version,
            'version_id': version_id,
            'registered_at': datetime.now().isoformat(),
            'model_path': model_filename,
            'scaler_type': 'unified_library',
            'metrics': metrics,
            'status': 'active',
            'metadata': metadata or {}
        }
        
        # Add to registry
        self.metadata['models'].append(model_info)
        self._save_metadata()
        
        # --- NEW: Prune old versions to save space ---
        self._prune_versions(ticker, keep_limit=3)
        
        print(f"Model registered (Unified): {ticker} v{version}")
        return model_info

    def _update_feature_library(self, ticker: str, feature_list: List[str]):
        """Save a stock's feature column list into the single global JSON file"""
        library = {}
        if os.path.exists(self.feature_library_file):
            try:
                with open(self.feature_library_file, 'r') as f:
                    library = json.load(f)
            except Exception:
                library = {}
        
        library[ticker] = feature_list
        
        with open(self.feature_library_file, 'w') as f:
            json.dump(library, f, indent=4)
        print(f"📋 Unified Storage: Feature list for {ticker} added to library.")

    def get_features_from_library(self, ticker: str) -> Optional[List[str]]:
        """Fetch a stock's feature list from the single global JSON file"""
        if not os.path.exists(self.feature_library_file):
            return None
            
        try:
            with open(self.feature_library_file, 'r') as f:
                library = json.load(f)
            return library.get(ticker)
        except Exception:
            return None

    def _update_scaler_library(self, ticker: str, scaler_data: Any):
        """Save a stock's scaler into the single global library file"""
        import pickle
        library = {}
        if os.path.exists(self.scaler_library_file):
            try:
                with open(self.scaler_library_file, 'rb') as f:
                    library = pickle.load(f)
            except Exception:
                library = {}
        
        library[ticker] = scaler_data
        
        with open(self.scaler_library_file, 'wb') as f:
            pickle.dump(library, f)
        print(f"💾 Unified Storage: Scaler for {ticker} added to library.")

    def get_scaler_from_library(self, ticker: str) -> Optional[Any]:
        """Fetch a stock's scaler from the single global library file"""
        import pickle
        if not os.path.exists(self.scaler_library_file):
            return None
            
        try:
            with open(self.scaler_library_file, 'rb') as f:
                library = pickle.load(f)
            return library.get(ticker)
        except Exception:
            return None

    def _prune_versions(self, ticker: str, keep_limit: int = 3):
        """
        Delete old local model versions to save disk space.
        Keeps the latest N versions and removes the rest.
        """
        try:
            # Get all versions for this ticker, sorted by version number descending
            ticker_models = sorted(
                [m for m in self.metadata['models'] if m['ticker'] == ticker],
                key=lambda x: x['version'],
                reverse=True
            )
            
            if len(ticker_models) <= keep_limit:
                return
                
            # Versions to remove
            to_remove = ticker_models[keep_limit:]
            removed_ids = [m['version_id'] for m in to_remove]
            
            # 1. Delete physical folders
            for model in to_remove:
                version_dir = os.path.dirname(model['model_path'])
                if os.path.exists(version_dir):
                    try:
                        shutil.rmtree(version_dir)
                        print(f"🧹 Pruning: Removed old storage for {model['version_id']}")
                    except Exception as e:
                        print(f"⚠️ Failed to delete pruned directory {version_dir}: {e}")
            
            # 2. Update metadata to remove these entries
            self.metadata['models'] = [
                m for m in self.metadata['models'] 
                if m['version_id'] not in removed_ids
            ]
            self._save_metadata()
            print(f"✅ Pruned {len(removed_ids)} old versions for {ticker}.")
            
        except Exception as e:
            print(f"⚠️ Error during model pruning for {ticker}: {e}")
    
    def get_best_model(self, ticker: str) -> Optional[Dict]:
        """
        Get the best model for a specific ticker.
        Prioritizes MLflow Model Registry (Production stage).
        Falls back to the LATEST local version.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Model information dictionary or None if no models found
        """
        # 1. Try MLflow first
        mlflow_model = self.get_mlflow_model(ticker, stage="Production")
        if mlflow_model:
            print(f"Using MLflow Production model for {ticker}")
            return mlflow_model
            
        # 2. Fallback to local
        active_models = [
            m for m in self.metadata['models'] 
            if m['ticker'] == ticker and m['status'] == 'active'
        ]
        
        if not active_models:
            return None
        
        # Return the LATEST version — architecture changes between versions
        # make val_loss comparison unreliable (different feature counts)
        best_model = max(
            active_models, 
            key=lambda x: x['version']
        )
        
        return best_model
    
    def get_latest_model(self, ticker: str) -> Optional[Dict]:
        """
        Get the most recently trained model for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Model information dictionary or None if no models found
        """
        active_models = [
            m for m in self.metadata['models'] 
            if m['ticker'] == ticker and m['status'] == 'active'
        ]
        
        if not active_models:
            return None
        
        return max(active_models, key=lambda x: x['version'])

    def get_mlflow_model(self, ticker: str, stage: str = "Production") -> Optional[Dict]:
        """
        Fetch model info from MLflow Model Registry by stage
        
        Args:
            ticker: Stock ticker symbol
            stage: Model stage ('None', 'Staging', 'Production', 'Archived')
        
        Returns:
            Dictionary with local artifact paths downloaded from MLflow
        """
        try:
            model_name = f"LSTM_{ticker}"
            versions = self.mlflow_client.get_latest_versions(model_name, stages=[stage])
            
            if not versions:
                return None
                
            latest_version = versions[0]
            run_id = latest_version.run_id
            
            # Download artifacts locally if not already there
            artifact_local_path = os.path.join(self.registry_path, f"mlflow_{ticker}_{stage}")
            os.makedirs(artifact_local_path, exist_ok=True)
            
            # Download scaler and model
            self.mlflow_client.download_artifacts(run_id, f"{ticker}_scaler.pkl", artifact_local_path)
            self.mlflow_client.download_artifacts(run_id, "model/data/model.h5", artifact_local_path)
            
            return {
                'ticker': ticker,
                'version': latest_version.version,
                'model_path': os.path.join(artifact_local_path, "model", "data", "model.h5"),
                'scaler_path': os.path.join(artifact_local_path, f"{ticker}_scaler.pkl"),
                'source': 'mlflow'
            }
        except Exception as e:
            print(f"MLflow model fetch failed for {ticker}: {e}")
            return None
    
    def list_models(self, ticker: Optional[str] = None, status: str = 'active') -> List[Dict]:
        """
        List all registered models
        
        Args:
            ticker: Filter by ticker symbol (optional)
            status: Filter by status ('active', 'archived', 'deprecated')
        
        Returns:
            List of model information dictionaries
        """
        models = self.metadata['models']
        
        if ticker:
            models = [m for m in models if m['ticker'] == ticker]
        
        if status:
            models = [m for m in models if m['status'] == status]
        
        return models
    
    def archive_model(self, version_id: str):
        """
        Archive a model (mark as inactive but keep data)
        
        Args:
            version_id: Unique version identifier
        """
        for model in self.metadata['models']:
            if model['version_id'] == version_id:
                model['status'] = 'archived'
                model['archived_at'] = datetime.now().isoformat()
                self._save_metadata()
                print(f"Model archived: {version_id}")
                return
        
        print(f"Model not found: {version_id}")
    
    def get_model_stats(self, ticker: str) -> Dict:
        """
        Get statistics for all models of a specific ticker
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary containing model statistics
        """
        models = [m for m in self.metadata['models'] if m['ticker'] == ticker]
        
        if not models:
            return {'ticker': ticker, 'total_versions': 0}
        
        active_models = [m for m in models if m['status'] == 'active']
        
        val_losses = [m['metrics'].get('val_loss', float('inf')) for m in active_models]
        
        return {
            'ticker': ticker,
            'total_versions': len(models),
            'active_versions': len(active_models),
            'best_val_loss': min(val_losses) if val_losses else None,
            'avg_val_loss': sum(val_losses) / len(val_losses) if val_losses else None,
            'latest_version': max([m['version'] for m in models])
        }


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    
    # Print all registered models
    models = registry.list_models()
    print(f"\n Total Models: {len(models)}")
    
    for model in models:
        print(f"\n{model['ticker']} v{model['version']}")
        print(f"  Status: {model['status']}")
        print(f"  Val Loss: {model['metrics'].get('val_loss', 'N/A')}")

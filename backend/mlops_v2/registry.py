from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib

from mlops_v2.settings import SETTINGS


@dataclass
class ModelPaths:
    xgb_model: Path
    lstm_model: Path
    scaler: Path
    metadata: Path


def get_model_paths(ticker: str) -> ModelPaths:
    ticker_dir = SETTINGS.model_dir / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ModelPaths(
        xgb_model=ticker_dir / "xgb.joblib",
        lstm_model=ticker_dir / "lstm.keras",
        scaler=ticker_dir / "scaler.joblib",
        metadata=ticker_dir / "metadata.json",
    )


def log_mlflow_run(
    ticker: str,
    xgb_model,
    lstm_model,
    metrics: Dict,
    drift_score: float,
    data_points: int,
) -> Optional[str]:
    try:
        import mlflow
        from mlops.config import MLOpsConfig

        mlflow.set_tracking_uri(MLOpsConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLOpsConfig.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"{ticker}_v2") as run:
            mlflow.set_tags({"ticker": ticker, "pipeline": "mlops_v2"})
            mlflow.log_metrics({
                "xgb_accuracy": float(metrics.get("xgb_accuracy", 0.0)),
                "lstm_val_loss": float(metrics.get("lstm_val_loss", 0.0)),
                "drift_score": float(drift_score),
                "data_points": float(data_points),
            })

            model_paths = get_model_paths(ticker)
            joblib.dump(xgb_model, model_paths.xgb_model)
            xgb_artifact_name = f"{ticker}_xgb_model"
            mlflow.log_artifact(str(model_paths.xgb_model), artifact_path=xgb_artifact_name)

            lstm_model.save(model_paths.lstm_model)
            mlflow.log_artifact(str(model_paths.lstm_model), artifact_path=f"{ticker}_lstm_model")

            # Best effort: if this is a TensorFlow model, also log as MLflow TF flavor.
            try:
                if hasattr(lstm_model, "to_json"):
                    import mlflow.tensorflow as mlflow_tf
                    mlflow_tf.log_model(lstm_model, artifact_path=f"{ticker}_lstm_model_tf")
            except Exception:
                pass

            _register_model_with_promotion(mlflow, run.info.run_id, ticker, metrics)
            return run.info.run_id

    except Exception:
        return None


def _register_model_with_promotion(mlflow_module, run_id: str, ticker: str, metrics: Dict) -> None:
    """Champion-challenger style auto-promotion rule."""
    client = mlflow_module.tracking.MlflowClient()

    model_pairs = [
        (f"{ticker}_xgb", f"runs:/{run_id}/{ticker}_xgb_model", metrics.get("xgb_accuracy", 0.0)),
        (f"{ticker}_lstm", f"runs:/{run_id}/{ticker}_lstm_model", 1.0 - float(metrics.get("lstm_val_loss", 1.0))),
    ]

    for name, uri, score in model_pairs:
        mv = mlflow_module.register_model(uri, name)

        try:
            prod_versions = client.get_latest_versions(name, stages=["Production"])
            if prod_versions:
                champion_run = client.get_run(prod_versions[0].run_id)
                champion_score = champion_run.data.metrics.get("xgb_accuracy", 0.0)
                should_promote = float(score) >= float(champion_score) * 0.98
            else:
                should_promote = True
        except Exception:
            should_promote = True

        target_stage = "Production" if should_promote else "Staging"
        client.transition_model_version_stage(name=name, version=mv.version, stage=target_stage, archive_existing_versions=should_promote)

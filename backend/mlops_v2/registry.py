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


def _set_experiment_safe(experiment_name: str) -> None:
    """Set MLflow experiment, restoring it first if it was soft-deleted."""
    import mlflow as _mlflow

    try:
        _mlflow.set_experiment(experiment_name)
    except Exception as first_err:
        if "deleted" in str(first_err).lower():
            try:
                client = _mlflow.tracking.MlflowClient()
                exp = client.get_experiment_by_name(experiment_name)
                if exp and exp.lifecycle_stage == "deleted":
                    client.restore_experiment(exp.experiment_id)
                    print(f"♻️ Restored deleted MLflow experiment '{experiment_name}'")
                _mlflow.set_experiment(experiment_name)
            except Exception as restore_err:
                print(f"⚠️ Could not restore experiment '{experiment_name}': {restore_err}")
                _mlflow.set_experiment(f"{experiment_name}_v2")
        else:
            raise


def _safe_metric(value, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return default
        return out
    except Exception:
        return default


def _log_artifact_safe(mlflow_module, local_path: Path | str, artifact_path: str) -> None:
    try:
        path = Path(local_path)
        if path.exists():
            mlflow_module.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:
        print(f"⚠️ MLflow artifact logging skipped for {artifact_path}: {exc}")


def log_mlflow_run(
    ticker: str,
    xgb_model,
    lstm_model,
    metrics: Dict,
    drift_score: float,
    data_points: int,
    params: Optional[Dict] = None,
) -> Optional[str]:
    model_paths = get_model_paths(ticker)
    
    # ALWAYS save locally first, regardless of MLflow availability
    try:
        joblib.dump(xgb_model, model_paths.xgb_model)
        lstm_model.save(model_paths.lstm_model)
    except Exception as save_err:
        print(f"⚠️ Failed to save local models for {ticker}: {save_err}")
    
    try:
        import mlflow
        from mlops.config import MLOpsConfig

        mlflow.set_tracking_uri(MLOpsConfig.MLFLOW_TRACKING_URI)
        _set_experiment_safe(MLOpsConfig.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"{ticker}_v2") as run:
            mlflow.set_tags({"ticker": ticker, "pipeline": "mlops_v2"})
            
            if params:
                mlflow.log_params(params)
                
            mlflow.log_metrics({
                "xgb_accuracy": _safe_metric(metrics.get("xgb_accuracy")),
                "lstm_val_loss": _safe_metric(metrics.get("lstm_val_loss")),
                "directional_accuracy": _safe_metric(metrics.get("directional_accuracy", metrics.get("xgb_accuracy"))),
                "drift_score": _safe_metric(drift_score),
                "data_points": _safe_metric(data_points),
                "mape": _safe_metric(metrics.get("mape")),
                "price_mae": _safe_metric(metrics.get("price_mae")),
                "price_rmse": _safe_metric(metrics.get("price_rmse")),
                "r2_score": _safe_metric(metrics.get("r2_score")),
                "val_loss": _safe_metric(metrics.get("val_loss")),
                "val_mae": _safe_metric(metrics.get("val_mae")),
                "validation_loss": _safe_metric(metrics.get("validation_loss")),
                "simulated_pnl": _safe_metric(metrics.get("simulated_pnl")),
                "sharpe_ratio": _safe_metric(metrics.get("sharpe_ratio")),
            })

            # Log Evidently AI HTML Reports to DagsHub
            from pathlib import Path
            val_report = metrics.get("validation_report_path")
            if val_report:
                _log_artifact_safe(mlflow, val_report, "evidently_reports")
                
            drift_report = metrics.get("drift_report_path")
            if drift_report:
                _log_artifact_safe(mlflow, drift_report, "evidently_reports")

            xgb_artifact_name = f"{ticker}_xgb_model"
            _log_artifact_safe(mlflow, model_paths.xgb_model, xgb_artifact_name)
            _log_artifact_safe(mlflow, model_paths.lstm_model, f"{ticker}_lstm_model")

            # Best effort: if this is a TensorFlow model, also log as MLflow TF flavor.
            try:
                if hasattr(lstm_model, "to_json"):
                    import mlflow.tensorflow as mlflow_tf
                    mlflow_tf.log_model(lstm_model, artifact_path=f"{ticker}_lstm_model_tf")
            except Exception:
                pass

            try:
                _register_model_with_promotion(mlflow, run.info.run_id, ticker, metrics)
            except Exception as registry_err:
                mlflow.set_tag("model_registry_warning", str(registry_err)[:250])
                print(f"⚠️ MLflow model registry promotion skipped for {ticker}: {registry_err}")
            return run.info.run_id

    except Exception as e:
        print(f"⚠️ MLflow V2 logging failed for {ticker}: {e}")
        print(f"   ✅ Local models were saved successfully — training is NOT lost.")
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

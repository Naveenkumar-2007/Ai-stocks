from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from mlops_v2.data_ingestion import DataIngestionService
from mlops_v2.drift import compute_drift
from mlops_v2.feature_engineering import FEATURE_COLUMNS, FeatureEngineer
from mlops_v2.monitoring import set_drift_score
from mlops_v2.registry import get_model_paths, log_mlflow_run
from mlops_v2.settings import SETTINGS
from mlops_v2.validation import DataValidator


@dataclass
class TrainResult:
    ticker: str
    trained: bool
    drift_score: float
    metrics: Dict
    reason: str
    run_id: Optional[str] = None


class _SklearnSequenceRegressor:
    """Fallback wrapper with Keras-like save/predict methods."""

    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def save(self, path) -> None:
        joblib.dump(self.estimator, path)

    def predict(self, X, verbose: int = 0):
        _ = verbose
        arr = np.asarray(X, dtype=np.float32)
        arr = arr.reshape((arr.shape[0], -1))
        pred = self.estimator.predict(arr)
        return np.asarray(pred, dtype=np.float32).reshape(-1, 1)


class TrainerV2:
    def __init__(self) -> None:
        self.ingestor = DataIngestionService()
        self.validator = DataValidator()
        self.fe = FeatureEngineer()

    def train_if_needed(self, ticker: str, force: bool = False) -> TrainResult:
        ticker = ticker.strip().upper()

        raw = self.ingestor.fetch_with_retries(ticker=ticker, days=200)
        validation = self.validator.validate(raw, ticker=ticker)
        if not validation.ok:
            print(
                f"[VALIDATION] {ticker} failed with {len(validation.errors)} error(s). "
                f"Report: {validation.report_path}"
            )
            return TrainResult(
                ticker=ticker,
                trained=False,
                drift_score=0.0,
                metrics={
                    "validation_report_path": validation.report_path,
                    "validation_checks_total": int(validation.checks_total),
                    "validation_checks_passed": int(validation.checks_passed),
                },
                reason=f"validation_failed: {validation.errors}",
            )

        raw_path = self.ingestor.persist_raw_parquet(ticker, raw)
        self.ingestor.persist_dvc_manifest(ticker, raw_path)

        artifacts = self.fe.compute_features(raw)
        features = artifacts.features
        if len(features) < SETTINGS.min_rows:
            return TrainResult(ticker=ticker, trained=False, drift_score=0.0, metrics={}, reason="not_enough_feature_rows")

        ref = features.iloc[:-SETTINGS.min_rows]
        cur = features.iloc[-SETTINGS.min_rows:]
        if ref.empty:
            drift_score = 1.0
        else:
            drift_result = compute_drift(ref, cur, report_name=f"{ticker}_drift")
            drift_score = drift_result.drift_score

        set_drift_score(ticker, drift_score)

        if not force and not self._should_train(ticker, drift_score):
            return TrainResult(ticker=ticker, trained=False, drift_score=drift_score, metrics={}, reason="drift_below_threshold_and_recently_trained")

        direction_target, magnitude_target = self.fe.build_targets(raw, features, horizon_days=SETTINGS.prediction_horizon_days)
        common_index = direction_target.index.intersection(magnitude_target.index)

        X = features.loc[common_index, FEATURE_COLUMNS]
        y_dir = direction_target.loc[common_index]
        y_mag = magnitude_target.loc[common_index]

        xgb_model, xgb_acc = self._train_xgb(X, y_dir)
        lstm_model, lstm_metrics = self._train_lstm(X, y_mag)

        if np.isnan(xgb_acc) or (xgb_acc < SETTINGS.min_directional_accuracy and not force):
            return TrainResult(ticker=ticker, trained=False, drift_score=drift_score, metrics={"xgb_accuracy": xgb_acc}, reason="xgb_accuracy_below_minimum")

        model_paths = get_model_paths(ticker)
        model_paths.scaler.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(artifacts.scaler, model_paths.scaler)

        try:
            val_size = min(20, len(X))
            X_val_pnl = X.iloc[-val_size:]
            pred_probs = xgb_model.predict_proba(X_val_pnl)[:, 1]
            pred_dirs = (pred_probs >= 0.5).astype(int)
            future_close = raw['Close'].shift(-SETTINGS.prediction_horizon_days)
            actual_returns = (future_close - raw['Close']) / raw['Close']
            val_returns = actual_returns.reindex(X_val_pnl.index).fillna(0)
            trade_returns = np.where(pred_dirs == 1, val_returns, -val_returns)
            simulated_pnl = float(np.sum(trade_returns) * 100)
            sharpe_ratio = float(np.mean(trade_returns) / (np.std(trade_returns) + 1e-9) * np.sqrt(252))
        except Exception as e:
            print(f"PnL Calculation failed: {e}")
            simulated_pnl = 0.0
            sharpe_ratio = 0.0

        metrics = {
            "xgb_accuracy": float(xgb_acc),
            "lstm_val_loss": float(lstm_metrics.get("val_loss", 1.0)),
            "data_points": int(len(X)),
            "validation_report_path": validation.report_path,
            "validation_checks_total": int(validation.checks_total),
            "validation_checks_passed": int(validation.checks_passed),
            "mape": float(lstm_metrics.get("mape", 0.0)),
            "price_mae": float(lstm_metrics.get("mae", 0.0)),
            "price_rmse": float(lstm_metrics.get("rmse", 0.0)),
            "directional_accuracy": float(xgb_acc),
            "r2_score": float(lstm_metrics.get("r2", 0.0)),
            "val_loss": float(lstm_metrics.get("val_loss", 0.0)),
            "val_mae": float(lstm_metrics.get("mae", 0.0)),
            "validation_loss": float(lstm_metrics.get("val_loss", 0.0)),
            "simulated_pnl": simulated_pnl,
            "sharpe_ratio": sharpe_ratio,
        }
        
        # Add training parameters that the user expects in DagsHub
        params = {
            "baseline": "None",
            "batch_size": 32,
            "class_weight": "None",
            "days": SETTINGS.prediction_horizon_days,
            "epochs": SETTINGS.lstm_max_epochs,
            "initial_epoch": 0,
            "min_delta": 0,
            "lstm_sequence_days": SETTINGS.lstm_sequence_days
        }
        
        run_id = log_mlflow_run(ticker, xgb_model, lstm_model, metrics=metrics, params=params, drift_score=drift_score, data_points=len(X))

        metadata = {
            "ticker": ticker,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "drift_score": drift_score,
            "metrics": metrics,
            "run_id": run_id,
            "feature_columns": FEATURE_COLUMNS,
        }
        model_paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return TrainResult(ticker=ticker, trained=True, drift_score=drift_score, metrics=metrics, reason="trained", run_id=run_id)

    def _should_train(self, ticker: str, drift_score: float) -> bool:
        if drift_score > SETTINGS.drift_threshold:
            return True

        metadata_path = get_model_paths(ticker).metadata
        if not metadata_path.exists():
            return True

        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            last = datetime.fromisoformat(payload["trained_at"].replace("Z", ""))
        except Exception:
            return True

        return datetime.utcnow() - last > timedelta(days=7)

    def _train_xgb(self, X: pd.DataFrame, y: pd.Series):
        from xgboost import XGBClassifier

        model = XGBClassifier(**SETTINGS.xgb_params)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            pred = (model.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
            scores.append(accuracy_score(y_val, pred))

        model.fit(X, y)
        return model, float(np.mean(scores))

    def _train_lstm(self, X: pd.DataFrame, y: pd.Series):
        import os

        seq_len = SETTINGS.lstm_sequence_days
        feat_count = min(SETTINGS.lstm_feature_count, X.shape[1])
        X_seq = X.iloc[:, :feat_count].to_numpy(dtype=np.float32, copy=True)
        y_seq = y.to_numpy(dtype=np.float32, copy=True)

        X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
        y_seq = np.nan_to_num(y_seq, nan=0.0, posinf=0.0, neginf=0.0)

        xs = []
        ys = []
        for i in range(seq_len, len(X_seq)):
            xs.append(X_seq[i - seq_len:i])
            ys.append(y_seq[i])

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)

        if len(xs) < 20:
            # Not enough rows for LSTM, return dummy high loss.
            class Dummy:
                def save(self, *_args, **_kwargs):
                    pass

            return Dummy(), 1.0

        split = int(len(xs) * 0.8)
        X_train, X_val = xs[:split], xs[split:]
        y_train, y_val = ys[:split], ys[split:]

        # Windows fallback: avoid native TF crashes while preserving sequence-learning behavior.
        if os.name == "nt":
            from sklearn.ensemble import HistGradientBoostingRegressor

            X_train_flat = X_train.reshape((X_train.shape[0], -1))
            X_val_flat = X_val.reshape((X_val.shape[0], -1))

            reg = HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                random_state=42,
            )
            reg.fit(X_train_flat, y_train)
            pred_val = reg.predict(X_val_flat).reshape(-1)
            
            from sklearn.metrics import mean_absolute_error, r2_score
            rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
            mae = float(mean_absolute_error(y_val, pred_val))
            r2 = float(r2_score(y_val, pred_val))
            
            nonzero_mask = y_val != 0
            if np.any(nonzero_mask):
                mape = float(np.mean(np.abs((y_val[nonzero_mask] - pred_val[nonzero_mask]) / y_val[nonzero_mask])) * 100)
            else:
                mape = 0.0
                
            metrics_dict = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "val_loss": rmse}
            return _SklearnSequenceRegressor(reg), metrics_dict

        # oneDNN can be unstable on some Windows CPU stacks for small recurrent models.
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(seq_len, feat_count)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss=tf.keras.losses.Huber())

        cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=SETTINGS.lstm_early_stop_patience, restore_best_weights=True)
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=SETTINGS.lstm_max_epochs,
            verbose=0,
            callbacks=[cb],
        )

        pred_val = model.predict(X_val, verbose=0).reshape(-1)
        
        from sklearn.metrics import mean_absolute_error, r2_score
        rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
        mae = float(mean_absolute_error(y_val, pred_val))
        r2 = float(r2_score(y_val, pred_val))
        
        nonzero_mask = y_val != 0
        if np.any(nonzero_mask):
            mape = float(np.mean(np.abs((y_val[nonzero_mask] - pred_val[nonzero_mask]) / y_val[nonzero_mask])) * 100)
        else:
            mape = 0.0
            
        val_loss = float(hist.history.get("val_loss", [rmse])[-1])
        metrics_dict = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "val_loss": val_loss}
        return model, metrics_dict

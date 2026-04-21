from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import joblib
import numpy as np

from mlops_v2.data_ingestion import DataIngestionService
from mlops_v2.feature_engineering import FEATURE_COLUMNS, FeatureEngineer
from mlops_v2.monitoring import inc_prediction, latency_timer, observe_latency
from mlops_v2.registry import get_model_paths
from mlops_v2.settings import SETTINGS


@dataclass
class InferencePayload:
    ticker: str
    prediction: float
    lower_95: float
    upper_95: float
    confidence: float
    direction_prob: float
    model_version: str
    features_used: List[str]
    data_freshness: str
    drift_score: float

    def as_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "prediction": self.prediction,
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
            "confidence": self.confidence,
            "direction_prob": self.direction_prob,
            "model_version": self.model_version,
            "features_used": self.features_used,
            "data_freshness": self.data_freshness,
            "drift_score": self.drift_score,
        }


class InferenceServiceV2:
    def __init__(self) -> None:
        self.ingestor = DataIngestionService()
        self.fe = FeatureEngineer()

    def predict(self, ticker: str) -> Dict:
        ticker = ticker.strip().upper()

        with latency_timer() as elapsed:
            raw = self.ingestor.fetch_with_retries(ticker=ticker, days=200)
            artifacts = self.fe.compute_features(raw)
            features = artifacts.features

            if features.empty:
                raise RuntimeError(f"No features computed for {ticker}")

            model_paths = get_model_paths(ticker)
            xgb = joblib.load(model_paths.xgb_model)
            scaler = joblib.load(model_paths.scaler)

            lstm = None
            lstm_is_tf = False
            if os.name == "nt":
                # Windows-safe path: skip TF import and use fallback model artifact.
                lstm = joblib.load(model_paths.lstm_model)
                lstm_is_tf = False
            else:
                try:
                    import tensorflow as tf

                    lstm = tf.keras.models.load_model(model_paths.lstm_model)
                    lstm_is_tf = True
                except Exception:
                    lstm = joblib.load(model_paths.lstm_model)
                    lstm_is_tf = False

            X_last = features.iloc[[-1]][FEATURE_COLUMNS]
            direction_prob = float(xgb.predict_proba(X_last)[0, 1])
            xgb_direction = 1.0 if direction_prob >= 0.5 else -1.0

            seq_len = SETTINGS.lstm_sequence_days
            feat_count = min(SETTINGS.lstm_feature_count, X_last.shape[1])
            X_seq = features.iloc[:, :feat_count].tail(seq_len).values
            if len(X_seq) < seq_len:
                pad = np.repeat(X_seq[:1], seq_len - len(X_seq), axis=0)
                X_seq = np.vstack([pad, X_seq])
            X_seq = np.expand_dims(X_seq, axis=0)
            if lstm_is_tf:
                lstm_mag = float(lstm.predict(X_seq, verbose=0)[0][0])
            else:
                lstm_mag = float(lstm.predict(X_seq.reshape((1, -1)))[0])

            prediction = SETTINGS.ensemble_direction_weight * xgb_direction + SETTINGS.ensemble_magnitude_weight * lstm_mag
            uncertainty = abs(xgb_direction - lstm_mag)
            lower_95 = prediction - 1.96 * uncertainty
            upper_95 = prediction + 1.96 * uncertainty
            confidence = max(0.0, min(1.0, 1.0 - uncertainty))

            metadata = {}
            if model_paths.metadata.exists():
                metadata = json.loads(model_paths.metadata.read_text(encoding="utf-8"))

            payload = InferencePayload(
                ticker=ticker,
                prediction=float(prediction),
                lower_95=float(lower_95),
                upper_95=float(upper_95),
                confidence=float(confidence),
                direction_prob=float(direction_prob),
                model_version=f"xgb_{metadata.get('run_id', 'na')}_lstm_{metadata.get('run_id', 'na')}",
                features_used=FEATURE_COLUMNS,
                data_freshness=datetime.utcnow().isoformat() + "Z",
                drift_score=float(metadata.get("drift_score", 0.0)),
            )

            inc_prediction(ticker)
            observe_latency(elapsed())
            return payload.as_dict()

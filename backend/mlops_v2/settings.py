from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class MLOpsV2Settings:
    base_dir: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    raw_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "raw")
    reports_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "monitoring" / "reports")
    model_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "mlops_v2" / "models")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "cache")

    tickers_file: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "mlops" / "stocks.json")

    prediction_horizon_days: int = 5
    min_rows: int = 50
    max_rows: int = 500
    min_history_days: int = 100

    drift_threshold: float = 0.25
    min_directional_accuracy: float = 0.52
    target_directional_accuracy: float = 0.60

    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
    })

    lstm_sequence_days: int = 60
    lstm_feature_count: int = 8
    lstm_max_epochs: int = 50
    lstm_early_stop_patience: int = 5

    ensemble_direction_weight: float = 0.7
    ensemble_magnitude_weight: float = 0.3

    def ensure_dirs(self) -> None:
        for p in [self.data_dir, self.raw_dir, self.reports_dir, self.model_dir, self.cache_dir]:
            p.mkdir(parents=True, exist_ok=True)


SETTINGS = MLOpsV2Settings()
SETTINGS.ensure_dirs()


def load_tickers() -> List[str]:
    import json
    import re

    if not SETTINGS.tickers_file.exists():
        return []
    tickers = json.loads(SETTINGS.tickers_file.read_text(encoding="utf-8"))
    pattern = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,19}$")
    seen = set()
    out: List[str] = []
    for t in tickers:
        symbol = str(t).strip().upper()
        if not symbol or symbol in seen:
            continue
        if not pattern.match(symbol):
            continue
        seen.add(symbol)
        out.append(symbol)
    return out






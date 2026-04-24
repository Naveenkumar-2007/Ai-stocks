from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from mlops_v2.settings import SETTINGS


@dataclass
class DriftResult:
    drift_score: float
    drifted_features: int
    report_path: Path


def compute_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, report_name: str) -> DriftResult:
    """Evidently drift score as ratio of drifted feature columns."""
    report_dir = SETTINGS.reports_dir / pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{report_name}.html"

    try:
        from evidently import Report  # type: ignore
        from evidently.metric_preset import DataDriftPreset  # type: ignore

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report.save_html(str(report_path))

        as_dict: Dict = report.as_dict()
        metrics = as_dict.get("metrics", [])
        drifted = 0
        total = 0
        for metric in metrics:
            result = metric.get("result", {})
            if "drift_by_columns" in result:
                for _, payload in result["drift_by_columns"].items():
                    total += 1
                    if payload.get("drift_detected"):
                        drifted += 1

        drift_score = (drifted / total) if total else 0.0
        return DriftResult(drift_score=drift_score, drifted_features=drifted, report_path=report_path)

    except Exception as e:
        import traceback
        error_msg = f"Drift check failed; defaulting to train-anyway policy.<br><br><b>Error details:</b><br><pre>{traceback.format_exc()}</pre>"
        report_path.write_text(error_msg, encoding="utf-8")
        return DriftResult(drift_score=1.0, drifted_features=len(reference_df.columns), report_path=report_path)

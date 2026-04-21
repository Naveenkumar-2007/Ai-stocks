from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import traceback

from mlops_v2.settings import SETTINGS, load_tickers
from mlops_v2.training import TrainerV2


def run_daily_drift_gated_training() -> Path:
    trainer = TrainerV2()
    tickers = load_tickers()

    results = []
    for ticker in tickers:
        try:
            result = trainer.train_if_needed(ticker=ticker, force=False)
            results.append({
                "ticker": result.ticker,
                "trained": result.trained,
                "drift_score": result.drift_score,
                "metrics": result.metrics,
                "reason": result.reason,
                "run_id": result.run_id,
                "status": "ok",
            })
        except Exception as exc:
            results.append({
                "ticker": ticker,
                "trained": False,
                "drift_score": 0.0,
                "metrics": {},
                "reason": "exception",
                "run_id": None,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
            })

    out_dir = SETTINGS.reports_dir / datetime.utcnow().strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "daily_training_summary.json"
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    report_path = run_daily_drift_gated_training()
    print(f"Daily drift-gated training summary: {report_path}")

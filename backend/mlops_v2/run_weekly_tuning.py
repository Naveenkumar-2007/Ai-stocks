from __future__ import annotations

import json
import logging
from pathlib import Path

from mlops_v2.data_ingestion import DataIngestionService
from mlops_v2.feature_engineering import FEATURE_COLUMNS, FeatureEngineer
from mlops_v2.settings import load_tickers
from mlops_v2.tuning import tune_xgb_for_sharpe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_weekly_tuning(max_stocks: int | None = None) -> int:
    tickers = load_tickers()
    stocks = tickers[: max_stocks or len(tickers)]
    ingestion = DataIngestionService()
    fe = FeatureEngineer()
    tuned = 0
    results = {}

    for ticker in stocks:
        try:
            raw = ingestion.fetch_with_retries(ticker)
            if raw is None or raw.empty:
                logger.warning("Skipping %s: no data", ticker)
                continue

            artifacts = fe.compute_features(raw)
            feats = artifacts.features
            if len(feats) < 120:
                logger.warning("Skipping %s: insufficient feature rows", ticker)
                continue

            y, fwd_returns = fe.build_targets(raw, feats, horizon_days=1)
            idx = feats.index.intersection(y.index).intersection(fwd_returns.index)
            X = feats.loc[idx, FEATURE_COLUMNS]
            y = y.loc[idx]
            fwd_returns = fwd_returns.loc[idx]

            if len(X) < 100:
                logger.warning("Skipping %s: too few samples after cleanup", ticker)
                continue

            best_params, best_sharpe = tune_xgb_for_sharpe(X, y, fwd_returns, n_trials=50)
            logger.info("%s tuned: best_sharpe=%.4f params=%s", ticker, best_sharpe, best_params)
            results[ticker] = {
                "best_sharpe": best_sharpe,
                "best_params": best_params,
            }
            tuned += 1

        except Exception as exc:
            logger.exception("Tuning failed for %s: %s", ticker, exc)

    output_dir = Path("monitoring/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "latest.json"
    output_file.write_text(json.dumps({"tuned": tuned, "results": results}, indent=2), encoding="utf-8")

    logger.info("Weekly tuning complete. Tuned tickers: %d", tuned)
    return tuned


if __name__ == "__main__":
    run_weekly_tuning()

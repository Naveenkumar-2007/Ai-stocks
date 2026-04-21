"""
Train all stocks listed in mlops/stocks.json and write metrics for DVC.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional

from mlops.config import MLOpsConfig
from mlops.training_pipeline import MLOpsTrainingPipeline


def _safe_remove(path: str) -> None:
	if os.path.isdir(path):
		shutil.rmtree(path, ignore_errors=True)
	elif os.path.isfile(path):
		try:
			os.remove(path)
		except OSError:
			pass


def clean_old_state() -> None:
	"""Clean previous MLflow and local training artifacts for a fresh run."""
	cleanup_targets = [
		"mlruns",
		os.path.join("mlops", "checkpoints"),
		os.path.join("mlops", "logs", "tensorboard"),
		os.path.join("mlops", "metrics"),
		os.path.join("mlops", "model_registry", "model_store"),
		os.path.join("mlops", "model_registry", "metadata.json"),
		os.path.join("mlops", "model_registry", "global_features.json"),
		os.path.join("mlops", "model_registry", "global_scalers.pkl"),
	]
	for target in cleanup_targets:
		_safe_remove(target)


def _to_float(value: Any) -> float | None:
	try:
		if value is None:
			return None
		return float(value)
	except (TypeError, ValueError):
		return None


def _aggregate_metrics(successful: List[Dict[str, Any]], total_requested: int, failed_count: int) -> Dict[str, Any]:
	rmse_values = [_to_float(r.get("metrics", {}).get("rmse")) for r in successful]
	mae_values = [_to_float(r.get("metrics", {}).get("mae")) for r in successful]
	mape_values = [_to_float(r.get("metrics", {}).get("mape")) for r in successful]
	r2_values = [_to_float(r.get("metrics", {}).get("r2")) for r in successful]
	da_values = [_to_float(r.get("metrics", {}).get("directional_accuracy")) for r in successful]

	rmse_values = [v for v in rmse_values if v is not None]
	mae_values = [v for v in mae_values if v is not None]
	mape_values = [v for v in mape_values if v is not None]
	r2_values = [v for v in r2_values if v is not None]
	da_values = [v for v in da_values if v is not None]

	best_by_mape = None
	if successful:
		with_mape = [r for r in successful if _to_float(r.get("metrics", {}).get("mape")) is not None]
		if with_mape:
			winner = min(with_mape, key=lambda x: float(x["metrics"]["mape"]))
			best_by_mape = {
				"ticker": winner.get("ticker"),
				"mape": _to_float(winner.get("metrics", {}).get("mape")),
				"version": winner.get("version"),
			}

	success_count = len(successful)
	success_rate = (success_count / total_requested * 100.0) if total_requested else 0.0

	return {
		"total_requested": total_requested,
		"success_count": success_count,
		"failed_count": failed_count,
		"success_rate": round(success_rate, 4),
		"avg_rmse": round(mean(rmse_values), 6) if rmse_values else None,
		"avg_mae": round(mean(mae_values), 6) if mae_values else None,
		"avg_mape": round(mean(mape_values), 6) if mape_values else None,
		"avg_r2": round(mean(r2_values), 6) if r2_values else None,
		"avg_directional_accuracy": round(mean(da_values), 6) if da_values else None,
		"best_by_mape": best_by_mape,
	}


def train_all_stocks(
	epochs: int,
	batch_size: int,
	days: int,
	clean_start: bool,
	max_stocks: Optional[int] = None,
	start_from: Optional[str] = None,
) -> Dict[str, Any]:
	if isinstance(start_from, str) and start_from.strip().lower() in {"", "none", "null"}:
		start_from = None

	if clean_start:
		clean_old_state()

	os.makedirs(os.path.join("mlops", "metrics"), exist_ok=True)

	tickers = MLOpsConfig.get_stocks()
	if start_from:
		if start_from not in tickers:
			raise ValueError(f"start_from ticker '{start_from}' not found in mlops/stocks.json")
		start_idx = tickers.index(start_from)
		tickers = tickers[start_idx:]

	if max_stocks is not None:
		if max_stocks <= 0:
			raise ValueError("max_stocks must be greater than 0")
		tickers = tickers[:max_stocks]

	if not tickers:
		raise ValueError("No tickers selected for training")

	pipeline = MLOpsTrainingPipeline()

	run_started = datetime.utcnow().isoformat() + "Z"
	started_at_epoch = time.time()

	successful: List[Dict[str, Any]] = []
	failed: List[Dict[str, Any]] = []

	for idx, ticker in enumerate(tickers, start=1):
		print(f"[{idx}/{len(tickers)}] Training {ticker}")
		try:
			model_info = pipeline.train_model(
				ticker=ticker,
				epochs=epochs,
				batch_size=batch_size,
				days=days,
			)

			if model_info.get("success") is False:
				failed.append(
					{
						"ticker": ticker,
						"status": "skipped",
						"error": model_info.get("message", "Skipped by pipeline"),
					}
				)
				continue

			successful.append(
				{
					"ticker": ticker,
					"version": model_info.get("version"),
					"status": "success",
					"metrics": model_info.get("metrics", {}),
				}
			)
		except Exception as exc:
			failed.append(
				{
					"ticker": ticker,
					"status": "failed",
					"error": str(exc),
				}
			)

	duration_seconds = round(time.time() - started_at_epoch, 2)
	aggregate = _aggregate_metrics(successful, len(tickers), len(failed))

	result = {
		"run_started_at": run_started,
		"run_finished_at": datetime.utcnow().isoformat() + "Z",
		"duration_seconds": duration_seconds,
		"training_config": {
			"epochs": epochs,
			"batch_size": batch_size,
			"days": days,
			"clean_start": clean_start,
			"max_stocks": max_stocks,
			"start_from": start_from,
		},
		"aggregate": aggregate,
		"successful": successful,
		"failed": failed,
	}

	summary_path = os.path.join("mlops", "metrics", "all_stocks_metrics.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2)

	# Minimal DVC-friendly metrics file.
	dvc_metrics = {
		"run_started_at": result["run_started_at"],
		"run_finished_at": result["run_finished_at"],
		"duration_seconds": result["duration_seconds"],
		**result["aggregate"],
	}
	dvc_metrics_path = os.path.join("mlops", "metrics", "summary.json")
	with open(dvc_metrics_path, "w", encoding="utf-8") as f:
		json.dump(dvc_metrics, f, indent=2)

	print("Training run complete.")
	print(f"Success: {result['aggregate']['success_count']} | Failed: {result['aggregate']['failed_count']}")
	print(f"Metrics written to: {summary_path}")

	return result


def main() -> int:
	parser = argparse.ArgumentParser(description="Train all stocks from mlops/stocks.json")
	parser.add_argument("--epochs", type=int, default=MLOpsConfig.DEFAULT_EPOCHS)
	parser.add_argument("--batch-size", type=int, default=MLOpsConfig.DEFAULT_BATCH_SIZE)
	parser.add_argument("--days", type=int, default=730)
	parser.add_argument("--max-stocks", type=int, default=None, help="Train only the first N selected stocks")
	parser.add_argument("--start-from", type=str, default=None, help="Start training from this ticker in stocks.json")
	parser.add_argument("--clean-start", action="store_true", help="Remove old mlruns and registry artifacts before training")
	args = parser.parse_args()

	train_all_stocks(
		epochs=args.epochs,
		batch_size=args.batch_size,
		days=args.days,
		clean_start=args.clean_start,
		max_stocks=args.max_stocks,
		start_from=args.start_from,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.time_sensor import TimeSensor

from mlops_v2.settings import load_tickers


def _fetch_validate_check_train(ticker: str) -> dict:
    from mlops_v2.training import TrainerV2

    trainer = TrainerV2()
    result = trainer.train_if_needed(ticker=ticker)
    return {
        "ticker": ticker,
        "trained": result.trained,
        "reason": result.reason,
        "drift_score": result.drift_score,
        "metrics": result.metrics,
    }


def _train_ticker_callable(ticker: str):
    return lambda **_kwargs: _fetch_validate_check_train(ticker)


with DAG(
    dag_id="stock_daily_training",
    description="Daily stock training with validation + drift gating",
    start_date=datetime(2026, 1, 1),
    schedule="30 22 * * 1-5",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": 3,
        "retry_delay": timedelta(minutes=2),
        "retry_exponential_backoff": True,
    },
    tags=["stocks", "mlops_v2"],
) as dag:

    wait_for_market_close = TimeSensor(
        task_id="wait_for_market_close",
        target_time=datetime.strptime("22:30", "%H:%M").time(),
    )

    tickers = load_tickers()

    # Parallelism is controlled by Airflow LocalExecutor + global parallelism=4.
    for ticker in tickers:
        train_task = PythonOperator(
            task_id=f"process_{ticker.replace('.', '_').replace('-', '_').lower()}",
            python_callable=_train_ticker_callable(ticker),
        )
        wait_for_market_close >> train_task

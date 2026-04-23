from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


try:
    from prometheus_client import Counter, Gauge, Histogram

    PREDICTIONS_TOTAL = Counter("predictions_total", "Total predictions made", ["ticker"])
    PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0))
    MODEL_ACCURACY_20D = Gauge("model_accuracy_20d", "Directional accuracy over last 20 predictions", ["ticker"])
    DRIFT_SCORE = Gauge("drift_score", "Latest drift score", ["ticker"])
    VIRAL_TICKER_TRACKING = Gauge("viral_ticker_tracking", "Viral Ticker Tracking", ["ticker"])
    SIMULATED_PNL = Gauge("simulated_pnl", "Simulated PnL % over last 20 trades", ["ticker"])
    SHARPE_RATIO = Gauge("sharpe_ratio", "Sharpe Ratio of last 20 simulated trades", ["ticker"])
except Exception:  # pragma: no cover
    PREDICTIONS_TOTAL = None
    PREDICTION_LATENCY = None
    MODEL_ACCURACY_20D = None
    DRIFT_SCORE = None
    VIRAL_TICKER_TRACKING = None
    SIMULATED_PNL = None
    SHARPE_RATIO = None


@contextmanager
def latency_timer():
    start = perf_counter()
    yield lambda: perf_counter() - start


def inc_prediction(ticker: str) -> None:
    if PREDICTIONS_TOTAL is not None:
        PREDICTIONS_TOTAL.labels(ticker=ticker).inc()


def observe_latency(seconds: float) -> None:
    if PREDICTION_LATENCY is not None:
        PREDICTION_LATENCY.observe(seconds)


def set_accuracy_20d(ticker: str, value: float) -> None:
    if MODEL_ACCURACY_20D is not None:
        MODEL_ACCURACY_20D.labels(ticker=ticker).set(value)


def set_drift_score(ticker: str, value: float) -> None:
    if DRIFT_SCORE is not None:
        DRIFT_SCORE.labels(ticker=ticker).set(value)

def set_viral_ticker(ticker: str, value: float) -> None:
    if VIRAL_TICKER_TRACKING is not None:
        VIRAL_TICKER_TRACKING.labels(ticker=ticker).set(value)

def set_simulated_pnl(ticker: str, value: float) -> None:
    if SIMULATED_PNL is not None:
        SIMULATED_PNL.labels(ticker=ticker).set(value)

def set_sharpe_ratio(ticker: str, value: float) -> None:
    if SHARPE_RATIO is not None:
        SHARPE_RATIO.labels(ticker=ticker).set(value)

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from mlops_v2.settings import SETTINGS

logger = logging.getLogger(__name__)


class DataIngestionService:
    """Fetches EOD data with retries and fallback provider."""

    def __init__(self) -> None:
        SETTINGS.ensure_dirs()

    def fetch_with_retries(self, ticker: str, days: int = 200, retries: int = 3) -> pd.DataFrame:
        from stock_api import get_stock_history

        ticker = ticker.strip().upper()
        last_error: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                df, info = get_stock_history(ticker, days=days, return_info=True)
                if not df.empty:
                    logger.info("Fetched %s rows for %s via %s", len(df), ticker, info.get("source", "unknown"))
                    return self._normalize(df)
            except Exception as exc:  # pragma: no cover - network bound
                last_error = exc
                logger.warning("Primary fetch failed (%s/%s) for %s: %s", attempt, retries, ticker, exc)

            backoff = 2 ** (attempt - 1)
            time.sleep(backoff)

        # Final fallback: cached data if present.
        cached = self._load_latest_cached(ticker)
        if not cached.empty:
            logger.warning("Using cached data for %s after provider failures.", ticker)
            return cached

        raise RuntimeError(f"Failed to fetch data for {ticker} after retries: {last_error}")

    def persist_raw_parquet(self, ticker: str, df: pd.DataFrame, as_of: Optional[datetime] = None) -> Path:
        as_of = as_of or datetime.utcnow()
        date_str = as_of.strftime("%Y-%m-%d")
        out_dir = SETTINGS.raw_dir / ticker.upper()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{date_str}.parquet"
        df.to_parquet(out_path, index=True)
        return out_path

    def persist_dvc_manifest(self, ticker: str, parquet_path: Path) -> Path:
        manifest_dir = SETTINGS.raw_dir / ticker.upper()
        manifest_path = manifest_dir / "latest_manifest.json"
        payload = {
            "ticker": ticker.upper(),
            "path": str(parquet_path).replace("\\", "/"),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_path

    def _load_latest_cached(self, ticker: str) -> pd.DataFrame:
        ticker_dir = SETTINGS.raw_dir / ticker.upper()
        if not ticker_dir.exists():
            return pd.DataFrame()

        parquet_files = sorted(ticker_dir.glob("*.parquet"), reverse=True)
        if not parquet_files:
            return pd.DataFrame()

        try:
            df = pd.read_parquet(parquet_files[0])
            return self._normalize(df)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
            out = out.set_index("datetime")
        if out.index.name is None:
            out.index.name = "datetime"
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out.sort_index()
        return out

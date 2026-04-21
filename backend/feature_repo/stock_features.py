from datetime import timedelta

from feast import Entity, FeatureView, Field  # type: ignore
from feast.types import Float32  # type: ignore
from feast.infra.offline_stores.file_source import FileSource  # type: ignore

stock = Entity(name="ticker", join_keys=["ticker"])

stock_features_source = FileSource(
    path="data/features/offline/features.parquet",
    event_timestamp_column="event_timestamp",
)

stock_feature_view = FeatureView(
    name="stock_feature_view",
    entities=[stock],
    ttl=timedelta(days=30),
    schema=[
        Field(name="rsi_14", dtype=Float32),
        Field(name="macd_diff", dtype=Float32),
        Field(name="bb_width_price", dtype=Float32),
        Field(name="volume_zscore_20", dtype=Float32),
        Field(name="return_5d", dtype=Float32),
        Field(name="atr_14", dtype=Float32),
        Field(name="stoch_k", dtype=Float32),
        Field(name="overnight_gap_pct", dtype=Float32),
        Field(name="return_1d", dtype=Float32),
        Field(name="volume_sma_ratio", dtype=Float32),
    ],
    online=True,
    source=stock_features_source,
)

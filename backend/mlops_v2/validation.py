from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
from typing import List

import pandas as pd

from mlops_v2.settings import SETTINGS


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    report_path: str
    checks_total: int
    checks_passed: int


class DataValidator:
    """Great Expectations validation for stock data before training."""

    def _build_html_report(self, payload: dict) -> str:
        rows = []
        for check in payload.get("checks", []):
            status = "PASS" if check.get("success") else "FAIL"
            rows.append(f"<tr><td>{check.get('check')}</td><td>{status}</td></tr>")

        errors = payload.get("errors", [])
        errors_block = "".join(f"<li>{err}</li>" for err in errors) if errors else "<li>None</li>"

        return (
            "<html><head><title>Validation Report</title></head><body>"
            f"<h2>Validation Report - {payload.get('ticker')}</h2>"
            f"<p><b>OK:</b> {payload.get('ok')}</p>"
            f"<p><b>Checks Passed:</b> {payload.get('checks_passed')} / {payload.get('checks_total')}</p>"
            "<h3>Errors</h3><ul>" + errors_block + "</ul>"
            "<h3>Checks</h3><table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Check</th><th>Status</th></tr>" + "".join(rows) + "</table>"
            f"<p><i>Generated at {payload.get('generated_at')}</i></p>"
            "</body></html>"
        )

    def validate(self, df: pd.DataFrame, ticker: str | None = None) -> ValidationResult:
        errors: List[str] = []
        checks: List[dict] = []

        report_dir = SETTINGS.reports_dir / pd.Timestamp.utcnow().strftime("%Y-%m-%d") / "validation"
        report_dir.mkdir(parents=True, exist_ok=True)
        safe_ticker = (ticker or "unknown").strip().upper().replace("/", "_")
        report_json_path = report_dir / f"{safe_ticker}_validation.json"
        report_html_path = report_dir / f"{safe_ticker}_validation.html"

        def _finalize() -> ValidationResult:
            checks_total = len(checks)
            checks_passed = sum(1 for c in checks if c.get("success"))
            payload = {
                "ticker": safe_ticker,
                "ok": len(errors) == 0,
                "errors": errors,
                "checks_total": checks_total,
                "checks_passed": checks_passed,
                "checks": checks,
                "generated_at": pd.Timestamp.utcnow().isoformat(),
            }
            report_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            report_html_path.write_text(self._build_html_report(payload), encoding="utf-8")
            return ValidationResult(
                ok=len(errors) == 0,
                errors=errors,
                report_path=str(report_html_path),
                checks_total=checks_total,
                checks_passed=checks_passed,
            )

        if df.empty:
            errors.append("Dataframe is empty")
            checks.append({"check": "dataframe_not_empty", "success": False})
            return _finalize()

        try:
            import great_expectations as gx  # type: ignore
        except Exception as exc:  # pragma: no cover
            errors.append(f"Great Expectations import failed: {exc}")
            checks.append({"check": "great_expectations_import", "success": False})
            return _finalize()

        # Great Expectations changed APIs across major versions.
        # Use the legacy API when available; otherwise run equivalent checks directly.
        try:
            ge_df = gx.from_pandas(df.reset_index())
            expectations = [
                ge_df.expect_column_values_to_not_be_null("Close"),
                ge_df.expect_column_values_to_be_between("Close", min_value=0.01, max_value=1_000_000),
                ge_df.expect_column_values_to_not_be_null("Volume"),
                ge_df.expect_column_values_to_be_between("Volume", min_value=1),
                ge_df.expect_column_values_to_be_unique("datetime"),
                ge_df.expect_table_row_count_to_be_between(min_value=SETTINGS.min_rows, max_value=SETTINGS.max_rows),
            ]

            for exp in expectations:
                check_name = exp.get("expectation_config", {}).get("expectation_type", "unknown_expectation")
                success = bool(exp.get("success", False))
                checks.append({"check": check_name, "success": success})
                if not success:
                    errors.append(check_name)
        except Exception:
            close = pd.to_numeric(df.get("Close"), errors="coerce")
            volume = pd.to_numeric(df.get("Volume"), errors="coerce")

            close_not_null = not close.isna().any()
            checks.append({"check": "expect_column_values_to_not_be_null_close", "success": close_not_null})
            if not close_not_null:
                errors.append("expect_column_values_to_not_be_null_close")

            close_range_ok = bool(close.between(0.01, 1_000_000).fillna(False).all())
            checks.append({"check": "expect_column_values_to_be_between_close", "success": close_range_ok})
            if not close_range_ok:
                errors.append("expect_column_values_to_be_between_close")

            volume_not_null = not volume.isna().any()
            checks.append({"check": "expect_column_values_to_not_be_null_volume", "success": volume_not_null})
            if not volume_not_null:
                errors.append("expect_column_values_to_not_be_null_volume")

            volume_min_ok = bool((volume >= 1).fillna(False).all())
            checks.append({"check": "expect_column_values_to_be_between_volume", "success": volume_min_ok})
            if not volume_min_ok:
                errors.append("expect_column_values_to_be_between_volume")

            dt = pd.to_datetime(df.index, utc=True, errors="coerce")
            dt_unique = not (dt.isna().any() or dt.duplicated().any())
            checks.append({"check": "expect_column_values_to_be_unique_datetime", "success": dt_unique})
            if not dt_unique:
                errors.append("expect_column_values_to_be_unique_datetime")

            row_count = len(df)
            row_count_ok = bool(SETTINGS.min_rows <= row_count <= SETTINGS.max_rows)
            checks.append({"check": "expect_table_row_count_to_be_between", "success": row_count_ok})
            if not row_count_ok:
                errors.append("expect_table_row_count_to_be_between")

        # Date recency/coverage check.
        index = pd.to_datetime(df.index, utc=True, errors="coerce")
        if index.isna().all():
            errors.append("datetime_index_invalid")
            checks.append({"check": "datetime_index_valid", "success": False})
        else:
            checks.append({"check": "datetime_index_valid", "success": True})
            min_dt = index.min()
            max_dt = index.max()
            history_ok = (max_dt - min_dt) >= timedelta(days=SETTINGS.min_history_days)
            checks.append({"check": "date_range_at_least_min_history_days", "success": history_ok})
            if not history_ok:
                errors.append("date_range_less_than_100_days")

        return _finalize()

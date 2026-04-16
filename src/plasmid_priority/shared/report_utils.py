"""Shared report utility functions for branch report cards.

Extracted from the duplicated definitions in bio_transfer/report.py,
clinical_hazard/report.py, geo_spread/report.py, and consensus/report.py.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def safe_series_value(frame: pd.DataFrame, column: str, default: Any = float("nan")) -> Any:
    """Safely extract a value from the first row of a DataFrame column."""
    if frame.empty or column not in frame.columns:
        return default
    return frame.iloc[0].get(column, default)


def metric_value(row: pd.Series, column: str) -> float:
    """Extract a numeric metric value from a report-card row."""
    return float(pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0])

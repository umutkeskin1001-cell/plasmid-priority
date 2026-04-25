from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LabelQualityResult:
    label_column: str
    status: str
    n_rows: int
    n_observed: int
    n_positive: int
    n_negative: int
    missing_fraction: float
    positive_fraction: float
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def assess_label_quality(
    frame: pd.DataFrame,
    label_column: str,
    *,
    min_positive: int = 5,
    min_negative: int = 2,
    max_missing_fraction: float = 0.5,
) -> LabelQualityResult:
    if label_column not in frame.columns:
        return LabelQualityResult(
            label_column=label_column,
            status="fail",
            n_rows=int(len(frame)),
            n_observed=0,
            n_positive=0,
            n_negative=0,
            missing_fraction=1.0,
            positive_fraction=0.0,
            reasons=("missing_label_column",),
        )

    labels = frame[label_column]
    observed = labels.dropna()
    n_rows = int(len(labels))
    n_observed = int(len(observed))
    numeric = pd.to_numeric(observed, errors="coerce").dropna()
    n_positive = int((numeric.astype(float) >= 1.0).sum())
    n_negative = int((numeric.astype(float) <= 0.0).sum())
    missing_fraction = float(1.0 - (n_observed / max(n_rows, 1)))
    positive_fraction = float(n_positive / max(len(numeric), 1))

    reasons: list[str] = []
    if missing_fraction > max_missing_fraction:
        reasons.append("missing_fraction_too_high")
    if n_positive < min_positive:
        reasons.append("insufficient_positive_cases")
    if n_negative < min_negative:
        reasons.append("insufficient_negative_cases")
    if numeric.nunique() < 2:
        reasons.append("label_has_single_class")

    return LabelQualityResult(
        label_column=label_column,
        status="pass" if not reasons else "fail",
        n_rows=n_rows,
        n_observed=n_observed,
        n_positive=n_positive,
        n_negative=n_negative,
        missing_fraction=missing_fraction,
        positive_fraction=positive_fraction,
        reasons=tuple(reasons),
    )

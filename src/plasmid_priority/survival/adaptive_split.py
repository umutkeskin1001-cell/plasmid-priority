"""Adaptive temporal splitting based on data density quantiles.

Replaces the fixed ``split_year=2015`` with a quantile-based dynamic split
that adapts to the data distribution. Sparse early years get a wider
pre-split window; dense middle years get a narrower one.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


class AdaptiveTemporalSplitter:
    """Compute a data-driven temporal split point.

    Uses the quantile of the year distribution to determine the split,
    with density-aware adjustment to ensure sufficient training data.

    Parameters
    ----------
    quantile : float
        Quantile of the year distribution to use as the baseline split
        (default 0.60, i.e. 60% of observations before split).
    min_pre_years : int
        Minimum number of distinct years required in the pre-split period.
    min_post_years : int
        Minimum number of distinct years required in the post-split period.
    strategy : {"quantile", "density_peak", "cumsum_gap"}
        How to determine the split:
        - ``quantile``: simple year quantile.
        - ``density_peak``: split before the highest-density year block.
        - ``cumsum_gap``: split at the largest cumulative gap in observations.
    """

    def __init__(
        self,
        *,
        quantile: float = 0.60,
        min_pre_years: int = 3,
        min_post_years: int = 2,
        strategy: Literal["quantile", "density_peak", "cumsum_gap"] = "quantile",
    ) -> None:
        self.quantile = float(np.clip(quantile, 0.1, 0.9))
        self.min_pre_years = int(min_pre_years)
        self.min_post_years = int(min_post_years)
        self.strategy = strategy
        self._split_year: int | None = None
        self._density_info: dict[str, object] = {}

    def fit(self, years: pd.Series | np.ndarray) -> "AdaptiveTemporalSplitter":
        """Learn the split year from observed years.

        Parameters
        ----------
        years : pd.Series | np.ndarray
            Observed isolation years (numeric, may contain NaN).

        Returns
        -------
        self
        """
        y = pd.to_numeric(pd.Series(years), errors="coerce").dropna().astype(int)
        if y.empty:
            raise ValueError("No valid years provided for adaptive split.")

        unique_years = np.sort(y.unique())
        if len(unique_years) < self.min_pre_years + self.min_post_years:
            # Fallback: simple median split
            split = int(np.median(unique_years))
            _log.warning(
                "Too few unique years (%d); falling back to median=%d",
                len(unique_years),
                split,
            )
            self._split_year = split
            return self

        if self.strategy == "quantile":
            split = int(np.quantile(y, self.quantile))
        elif self.strategy == "density_peak":
            split = self._density_peak_split(unique_years, y)
        elif self.strategy == "cumsum_gap":
            split = self._cumsum_gap_split(unique_years, y)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Enforce minimum pre/post windows
        pre_years = unique_years[unique_years <= split]
        post_years = unique_years[unique_years > split]
        if len(pre_years) < self.min_pre_years:
            split = int(unique_years[self.min_pre_years - 1])
            pre_years = unique_years[unique_years <= split]
            post_years = unique_years[unique_years > split]
        if len(post_years) < self.min_post_years:
            split = int(unique_years[-(self.min_post_years + 1)])

        self._split_year = int(split)
        self._density_info = {
            "strategy": self.strategy,
            "quantile": self.quantile,
            "unique_years": len(unique_years),
            "pre_years": len(pre_years),
            "post_years": len(post_years),
            "pre_observations": int((y <= split).sum()),
            "post_observations": int((y > split).sum()),
        }
        _log.info(
            "Adaptive split: year=%d, pre=%d obs, post=%d obs (%s strategy)",
            self._split_year,
            self._density_info["pre_observations"],
            self._density_info["post_observations"],
            self.strategy,
        )
        return self

    def _density_peak_split(
        self,
        unique_years: np.ndarray,
        y: pd.Series,
    ) -> int:
        """Find the densest contiguous block and split before it."""
        counts = y.value_counts().sort_index()
        # 3-year rolling sum to find peak density block
        rolling = counts.rolling(window=3, min_periods=1).sum()
        peak_year = int(rolling.idxmax())
        # Split at the year before the density ramp-up
        idx = int(np.searchsorted(unique_years, peak_year))
        split_idx = max(0, idx - 2)
        return int(unique_years[split_idx])

    def _cumsum_gap_split(
        self,
        unique_years: np.ndarray,
        y: pd.Series,
    ) -> int:
        """Split at the largest gap in the cumulative observation distribution."""
        counts = y.value_counts().sort_index()
        cumsum = counts.cumsum()
        total = cumsum.iloc[-1]
        # Normalize and find largest deviation from uniform
        uniform = np.linspace(0, total, len(cumsum))
        gaps = np.abs(cumsum.to_numpy() - uniform)
        gap_idx = int(np.argmax(gaps))
        return int(counts.index[gap_idx])

    @property
    def split_year(self) -> int:
        if self._split_year is None:
            raise RuntimeError("Splitter must be fitted before accessing split_year.")
        return self._split_year

    @property
    def density_info(self) -> dict[str, object]:
        return self._density_info.copy()

    def transform(self, df: pd.DataFrame, year_col: str = "resolved_year") -> pd.DataFrame:
        """Add split-derived columns to a DataFrame.

        Adds:
        - ``split_year`` (int)
        - ``pre_split`` (bool)
        - ``post_split`` (bool)
        """
        out = df.copy()
        years = pd.to_numeric(out[year_col], errors="coerce")
        out["split_year"] = self.split_year
        out["pre_split"] = years <= self.split_year
        out["post_split"] = years > self.split_year
        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        year_col: str = "resolved_year",
    ) -> pd.DataFrame:
        """Fit on ``df[year_col]`` and return transformed DataFrame."""
        self.fit(df[year_col])
        return self.transform(df, year_col=year_col)

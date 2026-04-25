"""Probabilistic multi-rater label fusion using Dawid-Skene EM.

Replaces hard binary labels (spread_label = 1 if >=3 new countries) with
soft probabilistic labels P(spread=1 | data) that account for uncertainty
in the country count threshold, metadata quality, and temporal censoring.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def _observed_label(value: object, *, n_classes: int) -> int:
    observed = int(float(value))  # type: ignore[arg-type]
    if observed < 0 or observed >= n_classes:
        raise ValueError(f"Observed rater label {observed} is outside [0, {n_classes - 1}]")
    return observed


class DawidSkeneLabelFuser:
    """Dawid-Skene expectation-maximization for multi-rater label fusion.

    Treats each "rater" as a different evidence source for the true label:
    - rater 0: country_count_threshold (hard 3+ rule)
    - rater 1: host_range_expansion (genera/families)
    - rater 2: clinical_escalation_proxy
    - rater 3: temporal_pattern (accelerating vs flat)
    - rater 4: geographic_diversity (Shannon entropy of countries)

    Each rater provides a noisy observation of the true latent label.
    EM iteratively estimates:
    - true class probabilities per sample
    - confusion matrices per rater

    Parameters
    ----------
    n_classes : int
        Number of latent classes (default 2: spread=0, spread=1).
    max_iter : int
        EM iterations.
    tol : float
        Convergence tolerance for log-likelihood.
    """

    def __init__(
        self,
        *,
        n_classes: int = 2,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> None:
        self.n_classes = int(n_classes)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._class_probs: np.ndarray | None = None
        self._confusion_matrices: dict[int, np.ndarray] | None = None
        self._rater_names: list[str] = []
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.log_likelihood_: float = float("nan")

    def fit(
        self,
        votes: pd.DataFrame,
        rater_names: list[str] | None = None,
    ) -> "DawidSkeneLabelFuser":
        """Run Dawid-Skene EM on a vote matrix.

        Parameters
        ----------
        votes : pd.DataFrame
            Rows = samples, columns = raters, values = observed labels (0/1).
            NaN = rater abstained.
        rater_names : list[str] | None
            Human-readable names for raters.

        Returns
        -------
        self
        """
        self._rater_names = rater_names or list(votes.columns)
        n_samples = len(votes)
        K = self.n_classes

        # Initialize: simple majority vote
        y_init = votes.mode(axis=1, dropna=True)[0].fillna(0).astype(int).to_numpy()
        class_counts = np.bincount(y_init, minlength=K) + 1  # Laplace smoothing
        self._class_probs = class_counts / class_counts.sum()

        # Confusion matrices: P(rater_observation | true_class)
        self._confusion_matrices = {}
        for r_idx, col in enumerate(votes.columns):
            cm = np.ones((K, K))  # Laplace smoothing
            for k in range(K):
                mask = y_init == k
                if mask.sum() == 0:
                    continue
                obs = votes.loc[mask, col].dropna().astype(int)
                for obs_val in range(K):
                    cm[k, obs_val] += (obs == obs_val).sum()
            # Normalize rows
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm = cm / row_sums
            self._confusion_matrices[r_idx] = cm

        # EM iterations
        log_likes: list[float] = []
        for iteration in range(self.max_iter):
            # E-step: estimate true label probabilities
            T = np.zeros((n_samples, K))
            for i in range(n_samples):
                for k in range(K):
                    prob = self._class_probs[k]
                    for r_idx, col in enumerate(votes.columns):
                        obs = votes.iloc[i, r_idx]
                        if pd.isna(obs):
                            continue
                        obs_int = _observed_label(obs, n_classes=K)
                        prob *= self._confusion_matrices[r_idx][k, obs_int]
                    T[i, k] = prob
            # Normalize
            row_sums = T.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            T = T / row_sums

            # M-step: update class priors and confusion matrices
            new_class_probs = T.mean(axis=0)
            new_cms: dict[int, np.ndarray] = {}
            for r_idx, col in enumerate(votes.columns):
                cm = np.ones((K, K))  # smoothing
                for k in range(K):
                    for i in range(n_samples):
                        obs = votes.iloc[i, r_idx]
                        if pd.isna(obs):
                            continue
                        obs_int = _observed_label(obs, n_classes=K)
                        cm[k, obs_int] += T[i, k]
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                cm = cm / row_sums
                new_cms[r_idx] = cm

            # Log-likelihood
            log_like = 0.0
            for i in range(n_samples):
                sample_ll = 0.0
                for k in range(K):
                    prob = new_class_probs[k]
                    for r_idx, col in enumerate(votes.columns):
                        obs = votes.iloc[i, r_idx]
                        if pd.isna(obs):
                            continue
                        obs_int = _observed_label(obs, n_classes=K)
                        prob *= new_cms[r_idx][k, obs_int]
                    sample_ll += prob
                log_like += np.log(max(sample_ll, 1e-12))
            log_likes.append(log_like)

            self._class_probs = new_class_probs
            self._confusion_matrices = new_cms
            self.n_iter_ = iteration + 1
            self.log_likelihood_ = float(log_like)

            if iteration > 0 and abs(log_likes[-1] - log_likes[-2]) < self.tol:
                self.converged_ = True
                _log.info("Dawid-Skene converged at iteration %d", iteration)
                break

        return self

    def predict_proba(self, votes: pd.DataFrame) -> pd.DataFrame:
        """Return P(true_label=k | votes) for each sample and class.

        Returns a DataFrame with columns ``prob_class_0``, ``prob_class_1``, ...
        """
        if self._class_probs is None or self._confusion_matrices is None:
            raise RuntimeError("Model must be fitted first.")

        n_samples = len(votes)
        K = self.n_classes
        T = np.zeros((n_samples, K))
        for i in range(n_samples):
            for k in range(K):
                prob = self._class_probs[k]
                for r_idx, col in enumerate(votes.columns):
                    obs = votes.iloc[i, r_idx]
                    if pd.isna(obs):
                        continue
                    obs_int = _observed_label(obs, n_classes=K)
                    prob *= self._confusion_matrices[r_idx][k, obs_int]
                T[i, k] = prob
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T = T / row_sums

        cols = [f"prob_class_{k}" for k in range(K)]
        return pd.DataFrame(T, index=votes.index, columns=cols)

    def predict(self, votes: pd.DataFrame) -> pd.Series:
        """Return the most probable true label per sample."""
        proba = self.predict_proba(votes)
        return proba.idxmax(axis=1).str.replace("prob_class_", "").astype(int)

    @property
    def rater_reliability(self) -> pd.DataFrame:
        """Return diagonal accuracy of each rater's confusion matrix."""
        if self._confusion_matrices is None:
            raise RuntimeError("Model not fitted.")
        rows = []
        for r_idx, cm in self._confusion_matrices.items():
            diag = np.diag(cm)
            rows.append(
                {
                    "rater": self._rater_names[r_idx]
                    if r_idx < len(self._rater_names)
                    else f"rater_{r_idx}",
                    "accuracy_class_0": float(diag[0]) if len(diag) > 0 else np.nan,
                    "accuracy_class_1": float(diag[1]) if len(diag) > 1 else np.nan,
                    "mean_accuracy": float(diag.mean()),
                }
            )
        return pd.DataFrame(rows)


def build_probabilistic_labels(
    records: pd.DataFrame,
    *,
    split_year: int = 2015,
    backbone_col: str = "backbone_id",
    year_col: str = "resolved_year",
    country_col: str = "country",
    host_genus_col: str = "host_genus",
    clinical_context_col: str = "clinical_context",
    horizon_years: int = 5,
) -> pd.DataFrame:
    """Build probabilistic labels from multiple weak evidence sources.

    Each backbone gets 5 rater votes:
    1. Country threshold: 1 if >=3 new countries in horizon.
    2. Host expansion: 1 if >=2 new genera in horizon.
    3. Clinical proxy: 1 if clinical fraction gain >= 0.15.
    4. Temporal acceleration: 1 if country count slope is positive.
    5. Geographic diversity: 1 if Shannon entropy of post-split countries > median.

    Dawid-Skene EM fuses these into P(spread=1 | data).

    Returns
    -------
    pd.DataFrame
        One row per backbone with columns:
        - ``backbone_id``
        - ``prob_spread_0``, ``prob_spread_1`` (from Dawid-Skene)
        - ``hard_spread_label`` (original threshold-based)
        - ``label_confidence`` (max probability)
        - ``label_noise_estimate`` (1 - confidence)
    """
    if records.empty:
        return pd.DataFrame(
            columns=[
                backbone_col,
                "prob_spread_0",
                "prob_spread_1",
                "hard_spread_label",
                "label_confidence",
                "label_noise_estimate",
            ]
        )
    df = records.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[backbone_col, year_col])
    if df.empty:
        return pd.DataFrame(
            columns=[
                backbone_col,
                "prob_spread_0",
                "prob_spread_1",
                "hard_spread_label",
                "label_confidence",
                "label_noise_estimate",
            ]
        )

    rows: list[dict[str, object]] = []
    for bb_id, group in df.groupby(backbone_col, sort=False):
        years = group[year_col].astype(float)
        future_mask = (years > split_year) & (years <= split_year + horizon_years)
        pre_mask = years <= split_year

        # Rater 0: country threshold
        pre_countries = (
            set(
                group.loc[pre_mask, country_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .unique()
            )
            if country_col in group.columns
            else set()
        )
        post_countries = (
            set(
                group.loc[future_mask, country_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .unique()
            )
            if country_col in group.columns
            else set()
        )
        new_countries = post_countries - pre_countries
        r0 = 1 if len(new_countries) >= 3 else 0

        # Rater 1: host expansion
        pre_genera = (
            set(
                group.loc[pre_mask, host_genus_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .unique()
            )
            if host_genus_col in group.columns
            else set()
        )
        post_genera = (
            set(
                group.loc[future_mask, host_genus_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", np.nan)
                .dropna()
                .unique()
            )
            if host_genus_col in group.columns
            else set()
        )
        new_genera = post_genera - pre_genera
        r1 = 1 if len(new_genera) >= 2 else 0

        # Rater 2: clinical proxy
        r2: float = np.nan
        r2_observed = False
        if (
            clinical_context_col in group.columns
            and bool(pre_mask.any())
            and bool(future_mask.any())
        ):
            pre_values = group.loc[pre_mask, clinical_context_col]
            post_values = group.loc[future_mask, clinical_context_col]
            if len(pre_values) > 0 and len(post_values) > 0:
                pre_clinical = (
                    pre_values.fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin({"clinical", "hospital", "patient", "human"})
                    .mean()
                )
                post_clinical = (
                    post_values.fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin({"clinical", "hospital", "patient", "human"})
                    .mean()
                )
                if pd.notna(pre_clinical) and pd.notna(post_clinical):
                    r2 = 1.0 if float(post_clinical) - float(pre_clinical) >= 0.15 else 0.0
                    r2_observed = True

        # Rater 3: temporal acceleration (slope of unique countries per year)
        r3 = 0
        if country_col in group.columns and not group.empty:
            yearly = (
                group.loc[pre_mask | future_mask, [year_col, country_col]]
                .dropna()
                .groupby(year_col)[country_col]
                .nunique()
                .sort_index()
            )
            if len(yearly) >= 2:
                x = yearly.index.to_numpy()
                y = yearly.to_numpy()
                slope = np.polyfit(x, y, 1)[0] if len(x) >= 2 else 0.0
                r3 = 1 if slope > 0.1 else 0

        # Rater 4: geographic diversity (Shannon entropy)
        r4 = 0
        if country_col in group.columns and len(post_countries) > 0:
            country_counts = group.loc[future_mask, country_col].value_counts()
            probs = country_counts / country_counts.sum()
            entropy: float = float(-np.sum(probs * np.log2(probs + 1e-12)))
            # Compare to a reference: ~log2(n_countries) is max
            max_ent = np.log2(max(len(post_countries), 2))
            r4 = 1 if entropy / max_ent > 0.5 else 0

        rows.append(
            {
                backbone_col: bb_id,
                "rater_country_threshold": r0,
                "rater_host_expansion": r1,
                "rater_clinical_proxy": r2,
                "rater_temporal_acceleration": r3,
                "rater_geo_diversity": r4,
                "rater_clinical_proxy_observed": r2_observed,
                "hard_spread_label": r0,  # original definition
                "n_new_countries": len(new_countries),
            }
        )

    vote_df = pd.DataFrame(rows)
    if vote_df.empty:
        return vote_df

    # Run Dawid-Skene
    rater_cols = [
        "rater_country_threshold",
        "rater_host_expansion",
        "rater_clinical_proxy",
        "rater_temporal_acceleration",
        "rater_geo_diversity",
    ]
    fuser = DawidSkeneLabelFuser(n_classes=2, max_iter=30)
    fuser.fit(vote_df[rater_cols], rater_names=rater_cols)
    proba = fuser.predict_proba(vote_df[rater_cols])

    result = vote_df[[backbone_col, "hard_spread_label", "n_new_countries"]].copy()
    result["prob_spread_0"] = proba["prob_class_0"].to_numpy()
    result["prob_spread_1"] = proba["prob_class_1"].to_numpy()
    result["label_confidence"] = result[["prob_spread_0", "prob_spread_1"]].max(axis=1)
    result["label_noise_estimate"] = 1.0 - result["label_confidence"]
    result["fused_spread_label"] = fuser.predict(vote_df[rater_cols]).to_numpy()
    result["label_fusion_converged"] = bool(fuser.converged_)
    result["label_fusion_iterations"] = int(fuser.n_iter_)
    result["label_fusion_log_likelihood"] = float(fuser.log_likelihood_)
    result["rater_clinical_proxy_observed"] = (
        vote_df["rater_clinical_proxy_observed"].astype(bool).to_numpy()
    )

    # Reliability diagnostics
    reliability = fuser.rater_reliability
    _log.info(
        "Probabilistic labels: mean confidence=%.3f, estimated noise=%.3f",
        result["label_confidence"].mean(),
        result["label_noise_estimate"].mean(),
    )
    for _, row in reliability.iterrows():
        _log.info(
            "Rater %s accuracy: class0=%.3f, class1=%.3f, mean=%.3f",
            row["rater"],
            row["accuracy_class_0"],
            row["accuracy_class_1"],
            row["mean_accuracy"],
        )

    return result

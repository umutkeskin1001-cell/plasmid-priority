"""
Exploratory script - not part of the canonical pipeline.

This one-off script checks whether polynomial interaction terms improve
the support-synergy style feature surface relative to a linear baseline.
Result: polynomial interactions did not produce a meaningful ROC AUC gain,
so the headline model remains a linear L2-regularized logistic regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from plasmid_priority.config import build_context


def main() -> int:
    context = build_context()
    scored = pd.read_csv(context.root / "data/scores/backbone_scored.tsv", sep="\t")
    scored = scored.loc[scored["spread_label"].notna()].copy()
    y = scored["spread_label"].astype(int).to_numpy(dtype=int)

    features = [
        "T_eff_norm",
        "H_specialization_norm",
        "A_eff_norm",
        "coherence_score",
        "backbone_purity_norm",
        "assignment_confidence_norm",
        "mash_neighbor_distance_train_norm",
        "replicon_architecture_norm",
        "clinical_context_fraction_norm",
        "A_recurrence_norm",
        "pmlst_coherence_norm",
        "ecology_context_diversity_norm",
        "H_external_host_range_support",
        "pmlst_presence_fraction_train",
        "amr_support_norm",
        "metadata_support_depth_norm",
        "H_external_host_range_norm",
        "external_t_synergy_norm",
    ]
    X = scored[features].fillna(0.0).to_numpy(dtype=float)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_linear: list[float] = []
    auc_poly: list[float] = []
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        linear = LogisticRegressionCV(scoring="roc_auc", max_iter=1000)
        linear.fit(X_train_scaled, y_train)
        auc_linear.append(roc_auc_score(y_test, linear.predict_proba(X_test_scaled)[:, 1]))

        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)
        poly_model = LogisticRegressionCV(scoring="roc_auc", max_iter=1000, penalty="l1", solver="liblinear")
        poly_model.fit(X_train_poly, y_train)
        auc_poly.append(roc_auc_score(y_test, poly_model.predict_proba(X_test_poly)[:, 1]))

    print(f"Linear AUC: {np.mean(auc_linear):.4f}")
    print(f"Poly+L1 AUC: {np.mean(auc_poly):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from plasmid_priority.modeling.ensemble_strategies import (
    EnsembleConfig,
    MetaSovereignEnsemble,
    create_meta_sovereign_model,
)


class _ColumnModel:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_ColumnModel":
        self.model.fit(X.loc[:, self.columns], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.loc[:, self.columns])


class EnsembleStrategyTests(unittest.TestCase):
    def _factory(self, name: str) -> _ColumnModel:
        mapping = {
            "model_a": ["x1"],
            "model_b": ["x2"],
            "model_c": ["x1", "x2"],
        }
        return _ColumnModel(mapping[name])

    def test_meta_sovereign_ensemble_predicts_with_real_uncertainty(self) -> None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {
                "x1": np.linspace(-2.0, 2.0, 80),
                "x2": np.sin(np.linspace(0.0, 3.0, 80)),
            }
        )
        y = ((X["x1"] + 0.8 * X["x2"]) > 0).astype(int).to_numpy()

        ensemble = MetaSovereignEnsemble(
            EnsembleConfig(base_models=["model_a", "model_b", "model_c"], n_folds=4)
        )
        ensemble.fit_base_models(X, y, self._factory)
        ensemble.fit_meta_learner(X, y)

        new_X = pd.DataFrame(
            {
                "x1": rng.normal(size=12),
                "x2": rng.normal(size=12),
            }
        )
        result = ensemble.predict(new_X)

        self.assertEqual(result["probability"].shape, (12,))
        self.assertEqual(result["uncertainty"].shape, (12,))
        self.assertEqual(result["confidence"].shape, (12,))
        self.assertTrue(np.all((result["probability"] >= 0.0) & (result["probability"] <= 1.0)))
        self.assertTrue(np.all((result["confidence"] >= 0.0) & (result["confidence"] <= 1.0)))
        self.assertFalse(np.allclose(result["uncertainty"], 0.1))

    def test_meta_sovereign_factory_calibrates_from_base_predictions(self) -> None:
        base_predictions = {
            "model_a": np.array([0.1, 0.2, 0.8, 0.9], dtype=float),
            "model_b": np.array([0.2, 0.3, 0.7, 0.8], dtype=float),
            "model_c": np.array([0.15, 0.25, 0.75, 0.85], dtype=float),
        }
        y_true = np.array([0, 0, 1, 1], dtype=int)

        result = create_meta_sovereign_model(base_predictions, y_true, method="adaptive_weighted")

        self.assertIn("probability", result)
        self.assertTrue(np.all((result["probability"] >= 0.0) & (result["probability"] <= 1.0)))
        self.assertEqual(set(result["weights"]), set(base_predictions))


if __name__ == "__main__":
    unittest.main()

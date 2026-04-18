from __future__ import annotations

import pandas as pd

from plasmid_priority.validation.falsification import build_outcome_permutation_falsification


def _toy_scored() -> pd.DataFrame:
    rows = []
    for index in range(40):
        rows.append(
            {
                "backbone_id": f"bb_{index}",
                "spread_label": 1 if index % 2 == 0 else 0,
                "feature_signal": float(index) / 39.0,
            }
        )
    return pd.DataFrame(rows)


def test_outcome_permutation_skips_without_model_evaluator() -> None:
    scored = _toy_scored()

    result = build_outcome_permutation_falsification(scored, "governance_linear", n_permutations=10)

    assert not result.empty
    assert str(result.loc[0, "status"]) == "skipped_missing_model_evaluator"


def test_outcome_permutation_runs_with_injected_model_evaluator() -> None:
    scored = _toy_scored()

    def _fake_evaluator(frame: pd.DataFrame, _model_name: str) -> dict[str, float]:
        prevalence = float(frame["spread_label"].mean())
        return {
            "status": "ok",
            "roc_auc": 0.75 if prevalence > 0.49 else 0.51,
            "average_precision": 0.70 if prevalence > 0.49 else 0.52,
        }

    result = build_outcome_permutation_falsification(
        scored,
        "governance_linear",
        n_permutations=10,
        model_evaluator=_fake_evaluator,
        known_model_names={"governance_linear"},
    )

    assert not result.empty
    assert str(result.loc[0, "status"]) == "completed"
    assert "auc_collapse_delta" in result.columns

"""Sidecar bridge for firthlogist running under a Python 3.10 environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        raise SystemExit(
            "usage: python -m plasmid_priority.modeling.firthlogist_bridge "
            "<input_npz> <output_json>"
        )
    input_path = Path(argv[0])
    output_path = Path(argv[1])

    from firthlogist import FirthLogisticRegression

    payload = np.load(input_path, allow_pickle=False)
    X = np.asarray(payload["X"], dtype=float)
    y = np.asarray(payload["y"], dtype=int)
    max_iter = int(payload["max_iter"][0]) if "max_iter" in payload.files else 100

    clf = FirthLogisticRegression(
        max_iter=max(max_iter, 25),
        skip_pvals=True,
        skip_ci=True,
    )
    clf.fit(X, y)

    result = {
        "beta": [float(clf.intercept_)]
        + [float(value) for value in np.asarray(clf.coef_, dtype=float)],
        "iterations_run": int(getattr(clf, "n_iter_", max_iter)),
        "converged": bool(getattr(clf, "converged_", True)),
        "solver": "firthlogist_sidecar",
    }
    output_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

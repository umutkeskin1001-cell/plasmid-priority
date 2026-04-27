# Plasmid Priority Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Plasmid Priority into a leakage-resistant, reproducible, scientifically defensible retrospective surveillance pipeline with stronger configuration contracts, validation gates, modular modeling internals, and higher critical-path test coverage.

**Architecture:** The plan starts with correctness gates at the temporal, label, and branch-context boundaries, then hardens model evaluation, caching, and pipeline orchestration. After the high-risk behavior is protected by tests, large implementation files are split behind stable public APIs so existing scripts keep working while internals become maintainable.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, LightGBM optional track, Pydantic v2, pytest, pytest-cov, Hypothesis, uv.

---

## Operating Rules

- Preserve public imports from `src/plasmid_priority/modeling/module_a.py`, `src/plasmid_priority/features/core.py`, and existing numbered scripts unless a task explicitly changes them.
- Treat `config/benchmarks.yaml`, `config.yaml`, and `src/plasmid_priority/protocol.py` as scientific contract inputs, not incidental defaults.
- Write a failing test before each behavioral change.
- Keep every task independently mergeable; commit after each task.
- Use `uv run pytest -q -o addopts='' ...` for targeted tests, then the quality gate in Task 13 before final merge.
- Register jCodemunch edits after implementation sessions with `register_edit` for changed paths when hooks are not auto-reindexing.

## Target End State

- Missing or invalid temporal metadata cannot silently enter training, feature normalization, or labels.
- Governance model validation defaults to temporal/group-aware folds, with random stratified CV retained only as an explicitly named discovery diagnostic.
- Branch runs load configuration from the actual `ProjectContext`, including layered config paths and alternate data roots.
- Weak label fusion treats unsupported evidence as abstention, validates rater labels, and records convergence diagnostics.
- Cache keys include data, feature schema, fit config, software fingerprint, and protocol hash; unsafe cache loads are limited to trusted local cache roots.
- Critical-path coverage gate includes temporal helpers, probabilistic labels, branch contracts, model folds, calibration, config validation, and cache manifests.
- `module_a.py` is reduced by moving fold generation, preprocessing, calibration, prediction evaluation, and cache glue into focused modules behind compatibility wrappers.

---

## File Responsibility Map

### Temporal Correctness

- Modify `src/plasmid_priority/shared/temporal.py`: canonical year coercion, required-year validation, temporal window masks, group/time split helpers.
- Modify `src/plasmid_priority/backbone/core.py`: replace `fillna(0)` temporal filtering with explicit validation or quarantine behavior.
- Modify `src/plasmid_priority/backbone/graph_clustering.py`: validate temporal windows and handle unknown years safely.
- Test `tests/test_temporal_contracts.py`: new unit tests for missing years, invalid windows, and mask behavior.
- Test `tests/test_backbone.py`: regression tests for training-only assignment and coherence with invalid years.
- Test `tests/test_backbone_graph_clustering.py`: regression tests for all-missing metadata and non-positive window size.

### Label Fusion

- Modify `src/plasmid_priority/labels/probabilistic.py`: rater abstention, rater label range checks, EM convergence report.
- Test `tests/test_probabilistic_labels.py`: unsupported clinical windows abstain, invalid rater values fail, convergence diagnostics are present.

### Model Evaluation

- Create `src/plasmid_priority/modeling/temporal_cv.py`: reusable temporal/group-aware splitters returning index arrays compatible with current OOF functions.
- Modify `src/plasmid_priority/modeling/module_a_support.py`: add `split_strategy` support while preserving `_stratified_folds`.
- Modify `src/plasmid_priority/modeling/module_a.py`: use governance default `temporal_group` folds and discovery default `stratified_repeated`.
- Test `tests/test_modeling_temporal_cv.py`: no group overlap, no future-to-past leakage, rare-class failures are actionable.
- Test `tests/test_modeling.py`: model evaluation reports selected split strategy.

### Branch Context

- Modify `src/plasmid_priority/shared/branch_base.py`: pass `ProjectContext` through `build_labels`, `build_features`, `evaluate_branch`, `build_predictions`, and `calibrate`.
- Modify `src/plasmid_priority/shared/branches.py`: replace no-arg and `None` config loading with context-aware loading.
- Modify branch CLIs in `src/plasmid_priority/geo_spread/cli.py`, `src/plasmid_priority/bio_transfer/cli.py`, `src/plasmid_priority/clinical_hazard/cli.py`, and `src/plasmid_priority/consensus/cli.py` only where signatures need context propagation.
- Test `tests/shared/test_branching.py` and `tests/test_hardening_data_root.py`: alternate data root and runtime config are honored.

### Configuration Contracts

- Modify `src/plasmid_priority/config.py`: Pydantic validators for split year, horizons, thresholds, consensus weights, OOD thresholds, and host phylogeny weights.
- Modify `tests/test_config.py`: invalid values raise precise `ValidationError`; valid weights normalize deterministically.
- Modify `docs/benchmark_contract.md` or generated contract source if docs are generated by `scripts/31_generate_scientific_contracts.py`.

### Cache Safety

- Modify `src/plasmid_priority/modeling/fold_cache.py`: include fit config hash, protocol hash, software fingerprint, and feature order in model cache keys.
- Modify `src/plasmid_priority/modeling/oof_cache.py` and `src/plasmid_priority/modeling/matrix_cache.py`: align key material and cache manifest metadata.
- Test `tests/test_model_caches.py`: hyperparameter changes miss cache; corrupted cache emits warning and recomputes.

### Coverage and Quality Gates

- Modify `pyproject.toml`: remove critical-path omissions from coverage or create a separate critical coverage command.
- Modify `.github/workflows/ci.yml`: add critical coverage job and temporal/leakage smoke tests.
- Test `tests/test_ci_workflow.py`: workflow includes the quality commands.

### Documentation

- Create `docs/adr/0001-temporal-validation-policy.md`: temporal metadata policy.
- Create `docs/adr/0002-governance-validation-strategy.md`: temporal/group CV rationale.
- Create `docs/adr/0003-probabilistic-label-fusion.md`: weak-label rater and abstention policy.
- Modify `README.md`: add concise release-quality command set and data-root rules.

---

## Phase 1: Correctness Fixes

### Task 1: Centralize Temporal Metadata Validation

**Files:**
- Modify: `src/plasmid_priority/shared/temporal.py`
- Modify: `src/plasmid_priority/backbone/core.py`
- Test: `tests/test_temporal_contracts.py`
- Test: `tests/test_backbone.py`

- [ ] **Step 1: Add failing tests for required-year behavior**

Create `tests/test_temporal_contracts.py` with:

```python
from __future__ import annotations

import pandas as pd
import pytest

from plasmid_priority.shared.temporal import (
    TemporalMetadataError,
    coerce_required_years,
    pre_split_mask,
)


def test_coerce_required_years_rejects_missing_years() -> None:
    frame = pd.DataFrame({"resolved_year": [2010, None, "bad"]})

    with pytest.raises(TemporalMetadataError, match="resolved_year"):
        coerce_required_years(frame, "resolved_year", context="unit-test")


def test_pre_split_mask_does_not_treat_missing_as_training() -> None:
    years = pd.Series([2014, None, "bad", 2016])

    mask = pre_split_mask(years, split_year=2015)

    assert mask.tolist() == [True, False, False, False]
```

Run:

```bash
uv run pytest -q tests/test_temporal_contracts.py -o addopts=''
```

Expected: fails because `TemporalMetadataError` and `coerce_required_years` do not exist.

- [ ] **Step 2: Implement required temporal helpers**

In `src/plasmid_priority/shared/temporal.py`, add:

```python
class TemporalMetadataError(ValueError):
    """Raised when temporal metadata required for leakage-safe evaluation is invalid."""


def coerce_required_years(
    frame: pd.DataFrame,
    year_column: str,
    *,
    context: str,
) -> pd.Series:
    if year_column not in frame.columns:
        raise TemporalMetadataError(f"{context}: missing required year column '{year_column}'")
    numeric = pd.to_numeric(frame[year_column], errors="coerce")
    invalid = numeric.isna()
    if bool(invalid.any()):
        examples = frame.loc[invalid, year_column].head(5).astype(str).tolist()
        raise TemporalMetadataError(
            f"{context}: {int(invalid.sum())} row(s) have invalid '{year_column}' values: {examples}"
        )
    return numeric.astype(int)
```

Update `pre_split_mask()` so NaN remains false:

```python
def pre_split_mask(
    years: pd.Series | Sequence[Any],
    *,
    split_year: int,
) -> pd.Series:
    numeric = _coerce_year_series(years)
    return numeric.notna() & (numeric <= float(split_year))
```

- [ ] **Step 3: Remove `fillna(0)` from backbone temporal filtering**

In `src/plasmid_priority/backbone/core.py`, change training-only assignment:

```python
from plasmid_priority.shared.temporal import coerce_required_years

years = coerce_required_years(
    assigned,
    "resolved_year",
    context="assign_backbone_ids_training_only",
)
training_mask = years <= int(split_year)
```

Change coherence:

```python
years = coerce_required_years(
    records,
    "resolved_year",
    context="compute_backbone_coherence",
)
training = records.loc[years <= int(split_year)].copy()
```

- [ ] **Step 4: Add backbone regression tests**

Append to `tests/test_backbone.py`:

```python
import pytest

from plasmid_priority.backbone.core import (
    assign_backbone_ids_training_only,
    compute_backbone_coherence,
)
from plasmid_priority.shared.temporal import TemporalMetadataError


def test_training_only_assignment_rejects_missing_resolved_year() -> None:
    records = pd.DataFrame(
        {
            "resolved_year": [2014, None],
            "primary_cluster_id": ["C1", "C2"],
            "predicted_mobility": ["mobilizable", "mobilizable"],
            "mpf_type": ["F", "F"],
            "primary_replicon": ["IncF", "IncF"],
            "sequence_length": [1000, 1000],
        }
    )

    with pytest.raises(TemporalMetadataError):
        assign_backbone_ids_training_only(records, split_year=2015)


def test_coherence_rejects_invalid_resolved_year() -> None:
    records = pd.DataFrame(
        {
            "resolved_year": [2014, "unknown"],
            "backbone_id": ["B1", "B1"],
            "predicted_mobility": ["mobilizable", "mobilizable"],
            "primary_replicon": ["IncF", "IncF"],
            "topology": ["circular", "circular"],
        }
    )

    with pytest.raises(TemporalMetadataError):
        compute_backbone_coherence(records, split_year=2015)
```

- [ ] **Step 5: Verify**

Run:

```bash
uv run pytest -q tests/test_temporal_contracts.py tests/test_backbone.py -o addopts=''
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/shared/temporal.py src/plasmid_priority/backbone/core.py tests/test_temporal_contracts.py tests/test_backbone.py
git commit -m "fix: enforce temporal metadata before training splits"
```

### Task 2: Harden Graph Temporal Versioning

**Files:**
- Modify: `src/plasmid_priority/backbone/graph_clustering.py`
- Test: `tests/test_backbone_graph_clustering.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_backbone_graph_clustering.py`:

```python
def test_temporal_versioning_all_missing_years_uses_unknown_window() -> None:
    clustering = MashLeidenClustering(temporal_window_years=5)
    clustering._cluster_map = {"backbone": {"seq1": "B1", "seq2": "B1"}}
    metadata = pd.DataFrame(
        {"sequence_accession": ["seq1", "seq2"], "resolved_year": [None, "bad"]}
    )

    clustering._apply_temporal_versioning(metadata, "sequence_accession", "resolved_year")

    assert clustering._cluster_map["backbone"]["seq1"] == "B1_vunknown"
    assert clustering._cluster_map["backbone"]["seq2"] == "B1_vunknown"


def test_temporal_versioning_rejects_non_positive_window() -> None:
    clustering = MashLeidenClustering(temporal_window_years=0)
    clustering._cluster_map = {"backbone": {"seq1": "B1"}}
    metadata = pd.DataFrame({"sequence_accession": ["seq1"], "resolved_year": [2010]})

    with pytest.raises(ValueError, match="temporal_window_years"):
        clustering._apply_temporal_versioning(metadata, "sequence_accession", "resolved_year")
```

Run:

```bash
uv run pytest -q tests/test_backbone_graph_clustering.py -o addopts=''
```

Expected: fails on current all-missing year behavior and missing validation.

- [ ] **Step 2: Implement temporal versioning safeguards**

In `_apply_temporal_versioning()`:

```python
window_size = int(self.temporal_window_years)
if window_size <= 0:
    raise ValueError("temporal_window_years must be a positive integer")

years = pd.to_numeric(metadata[year_col], errors="coerce")
valid_years = years.dropna()
if valid_years.empty:
    id_to_window = {
        str(row[id_col]).strip(): "unknown"
        for _, row in metadata.iterrows()
    }
else:
    min_year = int(valid_years.min())
    max_year = int(valid_years.max())
    windows = list(range(min_year, max_year + 1, window_size))
    id_to_window = {}
    for _, row in metadata.iterrows():
        sid = str(row[id_col]).strip()
        raw_year = pd.to_numeric(pd.Series([row.get(year_col)]), errors="coerce").iloc[0]
        if pd.isna(raw_year):
            id_to_window[sid] = "unknown"
            continue
        idx = max(0, int((int(raw_year) - min_year) // window_size))
        window_start = windows[min(idx, len(windows) - 1)]
        id_to_window[sid] = f"{window_start}_{window_start + window_size - 1}"
```

Keep the suffix append:

```python
self._cluster_map[level][sid] = f"{base}_v{window}"
```

- [ ] **Step 3: Verify**

Run:

```bash
uv run pytest -q tests/test_backbone_graph_clustering.py -o addopts=''
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/plasmid_priority/backbone/graph_clustering.py tests/test_backbone_graph_clustering.py
git commit -m "fix: handle unknown years in temporal backbone versioning"
```

### Task 3: Make Probabilistic Label Raters Abstain When Unsupported

**Files:**
- Modify: `src/plasmid_priority/labels/probabilistic.py`
- Test: `tests/test_probabilistic_labels.py`

- [ ] **Step 1: Add failing tests for abstention and label validation**

Append to `tests/test_probabilistic_labels.py`:

```python
def test_probabilistic_labels_abstain_for_empty_clinical_windows() -> None:
    records = pd.DataFrame(
        {
            "backbone_id": ["B1", "B1"],
            "resolved_year": [2010, 2011],
            "country": ["TR", "TR"],
            "host_genus": ["Escherichia", "Escherichia"],
            "clinical_context": ["clinical", "clinical"],
        }
    )

    labels = build_probabilistic_labels(records, split_year=2015, horizon_years=5)

    assert "rater_clinical_proxy_observed" in labels.columns
    assert labels["rater_clinical_proxy_observed"].tolist() == [False]


def test_dawid_skene_rejects_out_of_range_rater_values() -> None:
    votes = pd.DataFrame({"rater_a": [0, 1, 2]})
    fuser = DawidSkeneLabelFuser(n_classes=2)

    with pytest.raises(ValueError, match="outside"):
        fuser.fit(votes)
```

Run:

```bash
uv run pytest -q tests/test_probabilistic_labels.py -o addopts=''
```

Expected: fails because unsupported clinical evidence is encoded as `0` and range validation is absent.

- [ ] **Step 2: Add rater value validation**

Replace `_observed_label()` with:

```python
def _observed_label(value: object, *, n_classes: int) -> int:
    observed = int(float(value))  # type: ignore[arg-type]
    if observed < 0 or observed >= n_classes:
        raise ValueError(f"Observed rater label {observed} is outside [0, {n_classes - 1}]")
    return observed
```

Update every call:

```python
obs_int = _observed_label(obs, n_classes=K)
```

- [ ] **Step 3: Encode clinical unsupported evidence as NaN**

In `build_probabilistic_labels()`, change rater declarations:

```python
r2: float = np.nan
r2_observed = False
if clinical_context_col in group.columns and bool(pre_mask.any()) and bool(future_mask.any()):
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
```

Add to the row:

```python
"rater_clinical_proxy_observed": r2_observed,
```

Keep `rater_cols` limited to the five rater value columns so the observed flag is diagnostic only.

- [ ] **Step 4: Return diagnostics**

Add convergence attributes to `DawidSkeneLabelFuser`:

```python
self.n_iter_: int = 0
self.converged_: bool = False
self.log_likelihood_: float = float("nan")
```

Inside EM loop, set:

```python
self.n_iter_ = iteration + 1
self.log_likelihood_ = float(log_like)
if iteration > 0 and abs(log_likes[-1] - log_likes[-2]) < self.tol:
    self.converged_ = True
    _log.info("Dawid-Skene converged at iteration %d", iteration)
    break
```

Add output columns:

```python
result["label_fusion_converged"] = bool(fuser.converged_)
result["label_fusion_iterations"] = int(fuser.n_iter_)
result["label_fusion_log_likelihood"] = float(fuser.log_likelihood_)
result["rater_clinical_proxy_observed"] = vote_df["rater_clinical_proxy_observed"].astype(bool).to_numpy()
```

- [ ] **Step 5: Verify**

Run:

```bash
uv run pytest -q tests/test_probabilistic_labels.py -o addopts=''
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/labels/probabilistic.py tests/test_probabilistic_labels.py
git commit -m "fix: treat unsupported weak label evidence as abstention"
```

---

## Phase 2: Scientific Validation Strategy

### Task 4: Add Temporal Group Cross-Validation

**Files:**
- Create: `src/plasmid_priority/modeling/temporal_cv.py`
- Modify: `src/plasmid_priority/modeling/module_a_support.py`
- Modify: `src/plasmid_priority/modeling/module_a.py`
- Test: `tests/test_modeling_temporal_cv.py`
- Test: `tests/test_modeling.py`

- [ ] **Step 1: Add failing temporal CV tests**

Create `tests/test_modeling_temporal_cv.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plasmid_priority.modeling.temporal_cv import temporal_group_folds


def test_temporal_group_folds_have_no_group_overlap() -> None:
    frame = pd.DataFrame(
        {
            "resolved_year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
            "backbone_id": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "spread_label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    folds = temporal_group_folds(
        frame,
        label_column="spread_label",
        year_column="resolved_year",
        group_column="backbone_id",
        n_splits=3,
    )

    assert folds
    for train_idx, test_idx in folds:
        train_groups = set(frame.iloc[train_idx]["backbone_id"])
        test_groups = set(frame.iloc[test_idx]["backbone_id"])
        assert train_groups.isdisjoint(test_groups)
        assert frame.iloc[train_idx]["resolved_year"].max() < frame.iloc[test_idx]["resolved_year"].min()


def test_temporal_group_folds_require_both_classes_in_test_window() -> None:
    frame = pd.DataFrame(
        {
            "resolved_year": [2010, 2011, 2012, 2013],
            "backbone_id": ["A", "B", "C", "D"],
            "spread_label": [0, 0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="both classes"):
        temporal_group_folds(frame, n_splits=2)
```

Run:

```bash
uv run pytest -q tests/test_modeling_temporal_cv.py -o addopts=''
```

Expected: fails because module does not exist.

- [ ] **Step 2: Implement temporal group splitter**

Create `src/plasmid_priority/modeling/temporal_cv.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from plasmid_priority.shared.temporal import coerce_required_years


def _require_binary_labels(labels: pd.Series) -> None:
    observed = sorted(pd.to_numeric(labels, errors="coerce").dropna().astype(int).unique().tolist())
    if observed != [0, 1]:
        raise ValueError("Temporal group folds require both classes in the eligible frame")


def temporal_group_folds(
    frame: pd.DataFrame,
    *,
    label_column: str = "spread_label",
    year_column: str = "resolved_year",
    group_column: str = "backbone_id",
    n_splits: int = 5,
    min_train_years: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    if group_column not in frame.columns:
        raise ValueError(f"Missing group column: {group_column}")

    years = coerce_required_years(frame, year_column, context="temporal_group_folds")
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    eligible = frame.loc[labels.notna()].copy()
    eligible["_year"] = years.loc[eligible.index].to_numpy(dtype=int)
    eligible["_label"] = labels.loc[eligible.index].astype(int)
    _require_binary_labels(eligible["_label"])

    unique_years = sorted(eligible["_year"].unique().tolist())
    if len(unique_years) < max(int(n_splits), 2):
        raise ValueError("Not enough distinct years for temporal group folds")

    candidate_test_years = unique_years[int(min_train_years):]
    if not candidate_test_years:
        raise ValueError("Temporal group folds require at least one test year")

    selected_test_years = candidate_test_years[-int(n_splits):]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test_year in selected_test_years:
        train_mask = eligible["_year"] < int(test_year)
        test_mask = eligible["_year"] == int(test_year)
        train_groups = set(eligible.loc[train_mask, group_column].astype(str))
        test_groups = set(eligible.loc[test_mask, group_column].astype(str))
        overlap = train_groups & test_groups
        if overlap:
            train_mask &= ~eligible[group_column].astype(str).isin(overlap)
        train_labels = eligible.loc[train_mask, "_label"]
        test_labels = eligible.loc[test_mask, "_label"]
        if sorted(train_labels.unique().tolist()) != [0, 1]:
            continue
        if sorted(test_labels.unique().tolist()) != [0, 1]:
            continue
        train_idx = frame.index.get_indexer(eligible.loc[train_mask].index)
        test_idx = frame.index.get_indexer(eligible.loc[test_mask].index)
        folds.append((train_idx.astype(int), test_idx.astype(int)))

    if not folds:
        raise ValueError("No temporal group folds with both classes in train and test windows")
    return folds
```

- [ ] **Step 3: Add strategy resolver while preserving existing splitter**

In `src/plasmid_priority/modeling/module_a_support.py`, add:

```python
def build_model_folds(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    strategy: str,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    normalized = str(strategy or "stratified_repeated").strip().lower()
    if normalized == "stratified_repeated":
        return _stratified_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    if normalized == "temporal_group":
        from plasmid_priority.modeling.temporal_cv import temporal_group_folds

        return temporal_group_folds(frame, n_splits=n_splits)
    raise ValueError(f"Unsupported split_strategy: {strategy}")
```

- [ ] **Step 4: Wire governance models to temporal group folds**

In `module_a.py`, choose split strategy:

```python
def _default_split_strategy(model_name: str, fit_kwargs: dict[str, object] | None) -> str:
    explicit = _fit_kwarg_str(fit_kwargs, "split_strategy", "")
    if explicit:
        return explicit
    return "temporal_group" if get_model_track(model_name) == "governance" else "stratified_repeated"
```

Replace direct `_stratified_folds(...)` calls in evaluation entry points with `build_model_folds(...)`, passing the eligible frame.

Record metric:

```python
metrics["split_strategy"] = split_strategy
```

- [ ] **Step 5: Verify**

Run:

```bash
uv run pytest -q tests/test_modeling_temporal_cv.py tests/test_modeling.py tests/test_rolling_origin_validation.py -o addopts=''
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/modeling/temporal_cv.py src/plasmid_priority/modeling/module_a_support.py src/plasmid_priority/modeling/module_a.py tests/test_modeling_temporal_cv.py tests/test_modeling.py
git commit -m "feat: add temporal group validation folds"
```

### Task 5: Add Leakage Audit Gate to Model Evaluation

**Files:**
- Modify: `src/plasmid_priority/features/temporal_leak.py`
- Modify: `src/plasmid_priority/modeling/experiment_gates.py`
- Test: `tests/test_leakage.py`
- Test: `tests/test_experiment_gates.py`

- [ ] **Step 1: Add tests for mandatory leakage gate**

Add to `tests/test_experiment_gates.py`:

```python
def test_experiment_gates_fail_on_temporal_leakage_rows() -> None:
    audit = pd.DataFrame(
        {
            "feature_name": ["future_country_count"],
            "leakage_status": ["fail"],
            "reason": ["uses post-split evidence"],
        }
    )

    result = evaluate_experiment_gates(
        metrics={"average_precision": 0.8},
        temporal_leakage_audit=audit,
    )

    assert result.status == "fail"
    assert "future_country_count" in result.reasons[0]
```

Run:

```bash
uv run pytest -q tests/test_experiment_gates.py -o addopts=''
```

Expected: fails until the gate accepts and evaluates leakage audit input.

- [ ] **Step 2: Implement leakage gate**

In `evaluate_experiment_gates()`, add an optional `temporal_leakage_audit` parameter and this logic:

```python
if temporal_leakage_audit is not None and not temporal_leakage_audit.empty:
    status_col = temporal_leakage_audit.get("leakage_status", pd.Series(dtype=str))
    failed = temporal_leakage_audit.loc[status_col.astype(str).str.lower().eq("fail")]
    if not failed.empty:
        names = failed.get("feature_name", pd.Series(dtype=str)).astype(str).head(5).tolist()
        reasons.append(f"Temporal leakage audit failed for features: {names}")
        status = "fail"
```

- [ ] **Step 3: Verify**

Run:

```bash
uv run pytest -q tests/test_leakage.py tests/test_experiment_gates.py -o addopts=''
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/plasmid_priority/features/temporal_leak.py src/plasmid_priority/modeling/experiment_gates.py tests/test_leakage.py tests/test_experiment_gates.py
git commit -m "feat: gate experiments on temporal leakage audit"
```

---

## Phase 3: Runtime Context and Configuration Contracts

### Task 6: Propagate ProjectContext Through Branch APIs

**Files:**
- Modify: `src/plasmid_priority/shared/branch_base.py`
- Modify: `src/plasmid_priority/shared/branches.py`
- Test: `tests/shared/test_branching.py`
- Test: `tests/test_hardening_data_root.py`

- [ ] **Step 1: Add failing context propagation test**

Add to `tests/shared/test_branching.py`:

```python
def test_branch_run_full_branch_passes_context_to_load_config(monkeypatch) -> None:
    observed = {}

    class ContextAwareBranch(GeoSpreadBranch):
        def load_config(self, context):
            observed["context"] = context
            return super().load_config(context)

    context = build_context(PROJECT_ROOT)
    branch = ContextAwareBranch()
    scored = pd.DataFrame(
        {
            "backbone_id": ["B1", "B2"],
            "spread_label": [0, 1],
            "resolved_year": [2014, 2016],
        }
    )

    with pytest.raises(Exception):
        branch.run_full_branch(
            scored,
            context,
            split_year=2015,
            horizon_years=5,
            n_splits=2,
            n_repeats=1,
        )

    assert observed["context"] is context
```

The test allows downstream model failure because its purpose is verifying context handoff.

- [ ] **Step 2: Update abstract signatures**

In `Branch`, change methods to accept context:

```python
def build_labels(..., context: ProjectContext, ...) -> pd.DataFrame: ...
def build_features(self, scored: pd.DataFrame, *, context: ProjectContext) -> pd.DataFrame: ...
def evaluate_branch(..., context: ProjectContext, ...) -> dict[str, Any]: ...
def build_predictions(..., context: ProjectContext, model_name: str) -> pd.DataFrame: ...
def calibrate(..., context: ProjectContext, method: str = "isotonic") -> tuple[pd.DataFrame, pd.DataFrame]: ...
```

Update `run_full_branch()` calls accordingly:

```python
labeled = self.build_labels(..., context=context, ...)
featured = self.build_features(labeled, context=context)
results = self.evaluate_branch(featured, context=context, ...)
predictions = self.build_predictions(featured, context=context, model_name=primary_model)
calibrated, calibration_summary = self.calibrate(predictions, context=context)
```

- [ ] **Step 3: Update concrete branches**

In each concrete branch, use:

```python
config = self.load_config(context)
contract = build_branch_input_contract(config=context.config)
```

Replace `self.load_config(None)` with `self.load_config(context)`.

- [ ] **Step 4: Verify**

Run:

```bash
uv run pytest -q tests/shared/test_branching.py tests/test_hardening_data_root.py tests/geo_spread/test_pipeline.py tests/bio_transfer/test_train.py tests/clinical_hazard/test_train.py -o addopts=''
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/plasmid_priority/shared/branch_base.py src/plasmid_priority/shared/branches.py tests/shared/test_branching.py tests/test_hardening_data_root.py
git commit -m "fix: propagate project context through branch execution"
```

### Task 7: Enforce Pydantic Configuration Validation

**Files:**
- Modify: `src/plasmid_priority/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add failing validation tests**

Append to `tests/test_config.py`:

```python
def test_pipeline_config_rejects_invalid_horizon() -> None:
    with pytest.raises(ValidationError, match="horizon_years"):
        ProjectConfig.model_validate({"pipeline": {"horizon_years": -1}})


def test_pipeline_config_requires_consensus_weights_sum_to_one() -> None:
    payload = {
        "pipeline": {
            "consensus_weights": {
                "geo": 0.9,
                "bio_transfer": 0.9,
                "clinical_hazard": 0.9,
            }
        }
    }

    with pytest.raises(ValidationError, match="consensus_weights"):
        ProjectConfig.model_validate(payload)


def test_pipeline_config_rejects_ood_thresholds_outside_unit_interval() -> None:
    with pytest.raises(ValidationError, match="ood_thresholds"):
        ProjectConfig.model_validate({"pipeline": {"ood_thresholds": {"support": 1.5}}})
```

Run:

```bash
uv run pytest -q tests/test_config.py -o addopts=''
```

Expected: fails because validators are not strict enough.

- [ ] **Step 2: Add validators**

In `PipelineConfig`, add:

```python
from pydantic import field_validator, model_validator

@field_validator("split_year")
@classmethod
def _validate_split_year(cls, value: int) -> int:
    if value < 1900 or value > 2100:
        raise ValueError("split_year must be between 1900 and 2100")
    return value

@field_validator("horizon_years", "min_new_host_genera_for_transfer", "min_new_host_families_for_transfer")
@classmethod
def _validate_non_negative_int(cls, value: int) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    return value

@field_validator("clinical_escalation_thresholds", "ood_thresholds")
@classmethod
def _validate_unit_interval_mapping(cls, value: dict[str, float]) -> dict[str, float]:
    for key, raw in value.items():
        numeric = float(raw)
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"{key} must be in [0, 1]")
    return value

@model_validator(mode="after")
def _validate_consensus_weights(self) -> "PipelineConfig":
    if self.consensus_weights:
        total = sum(float(value) for value in self.consensus_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("consensus_weights must sum to 1.0")
    return self
```

- [ ] **Step 3: Verify**

Run:

```bash
uv run pytest -q tests/test_config.py tests/test_protocol.py -o addopts=''
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/plasmid_priority/config.py tests/test_config.py
git commit -m "fix: validate scientific pipeline configuration"
```

---

## Phase 4: Cache and Reproducibility

### Task 8: Harden Model Cache Keys and Manifests

**Files:**
- Modify: `src/plasmid_priority/modeling/fold_cache.py`
- Modify: `src/plasmid_priority/modeling/oof_cache.py`
- Modify: `src/plasmid_priority/modeling/matrix_cache.py`
- Test: `tests/test_model_caches.py`

- [ ] **Step 1: Add failing cache-key tests**

Append to `tests/test_model_caches.py`:

```python
def test_universal_model_cache_key_changes_with_fit_config(tmp_path: Path) -> None:
    cache = UniversalModelCache(tmp_path)

    key_a = cache._get_key(
        "governance_linear",
        ("T_eff_norm",),
        0,
        "datahash",
        fit_config_hash="alpha1",
        protocol_hash="protocol",
        software_hash="software",
    )
    key_b = cache._get_key(
        "governance_linear",
        ("T_eff_norm",),
        0,
        "datahash",
        fit_config_hash="alpha2",
        protocol_hash="protocol",
        software_hash="software",
    )

    assert key_a != key_b
```

Run:

```bash
uv run pytest -q tests/test_model_caches.py -o addopts=''
```

Expected: fails because `_get_key` does not accept the new key fields.

- [ ] **Step 2: Extend cache key material**

Change `_get_key()` signature:

```python
def _get_key(
    self,
    model_name: str,
    features: tuple[str, ...],
    fold_idx: int,
    data_hash: str,
    *,
    fit_config_hash: str,
    protocol_hash: str,
    software_hash: str,
) -> str:
    key_payload = {
        "model_name": model_name,
        "features": list(features),
        "fold_idx": int(fold_idx),
        "data_hash": data_hash,
        "fit_config_hash": fit_config_hash,
        "protocol_hash": protocol_hash,
        "software_hash": software_hash,
    }
    key_content = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key_content.encode("utf-8")).hexdigest()[:32]
```

Do not sort `features`; feature order matters for model matrices.

- [ ] **Step 3: Add manifest sidecar for joblib artifacts**

On save:

```python
manifest_path = path.with_suffix(".manifest.json")
atomic_write_json(
    manifest_path,
    {
        "model_name": model_name,
        "features": list(features),
        "fold_idx": int(fold_idx),
        "data_hash": data_hash,
        "fit_config_hash": fit_config_hash,
        "protocol_hash": protocol_hash,
        "software_hash": software_hash,
    },
)
```

On load, require the manifest path to exist and match requested key material before calling `joblib.load()`.

- [ ] **Step 4: Verify**

Run:

```bash
uv run pytest -q tests/test_model_caches.py tests/test_workflow.py -o addopts=''
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/plasmid_priority/modeling/fold_cache.py src/plasmid_priority/modeling/oof_cache.py src/plasmid_priority/modeling/matrix_cache.py tests/test_model_caches.py
git commit -m "fix: include scientific context in model cache keys"
```

---

## Phase 5: Critical Coverage and CI

### Task 9: Add Critical-Path Coverage Gate

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_ci_workflow.py`

- [ ] **Step 1: Add workflow expectation test**

In `tests/test_ci_workflow.py`, add:

```python
def test_ci_runs_critical_path_coverage_gate() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "critical-path coverage" in workflow.lower()
    assert "tests/test_probabilistic_labels.py" in workflow
    assert "tests/test_modeling_temporal_cv.py" in workflow
```

- [ ] **Step 2: Add CI command**

In `.github/workflows/ci.yml`, add a step:

```yaml
- name: Critical-path coverage
  run: >
    uv run pytest
    tests/test_temporal_contracts.py
    tests/test_probabilistic_labels.py
    tests/test_modeling_temporal_cv.py
    tests/test_leakage.py
    tests/test_config.py
    tests/test_model_caches.py
    --cov=src/plasmid_priority/shared/temporal.py
    --cov=src/plasmid_priority/labels/probabilistic.py
    --cov=src/plasmid_priority/modeling/temporal_cv.py
    --cov=src/plasmid_priority/config.py
    --cov-report=term-missing
    --cov-fail-under=90
    -o addopts=''
```

- [ ] **Step 3: Tighten coverage omit list**

In `pyproject.toml`, remove these critical modules from `tool.coverage.run.omit`:

```toml
"src/plasmid_priority/features/temporal_leak.py",
"src/plasmid_priority/labels/*.py",
"src/plasmid_priority/modeling/calibration.py",
"src/plasmid_priority/modeling/folds.py",
"src/plasmid_priority/modeling/threshold.py",
```

If broad removal causes unrelated research modules to enter coverage, replace the broad `labels/*.py` omission with specific optional-heavy files:

```toml
"src/plasmid_priority/labels/coteaching.py",
"src/plasmid_priority/labels/counterfactual.py",
```

- [ ] **Step 4: Verify**

Run:

```bash
uv run pytest -q tests/test_ci_workflow.py -o addopts=''
uv run pytest tests/test_temporal_contracts.py tests/test_probabilistic_labels.py tests/test_modeling_temporal_cv.py tests/test_leakage.py tests/test_config.py tests/test_model_caches.py --cov=src/plasmid_priority/shared/temporal.py --cov=src/plasmid_priority/labels/probabilistic.py --cov=src/plasmid_priority/modeling/temporal_cv.py --cov=src/plasmid_priority/config.py --cov-report=term-missing --cov-fail-under=90 -o addopts=''
```

Expected: both commands pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml tests/test_ci_workflow.py
git commit -m "ci: add critical path coverage gate"
```

---

## Phase 6: Modularization Without API Breakage

### Task 10: Split Model Evaluation Internals

**Files:**
- Create: `src/plasmid_priority/modeling/preprocessing.py`
- Create: `src/plasmid_priority/modeling/oof_predictions.py`
- Create: `src/plasmid_priority/modeling/evaluation_metrics.py`
- Modify: `src/plasmid_priority/modeling/module_a.py`
- Test: `tests/test_modeling.py`

- [ ] **Step 1: Snapshot current public API**

Add to `tests/test_modeling.py`:

```python
def test_module_a_public_api_remains_importable_after_split() -> None:
    from plasmid_priority.modeling.module_a import (
        evaluate_feature_columns,
        evaluate_model_name,
        fit_full_model_predictions,
        get_module_a_model_names,
        run_module_a,
    )

    assert callable(evaluate_feature_columns)
    assert callable(evaluate_model_name)
    assert callable(fit_full_model_predictions)
    assert callable(get_module_a_model_names)
    assert callable(run_module_a)
```

- [ ] **Step 2: Move preprocessing functions**

Create `preprocessing.py` and move these functions from `module_a.py`:

```python
_knownness_score_series
_knownness_design_matrix
_fit_knownness_residualizer
_resolve_knownness_residualizer_alpha
_prepare_feature_matrices
_ensure_feature_columns
```

Keep compatibility imports in `module_a.py`:

```python
from plasmid_priority.modeling.preprocessing import (
    _ensure_feature_columns,
    _fit_knownness_residualizer,
    _knownness_score_series,
    _prepare_feature_matrices,
    _resolve_knownness_residualizer_alpha,
)
```

- [ ] **Step 3: Move OOF prediction functions**

Create `oof_predictions.py` and move:

```python
_oof_lightgbm_predictions_from_eligible
_oof_hybrid_predictions_from_eligible
_oof_predictions_from_eligible
_oof_predictions_with_detail_from_eligible
```

Import dependencies explicitly from support/preprocessing modules. Keep wrappers in `module_a.py` only if tests or scripts import private names.

- [ ] **Step 4: Move metric assembly**

Create `evaluation_metrics.py` and move:

```python
_evaluate_prediction_set
_top_k_precision_recall calls and validation metric assembly glue
```

Expose a single function:

```python
def build_model_metric_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    knownness: np.ndarray | None = None,
) -> dict[str, float]:
    ...
```

- [ ] **Step 5: Verify no public breakage**

Run:

```bash
uv run pytest -q tests/test_modeling.py tests/test_model_integrity.py tests/test_module_a_compute_tiers.py -o addopts=''
uv run mypy src/plasmid_priority/modeling --hide-error-context
```

Expected: tests pass; mypy has no new errors in touched modeling modules.

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/modeling/preprocessing.py src/plasmid_priority/modeling/oof_predictions.py src/plasmid_priority/modeling/evaluation_metrics.py src/plasmid_priority/modeling/module_a.py tests/test_modeling.py
git commit -m "refactor: split module A evaluation internals"
```

### Task 11: Split Feature Engineering Internals

**Files:**
- Create: `src/plasmid_priority/features/host_taxonomy.py`
- Create: `src/plasmid_priority/features/amr_signals.py`
- Create: `src/plasmid_priority/features/context_signals.py`
- Modify: `src/plasmid_priority/features/core.py`
- Test: `tests/test_features.py`
- Test: `tests/test_properties.py`

- [ ] **Step 1: Add import stability test**

Append to `tests/test_features.py`:

```python
def test_feature_core_public_api_remains_importable_after_split() -> None:
    from plasmid_priority.features.core import (
        build_backbone_feature_table,
        compute_feature_A,
        compute_feature_H,
        compute_feature_T,
    )

    assert callable(build_backbone_feature_table)
    assert callable(compute_feature_A)
    assert callable(compute_feature_H)
    assert callable(compute_feature_T)
```

- [ ] **Step 2: Move host taxonomy helpers**

Move these from `core.py` to `host_taxonomy.py`:

```python
HOST_TAXONOMY_LEVELS
_taxonomy_rank_lookup
_normalized_taxon_identifier
_host_taxonomy_signature_series
_pairwise_host_taxonomy_distance
```

Import them back into `core.py` to avoid behavior changes.

- [ ] **Step 3: Move AMR token logic**

Move these to `amr_signals.py`:

```python
NON_PUBLIC_HEALTH_AMR_TERMS
LAST_RESORT_AMR_TERMS
_normalize_amr_class_token
_is_public_health_amr_class
_is_last_resort_amr_class
_gene_family
```

- [ ] **Step 4: Move context term constants**

Move these to `context_signals.py`:

```python
CLINICAL_CONTEXT_TERMS
ENVIRONMENTAL_CONTEXT_TERMS
HOST_ASSOCIATED_CONTEXT_TERMS
FOOD_CONTEXT_TERMS
```

- [ ] **Step 5: Verify**

Run:

```bash
uv run pytest -q tests/test_features.py tests/test_properties.py tests/test_feature_pipeline_integration.py -o addopts=''
uv run mypy src/plasmid_priority/features --hide-error-context
```

Expected: tests pass; no new mypy errors in touched feature modules.

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/features/host_taxonomy.py src/plasmid_priority/features/amr_signals.py src/plasmid_priority/features/context_signals.py src/plasmid_priority/features/core.py tests/test_features.py
git commit -m "refactor: split feature engineering helpers"
```

---

## Phase 7: Documentation and Scientific Contracts

### Task 12: Add Architecture Decision Records

**Files:**
- Create: `docs/adr/0001-temporal-validation-policy.md`
- Create: `docs/adr/0002-governance-validation-strategy.md`
- Create: `docs/adr/0003-probabilistic-label-fusion.md`
- Modify: `mkdocs.yml`
- Test: `tests/test_scientific_contract.py`

- [ ] **Step 1: Add ADR files**

Create `docs/adr/0001-temporal-validation-policy.md`:

```markdown
# ADR 0001: Temporal Validation Policy

## Status
Accepted

## Context
The project evaluates retrospective genomic surveillance signals. Missing or invalid year metadata cannot be interpreted as pre-split evidence because that creates training contamination.

## Decision
Rows used for training-only backbone assignment, coherence, feature normalization, labels, or governance validation must have valid `resolved_year` values. Unsupported temporal evidence is either rejected at contract boundaries or represented as an explicit abstention/unknown diagnostic.

## Consequences
Pipelines fail earlier on malformed input. Demo data and tests must include valid years for all records used in supervised evaluation.
```

Create `docs/adr/0002-governance-validation-strategy.md`:

```markdown
# ADR 0002: Governance Validation Strategy

## Status
Accepted

## Context
Random stratified cross-validation estimates discriminative capacity but does not establish retrospective deployment validity.

## Decision
Governance-track claims use temporal/group-aware validation by default. Discovery-track models may report random stratified diagnostics when labeled as discovery-only.

## Consequences
Governance metrics may be lower but better aligned with operational deployment. Reports must state the split strategy next to each headline metric.
```

Create `docs/adr/0003-probabilistic-label-fusion.md`:

```markdown
# ADR 0003: Probabilistic Label Fusion

## Status
Accepted

## Context
Weak label raters have different support conditions. Encoding unsupported evidence as a negative observation biases Dawid-Skene reliability estimates.

## Decision
Unsupported rater evidence is encoded as abstention. Rater observations are validated against the latent class domain. Label fusion emits convergence and reliability diagnostics.

## Consequences
Soft labels become more conservative when evidence is sparse, and reports can distinguish low-risk evidence from missing evidence.
```

- [ ] **Step 2: Add ADRs to docs navigation**

In `mkdocs.yml`, add:

```yaml
- Architecture Decisions:
  - Temporal Validation Policy: adr/0001-temporal-validation-policy.md
  - Governance Validation Strategy: adr/0002-governance-validation-strategy.md
  - Probabilistic Label Fusion: adr/0003-probabilistic-label-fusion.md
```

- [ ] **Step 3: Verify docs build**

Run:

```bash
uv run mkdocs build --strict
```

Expected: docs build completes without warnings.

- [ ] **Step 4: Commit**

```bash
git add docs/adr/0001-temporal-validation-policy.md docs/adr/0002-governance-validation-strategy.md docs/adr/0003-probabilistic-label-fusion.md mkdocs.yml
git commit -m "docs: record scientific validation decisions"
```

---

## Phase 8: Final Quality Gate

### Task 13: Full Verification and Release Readiness

**Files:**
- Modify only files needed for failures discovered by this task.

- [ ] **Step 1: Run static checks**

```bash
uv run ruff check .
uv run mypy src
```

Expected: both commands pass.

- [ ] **Step 2: Run critical tests**

```bash
uv run pytest tests/test_temporal_contracts.py tests/test_probabilistic_labels.py tests/test_modeling_temporal_cv.py tests/test_leakage.py tests/test_config.py tests/test_model_caches.py -q -o addopts=''
```

Expected: all selected tests pass.

- [ ] **Step 3: Run full tests**

```bash
uv run pytest -q -o addopts=''
```

Expected: full suite passes. If runtime is high, record elapsed time in the PR body.

- [ ] **Step 4: Run coverage**

```bash
uv run pytest --cov=src/plasmid_priority --cov-report=term-missing --cov-fail-under=70 -o addopts=''
```

Expected: coverage is at least 70 overall and at least 90 for the critical-path coverage command from Task 9.

- [ ] **Step 5: Run workflow smoke**

```bash
PLASMID_PRIORITY_VALIDATION_SCREEN_SPLITS=1 \
PLASMID_PRIORITY_VALIDATION_SCREEN_REPEATS=1 \
PLASMID_PRIORITY_VALIDATION_FINALIST_SPLITS=1 \
PLASMID_PRIORITY_VALIDATION_FINALIST_REPEATS=1 \
uv run python scripts/run_workflow.py --mode demo-pipeline
```

Expected: demo pipeline completes and writes step result artifacts without temporal metadata failures.

- [ ] **Step 6: Build docs**

```bash
uv run mkdocs build --strict
```

Expected: docs build completes without warnings.

- [ ] **Step 7: Commit final fixes**

```bash
git add .
git commit -m "test: verify hardened scientific workflow"
```

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---:|---|
| Temporal/group CV lowers headline metrics | High | Report discovery and governance tracks separately; update narrative to treat governance as deployment evidence. |
| Existing demo/sample data has missing years | Medium | Patch `scripts/generate_sample_data.py` to emit valid `resolved_year` for supervised rows. |
| Branch context signature changes touch many tests | Medium | Update `Branch.run_full_branch()` first, then concrete branches, then CLIs. Keep compatibility wrappers only if external callers need them. |
| Coverage gate exposes optional dependency modules | Medium | Use targeted critical coverage command and avoid broad optional-heavy modules in the gate. |
| Cache key changes invalidate prior caches | Low | Document cache version bump and let artifacts rebuild. |

## Success Criteria

- [ ] Zero open P1/P2 review findings from the latest audit.
- [ ] `resolved_year` missingness cannot enter training-only assignment, coherence, labels, or governance validation.
- [ ] Governance model results include `split_strategy=temporal_group`.
- [ ] Probabilistic labels include convergence diagnostics and rater-observed diagnostics.
- [ ] Critical-path coverage gate passes at 90 percent.
- [ ] Full test suite, ruff, mypy, and docs build pass.
- [ ] ADRs document the scientific decisions behind temporal validation, governance validation, and weak-label fusion.

## Implementation Order

1. Task 1: Temporal metadata validation.
2. Task 2: Graph temporal versioning.
3. Task 3: Probabilistic label abstention.
4. Task 4: Temporal group CV.
5. Task 5: Leakage audit gate.
6. Task 6: Branch context propagation.
7. Task 7: Config validation.
8. Task 8: Cache hardening.
9. Task 9: Coverage and CI.
10. Task 10: Model modularization.
11. Task 11: Feature modularization.
12. Task 12: ADR documentation.
13. Task 13: Full verification.

This order keeps scientific correctness ahead of refactoring, so later modularization does not move broken behavior into cleaner files.

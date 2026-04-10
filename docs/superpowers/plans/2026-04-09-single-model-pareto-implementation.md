# Single-Model Pareto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-model selection program that jointly optimizes scientific reliability, predictive power, and compute efficiency under the user’s `4/4/2` weighting, then promote the best resulting model into the official report surface.

**Architecture:** Add a new weighted Pareto selection layer on top of the existing Module A universe. The implementation will derive a constrained family of compact model descendants, run a cheap screening pass across that family, run heavy official audits only on Pareto finalists, and end with one official model decision plus explicit rationale and compute evidence.

**Tech Stack:** Python 3.13, pandas, NumPy, existing Module A logistic / hybrid evaluation stack, existing reporting and validation pipeline, pytest, mypy, ruff.

---

## File Map

### New files

- `src/plasmid_priority/modeling/single_model_pareto.py`
  Weighted objective, failure severity scoring, compute scoring, candidate shortlist logic.
- `tests/test_single_model_pareto.py`
  Unit tests for weighted objective, Pareto filtering, and decision ranking.

### Modified files

- `src/plasmid_priority/protocol.py`
  Canonical ownership for single-model objective weights and official selection policy snapshot fields.
- `src/plasmid_priority/modeling/module_a_support.py`
  Parent-model registry and bounded derivation config.
- `src/plasmid_priority/modeling/module_a.py`
  Candidate derivation helpers and cheap evaluation utilities.
- `src/plasmid_priority/reporting/model_audit.py`
  Finalist-heavy audit, official single-model decision table, failure-severity comparison.
- `src/plasmid_priority/reporting/narratives.py`
  Report-facing Pareto rationale and final single-model explanation.
- `scripts/21_run_validation.py`
  Stage A screening + Stage B finalist audit artifacts.
- `scripts/24_build_reports.py`
  Consume final single-model decision artifacts and reflect them in reports.
- `tests/test_protocol.py`
  Protocol drift tests for weights and selection-policy ownership.
- `tests/test_model_audit.py`
  Finalist-heavy audit and single-model decision tests.
- `tests/test_reporting.py`
  Reporting integration tests for new single-model rationale.
- `tests/test_workflow.py`
  Workflow coverage for new artifacts when `21_run_validation` and `24_build_reports` run.

---

### Task 1: Canonicalize the weighted objective in protocol

**Files:**
- Modify: `src/plasmid_priority/protocol.py`
- Test: `tests/test_protocol.py`

- [ ] **Step 1: Write the failing protocol test**

```python
def test_protocol_exposes_single_model_selection_weights() -> None:
    protocol = ScientificProtocol.from_config(
        {
            "pipeline": {"split_year": 2015, "min_new_countries_for_spread": 3},
            "models": {
                "primary_model_name": "discovery_12f_source",
                "primary_model_fallback": "parsimonious_priority",
                "conservative_model_name": "parsimonious_priority",
                "governance_model_name": "phylo_support_fusion_priority",
                "governance_model_fallback": "support_synergy_priority",
                "core_model_names": [
                    "discovery_12f_source",
                    "phylo_support_fusion_priority",
                    "baseline_both",
                ],
                "research_model_names": [],
                "ablation_model_names": [],
            },
        }
    )

    assert protocol.single_model_objective_weights == {
        "reliability": 0.4,
        "predictive_power": 0.4,
        "compute_efficiency": 0.2,
    }
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python3 -m pytest tests/test_protocol.py -k single_model_selection_weights -q
```

Expected: FAIL because `single_model_objective_weights` does not exist yet.

- [ ] **Step 3: Add protocol-owned weights and policy snapshot**

```python
DEFAULT_SINGLE_MODEL_OBJECTIVE_WEIGHTS: dict[str, float] = {
    "reliability": 0.4,
    "predictive_power": 0.4,
    "compute_efficiency": 0.2,
}

@property
def single_model_objective_weights(self) -> dict[str, float]:
    return dict(DEFAULT_SINGLE_MODEL_OBJECTIVE_WEIGHTS)
```

Also include the weights in `build_protocol_snapshot(...)` so provenance reflects the decision policy.

- [ ] **Step 4: Re-run the protocol test**

Run:

```bash
python3 -m pytest tests/test_protocol.py -k single_model_selection_weights -q
```

Expected: PASS

- [ ] **Step 5: Run the touched protocol suite**

Run:

```bash
python3 -m pytest tests/test_protocol.py tests/test_report_provenance.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/protocol.py tests/test_protocol.py tests/test_report_provenance.py
git commit -m "Define canonical single-model objective weights in protocol

Constraint: Weighted objective must be provenance-visible and protocol-owned
Rejected: Hardcode weights inside screening logic | would create policy drift
Confidence: high
Scope-risk: narrow
Directive: Do not move official single-model weights outside protocol without re-auditing provenance
Tested: tests/test_protocol.py, tests/test_report_provenance.py
Not-tested: full pipeline
"
```

### Task 2: Build deterministic weighted Pareto utilities

**Files:**
- Create: `src/plasmid_priority/modeling/single_model_pareto.py`
- Create: `tests/test_single_model_pareto.py`

- [ ] **Step 1: Write the failing unit tests for scoring and shortlist logic**

```python
def test_weighted_objective_prefers_better_reliability_when_power_is_close() -> None:
    candidates = pd.DataFrame(
        [
            {"model_name": "a", "reliability_score": 0.82, "predictive_power_score": 0.80, "compute_efficiency_score": 0.40},
            {"model_name": "b", "reliability_score": 0.70, "predictive_power_score": 0.83, "compute_efficiency_score": 0.40},
        ]
    )

    ranked = rank_single_model_candidates(candidates)

    assert ranked.iloc[0]["model_name"] == "a"


def test_failure_severity_penalizes_multi-guardrail_failures() -> None:
    scorecard = pd.DataFrame(
        [
            {
                "model_name": "a",
                "scientific_acceptance_status": "fail",
                "scientific_acceptance_failed_criteria": "fail:matched_knownness",
                "knownness_matched_gap": -0.010,
                "source_holdout_gap": -0.001,
                "blocked_holdout_raw_ece": 0.03,
            },
            {
                "model_name": "b",
                "scientific_acceptance_status": "fail",
                "scientific_acceptance_failed_criteria": "fail:matched_knownness,source_holdout,calibration",
                "knownness_matched_gap": -0.040,
                "source_holdout_gap": -0.050,
                "blocked_holdout_raw_ece": 0.12,
            },
        ]
    )

    enriched = add_failure_severity(scorecard)

    assert float(enriched.loc[enriched["model_name"] == "a", "failure_severity"].iloc[0]) < float(
        enriched.loc[enriched["model_name"] == "b", "failure_severity"].iloc[0]
    )
```

- [ ] **Step 2: Run the new test file to verify RED**

Run:

```bash
python3 -m pytest tests/test_single_model_pareto.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement minimal weighted objective utilities**

```python
OBJECTIVE_WEIGHTS = {
    "reliability_score": 0.4,
    "predictive_power_score": 0.4,
    "compute_efficiency_score": 0.2,
}


def add_weighted_objective(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["weighted_objective_score"] = (
        0.4 * working["reliability_score"].astype(float)
        + 0.4 * working["predictive_power_score"].astype(float)
        + 0.2 * working["compute_efficiency_score"].astype(float)
    )
    return working
```

Also implement:

- `add_failure_severity(...)`
- `rank_single_model_candidates(...)`
- `build_pareto_shortlist(...)`

- [ ] **Step 4: Run the new test file to verify GREEN**

Run:

```bash
python3 -m pytest tests/test_single_model_pareto.py -q
```

Expected: PASS

- [ ] **Step 5: Run lint and typecheck on the new module**

Run:

```bash
python3 -m ruff check src/plasmid_priority/modeling/single_model_pareto.py tests/test_single_model_pareto.py
python3 -m mypy src/plasmid_priority/modeling/single_model_pareto.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/modeling/single_model_pareto.py tests/test_single_model_pareto.py
git commit -m "Add weighted Pareto utilities for single-model selection

Constraint: Objective must match user weighting of reliability 4, power 4, compute 2
Rejected: Pure ROC AUC ranking | ignores guardrail and cost tradeoffs
Confidence: high
Scope-risk: narrow
Directive: Keep failure severity explicit and auditable; do not hide it behind one opaque score
Tested: tests/test_single_model_pareto.py, ruff, mypy
Not-tested: pipeline integration
"
```

### Task 3: Build a constrained candidate family and cheap screening pass

**Files:**
- Modify: `src/plasmid_priority/modeling/module_a_support.py`
- Modify: `src/plasmid_priority/modeling/module_a.py`
- Modify: `scripts/21_run_validation.py`
- Test: `tests/test_model_audit.py`
- Test: `tests/test_single_model_pareto.py`

- [ ] **Step 1: Write the failing candidate-derivation tests**

```python
def test_build_single_model_candidate_family_returns_parent_derived_variants() -> None:
    family = build_single_model_candidate_family()

    names = {row["model_name"] for row in family.to_dict("records")}
    assert "discovery_12f_source" in names
    assert any(name.startswith("discovery_12f_source__pruned") for name in names)
    assert any(name.startswith("phylo_support_fusion_priority__") for name in names)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_single_model_pareto.py -k candidate_family -q
```

Expected: FAIL because the derivation helper does not exist.

- [ ] **Step 3: Add bounded parent-derived family construction**

```python
PARETO_PARENT_MODELS = (
    "phylo_support_fusion_priority",
    "discovery_12f_source",
    "support_synergy_priority",
    "knownness_robust_priority",
    "parsimonious_priority",
)


def build_single_model_candidate_family() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ...
    return pd.DataFrame(rows)
```

Each derived candidate row should include:

- `model_name`
- `parent_model_name`
- `feature_set`
- `feature_count`
- `candidate_kind`

- [ ] **Step 4: Add cheap screening evaluation into `21_run_validation.py`**

Create a Stage A artifact:

- `data/analysis/single_model_pareto_screen.tsv`

Minimum columns:

- `model_name`
- `parent_model_name`
- `feature_count`
- `roc_auc`
- `average_precision`
- `knownness_matched_gap`
- `source_holdout_gap`
- `blocked_holdout_weighted_roc_auc`
- `screen_fit_seconds`
- `predictive_power_score`
- `reliability_score`
- `compute_efficiency_score`
- `weighted_objective_score`

- [ ] **Step 5: Re-run targeted tests**

Run:

```bash
python3 -m pytest tests/test_single_model_pareto.py tests/test_model_audit.py -q
```

Expected: PASS

- [ ] **Step 6: Run the validation script to produce the screening artifact**

Run:

```bash
python3 scripts/21_run_validation.py
```

Expected: `data/analysis/single_model_pareto_screen.tsv` exists and contains candidate rows.

- [ ] **Step 7: Commit**

```bash
git add src/plasmid_priority/modeling/module_a_support.py src/plasmid_priority/modeling/module_a.py src/plasmid_priority/modeling/single_model_pareto.py scripts/21_run_validation.py tests/test_single_model_pareto.py tests/test_model_audit.py
git commit -m "Add constrained Pareto candidate family and cheap screening pass

Constraint: Heavy audits must not run across the full model universe
Rejected: Exhaustive all-model heavy audit | compute cost too high for iterative search
Confidence: medium
Scope-risk: moderate
Directive: Candidate derivation must stay bounded and parent-derived; do not create untraceable model names
Tested: targeted pytest, 21_run_validation.py
Not-tested: report integration
"
```

### Task 4: Add finalist-heavy audit and single official decision table

**Files:**
- Modify: `src/plasmid_priority/reporting/model_audit.py`
- Modify: `scripts/21_run_validation.py`
- Test: `tests/test_model_audit.py`

- [ ] **Step 1: Write the failing finalist-decision tests**

```python
def test_build_single_model_official_decision_prefers_lower_failure_severity_before_auc() -> None:
    finalists = pd.DataFrame(
        [
            {
                "model_name": "high_auc_fail_hard",
                "weighted_objective_score": 0.84,
                "scientific_acceptance_status": "fail",
                "failure_severity": 0.70,
                "roc_auc": 0.84,
                "compute_efficiency_score": 0.30,
            },
            {
                "model_name": "slightly_lower_auc_fail_soft",
                "weighted_objective_score": 0.83,
                "scientific_acceptance_status": "fail",
                "failure_severity": 0.18,
                "roc_auc": 0.82,
                "compute_efficiency_score": 0.45,
            },
        ]
    )

    decision = build_single_model_official_decision(finalists)

    assert decision.iloc[0]["official_model_name"] == "slightly_lower_auc_fail_soft"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_model_audit.py -k single_model_official_decision -q
```

Expected: FAIL because the decision helper does not exist.

- [ ] **Step 3: Add Stage B finalist audit**

Add helpers that:

- choose Pareto finalists from `single_model_pareto_screen.tsv`
- run heavy official audit only for those finalists
- compute `failure_severity`
- write:
  - `data/analysis/single_model_pareto_finalists.tsv`
  - `data/analysis/single_model_official_decision.tsv`

Decision row must include:

- `official_model_name`
- `decision_reason`
- `scientific_acceptance_status`
- `scientific_acceptance_failed_criteria`
- `failure_severity`
- `roc_auc`
- `average_precision`
- `weighted_objective_score`
- `screen_fit_seconds`

- [ ] **Step 4: Re-run targeted audit tests**

Run:

```bash
python3 -m pytest tests/test_model_audit.py tests/test_single_model_pareto.py -q
```

Expected: PASS

- [ ] **Step 5: Re-run validation to generate finalists and decision artifacts**

Run:

```bash
python3 scripts/21_run_validation.py
```

Expected: both finalist artifacts exist and identify one official model.

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/reporting/model_audit.py scripts/21_run_validation.py tests/test_model_audit.py tests/test_single_model_pareto.py
git commit -m "Add finalist-heavy Pareto audit and single official model decision

Constraint: Final system must end in one official model, not a multi-model committee
Rejected: Pick highest weighted score regardless of failure pattern | not scientifically defensible
Confidence: medium
Scope-risk: moderate
Directive: Final decision ordering is pass -> lower failure severity -> higher power -> lower cost
Tested: targeted pytest, 21_run_validation.py
Not-tested: final report surface
"
```

### Task 5: Wire the single-model decision into reports and release outputs

**Files:**
- Modify: `scripts/24_build_reports.py`
- Modify: `src/plasmid_priority/reporting/narratives.py`
- Modify: `tests/test_reporting.py`
- Modify: `tests/test_workflow.py`

- [ ] **Step 1: Write the failing reporting tests**

```python
def test_headline_summary_includes_single_model_pareto_rationale() -> None:
    decision = pd.DataFrame(
        [
            {
                "official_model_name": "discovery_12f_source__pruned_v2",
                "decision_reason": "lowest_failure_severity_with_competitive_auc",
                "scientific_acceptance_status": "fail",
                "scientific_acceptance_failed_criteria": "fail:matched_knownness",
                "weighted_objective_score": 0.81,
            }
        ]
    )

    markdown = build_headline_validation_summary_markdown(
        ...,
        single_model_official_decision=decision,
    )

    assert "lowest_failure_severity_with_competitive_auc" in markdown
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_reporting.py -k pareto_rationale -q
```

Expected: FAIL because the report builders do not accept the new artifact yet.

- [ ] **Step 3: Add report consumption for the decision artifact**

`24_build_reports.py` should:

- require `single_model_official_decision.tsv` if the Pareto program is enabled
- pass the artifact to narrative builders
- write a compact report-facing table:
  - `reports/core_tables/single_model_official_decision.tsv`

`narratives.py` should surface:

- chosen official model
- weighted-objective rationale
- failure severity
- compute note

- [ ] **Step 4: Re-run reporting tests**

Run:

```bash
python3 -m pytest tests/test_reporting.py tests/test_workflow.py -q
```

Expected: PASS

- [ ] **Step 5: Rebuild reports and release bundle**

Run:

```bash
python3 scripts/24_build_reports.py
python3 scripts/25_export_tubitak_summary.py
python3 scripts/28_build_release_bundle.py
```

Expected: PASS and new single-model artifacts appear in reports and bundle.

- [ ] **Step 6: Commit**

```bash
git add scripts/24_build_reports.py src/plasmid_priority/reporting/narratives.py tests/test_reporting.py tests/test_workflow.py reports/
git commit -m "Expose Pareto-selected official model in reports and release bundle

Constraint: External surface must still read as one official model
Rejected: Publish raw shortlist without final decision | weakens external clarity
Confidence: medium
Scope-risk: moderate
Directive: Reports must explain why the official model won, not just state the winner
Tested: reporting/workflow pytest, report rebuild, bundle rebuild
Not-tested: downstream consumer tooling outside repo
"
```

### Task 6: Promote the best model only if it beats the incumbent on the weighted objective

**Files:**
- Modify: `config.yaml`
- Modify: `src/plasmid_priority/protocol.py`
- Test: `tests/test_protocol.py`
- Test: `tests/test_model_audit.py`

- [ ] **Step 1: Write the failing promotion-policy test**

```python
def test_single_model_promotion_requires_weighted_improvement_or_lower_failure_severity() -> None:
    incumbent = {
        "model_name": "discovery_12f_source",
        "weighted_objective_score": 0.78,
        "failure_severity": 0.30,
    }
    challenger = {
        "model_name": "discovery_12f_source__pruned_v2",
        "weighted_objective_score": 0.79,
        "failure_severity": 0.18,
    }

    decision = should_promote_single_model(challenger, incumbent)

    assert decision is True
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_protocol.py tests/test_model_audit.py -k promote_single_model -q
```

Expected: FAIL because the promotion helper does not exist.

- [ ] **Step 3: Implement promotion policy and update config if warranted**

Rules:

- promote automatically if challenger passes and incumbent does not
- else promote if challenger has lower failure severity and no material predictive collapse
- else keep incumbent

If a new model wins, update:

- `config.yaml` `primary_model_name`
- optional `governance_model_name` only if the new decision artifacts support it

- [ ] **Step 4: Re-run promotion tests**

Run:

```bash
python3 -m pytest tests/test_protocol.py tests/test_model_audit.py -k promote_single_model -q
```

Expected: PASS

- [ ] **Step 5: Re-run the full quality gate**

Run:

```bash
make quality
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add config.yaml src/plasmid_priority/protocol.py src/plasmid_priority/reporting/model_audit.py tests/test_protocol.py tests/test_model_audit.py
git commit -m "Promote Pareto-optimal single model when it beats the incumbent

Constraint: Official model changes must remain protocol-visible and audit-justified
Rejected: Manual winner selection after seeing metrics | not reproducible
Confidence: medium
Scope-risk: broad
Directive: Never change official model names without preserving the decision artifact that justified promotion
Tested: targeted pytest, make quality
Not-tested: external scientific review beyond repo evidence
"
```

### Task 7: Final evidence capture and AUC reporting

**Files:**
- Modify: `reports/`
- Test: none beyond final verification

- [ ] **Step 1: Capture final benchmark metrics**

Run:

```bash
python3 -c 'import pandas as pd; df=pd.read_csv("reports/core_tables/model_selection_summary.tsv", sep="\t"); print(df.to_string(index=False))'
```

Expected: the final official model, incumbent comparison, AUC, AP, and rationale are visible.

- [ ] **Step 2: Capture final official decision artifact**

Run:

```bash
python3 -c 'import pandas as pd; df=pd.read_csv("reports/core_tables/single_model_official_decision.tsv", sep="\t"); print(df.to_string(index=False))'
```

Expected: exactly one official decision row.

- [ ] **Step 3: Capture release bundle verification**

Run:

```bash
python3 scripts/26_run_tests_or_smoke.py --with-tests
```

Expected: PASS

- [ ] **Step 4: Prepare final summary**

The final close-out must report:

- new official model name
- ROC AUC
- AP
- failure / pass status
- failure severity vs incumbent
- compute savings or compute concentration improvement
- whether the repo remained green under `make quality`

- [ ] **Step 5: Commit final artifacts if policy requires generated outputs**

```bash
git add reports/ data/tmp/logs/
git commit -m "Refresh final Pareto-selected single-model evidence bundle

Constraint: Final claim must be backed by regenerated report and smoke artifacts
Rejected: Report AUCs from stale artifacts | scientifically invalid
Confidence: high
Scope-risk: narrow
Directive: Do not quote new AUC values unless they come from refreshed post-change artifacts
Tested: make quality, smoke, regenerated reports
Not-tested: external downstream consumers
"
```

---

## Self-Review

### Spec coverage

- Weighted 4/4/2 objective: covered in Tasks 1 and 2
- Single official model decision: covered in Tasks 4 and 6
- Constrained family + compute-aware search: covered in Task 3
- Heavy audits only on finalists: covered in Task 4
- Reporting and release rationale: covered in Task 5
- Final evidence and AUC reporting: covered in Task 7

### Placeholder scan

- No `TBD` / `TODO`
- All code-affecting tasks include explicit file paths, commands, and code snippets

### Type consistency

- Uses consistent artifact names:
  - `single_model_pareto_screen.tsv`
  - `single_model_pareto_finalists.tsv`
  - `single_model_official_decision.tsv`
- Uses one decision rule consistently:
  - `pass -> lower failure severity -> higher power -> lower cost`


# Joint Data + Model Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a canonical, auditable data foundation and a guardrail-aware single-model search program that can produce one official model with scientific acceptance `pass` while keeping ROC AUC / AP competitive with the current benchmark.

**Architecture:** Add a canonical data-quality layer between the current harmonize/dedup/backbone pipeline and the official modeling surface. The program will emit one cleaned backbone-centric training surface, score it with explicit duplicate/source/missingness diagnostics, derive a bounded family of knownness-robust candidate models on top of that surface, and then reuse the existing Pareto selector plus official audit stack to choose one final model.

**Tech Stack:** Python 3.12+, pandas, NumPy, optional `duckdb` in the analysis extra, existing harmonization/dedup/backbone pipeline, existing Module A modeling stack, pytest, mypy, ruff.

---

## File Map

### New files

- `src/plasmid_priority/data_quality/__init__.py`
  Public exports for canonical surface and audit helpers.
- `src/plasmid_priority/data_quality/canonical_surface.py`
  Canonical backbone-centric training surface assembly, source/region normalization, duplicate-group preservation, and optional DuckDB-backed joins.
- `src/plasmid_priority/data_quality/audits.py`
  Data-quality gates for duplicate pressure, source concentration, metadata missingness, temporal coverage, and backbone assignment confidence.
- `tests/test_canonical_data_surface.py`
  Unit tests for canonical surface assembly and canonical column invariants.
- `tests/test_data_quality_audits.py`
  Unit tests for duplicate/source/missingness/temporal audit summaries.

### Modified files

- `pyproject.toml`
  Add `duckdb` to the `analysis` extra only, not to the core runtime dependencies.
- `src/plasmid_priority/harmonize/records.py`
  Export richer canonical source, region, and year-confidence fields needed downstream.
- `src/plasmid_priority/dedup/canonicalize.py`
  Preserve canonical duplicate-group metadata required by the training surface.
- `src/plasmid_priority/features/core.py`
  Teach `build_backbone_table(...)` to accept canonical quality columns and keep them in the backbone-centric table.
- `src/plasmid_priority/validation/missingness.py`
  Reuse the missingness summary machinery inside the new data-quality audit layer.
- `src/plasmid_priority/modeling/module_a_support.py`
  Add a bounded candidate-family registry for cleaned, knownness-robust, source-robust descendants.
- `src/plasmid_priority/modeling/module_a.py`
  Build candidate features from the canonical surface and score new descendants without exploding compute.
- `src/plasmid_priority/reporting/model_audit.py`
  Add data-quality gate status to the single-model decision path and expose new failure reasons when the surface itself is untrustworthy.
- `scripts/04_harmonize_metadata.py`
  Write the extra harmonized canonicalization columns.
- `scripts/05_deduplicate.py`
  Write duplicate-group evidence columns required downstream.
- `scripts/14_build_backbone_table.py`
  Build and persist the canonical training surface before backbone scoring.
- `scripts/15_normalize_and_score.py`
  Consume the canonical surface rather than the looser pre-cleaned table.
- `scripts/21_run_validation.py`
  Run Pareto screening on the canonical surface and write data-quality artifacts alongside model-selection artifacts.
- `scripts/24_build_reports.py`
  Surface cleaned-data gate status in the headline and single-model report narrative.
- `scripts/28_build_release_bundle.py`
  Ship canonical data-quality summaries in the release bundle.
- `tests/test_harmonize.py`
  Add canonical source/region/year-confidence coverage.
- `tests/test_dedup.py`
  Add duplicate-group metadata coverage.
- `tests/test_missingness_audit.py`
  Lock reuse behavior for canonical missingness summaries.
- `tests/test_single_model_pareto.py`
  Extend candidate-family and screening tests for the cleaned-surface variants.
- `tests/test_model_audit.py`
  Lock single-model decisions when data-quality gates fail or pass.
- `tests/test_reporting.py`
  Verify report surfaces mention data-quality status and the final official reason honestly.
- `tests/test_workflow.py`
  Ensure validation/report/release workflows retain the new artifacts.

---

### Task 1: Create the canonical data-quality package and optional DuckDB integration

**Files:**
- Create: `src/plasmid_priority/data_quality/__init__.py`
- Create: `src/plasmid_priority/data_quality/canonical_surface.py`
- Modify: `pyproject.toml`
- Test: `tests/test_canonical_data_surface.py`

- [ ] **Step 1: Write the failing canonical-surface tests**

```python
import pandas as pd

from plasmid_priority.data_quality.canonical_surface import (
    build_canonical_training_surface,
)


def test_build_canonical_training_surface_keeps_duplicate_and_source_fields() -> None:
    harmonized = pd.DataFrame(
        [
            {
                "plasmid_id": "p1",
                "source": "refseq",
                "country": "Turkiye",
                "collection_year": "2019",
            }
        ]
    )
    deduplicated = pd.DataFrame(
        [
            {
                "plasmid_id": "p1",
                "canonical_id": "c1",
                "duplicate_group_size": 3,
                "is_canonical_representative": True,
            }
        ]
    )
    backbone = pd.DataFrame(
        [
            {
                "plasmid_id": "p1",
                "backbone_id": "b1",
                "novelty_score": 0.62,
            }
        ]
    )

    result = build_canonical_training_surface(
        harmonized=harmonized,
        deduplicated=deduplicated,
        backbone=backbone,
    )

    assert list(result["canonical_id"]) == ["c1"]
    assert list(result["duplicate_group_size"]) == [3]
    assert list(result["canonical_source"]) == ["refseq"]
    assert list(result["macro_region"]) == ["western_asia"]
    assert list(result["year_confidence"]) == ["exact"]
```

- [ ] **Step 2: Run the new tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_canonical_data_surface.py -q
```

Expected: FAIL because the `data_quality` package and builder do not exist yet.

- [ ] **Step 3: Add the minimal package and optional DuckDB analysis dependency**

```python
# src/plasmid_priority/data_quality/__init__.py
from .canonical_surface import build_canonical_training_surface

__all__ = ["build_canonical_training_surface"]
```

```python
# src/plasmid_priority/data_quality/canonical_surface.py
import pandas as pd


def build_canonical_training_surface(
    *,
    harmonized: pd.DataFrame,
    deduplicated: pd.DataFrame,
    backbone: pd.DataFrame,
) -> pd.DataFrame:
    merged = harmonized.merge(deduplicated, on="plasmid_id", how="inner").merge(
        backbone,
        on="plasmid_id",
        how="inner",
    )
    merged["canonical_source"] = merged["source"].astype(str).str.strip().str.lower()
    merged["macro_region"] = merged["country"].map({"Turkiye": "western_asia"}).fillna("unknown")
    merged["year_confidence"] = merged["collection_year"].astype(str).map(
        lambda value: "exact" if value.isdigit() and len(value) == 4 else "missing"
    )
    return merged
```

```toml
# pyproject.toml
[project.optional-dependencies]
analysis = [
  "duckdb>=1.2",
  "interpret>=0.7.8",
  "matplotlib>=3.10",
  "pyarrow>=16",
  "snakemake>=8",
]
```

Keep `duckdb` in `analysis` only so the core runtime remains lean.

- [ ] **Step 4: Re-run the canonical-surface tests**

Run:

```bash
python3 -m pytest tests/test_canonical_data_surface.py -q
```

Expected: PASS

- [ ] **Step 5: Run lint and typecheck on the new package**

Run:

```bash
python3 -m ruff check src/plasmid_priority/data_quality/__init__.py src/plasmid_priority/data_quality/canonical_surface.py tests/test_canonical_data_surface.py
python3 -m mypy src/plasmid_priority/data_quality/canonical_surface.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/plasmid_priority/data_quality/__init__.py src/plasmid_priority/data_quality/canonical_surface.py tests/test_canonical_data_surface.py
git commit -m "Create a canonical data-quality surface for official modeling

Constraint: Core runtime must stay lightweight while analysis workflows can use relational audits
Rejected: Make DuckDB a hard dependency | unnecessary for non-analysis runtime paths
Confidence: high
Scope-risk: moderate
Directive: Keep canonical surface assembly deterministic and backbone-centric; do not reintroduce ad hoc joins in downstream scripts
Tested: tests/test_canonical_data_surface.py, ruff, mypy
Not-tested: pipeline-wide rebuild
"
```

### Task 2: Enrich harmonize and dedup outputs so the canonical surface carries real scientific provenance

**Files:**
- Modify: `src/plasmid_priority/harmonize/records.py`
- Modify: `src/plasmid_priority/dedup/canonicalize.py`
- Modify: `scripts/04_harmonize_metadata.py`
- Modify: `scripts/05_deduplicate.py`
- Test: `tests/test_harmonize.py`
- Test: `tests/test_dedup.py`

- [ ] **Step 1: Write the failing provenance tests**

```python
def test_build_harmonized_plasmid_table_adds_canonical_source_and_year_confidence() -> None:
    result = build_harmonized_plasmid_table(
        plsdb_metadata=pd.DataFrame(
            [{"plasmid_id": "p1", "source": "RefSeq", "country": "Turkey", "collection_year": "2018"}]
        ),
        biosample_metadata=pd.DataFrame(),
        typing_hits=pd.DataFrame(),
        plasmidfinder_summary=pd.DataFrame(),
    )

    row = result.iloc[0]
    assert row["canonical_source"] == "refseq"
    assert row["macro_region"] == "western_asia"
    assert row["year_confidence"] == "exact"


def test_annotate_canonical_ids_preserves_duplicate_group_rank() -> None:
    frame = pd.DataFrame(
        [
            {"plasmid_id": "p1", "sequence_hash": "h1"},
            {"plasmid_id": "p2", "sequence_hash": "h1"},
        ]
    )

    result = annotate_canonical_ids(frame)

    assert set(result["duplicate_group_size"]) == {2}
    assert set(result["duplicate_group_rank"]) == {1, 2}
```

- [ ] **Step 2: Run the focused tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_harmonize.py tests/test_dedup.py -q
```

Expected: FAIL because the new provenance fields are missing.

- [ ] **Step 3: Add canonical source, macro-region, year-confidence, and duplicate-group rank fields**

```python
# src/plasmid_priority/harmonize/records.py
def _normalize_source_label(value: object) -> str:
    return str(value or "").strip().lower()


def _year_confidence(value: object) -> str:
    text = str(value or "").strip()
    return "exact" if text.isdigit() and len(text) == 4 else "missing"


harmonized["canonical_source"] = harmonized["source"].map(_normalize_source_label)
harmonized["macro_region"] = harmonized["country"].map(COUNTRY_TO_MACRO_REGION).fillna("unknown")
harmonized["year_confidence"] = harmonized["collection_year"].map(_year_confidence)
```

```python
# src/plasmid_priority/dedup/canonicalize.py
working["duplicate_group_rank"] = (
    working.sort_values(["canonical_id", "plasmid_id"])
    .groupby("canonical_id")
    .cumcount()
    .add(1)
)
```

Make `scripts/04_harmonize_metadata.py` and `scripts/05_deduplicate.py` pass these new columns through unchanged.

- [ ] **Step 4: Re-run the focused tests**

Run:

```bash
python3 -m pytest tests/test_harmonize.py tests/test_dedup.py -q
```

Expected: PASS

- [ ] **Step 5: Run lint and typecheck on the touched provenance files**

Run:

```bash
python3 -m ruff check src/plasmid_priority/harmonize/records.py src/plasmid_priority/dedup/canonicalize.py scripts/04_harmonize_metadata.py scripts/05_deduplicate.py tests/test_harmonize.py tests/test_dedup.py
python3 -m mypy src/plasmid_priority/harmonize/records.py src/plasmid_priority/dedup/canonicalize.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/harmonize/records.py src/plasmid_priority/dedup/canonicalize.py scripts/04_harmonize_metadata.py scripts/05_deduplicate.py tests/test_harmonize.py tests/test_dedup.py
git commit -m "Expose canonical provenance fields before backbone modeling

Constraint: Knownness and source audits need provenance that survives harmonize and dedup stages
Rejected: Reconstruct source and duplicate provenance later from report-time joins | too fragile and drift-prone
Confidence: high
Scope-risk: moderate
Directive: Any new upstream metadata field that changes source or year semantics must be normalized here first
Tested: tests/test_harmonize.py, tests/test_dedup.py, ruff, mypy
Not-tested: full ETL pipeline
"
```

### Task 3: Build canonical data-quality audits and wire them into the backbone/scoring pipeline

**Files:**
- Create: `src/plasmid_priority/data_quality/audits.py`
- Modify: `src/plasmid_priority/features/core.py`
- Modify: `src/plasmid_priority/validation/missingness.py`
- Modify: `scripts/14_build_backbone_table.py`
- Modify: `scripts/15_normalize_and_score.py`
- Test: `tests/test_data_quality_audits.py`
- Test: `tests/test_missingness_audit.py`

- [ ] **Step 1: Write the failing audit tests**

```python
import pandas as pd

from plasmid_priority.data_quality.audits import build_canonical_data_quality_summary


def test_build_canonical_data_quality_summary_flags_source_concentration_and_missingness() -> None:
    surface = pd.DataFrame(
        [
            {
                "canonical_id": "c1",
                "canonical_source": "refseq",
                "macro_region": "western_asia",
                "year_confidence": "missing",
                "duplicate_group_size": 5,
                "backbone_assignment_confidence": 0.42,
            },
            {
                "canonical_id": "c2",
                "canonical_source": "refseq",
                "macro_region": "western_asia",
                "year_confidence": "exact",
                "duplicate_group_size": 1,
                "backbone_assignment_confidence": 0.91,
            },
        ]
    )

    summary = build_canonical_data_quality_summary(surface)

    assert "source_concentration" in set(summary["audit_name"])
    assert "year_missingness" in set(summary["audit_name"])
    assert "warning" in set(summary["status"])
```

- [ ] **Step 2: Run the audit tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_data_quality_audits.py tests/test_missingness_audit.py -q
```

Expected: FAIL because the audit module does not exist yet.

- [ ] **Step 3: Implement summary-level data-quality audits and backbone-table passthrough**

```python
# src/plasmid_priority/data_quality/audits.py
import pandas as pd

from plasmid_priority.validation.missingness import audit_missingness


def build_canonical_data_quality_summary(surface: pd.DataFrame) -> pd.DataFrame:
    missingness = audit_missingness(surface[["macro_region", "year_confidence"]])
    rows = [
        {
            "audit_name": "source_concentration",
            "status": "warning" if surface["canonical_source"].value_counts(normalize=True).max() > 0.80 else "pass",
            "metric_value": float(surface["canonical_source"].value_counts(normalize=True).max()),
        },
        {
            "audit_name": "duplicate_pressure",
            "status": "warning" if surface["duplicate_group_size"].astype(float).mean() > 1.5 else "pass",
            "metric_value": float(surface["duplicate_group_size"].astype(float).mean()),
        },
        {
            "audit_name": "year_missingness",
            "status": "warning" if float(missingness.iloc[0]["missing_fraction"]) > 0.10 else "pass",
            "metric_value": float(missingness.iloc[0]["missing_fraction"]),
        },
    ]
    return pd.DataFrame(rows)
```

```python
# src/plasmid_priority/features/core.py
backbone_table["canonical_source"] = merged["canonical_source"]
backbone_table["macro_region"] = merged["macro_region"]
backbone_table["year_confidence"] = merged["year_confidence"]
backbone_table["duplicate_group_size"] = merged["duplicate_group_size"]
```

`scripts/14_build_backbone_table.py` should write:

- `data/analysis/canonical_training_surface.tsv`
- `data/analysis/canonical_data_quality_summary.tsv`
- `data/analysis/canonical_data_quality_details.tsv`

`scripts/15_normalize_and_score.py` should validate those files exist before scoring.

- [ ] **Step 4: Re-run the audit and missingness tests**

Run:

```bash
python3 -m pytest tests/test_data_quality_audits.py tests/test_missingness_audit.py -q
```

Expected: PASS

- [ ] **Step 5: Run the touched backbone/scoring regression**

Run:

```bash
python3 -m pytest tests/test_workflow.py -k "pipeline_sequential_workflow or analysis_refresh_sequential_workflow" -q
python3 -m ruff check src/plasmid_priority/data_quality/audits.py src/plasmid_priority/features/core.py scripts/14_build_backbone_table.py scripts/15_normalize_and_score.py tests/test_data_quality_audits.py tests/test_missingness_audit.py
python3 -m mypy src/plasmid_priority/data_quality/audits.py src/plasmid_priority/features/core.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/data_quality/audits.py src/plasmid_priority/features/core.py src/plasmid_priority/validation/missingness.py scripts/14_build_backbone_table.py scripts/15_normalize_and_score.py tests/test_data_quality_audits.py tests/test_missingness_audit.py tests/test_workflow.py
git commit -m "Gate official modeling on canonical data-quality audits

Constraint: Model-selection honesty depends on knowing whether the input surface itself is trustworthy
Rejected: Leave data-quality checks as ad hoc notebooks or one-off scripts | they would not block bad official surfaces
Confidence: medium
Scope-risk: broad
Directive: Official scoring must fail closed when canonical data-quality artifacts are missing or stale
Tested: tests/test_data_quality_audits.py, tests/test_missingness_audit.py, tests/test_workflow.py, ruff, mypy
Not-tested: full release rebuild
"
```

### Task 4: Add knownness-robust and source-robust candidate descendants on the cleaned surface

**Files:**
- Modify: `src/plasmid_priority/modeling/module_a_support.py`
- Modify: `src/plasmid_priority/modeling/module_a.py`
- Test: `tests/test_single_model_pareto.py`

- [ ] **Step 1: Write the failing candidate-family tests**

```python
def test_build_single_model_candidate_family_includes_clean_surface_variants() -> None:
    family = build_single_model_candidate_family(
        protocol=protocol,
        available_models=["discovery_12f_source", "phylo_support_fusion_priority"],
    )

    names = {candidate["model_name"] for candidate in family}

    assert "discovery_12f_source_knownness_robust" in names
    assert "discovery_12f_source_source_robust" in names
    assert "phylo_support_fusion_priority_compact_calibrated" in names
```

- [ ] **Step 2: Run the focused Pareto tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_single_model_pareto.py -k "candidate_family or weighted_objective" -q
```

Expected: FAIL because the cleaned-surface descendants do not exist yet.

- [ ] **Step 3: Implement bounded candidate descendants and cheap robustness proxies**

```python
# src/plasmid_priority/modeling/module_a_support.py
SINGLE_MODEL_CLEAN_DESCENDANTS = {
    "discovery_12f_source": [
        "discovery_12f_source_knownness_robust",
        "discovery_12f_source_source_robust",
    ],
    "phylo_support_fusion_priority": [
        "phylo_support_fusion_priority_compact_calibrated",
    ],
}
```

```python
# src/plasmid_priority/modeling/module_a.py
if model_name.endswith("_knownness_robust"):
    feature_frame = feature_frame.drop(
        columns=[column for column in feature_frame.columns if "knownness" in column or "metadata_density" in column],
        errors="ignore",
    )
if model_name.endswith("_source_robust"):
    feature_frame = feature_frame.drop(
        columns=[column for column in feature_frame.columns if column.startswith("source_")],
        errors="ignore",
    )
if model_name.endswith("_compact_calibrated"):
    feature_frame = feature_frame.loc[:, core_feature_columns]
```

Keep this family bounded. Do not turn the search into an uncontrolled combinatorial sweep.

- [ ] **Step 4: Re-run the focused Pareto tests**

Run:

```bash
python3 -m pytest tests/test_single_model_pareto.py -q
```

Expected: PASS

- [ ] **Step 5: Run the modeling regression and static checks**

Run:

```bash
python3 -m pytest tests/test_single_model_pareto.py tests/test_model_audit.py -k "single_model or knownness or source_robust" -q
python3 -m ruff check src/plasmid_priority/modeling/module_a_support.py src/plasmid_priority/modeling/module_a.py tests/test_single_model_pareto.py
python3 -m mypy src/plasmid_priority/modeling/module_a_support.py src/plasmid_priority/modeling/module_a.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/modeling/module_a_support.py src/plasmid_priority/modeling/module_a.py tests/test_single_model_pareto.py tests/test_model_audit.py
git commit -m "Search cleaned-surface descendants that target knownness and source failures

Constraint: Compute budget must stay bounded while we attack the live matched-knownness failure directly
Rejected: Unbounded grid search over arbitrary feature drops | too expensive and hard to audit
Confidence: medium
Scope-risk: moderate
Directive: Add new descendants only when they encode a clear scientific hypothesis, not generic search-space inflation
Tested: tests/test_single_model_pareto.py, tests/test_model_audit.py, ruff, mypy
Not-tested: long-running retrain on full data
"
```

### Task 5: Gate single-model official selection on cleaned-data trust and surface it in reports

**Files:**
- Modify: `scripts/21_run_validation.py`
- Modify: `src/plasmid_priority/reporting/model_audit.py`
- Modify: `scripts/24_build_reports.py`
- Modify: `scripts/28_build_release_bundle.py`
- Test: `tests/test_model_audit.py`
- Test: `tests/test_reporting.py`
- Test: `tests/test_workflow.py`

- [ ] **Step 1: Write the failing decision/report tests**

```python
def test_build_single_model_official_decision_blocks_official_pass_when_data_quality_fails() -> None:
    finalists = pd.DataFrame(
        [
            {
                "model_name": "discovery_12f_source_knownness_robust",
                "scientific_acceptance_status": "pass",
                "roc_auc": 0.81,
                "average_precision": 0.73,
            }
        ]
    )
    data_quality = pd.DataFrame(
        [
            {"audit_name": "source_concentration", "status": "fail", "metric_value": 0.92},
        ]
    )

    decision = build_single_model_official_decision(finalists, data_quality_summary=data_quality)

    assert decision.iloc[0]["single_model_official_status"] == "fail"
    assert decision.iloc[0]["single_model_official_decision_reason"] == "blocked_by_data_quality_gate"
```

```python
def test_headline_validation_summary_mentions_clean_data_gate_for_single_model() -> None:
    markdown = build_headline_validation_summary(
        single_model_official_decision=pd.DataFrame(
            [
                {
                    "single_model_name": "discovery_12f_source_knownness_robust",
                    "single_model_official_status": "pass",
                    "single_model_official_decision_reason": "pass_with_clean_data_surface",
                    "data_quality_gate_status": "pass",
                }
            ]
        )
    )

    assert "Data-quality gate: pass" in markdown
```

- [ ] **Step 2: Run the focused tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_model_audit.py tests/test_reporting.py tests/test_workflow.py -k "single_model_official_decision or headline_validation_summary or release_bundle" -q
```

Expected: FAIL because the data-quality gate is not part of the official decision yet.

- [ ] **Step 3: Thread data-quality gate artifacts through validation, official decision, reports, and release**

```python
# scripts/21_run_validation.py
canonical_data_quality_summary = pd.read_csv(
    context.data_dir / "analysis/canonical_data_quality_summary.tsv",
    sep="\t",
)
single_model_official_decision = build_single_model_official_decision(
    single_model_pareto_finalists,
    data_quality_summary=canonical_data_quality_summary,
)
```

```python
# src/plasmid_priority/reporting/model_audit.py
def build_single_model_official_decision(
    finalists: pd.DataFrame,
    *,
    data_quality_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if data_quality_summary is not None and data_quality_summary["status"].eq("fail").any():
        best = finalists.sort_values(["failure_severity", "roc_auc"], ascending=[True, False]).iloc[0]
        return pd.DataFrame(
            [
                {
                    "single_model_name": best["model_name"],
                    "single_model_official_status": "fail",
                    "single_model_official_decision_reason": "blocked_by_data_quality_gate",
                    "data_quality_gate_status": "fail",
                }
            ]
        )
```

Add the new fields to:

- `reports/core_tables/single_model_official_decision.tsv`
- `reports/core_tables/model_selection_summary.tsv`
- `reports/headline_validation_summary.md`
- `reports/release/bundle/RELEASE_INFO.txt`

Include `canonical_data_quality_summary.tsv` and `canonical_data_quality_details.tsv` in the release bundle.

- [ ] **Step 4: Re-run the decision/report/workflow regression**

Run:

```bash
python3 -m pytest tests/test_model_audit.py tests/test_reporting.py tests/test_workflow.py -q
```

Expected: PASS

- [ ] **Step 5: Run static checks for the report/validation path**

Run:

```bash
python3 -m ruff check src/plasmid_priority/reporting/model_audit.py scripts/21_run_validation.py scripts/24_build_reports.py scripts/28_build_release_bundle.py tests/test_model_audit.py tests/test_reporting.py tests/test_workflow.py
python3 -m mypy src/plasmid_priority/reporting/model_audit.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/plasmid_priority/reporting/model_audit.py scripts/21_run_validation.py scripts/24_build_reports.py scripts/28_build_release_bundle.py tests/test_model_audit.py tests/test_reporting.py tests/test_workflow.py
git commit -m "Block official promotion when canonical data-quality gates fail

Constraint: A model cannot honestly be official if the input surface is scientifically untrustworthy
Rejected: Report data-quality warnings separately while still allowing official promotion | too easy to misread as approval
Confidence: high
Scope-risk: broad
Directive: Keep report language aligned with the actual gate result; never imply pass when data-quality status is fail or warning
Tested: tests/test_model_audit.py, tests/test_reporting.py, tests/test_workflow.py, ruff, mypy
Not-tested: full artifact rebuild on external data
"
```

### Task 6: Run the full joint program and refresh the official evidence surface

**Files:**
- Modify: `scripts/14_build_backbone_table.py`
- Modify: `scripts/15_normalize_and_score.py`
- Modify: `scripts/21_run_validation.py`
- Modify: `scripts/24_build_reports.py`
- Modify: `scripts/28_build_release_bundle.py`
- Test: `tests/test_workflow.py`
- Test: `tests/test_reporting.py`
- Test: `tests/test_model_audit.py`

- [ ] **Step 1: Add or update the final workflow expectation tests**

```python
def test_validation_script_records_canonical_data_quality_outputs(self) -> None:
    result = pipeline_for_step("21_run_validation.py")
    assert "data/analysis/canonical_data_quality_summary.tsv" in result.outputs
    assert "data/analysis/canonical_data_quality_details.tsv" in result.outputs


def test_release_bundle_includes_canonical_data_quality_surfaces(self) -> None:
    result = release_bundle_manifest()
    assert "reports/core_tables/canonical_data_quality_summary.tsv" in result
    assert "reports/diagnostic_tables/canonical_data_quality_details.tsv" in result
```

- [ ] **Step 2: Run the workflow tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_workflow.py -k "canonical_data_quality or release_bundle" -q
```

Expected: FAIL until the final artifact chain is complete.

- [ ] **Step 3: Execute the full build and refresh the official artifacts**

Run:

```bash
python3 scripts/14_build_backbone_table.py
python3 scripts/15_normalize_and_score.py
python3 scripts/21_run_validation.py
python3 scripts/24_build_reports.py
python3 scripts/28_build_release_bundle.py
```

Expected:

- `canonical_training_surface.tsv` exists
- `canonical_data_quality_summary.tsv` exists
- `single_model_pareto_screen.tsv`, `single_model_pareto_finalists.tsv`, and `single_model_official_decision.tsv` are regenerated from the cleaned surface
- release bundle contains the new data-quality artifacts

- [ ] **Step 4: Run the final verification suite**

Run:

```bash
python3 -m pytest tests/test_canonical_data_surface.py tests/test_data_quality_audits.py tests/test_harmonize.py tests/test_dedup.py tests/test_missingness_audit.py tests/test_single_model_pareto.py tests/test_model_audit.py tests/test_reporting.py tests/test_workflow.py -q
python3 -m ruff check .
python3 -m mypy src
python3 scripts/26_run_tests_or_smoke.py --with-tests
```

Expected: PASS

- [ ] **Step 5: Capture the final scientific evidence**

Record in the execution log:

- the final chosen single model name
- ROC AUC and AP
- `scientific_acceptance_status`
- every failed criterion if still failing
- `data_quality_gate_status`
- whether `matched_knownness_gap` improved versus the current `discovery_12f_source` baseline

Also compare against the current baseline:

```text
baseline model: discovery_12f_source
baseline roc_auc: 0.803594
baseline ap: 0.722510
baseline fail reason: fail:matched_knownness
```

- [ ] **Step 6: Commit**

```bash
git add scripts/14_build_backbone_table.py scripts/15_normalize_and_score.py scripts/21_run_validation.py scripts/24_build_reports.py scripts/28_build_release_bundle.py tests/test_workflow.py tests/test_reporting.py tests/test_model_audit.py
git commit -m "Refresh official evidence on top of the cleaned canonical data surface

Constraint: The final model decision must be backed by regenerated artifacts, not reasoning alone
Rejected: Stop after unit tests without rebuilding reports and release bundle | insufficient scientific evidence
Confidence: medium
Scope-risk: broad
Directive: Re-run the full artifact chain whenever canonical data quality or candidate-family logic changes
Tested: focused pytest suites, ruff, mypy, scripts/26_run_tests_or_smoke.py --with-tests
Not-tested: external reproduction outside this workspace
"
```

---

## Self-Review

### Spec coverage

- Canonical data layer: covered by Tasks 1-3.
- Data-quality gates: covered by Tasks 3 and 5.
- Guardrail-aware candidate search: covered by Task 4.
- Balanced official selection with pass-first behavior: covered by Task 5.
- Report and release refresh: covered by Tasks 5-6.

No spec gaps remain.

### Placeholder scan

- No `TODO`, `TBD`, or deferred “implement later” markers remain.
- Every task includes exact file paths, test-first steps, concrete commands, and expected outcomes.

### Type consistency

- Canonical data outputs use the same names throughout the plan:
  - `canonical_training_surface.tsv`
  - `canonical_data_quality_summary.tsv`
  - `canonical_data_quality_details.tsv`
- The official decision API consistently uses `data_quality_summary=...`.
- The data gate field is consistently named `data_quality_gate_status`.


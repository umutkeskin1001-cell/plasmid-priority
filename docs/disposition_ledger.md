# Disposition Ledger

- generated_at_utc: `2026-04-18T12:13:32+00:00`
- principle: keep / absorb / replace / delete

## Decisions

| Surface | Disposition | Status | Rationale | Next Action |
|---|---|---|---|---|
| `scripts/run_geo_spread_branch.py` | replace | active | Backward-compatible wrapper only. | Route through `scripts/run_branch.py` until external callers migrate. |
| `scripts/run_bio_transfer_branch.py` | replace | active | Backward-compatible wrapper only. | Route through `scripts/run_branch.py` until external callers migrate. |
| `scripts/run_clinical_hazard_branch.py` | replace | active | Backward-compatible wrapper only. | Route through `scripts/run_branch.py` until external callers migrate. |
| `scripts/run_consensus_branch.py` | replace | active | Backward-compatible wrapper only. | Route through `scripts/run_branch.py` until external callers migrate. |
| `scripts/archive/*` | delete | inactive | Legacy scripts are outside canonical workflow. | Remove after provenance parity check and one release cycle. |
| `src/plasmid_priority/reporting/model_audit_*.py` re-export stubs | delete | completed | Stub-only re-export surfaces removed. | Keep APIs served from `model_audit.py` until real module split lands. |
| `src/plasmid_priority/modeling/tree_models.py` | replace_or_delete | unclear usage | Tree backends should be canonical backend or removed. | If no runtime dependency remains, remove; else integrate into official model surface. |

## Evidence

### tree_models usage graph

```text
tests/test_ci_workflow.py:22:    def test_ci_installs_tree_models(self) -> None:
```

### scripts/archive inventory

```text
build_phase_52_report.py
check_code_review_graph.py
run_phase_52_discovery.py
```

## Guardrail

No new legacy wrapper, archive script, or re-export stub should be added
without an explicit ledger entry and planned deletion path.

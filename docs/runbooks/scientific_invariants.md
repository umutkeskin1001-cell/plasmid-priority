# Scientific Invariants

The repository treats release claims as constrained scientific surfaces, not free-form narratives.

## Non-Optional Invariants

The freeze and equivalence checks enforce these invariants:

- `ranking_drift`: top-100 candidate overlap must stay above the configured minimum
- `metric_drift`: headline metric absolute delta must stay within the configured tolerance
- `row_count_drift`: canonical table row counts must stay within allowed change
- `schema_drift`: canonical artifact schemas must not contract unexpectedly

These invariants are implemented in `src/plasmid_priority/governance/freeze.py`.

## Release-Time Scientific Gates

- `make scientific-contract-gate`
- `make artifact-integrity`
- `make release-readiness`
- `make protocol-freshness`

## Interpretation Rule

- A stronger metric does not override a failed invariant.
- Additive documentation is acceptable; silent contract drift is not.
- Generated reviewer-pack artifacts must match the current protocol documents.

## Change Rule

When a scientific invariant changes, update the contract source, test coverage, and reviewer-facing documentation together.

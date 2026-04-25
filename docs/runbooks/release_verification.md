# Release Verification Runbook

This repository has one canonical release-grade verification path:

```bash
make verify-release
```

The target runs the same gates that define a publishable state:

- protocol freshness
- import contract
- runtime budget gate
- scientific contract gate
- artifact integrity gate
- release readiness gate
- lint
- typecheck
- full-suite coverage gate
- critical-path coverage gate
- dependency security audit
- smoke surface
- strict docs build

The orchestration lives in `scripts/46_verify_release.py`. The script writes machine-readable and markdown summaries under `reports/release/`.

## Operator Rule

- Use `make quality` for a fast local signal.
- Use `make verify-release` before claiming a release-ready state.
- Treat any manual command sequence as advisory only; the script is the authority.

## Expected Outputs

- `reports/release/release_verification_summary.json`
- `reports/release/release_verification_summary.md`
- `reports/release/release_readiness_report.json`
- `reports/release/release_readiness_report.md`

## Failure Handling

1. Read the first failed step from `release_verification_summary.md`.
2. Fix the underlying issue, not the summary artifact.
3. Re-run `make verify-release` from the repo root.

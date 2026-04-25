# Quality Scoreboard

This scoreboard defines the minimum release-grade bar for the repository.

| Surface | Command | Pass condition |
| --- | --- | --- |
| Style | `make lint` | `ruff` reports zero violations |
| Typing | `make typecheck` | `mypy` reports zero blocking errors |
| Full coverage | `make test-cov` | overall coverage `>= 70%` |
| Critical coverage | `make critical-coverage` | targeted modules `>= 90%` |
| Scientific contract | `make scientific-contract-gate` | release scientific contract passes |
| Artifact integrity | `make artifact-integrity` | canonical artifact contract passes |
| Release readiness | `make release-readiness` | go/no-go checklist passes |
| Runtime budgets | `make runtime-budget-gate` | smoke-local budget holds |
| Import layering | `make import-contract` | dependency contract passes |
| Protocol freshness | `make protocol-freshness` | generated protocol docs are current |
| Security | `make security` | `pip-audit` passes allowlist policy |
| Smoke path | `make smoke` | smoke script and tests pass |
| Docs | `make docs-check` | `mkdocs build --strict` passes |

## Canonical Summary

`make verify-release` is the only command that is expected to satisfy every row in one pass.

## Update Rule

If a threshold changes, update the command target, this scoreboard, and the release-readiness contract in the same commit.

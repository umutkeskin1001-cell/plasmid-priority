# Plasmid Priority - Ruthless Full Audit

Date: 2026-04-26  
Scope: Full repo health, build/test gates, static quality, architecture drift, API/runtime risk.  
Repo ID: `local/plasmid-priority-e0d92cc3`

## 1) Executive Summary

This branch is currently not releasable.

- Hard blockers:
  - 8 syntax-corruption points in production scripts (compile/test collection breaks).
  - `make lint` fails with 157 issues.
  - `make typecheck` fails with 105 errors in core files.
  - `make test-cov` and `make smoke` fail at collection because of syntax errors.
  - `make protocol-freshness` and `make scientific-contract-gate` fail (blocked by `scripts/31_generate_scientific_contracts.py` syntax error).
- Structural risk:
  - Dead/unreachable code signal is very high (`dead_code_pct=41.1`, `dead_count=1242`).
  - Large orphaned refactor surface (`modeling/evaluation.py`, `modeling/metrics.py`, `pipeline/branches.py` effectively disconnected).
  - Critical modules are excluded from coverage gates.
- API risk:
  - `/score` path returns fallback `0.0` scores on artifact failures instead of fail-closed error.

## 2) Reproduction Evidence (Commands)

- Syntax check:
  - `.venv/bin/python -m compileall -q src scripts tests` -> fails.
- Lint:
  - `make lint` -> fails (`157 errors`).
- Type check:
  - `make typecheck` -> fails (`105 errors in 4 files`).
- Test coverage gate:
  - `make test-cov` -> fails during collection (syntax error in `scripts/28_build_release_bundle.py`).
- Smoke:
  - `make smoke` -> fails during collection (same syntax error).
- Release readiness:
  - `make release-readiness` -> fails (`status=fail`; failed checks: `graphql_surface_present`, `async_batch_surface_present`).
- Security:
  - `make security` -> passes, but with ignored vulns (`CVE-2025-69872`, `CVE-2026-3219`).

## 3) P0 - Release Blocking Defects

### P0.1 Syntax corruption in scripts (hard compile break)

Malformed lines contain merged statements such as `...exc_info=True)return ...`.

Affected files/lines:

1. `scripts/28_build_release_bundle.py:101`
2. `scripts/28_build_release_bundle.py:338`
3. `scripts/28b_run_sovereign_ensemble.py:88`
4. `scripts/29_build_experiment_registry.py:69`
5. `scripts/29_train_sovereign.py:67`
6. `scripts/31_generate_scientific_contracts.py:298`
7. `scripts/31_generate_scientific_contracts.py:431`
8. `scripts/profile_workflow.py:132`

Impact:

- CI compile step fails.
- Pytest collection fails immediately for modules importing these scripts.
- Smoke and coverage gates cannot execute meaningful checks.

---

### P0.2 `scripts/15_normalize_and_score.py` contains immediate runtime defects

`ruff` findings indicate unresolved and conflicting symbols:

- `F821 Undefined name _cache_key_path` (`scripts/15_normalize_and_score.py:99`)
- `F821 Undefined name _cache_key_payload` (`scripts/15_normalize_and_score.py:115`)
- `F823 Local variable cache_key_path referenced before assignment` (`line 92`)
- Additional dead/unused assignments in same block.

Impact:

- Script execution can fail at runtime even after syntax blockers are fixed.

---

### P0.3 API scoring silently degrades to fake scores on artifact failure

In `src/plasmid_priority/api/app.py`:

- `_fallback_score()` returns `{"priority_index": 0.0}` (`line ~74`).
- `ModelRegistry.score_backbones()` falls back to `_fallback_score` if registry unavailable or errors (`line ~222 onward).
- `/score` and `/score/backbones` still return `status="ok"` with these fallback values (`line ~313 onward).

Impact:

- Silent correctness failure for product API clients (looks successful but values are synthetic zeros).
- Inconsistency with `/explain` and `/evidence`, which properly fail with HTTP 503 when artifacts are unavailable.

## 4) P1 - High Severity Structural Defects

### P1.1 CI blind spot for API tests

- `tests/test_api_security.py:5` and `tests/test_api_artifact_registry.py:86` use `pytest.importorskip("fastapi")`.
- CI install step is:
  - `.github/workflows/ci.yml:30` -> `python -m pip install -e ".[analysis,dev,tree-models]"`
- `api` extra (`fastapi`, `uvicorn`) is defined in `pyproject.toml` but not installed in CI.

Impact:

- API security tests can be skipped silently in CI.
- Important security behavior can regress without failing pipeline.

---

### P1.2 Coverage policy excludes high-risk modules

`pyproject.toml` coverage omit list includes multiple core areas:

- `src/plasmid_priority/modeling/module_a.py`
- `src/plasmid_priority/features/core.py`
- `src/plasmid_priority/reporting/figures.py`
- many advanced/survival/modeling modules

At the same time, complexity hotspots are concentrated exactly in these areas.

Impact:

- Reported coverage does not reflect risk surface.
- Regressions in core model/report logic are less likely to be detected.

---

### P1.3 Orphaned/rotten branch architecture chain

Importer analysis:

- `src/plasmid_priority/pipeline/branches.py` -> `importer_count=0`
- `src/plasmid_priority/shared/branch_base.py` is only imported by `pipeline/branches.py` (which itself is unreferenced)

Type-check corroboration:

- `pipeline/branches.py` has extensive signature mismatches against branch packages (many `call-arg` and `arg-type` errors from `make typecheck`).

Impact:

- A full branch orchestration path exists in code but is not integrated and currently inconsistent with real APIs.
- High maintenance burden and confusion during future edits.

---

### P1.4 Partial split refactor left disconnected modules

Files introduced but effectively disconnected:

- `src/plasmid_priority/modeling/evaluation.py` -> `importer_count=0`, coupling `instability=1.0`
- `src/plasmid_priority/modeling/metrics.py` -> `importer_count=0`, coupling `isolated`

At the same time, `module_a.py` still contains overlapping functionality and remains the dominant path.

Impact:

- Duplicate logic and drift risk.
- Refactor debt accumulates with no runtime value.

---

### P1.5 Governance gate drift: release-readiness requires removed surfaces

Current `make release-readiness` output fails even when major checks pass because:

- `graphql_surface_present = false`
- `async_batch_surface_present = false`

(`reports/release/release_readiness_report.json`, generated `2026-04-26T16:49:17+00:00`)

Impact:

- Product direction and gate policy are inconsistent.
- Team can ship gate failures that are policy-obsolete rather than quality-real.

## 5) P2 - Medium Severity Quality and Reliability Defects

### P2.1 Over-broad exception handling with silent fallback/pass

Examples:

- `src/plasmid_priority/io/table_io.py:201,214` -> broad `except Exception`, then `pass`
- `src/plasmid_priority/modeling/fold_cache.py:250` -> silent failure on save
- `src/plasmid_priority/modeling/oof_cache.py:56`
- `src/plasmid_priority/annotate/sequence_cache.py:95`
- `src/plasmid_priority/sensitivity/variant_cache.py:101,113,184`

Impact:

- Data corruption and I/O errors can be hidden and surface much later as model/report anomalies.

---

### P2.2 Repository hygiene collapse from ad-hoc fixer scripts in root

Root contains untracked fixer/generator scripts (`fix_exceptions.py`, `fix_imports.py`, `fix_app.py`, `generate_split.py`, `split_ast.py`, etc.) and these are included by lint scope.

Observed effects:

- Lint fails heavily on these files.
- `fix_imports.py` includes `git checkout` mutation behavior.
- `fix_exceptions.py` regex rewriting aligns with the style of corruption now seen in production scripts.

Impact:

- Tooling noise, unstable automation, increased accidental breakage risk.

---

### P2.3 Monolithic hotspot concentration (maintainability risk)

Key complexity hotspots:

- `src/plasmid_priority/reporting/model_audit.py::build_primary_model_selection_summary`
  - cyclomatic `168`, params `12`, lines `600`
- `scripts/25_export_tubitak_summary.py::main`
  - cyclomatic `102`, lines `469`
- `scripts/21_run_validation.py::main`
  - cyclomatic `82`, lines `1248`
- `scripts/run_workflow.py::run_workflow`
  - cyclomatic `60`, lines `318`

File scale:

- `src/plasmid_priority/reporting/build_reports_script.py`: `5721` lines
- `src/plasmid_priority/reporting/model_audit.py`: `5792` lines
- `src/plasmid_priority/modeling/module_a.py`: `3400` lines
- `src/plasmid_priority/features/core.py`: `1879` lines

Impact:

- High regression probability; difficult reviewability and safe refactoring.

## 6) Signals from Existing `optimize.md` / `suggest.md`

Current status versus those documents:

- `API fail-closed auth` recommendation appears already implemented:
  - `src/plasmid_priority/api/app.py` uses `hmac.compare_digest`
  - server missing key returns HTTP 500 (fail-closed).
- However, large parts of the suggested cleanup are still incomplete:
  - split modules exist but are not fully integrated,
  - silent exception handling remains widespread,
  - pipeline/branch architecture duplication/drift persists.

Note:

- User-facing earlier filename request referenced `suggestion.md`; repository currently has `suggest.md` (untracked in this branch state).

## 7) Prioritized Fix Order (Strict)

1. Fix all 8 syntax-corruption points in scripts (compile green first).
2. Fix `scripts/15_normalize_and_score.py` unresolved names/runtime defects.
3. Make `/score` fail-closed on artifact unavailability (align with `/explain`/`/evidence`).
4. Ensure CI installs `api` extra or equivalent so FastAPI tests cannot silently skip.
5. Remove or quarantine root fixer scripts from lint scope; stop mutation scripts from touching production files.
6. Decide one branch orchestration path; remove or fully repair `pipeline/branches.py` + `shared/branch_base.py` chain.
7. Complete split-refactor integration (or remove disconnected modules) for `modeling/evaluation.py` and `modeling/metrics.py`.
8. Reduce coverage omissions on core risk modules and rebalance tests around actual hotspots.

## 8) Final Verdict

Project has substantial engineering value but current branch state is unstable:

- Compile/test gates are hard-broken.
- Core behavior has silent-degradation paths.
- Architecture contains disconnected and duplicated subsystems.

Do not tag/release from this state before P0 + P1 items are closed.

## 9) Remediation Update (2026-04-26, same-day follow-up)

This section reflects fixes applied after the initial snapshot above.

### Fixed in this pass

1. **All 8 syntax-corruption points** in scripts were repaired.
   - `python3 -m compileall -q src scripts tests` now passes.
2. **`scripts/15_normalize_and_score.py` cache-key runtime breakage** fixed.
   - Removed undefined `_cache_key_path` / `_cache_key_payload` usage.
   - Removed local name shadowing of `cache_key_path`.
   - Unified cache payload construction.
3. **API scoring fail-closed behavior** implemented for artifact unavailability.
   - `/score` and `/score/backbones` now raise 503 when artifact registry is unavailable.
   - Missing per-row IDs still zero-fill only when artifact scoring itself succeeds.
4. **CI install surface updated** to include API dependencies.
   - `.github/workflows/ci.yml` install step now includes `api` extra.
5. **Release-readiness policy drift fixed**.
   - Removed legacy GraphQL/async-batch requirements.
   - Added checks for current REST product surface and SDK deprecation alignment.
6. **Phase-5 productization tests updated** to current API surface.
7. **API security and artifact-registry tests updated** for fail-closed auth and fail-closed artifact behavior.

### Verification results after remediation

- Targeted regression suite:
  - `uv run python -m pytest tests/test_release_readiness.py tests/test_phase5_productization.py tests/test_ci_workflow.py tests/test_api_security.py tests/test_api_artifact_registry.py`
  - **Result:** `23 passed`.
- Gates:
  - `make runtime-budget-gate artifact-integrity` -> **pass**
  - `make scientific-contract-gate` -> **pass**
  - `make release-readiness` -> **pass**
    - `reports/release/release_readiness_report.json`: `status = pass`, `failed_checks = []`.

### Still open (not solved in this pass)

1. **Global lint gate still red**:
   - `make lint` -> **137 errors**.
   - Dominant causes: ad-hoc untracked fixer scripts in repo root (`fix_*.py`, `generate_split.py`, `split_ast.py`, etc.) and existing broad-style debt outside this remediation scope.
2. **Global mypy gate still red**:
   - `make typecheck` -> **105 errors in 4 files**.
   - Concentrated in `src/plasmid_priority/reporting/build_reports_script.py`, `src/plasmid_priority/pipeline/branches.py`, and new/changed modeling split files.
3. **Protocol freshness still red**:
   - `make protocol-freshness` fails due generated scientific-contract docs drifting from committed versions (hash/surface mismatch), triggered while scored artifacts are missing.

### Additional remediation after this update section was written

- `make smoke` failure fixed:
  - Root cause: pairwise branch in `build_logistic_convergence_audit` built diagnostics but never appended audit rows.
  - Fix applied in:
    - `src/plasmid_priority/modeling/module_a.py`
    - `src/plasmid_priority/modeling/evaluation.py`
  - Validation:
    - `uv run python -m pytest tests/test_modeling.py::ModelingTests::test_convergence_audit_includes_pairwise_model_rows` -> pass
    - `make smoke` -> pass

- `build_protocol_snapshot` compatibility surface restored:
  - Re-added backward-compatible keys (`benchmark_scope`, `acceptance_thresholds`, `single_model_objective_weights`, etc.) to align provenance/tests/docs generation.
  - Validation:
    - `uv run python -m pytest tests/test_report_provenance.py::test_protocol_snapshot_includes_single_model_selection_weights` -> pass

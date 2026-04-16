# CI Failure Triage Runbook

Quick reference for diagnosing CI failures.

## Failure Classification

### 1. LightGBM Import Failure

**Symptom:**
```
AssertionError: _HAS_LIGHTGBM is False — LightGBM not installed.
```

**Cause:** `tree-models` extra not installed.

**Fix:**
```bash
pip install -e ".[analysis,dev,tree-models]"
# CI: ensure ci.yml has -e ".[analysis,dev,tree-models]"
```

---

### 2. Coverage Below Threshold

**Symptom:**
```
FAIL Required test coverage of 80% not reached. Total coverage: 7X%
```

**Fix:**
```bash
# Check which modules are under-covered
make test-cov
open htmlcov/index.html

# Add tests for uncovered lines
python -m pytest tests/ -x --cov=src --cov-report=term-missing
```

---

### 3. Type Check (mypy) Failure

**Symptom:**
```
error: Incompatible return value type (got "X", expected "Y")
```

**Fix:**
```bash
make typecheck 2>&1 | head -30
# Fix type hints in affected file
# If false positive, add: # type: ignore[assignment]
```

---

### 4. Lint (ruff) Failure

**Symptom:**
```
E501 Line too long (105 > 100 characters)
```

**Fix:**
```bash
make lint-fix
# Or manually for specific files:
python -m ruff check src/plasmid_priority/modeling/module_a.py --fix
```

---

### 5. Test Failure: Config Not Found

**Symptom:**
```
pytest.skip: config.yaml not found — skipping split_year check
```

This is expected in CI (no data). Run `make test-cov` to confirm skip, not failure.

---

### 6. Smoke Test Failure

**Symptom:**
```
SMOKE FAILED — 1 issue(s) found
  • Cannot import plasmid_priority.modeling: ...
```

**Fix:**
```bash
python scripts/26_run_tests_or_smoke.py --smoke-only
# Will print exact import error — fix the named module
```

---

### 7. Security Audit Failure

**Symptom:**
```
plasmid-priority 0.3.0 depends on numpy==X.Y.Z (GHSA-XXXX)
```

**Fix:**
```bash
make security 2>&1
# Update the affected dependency in pyproject.toml
pip install -e ".[analysis,dev,tree-models]" --upgrade
# Or pin to safe version:
# numpy>=2.0.1  # (example: exclude vulnerable range)
```

---

## Emergency Bypasses (ONLY for non-breaking failures)

```bash
# Skip failing test temporarily (add to test file):
@pytest.mark.skip(reason="tracked in issue #XX — fix by YYYY-MM-DD")

# Skip type check for a specific line:
x: Any = some_call()  # type: ignore[assignment]

# Do NOT skip leakage tests or coverage threshold permanently.
```

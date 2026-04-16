# Model Update Runbook

This runbook documents the procedure for updating the primary or governance model.

## Changing the Primary Model

### 1. Evaluate the Candidate

```bash
# Run full validation with the candidate model:
python scripts/16_run_module_A.py
python scripts/21_run_validation.py

# Check candidate against primary in model_comparison_summary.tsv:
cat data/analysis/model_comparison_summary.tsv | grep -E "model_name|discovery_boosted|new_candidate"
```

The new primary model must satisfy ALL of the following:
- [ ] ROC AUC > current primary (selection-adjusted permutation p < 0.01)
- [ ] PR AUC ≥ current primary (positive AP lift maintained)
- [ ] Brier score ≤ current primary (calibration not degraded)
- [ ] VIF audit: no "critical" concern features (`vif_audit_summary.tsv`)
- [ ] p/n ratio < 0.1 (`model_pn_ratio.tsv`)
- [ ] Passes all holdout tests (blocked_holdout, spatial_holdout)

### 2. Update config.yaml

```yaml
models:
  primary_model_name: new_candidate_model   # ← change here
  primary_model_fallback: parsimonious_priority  # keep fallback
```

### 3. Add to core_model_names (if not already there)

```yaml
  core_model_names:
    - baseline_both
    - governance_linear
    - new_candidate_model  # ← add here
    - ...
```

### 4. Verify Smoke Tests Pass

```bash
python scripts/26_run_tests_or_smoke.py --smoke-only
# Expected: "Primary model 'new_candidate_model' found with N features."
```

### 5. Rerun Test Suite

```bash
make test
# test_model_integrity.py::test_primary_model_in_feature_sets must pass
```

### 6. Update Reports

```bash
make reports
make tubitak-summary
```

### 7. Commit with Conventional Commit

```bash
git add config.yaml
git commit -m "feat(model): promote new_candidate_model to primary

ROC AUC: 0.XX (selection-adj p < 0.001)
Previous primary: discovery_boosted (ROC AUC: 0.XX)
Reason: higher discriminability on low-knownness backbones

Resolves #XX"
```

---

## Changing the Governance Model

Same procedure, but governance model requirements are stricter:
- [ ] Coefficients must be interpretable (linear backbone preferred)
- [ ] Calibration ECE < 0.05 (much stricter than primary)
- [ ] Model simplicity score must justify complexity addition
- [ ] All governance-specific features (coherence, purity, assignment confidence) retained

---

## Emergency Rollback

If a model update causes CI failures:

```bash
git revert HEAD  # rolls back config change
git push
# CI will rerun with previous model
```

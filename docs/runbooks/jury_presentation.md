# Jury Preparation Runbook

This runbook prepares the team for the TÜBİTAK jury presentation.

## Pre-Presentation Checklist

### 1. Regenerate All Outputs (1 day before)

```bash
# Ensure data disk is mounted
ls $PLASMID_PRIORITY_DATA_ROOT

# Full pipeline refresh
make full-local DATA_ROOT=$PLASMID_PRIORITY_DATA_ROOT

# Generate TÜBİTAK summary
make tubitak-summary

# Rebuild reports
make reports
```

Verify `reports/tubitak_final_metrics.txt` exists and is recent (check timestamp).

### 2. Verify Key Metrics

```bash
cat reports/tubitak_final_metrics.txt
```

Expected: ROC AUC, PR AUC, Brier Score, positive prevalence, permutation p-value.

### 3. Run Quality Gate

```bash
make quality
```

All checks must pass (lint, typecheck, tests, smoke, security).

---

## Anticipated Jury Questions & Answers

### Q1: "Your AUC is high — isn't that just because negatives dominate?"

**A:** Yes, this is a class-imbalanced task. The eligible backbone cohort has a positive
prevalence of approximately X% (see `reports/tubitak_final_metrics.txt`, line `Temel Oran`).
We address this with:
1. **Sample weighting**: `class_balanced + knownness_balanced` weighting in all models.
2. **PR AUC**: We report Average Precision (PR AUC) which is prevalence-sensitive.
3. **AP Lift**: We report lift over the prevalence baseline, not raw AP.
4. **Permutation null**: We permute labels and verify AUC collapses to ~0.5.

### Q2: "Did you search 44 models and pick the best one? That's p-hacking."

**A:** Yes, we search 44 candidate models. However:
1. **Selection-adjusted permutation p-value**: We use a selection-adjusted null that
   accounts for the multiple-model search. The reported p-value is NOT the naive per-model
   permutation p-value.
2. **Honest pre-registration**: The primary model (`discovery_boosted`) is declared in
   `config.yaml` upfront, not discovered post-hoc.
3. **Held-out validation**: We use repeated stratified k-fold OOF (out-of-fold) prediction,
   not a single train/test split.
4. **BH correction**: Model comparison table includes Benjamini-Hochberg FDR-corrected
   q-values across all model comparisons.

### Q3: "Can this be used in clinics?"

**A:** **No.** This is a retrospective surveillance tool. It cannot:
- Make individual patient diagnoses
- Predict clinical outcomes for any specific patient
- Replace clinical microbiology workflows
- Be used for real-time treatment decisions

The model labels (`geo_spread`, `bio_transfer`, `clinical_hazard`) are retrospective
epidemiological signals, not prospective clinical predictions.

### Q4: "How do you know the model isn't learning data artifacts?"

**A:** Multiple safeguards:
1. **Temporal split**: Training ≤ 2015, test > 2015. No future data leaks into training.
2. **Knownness residualization**: We partial out sequence abundance/knownness from features
   to prevent highly-observed backbones from dominating scores.
3. **Discovery contract**: `discovery_contract.py` has hard vetoes against leaky features.
4. **Outcome permutation test**: `falsification.py` verifies AUC collapses under label
   permutation — if model learned artifacts, AUC would stay high under permutation.
5. **Negative controls**: Random-score baseline and label-permutation null included in all
   comparison tables.

### Q5: "How were the consensus weights (geo: 0.5, bio: 0.25, clinical: 0.25) chosen?"

**A:** The weights reflect the scientific rationale that:
- **geo_spread (0.5)**: Geographic spread is the most directly measurable and reliable
  surveillance signal. It reflects actual mobility across borders.
- **bio_transfer (0.25)**: Host expansion is important but noisier due to host taxonomy
  database limitations.
- **clinical_hazard (0.25)**: Clinical escalation signals are important but depend on
  clinical metadata quality which varies by country.

Sensitivity analysis (`22_run_sensitivity.py`) shows that ±0.1 weight changes do not
significantly alter the candidate shortlist (Spearman ρ > 0.95 for top-50 rankings).

### Q6: "What if your external dependencies (firthlogist, PLSDB) are unavailable?"

**A:** All external dependencies have documented fallback paths:

| Dependency | Fallback |
|------------|----------|
| `lightgbm` | `HistGradientBoostingClassifier` (sklearn) |
| `interpret` (EBM) | `HistGradientBoostingClassifier` |
| `firthlogist` | standard logistic regression |
| PLSDB data | pipeline fails fast with clear error |
| AMRFinderPlus | AMR module skipped, probe marked absent |

### Q7: "Your training cohort is ≤2015 — isn't PLSDB too small then?"

**A:** This is intentional. The temporal split design means:
- Training: backbones with ≥1 member deposited ≤2015 (N≈X backbones)
- Testing: geographic/host spread observed 2016–2020 (horizon_years=5)

The Knownness-stratified analysis (`knownness_stratified_performance.tsv`) shows the model
performs above baseline even for low-abundance backbones (lower half of knownness quartile),
verifying we're not just recovering well-known resistant clones.

---

## Slide Deck Checklist

- [ ] Title slide with scientific boundary disclaimer
- [ ] Data pipeline diagram (slides 2-3)
- [ ] T/H/A feature explanation (slide 4)
- [ ] ROC + calibration figures (slides 5-6)
- [ ] Top candidate shortlist (slide 7)
- [ ] Comparison: discovery vs governance vs baseline (slide 8)
- [ ] Permutation null visualization (slide 9)
- [ ] Limitations slide (slide 10) — mandatory

---

## Emergency Fallback (if make quality fails)

If quality gate fails on presentation day:

```bash
# Run only lighting smoke check
python scripts/26_run_tests_or_smoke.py --smoke-only

# If LightGBM missing:
pip install lightgbm

# If tests fail:
python -m pytest tests/ -x -q --ignore=tests/test_figures.py --tb=short
# (figures tests often fail without data)
```

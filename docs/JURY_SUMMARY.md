# Plasmid Priority — TÜBİTAK Jury Technical Summary

> **Revision:** April 2026 | **Version:** 0.3.0
> **Contact:** See repository README for principal investigator details.

---

## ⚠️ Scientific Scope

This system is a **retrospective genomic surveillance prioritization tool**.

It **does NOT**:
- Make clinical diagnoses or treatment recommendations
- Predict individual patient outcomes
- Replace clinical microbiology or infection control workflows
- Operate in real-time on patient samples

All outputs are epidemiological retrospective signals intended for **population-level surveillance** interpretation by qualified specialists.

---

## 1. What We Built

A computational framework that assigns priority scores to operational plasmid backbone classes — groups of related plasmids sharing a conserved genomic backbone — based on their retrospective spread signals.

### Three Branch Surfaces

| Branch | Signal | Outcome Definition |
|--------|--------|--------------------|
| `geo_spread` | Geographic spread | ≥3 new countries in 5-year horizon |
| `bio_transfer` | Host expansion | ≥2 new host genera in 5-year horizon |
| `clinical_hazard` | Clinical escalation | Fraction gain in last-resort/MDR classes |

These branches are fused into a `consensus` risk score using defensible weights (see below).

### Two Model Tracks

| Track | Model | Purpose |
|-------|-------|---------|
| **Discovery** | `discovery_boosted` (LightGBM) | Maximum discrimination for retrospective prioritization |
| **Governance** | `governance_linear` (Logistic) | Interpretable, calibrated, guardrail-aware deployment |

---

## 2. How We Built It

### Temporal Design (No Leakage)

```
TRAIN: Backbone members deposited ≤ 2015
  └── Feature computation (T, H, A signals)
  └── Knownness features (member count, countries, RefSeq share)

TEST: Spread/transfer/escalation observed 2016–2020 (horizon = 5 years)
  └── Outcome labels (spread_label, bio_label, clinical_label)
```

**No test-period data enters training features.** Enforced by:
1. `features/core.py`: `_training_period_records()` strict cutoff
2. `tests/test_leakage.py`: automated leakage unit tests
3. `validation/boundaries.py`: pipeline invariant checks

### Cross-Validation Design

- **Repeated stratified K-fold** (5 splits × 5 repeats = 25 folds)
- **Out-of-fold (OOF)** predictions — no test-set information in any final prediction
- **Class-balanced + IPW-balanced sample weighting** to handle prevalence imbalance

### Feature Engineering (T / H / A Framework)

| Feature Family | Description | Key Features |
|----------------|-------------|--------------|
| **T** (Transfer) | Mobilization potential | `T_eff_norm`, `A_recurrence_norm`, IncGroup diversity |
| **H** (Host) | Host diversity & phylogenetic breadth | `H_phylogenetic_specialization_norm`, evenness |
| **A** (AMR) | Resistance burden | `A_eff_norm`, `amr_gene_burden_norm`, class richness |
| **Structural** | Backbone quality | coherence, purity, assignment confidence |

---

## 3. Key Numbers

> Note: Fill in actual values from `reports/tubitak_final_metrics.txt` before presentation.

| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC AUC (primary) | see `tubitak_final_metrics.txt` | see file |
| Average Precision | see file | see file |
| AP Lift over baseline | see file | — |
| Brier Score | see file | see file |
| Positive prevalence | see file | — |
| n eligible backbones | see file | — |
| n positive backbones | see file | — |
| Permutation p-value (selection-adj) | see file | — |
| Best Spearman ρ (count alignment) | see file | see file |

### Sample Weighting Strategy

```
class_balanced + knownness_balanced (IPW)
```

This addresses both class imbalance (rare positive events) and knownness bias
(highly-observed backbones dominating the signal).

---

## 4. Statistical Rigor

### Model Selection (44 Candidates → 1 Primary)

- 44 candidate models evaluated across the research model registry
- **Benjamini-Hochberg FDR correction** applied across all model comparisons
- **Selection-adjusted permutation p-value** reported (not naive per-model p-value)
- Single-model Pareto screen → finalist audit → official decision pipeline

### Negative Controls

| Control | Description | Expected Result |
|---------|-------------|-----------------|
| Label permutation | Permute `spread_label` | AUC → ~0.5 |
| Random score | Uniform random predictions | AUC → ~0.5 |
| Outcome falsification (`falsification.py`) | Multiple permutation runs | All AUC < threshold |

### Holdout Validations

- **Blocked geographic holdout**: Model evaluated on held-out macro-regions
- **Temporal rolling origin**: AUC stability across different split years
- **Knownness-stratified performance**: Verified model works for low-abundance backbones

---

## 5. Consensus Weights Rationale

```yaml
geo_spread:    0.50  # Most directly measurable signal
bio_transfer:  0.25  # Important but noisier
clinical_hazard: 0.25  # Quality varies by country
```

**Sensitivity analysis** (`22_run_sensitivity.py`): ±0.1 weight changes yield Spearman ρ > 0.95
for top-50 backbone rankings — results are robust to weight perturbation.

---

## 6. What We Did NOT Claim

The following are **explicitly out of scope** and are stated in all reports:

- ❌ Causal transmission inference
- ❌ Prospective outbreak prediction
- ❌ Individual plasmid risk (only backbone-class level)
- ❌ Clinical outcome prediction (mortality, patient prognosis)
- ❌ "Ground truth" spread validation (retrospective by design)
- ❌ External population validation (prospective cohort not available)

---

## 7. Technical Infrastructure

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Primary model | LightGBM 4.x |
| Interpretable model | Logistic regression + Firth penalty |
| Nonlinear alternative | InterpretML (EBM) / HistGradientBoosting (fallback) |
| Data layer | DuckDB + pandas |
| Validation | Custom metrics (tie-invariant AP), bootstrap CI (2000 iterations) |
| Multicollinearity | VIF audit (`data/analysis/vif_audit_summary.tsv`) |
| Reproducibility | `uv.lock` + multi-stage Docker |
| CI | GitHub Actions (Python 3.12 + 3.13, tree-models tested) |

---

## 8. FAQ for Jury

**Q: Why not use a deep learning model?**
A: Sample size (~N backbones eligible) limits deep learning benefit. Interpretable models
(LightGBM + logistic) outperform or match in this regime and provide feature attribution.

**Q: What database was used?**
A: PLSDB + RefSeq plasmid sequences + Pathogen Detection metadata. All public databases.

**Q: How do you handle the class imbalance?**
A: Class-balanced + IPW-balanced sample weighting. We report AP (prevalence-sensitive)
alongside ROC AUC. See sınıf dengesizliği note in `tubitak_final_metrics.txt`.

**Q: Can this system be updated as new sequences are deposited?**
A: Yes. Pipeline is fully reproducible. Running `make full-local` with an updated data root
will regenerate all outputs from scratch with new data.

**Q: What makes this different from existing plasmid databases?**
A: Existing databases (PlasmidFinder, MOB-suite) focus on classification, not risk
prioritization. This is the first framework to fuse T/H/A surveillance signals at the
backbone-class level with a formally validated retrospective design.

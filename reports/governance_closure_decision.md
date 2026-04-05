# Governance Temporal Evidence Closure Decision

**Document Type**: Final Closure Memo — Governance Promotion Deferred  
**Date**: Generated from batch artifacts  
**Status**: FINAL — No Further Governance Action Required at This Time

---

## A. Current Governance State

### Official Model (Retained)
- **Model**: `phylo_support_fusion_priority`
- **Role**: OFFICIAL GOVERNANCE BASELINE
- **Raw AUC**: 0.8272 [0.800, 0.852]
- **ECE**: 0.0852
- **Temporal Evidence Status**: partially_evaluated (1 window)

### Challenger Model (Held)
- **Model**: `governance_15f_pruned`
- **Role**: CHALLENGER — NOT PROMOTED
- **Raw AUC**: 0.8234 [0.795, 0.849]
- **ECE**: 0.0896
- **Δ vs Baseline**: -0.0037
- **paired_delong_p**: 0.405 (not significant)
- **Temporal Evidence Status**: partially_evaluated (1 window)
- **Classification**: STABILITY_PRESERVING

### Latest Phase 6.1 Metrics Comparison

| Metric | phylo_support_fusion_priority | governance_15f_pruned | Assessment |
|--------|------------------------------|----------------------|------------|
| Raw AUC | 0.8272 | 0.8234 | Baseline higher |
| ECE | 0.0852 | 0.0896 | Both acceptable |
| AUC Δ | — | -0.0037 | Within tolerance |
| paired_delong_p | — | 0.405 | Not significant |
| Gate Overall | N/A | **FAILED** | p-value gate not passed |

### Latest Temporal Evidence Status

| Model | Temporal Status | Windows | Mean Rolling AUC | Gap vs OOF |
|-------|-----------------|---------|------------------|------------|
| phylo_support_fusion_priority | partially_evaluated | 1 | 0.8272 | -0.00003 |
| governance_15f_pruned | partially_evaluated | 1 | 0.8244 | +0.0010 |

**Evidence Coverage**: Only 1 temporal window available (2015→2023, training_only). Full multi-window matrix (2012-2018 splits × 1/3/5/8 year horizons) not present in current data.

---

## B. Why Promotion Is Deferred

### 1. Pruning Candidate Is Not Clearly Superior
- **AUC comparison**: Challenger AUC is actually *lower* than baseline (-0.0037)
- **Statistical significance**: paired_delong_p = 0.405 indicates NO significant difference
- **Gain classification**: TIE/NOISE — not a meaningful improvement
- **Gate evaluation**: FAILED `p_pass_or_status` due to high p-value

### 2. Temporal Evidence Is Insufficient
- **Coverage**: Only 1 window evaluated vs. required 10+ for reliable assessment
- **Status**: Both models marked `partially_evaluated` — insufficient for promotion decision
- **Recommendation logic**: Conservative threshold requires `evaluated` status (≥10 windows) for challenger promotion

### 3. Rolling Temporal Evidence Not Strong Enough
- **Temporal gap**: Challenger shows slightly larger degradation vs OOF (+0.0010) than baseline (-0.00003)
- **ECE concern**: Mean ECE in temporal window (0.366) is much higher than OOF ECE (~0.09), suggesting calibration instability
- **Single-window risk**: Cannot assess stability across varying conditions

### 4. Full Temporal Campaign Deferred (Compute-Cost Tradeoff)
**Rationale for Deferral**:
- A full multi-window temporal validation campaign requires:
  - Multiple split years (2012-2018): 7 distinct training cuts
  - Multiple horizon lengths (1, 3, 5, 8 years): 4 horizons per split
  - Multiple assignment modes (all_records, training_only): 2 modes per configuration
  - **Total**: ~56 distinct temporal window evaluations
  - Each window requires OOF prediction generation with 5-fold × 3-repeat cross-validation
  - Estimated compute: 2-4 hours on current laptop hardware

**Decision**: The marginal governance improvement potential (pruned model is stability-preserving but not superior) does not justify immediate compute investment. Full temporal validation is **deferred** until:
- Stronger compute hardware is available, OR
- Governance challenger shows clearer superiority signals in preliminary evaluation

---

## C. Current Decision

### Explicit Statement

**GOVERNANCE OFFICIAL REMAINS UNCHANGED**: `phylo_support_fusion_priority` continues as the official governance model.

**GOVERNANCE CHALLENGER REMAINS RECORDED**: `governance_15f_pruned` is retained as challenger for future reconsideration.

**NO FURTHER GOVERNANCE PROMOTION ACTION IS TAKEN NOW**.

### Decision Rationale (Conservative)

Under stability-first governance semantics, baseline replacement requires compelling evidence of:
1. **Superior performance** (AUC gain ≥ 0.025 with significance), OR
2. **Statistically equivalent performance** (p > 0.05) with **material ancillary benefits** (substantial parsimony, improved interpretability, reduced compute cost)

governance_15f_pruned satisfies criterion #2 partially (p=0.405, AUC loss within tolerance) but:
- Does not demonstrate compelling ancillary benefits that justify replacement
- Temporal stability evidence is insufficient (1 window only)
- Pruning alone is not sufficient justification for baseline replacement

Therefore, **conservative retention of the proven baseline is the appropriate action**.

---

## D. Future Options (Optional Future Work Only)

The following are **not immediate actions** but potential future work if governance evidence needs strengthening:

### Option 1: Governance Hybrid Stretch Evaluation (Phase 6.2)
- **Description**: Evaluate hybrid feature combinations or ensemble approaches
- **When to consider**: If `governance_15f_pruned` shows stronger preliminary signals
- **Compute cost**: Medium (1-2 hours)
- **Current status**: Optional — not required now

### Option 2: Fuller Temporal Evidence Generation on Stronger Hardware
- **Description**: Run complete 56-window temporal validation matrix
- **When to consider**: When high-confidence temporal robustness assessment is required
- **Compute cost**: High (2-4 hours on laptop; <30 min on workstation/cloud)
- **Current status**: **DEFERRED** — blocked on hardware, not priority

### Option 3: No Immediate Action (Selected)
- **Description**: Maintain current state and revisit if new evidence emerges
- **When to consider**: Current evidence insufficient to justify change
- **Compute cost**: None
- **Current status**: **ACTIVE**

---

## E. Artifact Record

### Source Artifacts Consulted
- `reports/phase_61_per_model_summary.tsv`
- `reports/phase_61_gate_evaluation.tsv`
- `reports/phase_61_recommendation.md`
- `reports/governance_temporal_evidence.tsv`
- `reports/governance_temporal_recommendation.md`
- `reports/final_model_promotion_decision.md`
- `reports/final_model_promotion_summary.tsv`
- `reports/final_model_promotion_tokens.json`

### Generated Closure Artifacts
- `reports/governance_closure_decision.md` (this memo)
- `reports/governance_closure_summary.tsv` (summary table)
- `reports/governance_next_step_token.json` (machine-readable token)

---

## F. Statistical Honesty Notes

- **paired_delong_p = 0.405**: Indicates NO statistically significant difference between challenger and baseline. This is a **paired DeLong test p-value**, NOT a selection-adjusted permutation-null p-value.
- **Temporal evidence**: Only 1 window available; "partially_evaluated" is honest reporting of evidence insufficiency.
- **ECE values**: OOF ECE (~0.09) vs temporal ECE (0.37) suggests calibration instability — warrants caution.

---

## G. Final Closure Statement

**The governance temporal-evidence line of work is now CLOSED.**

**Current State**:
- Official: `phylo_support_fusion_priority`
- Challenger: `governance_15f_pruned` (held, not promoted)
- Decision: DEFER_PROMOTION
- Next Step: NO_IMMEDIATE_ACTION

**No config.yaml changes required.**  
**No immediate governance action required.**  
**No further temporal experiments scheduled.**

---

*Governance temporal evidence closure generated per conservative promotion protocol.*

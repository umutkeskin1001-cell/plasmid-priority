# Final Model Promotion Decision

**Document Type**: Post-Batch Model Promotion Decision Package  
**Generated From**: Phase 5.2 Discovery + Phase 6.1 Governance Pruning  
**Date**: Auto-generated from batch artifacts  
**Status**: CONFIDENTIAL — Internal Decision Memo

---

## A. Discovery Decision

### Current Baseline
- **Model**: bio_clean_priority
- **Raw AUC**: 0.7454
- **95% CI**: [0.714, 0.778]
- **ECE**: 0.0634

### Evaluated Candidates

| Model | Raw AUC | AUC CI | ECE | paired_delong_p | Δ vs Baseline |
|-------|---------|--------|-----|-----------------|---------------|
| discovery_9f_source | 0.7920 | [0.761, 0.820] | 0.0414 | 2.82e-06 | +0.0466 |
| discovery_12f_source | 0.8036 | [0.775, 0.831] | 0.0491 | 9.15e-08 | +0.0582 |
| discovery_12f_class_balanced | 0.8131 | [0.785, 0.841] | 0.0938 | 3.69e-08 | +0.0676 |

### Raw AUC Comparison
- **Absolute winner**: discovery_12f_class_balanced (0.8131)
- **Runner-up**: discovery_12f_source (0.8036)
- **Third**: discovery_9f_source (0.7920)
- All candidates show MEANINGFUL gain vs baseline (Δ > 0.025)

### Calibration Comparison
- **Best calibrated**: discovery_9f_source (ECE 0.0414)
- **Acceptable**: discovery_12f_source (ECE 0.0491)
- **Near threshold**: discovery_12f_class_balanced (ECE 0.0938, 94% of 0.10 threshold)

### paired_delong_p Caveat
All candidates show paired_delong_p < 0.001 vs baseline, indicating statistically significant improvement. However, these are **paired DeLong p-values**, NOT selection-adjusted permutation-null p-values. Selection-adjusted inference accounting for post-hoc model selection from 3 candidates would require additional `build_selection_adjusted_permutation_null()` analysis.

### Final Conservative Recommendation

| Role | Model | Rationale |
|------|-------|-----------|
| **OFFICIAL** | discovery_12f_source | Strong AUC improvement (+0.0582) with acceptable calibration (ECE 0.0491). Balances predictive power with robust calibration. |
| **CHALLENGER** | discovery_12f_class_balanced | Highest raw AUC (+0.0676) but calibration near threshold (ECE 0.0938). Requires further validation before official promotion. |

### Rationale (One Paragraph)

The discovery batch produced three strong candidates, all materially improving over bio_clean_priority. While discovery_12f_class_balanced achieves the highest raw AUC (0.8131), its calibration error (ECE 0.0938) approaches the 0.10 acceptance threshold, creating deployment risk. In contrast, discovery_12f_source achieves nearly comparable AUC (0.8036, only 0.0095 lower) with substantially better calibration (ECE 0.0491 vs 0.0938). Under conservative scientific promotion principles that prioritize calibration robustness alongside AUC gains, discovery_12f_source is the safer OFFICIAL promotion, with discovery_12f_class_balanced retained as CHALLENGER pending additional validation evidence (particularly calibration stability across temporal splits).

---

## B. Governance Decision

### Current Baseline
- **Model**: phylo_support_fusion_priority
- **Raw AUC**: 0.8272
- **95% CI**: [0.800, 0.852]
- **ECE**: 0.0852

### Pruning Candidate
- **Model**: governance_15f_pruned
- **Raw AUC**: 0.8234
- **95% CI**: [0.795, 0.849]
- **ECE**: 0.0896
- **Δ vs Baseline**: -0.0037
- **paired_delong_p**: 0.405

### Raw AUC Comparison
- AUC loss: -0.0037 (within stability-preserving tolerance of < 0.015)
- paired_delong_p = 0.405 indicates no statistically significant difference vs baseline
- Gain class: TIE/NOISE

### Calibration Comparison
- Baseline ECE: 0.0852
- Candidate ECE: 0.0896 (acceptable, ECE < 0.10)
- No material calibration degradation

### Rolling/Temporal Evidence Status
**Status**: **not_evaluated**

The Phase 6.1 execution did not include rolling-origin or temporal stability evaluation. Per governance protocol, this is explicitly reported as missing rather than fabricating boolean passes. Absence of temporal evidence constrains promotion confidence — we cannot claim temporal robustness for governance_15f_pruned.

### Gate Evaluation Summary

| Model | ECE < 0.10 | paired_delong_p < 0.05 | Rolling Origin | Overall |
|-------|------------|------------------------|----------------|---------|
| phylo_support_fusion_priority | ✓ | N/A (baseline) | N/A | N/A |
| governance_15f_pruned | ✓ | ✗ (p=0.405) | not_evaluated | ✗ |

### Governance Classification
- **Classification**: STABILITY_PRESERVING
- **Not**: SUPERIOR (AUC gain < 0.025 and p-value not significant)
- **Not**: REJECTED (AUC loss < 0.015, ECE acceptable)

### Final Conservative Recommendation

| Role | Model | Rationale |
|------|-------|-----------|
| **OFFICIAL** | phylo_support_fusion_priority | Maintains highest raw AUC with acceptable calibration. No compelling evidence requires replacement. |
| **CHALLENGER** | governance_15f_pruned | Stability-preserving (AUC loss only -0.0037), more parsimonious, but temporal evidence incomplete. Candidate for future replacement if temporal robustness confirmed. |

### Rationale (One Paragraph)

The governance pruning evaluation classifies governance_15f_pruned as STABILITY_PRESERVING, not SUPERIOR. The paired DeLong test (p=0.405) shows no statistically significant difference from the phylo_support_fusion_priority baseline, and the raw AUC is marginally lower (-0.0037). While this AUC loss is within tolerance for a parsimonious replacement, the absence of rolling-origin or temporal stability evidence precludes confident promotion. Under stability-first semantics appropriate for governance decisions, we retain the established baseline as OFFICIAL and designate the pruning candidate as CHALLENGER pending temporal validation. The pruning candidate offers feature-set simplification that may prove valuable, but insufficient evidence exists to justify immediate official replacement.

---

## C. Promotion Summary

### Discovery Surface

| Model | Promotion Status | Rationale |
|-------|-----------------|-----------|
| bio_clean_priority | **REPLACED** (baseline) | Superseded by discovery candidates |
| discovery_9f_source | **REJECTED** | Weaker AUC than 12f alternatives, no distinct advantage |
| discovery_12f_source | **OFFICIAL** | Strong AUC + acceptable calibration, conservative choice |
| discovery_12f_class_balanced | **CHALLENGER** | Highest raw AUC but calibration near threshold |

### Governance Surface

| Model | Promotion Status | Rationale |
|-------|-----------------|-----------|
| phylo_support_fusion_priority | **OFFICIAL** | Baseline maintains best raw AUC, no compelling replacement evidence |
| governance_15f_pruned | **CHALLENGER** | Stability-preserving, parsimonious, but temporal evidence incomplete |

---

## D. Optional Config Preview

The following sections show candidate config.yaml changes under different promotion strategies. **These are previews only — config.yaml has NOT been modified.**

### Conservative Promotion (Recommended)

```yaml
models:
  primary_model_name: "discovery_12f_source"      # Changed from bio_clean_priority
  primary_model_fallback: "bio_clean_priority"    # Fallback to proven baseline
  conservative_model_name: "bio_clean_priority"   # Unchanged
  governance_model_name: "phylo_support_fusion_priority"  # Unchanged
  governance_model_fallback: "support_synergy_priority"   # Unchanged

  research_model_names:
    # ... existing list ...
    - "discovery_12f_class_balanced"  # Retain as challenger
    - "governance_15f_pruned"         # Retain as challenger
```

### Aggressive Experimental Promotion (Not Recommended)

```yaml
models:
  primary_model_name: "discovery_12f_class_balanced"  # Raw AUC winner
  primary_model_fallback: "discovery_12f_source"      # Safer fallback
  governance_model_name: "governance_15f_pruned"        # Pruned candidate
```

**Decision**: No config.yaml changes applied at this time. Conservative promotion deferred to human review.

---

## Artifact Locations

- **This Decision Memo**: `reports/final_model_promotion_decision.md`
- **Summary TSV**: `reports/final_model_promotion_summary.tsv`
- **Machine-Readable Tokens**: `reports/final_model_promotion_tokens.json`

---

*Generated from Phase 5.2 Discovery and Phase 6.1 Governance artifacts per promotion protocol.*

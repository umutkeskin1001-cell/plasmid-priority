# Sovereign Model Optimization Design

## Objective

Improve `sovereign_precision_priority` so it is stronger across four axes at the same time:

- discrimination (`roc_auc`, `average_precision`)
- probability quality (`brier_score`, `expected_calibration_error`, `log_loss`)
- robustness (`knownness_matched_gap`, source/region/blocked-holdout performance)
- simplicity and stability (fewer redundant features, less variance, easier interpretation)

The design goal is not "maximum ROC AUC at any cost". The goal is a model that remains strong after calibration, under subgroup stress, and after modest feature pruning.

## Current State

Current sovereign training from [scripts/29_train_sovereign.py](/Users/umut/Projeler/plasmid-priority/scripts/29_train_sovereign.py) produces:

- `roc_auc = 0.8294628459647361`
- `average_precision = 0.7639671856428951`
- `brier_score = 0.1655842860624284`
- `calibrated_roc_auc = 0.8360318803034709`

This beats the current best recorded model in `module_a_metrics.json` (`phylo_support_fusion_priority`, `roc_auc = 0.827138791227189`) on raw ROC AUC, but the current sovereign script is not using the full repo-native training stack.

## Key Findings

### 1. The sovereign script is underpowered relative to the repo's native modeling path

`config.yaml` already defines a richer fit surface for `sovereign_precision_priority`, including:

- `sample_weight_mode: class_balanced+knownness_balanced`
- `preprocess_mode: knownness_residualized`
- grouped residualization alphas
- calibration-related config flags

However, the current standalone sovereign script uses:

- plain `LogisticRegression`
- only `class_weight="balanced"`
- no repo-native knownness residualization
- no repo-native model scoring surface

This means the "current sovereign result" is likely not the strongest or most trustworthy version of the model.

### 2. The sovereign feature surface is close to the best existing model already

Compared with `phylo_support_fusion_priority`, sovereign changes only a small part of the feature surface:

- sovereign adds: `A_raw_norm`, `H_phylogenetic_specialization_norm`, `T_raw_norm`
- best existing model instead uses: `amr_clinical_escalation_norm`, `amr_mechanism_diversity_norm`, `last_resort_convergence_norm`, `silent_carrier_risk_norm`

This means the next gains will probably come from training protocol, evaluation discipline, and pruning, not from a wholesale redesign.

### 3. The sovereign feature set contains cleanup opportunities

Observed issues on the labeled cohort:

- `assignment_confidence_norm` is constant (`1.0`) and contributes no discriminative information
- several feature pairs are almost redundant:
  - `host_phylogenetic_dispersion_norm` vs `evolutionary_jump_score_norm`
  - `clinical_context_fraction_norm` vs `context_support_guard_norm`
  - `T_eff_norm` vs `T_raw_norm` / `external_t_synergy_norm`

Quick ablations show:

- removing the constant feature causes negligible performance change
- a compact sovereign variant can stay near current AUC
- blindly adding the "missing best-model features" does not improve performance

## Approaches Considered

### Approach A: Strengthen only the standalone sovereign script

Pros:

- fastest to iterate
- isolated blast radius

Cons:

- duplicates modeling logic already present in Module A
- weaker evaluation discipline
- easier to overfit to a single metric

### Approach B: Promote sovereign to a first-class Module A optimization target

Pros:

- uses the repo's actual fit config and preprocessing path
- evaluates calibration, knownness, and holdout robustness together
- easier to compare fairly with existing benchmark models

Cons:

- more plumbing work up front
- slightly slower iteration

### Approach C: Replace sovereign with an ensemble or nonlinear stack

Pros:

- highest raw upside ceiling

Cons:

- higher complexity
- weaker interpretability
- greater overfitting and governance risk

## Recommended Design

Use **Approach B**.

Build a `sovereign_v2` optimization workflow inside the existing Module A evaluation surface, then keep the standalone script only as a convenience wrapper or remove it once redundant.

### Design Principles

1. Do not optimize ROC AUC alone.
2. Prefer a slightly smaller model if discrimination is unchanged and calibration/robustness improve.
3. Treat dead config and dead features as bugs.
4. Make sovereign win on the same scoreboard used by the rest of the repository.

## Proposed Solution

### Phase 1: Make sovereign use the real training path

- route sovereign through Module A fit/evaluation helpers instead of bespoke training logic
- ensure `MODEL_FIT_CONFIG["sovereign_precision_priority"]` is actually used
- expose sovereign metrics using the same artifact surface as other named models

Expected benefit:

- fair apples-to-apples comparison
- access to knownness-aware preprocessing and evaluation

### Phase 2: Clean the feature surface

- remove constant features from sovereign
- prune near-duplicate features one pair at a time
- keep changes small and testable

Start with:

- drop `assignment_confidence_norm`
- test one representative from each near-duplicate pair instead of both

Expected benefit:

- lower variance
- simpler coefficients
- easier calibration

### Phase 3: Optimize on a multi-metric scorecard

Select sovereign revisions with a bounded objective:

- maximize `roc_auc`
- then prefer lower `brier_score`
- then prefer lower `expected_calibration_error`
- then prefer lower `knownness_matched_gap`
- then prefer lower feature count

Reject candidates that:

- lose materially on source/region/blocked holdouts
- improve only after suspicious calibration tricks
- depend on a constant or low-information feature

### Phase 4: Tighten calibration honestly

- avoid treating same-sample post-hoc calibration as final evidence
- add a cleaner calibration comparison path using the repo's evaluation surface
- prefer calibration methods that improve Brier/ECE without destabilizing rank metrics

## Acceptance Criteria

The new sovereign candidate is accepted only if all are true:

1. `roc_auc` is at least as good as current sovereign within a small tolerance.
2. `brier_score` and/or `expected_calibration_error` improve meaningfully.
3. `knownness_matched_gap` does not worsen.
4. source/spatial/blocked-holdout performance does not degrade materially.
5. feature count is reduced or unchanged unless a larger model wins clearly on multiple axes.

## Implementation Shape

Expected code areas:

- `config.yaml`
- `src/plasmid_priority/modeling/module_a.py`
- `src/plasmid_priority/modeling/module_a_support.py`
- `scripts/29_train_sovereign.py`
- tests covering sovereign fit/evaluation behavior

## Risks

- optimizing too aggressively for ROC AUC can hide calibration regressions
- using same OOF predictions for final calibration summaries can overstate probability quality
- feature pruning can remove "insurance" features that matter only in subgroup stress tests

## Non-Goals

- no new external dependencies unless existing tooling is insufficient
- no uncontrolled switch to opaque nonlinear models as the default sovereign path
- no report-layer claim changes before the underlying evaluation surface is updated

## Recommendation

Proceed with a conservative `sovereign_v2` upgrade that:

- reuses Module A's native fit/evaluation pipeline
- removes dead or redundant features
- ranks candidates on a multi-metric scorecard rather than raw AUC alone

This gives the best chance of making sovereign stronger in practice, not just stronger on one headline number.

# Single-Model Pareto Design

## Context

The repository now has a stricter and more honest scientific audit surface:

- canonical protocol-owned official thresholds
- protocol-aligned selection-adjusted official null
- explicit scientific acceptance failures in reporting
- repeatable smoke, report, and release evidence

That work improved defendability, but it did not materially improve predictive performance.
Current evidence shows that the strongest predictive models still fail the frozen official
acceptance gate, especially on `matched_knownness`, `source_holdout`, and sometimes
`calibration`.

The user now wants a single official model that maximizes three objectives together:

1. reliability / scientific defendability
2. predictive power
3. compute efficiency

The stated weighting is:

- reliability: `4`
- predictive power: `4`
- compute efficiency: `2`

This implies a weighted Pareto objective rather than a pure AUC-max search.

## Problem

The current model surface has two systemic issues:

1. The best-AUC candidates are not the most reliable candidates.
2. Heavy audits are expensive and should not be run across the full model universe.

If we continue to optimize only AUC/AP, we will keep landing on models that look strong but
fail the scientific gate. If we optimize only reliability, we risk collapsing predictive power.
If we run the full official audit on too many candidates, compute becomes impractical.

## Goal

Design and implement a new single-model selection program that produces one official model
optimized for the weighted objective:

`0.4 reliability + 0.4 predictive power + 0.2 compute efficiency`

The final model should be the best single official candidate under that objective, not merely
the highest-AUC candidate.

## Non-Goals

- Preserving the current official model names.
- Preserving the current single-pass evaluation shape.
- Running the full heavy audit on every model candidate.
- Optimizing only for leaderboard metrics.
- Adding complexity that does not improve the weighted objective.

## Design Principles

1. Reliability and predictive power are co-primary objectives.
2. Compute should constrain the search strategy, not define scientific truth.
3. Heavy audits should be reserved for finalists.
4. The final system must still end in one official model.
5. Model choice must be defensible from artifacts, not intuition.

## Candidate Approaches

### Option A: AUC-first larger search

Search more feature combinations and keep the best metric model.

Pros:
- highest chance of small headline-metric gains

Cons:
- likely repeats the current failure mode
- worst compute profile
- weak scientific defendability

Rejected because it overweights the wrong objective.

### Option B: Reliability-first compact model

Aggressively prune features and keep the most robust low-compute model.

Pros:
- best chance of improving guardrails quickly
- best compute profile

Cons:
- likely loses too much AUC/AP
- can become scientifically conservative but operationally weaker

Rejected as the sole strategy because it underweights predictive power.

### Option C: Weighted Pareto single-model program

Use a two-stage candidate program:

- Stage A: cheap screening over a constrained candidate family
- Stage B: heavy official audit only on Pareto-shortlisted finalists

Pros:
- directly matches the 4/4/2 weighting
- controls compute while still searching broadly
- gives a principled single-model decision rule

Chosen option.

## Chosen Design

### 1. Constrained candidate family

Do not search the entire existing model universe blindly. Start from the current strongest
surfaces and generate compact, interpretable descendants:

- `phylo_support_fusion_priority`
- `discovery_12f_source`
- `support_synergy_priority`
- `knownness_robust_priority`
- `parsimonious_priority`

Generate derived candidates by applying bounded transforms:

- feature pruning
- source-sensitive feature removal
- knownness-sensitive feature removal or residualization
- calibration-aware variants
- smaller discovery-only variants
- simpler governance-aware fusion variants

Every derived candidate must be explainable as a small edit from a known parent.

### 2. Weighted screening objective

Each candidate gets three scores:

#### Predictive Power Score

Based on:

- ROC AUC
- average precision
- shortlist utility / decision yield

#### Reliability Score

Based on:

- matched-knownness gap
- source-holdout gap
- spatial holdout
- blocked holdout
- calibration quality
- leakage review flag
- selection-adjusted official null, when available for finalists

#### Compute Score

Based on:

- feature count
- per-fit wall time
- total evaluation wall time
- whether heavy official audit is required

The aggregate search objective is weighted:

`0.4 reliability + 0.4 predictive power + 0.2 compute`

### 3. Two-stage evaluation

#### Stage A: cheap candidate screening

Run across the whole constrained family:

- OOF ROC AUC / AP
- matched-knownness gap
- source holdout gap
- calibration proxy
- feature count
- timed evaluation cost

Use this stage to eliminate obviously dominated candidates.

#### Stage B: heavy finalist audit

Run only on a small Pareto shortlist:

- frozen scientific acceptance audit
- selection-adjusted official null
- blocked holdout and calibration detail
- report-facing benchmark panel generation

This is where the canonical heavy compute is spent.

### 4. Single-model decision rule

The final official model is selected by this order:

1. Prefer candidates that `pass` the frozen official gate.
2. If no model passes, prefer the lowest failure severity.
3. Within that set, prefer higher predictive power.
4. If still tied, prefer lower compute cost.

Failure severity is not binary. A model failing only one guardrail with a small margin is better
than a model failing multiple guardrails by large margins.

### 5. Final output surface

The system still ends in one official model, but with better evidence:

- one official model decision
- one weighted Pareto rationale
- one explicit compute cost summary
- one explicit reliability breakdown

This keeps the external surface simple while making the internal selection process much stronger.

## Architecture Changes

### New concepts

- candidate derivation registry
- weighted screening score
- failure severity score
- evaluation cost tracker
- Pareto shortlist builder

### Existing surfaces to adapt

- `module_a.py`
  candidate generation and cheap screening
- `model_audit.py`
  severity scoring, finalist-heavy audit, single-model decision logic
- `scripts/21_run_validation.py`
  staged evaluation path
- `scripts/24_build_reports.py`
  single-model Pareto rationale in reporting
- `protocol.py`
  if the chosen official model changes, protocol ownership remains canonical

## Testing Strategy

The implementation must be test-first and staged.

### Unit

- candidate derivation produces expected bounded families
- weighted objective is deterministic
- failure severity scoring is deterministic
- compute tracker records stable, comparable cost summaries
- Pareto shortlist logic eliminates dominated candidates correctly

### Integration

- staged evaluation returns the same finalists on a stable fixture
- heavy audits only run on shortlist candidates
- final official model selection is reproducible
- reports consume the new single-model rationale correctly

### End-to-end

- `make quality`
- `scripts/26_run_tests_or_smoke.py --with-tests`
- refreshed release bundle

## Success Criteria

The design succeeds only if all are true:

1. The repository ends with one official model.
2. That model is better than the current official model on the weighted 4/4/2 objective.
3. Heavy compute is focused on finalists rather than the whole surface.
4. The final rationale is explicit enough to defend under audit.
5. The resulting official model is either:
   - an actual frozen-gate pass, or
   - the least-bad scientifically defensible option with clearly smaller failure severity and
     preserved predictive power.

## Risks

### Risk 1: Search complexity grows too much

Mitigation:
- constrain candidate family to parent-derived compact variants

### Risk 2: Reliability improves but AUC drops too much

Mitigation:
- weighted objective keeps predictive power co-primary

### Risk 3: Compute tracker changes the search but not the result

Mitigation:
- require compute metrics to influence shortlist pruning explicitly

### Risk 4: No candidate fully passes official acceptance

Mitigation:
- optimize failure severity honestly instead of hiding the result

## Recommendation

Proceed with implementation planning for the weighted Pareto single-model program.
This is the most direct way to honor the user’s actual weighting:

- not AUC-only
- not reliability-only
- not compute-only
- one official model, selected by a defensible engineering process

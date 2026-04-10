# Joint Data + Model Reliability Design

## Context

The repository now has a substantially stronger scientific audit surface:

- frozen scientific acceptance thresholds are explicit and enforced
- selection-adjusted null evidence is canonicalized
- report and release surfaces expose failures instead of hiding them
- a single-model Pareto selection layer exists and is wired into reporting

That work improved honesty and reproducibility, but it did not solve the main scientific problem.
The current best single-model candidate still fails the official gate because its signal weakens too
much under matched-knownness evaluation.

Current live evidence:

- official single-model candidate: `discovery_12f_source`
- ROC AUC: `0.8036`
- AP: `0.7225`
- acceptance status: `fail`
- failure reason: `fail:matched_knownness`

This strongly suggests that the next improvement cycle cannot be only a model-search problem.
The model surface is likely entangled with data quality, metadata density, source imbalance, and
entity duplication effects. The user wants a joint program that improves both:

1. scientific reliability
2. predictive power

with the explicit rule that scientific acceptance `pass` is required, but AUC degradation should be
kept minimal.

## Goal

Build a joint data-cleanup and model-selection program that:

- upgrades the data surface into a more canonical, auditable backbone-centric training table
- reduces knownness-driven and source-driven artifacts before model fitting
- searches for new official candidates on top of the cleaned data surface
- ends with a single official model that passes scientific acceptance if possible
- keeps predictive performance close to, or better than, the current discovery benchmark

## Success Criteria

The program is successful only if the final output demonstrates all of the following:

1. a canonical cleaned backbone dataset with explicit provenance and quality diagnostics
2. reduced data ambiguity around duplicates, source identity, geographic labels, and metadata sparsity
3. at least one official single-model candidate that clears the frozen scientific acceptance gate,
   or a clearly documented proof that the current feature family cannot do so
4. a final official model whose ROC AUC / AP is competitive with the current `discovery_12f_source`
   benchmark
5. regenerated report and release artifacts that reflect the new official decision honestly

For this program, "competitive" means the final chosen pass candidate should stay close enough to the
current benchmark that the scientific gain is not achieved by collapsing predictive value. The exact
tradeoff is not pre-fixed by a single number; it is decided by the final official selection rule.

## Non-Goals

- preserving the exact current data-layout contracts if they hide ambiguity
- preserving current model names
- preserving current intermediate TSV-only pipeline shape
- introducing infrastructure complexity that does not improve scientific quality
- optimizing only AUC at the expense of defendability

## Problem Statement

The present system has two coupled weaknesses:

### 1. Data ambiguity leaks into modeling

The pipeline already has harmonization and deduplication stages, but the effective modeling surface
still appears to carry residual ambiguity from:

- duplicate or near-duplicate biological records
- uneven metadata richness across sources
- source-identity imbalance
- geography and year standardization inconsistencies
- annotation confidence asymmetry
- backbone-level entity instability

These issues can inflate apparent signal and especially damage matched-knownness robustness.

### 2. Model search currently starts too late in the causal chain

The new Pareto model selector works correctly, but it is selecting from a candidate universe built on
the current scored backbone table. If the scored table itself still carries knownness-linked or
source-linked distortions, the model search is downstream of the real problem.

## Chosen Direction

Use a **joint program** with two coordinated lanes:

1. `Canonical Data Foundation`
2. `Guardrail-Aware Model Search`

These are part of one program, but they are not equal in ordering. The model lane is allowed to
explore continuously, yet official promotion decisions are blocked until the data lane clears quality
gates.

This creates a controlled loop:

`clean data surface -> audit data quality -> generate candidate models -> run acceptance audits -> feed failures back into data and feature cleanup`

## Why Not Other Options

### Option A: Data-first only

Pros:

- most scientifically conservative
- reduces upstream ambiguity before any modeling work

Cons:

- too slow to learn whether data fixes actually improve model guardrails
- gives delayed feedback on whether the cleaned surface helps the model

Rejected as the sole strategy because it sacrifices useful learning cycles.

### Option B: Model-first only

Pros:

- fastest path to a new candidate model

Cons:

- likely repeats the exact current failure mode
- optimizes on a still-ambiguous data surface

Rejected because the live evidence already says this is insufficient.

### Option C: Joint program with quality gates

Pros:

- lets us improve data and modeling in one program
- keeps fast learning cycles
- still prevents premature official promotion on dirty data

Chosen option.

## Architecture

The program has five layers.

### 1. Canonical Data Layer

Create one explicit backbone-centric canonical training surface that becomes the only authoritative
input for official modeling.

Responsibilities:

- stable entity resolution at the backbone / canonical record level
- canonical source labels
- canonical region labels
- confidence-aware metadata joins
- explicit duplicate-group identity and representative choice
- explicit temporal eligibility and training/test lineage fields

This layer may still emit TSV artifacts, but it should behave as a single canonical relational
surface. DuckDB is appropriate here as an execution engine and audit layer, not because it magically
improves accuracy, but because it gives deterministic joins, schema checks, reusable SQL audit
queries, and a cleaner path to reproducible entity assembly.

### 2. Data Quality Gate Layer

Before model candidates can be considered official, the canonical data layer must emit diagnostics
that quantify whether the training surface is trustworthy.

Required audit themes:

- duplicate and near-duplicate pressure
- canonical-id stability
- source concentration and source leakage risk
- geographic label quality and macro-region coverage
- year completeness and temporal contamination risk
- metadata missingness profile
- annotation confidence profile
- backbone purity / assignment confidence distribution

This layer does not choose the model; it decides whether the input surface is acceptable for official
model search.

### 3. Feature and Candidate Generation Layer

Build new single-model candidates on top of the cleaned canonical data surface. The candidate family
should remain bounded and interpretable.

Priority candidate directions:

- compact discovery descendants of `discovery_12f_source`
- cleaned support-fusion descendants of `phylo_support_fusion_priority`
- explicit knownness-deconfounded descendants
- source-robust variants
- calibration-aware compact variants

The candidate family should explicitly test whether knownness-leaning features or metadata-density
features are the main driver of the acceptance failure.

### 4. Scientific Selection Layer

Reuse the current Pareto selector, but run it on the cleaned data surface.

Official promotion rule:

1. prefer candidates with `scientific_acceptance_status = pass`
2. among pass candidates, maximize predictive power
3. among near-ties, prefer lower compute and lower complexity
4. if no candidate passes, publish the best available failure-severity ranking but do not pretend the
   system is deployment-ready

This is the `balanced` rule the user selected: pass is mandatory for true success, but predictive
power must remain competitive.

### 5. Report / Release Surface

The final official decision must remain externally simple:

- one official model
- one clear acceptance result
- one clear rationale
- one data-quality summary

But internally, the release bundle must include enough evidence for audit:

- canonical single-model decision
- Pareto screen artifact
- finalist audit artifact
- cleaned-data quality summaries
- updated release info

## Data Strategy

The data program should focus on the sources of likely matched-knownness inflation.

### A. Entity resolution and duplication

Strengthen:

- exact duplicate canonicalization
- near-duplicate clustering where scientifically justified
- representative-row selection rules
- duplicate-group lineage visibility in downstream tables

### B. Metadata canonicalization

Standardize:

- source identity
- country and macro-region mapping
- year parsing and uncertainty handling
- host taxonomy fields
- replicon-family normalization
- annotation token cleanup for AMR and mobility metadata

### C. Confidence-aware joins

The pipeline should distinguish:

- trusted missingness
- unknown / absent annotation
- parse failure
- low-confidence mapping

This matters because current models can accidentally treat metadata richness as biological evidence.

### D. Knownness-aware data diagnostics

The cleaned surface must expose whether metadata density, duplicate group size, source composition,
country count, or record count are strongly aligned with the target or with the model score.

This is the most important diagnostic family because the live failure is `matched_knownness`.

## Modeling Strategy

The new model-search program should focus on candidates that are more likely to break the
matched-knownness failure while protecting AUC.

### Candidate families to prioritize

1. `deconfounded discovery descendants`
   Discovery-core models where features most correlated with knownness-rich documentation are removed,
   downweighted, or residualized.

2. `cleaned support fusion descendants`
   High-signal models that keep biological power but reduce reliance on support-density and metadata
   richness effects.

3. `compact calibration-aware variants`
   Compact models that may lose a little raw AUC but gain enough calibration and stability to become
   pass candidates.

4. `source-robust variants`
   Candidates explicitly regularized or pruned against source-linked leakage.

### Selection logic

Continue to use weighted Pareto search, but the cleaned-data program should add one stronger
constraint:

- candidates that remain far below the matched-knownness threshold should be deprioritized even if
  they have strong AUC

This does not mean hardcoding away useful signal; it means the search must stop rewarding models that
clearly win on likely documentation artifacts.

## Compute Strategy

Compute should be spent where scientific value is highest.

### Cheap stages

- data-quality audits on canonical tables
- Stage A Pareto screening
- feature-family ablations
- knownness-correlation diagnostics

### Expensive stages

- finalist-heavy audit
- selection-adjusted null
- final report/release rebuild

DuckDB, if introduced, should be used mainly to reduce data-wrangling inefficiency and silent join
errors, not as a replacement for the scientific model stack.

## Deliverables

The full program should end with these deliverables:

1. canonical cleaned backbone training surface
2. explicit data-quality audit tables
3. upgraded candidate-generation logic
4. updated single-model official decision
5. regenerated headline/report/release artifacts
6. clear statement of whether the final official model truly passes scientific acceptance

## Risks

### Risk 1: We clean data but lose too much AUC

Mitigation:

- keep a bounded candidate family
- evaluate compact and richer variants in parallel
- only accept predictive loss when it materially improves acceptance posture

### Risk 2: We improve AUC but still fail matched-knownness

Mitigation:

- explicitly prioritize knownness-deconfounded candidates
- treat knownness diagnostics as a first-class selection input

### Risk 3: Data cleanup changes semantics invisibly

Mitigation:

- protocol-visible cleaning rules
- explicit before/after audit tables
- deterministic canonicalization logic

### Risk 4: Compute cost explodes

Mitigation:

- keep heavy audits finalist-only
- use canonical relational execution for data assembly
- cache stable data-quality and screen artifacts

## Decision

Proceed with a single coordinated program:

- build a canonical, auditable cleaned data surface
- add strict data-quality gates
- search new official single-model candidates on top of that cleaned surface
- require scientific acceptance `pass` for true success
- keep AUC/AP as a co-primary objective rather than sacrificing predictive power blindly

This is the highest-leverage path to both a stronger scientific tool and a stronger model.

# Jury Brief

## Core Claim

This framework retrospectively prioritizes plasmid backbone surveillance units using pre-2016 genomic and ecological features, then tests whether those same backbone classes later show multi-country visibility increase.

## Formal Hypotheses

- **H0 (null)**: A <=2015 T/H/A-derived priority signal has no discriminative association with post-2015 multi-country visibility expansion.
- **H1 (alternative)**: The same priority signal is positively associated with post-2015 multi-country visibility expansion.
- **Significance criterion**: empirical permutation p-value below the predeclared threshold for the current headline model.

## Current Benchmark

- Discovery benchmark: `bio-clean model` | ROC AUC `0.747` | AP `0.660`.
- Counts-only baseline: `baseline_both` | ROC AUC `0.722` | AP `0.647`.
- Conservative benchmark: `parsimonious_priority` | ROC AUC `0.752` | AP `0.664`.
- Source-only control: `source_only` | ROC AUC `0.452`.
- Strongest audited metric model: `phylo-support fusion model` | ROC AUC `0.827` | AP `0.766`.
- Governance watch-only: `phylo-support fusion model` | ROC AUC `0.827` | AP `0.766` | strict `fail`.
- Selection-adjusted official-model permutation audit for the headline ROC AUC: `p 0.005`; the older fixed-score label-permutation entry is retained only as an exploratory appendix diagnostic.
- Delta vs counts-only baseline: `0.024, 95% CI [-0.019, 0.065]`.

## Primary Model Selection Rationale

Model selection was not driven by a single metric. We jointly considered ROC AUC, average precision, lower-knownness behavior, matched-knownness/source performance, source holdout robustness, and practical shortlist yield. In the current scorecard, the headline model ranks `13/21` overall; `2` within the discovery track.

Operationally, the headline model is preferred because it keeps the strongest balance between discrimination and shortlist usefulness. In this refresh, the primary model also preserves a top-10 precision of 1.0 while remaining clearly above the counts-only baseline in matched-knownness auditing.
Governance track logic is kept separate from discovery-track optimization even when the shortlisted candidates partially overlap.

## Strict Test Interpretation

The strict matched-knownness/source-holdout test isolates the hardest low-knownness and low-support slice of the dataset. No current model fully passes this acceptance rule.

This should be interpreted as a data-limited regime, not as evidence that the entire methodology collapses. The primary dataset contains 989 eligible backbone classes, but the strict slice is materially smaller and therefore noisier. We report this limitation explicitly instead of hiding it behind the stronger overall metrics.

## Decision Readout

- Outcome definition: later visibility in at least `3` new countries.
- Operational watchlist mix: `22` action + `18` review + `10` abstain rows.
- False-negative audit: `50` later positives remain outside the practical shortlist; dominant miss drivers are `low_assignment_confidence, low_training_members, low_knownness`.
- `operational_risk_watchlist.tsv` is the calibrated deployment-facing table for the current shortlist.
- Current operational watchlist mix: `22` action + `18` review + `10` abstain rows.
- This remains a shortlist-prioritization benchmark rather than an exhaustive detector for every later positive backbone.
- Matched knownness/source strata: primary `0.701` vs baseline `0.594` weighted ROC AUC.
- Raw later new-country count alignment: Spearman ρ `0.624` [0.580, 0.662].
- Weighted new-country burden alignment: Spearman ρ `0.620`.
- Spatial holdout audit: weighted ROC AUC `0.735` across `5` held-out dominant regions; hardest region `Asia` at ROC AUC `0.697`.
- Discovery shortlist agreement with the strongest audited metric model: top-25 overlap: `8/25`; top-50 overlap: `22/50`.
- A knownness-gated audit model (`adaptive_natural_priority`) remains useful for lower-knownness stress testing but is not the headline benchmark.
- Observed host-diversity terms should be interpreted cautiously because they partly behave like sampling saturation / knownness signals.
- Supportive external layers are descriptive context only; AMRFinder is optional and not required for the headline benchmark.
- Only three models are official in the jury-facing narrative: discovery, governance watch-only, and baseline.

## Zero-Floor Component Behavior

When a backbone lacks direct evidence for a component, the normalized contribution is explicitly allowed to stay at zero rather than being imputed upward by unrelated metadata support.

## OLS Residual Approach

Knownness-sensitive H-support terms are residualized against knownness proxies so that the retained signal is not a disguised count effect.

## Turkey Context

WHO's 2025 GLASS summary highlights a high antibiotic-resistance burden in the Eastern Mediterranean region. For Türkiye, this makes Enterobacterales-focused genomic surveillance directly relevant.

Within the ECDC/WHO Europe surveillance framing, carbapenem-resistant *Klebsiella pneumoniae* and ESBL-producing *Escherichia coli* remain core public-health concerns. A backbone-level prioritization system is therefore operationally meaningful for Turkish genomic AMR surveillance, even though this project does not claim clinical decision support.

## Interpretation Guardrails

- No external validation claim is made.
- T, H and A features are computed only from `resolved_year <= 2015` rows.
- The outcome is later country visibility increase, not direct biological fitness or transmission proof.
- Opportunity bias is a declared limitation: backbones seen earlier have longer time-at-risk.

## Release Surface

- `blocked_holdout_summary.tsv` records the blocked source/region stress test used for the internal audit layer.
- `calibration_threshold_summary.png` captures the compact calibration/threshold view used in slide decks.
- `reports/core_figures/` contains the rest of the presentation-ready figure pack.

# Jury Brief

## Core Claim

This framework retrospectively prioritizes plasmid backbone surveillance units using training-period genomic and ecological features, then asks whether those same backbones later show new-country visibility increase.

## Formal Hypotheses

- **H0 (null)**: A <=2015 T/H/A-derived priority signal has no discriminative association with post-2015 multi-country visibility expansion (ROC AUC = 0.50).
- **H1 (alternative)**: The same priority signal is positively associated with post-2015 multi-country visibility expansion (ROC AUC > 0.50).
- **Significance criterion**: empirical permutation p-value < 0.01, using the current primary model's feature set and the same L2 / weighting configuration as the headline evaluation.

## Current Benchmark and Audit Context

- Current primary benchmark: `support-synergy biological model` with ROC AUC `0.814` and AP `0.737`.
- Strongest audited metric model: `support-synergy biological model` with ROC AUC `0.814` and AP `0.737`.
- Conservative benchmark: `bio-clean model` with ROC AUC `0.761` and AP `0.668`.
- Counts-only baseline: `baseline_both` with ROC AUC `0.723` and AP `0.648`.
- Source-only control: `source_only` with ROC AUC `0.452`.
- Primary-model selection rationale: current primary is also the strongest current single-model benchmark, so the headline and strongest audited metric model now coincide

## Decision-Support Readout

- Reviewer shortlist size: `10` established high-risk + `10` novel-signal candidates in `candidate_portfolio.tsv`.
- Read candidate outputs in this order: `candidate_portfolio.tsv` -> `candidate_evidence_matrix.tsv` -> `candidate_threshold_flip.tsv`.
- Main outcome threshold: `3` later new countries; use `candidate_threshold_flip.tsv` to see which candidates are definition-sensitive.
- In the multi-objective selection scorecard, the primary model ranks `1/18` after combining overall AUC, AP, lower-half/q1 knownness, matched-knownness, source holdout, and a knownness-dependence penalty.
- Augmented biological audit model: `natural_auc_priority` with ROC AUC `0.786` and AP `0.697`; this model adds external host-range, backbone purity, assignment confidence, mash-based novelty, and replicon architecture without changing the current headline benchmark.
- Knownness-robust biological audit model: `knownness_robust_priority` with ROC AUC `0.804` and AP `0.713`; this variant keeps the biological core but replaces external host-range with recurrent AMR structure, pMLST coherence, and eco-clinical context under class+knownness balancing.
- Support-calibrated biological model: `support_calibrated_priority` with ROC AUC `0.808` and AP `0.725`; this variant keeps the knownness-robust biological core but makes annotation support explicit through host-range support, pMLST presence, and AMR support depth.
- Support-synergy biological model: `support_synergy_priority` with ROC AUC `0.814` and AP `0.737`; this variant keeps the support-calibrated core but adds metadata support depth, external host-range magnitude, and host-range x transfer synergy to recover sparse-support errors without adding count proxies.
- Error-focused host-transfer synergy model: `host_transfer_synergy_priority` with ROC AUC `0.804` and AP `0.714`; this variant adds explicit host-range x transfer coupling to recover sparse-backbone mistakes without introducing direct knownness counts.
- Threat-architecture audit model: `threat_architecture_priority` with ROC AUC `0.804` and AP `0.713`; this variant keeps the host-transfer coupling but adds AMR clinical-threat burden plus replicon multiplicity to recover sparse-backbone misses with biologically interpretable structure.
- Taxonomy-aware H audit model: `phylogeny_aware_priority` with ROC AUC `0.779` and AP `0.697`; this variant preserves the augmented biological core but swaps the H axis for a lineage-aware host specialization signal.
- Structure-aware biological audit model: `structured_signal_priority` with ROC AUC `0.784` and AP `0.681`; this variant keeps the taxonomy-aware H axis and adds host evenness plus recurrent AMR structure.
- Current primary top-10 yield: precision `1.000`, recall `0.028`.
- Conservative top-10 yield: precision `0.900`, recall `0.025`.
- Counts-only baseline top-10 yield: precision `0.900`, recall `0.025`.
- Current-primary vs strongest-metric top-10 overlap: `10/10` candidates; top-25 overlap: `25/25`; top-50 overlap: `50/50`.
- Top-25 is the more decision-relevant cut: current primary precision `1.000`, recall `0.069`.
- Published-primary top-25 contains `0` lower-knownness candidates, so the `novel_signal` track should be read as a separate exploratory watchlist rather than as the same shortlist.
- Lowest-knownness quartile performance is weaker (primary ROC AUC `0.979`), so early-signal claims should stay conservative.
- A knownness-gated audit model (`adaptive_natural_priority`) keeps `natural_auc_priority` in the upper-knownness half and uses leakage-free OOF base plus OOF specialist scoring in the lower-knownness half, reaching ROC AUC `0.867` with AP `0.779`.
- The strongest knownness-gated audit model (`adaptive_support_synergy_blend_priority`) uses `support-synergy biological model` as the base score and applies specialist weight `0.50` within the lower-knownness half, reaching ROC AUC `0.934` with AP `0.890`; this remains a routing audit rather than a replacement headline benchmark.
- Gate consistency audit: for `adaptive_support_synergy_blend_priority`, the `99` backbones closest to the active routing boundary showed mean |Δscore| `0.071`, p90 |Δscore| `0.149`, and route Spearman `0.948` under counterfactual route switching; this gate is currently tiered `moderate`.
- Source-balanced reruns average ROC AUC `0.889`, so source composition remains a real robustness caveat rather than a fully neutral nuisance factor.
- Within matched visibility/source strata, the current primary model still exceeds the counts-only baseline (`0.698` vs `0.691` weighted ROC AUC).
- Secondary outcome auditing now includes macro-region jumps; the strongest audited model reaches ROC AUC `0.910` on that harder geographic expansion endpoint.
- Ranking scores are also compared against upload-weighted new-country burden; the best audited Spearman correlation is `0.848`.
- The same ranking family is now also audited against the raw later new-country count; the strongest Spearman alignment is `0.878`.
- Backbone-level metadata quality is quantified directly (mean score `0.574`) and reused in the false-negative audit instead of being left as an informal caveat.

## Interpretation Guardrails

- T, H, A, coherence, and mobility-support features are computed only from `resolved_year <= 2015` rows.
- The outcome uses only later-period country visibility (`2016-2023`), not training-period feature inputs.
- Supportive WHO/Pathogen Detection/CARD/MOB-suite layers are descriptive context only; AMRFinder is optional and not required for the headline benchmark or its main claims.
- This is a shortlist-prioritization benchmark, not an exhaustive detection system for every later positive backbone.
- Source-only performance is weak, but source-balanced reruns still matter, so source composition should be treated as a real robustness caveat.
- Observed host-diversity terms should not be read as direct biological host range; in practice they partly behave like sampling saturation / knownness signals.
- Additional biological audit tables now report external host-range support, backbone purity, assignment confidence, replicon architecture, and training-only mash novelty so AUC gains can be inspected mechanistically rather than treated as a black-box metric jump.
- The `adaptive_*` family changes only how the pre-2015 low-knownness cohort receives the specialist score, using either a hard switch or a partial blend; it does not relax the temporal split or alter the outcome definition.
- Primary-model choice is not driven by a single headline metric; overall discrimination, low-knownness behaviour, matched-strata behaviour, and source robustness are read together.
- Candidate interpretation is explicitly multi-table: portfolio + evidence matrix + threshold flips + consensus/risk context, not a single raw score rank.
- Only the current primary benchmark and the conservative benchmark should be treated as headline models; the rest of the model zoo is exploratory audit context, not multiple-comparison-free confirmatory evidence.
- Bootstrap intervals resample the analysis unit itself (backbone rows), so the uncertainty intervals are already computed at backbone granularity rather than at raw-sequence granularity.
- Spatial generalization is now audited separately via strict `dominant_region_train` holdouts, complementing the temporal split with an explicit out-of-distribution check.
- Opportunity bias is not fully removable in a retrospective archive: backbones first seen earlier have longer time-at-risk for later new-country visibility. This remains a declared limitation.
- Eligibility is intentionally restricted to backbones with 1-3 training countries; the system is meant for early-stage surveillance triage, not for all already-global backbones.
- Country metadata completeness is reported separately in `country_quality_summary.tsv`; missing countries must not be over-interpreted as true geographic narrowness.
- The project uses two permutation paradigms on purpose: the headline null asks whether the observed signal exceeds randomized labels, whereas model-comparison nulls ask whether one score family truly beats another.
- Ethical scope: only public genome and country metadata are used; the framework does not infer patient identity and is not a clinical diagnostic tool.

## Feature-Interpretation Note

- Highest dropout impact: `metadata_support_depth_norm` with ROC AUC drop `0.054` when removed.
- Highest ablation impact and strongest directional coefficient need not be the same signal; T-ablation and host-diversity interpretation answer different questions.

## Boundary Conditions

- This is a retrospective association framework, not a causal, mechanistic, or clinical prediction system.
- The outcome is later visibility increase, not direct proof of transmission fitness or public-health impact.
- Backbone definitions are operational surveillance units and should not be presented as biological truth claims.
- The current primary benchmark is the official headline ranking, but adaptive and exploratory audits remain useful for stress-testing its behavior.
- Current scored backbone count: `6841`.

## Zero-Floor Component Behavior

- When a raw T/H/A component is genuinely zero, its normalized counterpart remains 0.0. The arithmetic headline score therefore behaves like an average across only the active evidence axes for many backbones; this is deliberate and should not be narrated as if every backbone carries three equally active signals.

## OLS Residual Approach

- `H_support_norm_residual` is computed with ordinary least squares against visibility/knownness proxies. The goal is deterministic proxy-subtraction for audit purposes, not a causal robust-regression claim.

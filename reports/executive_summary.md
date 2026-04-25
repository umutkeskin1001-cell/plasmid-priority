# Executive Summary

Plasmid Priority is a retrospective surveillance ranking framework for plasmid backbone classes. It does not claim causal spread prediction; it asks whether pre-2016 genomic signals are associated with post-2015 international visibility increase.

The Seer (conditional benchmark candidate): `discovery-boosted primary model` | ROC AUC `0.811` | AP `0.721`.
The Guard (governance watch-only): `governance linear model` | ROC AUC `0.692` | AP `0.604`.
The Baseline: `counts-only baseline` | ROC AUC `0.736` | AP `0.648`.

Benchmark scope: Benchmark scope note: the headline benchmark does not clear the frozen scientific acceptance gate, so the narrative remains conditional and benchmark-limited.
Calibration note: fixed-bin ECE, max calibration error, calibration slope, and calibration intercept are reported explicitly, rather than being treated as uninterpreted summary numbers.

## Method Overview

```text
Raw Data (PLSDB + RefSeq + Pathogen Detection)
                |
                v
      Harmonization and Deduplication
                |
                v
      Backbone Assignment (MOB-suite style)
                |
                v
       Temporal Split: <=2015 | >2015
                |
                v
        T / H / A Feature Extraction
                |
                v
   L2-Regularized Logistic Regression (OOF)
          /                           \
         v                             v
 Discovery Track          Governance Watch-only Track
                \               /
                 v             v
            Candidate Portfolio + Risk Tiers
```

## Validation Posture

- No external validation claim is made.
- Validation is framed as temporal holdout, source holdout, knownness-matched auditing, and an internal high-integrity subset audit.
- False-negative audit: `50` later positives remain outside the practical shortlist; dominant miss drivers are `low_assignment_confidence, low_training_members, low_knownness`.
- Rolling-origin validation: outer split years 2014 to 2017 across horizons 1,3,5 with assignment modes all_records,training_only; ROC AUC mean 0.761 (range 0.546 to 0.858).

## Ranking Stability

- `candidate_rank_stability.tsv` records candidate rank stability across bootstrap resamples; the strongest stable backbone `AA316` remains in the top-`15` set at frequency `1.00`.
- `candidate_variant_consistency.tsv` records candidate rank stability across model variants; the strongest stable backbone `AA171` remains in the top-`15` set at frequency `1.00`.

## Release Surface

- `6` case studies are exported in `candidate_case_studies.tsv`.
- Jury-facing narrative lives in `jury_brief.md` and `ozet_tr.md`.
- `frozen_scientific_acceptance_audit.tsv` records the headline acceptance gate across matched-knownness, source holdout, spatial holdout, calibration, and leakage review.
- Blocked holdout audit is exported in `blocked_holdout_summary.tsv`.
- `nonlinear_deconfounding_audit.tsv` records the nonlinear deconfounding check used to keep knownness residualization transparent.
- `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv`, and `macro_region_jump_outcome.tsv` record the alternative-endpoint stress tests for ordinal, exposure-adjusted, and macro-region jump outcomes.
- `prospective_candidate_freeze.tsv` and `annual_candidate_freeze_summary.tsv` record the quasi-prospective freeze surface used to check whether the shortlist survives a forward-looking holdout.
- `future_sentinel_audit.tsv`, `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv`, and `amr_uncertainty_summary.tsv` record the leakage canary, graph audit, counterfactual shortlist comparison, geographic-jump diagnostic, and AMR-uncertainty summary.
- Candidate rank stability is exported in `candidate_rank_stability.tsv` and model-variant consistency is exported in `candidate_variant_consistency.tsv`.
- `calibration_threshold_summary.png` is exported as a compact calibration/threshold diagnostic when threshold-sensitivity data are available.
- Figures in `reports/core_figures/` are presentation-ready.

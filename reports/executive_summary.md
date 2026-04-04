# Executive Summary

Plasmid Priority is a retrospective surveillance ranking framework for plasmid backbone classes. It does not claim causal spread prediction; it asks whether pre-2016 genomic signals are associated with post-2015 international visibility increase.

The Seer (headline model): `bio-clean model` | ROC AUC `0.747` | AP `0.660`.
The Guard (governance watch-only): `phylo-support fusion model` | ROC AUC `0.827` | AP `0.766`.
The Baseline: `counts-only baseline` | ROC AUC `0.722` | AP `0.647`.

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
- Rolling-origin validation: outer split years 2012 to 2018 across horizons 1,3,5,8 with assignment modes all_records,training_only; ROC AUC mean 0.749 (range 0.636 to 0.914).

## Country Missingness

- bio-clean model country-missingness audit (`country_missingness_bounds.tsv`, `country_missingness_sensitivity.tsv`): observed labels mark 362/989 eligible backbones positive; midpoint / optimistic / weighted interpretations shift 75/89/42 labels and yield 437/451/404 positives. Sensitivity across those label variants spans ROC AUC 0.744 to 0.757 and AP 0.660 to 0.732..

## Ranking Stability

- `candidate_rank_stability.tsv` records candidate rank stability across bootstrap resamples; the strongest stable backbone `AA175` remains in the top-`25` set at frequency `1.00`.
- `candidate_variant_consistency.tsv` records candidate rank stability across model variants; the strongest stable backbone `AA324` remains in the top-`25` set at frequency `0.88`.

## Release Surface

- `6` case studies are exported in `candidate_case_studies.tsv`.
- Jury-facing narrative lives in `jury_brief.md` and `ozet_tr.md`.
- Blocked holdout audit is exported in `blocked_holdout_summary.tsv`.
- Candidate rank stability is exported in `candidate_rank_stability.tsv` and model-variant consistency is exported in `candidate_variant_consistency.tsv`.
- `calibration_threshold_summary.png` is exported as a compact calibration/threshold diagnostic when threshold-sensitivity data are available.
- Figures in `reports/core_figures/` are presentation-ready.

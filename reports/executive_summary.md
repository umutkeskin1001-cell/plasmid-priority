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

## Release Surface

- `6` case studies are exported in `candidate_case_studies.tsv`.
- Jury-facing narrative lives in `jury_brief.md` and `ozet_tr.md`.
- Blocked holdout audit is exported in `blocked_holdout_summary.tsv`: `bio-clean model` blocked holdout audit (dominant_source + dominant_region_train): weighted ROC AUC `0.735` across `4` blocked groups; hardest group `dominant_region_train:Europe` at ROC AUC `0.701`.
- `calibration_threshold_summary.png` is exported as a compact calibration/threshold diagnostic when threshold-sensitivity data are available.
- Figures in `reports/core_figures/` are presentation-ready.

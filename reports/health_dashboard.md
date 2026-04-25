# Pipeline Health Dashboard - 2026-04-25T21:31:02+00:00

## Protocol Status
- Hash: c6ede6432592
- Split year: 2015 | Horizon: 5y | Accept thresholds: ECE<=0.05, p<=0.01

## Data Status
| Asset | Hash | Size | Status |
|-------|------|------|--------|
| plsdb_sequences_fasta | e654c106d8a1 | 7425668755 | OK |
| refseq_plasmids_fasta | f25b59541947 | 10562636529 | OK |
| assembly_summary_refseq | efe199629e10 | 225872454 | OK |
| biosample_csv | 3ef01dfc2d3d | 11700408 | OK |
| amr_tsv | dea80560f0b6 | 37967165 | OK |
| plsdb_metadata_tsv | 1c599c9a900b | 16833553 | OK |
| who_mia_pdf | 3b81e2301ba7 | 497863 | OK |
| plsdb_mashdb_sim_tsv | e5aba1955e25 | 13731287 | OK |

## Quality Gates
| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| ece | 0.0918 | 0.0500 | FAIL |
| selection_adjusted_p | 0.0909 | 0.0100 | FAIL |
| matched_knownness_gap | -0.0583 | -0.0050 | FAIL |
| source_holdout_gap | -0.2983 | -0.0050 | FAIL |
| spatial_holdout_gap | -0.1010 | -0.0300 | FAIL |
| calibration_slope | 1.0280 | [0.8500 - 1.1500] | PASS |
| calibration_intercept | -0.5282 | [-0.1000 - 0.1000] | FAIL |

## Release Readiness
FAIL

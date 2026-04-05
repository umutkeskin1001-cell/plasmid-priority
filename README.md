# Plasmid Priority

Plasmid Priority is a retrospective genomic surveillance prioritization framework for operational plasmid backbone classes. It scores backbones with mobility (`T`), observed host diversity (`H`), and AMR burden/consistency (`A`), then tests whether higher-priority backbones are retrospectively associated with later new-country visibility increase.

The project now has two explicit model tracks:

- `discovery`: the strongest held-out discrimination model for retrospective prioritization.
- `governance`: the guardrail-aware model used for deployment-style interpretation, matched-knownness auditing, and report safety checks.

## Scientific Boundaries

This repository does not claim to predict true biological spread, prove transmission, or act as a clinical risk tool. The primary outcome is a visibility-based retrospective label, not direct epidemiological spread.

## Core Data Sources

Current computational inputs in the fast pipeline:

- PLSDB FASTA + canonical metadata + side tables
- RefSeq plasmid FASTA
- Pathogen Detection metadata tables
- WHO MIA text extraction for curated reference validation
- Local CARD and MOB-suite archives for descriptive support
- Local AMRFinderPlus database for the supportive concordance probe (required by the current manifest; the `amrfinder` executable itself may be absent, in which case the probe is skipped)

Repository-retained reserve assets not required by the current fast analytical path:

- RefSeq assembly summary
- NCBI taxonomy dump
- Project-local ResFinder and PlasmidFinder databases

Supportive layers include the WHO Medically Important Antimicrobials document, Pathogen Detection metadata, the local CARD archive, the local MOB-suite reference archive, and, when the `amrfinder` executable is available, a small-panel AMRFinder concordance check. In this repository, these layers are descriptive support or sanity-check layers only; they are never used as model training features and must not be presented as standalone external validation claims.

## Repository Layout

- `src/plasmid_priority/`: reusable package code
- `scripts/`: numbered pipeline entry points
- `data/manifests/`: path authority and machine-readable data contract
- `data/experiments/`: exploratory search outputs and non-canonical experiment artifacts
- `tests/`: unit and smoke-level checks

## Bu Çalışmanın Katkısı

Plasmid omurga düzeyinde biyolojik sürveyans önceliklendirmesi, çoğu zaman tür veya tekil plazmid odaklı çerçevelerin gölgesinde kalmaktadır. Bu depo, omurga sınıfını doğrudan analiz birimi yaparak bu boşluğu hedefler.

Bu çalışma:

1. Transfer mobilizasyonu (`T`), konak çeşitliliği (`H`) ve AMR yükü (`A`) sinyallerini birleştiren omurga-düzeyi bir önceliklendirme çerçevesi sunar.
2. Retrospektif zamansal tasarımla, 2015 öncesi sinyallerin 2015 sonrası coğrafi görünürlük artışıyla ilişkili olduğunu gösterir.
3. Discovery ve governance olmak üzere iki paralel model hattı kullanarak yüksek ayırıcılık ile temkinli karar katmanını birbirinden ayırır.

## Tekrar Üretilebilirlik

Bu analiz Python 3.13 ve `uv` ortam yöneticisi ile üretilmiştir.

```bash
pip install uv
uv sync
```

`uv.lock`, bağımlılıkları sabit sürümlerle kilitler. Çalışma yüzeyi Python 3.12+ üzerinde de desteklenir.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis]"  # Installs the package and analysis dependencies
python scripts/01_check_inputs.py
python scripts/26_run_tests_or_smoke.py
```

## Current Scope

The current implementation now covers the main retrospective pipeline:

- manifest-driven input validation
- canonical metadata harmonization and deduplication
- operational backbone assignment and T/H/A feature generation
- retrospective Module A scoring and validation
- conservative and proxy-audit model comparisons
- subgroup validation, coefficient audit, and feature-dropout analysis
- discovery/governance split with guardrail-aware selection and calibrated risk reporting
- exploratory Module B AMR comparison
- supportive Module C Pathogen Detection metadata analysis, including clinical/environmental strata
- supportive CARD ontology and MOB-suite literature/cluster support analysis
- final report tables and figures under `reports/`

## Interpreting Results

The results of the pipeline are structured to be directly usable for scientific presentations and jury evaluations:

1. **Narratives** (`reports/jury_brief.md` & `reports/ozet_tr.md`): The main English and Turkish narrative summaries for jury-facing interpretation, with a short release-surface note that points to the blocked-holdout audit and calibration/threshold figure.
2. **Canonical One-Page Summary** (`reports/headline_validation_summary.md`): The single-sheet validation surface with discovery, governance, baseline, and confirmatory metrics.
3. **Core Figures** (`reports/core_figures/`): The high-impact visualizations ready for slide decks, including the compact calibration/threshold diagnostic when threshold-sensitivity data are available.
4. **Core Report Tables** (`reports/core_tables/`): The curated shortlist, model-selection, and portfolio tables that belong in presentations and handouts.
5. **Canonical Analysis Tables** (`data/analysis/`): The full machine-readable audit outputs. Large technical tables live here instead of being mirrored into multiple report folders.
6. **Experimental Artifacts** (`data/experiments/`): Model-search outputs, exploratory sweeps, and their checksum registry.
7. **Compact Summary Output**: `reports/tubitak_final_metrics.txt` carries the exact TÜBİTAK-ready headline metrics.

Canonical outputs live under `reports/`. Frozen release snapshots live under `reports/release/bundle/`. Each release workflow run refreshes the snapshot from the current canonical report surface; `reports/release/` stays git-ignored by design because it is a generated export layer.

The release surface also includes `blocked_holdout_summary.tsv` for the blocked-holdout stress test and `calibration_threshold_summary.png` for the combined calibration/threshold view, and the jury brief now points directly to those artifacts.

The curated shortlist is not a raw top-score dump. It is source-diverse and low-knownness-aware so that the portfolio keeps both operationally strong candidates and early-signal candidates in view.

## Recommended Run Order

```bash
make pipeline
```

If `.venv/` exists, the `Makefile` automatically uses `.venv/bin/python`.

To just rebuild the TÜBİTAK summary output after modifying how metrics are pulled:
```bash
make tubitak-summary
```

## Runtime Modes

The pipeline now supports an external data root via `PLASMID_PRIORITY_DATA_ROOT`.
This lets you keep large `data/*` trees on an external USB volume while leaving
`reports/*` on the laptop.

- `make fast-local SOURCE_DATA_ROOT=/Volumes/PLASMID_USB/data`: refresh the small local report cache from a full data root, then render reports from that cache. If `SOURCE_DATA_ROOT` is omitted, the command uses the already-seeded local cache.
- `make full-local DATA_ROOT=/Volumes/PLASMID_USB/data`: run against an explicit external `data` root. If `DATA_ROOT` is omitted in an interactive shell, the runner prompts for it. If the path is unavailable, the command fails fast.

Both modes keep `reports/` under the repository on the laptop. Only `data/*`
is redirected. If scoring/features/backbone logic changes, use `full-local`.

To run the same local quality gate used by CI:
```bash
make quality
```

To remove stale generated clutter before a fresh rebuild:
```bash
make clean-generated
```

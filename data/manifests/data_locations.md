# Data Locations

This document is the human-readable path authority for the current repository snapshot. If other notes or stale documents disagree with the workspace, the local files listed here win.

Snapshot freeze date for the current manifest: `2026-03-22`.

## Core Required Assets

| Key | Path | Notes |
| --- | --- | --- |
| `plsdb_sequences_fasta` | `data/raw/plsdb_sequences.fasta` | Canonical PLSDB sequence FASTA |
| `plsdb_metadata_tsv` | `data/raw/plsdb_metadata.tsv` | Canonical metadata derived from `nuccore.csv` |
| `plsdb_meta_tables_dir` | `data/raw/plsdb_meta_tables/` | Includes `nuccore.csv`, `taxonomy.csv`, `plasmidfinder.csv`, `amr.tsv` |
| `refseq_plasmids_fasta` | `data/raw/refseq_plasmids.fasta` | Combined RefSeq plasmid FASTA |
| `refseq_chunk_dir` | `data/raw/refseq_chunks/` | Raw chunk archive used to build the combined FASTA |
| `amrfinder_db_dir` | `data/external/amrfinder_db/` | Project-local release directory is auto-detected |

## Supportive Assets

| Key | Path | Notes |
| --- | --- | --- |
| `who_mia_pdf` | `data/raw/who-mia-list-2024.pdf` | WHO MIA PDF; not a bacterial priority-pathogen list |
| `who_mia_text` | `data/raw/who-mia-list-2024.txt` | Local text extraction used to validate the curated WHO MIA class catalog |
| `pathogen_detection_metadata` | `data/external/pathogen_detection/combined_metadata.tsv` | Metadata-only descriptive layer |
| `pathogen_detection_clinical` | `data/external/pathogen_detection/clinical.tsv` | Optional clinical subset for stratified descriptive support |
| `pathogen_detection_environmental` | `data/external/pathogen_detection/environmental.tsv` | Optional environmental subset for stratified descriptive support |

## Optional Assets

| Key | Path | Notes |
| --- | --- | --- |
| `refseq_assembly_summary` | `data/raw/assembly_summary_refseq.txt` | Retained reserve metadata; not required by the current fast analytical path |
| `taxonomy_dump_dir` | `data/raw/taxdump/` | Retained reserve taxonomy data; not required by the current fast analytical path |
| `resfinder_fasta` | `data/external/resfinder_db/all.fsa` | Retained reserve database with colocated BLAST indexes |
| `plasmidfinder_fasta` | `data/external/plasmidfinder_db/plasmidfinder_all.fsa` | Retained reserve database with colocated BLAST indexes |
| `mobsuite_db_tar` | `data/external/mobsuite_db/data.tar` | Optional mobility support |
| `card_archive` | `data/external/card/card-data.tar.bz2` | Optional AMR support |
| `uniprot_sprot_dat` | `data/external/uniprot/uniprot_sprot.dat` | Local file is uncompressed `.dat` |
| `kegg_pathways` | `data/external/kegg/pathways.tsv` | Optional descriptive layer |
| `kegg_ko_pathway` | `data/external/kegg/ko_pathway.tsv` | Optional descriptive layer |
| `kegg_ko_list` | `data/external/kegg/ko_list.tsv` | Local replacement for `ko_genes.tsv` |

## Derived Assets

| Key | Path | Notes |
| --- | --- | --- |
| `bronze_all_plasmids_fasta` | `data/bronze/all_plasmids.fasta` | Generated from PLSDB + RefSeq FASTA inputs |

## Accepted Deviations

- Pathogen Detection sequence layer `combined.fna.gz` is intentionally absent at this stage. The repository uses metadata-only descriptive support.
- The WHO file in this workspace is the Medically Important Antimicrobials list, not a bacterial priority-pathogen list.
- UniProt is stored as `uniprot_sprot.dat`, not `uniprot_sprot.dat.gz`.
- KEGG uses `ko_list.tsv` instead of `ko_genes.tsv`.

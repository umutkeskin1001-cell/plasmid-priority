#!/usr/bin/env python3
r'''Generate a minimal synthetic dataset for demonstration purposes.

This script creates a compact (~15-20 plasmids) dataset that exercises the
entire pipeline from bronze → silver → features → scoring → reporting.

Usage:
    python scripts/generate_sample_data.py [--output-dir data/sample]

The generated data follows the same schema as the production pipeline but
with synthetic, non-real plasmid sequences and metadata.
'''

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Set random seed for reproducible runs
random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_PLASMID_COUNT = 18
SAMPLE_COUNTRIES = [
    'USA', 'DEU', 'GBR', 'FRA', 'ESP', 'ITA', 'NLD', 'BEL', 'AUT', 'CHE',
    'JPN', 'CHN', 'KOR', 'AUS', 'IND', 'BRA', 'MEX', 'ZAF', 'EGY', 'NGA',
]
SAMPLE_GENERA = ['Escherichia', 'Klebsiella', 'Salmonella', 'Pseudomonas', 'Acinetobacter']
SAMPLE_SPECIES = ['coli', 'pneumoniae', 'enterica', 'aeruginosa', 'baumannii']
SAMPLE_REPLICONS = ['IncF', 'IncI', 'IncX', 'IncH', 'IncN', 'IncW', 'IncP', 'IncQ']
SAMPLE_AMR_GENES = ['blaTEM', 'blaCTX-M', 'blaOXA', 'blaKPC', 'blaNDM', 'aac(6)', 'qnr', 'sul', 'tet']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_dna(length: int) -> str:
    '''Generate random DNA sequence.'''
    return ''.join(random.choices('ACGT', k=length))


def generate_sequence_id() -> str:
    '''Generate a realistic-looking sequence ID.'''
    prefix = random.choice(['NZ', 'NC', 'NR', 'AP'])
    number = random.randint(100000, 9999999)
    version = random.randint(1, 99)
    return f'{prefix}{number}.{version}'


# ---------------------------------------------------------------------------
# Core data generators
# ---------------------------------------------------------------------------

def generate_fasta(output_path: Path, count: int) -> dict[str, int]:
    '''Generate minimal plasmid FASTA files.'''
    records = []
    total_bases = 0

    for i in range(count):
        acc = generate_sequence_id()
        length = random.randint(2000, 150000)  # Realistic plasmid sizes
        sequence = random_dna(length)
        records.append(f'>{acc}\n{sequence}\n')
        total_bases += length

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(records), encoding='utf-8')

    return {'record_count': count, 'base_count': total_bases}


def generate_plsdb_metadata(output_path: Path, count: int) -> list[dict]:
    '''Generate PLSDB-style metadata table.'''
    rows = []
    taxa = list(range(1000, 1000 + count * 2))
    random.shuffle(taxa)

    for i in range(count):
        acc = generate_sequence_id()
        length = random.randint(2000, 150000)
        year = random.randint(2010, 2024)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date = f'{year}-{month:02d}-{day:02d}'
        source = random.choice(['RefSeq', 'GenBank'])
        rows.append({
            'NUCCORE_ACC': acc,
            'NUCCORE_CreateDate': date,
            'NUCCORE_Length': length,
            'NUCCORE_Source': source,
            'TAXONOMY_UID': taxa[i],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    return rows


def generate_taxonomy_csv(output_path: Path, rows: list[dict]) -> None:
    '''Generate taxonomy side table.'''
    taxa_data = {}
    for row in rows:
        uid = row['TAXONOMY_UID']
        genus = random.choice(SAMPLE_GENERA)
        species = random.choice(SAMPLE_SPECIES)
        taxa_data[uid] = {
            'TAXONOMY_UID': uid,
            'TAXONOMY_genus': genus,
            'TAXONOMY_species': species,
            'TAXONOMY_parent_uid': uid // 10 if uid > 1000 else 2,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(taxa_data.values()))
    df.to_csv(output_path, index=False)


def generate_plasmidfinder_csv(output_path: Path, count: int) -> None:
    '''Generate plasmidfinder typing table.'''
    rows = []
    for i in range(count):
        acc = generate_sequence_id()
        replicon = random.choice(SAMPLE_REPLICONS)
        identity = random.uniform(85.0, 100.0)
        coverage = random.uniform(80.0, 100.0)
        rows.append({
            'NUCCORE_ACC': acc,
            'typing': replicon,
            'identity': round(identity, 2),
            'coverage': round(coverage, 2),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def generate_amr_tsv(output_path: Path, count: int) -> None:
    '''Generate AMR gene table.'''
    rows = []
    for i in range(count):
        acc = generate_sequence_id()
        gene = random.choice(SAMPLE_AMR_GENES) + random.choice(['1a', '1b', '2', '3'])
        drug = random.choice(['AMP', 'CTX', 'CAZ', 'MEM', 'GEN', 'AMI'])
        rows.append({
            'NUCCORE_ACC': acc,
            'gene_symbol': gene,
            'drug_class': drug,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)


def generate_pathogen_detection_metadata(output_path: Path, count: int) -> None:
    '''Generate Pathogen Detection metadata.'''
    rows = []
    for i in range(count):
        organism = f'{random.choice(SAMPLE_GENERA)} {random.choice(SAMPLE_SPECIES)}'
        country = random.choice(SAMPLE_COUNTRIES)
        assembly = generate_sequence_id()
        amr = ', '.join(random.sample(SAMPLE_AMR_GENES, k=random.randint(1, 3)))
        host = random.choice(['Human', 'Animal', 'Environmental', 'Food'])
        rows.append({
            '#Organism group': organism,
            'Location': country,
            'Assembly': assembly,
            'AMR genotypes': amr,
            'Host': host,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)


def generate_amrfinder_db(output_path: Path) -> None:
    '''Generate minimal AMRFinder database structure.'''
    release_dir = output_path / '2024-01-01.1'
    release_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal version file
    (release_dir / 'version.txt').write_text('2024-01-01.1\n', encoding='utf-8')

    # Create minimal FASTA files
    (release_dir / 'AMRProt.fa').write_text(
        '>test_protein_1\nMRILIIFCSVLILGLIGTANSK\n',
        encoding='utf-8'
    )
    (release_dir / 'AMR_CDS.fa').write_text(
        '>test_gene_1\nATGAGAATACTTATAATTAA\n',
        encoding='utf-8'
    )


def generate_combined_metadata_for_bio_transfer(output_path: Path, count: int) -> None:
    '''Generate combined metadata for bio_transfer branch.'''
    rows = []
    for i in range(count):
        acc = generate_sequence_id()
        organism = f'{random.choice(SAMPLE_GENERA)} {random.choice(SAMPLE_SPECIES)}'
        country = random.choice(SAMPLE_COUNTRIES)
        year = random.randint(2015, 2024)
        genus = random.choice(SAMPLE_GENERA)
        species = random.choice(SAMPLE_SPECIES)
        replicon = random.choice(SAMPLE_REPLICONS)
        amr_genes = random.randint(0, 5)
        host = random.choice(['Human', 'Animal', 'Environmental'])
        clinical = random.choice([0, 1])

        rows.append({
            'sequence_accession': acc,
            'organism': organism,
            'country': country,
            'year': year,
            'genus': genus,
            'species': species,
            'replicon_type': replicon,
            'amr_gene_count': amr_genes,
            'host_source': host,
            'clinical_context': clinical,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'sample',
        help='Output directory for sample data (default: data/sample)',
    )
    parser.add_argument(
        '--count', '-n',
        type=int,
        default=SAMPLE_PLASMID_COUNT,
        help=f'Number of sample plasmids (default: {SAMPLE_PLASMID_COUNT})',
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite existing sample data',
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(f'Sample data already exists at {output_dir}')
        print('Use --force to overwrite or --output-dir to specify a different location.')
        return 1

    print(f'Generating sample data in {output_dir}')
    print(f'  Plasmid count: {args.count}')

    # Raw data
    plsdb_fasta = output_dir / 'raw' / 'plsdb_sequences.fasta'
    plsdb_meta = output_dir / 'raw' / 'plsdb_metadata.tsv'
    meta_tables = output_dir / 'raw' / 'plsdb_meta_tables'
    refseq_fasta = output_dir / 'raw' / 'refseq_plasmids.fasta'

    # External data
    amrfinder = output_dir / 'external' / 'amrfinder_db'
    pathogen = output_dir / 'external' / 'pathogen_detection'

    print('\n[1/8] Generating PLSDB FASTA...')
    stats = generate_fasta(plsdb_fasta, args.count)
    record_count = stats['record_count']
    base_count = stats['base_count']
    print(f'      Created {record_count} records, {base_count:,} bases')

    print('\n[2/8] Generating PLSDB metadata...')
    meta_rows = generate_plsdb_metadata(plsdb_meta, args.count)
    print(f'      Created {len(meta_rows)} metadata rows')

    print('\n[3/8] Generating taxonomy table...')
    generate_taxonomy_csv(meta_tables / 'taxonomy.csv', meta_rows)
    print('      Created taxonomy.csv')

    print('\n[4/8] Generating plasmidfinder table...')
    generate_plasmidfinder_csv(meta_tables / 'plasmidfinder.csv', args.count)
    print('      Created plasmidfinder.csv')

    print('\n[5/8] Generating AMR table...')
    generate_amr_tsv(meta_tables / 'amr.tsv', args.count)
    print('      Created amr.tsv')

    print('\n[6/8] Generating RefSeq FASTA...')
    stats2 = generate_fasta(refseq_fasta, args.count // 3)
    record_count2 = stats2['record_count']
    base_count2 = stats2['base_count']
    print(f'      Created {record_count2} records, {base_count2:,} bases')

    print('\n[7/8] Generating AMRFinder database...')
    generate_amrfinder_db(amrfinder)
    print('      Created minimal AMRFinder database structure')

    print('\n[8/8] Generating Pathogen Detection metadata...')
    generate_pathogen_detection_metadata(pathogen / 'combined_metadata.tsv', args.count)
    print('      Created combined_metadata.tsv')

    # Generate bio_transfer sample data
    bio_dir = output_dir / 'bio_transfer'
    print('\n[Bonus] Generating bio_transfer sample data...')
    generate_combined_metadata_for_bio_transfer(bio_dir / 'combined_metadata.tsv', args.count)
    print('      Created bio_transfer/combined_metadata.tsv')

    # Create a manifest file
    manifest = {
        'version': '1.0.0',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'plasmid_count': args.count,
        'purpose': 'Demo dataset for Plasmid Priority pipeline demonstration',
        'note': 'All sequences and metadata are synthetic and do not represent real plasmids',
    }
    manifest_path = output_dir / 'SAMPLE_DATA_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print('\n✓ Sample data generated successfully!')
    print(f'  Location: {output_dir}')
    print(f'  Manifest: {manifest_path}')
    print('\nTo use this data, run:')
    print(f'  export PLASMID_PRIORITY_DATA_ROOT={output_dir}')
    print('  make demo')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

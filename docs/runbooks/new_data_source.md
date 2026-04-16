# New Data Source Runbook

This runbook guides the process of adding a new plasmid metadata data source to the pipeline.

## Prerequisites

- The new data source must be indexed in `data/manifests/data_contract.json`
- New accessions must not overlap with existing canonical accessions (deduplication handles this)
- Year and country metadata must be harmonizable to ISO formats

## Steps

### 1. Add to Data Contract

Edit `data/manifests/data_contract.json`:
```json
{
  "required_inputs": {
    "new_source_table": {
      "path": "data/raw/new_source.tsv",
      "description": "New plasmid source — tab-separated with columns: accession, year, country, ...",
      "required_columns": ["accession", "year", "country"],
      "optional": false
    }
  }
}
```

### 2. Add Harmonization Logic

In `src/plasmid_priority/harmonize/records.py`, add a new reader function:
```python
def harmonize_new_source(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize new_source records to canonical format."""
    return df.rename(columns={
        "accession": "sequence_accession",
        "year": "resolved_year",
        "country": "resolved_country",
    })
```

### 3. Update Bronze Table Builder

In `scripts/03_build_bronze_table.py`, add:
```python
new_source_path = context.data_dir / "raw/new_source.tsv"
if new_source_path.exists():
    new_source = pd.read_csv(new_source_path, sep="\t")
    new_source = harmonize_new_source(new_source)
    new_source["data_source"] = "new_source"
    frames.append(new_source)
```

### 4. Run Validation

```bash
make check-inputs
python scripts/03_build_bronze_table.py
python scripts/05_deduplicate.py  # handles overlaps
```

### 5. Verify No Leakage

Verify the new source's post-split_year records are excluded from features:
```bash
python -m pytest tests/test_leakage.py -v
```

### 6. Rerun Full Pipeline

```bash
make full-local DATA_ROOT=$PLASMID_PRIORITY_DATA_ROOT
make quality
```

## Checklist

- [ ] Data contract updated
- [ ] Harmonization function added and tested
- [ ] Bronze builder updated
- [ ] Deduplication verified (no duplicate accessions)
- [ ] Feature leakage tests pass
- [ ] VIF audit results checked (`data/analysis/vif_audit_summary.tsv`)
- [ ] p/n ratio checked (`data/analysis/model_pn_ratio.tsv`)
- [ ] Full pipeline reruns cleanly

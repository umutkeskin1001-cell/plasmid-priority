from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "advanced_audits_script",
    PROJECT_ROOT / "scripts/27b_run_advanced_audits.py",
)
assert SPEC is not None and SPEC.loader is not None
advanced_audits_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = advanced_audits_script
SPEC.loader.exec_module(advanced_audits_script)


def test_load_metadata_quality_inputs_uses_canonical_raw_tables(tmp_path: Path) -> None:
    raw_dir = tmp_path / "plsdb_meta_tables"
    raw_dir.mkdir()
    pd.DataFrame(
        [
            {
                "NUCCORE_UID": "101",
                "STATUS": "Complete Genome",
                "NUCCORE_Completeness": "complete",
            }
        ]
    ).to_csv(raw_dir / "nuccore.csv", index=False)
    pd.DataFrame(
        [
            {
                "NUCCORE_UID": "101",
                "BIOSAMPLE_pathogenicity": "pathogenic",
                "DISEASE_tags": "sepsis",
                "ECOSYSTEM_tags": "clinical",
            }
        ]
    ).to_csv(raw_dir / "biosample.csv", index=False)
    pd.DataFrame(
        [
            {
                "NUCCORE_ACC": "NZ_CP000001.1",
                "NUCCORE_Completeness": "complete",
                "NUCCORE_DuplicatedEntry": "false",
            }
        ]
    ).to_csv(raw_dir / "nucc_identical.csv", index=False)

    assembly, biosample, nucc_identical = advanced_audits_script._load_metadata_quality_inputs(
        raw_dir
    )

    assert list(assembly.columns) == [
        "NUCCORE_UID",
        "ASSEMBLY_Status",
        "ASSEMBLY_coverage",
        "ASSEMBLY_SeqReleaseDate",
        "NUCCORE_Completeness",
    ]
    assert assembly.loc[0, "ASSEMBLY_Status"] == "Complete Genome"
    assert assembly.loc[0, "NUCCORE_Completeness"] == "complete"
    assert pd.isna(assembly.loc[0, "ASSEMBLY_coverage"])
    assert pd.isna(assembly.loc[0, "ASSEMBLY_SeqReleaseDate"])
    assert biosample.loc[0, "DISEASE_tags"] == "sepsis"
    assert bool(nucc_identical.loc[0, "NUCCORE_DuplicatedEntry"]) is False

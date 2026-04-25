"""Literature-backed validation surfaces built from curated evidence."""


from pathlib import Path

import pandas as pd

from plasmid_priority.io.table_io import read_table
from plasmid_priority.utils.files import ensure_directory

_PATTERN_CATEGORY: tuple[tuple[str, str], ...] = (
    ("IncF", "ESBL_high_risk"),
    ("IncHI2", "MDR_high_risk"),
    ("IncI", "Emerging_spread"),
    ("IncX", "High_mobility"),
    ("IncA/C", "Broad_host_range"),
    ("IncN", "Rapid_transfer"),
)


def _resolve_literature_path(project_root: Path) -> Path:
    for candidate in (
        project_root / "data" / "curated" / "literature_evidence.parquet",
        project_root / "data" / "curated" / "literature_evidence.tsv",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "literature evidence not found at data/curated/literature_evidence.parquet|.tsv",
    )


def build_literature_support_matrix(
    scored: pd.DataFrame,
    literature: pd.DataFrame,
    *,
    top_k: int = 100,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "backbone_id",
                "priority_index",
                "pattern",
                "risk_category",
                "literature_match_count",
                "example_pmids",
                "claim_level",
            ],
        )
    working = scored.copy()
    if "priority_index" not in working.columns:
        working["priority_index"] = 0.0
    working = working.sort_values("priority_index", ascending=False).head(top_k).copy()
    titles = literature.get("title", pd.Series(dtype=str)).astype(str).str.lower()
    pmids = literature.get("pmid", pd.Series(dtype=str)).astype(str)

    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        backbone_id = str(row.get("backbone_id", ""))
        pid = float(row.get("priority_index", 0.0))
        matched_pattern = ""
        matched_category = "unmapped"
        for pattern, category in _PATTERN_CATEGORY:
            if pattern.lower() in backbone_id.lower():
                matched_pattern = pattern
                matched_category = category
                break
        if matched_pattern:
            mask = titles.str.contains(matched_pattern.lower(), regex=False, na=False)
            hit_count = int(mask.sum())
            sample_pmids = ";".join(pmids.loc[mask].head(3).tolist())
        else:
            hit_count = 0
            sample_pmids = ""
        rows.append(
            {
                "backbone_id": backbone_id,
                "priority_index": pid,
                "pattern": matched_pattern,
                "risk_category": matched_category,
                "literature_match_count": hit_count,
                "example_pmids": sample_pmids,
                "claim_level": "literature_supported" if hit_count > 0 else "proxy",
            },
        )
    return pd.DataFrame(rows)


def build_literature_inventory(literature: pd.DataFrame) -> pd.DataFrame:
    if literature.empty:
        return pd.DataFrame(columns=["pub_year", "n_records"])
    frame = literature.copy()
    pub_year = frame["pub_year"] if "pub_year" in frame.columns else pd.Series(dtype=str)
    frame["pub_year"] = pub_year.astype(str).str.strip()
    summary = (
        frame.groupby("pub_year", dropna=False)
        .size()
        .reset_index(name="n_records")
        .sort_values("pub_year")
    )
    return summary


def generate_literature_validation_artifacts(
    project_root: Path,
    *,
    top_k: int = 100,
) -> dict[str, object]:
    scored_path = project_root / "data" / "scores" / "backbone_scored.tsv"
    if not scored_path.exists():
        scored_path = project_root / "data" / "scores" / "backbone_scored.parquet"
    if not scored_path.exists():
        raise FileNotFoundError(
            "scored artifact missing at data/scores/backbone_scored.tsv|.parquet"
        )

    literature_path = _resolve_literature_path(project_root)
    scored = read_table(scored_path, columns=["backbone_id", "priority_index"])
    literature = read_table(literature_path, columns=["pmid", "title", "pub_year"])

    matrix = build_literature_support_matrix(scored, literature, top_k=top_k)
    inventory = build_literature_inventory(literature)

    core_dir = ensure_directory(project_root / "reports" / "core_tables")
    diag_dir = ensure_directory(project_root / "reports" / "diagnostic_tables")
    matrix_path = core_dir / "literature_validation_matrix.tsv"
    inventory_path = diag_dir / "literature_evidence_inventory.tsv"
    matrix.to_csv(matrix_path, sep="\t", index=False)
    inventory.to_csv(inventory_path, sep="\t", index=False)
    return {
        "matrix_path": str(matrix_path),
        "inventory_path": str(inventory_path),
        "top_k": int(top_k),
        "n_matrix_rows": int(len(matrix)),
        "n_inventory_rows": int(len(inventory)),
    }

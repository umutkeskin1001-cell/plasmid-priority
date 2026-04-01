from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pandas as pd


from plasmid_priority.reporting.amrfinder_support import (
    build_amrfinder_concordance_tables,
    parse_amrfinder_probe_report,
    write_selected_fasta_records,
)


class AmrFinderSupportTests(unittest.TestCase):
    def test_parse_amrfinder_probe_report_aggregates_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "probe.tsv"
            report_path.write_text(
                "\n".join(
                    [
                        "Protein id\tContig id\tElement symbol\tClass",
                        "NA\tseq1\ttet(L)\tTETRACYCLINE",
                        "NA\tseq1\ttet(M)\tTETRACYCLINE",
                        "NA\tseq1\term(B)\tMACROLIDE",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = parse_amrfinder_probe_report(report_path)

        row = parsed.iloc[0]
        self.assertEqual(row["sequence_accession"], "seq1")
        self.assertEqual(int(row["amrfinder_hit_count"]), 3)
        self.assertIn("TETRACYCLINE", row["amrfinder_class_tokens"])

    def test_build_amrfinder_concordance_tables_returns_expected_jaccard(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "backbone_id": "bb1",
                    "priority_group": "high",
                    "priority_index": 0.9,
                    "sequence_accession": "seq1",
                }
            ]
        )
        amr_consensus = pd.DataFrame(
            [
                {
                    "sequence_accession": "seq1",
                    "amr_gene_symbols": "tet(L),erm(B)",
                    "amr_drug_classes": "TETRACYCLINE,MACROLIDE ANTIBIOTIC",
                }
            ]
        )
        amrfinder_probe = pd.DataFrame(
            [
                {
                    "sequence_accession": "seq1",
                    "amrfinder_gene_symbols": "tet(L),erm(B),aadA",
                    "amrfinder_class_tokens": "TETRACYCLINE,MACROLIDE,AMINOGLYCOSIDE",
                    "amrfinder_hit_count": 3,
                }
            ]
        )

        detail, summary = build_amrfinder_concordance_tables(panel, amr_consensus, amrfinder_probe)
        self.assertAlmostEqual(float(detail.iloc[0]["gene_jaccard"]), 2 / 3)
        self.assertAlmostEqual(float(detail.iloc[0]["class_jaccard"]), 2 / 3)
        self.assertEqual(int(summary.loc[summary["priority_group"] == "high", "n_with_any_amr_evidence"].iloc[0]), 1)
        self.assertAlmostEqual(
            float(summary.loc[summary["priority_group"] == "high", "mean_gene_jaccard_nonempty"].iloc[0]),
            2 / 3,
        )
        self.assertEqual(summary.iloc[-1]["priority_group"], "overall")

    def test_build_amrfinder_concordance_tables_handles_missing_probe_hits(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "backbone_id": "bb2",
                    "priority_group": "low",
                    "priority_index": 0.1,
                    "sequence_accession": "seq_missing",
                }
            ]
        )
        amr_consensus = pd.DataFrame(
            [
                {
                    "sequence_accession": "seq_missing",
                    "amr_gene_symbols": "tet(L)",
                    "amr_drug_classes": "TETRACYCLINE",
                }
            ]
        )
        amrfinder_probe = pd.DataFrame(
            columns=[
                "sequence_accession",
                "amrfinder_gene_symbols",
                "amrfinder_class_tokens",
                "amrfinder_hit_count",
            ]
        )

        detail, summary = build_amrfinder_concordance_tables(panel, amr_consensus, amrfinder_probe)
        self.assertEqual(int(detail.iloc[0]["amrfinder_hit_count"]), 0)
        self.assertFalse(bool(detail.iloc[0]["amrfinder_any_hit"]))
        self.assertTrue(bool(detail.iloc[0]["any_amr_evidence"]))
        self.assertEqual(int(summary.loc[summary["priority_group"] == "low", "n_with_any_amr_evidence"].iloc[0]), 1)
        self.assertEqual(
            float(summary.loc[summary["priority_group"] == "low", "mean_gene_jaccard_nonempty"].iloc[0]),
            0.0,
        )
        self.assertEqual(float(summary.loc[summary["priority_group"] == "low", "mean_shared_gene_count"].iloc[0]), 0.0)

    def test_write_selected_fasta_records_extracts_requested_accessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "all.fasta"
            output_path = Path(tmp_dir) / "selected.fasta"
            input_path.write_text(
                ">a one\nAAAA\n>b two\nCCCC\n>c three\nGGGG\n",
                encoding="utf-8",
            )
            result = write_selected_fasta_records(input_path, ["b", "c"], output_path)
            content = output_path.read_text(encoding="utf-8")

        self.assertEqual(result["found"], 2)
        self.assertIn(">b two", content)
        self.assertIn(">c three", content)
        self.assertNotIn(">a one", content)


if __name__ == "__main__":
    unittest.main()

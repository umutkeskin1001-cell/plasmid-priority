from __future__ import annotations

import tempfile
from pathlib import Path

from plasmid_priority.cache import ArtifactCache
from plasmid_priority.utils.files import path_signature_with_hash


def test_artifact_cache_publish_and_restore_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        cache = ArtifactCache(root / "cache")

        input_path = root / "input.tsv"
        output_path = root / "output.tsv"
        summary_path = root / "summary.json"

        input_path.write_text("id\tv\n1\tA\n", encoding="utf-8")
        output_path.write_text("id\tscore\n1\t0.9\n", encoding="utf-8")

        summary = {
            "script_name": "15_normalize_and_score",
            "status": "ok",
            "input_manifest": {
                "input.tsv": path_signature_with_hash(input_path, include_file_hash=True)
            },
            "output_manifest": {
                "output.tsv": path_signature_with_hash(output_path, include_file_hash=True),
            },
            "output_files_written": [str(output_path)],
        }

        manifest = cache.publish(
            step_name="15_normalize_and_score",
            cache_key="a" * 64,
            cache_key_payload={"example": "payload"},
            summary=summary,
            output_paths=[output_path],
        )

        output_path.unlink()
        restored = cache.restore(manifest, summary_path=summary_path)
        assert restored is True
        assert output_path.exists()
        assert summary_path.exists()
